#!/usr/bin/env python3
"""Runner for trajectory privacy DCR/NNDR comparison metrics.

Writes results in the standard Seq2Synth layout:
``results/{dataset}/{model}/trajectory_privacy/{variant}.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

SEQ2SYNTH_DIR = Path(__file__).resolve().parents[3]
if str(SEQ2SYNTH_DIR) not in sys.path:
    sys.path.insert(0, str(SEQ2SYNTH_DIR))

from paths import RESULTS_DIR
from seq2synth.metrics.trajectory_privacy.trajectory_privacy_metrics import (
    DATASET_CONFIGS,
    _nan_to_none,
    print_summary,
    run_dataset_model,
)

DEFAULT_DATASETS = ["airbnb_tabdit", "berka_tabdit", "rossmann_tabdit"]
DEFAULT_VARIANTS = ["default"]


def _summary_value(metric: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = metric.get(key)
        if isinstance(value, (int, float, np.floating)) and not isinstance(value, bool):
            value = float(value)
            return None if math.isnan(value) else value
    return None


def _format_payload(result: dict[str, Any], variant: str, elapsed_seconds: float) -> dict[str, Any]:
    ngram = result.get("ngram") or {}
    per_metric = {
        "dcr": result["dcr"],
        "nndr": result["nndr"],
        "temporal_dcr": result["temporal_dcr"],
        "temporal_nndr": result["temporal_nndr"],
    }
    for key, value in ngram.items():
        per_metric[f"ngram_{key}"] = value

    summary = {
        "dcr": _summary_value(result["dcr"], "median"),
        "nndr": _summary_value(result["nndr"], "median"),
        "temporal_dcr": _summary_value(result["temporal_dcr"], "mean_over_T"),
        "temporal_nndr": _summary_value(result["temporal_nndr"], "mean_over_T"),
    }
    for key, value in ngram.items():
        summary[f"ngram_{key}"] = _summary_value(value, "score")

    return {
        "summary": summary,
        "per_metric": per_metric,
        "metadata": {
            "status": "ok",
            "dataset": result["dataset"],
            "model": result["model"],
            "variant": variant,
            "key_col": result.get("key_col"),
            "time_col": result.get("time_col"),
            "num_cols_used": result.get("feature_cols", []),
            "n_real_rows": result.get("n_real_rows"),
            "n_synth_rows": result.get("n_synth_rows"),
            "ngram_n1": result.get("ngram_n1"),
            "ngram_n2": result.get("ngram_n2"),
            "elapsed_seconds": round(elapsed_seconds, 2),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        "--datasets",
        dest="datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        choices=list(DATASET_CONFIGS.keys()),
    )
    parser.add_argument("--model", "--models", dest="models", nargs="+", default=None)
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output", default=None, help="Optional aggregate JSON output path.")
    parser.add_argument("--ngram-n1", type=int, default=1)
    parser.add_argument("--ngram-n2", type=int, default=3)
    parser.add_argument("--ngram-max-bins", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_results: list[dict[str, Any]] = []

    unsupported = [variant for variant in args.variants if variant != "default"]
    if unsupported:
        print(
            "Only the default postprocessed files are supported by the current "
            f"DCR/NNDR implementation; skipping variants: {unsupported}",
            flush=True,
        )

    variants = [variant for variant in args.variants if variant == "default"] or ["default"]

    for dataset in args.datasets:
        cfg = DATASET_CONFIGS[dataset]
        models = args.models if args.models else cfg["models"]

        for model in models:
            for variant in variants:
                t0 = time.time()
                try:
                    result = run_dataset_model(
                        dataset,
                        model,
                        cfg,
                        ngram_n1=args.ngram_n1,
                        ngram_n2=args.ngram_n2,
                        ngram_max_bins=args.ngram_max_bins,
                    )
                    if result is None:
                        continue

                    elapsed = time.time() - t0
                    payload = _format_payload(result, variant, elapsed)
                    out_dir = args.results_dir / dataset / model / "trajectory_privacy"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"{variant}.json"
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(_nan_to_none(payload), f, indent=2, default=str)
                    print(f"  wrote {out_path}", flush=True)
                    all_results.append(result)
                except Exception as exc:  # noqa: BLE001
                    elapsed = time.time() - t0
                    out_dir = args.results_dir / dataset / model / "trajectory_privacy"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"{variant}.json"
                    payload = {
                        "summary": {},
                        "metadata": {
                            "status": "error",
                            "dataset": dataset,
                            "model": model,
                            "variant": variant,
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                            "elapsed_seconds": round(elapsed, 2),
                        },
                    }
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(_nan_to_none(payload), f, indent=2, default=str)
                    print(f"  ERROR [{dataset}][{model}][{variant}]: {exc}", flush=True)
                    print(f"  wrote {out_path}", flush=True)

    if all_results:
        print_summary(all_results, ngram_n1=args.ngram_n1, ngram_n2=args.ngram_n2)

    if args.output:
        out_path = Path(args.output)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(_nan_to_none(all_results), f, indent=2, default=str)
        print(f"\nAggregate results saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
