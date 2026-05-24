#!/usr/bin/env python3
"""Runner for privacy DCR/NNDR comparison metrics.

Writes results in the standard Seq2Synth layout:
``results/{dataset}/{model}/privacy/{variant}.json``.
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
from seq2synth.metrics.privacy.privacy_metrics import (
    available_trajectory_datasets,
    _nan_to_none,
    get_dataset_config,
    print_summary,
    run_dataset_model,
)

DEFAULT_DATASETS = available_trajectory_datasets()
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
        "cs_dcr": result["cs_dcr"],
        "cs_nndr": result["cs_nndr"],
    }
    for key, value in ngram.items():
        per_metric[f"ngram_{key}"] = value
    if result.get("table_results"):
        per_metric["table_results"] = result["table_results"]

    summary = {
        "dcr": _summary_value(result["dcr"], "median"),
        "nndr": _summary_value(result["nndr"], "median"),
        "cs_dcr": _summary_value(result["cs_dcr"], "mean_over_T"),
        "cs_nndr": _summary_value(result["cs_nndr"], "mean_over_T"),
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
            "table_name": result.get("table_name"),
            "table_names": result.get("table_names"),
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
    )
    parser.add_argument("--model", "--models", dest="models", nargs="+", default=None)
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--results_base", dest="results_dir", type=Path)
    parser.add_argument("--parallel", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--workers", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--n_jobs", type=int, default=None, help=argparse.SUPPRESS)
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
        try:
            cfg = get_dataset_config(dataset)
        except Exception as exc:  # noqa: BLE001
            print(f"  SKIP [{dataset}]: {exc}", flush=True)
            continue
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
                    out_dir = args.results_dir / dataset / model / "privacy"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"{variant}.json"
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(_nan_to_none(payload), f, indent=2, default=str)
                    print(f"  wrote {out_path}", flush=True)
                    all_results.append(result)
                except Exception as exc:  # noqa: BLE001
                    elapsed = time.time() - t0
                    out_dir = args.results_dir / dataset / model / "privacy"
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
