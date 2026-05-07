#!/usr/bin/env python3
"""Metadata-driven runner for Seq2Synth structural metrics.

The runner discovers tables, relationships, real files, synthetic variants,
and applicable structural metric specs from each dataset's metadata.json at
runtime. Paths are resolved relative to the local seq2synth package checkout.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

TIMESYNTH_DIR = Path(__file__).parents[2]
BENCHMARK_DIR = TIMESYNTH_DIR.parent
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from configs import (  # noqa: E402
    DATETIME_SDTYPES,
    NUMERIC_SDTYPES,
    VARIANT_SUFFIXES,
    available_datasets,
    available_models,
    load_dataset_config,
    resolve_datasets,
)
from paths import RESULTS_DIR  # noqa: E402
from seq2synth.metrics.structural.dynamic_khop_correlation_sim import (  # noqa: E402
    DynamicKHopCorrelationSimilarity,
)
from seq2synth.metrics.structural.multi_parent_conditional_sim import (  # noqa: E402
    MultiParentConditionalSimilarity,
)
from seq2synth.metrics.structural.sequence_length_similarity import (  # noqa: E402
    SequenceLengthSimilarity,
)
from seq2synth.metrics.structural.temporal_cardinality_shape_sim import (  # noqa: E402
    TemporalCardinalityShapeSimilarity,
)

logger = logging.getLogger("seq2synth.metrics.structural.run_structural")


@dataclass(frozen=True)
class MetricSpec:
    name: str
    klass: Callable[..., Any]
    init: dict[str, Any]
    requires: list[str]
    variants: tuple[str, ...]


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        val = float(obj)
        return None if math.isnan(val) else val
    if isinstance(obj, float):
        return None if math.isnan(obj) else obj
    if isinstance(obj, (np.ndarray, pd.Series)):
        return _json_safe(list(obj))
    return obj


def _load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    with open(path, "rb") as f:
        if f.read(4) == b"PAR1":
            return pd.read_parquet(path)
    return pd.read_csv(path)


def _score_str(score: Any) -> str:
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return "nan"
    return f"{score:.4f}"


def _trim_details(result: dict[str, Any]) -> dict[str, Any]:
    details = result.get("details")
    if isinstance(details, dict):
        for key in ("per_bin", "per_cell", "per_window"):
            details.pop(key, None)
    return result


def _metric_family(metric_name: str) -> str:
    family = metric_name.split("__", 1)[0]
    display_names = {
        "sequence_length_similarity": "SequenceLengthSimilarity",
        "temporal_cardinality_shape_similarity": "TemporalCardinalityShapeSimilarity",
        "dynamic_khop": "DynamicKHopCorrelationSimilarity",
        "multi_parent_conditional_similarity": "MultiParentConditionalSimilarity",
    }
    return display_names.get(family, family)


def _mean_scores_by_family(metrics_results: dict[str, Any]) -> dict[str, Any]:
    grouped: dict[str, list[float]] = {}
    for metric_name, result in metrics_results.items():
        if not isinstance(result, dict):
            continue
        score = result.get("score")
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            score = float(score)
            if not math.isnan(score):
                grouped.setdefault(_metric_family(metric_name), []).append(score)
    return {
        family: sum(scores) / len(scores) if scores else None
        for family, scores in grouped.items()
    }


def _per_metric_scores(metrics_results: dict[str, Any]) -> dict[str, dict[str, Any]]:
    per_metric: dict[str, dict[str, Any]] = {}
    for metric_name, result in metrics_results.items():
        if not isinstance(result, dict):
            continue
        score = result.get("score")
        if isinstance(score, float) and math.isnan(score):
            score = None
        per_metric.setdefault(_metric_family(metric_name), {})[metric_name] = score
    return per_metric


def _metric_variant_kind(variant_label: str) -> str:
    return "raw" if variant_label == "raw" else "postprocessed"


def _metadata_schema(cfg: Any) -> dict[str, Any]:
    return {
        "tables": {
            name: {
                "primary_key": table.primary_key,
                "numerical_cols": table.numeric_cols,
            }
            for name, table in cfg.tables.items()
        },
        "relationships": [
            {
                "parent": rel["parent_table_name"],
                "child": rel["child_table_name"],
                "fk": rel["child_foreign_key"],
            }
            for rel in cfg.relationships
            if {
                "parent_table_name",
                "child_table_name",
                "child_foreign_key",
            }.issubset(rel)
        ],
    }


def _datetime_cols(table: Any) -> list[str]:
    return [
        name
        for name, info in table.columns.items()
        if info.get("sdtype") in DATETIME_SDTYPES
    ]


def _numeric_cols(table: Any) -> list[str]:
    return [
        name
        for name, info in table.columns.items()
        if info.get("sdtype") in NUMERIC_SDTYPES
    ]


def _sequence_key(cfg: Any, table_name: str) -> str | None:
    rel = next(
        (
            rel
            for rel in cfg.relationships
            if rel.get("child_table_name") == table_name
        ),
        None,
    )
    if rel:
        return rel.get("child_foreign_key")
    table = cfg.tables[table_name]
    if table.id_cols:
        return table.id_cols[0]
    return table.primary_key


def _evaluation_tables(cfg: Any) -> list[str]:
    children = list(dict.fromkeys(
        rel["child_table_name"]
        for rel in cfg.relationships
        if rel.get("child_table_name") in cfg.tables
    ))
    if children:
        return children
    return list(cfg.tables)


def _infer_time_bin(
    real_bundle: dict[str, pd.DataFrame],
    table_name: str,
    time_col: str,
) -> str:
    series = pd.to_datetime(real_bundle[table_name][time_col], errors="coerce")
    values = series.dropna().sort_values().drop_duplicates()
    diffs = values.diff().dropna()
    if diffs.empty:
        return "1D"
    median_seconds = float(diffs.dt.total_seconds().median())
    if median_seconds >= 28 * 24 * 3600:
        return "30D"
    if median_seconds >= 7 * 24 * 3600:
        return "7D"
    if median_seconds >= 24 * 3600:
        return "1D"
    return "1H"


def _build_metric_specs(cfg: Any, real_bundle: dict[str, pd.DataFrame]) -> list[MetricSpec]:
    specs: list[MetricSpec] = []
    eval_tables = _evaluation_tables(cfg)

    for table_name in eval_tables:
        key_col = _sequence_key(cfg, table_name)
        if key_col:
            specs.append(
                MetricSpec(
                    name=f"sequence_length_similarity__{table_name}",
                    klass=SequenceLengthSimilarity,
                    init={"table_name": table_name, "flow_key_col": key_col},
                    requires=[table_name],
                    variants=("raw",),
                )
            )

    for rel in cfg.relationships:
        parent = rel.get("parent_table_name")
        child = rel.get("child_table_name")
        fk = rel.get("child_foreign_key")
        if parent not in cfg.tables or child not in cfg.tables or not fk:
            continue
        child_dt = _datetime_cols(cfg.tables[child])
        if child_dt:
            time_bin = _infer_time_bin(real_bundle, child, child_dt[0])
            specs.append(
                MetricSpec(
                    name=f"temporal_cardinality_shape_similarity__{child}",
                    klass=TemporalCardinalityShapeSimilarity,
                    init={
                        "parent_table": parent,
                        "child_table": child,
                        "foreign_key": fk,
                        "time_col": child_dt[0],
                        "window": time_bin,
                    },
                    requires=[child],
                    variants=("raw",),
                )
            )

        parent_nums = _numeric_cols(cfg.tables[parent])
        child_nums = _numeric_cols(cfg.tables[child])
        for parent_col in parent_nums[:1]:
            for child_col in child_nums[:5]:
                if child_dt:
                    time_bin = _infer_time_bin(real_bundle, child, child_dt[0])
                    specs.append(
                        MetricSpec(
                            name=(
                                f"dynamic_khop__{parent_col}__{child_col}"
                                .replace(" ", "_")
                            ),
                            klass=DynamicKHopCorrelationSimilarity,
                            init={
                                "variable_a": (parent, parent_col),
                                "variable_b": (child, child_col),
                                "time_col": (child, child_dt[0]),
                                "time_bin": time_bin,
                            },
                            requires=[child],
                            variants=("postprocessed",),
                        )
                    )

    parents_by_child: dict[str, list[str]] = {}
    for rel in cfg.relationships:
        parent = rel.get("parent_table_name")
        child = rel.get("child_table_name")
        if parent in cfg.tables and child in cfg.tables:
            parents_by_child.setdefault(child, []).append(parent)

    for child, parents in parents_by_child.items():
        parents = list(dict.fromkeys(parents))
        if len(parents) < 2:
            continue
        child_dt = _datetime_cols(cfg.tables[child])
        child_nums = _numeric_cols(cfg.tables[child])
        if not child_dt or not child_nums:
            continue
        time_bin = _infer_time_bin(real_bundle, child, child_dt[0])
        condition_cols: dict[str, str] = {}
        for parent in parents:
            table = cfg.tables[parent]
            cols = table.categorical_cols or table.numeric_cols
            if cols:
                condition_cols[parent] = cols[0]
        selected_parents = [p for p in parents if p in condition_cols]
        if len(selected_parents) < 2:
            continue
        specs.append(
            MetricSpec(
                name=f"multi_parent_conditional_similarity__{child}",
                klass=MultiParentConditionalSimilarity,
                init={
                    "child_table": child,
                    "parent_tables": selected_parents,
                    "target_col": child_nums[0],
                    "condition_cols": condition_cols,
                    "time_col": child_dt[0],
                    "time_bin": time_bin,
                    "n_numeric_bins": 4,
                    "min_group_size": 10,
                },
                requires=[child],
                variants=("postprocessed",),
            )
        )

    return specs


def _table_columns(path: Path) -> set[str]:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return set(pd.read_parquet(path).columns)
    return set(pd.read_csv(path, nrows=0).columns)


def _discover_real_file(cfg: Any, table_name: str) -> Path:
    expected = set(cfg.tables[table_name].columns)
    candidates = sorted(
        path
        for path in cfg.real_dir.iterdir()
        if path.suffix.lower() in {".csv", ".parquet", ".pq"}
    )

    exact = [
        path
        for path in candidates
        if path.stem == table_name and expected.issubset(_table_columns(path))
    ]
    if exact:
        return exact[0]

    matching = [
        path
        for path in candidates
        if expected and expected.issubset(_table_columns(path))
    ]
    if matching:
        return matching[0]

    raise FileNotFoundError(
        f"Could not discover real file for table '{table_name}' in {cfg.real_dir}"
    )


def _load_real_bundle(cfg: Any) -> dict[str, pd.DataFrame]:
    bundle: dict[str, pd.DataFrame] = {}
    for table_name in cfg.tables:
        path = _discover_real_file(cfg, table_name)
        logger.info("Loading real[%s] = %s", table_name, path)
        bundle[table_name] = _load_table(path)
    return bundle


def _variant_path(model_dir: Path, table_name: str, variant: str) -> Path | None:
    suffix = VARIANT_SUFFIXES[variant]
    candidates = [model_dir / f"{table_name}{suffix}.csv"]
    if variant != "raw":
        candidates.append(model_dir / f"{table_name}{VARIANT_SUFFIXES['default']}.csv")
    candidates.append(model_dir / f"{table_name}.csv")
    return next((path for path in candidates if path.exists()), None)


def _discover_variants(
    cfg: Any,
    model_dir: Path,
    variants: list[str] | None,
) -> dict[str, dict[str, str]]:
    wanted = variants or list(VARIANT_SUFFIXES)
    eval_tables = _evaluation_tables(cfg)
    discovered: dict[str, dict[str, str]] = {}
    for variant in wanted:
        if variant not in VARIANT_SUFFIXES:
            logger.warning("[%s] unknown variant '%s'; skipping", cfg.name, variant)
            continue
        table_files: dict[str, str] = {}
        for table_name in eval_tables:
            path = _variant_path(model_dir, table_name, variant)
            if path is not None:
                table_files[table_name] = path.name
        if table_files:
            discovered[variant] = table_files
    return discovered


def _load_synth_bundle(
    cfg: Any,
    model_dir: Path,
    variant_map: dict[str, str],
    real_bundle: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    bundle: dict[str, pd.DataFrame] = {}
    missing: list[str] = []

    for table_name in cfg.tables:
        if table_name in variant_map:
            path = model_dir / variant_map[table_name]
        else:
            path = model_dir / f"{table_name}.csv"
            if not path.exists():
                bundle[table_name] = real_bundle[table_name]
                continue
        if not path.exists():
            missing.append(path.name)
            continue
        bundle[table_name] = _load_table(path)
    return bundle, missing


def _run_variant(
    cfg: Any,
    schema: dict[str, Any],
    specs: list[MetricSpec],
    model: str,
    model_dir: Path,
    variant_label: str,
    variant_map: dict[str, str],
    real_bundle: dict[str, pd.DataFrame],
    out_root: Path,
) -> int:
    synth_bundle, missing = _load_synth_bundle(cfg, model_dir, variant_map, real_bundle)
    if missing:
        logger.warning(
            "[%s][%s][%s] missing synthetic files: %s",
            cfg.name,
            model,
            variant_label,
            missing,
        )

    metrics_results: dict[str, Any] = {}
    skipped_metrics: dict[str, str] = {}
    variant_kind = _metric_variant_kind(variant_label)
    for spec in specs:
        if variant_kind not in spec.variants:
            skipped_metrics[spec.name] = (
                f"metric is for {', '.join(spec.variants)} variants; "
                f"current variant is {variant_kind}"
            )
            logger.info(
                "[%s][%s][%s] %s skipped (%s-only metric)",
                cfg.name,
                model,
                variant_label,
                spec.name,
                "/".join(spec.variants),
            )
            continue
        missing_req = [table for table in spec.requires if table not in synth_bundle]
        if missing_req:
            result = {
                "score": None,
                "details": {"reason": f"missing required table(s): {missing_req}"},
            }
        else:
            try:
                metric = spec.klass(**spec.init)
                result = metric.compute(real_bundle, synth_bundle, schema)
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "[%s][%s][%s] %s failed: %s",
                    cfg.name,
                    model,
                    variant_label,
                    spec.name,
                    exc,
                )
                result = {"score": None, "details": {"reason": f"exception: {exc}"}}
        _trim_details(result)
        metrics_results[spec.name] = result
        logger.info(
            "[%s][%s][%s] %s = %s",
            cfg.name,
            model,
            variant_label,
            spec.name,
            _score_str(result.get("score")),
        )

    out_dir = out_root / cfg.name / model / "structural"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"struct_{variant_label}.json"
    payload = {
        "summary": _mean_scores_by_family(metrics_results),
        "per_metric": _per_metric_scores(metrics_results),
        "details": metrics_results,
        "metadata": {
            "dataset": cfg.name,
            "model": model,
            "variant": variant_label,
            "variant_kind": variant_kind,
            "synth_files": variant_map,
            "skipped_metrics": skipped_metrics,
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2)
    logger.info("Wrote %s", out_path)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", nargs="+", default=["all"])
    parser.add_argument("--model", nargs="+", default=["all"])
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--out_root", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run model/variant evaluations concurrently with CPU worker threads.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker threads used with --parallel.",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=None,
        help="Alias for --workers.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    datasets = resolve_datasets(args.dataset)
    worker_count = args.workers if args.workers is not None else args.n_jobs
    worker_count = worker_count or min(32, (os.cpu_count() or 1) + 4)
    exit_code = 0
    for dataset in datasets:
        cfg = load_dataset_config(dataset)
        models = (
            available_models(dataset)
            if args.model == ["all"] or "all" in args.model
            else args.model
        )
        if not models:
            logger.warning("No models under %s; skipping %s", cfg.synth_dir, dataset)
            continue

        try:
            real_bundle = _load_real_bundle(cfg)
        except FileNotFoundError as exc:
            logger.error("[%s] real files missing: %s", dataset, exc)
            exit_code = exit_code or 2
            continue

        schema = _metadata_schema(cfg)
        specs = _build_metric_specs(cfg, real_bundle)
        if not specs:
            logger.warning("[%s] no applicable structural metrics", dataset)
            continue

        tasks = []
        for model in models:
            model_dir = cfg.synth_dir / model
            if not model_dir.is_dir():
                logger.warning("Skipping missing model dir: %s", model_dir)
                continue
            variants = _discover_variants(cfg, model_dir, args.variants)
            if not variants:
                logger.warning("[%s][%s] no matching variants", dataset, model)
                continue
            for variant_label, variant_map in variants.items():
                tasks.append(
                    {
                        "cfg": cfg,
                        "schema": schema,
                        "specs": specs,
                        "model": model,
                        "model_dir": model_dir,
                        "variant_label": variant_label,
                        "variant_map": variant_map,
                        "real_bundle": real_bundle,
                        "out_root": args.out_root,
                    }
                )
        if args.parallel and len(tasks) > 1:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [executor.submit(_run_variant, **task) for task in tasks]
                for future in as_completed(futures):
                    exit_code = exit_code or future.result()
        else:
            for task in tasks:
                exit_code = exit_code or _run_variant(**task)
    return exit_code


DATASETS = {name: {} for name in available_datasets()}


if __name__ == "__main__":
    sys.exit(main())
