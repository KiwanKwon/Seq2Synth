#!/usr/bin/env python3
"""Aggregate Seq2Synth result JSON files into a benchmark-style CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).parent
RESULTS_ROOT = ROOT / "results"
DEFAULT_VARIANT_OUTPUT = RESULTS_ROOT / "benchmark_results.csv"
DEFAULT_AGGREGATED_OUTPUT = RESULTS_ROOT / "aggregated_main.csv"

DEFAULT_VARIANT = "default"
LONGITUDINAL_DEFAULT_VARIANT = "interp_fill"
RAW_VARIANT = "raw"
KNOWN_VARIANTS = (
    "interp_fill",
    "prev_fill",
    "id_fill",
    "default",
    "raw",
)

STRUCTURAL_METRIC_ALIASES = {
    "sequence_length_similarity": "SequenceLengthSimilarity",
    "temporal_cardinality_shape_similarity": "TemporalCardinalityShapeSimilarity",
    "multi_parent_conditional_similarity": "MultiParentConditionalSimilarity",
    "dynamic_khop": "DynamicKHopCorrelationSimilarity",
}


METRIC_GROUPS = [
    (
        "timestamp",
        "ts",
        [
            "TemporalOrderConsistency",
            "TimestampUniqueness",
            "TrajectoryDurationSimilarity",
            "TemporalRangeCompliance",
            "RegularityConsistency",
            "GridCompleteness",
            "TimeIntervalDistributionSim",
        ],
    ),
    (
        "sd",
        "sd",
        [
            "KSComplement",
            "TVComplement",
            "StatisticSimilarity",
            "CorrelationSim",
            "RangeCoverage",
            "CategoryCoverage",
            "ContingencySim",
            "StatisticMSAS",
        ],
    ),
    (
        "cross_sectional",
        "cs",
        [
            "CS-KSComplement",
            "CS-TVComplement",
            "CS-StatisticSimilarity",
            "CS-CorrelationSim",
            "CS-RangeCoverage",
            "CS-CategoryCoverage",
            "CS-ContingencySim",
        ],
    ),
    (
        "structural",
        "struct",
        [
            "SequenceLengthSimilarity",
            "TemporalCardinalityShapeSimilarity",
            "MultiParentConditionalSimilarity",
            "DynamicKHopCorrelationSimilarity",
        ],
    ),
    (
        "longitudinal",
        "long",
        [
            "FirstDifferenceKSComplement",
            "TransitionMatrixTVDComplement",
            "AutoCorrelationSimilarity",
            "TTWassersteinDistance",
        ],
    ),
    (
        "trajectory_privacy",
        "tprivacy",
        [
            "dcr",
            "nndr",
            "cs_dcr",
            "cs_nndr",
            "ngram_n1",
            "ngram_n3",
        ],
    ),
]

AGGREGATED_GROUPS = ("sd", "ts", "cs", "long", "struct", "tprivacy")


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def numeric_or_blank(value: Any) -> Any:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isnan(value) or math.isinf(value):
            return ""
        return value
    return ""


def mean_numeric(values: list[Any]) -> Any:
    nums = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not nums:
        return ""
    return sum(nums) / len(nums)


def is_numeric(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and not math.isnan(value)
        and not math.isinf(value)
    )


def structural_metric_family(metric_name: str) -> str | None:
    if metric_name in STRUCTURAL_METRIC_ALIASES.values():
        return metric_name

    normalized = metric_name.strip().lower()
    for prefix, family in STRUCTURAL_METRIC_ALIASES.items():
        if normalized == prefix:
            return family
        if normalized.startswith(f"{prefix}__") or normalized.startswith(f"{prefix}_"):
            return family
        if metric_name.startswith(f"{family}_"):
            return family
    return None


def aggregate_structural_payload(payload: dict[str, Any]) -> dict[str, Any]:
    summary_groups: dict[str, list[Any]] = defaultdict(list)
    for metric, value in (payload.get("summary", {}) or {}).items():
        family = structural_metric_family(metric)
        if family and is_numeric(value):
            summary_groups[family].append(value)

    aggregated = {
        family: mean_numeric(values)
        for family, values in summary_groups.items()
    }

    per_metric_groups: dict[str, list[Any]] = defaultdict(list)
    for metric, value in (payload.get("per_metric", {}) or {}).items():
        family = structural_metric_family(metric)
        if isinstance(value, dict):
            for child_metric, child_value in value.items():
                child_family = structural_metric_family(child_metric) or family
                if child_family and is_numeric(child_value):
                    per_metric_groups[child_family].append(child_value)
        elif family and is_numeric(value):
            per_metric_groups[family].append(value)

    if per_metric_groups:
        aggregated.update(
            {
                family: mean_numeric(values)
                for family, values in per_metric_groups.items()
            }
        )
        return aggregated

    detail_groups: dict[str, list[Any]] = defaultdict(list)
    for metric, value in (payload.get("details", {}) or {}).items():
        family = structural_metric_family(metric)
        if isinstance(value, dict) and family and is_numeric(value.get("score")):
            detail_groups[family].append(value["score"])
    aggregated.update(
        {
            family: mean_numeric(values)
            for family, values in detail_groups.items()
        }
    )
    return aggregated


def variant_from_metric_file(path: Path, prefix: str) -> str | None:
    stem = path.stem
    if not stem.startswith(prefix):
        return None
    return stem.removeprefix(prefix)


def sd_variant_from_file(path: Path) -> str | None:
    stem = path.stem
    if not stem.startswith("sd_metrics_"):
        return None
    for variant in KNOWN_VARIANTS:
        if stem == f"sd_metrics_{variant}" or stem.endswith(f"_{variant}"):
            return variant
    return stem.rsplit("_", 1)[-1]


def variant_from_path(path: Path) -> str | None:
    stem = path.stem
    for variant in KNOWN_VARIANTS:
        if stem == variant or stem.endswith(f"_{variant}"):
            return variant
    return None


def discover_rows(results_root: Path) -> list[tuple[str, str]]:
    rows: set[tuple[str, str]] = set()
    for model_dir in results_root.glob("*/*"):
        if not model_dir.is_dir():
            continue
        dataset = model_dir.parent.name
        model = model_dir.name
        rows.add((dataset, model))

    return sorted(rows)


def discover_variant_rows(results_root: Path) -> list[tuple[str, str, str]]:
    rows: set[tuple[str, str, str]] = set()
    for model_dir in results_root.glob("*/*"):
        if not model_dir.is_dir():
            continue
        variants = {
            variant
            for path in model_dir.glob("*/*.json")
            for variant in [variant_from_path(path)]
            if variant is not None
        }
        if (model_dir / "timestamp" / f"{model_dir.name}.json").exists():
            variants.add(RAW_VARIANT)
        if not variants:
            variants = {DEFAULT_VARIANT}
        for variant in variants:
            rows.add((model_dir.parent.name, model_dir.name, variant))

    variant_order = {variant: idx for idx, variant in enumerate(KNOWN_VARIANTS)}
    return sorted(
        rows,
        key=lambda row: (
            row[0],
            row[1],
            variant_order.get(row[2], len(KNOWN_VARIANTS)),
            row[2],
        ),
    )


def read_single_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return load_json(path).get("summary", {}) or {}


def read_first_summary(paths: list[Path]) -> dict[str, Any]:
    for path in paths:
        summary = read_single_summary(path)
        if summary:
            return summary
    return {}


def normalize_trajectory_privacy_summary(summary: dict[str, Any]) -> dict[str, Any]:
    summary = dict(summary)
    if "cs_dcr" not in summary and "temporal_dcr" in summary:
        summary["cs_dcr"] = summary["temporal_dcr"]
    if "cs_nndr" not in summary and "temporal_nndr" in summary:
        summary["cs_nndr"] = summary["temporal_nndr"]
    return summary


def read_structural_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return aggregate_structural_payload(load_json(path))


def read_sd_summary(model_dir: Path, variant: str) -> dict[str, Any]:
    summaries = []
    for path in sorted((model_dir / "sd").glob(f"sd_metrics_*_{variant}.json")):
        summaries.append(load_json(path).get("summary", {}) or {})

    direct = model_dir / "sd" / f"sd_metrics_{variant}.json"
    if direct.exists():
        summaries.append(load_json(direct).get("summary", {}) or {})

    merged: dict[str, list[Any]] = defaultdict(list)
    for summary in summaries:
        for metric, value in summary.items():
            merged[metric].append(value)
    return {metric: mean_numeric(values) for metric, values in merged.items()}


def read_structural_summary(model_dir: Path) -> dict[str, Any]:
    raw_summary = read_structural_file(model_dir / "structural" / f"struct_{RAW_VARIANT}.json")
    default_summary = read_structural_file(model_dir / "structural" / f"struct_{DEFAULT_VARIANT}.json")
    return {
        "SequenceLengthSimilarity": raw_summary.get("SequenceLengthSimilarity"),
        "TemporalCardinalityShapeSimilarity": raw_summary.get("TemporalCardinalityShapeSimilarity"),
        "MultiParentConditionalSimilarity": default_summary.get("MultiParentConditionalSimilarity"),
        "DynamicKHopCorrelationSimilarity": default_summary.get("DynamicKHopCorrelationSimilarity"),
    }


def read_structural_variant_summary(model_dir: Path, variant: str) -> dict[str, Any]:
    if variant == RAW_VARIANT:
        raw_summary = read_structural_file(model_dir / "structural" / f"struct_{RAW_VARIANT}.json")
        return {
            "SequenceLengthSimilarity": raw_summary.get("SequenceLengthSimilarity"),
            "TemporalCardinalityShapeSimilarity": raw_summary.get("TemporalCardinalityShapeSimilarity"),
        }

    if variant == DEFAULT_VARIANT:
        default_summary = read_structural_file(model_dir / "structural" / f"struct_{DEFAULT_VARIANT}.json")
        return {
            "MultiParentConditionalSimilarity": default_summary.get("MultiParentConditionalSimilarity"),
            "DynamicKHopCorrelationSimilarity": default_summary.get("DynamicKHopCorrelationSimilarity"),
        }

    return {}


def read_longitudinal_summary(model_dir: Path, variant: str) -> dict[str, Any]:
    summary = read_single_summary(model_dir / "longitudinal" / f"longitudinal_{variant}.json")
    if "TransitionMatrixTVDComplement" not in summary and "TransitionMatrixSimilarity" in summary:
        summary = dict(summary)
        summary["TransitionMatrixTVDComplement"] = summary["TransitionMatrixSimilarity"]
    return summary


def read_timestamp_variant_summary(model_dir: Path, model: str, variant: str) -> dict[str, Any]:
    paths = [
        model_dir / "timestamp" / f"{model}_{variant}.json",
        model_dir / "timestamp" / f"{variant}.json",
    ]
    if variant == RAW_VARIANT:
        paths.append(model_dir / "timestamp" / f"{model}.json")
    return read_first_summary(paths)


def read_privacy_variant_summary(model_dir: Path, variant: str) -> dict[str, Any]:
    return normalize_trajectory_privacy_summary(
        read_first_summary(
            [
                model_dir / "trajectory_privacy" / f"{variant}.json",
                model_dir / "trajectory_privacy" / f"privacy_{variant}.json",
            ]
        )
    )


def collect_summaries(results_root: Path, dataset: str, model: str) -> dict[str, dict[str, Any]]:
    model_dir = results_root / dataset / model
    summaries = {
        "timestamp": read_timestamp_variant_summary(model_dir, model, RAW_VARIANT),
        "sd": read_sd_summary(model_dir, DEFAULT_VARIANT),
        "cross_sectional": read_single_summary(model_dir / "cross_sectional" / f"cs_metrics_{DEFAULT_VARIANT}.json"),
        "structural": read_structural_summary(model_dir),
        "longitudinal": read_longitudinal_summary(model_dir, LONGITUDINAL_DEFAULT_VARIANT),
        "trajectory_privacy": read_privacy_variant_summary(model_dir, DEFAULT_VARIANT),
    }
    return summaries


def collect_variant_summaries(
    results_root: Path,
    dataset: str,
    model: str,
    variant: str,
) -> dict[str, dict[str, Any]]:
    model_dir = results_root / dataset / model
    return {
        "timestamp": read_timestamp_variant_summary(model_dir, model, variant),
        "sd": read_sd_summary(model_dir, variant),
        "cross_sectional": read_single_summary(
            model_dir / "cross_sectional" / f"cs_metrics_{variant}.json"
        ),
        "structural": read_structural_variant_summary(model_dir, variant),
        "longitudinal": read_longitudinal_summary(model_dir, variant),
        "trajectory_privacy": read_privacy_variant_summary(model_dir, variant),
    }


def row_group_average(row: dict[str, Any], prefix: str) -> Any:
    values = [
        value
        for column, value in row.items()
        if column.startswith(f"{prefix}_") and is_numeric(value)
    ]
    return mean_numeric(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--variant-output", type=Path, default=DEFAULT_VARIANT_OUTPUT)
    parser.add_argument("--aggregated-output", type=Path, default=DEFAULT_AGGREGATED_OUTPUT)
    args = parser.parse_args()

    rows = discover_rows(args.results_root)
    columns = ["dataset", "model"]
    for _, prefix, metrics in METRIC_GROUPS:
        columns.extend(f"{prefix}_{metric}" for metric in metrics)
    variant_rows = discover_variant_rows(args.results_root)
    variant_columns = ["dataset", "model", "variant", *columns[2:]]
    aggregated_columns = ["dataset", "model"]
    aggregated_columns.extend(f"{prefix}_avg" for prefix in AGGREGATED_GROUPS)

    args.variant_output.parent.mkdir(parents=True, exist_ok=True)
    args.aggregated_output.parent.mkdir(parents=True, exist_ok=True)
    aggregated_rows: list[dict[str, Any]] = []
    for dataset, model in rows:
        summaries = collect_summaries(args.results_root, dataset, model)
        row: dict[str, Any] = {"dataset": dataset, "model": model}
        for group, prefix, metrics in METRIC_GROUPS:
            summary = summaries[group]
            for metric in metrics:
                row[f"{prefix}_{metric}"] = numeric_or_blank(summary.get(metric))
        aggregated_rows.append(
            {
                "dataset": dataset,
                "model": model,
                **{
                    f"{prefix}_avg": row_group_average(row, prefix)
                    for prefix in AGGREGATED_GROUPS
                },
            }
        )

    with args.variant_output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=variant_columns)
        writer.writeheader()
        for dataset, model, variant in variant_rows:
            summaries = collect_variant_summaries(args.results_root, dataset, model, variant)
            row: dict[str, Any] = {"dataset": dataset, "model": model, "variant": variant}
            for group, prefix, metrics in METRIC_GROUPS:
                summary = summaries[group]
                for metric in metrics:
                    row[f"{prefix}_{metric}"] = numeric_or_blank(summary.get(metric))
            writer.writerow(row)

    with args.aggregated_output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=aggregated_columns)
        writer.writeheader()
        writer.writerows(aggregated_rows)

    print(f"Wrote {len(variant_rows)} rows to {args.variant_output}")
    print(f"Wrote {len(aggregated_rows)} rows to {args.aggregated_output}")


if __name__ == "__main__":
    main()
