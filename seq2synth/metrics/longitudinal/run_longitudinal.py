from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TIMESYNTH_DIR = Path(__file__).parents[2]
BENCHMARK_DIR = TIMESYNTH_DIR.parent
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from configs import (  # noqa: E402
    CATEGORICAL_SDTYPES,
    DATETIME_SDTYPES,
    ID_SDTYPES,
    NUMERIC_SDTYPES,
    TABLE_TIME_COL_OVERRIDES,
    VARIANT_SUFFIXES,
    available_models,
    discover_synth_variants,
    load_dataset_config,
    resolve_datasets,
)
from paths import RESULTS_DIR, ROOT_DIR  # noqa: E402
from seq2synth.metrics.cross_sectional.run_cross_sectional import (  # noqa: E402
    flatten_table,
    load_data,
)
from seq2synth.metrics.longitudinal.longitudinal_metrics import (  # noqa: E402
    DEFAULT_TT_WASSERSTEIN_MAX_FLOWS,
    DEFAULT_TT_WASSERSTEIN_NUM_ITER_MAX,
    DEFAULT_TT_WASSERSTEIN_SAMPLE_SEED,
    compute_all_longitudinal_metrics,
    validate_columns,
)


PREFERRED_TIME_COLS = (
    "Date",
    "t_dat",
    "I_DATE",
    "MONTHLY REPORTING PERIOD",
    "Monthly Reporting Period",
    "starttime",
    "time_cycles",
    "start_time",
    "time",
    "Year",
)

LONGITUDINAL_TIME_OVERRIDES: dict[str, dict[str, Any]] = {
    "airbnb_tabdit": {"time_col": "row_rank", "kind": "row_rank"},
    "berka_tabdit": {"time_col": "Date", "components": ("Year", "Month", "Day")},
    "citi_bike": {"time_col": "Date", "source": "starttime"},
    "google_cluster": {
        "primary_table": "task_usage",
        "key_col": "job_id",
        "time_col": "start_time",
    },
    "rossmann_tabdit": {"time_col": "Date", "components": ("Date_month", "Date_day")},
}


def _variant_table_path(model_dir: Path, table_name: str, variant: str) -> Path:
    suffix = VARIANT_SUFFIXES[variant]
    return model_dir / f"{table_name}{suffix}.csv"


def _infer_primary_table(cfg) -> str | None:
    override = LONGITUDINAL_TIME_OVERRIDES.get(cfg.name, {})
    primary = override.get("primary_table")
    if primary in cfg.tables:
        return primary
    child_tables = cfg.child_tables
    candidates = child_tables or list(cfg.tables)
    for table_name in candidates:
        table = cfg.tables[table_name]
        if table.datetime_cols:
            return table_name
    for table_name in candidates:
        table = cfg.tables[table_name]
        if any(col in table.columns for col in PREFERRED_TIME_COLS):
            return table_name
    return candidates[0] if candidates else None


def _infer_key_col(cfg, table_name: str) -> str | None:
    override = LONGITUDINAL_TIME_OVERRIDES.get(cfg.name, {})
    key_col = override.get("key_col")
    if key_col in cfg.tables[table_name].columns:
        return key_col
    rels = [r for r in cfg.relationships if r.get("child_table_name") == table_name]
    preferred_tokens = ("user", "customer", "loan", "store", "bike", "unit", "subject")
    for rel in rels:
        key = rel.get("child_foreign_key")
        if key and any(tok in key.lower() for tok in preferred_tokens):
            return key
    if rels and rels[0].get("child_foreign_key"):
        return rels[0]["child_foreign_key"]
    table = cfg.tables[table_name]
    if table.primary_key:
        return table.primary_key
    for col in table.id_cols:
        if any(tok in col.lower() for tok in preferred_tokens):
            return col
    for col in table.columns:
        if any(tok in col.lower() for tok in preferred_tokens):
            return col
    return table.id_cols[0] if table.id_cols else None


def _infer_time_col(cfg, table_name: str) -> str | None:
    override = LONGITUDINAL_TIME_OVERRIDES.get(cfg.name)
    if override:
        return override["time_col"]
    table = cfg.tables[table_name]
    table_override = TABLE_TIME_COL_OVERRIDES.get(cfg.name, {}).get(table_name)
    if table_override and table_override in table.columns:
        return table_override
    for col in PREFERRED_TIME_COLS:
        if col in table.columns:
            return col
    if table.datetime_cols:
        return table.datetime_cols[0]
    for col in table.numeric_cols:
        lower = col.lower()
        if "time" in lower or "cycle" in lower or lower in {"step", "month", "months_balance"}:
            return col
    return None


def _apply_longitudinal_time(df: pd.DataFrame, dataset: str, key_col: str | None) -> pd.DataFrame:
    spec = LONGITUDINAL_TIME_OVERRIDES.get(dataset)
    if not spec:
        return df
    time_col = spec["time_col"]
    if time_col in df.columns:
        return df
    df = df.copy()
    if spec.get("kind") == "row_rank":
        if key_col is None or key_col not in df.columns:
            df[time_col] = np.arange(len(df))
        else:
            df[time_col] = df.groupby(key_col).cumcount()
        return df
    if spec.get("source") == "starttime" and "starttime" in df.columns:
        df[time_col] = pd.to_datetime(df["starttime"], errors="coerce").dt.strftime("%Y-%m-%d")
        return df
    components = spec.get("components", ())
    if not all(c in df.columns for c in components):
        return df
    if dataset == "rossmann_tabdit":
        df[time_col] = (
            pd.to_numeric(df["Date_month"], errors="coerce").astype("Int64").astype(str).str.zfill(2)
            + "-"
            + pd.to_numeric(df["Date_day"], errors="coerce").astype("Int64").astype(str).str.zfill(2)
        )
        return df
    year_col, month_col, day_col = components
    years = pd.to_numeric(df[year_col], errors="coerce")
    if dataset == "berka_tabdit":
        years = years.map(lambda y: y + 1900 if pd.notna(y) and 0 <= y < 100 else y)
    months = pd.to_numeric(df[month_col], errors="coerce")
    days = pd.to_numeric(df[day_col], errors="coerce")
    valid = years.notna() & months.notna() & days.notna()
    df[time_col] = pd.Series(pd.NA, index=df.index, dtype="string")
    df.loc[valid, time_col] = (
        years[valid].astype(int).astype(str).str.zfill(4)
        + "-"
        + months[valid].astype(int).astype(str).str.zfill(2)
        + "-"
        + days[valid].astype(int).astype(str).str.zfill(2)
    )
    return df


def _coerce_time(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    if time_col not in df.columns:
        return df
    df = df.copy()
    if pd.api.types.is_numeric_dtype(df[time_col]):
        df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
        return df
    converted = pd.to_datetime(df[time_col], errors="coerce")
    if converted.notna().mean() >= 0.8:
        df[time_col] = converted
        return df
    numeric = pd.to_numeric(df[time_col], errors="coerce")
    if numeric.notna().mean() >= 0.8:
        df[time_col] = numeric
    return df


def _ancestor_tables(cfg, primary_table: str) -> list[str]:
    tables = [primary_table]
    queue = [primary_table]
    seen = {primary_table}
    while queue:
        child = queue.pop(0)
        for rel in cfg.relationships:
            if rel.get("child_table_name") != child:
                continue
            parent = rel.get("parent_table_name")
            if parent not in cfg.tables or parent in seen:
                continue
            seen.add(parent)
            tables.append(parent)
            queue.append(parent)
    return tables


def _feature_columns(cfg, table_names: list[str], key_col: str | None, time_col: str) -> tuple[list[str], list[str]]:
    exclude = {time_col}
    if key_col:
        exclude.add(key_col)
    override = LONGITUDINAL_TIME_OVERRIDES.get(cfg.name, {})
    exclude.add(override.get("source"))
    exclude.update(override.get("components", ()))
    num_cols: list[str] = []
    cat_cols: list[str] = []
    for table_name in table_names:
        table = cfg.tables[table_name]
        for col, info in table.columns.items():
            sdtype = info.get("sdtype")
            if col in exclude or sdtype in ID_SDTYPES or sdtype in DATETIME_SDTYPES:
                continue
            if sdtype in NUMERIC_SDTYPES and col not in num_cols:
                num_cols.append(col)
            elif sdtype in CATEGORICAL_SDTYPES and col not in cat_cols:
                cat_cols.append(col)
    return num_cols, cat_cols


def _prepare_frames(cfg, model_dir: Path, variant: str):
    primary = _infer_primary_table(cfg)
    if primary is None:
        raise ValueError(f"No table found in metadata for {cfg.name}")
    key_col = _infer_key_col(cfg, primary)
    time_col = _infer_time_col(cfg, primary)
    if key_col is None:
        raise ValueError(f"Could not infer key column for {cfg.name}/{primary}")
    if time_col is None:
        raise ValueError(f"Could not infer time column for {cfg.name}/{primary}")

    real_df = flatten_table(cfg, primary, cfg.real_dir)
    synth_df = flatten_table(cfg, primary, cfg.real_dir, model_dir, variant)
    real_df = _apply_longitudinal_time(real_df, cfg.name, key_col)
    synth_df = _apply_longitudinal_time(synth_df, cfg.name, key_col)
    real_df = _coerce_time(real_df, time_col)
    synth_df = _coerce_time(synth_df, time_col)

    num_cols, cat_cols = _feature_columns(cfg, _ancestor_tables(cfg, primary), key_col, time_col)
    num_cols = [c for c in num_cols if c in real_df.columns and c in synth_df.columns]
    cat_cols = [c for c in cat_cols if c in real_df.columns and c in synth_df.columns]
    return primary, key_col, time_col, real_df, synth_df, num_cols, cat_cols


def _nan_to_none(obj):
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    return obj


def run_dataset(
    dataset: str,
    models: list[str] | None = None,
    variants: list[str] | None = None,
    dry_run: bool = False,
    flow_independence: str = "dependent",
    max_lag: int = 12,
    n_fft_components: int = 10,
    zscore_reference: str = "real",
    fill_method: str = "edge_nearest",
    skip_tt_wasserstein: bool = False,
    compute_tt_standalone: bool = False,
    tt_num_iter_max: int = DEFAULT_TT_WASSERSTEIN_NUM_ITER_MAX,
    tt_max_flows: int | None = DEFAULT_TT_WASSERSTEIN_MAX_FLOWS,
    tt_sample_seed: int = DEFAULT_TT_WASSERSTEIN_SAMPLE_SEED,
) -> int:
    cfg = load_dataset_config(dataset)
    primary = _infer_primary_table(cfg)
    if primary is None:
        print(f"[WARN] No primary table inferred for {dataset}")
        return 0
    selected_models = available_models(dataset) if not models or "all" in models else models
    exit_code = 0
    for model in selected_models:
        model_dir = cfg.synth_dir / model
        if not model_dir.exists():
            print(f"[WARN] Missing model directory: {model_dir}")
            continue
        detected = discover_synth_variants(model_dir, primary)
        if variants:
            detected = [(v, p) for v, p in detected if v in variants]
        if not detected:
            print(f"[WARN] No synthetic variants for {dataset}/{model}/{primary}")
            continue
        for variant, synth_path in detected:
            output_dir = RESULTS_DIR / dataset / model / "longitudinal"
            output_path = output_dir / f"longitudinal_{variant}.json"
            if dry_run:
                print(
                    f"[DRY RUN] {dataset}/{model}/{variant}: "
                    f"primary={primary} synth={synth_path.relative_to(ROOT_DIR)} "
                    f"output={output_path.relative_to(ROOT_DIR)}"
                )
                continue
            try:
                print(f"========== [{dataset}] {model} / {variant} ==========")
                _, key_col, time_col, real_df, synth_df, num_cols, cat_cols = _prepare_frames(
                    cfg, model_dir, variant
                )
                if not num_cols and not cat_cols:
                    print(f"[WARN] No common feature columns for {dataset}/{model}/{variant}")
                    continue
                validate_columns(real_df, key_col, time_col, num_cols, cat_cols, "Real")
                validate_columns(synth_df, key_col, time_col, num_cols, cat_cols, "Synth")
                results = compute_all_longitudinal_metrics(
                    real_df,
                    synth_df,
                    key_col,
                    time_col,
                    num_cols,
                    cat_cols,
                    flow_independence,
                    max_lag,
                    n_fft_components,
                    zscore_reference=zscore_reference,
                    fill_method=fill_method,
                    skip_tt_wasserstein=skip_tt_wasserstein,
                    compute_tt_standalone=compute_tt_standalone,
                    tt_num_iter_max=tt_num_iter_max,
                    tt_max_flows=tt_max_flows,
                    tt_sample_seed=tt_sample_seed,
                )
                results["metadata"].update(
                    {
                        "dataset": dataset,
                        "model": model,
                        "variant": variant,
                        "primary_table": primary,
                        "key_col": key_col,
                        "time_col": time_col,
                        "num_cols": num_cols,
                        "cat_cols": cat_cols,
                        "skip_tt_wasserstein": skip_tt_wasserstein,
                        "compute_tt_standalone": compute_tt_standalone,
                        "tt_num_iter_max": tt_num_iter_max,
                        "tt_max_flows": tt_max_flows,
                        "tt_sample_seed": tt_sample_seed,
                    }
                )
                output_dir.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(_nan_to_none(results), f, indent=2, default=str)
                print(f"[INFO] Saved: {output_path}")
            except Exception as exc:
                exit_code = 1
                print(f"[ERROR] {dataset}/{model}/{variant} failed: {exc}", file=sys.stderr)
    return exit_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=None, help="Datasets to evaluate, or all.")
    parser.add_argument("--models", nargs="+", default=None, help="Model folders to run, or all.")
    parser.add_argument("--variants", nargs="+", default=None, help="Postprocessing variants to run.")
    parser.add_argument("--dry_run", action="store_true", help="Print planned runs without executing.")
    parser.add_argument(
        "--flow_independence",
        choices=["independent", "dependent"],
        default="dependent",
        help="Flow independence mode for AutoCorrelation.",
    )
    parser.add_argument("--max_lag", type=int, default=12)
    parser.add_argument("--n_fft_components", type=int, default=10)
    parser.add_argument("--zscore_reference", choices=["real", "pooled", "none"], default="real")
    parser.add_argument("--fill_method", default="edge_nearest")
    parser.add_argument(
        "--skip_tt_wasserstein",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip TT-Wasserstein computation.",
    )
    parser.add_argument(
        "--tt_standalone",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Compute per-feature standalone TT-Wasserstein distances. "
            "Default is off to keep the main TT distance and component "
            "contributions while skipping extra per-feature OT solves."
        ),
    )
    parser.add_argument(
        "--tt_num_iter_max",
        type=int,
        default=DEFAULT_TT_WASSERSTEIN_NUM_ITER_MAX,
        help="POT EMD numItermax for exact TT-Wasserstein.",
    )
    parser.add_argument(
        "--tt_max_flows",
        type=int,
        default=DEFAULT_TT_WASSERSTEIN_MAX_FLOWS,
        help=(
            "Maximum trajectories per side for exact TT-Wasserstein. "
            "Use 0 to disable deterministic TT-W trajectory sampling."
        ),
    )
    parser.add_argument(
        "--tt_sample_seed",
        type=int,
        default=DEFAULT_TT_WASSERSTEIN_SAMPLE_SEED,
        help="Seed for deterministic TT-W trajectory sampling.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run datasets concurrently with CPU worker threads.",
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    targets = resolve_datasets(args.datasets or ["all"])
    worker_count = args.workers if args.workers is not None else args.n_jobs
    worker_count = worker_count or min(32, (os.cpu_count() or 1) + 4)

    def _run_target(dataset: str) -> int:
        return run_dataset(
            dataset,
            models=args.models,
            variants=args.variants,
            dry_run=args.dry_run,
            flow_independence=args.flow_independence,
            max_lag=args.max_lag,
            n_fft_components=args.n_fft_components,
            zscore_reference=args.zscore_reference,
            fill_method=args.fill_method,
            skip_tt_wasserstein=args.skip_tt_wasserstein,
            compute_tt_standalone=args.tt_standalone,
            tt_num_iter_max=args.tt_num_iter_max,
            tt_max_flows=args.tt_max_flows,
            tt_sample_seed=args.tt_sample_seed,
        )

    exit_code = 0
    if args.parallel and len(targets) > 1:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            for result in executor.map(_run_target, targets):
                exit_code = exit_code or result
    else:
        for dataset in targets:
            exit_code = exit_code or _run_target(dataset)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
