from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

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
    TableConfig,
    available_datasets,
    available_models,
    discover_synth_variants,
    load_dataset_config,
    resolve_datasets,
)
from paths import RESULTS_DIR, ROOT_DIR  # noqa: E402


PREFERRED_TIME_COLS = (
    "Date",
    "date_first_booking",
    "t_dat",
    "I_DATE",
    "MONTHLY REPORTING PERIOD",
    "Monthly Reporting Period",
    "starttime",
    "time_cycles",
    "start_time",
    "time",
)

DERIVED_TIME_SPECS: dict[str, dict[str, Any]] = {
    "airbnb_tabdit": {
        "time_col": "date_first_booking",
        "components": (
            "date_first_booking_year",
            "date_first_booking_month",
            "date_first_booking_day",
        ),
    },
    "berka_tabdit": {"time_col": "Date", "components": ("Year", "Month", "Day")},
    "rossmann_tabdit": {"time_col": "Date", "components": ("Date_month", "Date_day")},
    "citi_bike": {"time_col": "Date", "source": "starttime"},
}


def _sniff_format(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".parquet", ".pq"}:
        return "parquet"
    if path.is_file():
        with open(path, "rb") as f:
            if f.read(4) == b"PAR1":
                return "parquet"
    return "csv"


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if _sniff_format(path) == "parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _table_path(base_dir: Path, table: TableConfig, variant: str | None = None) -> Path:
    if variant is None or variant == "raw":
        return base_dir / table.filename
    suffix = VARIANT_SUFFIXES[variant]
    return base_dir / f"{table.name}{suffix}.csv"


def _load_table(
    table: TableConfig,
    real_dir: Path,
    synth_dir: Path | None = None,
    variant: str | None = None,
) -> pd.DataFrame:
    if synth_dir is not None:
        synth_path = _table_path(synth_dir, table, variant)
        if synth_path.exists():
            return load_data(synth_path)
        raw_synth_path = _table_path(synth_dir, table, "raw")
        if raw_synth_path.exists():
            return load_data(raw_synth_path)
    return load_data(real_dir / table.filename)


def _normalize_merge_key(series: pd.Series) -> pd.Series:
    if pd.api.types.is_integer_dtype(series):
        return series.astype("Int64").astype("string")
    if pd.api.types.is_float_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        finite = numeric.dropna()
        if finite.empty or np.all(np.isclose(finite, np.round(finite))):
            return numeric.round().astype("Int64").astype("string")
        return numeric.astype("string")
    return series.astype("string").str.strip()


def _is_integer_like(series: pd.Series) -> bool:
    if pd.api.types.is_integer_dtype(series):
        return True
    if not pd.api.types.is_float_dtype(series):
        return False
    numeric = pd.to_numeric(series.dropna(), errors="coerce")
    if numeric.empty or numeric.isna().any():
        return False
    return bool(np.all(np.isclose(numeric, np.round(numeric))))


def _categorical_labels(series: pd.Series, *, integer_like: bool) -> pd.Series:
    if not integer_like:
        return series.astype("string").str.strip()
    numeric = pd.to_numeric(series, errors="coerce")
    if series.notna().any() and numeric[series.notna()].isna().any():
        return series.astype("string").str.strip()
    finite = numeric.dropna()
    if not finite.empty and not np.all(np.isclose(finite, np.round(finite))):
        return numeric.astype("string")
    return numeric.round().astype("Int64").astype("string")


def _is_boolean_like(series: pd.Series) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return False
    if pd.api.types.is_bool_dtype(non_null):
        return True
    labels = non_null.astype("string").str.strip().str.lower()
    return bool(labels.isin({"true", "false", "1", "0", "1.0", "0.0"}).all())


def _boolean_categorical_labels(series: pd.Series) -> pd.Series:
    labels = series.astype("string").str.strip()
    lowered = labels.str.lower()
    mapped = labels.copy()
    mapped = mapped.mask(lowered.isin({"true", "1", "1.0"}), "True")
    mapped = mapped.mask(lowered.isin({"false", "0", "0.0"}), "False")
    return mapped


def _align_categorical_columns(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    cat_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not cat_cols:
        return real_df, synth_df
    real_df = real_df.copy()
    synth_df = synth_df.copy()
    for col in cat_cols:
        if _is_boolean_like(real_df[col]):
            real_df[col] = _boolean_categorical_labels(real_df[col])
            synth_df[col] = _boolean_categorical_labels(synth_df[col])
            continue
        integer_like = _is_integer_like(real_df[col])
        real_df[col] = _categorical_labels(real_df[col], integer_like=integer_like)
        synth_df[col] = _categorical_labels(synth_df[col], integer_like=integer_like)
    return real_df, synth_df


def _merge_parent(child: pd.DataFrame, parent: pd.DataFrame, child_key: str, parent_key: str) -> pd.DataFrame:
    parent_cols = [parent_key] + [c for c in parent.columns if c != parent_key and c not in child.columns]
    left = child.copy()
    right = parent[parent_cols].copy()
    if left[child_key].dtype != right[parent_key].dtype:
        left[child_key] = _normalize_merge_key(left[child_key])
        right[parent_key] = _normalize_merge_key(right[parent_key])
    return left.merge(right, left_on=child_key, right_on=parent_key, how="left")


def flatten_table(
    cfg,
    table_name: str,
    real_dir: Path,
    synth_dir: Path | None = None,
    variant: str | None = None,
    seen: set[str] | None = None,
) -> pd.DataFrame:
    seen = set() if seen is None else set(seen)
    if table_name in seen:
        return _load_table(cfg.tables[table_name], real_dir, synth_dir, variant)
    seen.add(table_name)

    table = cfg.tables[table_name]
    df = _load_table(table, real_dir, synth_dir, variant)
    for rel in cfg.relationships:
        if rel.get("child_table_name") != table_name:
            continue
        parent_name = rel.get("parent_table_name")
        child_key = rel.get("child_foreign_key")
        parent_key = rel.get("parent_primary_key")
        if parent_name not in cfg.tables or child_key not in df.columns:
            continue
        parent = flatten_table(cfg, parent_name, real_dir, synth_dir, variant, seen)
        if parent_key in parent.columns:
            df = _merge_parent(df, parent, child_key, parent_key)
    return df


def _infer_primary_table(cfg) -> str | None:
    child_tables = cfg.child_tables
    if not child_tables:
        return next(iter(cfg.tables), None)
    for table_name in child_tables:
        table = cfg.tables[table_name]
        if table.datetime_cols or any(c in table.columns for c in PREFERRED_TIME_COLS):
            return table_name
    return child_tables[0]


def _infer_key_col(cfg, table_name: str) -> str | None:
    rels = [r for r in cfg.relationships if r.get("child_table_name") == table_name]
    preferred_tokens = ("user", "customer", "loan", "store", "bike", "unit")
    for rel in rels:
        key = rel.get("child_foreign_key")
        if key and any(tok in key.lower() for tok in preferred_tokens):
            return key
    if rels and rels[0].get("child_foreign_key"):
        return rels[0]["child_foreign_key"]
    table = cfg.tables[table_name]
    return table.primary_key or (table.id_cols[0] if table.id_cols else None)


def _infer_time_col(cfg, table_name: str) -> str | None:
    configured = (getattr(cfg, "table_time_cols", None) or {}).get(table_name)
    if configured and configured in cfg.tables[table_name].columns:
        return configured
    derived = DERIVED_TIME_SPECS.get(cfg.name)
    if derived:
        return derived["time_col"]
    table = cfg.tables[table_name]
    override = TABLE_TIME_COL_OVERRIDES.get(cfg.name, {}).get(table_name)
    if override and override in table.columns:
        return override
    for col in PREFERRED_TIME_COLS:
        if col in table.columns:
            return col
    if table.datetime_cols:
        return table.datetime_cols[0]
    for col in table.numeric_cols:
        lower = col.lower()
        if "time" in lower or "cycle" in lower or lower in {"step", "month"}:
            return col
    return None


def _apply_derived_time(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    spec = DERIVED_TIME_SPECS.get(dataset)
    if not spec:
        return df
    time_col = spec["time_col"]
    if time_col in df.columns:
        return df
    df = df.copy()
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
    non_null = df[time_col].notna()
    parse_rate = converted[non_null].notna().mean() if non_null.any() else 0.0
    if parse_rate >= 0.8:
        df[time_col] = converted
    return df


def _source_feature_columns(cfg, table_names: list[str], time_col: str, key_col: str | None) -> tuple[list[str], list[str]]:
    exclude = {time_col}
    if key_col:
        exclude.add(key_col)
    derived = DERIVED_TIME_SPECS.get(cfg.name, {})
    exclude.update(derived.get("components", ()))
    exclude.add(derived.get("source"))

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


def _related_tables(cfg, primary_table: str) -> list[str]:
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

def _add_sibling_aggregates(
    cfg,
    df: pd.DataFrame,
    real_dir: Path,
    primary_table: str,
    key_col: str | None,
    time_col: str,
    synth_dir: Path | None = None,
    variant: str | None = None,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    if key_col is None or key_col not in df.columns:
        return df, [], {}
    return _merge_sibling_aggregates(
        cfg,
        df,
        real_dir,
        primary_table,
        key_col,
        time_col,
        synth_dir=synth_dir,
        variant=variant,
    )


def _merge_sibling_aggregates(
    cfg,
    df: pd.DataFrame,
    real_dir: Path,
    primary_table: str,
    key_col: str,
    time_col: str,
    synth_dir: Path | None = None,
    variant: str | None = None,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    added: list[str] = []
    diagnostics: dict[str, Any] = {}
    join_keys = [key_col]
    for sibling_name in cfg.child_tables:
        if sibling_name == primary_table or sibling_name not in cfg.tables:
            continue
        sibling_table = cfg.tables[sibling_name]
        if key_col not in sibling_table.columns:
            continue
        sibling = _load_table(sibling_table, real_dir, synth_dir, variant)
        numeric_cols = [
            c
            for c in sibling_table.numeric_cols
            if c not in {key_col, time_col} and c in sibling.columns and c not in df.columns
        ]
        if not numeric_cols:
            continue
        if df[key_col].dtype != sibling[key_col].dtype:
            df = df.copy()
            sibling = sibling.copy()
            df[key_col] = _normalize_merge_key(df[key_col])
            sibling[key_col] = _normalize_merge_key(sibling[key_col])
        agg = sibling.groupby(join_keys, as_index=False)[numeric_cols].sum()
        before_rows = len(df)
        df = df.merge(agg, on=join_keys, how="left")
        matched_rows = int(df[numeric_cols[0]].notna().sum()) if numeric_cols else 0
        diagnostics[sibling_name] = {
            "join_mode": "id_only",
            "join_keys": join_keys,
            "numeric_cols": numeric_cols,
            "matched_rows": matched_rows,
            "total_rows": int(before_rows),
            "coverage": float(matched_rows / before_rows) if before_rows else 0.0,
        }
        added.extend(numeric_cols)
    return df, added, diagnostics


def _filter_numeric_columns_with_partition_support(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    time_col: str,
    num_cols: list[str],
) -> list[str]:
    real_times = set(real_df[time_col].dropna().unique())
    synth_times = set(synth_df[time_col].dropna().unique())
    common_times = real_times & synth_times
    if not common_times:
        return num_cols
    real_eval = real_df[real_df[time_col].isin(common_times)]
    synth_eval = synth_df[synth_df[time_col].isin(common_times)]
    supported: list[str] = []
    for col in num_cols:
        real_values = pd.to_numeric(real_eval[col], errors="coerce").dropna()
        synth_values = pd.to_numeric(synth_eval[col], errors="coerce").dropna()
        if len(real_values) > 0 and len(synth_values) > 0:
            supported.append(col)
    return supported


def get_eval_timestamps(real_df: pd.DataFrame, time_col: str) -> list[Any]:
    return sorted(real_df[time_col].dropna().unique())


def get_cross_sectional_slice(grouped, t, feature_col: str) -> pd.Series:
    if t not in grouped.groups:
        return pd.Series(dtype=float)
    group = grouped.get_group(t)
    if feature_col not in group.columns:
        return pd.Series(dtype=float)
    return group[feature_col].dropna()


def cs__ks_complement(real_grouped, synth_grouped, times, num_col: str) -> float:
    """
    Compute cross-sectional KS complement for a numeric column.

    At each timestamp, runs a two-sample Kolmogorov-Smirnov test between
    the real and synthetic distributions of ``num_col``, then maps the KS
    statistic to a similarity score as ``1 - KS``. Averages across all
    timestamps.

    Higher values indicate better distributional fidelity. Returns NaN
    if no timestamp has data on both sides.

    Parameters
    ----------
    real_grouped : DataFrameGroupBy
        Real data grouped by time key.
    synth_grouped : DataFrameGroupBy
        Synthetic data grouped by time key.
    times : iterable
        Timestamps to evaluate.
    num_col : str
        Name of the numeric column to evaluate.
    """
    scores = []
    for t in times:
        real_slice = pd.to_numeric(get_cross_sectional_slice(real_grouped, t, num_col), errors="coerce").dropna()
        synth_slice = pd.to_numeric(get_cross_sectional_slice(synth_grouped, t, num_col), errors="coerce").dropna()
        if len(real_slice) == 0 or len(synth_slice) == 0:
            continue
        ks_stat, _ = ks_2samp(real_slice.values, synth_slice.values)
        scores.append(1.0 - ks_stat)
    return float(np.mean(scores)) if scores else np.nan


def cs__statistic_similarity(real_grouped, synth_grouped, times, num_col: str, epsilon: float = 1e-10) -> float:
    """
    Compute cross-sectional statistic similarity for a numeric column.

    For each of mean, median, and std: computes the statistic of ``num_col``
    at every timestamp for both real and synthetic data, then scores each
    timestamp as ``1 - |real_stat - synth_stat| / range_real``, clipped
    to the ``[0, 1]`` range.
    Averages per-statistic scores, then averages across the three statistics.

    Higher values indicate better alignment of summary statistics. Returns
    NaN if no usable timestamps exist.

    Parameters
    ----------
    real_grouped : DataFrameGroupBy
        Real data grouped by time key.
    synth_grouped : DataFrameGroupBy
        Synthetic data grouped by time key.
    times : iterable
        Timestamps to evaluate.
    num_col : str
        Name of the numeric column to evaluate.
    epsilon : float
        Small constant added to the range denominator to avoid division by zero.
    """
    stat_funcs = {"mean": np.mean, "median": np.median, "std": np.std}
    all_scores = []
    for stat_func in stat_funcs.values():
        real_stats = {}
        synth_stats = {}
        for t in times:
            real_slice = pd.to_numeric(get_cross_sectional_slice(real_grouped, t, num_col), errors="coerce").dropna()
            synth_slice = pd.to_numeric(get_cross_sectional_slice(synth_grouped, t, num_col), errors="coerce").dropna()
            if len(real_slice) > 0:
                real_stats[t] = stat_func(real_slice.values)
            if len(synth_slice) > 0:
                synth_stats[t] = stat_func(synth_slice.values)
        if not real_stats:
            continue
        real_values = list(real_stats.values())
        range_real = max(real_values) - min(real_values)
        scores = [
            np.clip(
                1.0 - abs(real_stats[t] - synth_stats[t]) / (range_real + epsilon),
                0.0,
                1.0,
            )
            # 1.0 - abs(real_stats[t] - synth_stats[t]) / (range_real + epsilon)
            for t in times
            if t in real_stats and t in synth_stats
        ]
        if scores:
            all_scores.append(np.mean(scores))
    return float(np.mean(all_scores)) if all_scores else np.nan


def cs__range_coverage(real_grouped, synth_grouped, times, num_col: str) -> float:
    """
    Compute cross-sectional range coverage for a numeric column.

    At each timestamp, measures the fraction of synthetic values that fall
    within the observed min–max range of the real data. Averages across
    all timestamps.

    Higher values indicate the synthetic data stays within the real value
    range. Returns NaN if no timestamp has data on both sides.

    Parameters
    ----------
    real_grouped : DataFrameGroupBy
        Real data grouped by time key.
    synth_grouped : DataFrameGroupBy
        Synthetic data grouped by time key.
    times : iterable
        Timestamps to evaluate.
    num_col : str
        Name of the numeric column to evaluate.
    """
    scores = []
    for t in times:
        real_slice = pd.to_numeric(get_cross_sectional_slice(real_grouped, t, num_col), errors="coerce").dropna()
        synth_slice = pd.to_numeric(get_cross_sectional_slice(synth_grouped, t, num_col), errors="coerce").dropna()
        if len(real_slice) == 0 or len(synth_slice) == 0:
            continue
        scores.append(((synth_slice >= real_slice.min()) & (synth_slice <= real_slice.max())).mean())
    return float(np.mean(scores)) if scores else np.nan


def cs__tv_complement(real_grouped, synth_grouped, times, cat_col: str) -> float:
    """
    Compute cross-sectional total variation complement for a categorical column.

    At each timestamp, computes the total variation distance (TVD) between the
    real and synthetic category distributions, then maps it to a similarity
    score as ``1 - TVD``. Averages across all timestamps.

    Higher values indicate better categorical distributional fidelity. Returns
    NaN if no timestamp has data on both sides.

    Parameters
    ----------
    real_grouped : DataFrameGroupBy
        Real data grouped by time key.
    synth_grouped : DataFrameGroupBy
        Synthetic data grouped by time key.
    times : iterable
        Timestamps to evaluate.
    cat_col : str
        Name of the categorical column to evaluate.
    """
    scores = []
    for t in times:
        real_slice = get_cross_sectional_slice(real_grouped, t, cat_col).astype(str)
        synth_slice = get_cross_sectional_slice(synth_grouped, t, cat_col).astype(str)
        if len(real_slice) == 0 or len(synth_slice) == 0:
            continue
        all_categories = set(real_slice.unique()) | set(synth_slice.unique())
        real_counts = real_slice.value_counts(normalize=True)
        synth_counts = synth_slice.value_counts(normalize=True)
        tvd = 0.5 * sum(abs(real_counts.get(c, 0.0) - synth_counts.get(c, 0.0)) for c in all_categories)
        scores.append(1.0 - tvd)
    return float(np.mean(scores)) if scores else np.nan


def cs__category_coverage(real_grouped, synth_grouped, times, cat_col: str) -> float:
    """
    Compute cross-sectional category coverage for a categorical column.

    At each timestamp, measures the fraction of real categories that also
    appear in the synthetic data. Averages across all timestamps.

    Higher values indicate the synthetic data reproduces a broader set of
    real categories. Returns NaN if no timestamp has real data.

    Parameters
    ----------
    real_grouped : DataFrameGroupBy
        Real data grouped by time key.
    synth_grouped : DataFrameGroupBy
        Synthetic data grouped by time key.
    times : iterable
        Timestamps to evaluate.
    cat_col : str
        Name of the categorical column to evaluate.
    """
    scores = []
    for t in times:
        real_slice = get_cross_sectional_slice(real_grouped, t, cat_col).astype(str)
        synth_slice = get_cross_sectional_slice(synth_grouped, t, cat_col).astype(str)
        if len(real_slice) == 0:
            continue
        real_cats = set(real_slice.unique())
        synth_cats = set(synth_slice.unique()) if len(synth_slice) > 0 else set()
        scores.append(len(real_cats & synth_cats) / len(real_cats))
    return float(np.mean(scores)) if scores else np.nan


def cs__correlation_similarity(real_grouped, synth_grouped, times, col_x: str, col_y: str) -> float:
    """
    Compute cross-sectional Pearson correlation similarity between two numeric columns.

    At each timestamp, computes the Pearson correlation of ``col_x`` and ``col_y``
    for both real and synthetic data, then scores their agreement as
    ``1 - |rho_real - rho_synth| / 2``. Averages across all valid timestamps.

    Timestamps with fewer than 3 rows or zero variance in either column are
    skipped. Higher values indicate better preservation of pairwise linear
    relationships. Returns NaN if no valid timestamps exist.

    Parameters
    ----------
    real_grouped : DataFrameGroupBy
        Real data grouped by time key.
    synth_grouped : DataFrameGroupBy
        Synthetic data grouped by time key.
    times : iterable
        Timestamps to evaluate.
    col_x : str
        Name of the first numeric column.
    col_y : str
        Name of the second numeric column.
    """
    scores = []
    for t in times:
        if t not in real_grouped.groups or t not in synth_grouped.groups:
            continue
        real_common = real_grouped.get_group(t)[[col_x, col_y]].apply(pd.to_numeric, errors="coerce").dropna()
        synth_common = synth_grouped.get_group(t)[[col_x, col_y]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(real_common) < 3 or len(synth_common) < 3:
            continue
        if real_common[col_x].std() == 0 or real_common[col_y].std() == 0:
            continue
        if synth_common[col_x].std() == 0 or synth_common[col_y].std() == 0:
            continue
        rho_real = real_common[col_x].corr(real_common[col_y])
        rho_synth = synth_common[col_x].corr(synth_common[col_y])
        if not np.isnan(rho_real) and not np.isnan(rho_synth):
            scores.append(1.0 - abs(rho_real - rho_synth) / 2.0)
    return float(np.mean(scores)) if scores else np.nan


def cs__contingency_similarity(real_grouped, synth_grouped, times, col_a: str, col_b: str) -> float:
    """
    Compute cross-sectional contingency (joint category) similarity between two columns.

    At each timestamp, forms the joint distribution of ``(col_a, col_b)`` pairs
    for both real and synthetic data, then scores their agreement as
    ``1 - TVD`` where TVD is the total variation distance over all pair
    combinations. Averages across all timestamps.

    Higher values indicate better preservation of pairwise categorical
    co-occurrence structure. Returns NaN if no timestamp has data on both sides.

    Parameters
    ----------
    real_grouped : DataFrameGroupBy
        Real data grouped by time key.
    synth_grouped : DataFrameGroupBy
        Synthetic data grouped by time key.
    times : iterable
        Timestamps to evaluate.
    col_a : str
        Name of the first categorical column.
    col_b : str
        Name of the second categorical column.
    """
    scores = []
    for t in times:
        if t not in real_grouped.groups or t not in synth_grouped.groups:
            continue
        real_sub = real_grouped.get_group(t)[[col_a, col_b]].dropna()
        synth_sub = synth_grouped.get_group(t)[[col_a, col_b]].dropna()
        if len(real_sub) == 0 or len(synth_sub) == 0:
            continue
        real_pairs = real_sub[col_a].astype(str) + "|||" + real_sub[col_b].astype(str)
        synth_pairs = synth_sub[col_a].astype(str) + "|||" + synth_sub[col_b].astype(str)
        real_probs = real_pairs.value_counts(normalize=True)
        synth_probs = synth_pairs.value_counts(normalize=True)
        all_pairs = set(real_probs.index) | set(synth_probs.index)
        tvd = 0.5 * sum(abs(real_probs.get(pair, 0.0) - synth_probs.get(pair, 0.0)) for pair in all_pairs)
        scores.append(1.0 - tvd)
    return float(np.mean(scores)) if scores else np.nan


def _safe_mean(values: dict[str, float]) -> float:
    nums = [v for v in values.values() if isinstance(v, (int, float)) and not np.isnan(v)]
    return float(np.mean(nums)) if nums else np.nan


def compute_all_cross_sectional_metrics(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    time_col: str,
    key_col: str | None,
    num_cols: list[str],
    cat_cols: list[str],
) -> dict[str, Any]:
    required = [time_col] + ([key_col] if key_col else []) + list(num_cols) + list(cat_cols)
    for col in required:
        if col not in real_df.columns:
            raise ValueError(f"Column '{col}' not found in real data")
        if col not in synth_df.columns:
            raise ValueError(f"Column '{col}' not found in synthetic data")

    times = get_eval_timestamps(real_df, time_col)
    if not times:
        raise ValueError("No valid timestamps found in real data")

    print(f"[INFO] Found {len(times)} unique timestamps for evaluation")
    print("[INFO] Grouping data by timestamp...")
    real_grouped = real_df.groupby(time_col)
    synth_grouped = synth_df.groupby(time_col)

    ks_scores = {c: cs__ks_complement(real_grouped, synth_grouped, times, c) for c in num_cols}
    stat_scores = {c: cs__statistic_similarity(real_grouped, synth_grouped, times, c) for c in num_cols}
    range_scores = {c: cs__range_coverage(real_grouped, synth_grouped, times, c) for c in num_cols}
    tv_scores = {c: cs__tv_complement(real_grouped, synth_grouped, times, c) for c in cat_cols}
    coverage_scores = {c: cs__category_coverage(real_grouped, synth_grouped, times, c) for c in cat_cols}
    corr_scores = {
        f"{a} x {b}": cs__correlation_similarity(real_grouped, synth_grouped, times, a, b)
        for a, b in combinations(num_cols, 2)
    }
    contingency_pairs = list(combinations(cat_cols, 2))
    cont_scores = {
        f"{a} x {b}": cs__contingency_similarity(real_grouped, synth_grouped, times, a, b)
        for a, b in contingency_pairs
    }

    return {
        "summary": {
            "CS-KSComplement": _safe_mean(ks_scores),
            "CS-TVComplement": _safe_mean(tv_scores),
            "CS-StatisticSimilarity": _safe_mean(stat_scores),
            "CS-CorrelationSim": _safe_mean(corr_scores),
            "CS-RangeCoverage": _safe_mean(range_scores),
            "CS-CategoryCoverage": _safe_mean(coverage_scores),
            "CS-ContingencySim": _safe_mean(cont_scores),
        },
        "per_feature": {
            "CS-KSComplement": ks_scores,
            "CS-StatisticSimilarity": stat_scores,
            "CS-RangeCoverage": range_scores,
            "CS-TVComplement": tv_scores,
            "CS-CategoryCoverage": coverage_scores,
            "CS-CorrelationSim": corr_scores,
            "CS-ContingencySim": cont_scores,
        },
        "metadata": {
            "n_timestamps": len(times),
            "timestamp_range": [str(times[0]), str(times[-1])],
            "n_num_cols": len(num_cols),
            "n_cat_cols": len(cat_cols),
            "num_cols": num_cols,
            "cat_cols": cat_cols,
            "key_col": key_col,
            "time_col": time_col,
            "contingency_pair_count": len(contingency_pairs),
        },
    }


def _nan_to_none(obj):
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def _prepare_dataset_frames(cfg, model_dir: Path, variant: str):
    primary = _infer_primary_table(cfg)
    if primary is None:
        raise ValueError(f"No table found in metadata for {cfg.name}")
    key_col = _infer_key_col(cfg, primary)
    time_col = _infer_time_col(cfg, primary)
    if time_col is None:
        raise ValueError(f"Could not infer time column for {cfg.name}/{primary}")

    real_df = flatten_table(cfg, primary, cfg.real_dir)
    synth_df = flatten_table(cfg, primary, cfg.real_dir, model_dir, variant)
    real_df = _apply_derived_time(real_df, cfg.name)
    synth_df = _apply_derived_time(synth_df, cfg.name)

    real_df, sibling_nums, real_sibling_join = _add_sibling_aggregates(
        cfg, real_df, cfg.real_dir, primary, key_col, time_col
    )
    synth_df, _, synth_sibling_join = _add_sibling_aggregates(
        cfg, synth_df, cfg.real_dir, primary, key_col, time_col, model_dir, variant
    )

    real_df = _coerce_time(real_df, time_col)
    synth_df = _coerce_time(synth_df, time_col)

    related = _related_tables(cfg, primary)
    num_cols, cat_cols = _source_feature_columns(cfg, related, time_col, key_col)
    for col in sibling_nums:
        if col not in num_cols:
            num_cols.append(col)
    num_cols = [c for c in num_cols if c in real_df.columns and c in synth_df.columns]
    num_cols = _filter_numeric_columns_with_partition_support(real_df, synth_df, time_col, num_cols)
    cat_cols = [c for c in cat_cols if c in real_df.columns and c in synth_df.columns]
    real_df, synth_df = _align_categorical_columns(real_df, synth_df, cat_cols)
    sibling_join_metadata = {
        "mode": "id_only",
        "real": real_sibling_join,
        "synthetic": synth_sibling_join,
    }
    return primary, key_col, time_col, real_df, synth_df, num_cols, cat_cols, sibling_join_metadata


def run_dataset(
    dataset: str,
    models: list[str] | None = None,
    variants: list[str] | None = None,
    dry_run: bool = False,
) -> int:
    cfg = load_dataset_config(dataset)
    if not cfg.supports_cross_sectional:
        print(
            "[WARN] Skipping "
            f"{dataset}: cross-sectional metrics require "
            "time_representation=absolute and trajectory_independence=dependent"
        )
        return 0
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
            output_dir = RESULTS_DIR / dataset / model / "cross_sectional"
            output_path = output_dir / f"cs_metrics_{variant}.json"
            if dry_run:
                print(
                    f"[DRY RUN] {dataset}/{model}/{variant}: "
                    f"primary={primary} synth={synth_path.relative_to(ROOT_DIR)} "
                    f"output={output_path.relative_to(ROOT_DIR)}"
                )
                continue
            try:
                print(f"========== [{dataset}] {model} / {variant} ==========")
                _, key_col, time_col, real_df, synth_df, num_cols, cat_cols, sibling_join_metadata = _prepare_dataset_frames(
                    cfg, model_dir, variant
                )
                if not num_cols and not cat_cols:
                    print(f"[WARN] No common feature columns for {dataset}/{model}/{variant}")
                    continue
                results = compute_all_cross_sectional_metrics(
                    real_df,
                    synth_df,
                    time_col=time_col,
                    key_col=key_col,
                    num_cols=num_cols,
                    cat_cols=cat_cols,
                )
                results["metadata"].update(
                    {
                        "dataset": dataset,
                        "model": model,
                        "variant": variant,
                        "primary_table": primary,
                        "sibling_join": sibling_join_metadata,
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
    known = set(available_datasets())
    worker_count = args.workers if args.workers is not None else args.n_jobs
    worker_count = worker_count or min(32, (os.cpu_count() or 1) + 4)

    def _run_target(dataset: str) -> int:
        if dataset not in known:
            print(f"[WARN] Unknown dataset skipped: {dataset}")
            return 0
        return run_dataset(
            dataset,
            models=args.models,
            variants=args.variants,
            dry_run=args.dry_run,
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
