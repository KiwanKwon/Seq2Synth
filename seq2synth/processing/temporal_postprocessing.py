"""
Unified temporal post-processing for the linked benchmark datasets.

This file replaces the dataset-specific scripts in
``benchmark/temporal_postprocessing`` with one configurable pipeline. Paths can
be absolute, relative to the current working directory, or relative to
``benchmark/data``. For example, from this file ``../../data/real/rossmann`` is
the benchmark data root.

Dataset presets
---------------
airbnb_tabdit
    real=real/airbnb_tabdit/child.csv, synth=synthetic/airbnb_tabdit/<model>/child.csv
    key_col=user, time_col=secs_elapsed, time_kind=duration, sort=(key,row_index),
    irregular. Stage 2 clips durations above the real global maximum.

berka_tabdit
    real=real/berka_tabdit/child.csv, synth=synthetic/berka_tabdit/<model>/child.csv
    key_col=user, time_col=Date, date_from=Year/Month/Day, irregular.
    Invalid synthetic dates and dates absent from real data are removed first.

citi_bike
    real=real/citi_bike/citi_bike.csv, synth=synthetic/citi_bike/<model>/citi_bike.csv
    key_col=bikeid, time_col=starttime, irregular, no dedup in the original script.

cmapss
    real=real/cmapss/cmapss.csv, synth=synthetic/cmapss/<model>/cmapss.csv
    key_col=unit_nr, time_col=time_cycles, time_kind=integer, regular.
    Stage 4 uses an integer grid.

coupon
    real=real/coupon/{coupon_detail,coupon_visit}.csv,
    synth=synthetic/coupon/<model>/{coupon_detail,coupon_visit}.csv
    key_col=USER_ID, time_col=I_DATE, irregular, multi-table preset.

fanniemae
    real=real/fanniemae/child.csv, synth=synthetic/fanniemae/<model>/child.csv
    key_col and time_col are dataset/run dependent, regular. Stage 4 uses monthly
    grid imputation. Override --key_col/--time_col when needed.

freddiemac
    real=real/freddiemac/hist.csv, synth=synthetic/freddiemac/<model>/hist.csv
    key_col and time_col are dataset/run dependent, regular. Stage 4 uses monthly
    grid imputation. Override --key_col/--time_col when needed.

google_cluster
    real=real/google_cluster/<table>.csv, synth=synthetic/google_cluster/<model>/<table>.csv
    table options:
      task_events: key_col=machine_id, time_col=time, irregular
      task_usage: key_col=machine_id, time_col=start_time, regular numeric grid,
                  snap step=300000000, also snaps end_time when present
      job_events: key_col=job_id, time_col=time, irregular
      task_constraints: key_col=job_id, time_col=time, irregular

hnm
    real=real/hnm/transactions_train.csv,
    synth=synthetic/hnm/<model>/transactions_train.csv
    key_col=customer_id, time_col=t_dat, irregular.

home_credit
    real=real/home_credit/<table>.csv, synth=synthetic/home_credit/<model>/<table>.csv
    application is the root table and is skipped. Temporal child table options:
      bureau: key_col=SK_ID_CURR, time_col=DAYS_CREDIT, irregular,
              dedup on SK_ID_BUREAU
      bureau_balance: key_col=SK_ID_BUREAU, time_col=MONTHS_BALANCE, regular
      previous_application: key_col=SK_ID_CURR, time_col=DAYS_DECISION, irregular
      POS_CASH_balance: key_col=SK_ID_PREV, time_col=MONTHS_BALANCE, regular
      installments_payments: key_col=SK_ID_PREV, time_col=DAYS_INSTALMENT,
                             irregular, sequence check on NUM_INSTALMENT_NUMBER
      credit_card_balance: key_col=SK_ID_PREV, time_col=MONTHS_BALANCE, regular

ptbxl
    real=real/ptbxl/records.csv, synth=synthetic/ptbxl/<model>/records.csv
    key_col=ecg_id, time_col=step, time_kind=integer, regular. Dedup is on
    (key_col,time_col), matching the original script.

rossmann
    real=real/rossmann/historical.csv,
    synth=synthetic/rossmann/<model>/historical.csv
    key_col and time_col are dataset/run dependent, regular. Stage 4 uses daily
    grid imputation. Override --key_col/--time_col when needed.

rossmann_tabdit
    real=real/rossmann_tabdit/historical.csv,
    synth=synthetic/rossmann_tabdit/<model>/historical.csv
    key_col=user, time_col=Date, date_from=Date_month/Date_day, regular.
    Invalid synthetic month/day combinations are removed first.

walmart
    real=real/walmart/{depts,features}.csv,
    synth=synthetic/walmart/<model>/{depts,features}.csv
    key_col=Store, time_col=Date, regular, multi-table preset. Stage 4 uses the
    real weekly grid.

Pipeline
--------
1. Sort by key/time. Airbnb can sort by key/row_index because secs_elapsed is
   a relative duration, not a timestamp.
2. Handle temporal values: boundary trim for ordinary timestamps, feature clip
   for durations, or snap/assign regular grids where configured.
3. Deduplicate according to the preset: full-row, key/time, or none.
4. If --is_regular is true, insert missing grid rows. Fill behavior is selected
   by --method: sparse, id_fill, prev_fill, interp_fill.


run example:
------------
python3 temporal_postprocessing.py --dataset coupon --model all --method sparse


python3 temporal_postprocessing.py --dataset airbnb_tabdit --model all


"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def find_benchmark_dir() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "data").is_dir() and (parent / "seq2synth").is_dir():
            return parent
        candidate = parent / "benchmark"
        if (candidate / "data").is_dir():
            return candidate
        if parent.name == "benchmark" and (parent / "data").is_dir():
            return parent
    return current.parents[3] / "benchmark"


BENCHMARK_DIR = find_benchmark_dir()
DATA_DIR = BENCHMARK_DIR / "data"


@dataclass(frozen=True)
class DatasetConfig:
    dataset: str
    real: str
    synth: str
    key_col: Optional[str]
    time_col: str
    table_files: Tuple[str, ...] = ()
    time_kind: str = "datetime"  # datetime, date_string, integer, numeric, duration
    is_regular: bool = False
    grid_kind: str = "real_unique"  # real_unique, date_range, integer, numeric_step
    grid_freq: Optional[str] = None
    grid_anchor: int = 0
    grid_step: Optional[int] = None
    date_builder: Optional[str] = None  # ymd_1900, month_day
    year_col: str = "Year"
    month_col: str = "Month"
    day_col: str = "Day"
    row_index_col: Optional[str] = None
    time_action: str = "boundary_trim"  # boundary_trim, duration_clip, snap_real_grid, assign_daily, assign_monthly
    dedup: str = "full_row"  # full_row, key_time, none
    dedup_keys: Tuple[str, ...] = ()
    sequence_col: Optional[str] = None


PRESETS: Dict[str, DatasetConfig] = {
    "airbnb_tabdit": DatasetConfig(
        "airbnb_tabdit", "real/airbnb_tabdit/child.csv",
        "synthetic/airbnb_tabdit/{model}/child.csv", "user", "secs_elapsed",
        time_kind="duration", row_index_col="Unnamed: 0", time_action="duration_clip",
    ),
    "berka_tabdit": DatasetConfig(
        "berka_tabdit", "real/berka_tabdit/child.csv",
        "synthetic/berka_tabdit/{model}/child.csv", "user", "Date",
        time_kind="date_string", date_builder="ymd_1900",
    ),
    "citi_bike": DatasetConfig(
        "citi_bike", "real/citi_bike/citi_bike.csv",
        "synthetic/citi_bike/{model}/citi_bike.csv", "bikeid", "starttime",
        dedup="none",
    ),
    "cmapss": DatasetConfig(
        "cmapss", "real/cmapss/cmapss.csv",
        "synthetic/cmapss/{model}/cmapss.csv", "unit_nr", "time_cycles",
        time_kind="integer", is_regular=True, grid_kind="integer",
        dedup="full_row",
    ),
    "coupon": DatasetConfig(
        "coupon", "real/coupon/{table}.csv",
        "synthetic/coupon/{model}/{table}.csv", "USER_ID", "I_DATE",
        table_files=("coupon_detail.csv", "coupon_visit.csv"),
    ),
    "fanniemae": DatasetConfig(
        "fanniemae", "real/fanniemae/child.csv",
        "synthetic/fanniemae/{model}/child.csv", None, "timestamp",
        is_regular=True, grid_kind="date_range", grid_freq="MS",
        time_action="assign_monthly",
    ),
    "freddiemac": DatasetConfig(
        "freddiemac", "real/freddiemac/hist.csv",
        "synthetic/freddiemac/{model}/hist.csv", None, "timestamp",
        is_regular=True, grid_kind="date_range", grid_freq="MS",
        time_action="assign_monthly",
    ),
    "google_cluster": DatasetConfig(
        "google_cluster", "real/google_cluster/{table}.csv",
        "synthetic/google_cluster/{model}/{table}.csv", "machine_id", "time",
        time_kind="numeric",
    ),
    "hnm": DatasetConfig(
        "hnm", "real/hnm/transactions_train.csv",
        "synthetic/hnm/{model}/transactions_train.csv", "customer_id", "t_dat",
    ),
    "home_credit": DatasetConfig(
        "home_credit", "real/home_credit/{table}.csv",
        "synthetic/home_credit/{model}/{table}.csv", "SK_ID_CURR", "DAYS_CREDIT",
        time_kind="integer",
    ),
    "ptbxl": DatasetConfig(
        "ptbxl", "real/ptbxl/records.csv",
        "synthetic/ptbxl/{model}/records.csv", "ecg_id", "step",
        time_kind="integer", is_regular=True, grid_kind="integer",
        dedup="key_time",
    ),
    "rossmann": DatasetConfig(
        "rossmann", "real/rossmann/historical.csv",
        "synthetic/rossmann/{model}/historical.csv", None, "Date",
        is_regular=True, grid_kind="date_range", grid_freq="D",
        time_action="assign_daily",
    ),
    "rossmann_tabdit": DatasetConfig(
        "rossmann_tabdit", "real/rossmann_tabdit/historical.csv",
        "synthetic/rossmann_tabdit/{model}/historical.csv", "user", "Date",
        time_kind="date_string", is_regular=True, grid_kind="real_unique",
        date_builder="month_day", month_col="Date_month", day_col="Date_day",
    ),
    "walmart": DatasetConfig(
        "walmart", "real/walmart/{table}.csv",
        "synthetic/walmart/{model}/{table}.csv", "Store", "Date",
        table_files=("depts.csv", "features.csv"), is_regular=True,
        grid_kind="real_unique", time_action="snap_real_grid",
    ),
}

GOOGLE_TABLES: Dict[str, DatasetConfig] = {
    "task_events": replace(PRESETS["google_cluster"], key_col="machine_id", time_col="time", is_regular=False),
    "task_usage": replace(
        PRESETS["google_cluster"], key_col="machine_id", time_col="start_time",
        is_regular=True, grid_kind="numeric_step", grid_step=300_000_000,
        time_action="snap_real_grid",
    ),
    "job_events": replace(PRESETS["google_cluster"], key_col="job_id", time_col="time", is_regular=False),
    "task_constraints": replace(PRESETS["google_cluster"], key_col="job_id", time_col="time", is_regular=False),
}

HOME_CREDIT_TABLES: Dict[str, DatasetConfig] = {
    "bureau": replace(
        PRESETS["home_credit"],
        key_col="SK_ID_CURR",
        time_col="DAYS_CREDIT",
        is_regular=False,
        dedup="custom",
        dedup_keys=("SK_ID_BUREAU",),
    ),
    "bureau_balance": replace(
        PRESETS["home_credit"],
        key_col="SK_ID_BUREAU",
        time_col="MONTHS_BALANCE",
        is_regular=True,
        grid_kind="integer",
        grid_anchor=0,
        grid_step=1,
        time_action="snap_integer_grid",
        dedup="key_time",
    ),
    "previous_application": replace(
        PRESETS["home_credit"],
        key_col="SK_ID_CURR",
        time_col="DAYS_DECISION",
        is_regular=False,
        dedup="key_time",
    ),
    "POS_CASH_balance": replace(
        PRESETS["home_credit"],
        key_col="SK_ID_PREV",
        time_col="MONTHS_BALANCE",
        is_regular=True,
        grid_kind="integer",
        grid_anchor=0,
        grid_step=1,
        time_action="snap_integer_grid",
        dedup="key_time",
    ),
    "installments_payments": replace(
        PRESETS["home_credit"],
        key_col="SK_ID_PREV",
        time_col="DAYS_INSTALMENT",
        is_regular=False,
        dedup="custom",
        dedup_keys=("SK_ID_PREV", "NUM_INSTALMENT_NUMBER", "NUM_INSTALMENT_VERSION"),
        sequence_col="NUM_INSTALMENT_NUMBER",
    ),
    "credit_card_balance": replace(
        PRESETS["home_credit"],
        key_col="SK_ID_PREV",
        time_col="MONTHS_BALANCE",
        is_regular=True,
        grid_kind="integer",
        grid_anchor=0,
        grid_step=1,
        time_action="snap_integer_grid",
        dedup="key_time",
    ),
}


def resolve_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    cwd_path = Path.cwd() / p
    if cwd_path.exists():
        return cwd_path
    return DATA_DIR / p


def _sniff_format(path: Path) -> str:
    if path.suffix.lower() in (".parquet", ".pq"):
        return "parquet"
    if path.suffix.lower() in (".csv", ".tsv", ".txt"):
        return "csv"
    if path.is_file():
        with path.open("rb") as f:
            if f.read(4) == b"PAR1":
                return "parquet"
    return "csv"


def load_data(path: Path) -> pd.DataFrame:
    if _sniff_format(path) == "parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def save_data(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in (".parquet", ".pq"):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def output_path_for(synth_path: Path, method: str) -> Path:
    suffix = "_postprocessed" if method == "sparse" else f"_postprocessed_{method}"
    ext = synth_path.suffix or ".csv"
    return synth_path.parent / f"{synth_path.stem}{suffix}{ext}"


def metrics_path_for(output_path: Path) -> Path:
    return output_path.parent / f"{output_path.stem}_metrics.json"


def validate_columns(df: pd.DataFrame, cols: Iterable[Optional[str]], name: str) -> None:
    missing = [c for c in cols if c and c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {name}: {', '.join(missing)}")


def build_date_column(df: pd.DataFrame, cfg: DatasetConfig) -> pd.DataFrame:
    if cfg.date_builder is None:
        return df
    df = df.copy()
    if cfg.date_builder == "ymd_1900":
        year = df[cfg.year_col].astype(float).fillna(-1).astype(int)
        month = df[cfg.month_col].astype(float).fillna(-1).astype(int)
        day = df[cfg.day_col].astype(float).fillna(-1).astype(int)
        date_str = (
            year.map(lambda y: f"{1900 + y:04d}" if y >= 0 else "-001")
            + "-"
            + month.map(lambda m: f"{m:02d}")
            + "-"
            + day.map(lambda d: f"{d:02d}")
        )
        parsed = pd.to_datetime(date_str, format="%Y-%m-%d", errors="coerce")
        df[cfg.time_col] = parsed.dt.strftime("%Y-%m-%d").where(parsed.notna(), np.nan)
        return df
    if cfg.date_builder == "month_day":
        month = df[cfg.month_col].astype(float).fillna(-1).astype(int)
        day = df[cfg.day_col].astype(float).fillna(-1).astype(int)
        df[cfg.time_col] = month.map(lambda m: f"{m:02d}") + "-" + day.map(lambda d: f"{d:02d}")
        bad = df[cfg.month_col].isna() | df[cfg.day_col].isna()
        df.loc[bad, cfg.time_col] = np.nan
        return df
    raise ValueError(f"Unknown date_builder: {cfg.date_builder}")


def coerce_time(df: pd.DataFrame, cfg: DatasetConfig, name: str) -> pd.DataFrame:
    df = df.copy()
    if cfg.time_kind == "datetime":
        df[cfg.time_col] = pd.to_datetime(df[cfg.time_col], errors="coerce")
    elif cfg.time_kind in {"integer", "numeric", "duration"}:
        df[cfg.time_col] = pd.to_numeric(df[cfg.time_col], errors="coerce")
        if cfg.time_kind == "integer":
            df[cfg.time_col] = df[cfg.time_col].round().astype("Int64")
    elif cfg.time_kind == "date_string":
        df[cfg.time_col] = df[cfg.time_col].astype("string")
    else:
        raise ValueError(f"Unknown time_kind: {cfg.time_kind}")
    n_bad = int(df[cfg.time_col].isna().sum())
    if n_bad:
        print(f"[{name}] {n_bad:,} rows have missing/unparseable {cfg.time_col}.")
    return df


def filter_invalid_synth_dates(synth_df: pd.DataFrame, real_df: pd.DataFrame, cfg: DatasetConfig) -> pd.DataFrame:
    if cfg.date_builder is None:
        return synth_df
    before = len(synth_df)
    real_dates = set(real_df[cfg.time_col].dropna().unique())
    result = synth_df.dropna(subset=[cfg.time_col]).copy()
    result = result[result[cfg.time_col].isin(real_dates)].copy()
    dropped = before - len(result)
    if dropped:
        print(f"[Pre-filter] Dropped {dropped:,} synthetic rows with invalid dates.")
    return result


def stage1_sort(synth_df: pd.DataFrame, cfg: DatasetConfig) -> pd.DataFrame:
    key_col = require_key(cfg)
    if cfg.row_index_col:
        cols = [key_col, cfg.row_index_col]
    else:
        cols = [key_col, cfg.time_col]
    result = synth_df.sort_values(cols, kind="stable").reset_index(drop=True)
    print(f"[Stage 1] Sorted by {cols}. Rows: {len(result):,}")
    return result


def require_key(cfg: DatasetConfig) -> str:
    if not cfg.key_col:
        raise ValueError(f"--key_col is required for dataset={cfg.dataset}")
    return cfg.key_col


def snap_to_real_grid(synth_df: pd.DataFrame, real_df: pd.DataFrame, cfg: DatasetConfig) -> pd.DataFrame:
    real_grid = pd.Series(sorted(real_df[cfg.time_col].dropna().unique()))
    if real_grid.empty:
        return synth_df.copy()
    result = synth_df.copy()
    if cfg.time_kind == "datetime":
        grid_ns = pd.to_datetime(real_grid).astype("int64").to_numpy()
        synth_time = pd.to_datetime(result[cfg.time_col], errors="coerce")
        synth_ns = synth_time.astype("int64").to_numpy()
    else:
        grid_ns = pd.to_numeric(real_grid, errors="coerce").to_numpy(dtype=float)
        synth_ns = pd.to_numeric(result[cfg.time_col], errors="coerce").to_numpy(dtype=float)
    idx = np.searchsorted(grid_ns, synth_ns)
    idx = np.clip(idx, 0, len(grid_ns) - 1)
    prev = np.clip(idx - 1, 0, len(grid_ns) - 1)
    choose_prev = np.abs(grid_ns[prev] - synth_ns) <= np.abs(grid_ns[idx] - synth_ns)
    snap_idx = np.where(choose_prev, prev, idx)
    result[cfg.time_col] = real_grid.iloc[snap_idx].to_numpy()
    if cfg.dataset == "google_cluster" and cfg.time_col == "start_time" and "end_time" in result.columns:
        end = pd.to_numeric(result["end_time"], errors="coerce")
        start_before = pd.to_numeric(synth_df[cfg.time_col], errors="coerce")
        duration = end - start_before
        result["end_time"] = pd.to_numeric(result[cfg.time_col], errors="coerce") + duration
    return result


def snap_to_integer_grid(synth_df: pd.DataFrame, cfg: DatasetConfig) -> pd.DataFrame:
    step = cfg.grid_step or 1
    result = synth_df.copy()
    values = pd.to_numeric(result[cfg.time_col], errors="coerce")
    snapped = (((values - cfg.grid_anchor) / step).round() * step + cfg.grid_anchor)
    result[cfg.time_col] = snapped.astype("Int64")
    return result


def stage2_time_handling(
    synth_df: pd.DataFrame,
    real_df: pd.DataFrame,
    cfg: DatasetConfig,
) -> pd.DataFrame:
    if cfg.time_action == "duration_clip":
        real_max = real_df[cfg.time_col].max()
        result = synth_df[synth_df[cfg.time_col] <= real_max].copy()
        print(f"[Stage 2] Clipped durations above real max={real_max}. Dropped {len(synth_df) - len(result):,}.")
        return result
    if cfg.time_action in {"snap_real_grid", "snap_integer_grid", "assign_daily", "assign_monthly"}:
        if cfg.time_action == "snap_real_grid":
            result = snap_to_real_grid(synth_df, real_df, cfg)
        elif cfg.time_action == "snap_integer_grid":
            result = snap_to_integer_grid(synth_df, cfg)
        else:
            result = synth_df.copy()
        if cfg.time_action == "assign_daily":
            result[cfg.time_col] = pd.to_datetime(result[cfg.time_col], errors="coerce").dt.normalize()
        if cfg.time_action == "assign_monthly":
            result[cfg.time_col] = pd.to_datetime(result[cfg.time_col], errors="coerce").dt.to_period("M").dt.to_timestamp()
        synth_df = result
    real_min = real_df[cfg.time_col].min()
    real_max = real_df[cfg.time_col].max()
    result = synth_df[(synth_df[cfg.time_col] >= real_min) & (synth_df[cfg.time_col] <= real_max)].copy()
    print(f"[Stage 2] Boundary trim [{real_min}, {real_max}]. Dropped {len(synth_df) - len(result):,}.")
    return result


def stage3_dedup(synth_df: pd.DataFrame, cfg: DatasetConfig) -> pd.DataFrame:
    if cfg.dedup == "none":
        print("[Stage 3] Skipped deduplication.")
        return synth_df.copy()
    before = len(synth_df)
    if cfg.dedup_keys:
        result = synth_df.drop_duplicates(subset=list(cfg.dedup_keys), keep="first").copy()
        label = ",".join(cfg.dedup_keys)
    elif cfg.dedup == "key_time":
        result = synth_df.drop_duplicates(subset=[require_key(cfg), cfg.time_col], keep="first").copy()
        label = "key_time"
    elif cfg.dedup == "full_row":
        result = synth_df.drop_duplicates(keep="first").copy()
        label = "full_row"
    else:
        raise ValueError(f"Unknown dedup mode: {cfg.dedup}")
    print(f"[Stage 3] Deduplicated ({label}). Dropped {before - len(result):,}.")
    return result


def infer_feature_cols(df: pd.DataFrame, cfg: DatasetConfig, num_cols: Sequence[str], cat_cols: Sequence[str]) -> Tuple[List[str], List[str]]:
    protected = {require_key(cfg), cfg.time_col, "_is_imputed"}
    all_features = [c for c in df.columns if c not in protected]
    nums = list(num_cols) if num_cols else [c for c in all_features if pd.api.types.is_numeric_dtype(df[c])]
    cats = list(cat_cols) if cat_cols else [c for c in all_features if c not in nums]
    return nums, cats


def make_grid(group: pd.DataFrame, real_df: pd.DataFrame, cfg: DatasetConfig, key: Any) -> List[Any]:
    time_col = cfg.time_col
    if cfg.grid_kind == "integer":
        step = cfg.grid_step or 1
        start = int(group[time_col].min())
        end = int(group[time_col].max())
        return list(range(start, end + 1, step))
    if cfg.grid_kind == "numeric_step":
        step = cfg.grid_step or 1
        real_per_key = real_df.groupby(require_key(cfg))[time_col].agg(["min", "max"])
        if key in real_per_key.index:
            start, end = real_per_key.loc[key, "min"], real_per_key.loc[key, "max"]
        else:
            start, end = real_df[time_col].min(), real_df[time_col].max()
        start = max(start, group[time_col].min())
        return list(np.arange(int(start), int(end) + step, step))
    if cfg.grid_kind == "date_range":
        freq = cfg.grid_freq or "D"
        real_per_key = real_df.groupby(require_key(cfg))[time_col].agg(["min", "max"])
        if key in real_per_key.index:
            start, end = real_per_key.loc[key, "min"], real_per_key.loc[key, "max"]
        else:
            start, end = real_df[time_col].min(), real_df[time_col].max()
        start = max(pd.Timestamp(start), pd.Timestamp(group[time_col].min()))
        return list(pd.date_range(start=start, end=pd.Timestamp(end), freq=freq))
    real_grid = sorted(real_df[time_col].dropna().unique())
    real_per_key = real_df.groupby(require_key(cfg))[time_col].agg(["min", "max"])
    if key in real_per_key.index:
        start, end = real_per_key.loc[key, "min"], real_per_key.loc[key, "max"]
    else:
        start, end = real_grid[0], real_grid[-1]
    start = max(start, group[time_col].min())
    return [x for x in real_grid if start <= x <= end]


def fill_imputed_rows(merged: pd.DataFrame, method: str, num_cols: List[str], cat_cols: List[str], cfg: DatasetConfig) -> pd.DataFrame:
    if method == "sparse" or "_is_imputed" not in merged.columns:
        return merged
    imp = merged["_is_imputed"]
    if not bool(imp.any()):
        return merged
    if method == "id_fill":
        observed = ~imp
        for col in num_cols:
            merged.loc[imp, col] = pd.to_numeric(merged.loc[observed, col], errors="coerce").mean()
        for col in cat_cols:
            mode = merged.loc[observed, col].mode(dropna=True)
            if not mode.empty:
                merged.loc[imp, col] = mode.iloc[0]
    elif method == "prev_fill":
        merged[num_cols + cat_cols] = merged[num_cols + cat_cols].ffill()
    elif method == "interp_fill":
        for col in num_cols:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").interpolate(method="linear", limit_direction="both")
        for col in cat_cols:
            merged[col] = merged[col].ffill().bfill()
    else:
        raise ValueError(f"Unsupported method for imputation: {method}")
    return merged


def stage4_grid_imputation(
    synth_df: pd.DataFrame,
    real_df: pd.DataFrame,
    cfg: DatasetConfig,
    method: str,
    num_cols: Sequence[str],
    cat_cols: Sequence[str],
) -> Tuple[pd.DataFrame, int]:
    if not cfg.is_regular:
        print("[Stage 4] Skipped grid imputation.")
        out = synth_df.copy()
        out["_is_imputed"] = False
        return out, 0
    key_col = require_key(cfg)
    nums, cats = infer_feature_cols(synth_df, cfg, num_cols, cat_cols)
    frames: List[pd.DataFrame] = []
    total_imputed = 0
    for key, group in synth_df.groupby(key_col, sort=False):
        group = group.sort_values(cfg.time_col).copy()
        grid = make_grid(group, real_df, cfg, key)
        if not grid:
            group["_is_imputed"] = False
            frames.append(group)
            continue
        marked = group.copy()
        marked["_is_original"] = True
        grid_df = pd.DataFrame({key_col: key, cfg.time_col: grid})
        merged = grid_df.merge(marked, on=[key_col, cfg.time_col], how="left")
        merged["_is_imputed"] = merged["_is_original"].isna()
        merged = merged.drop(columns=["_is_original"])
        total_imputed += int(merged["_is_imputed"].sum())
        merged = fill_imputed_rows(merged, method, nums, cats, cfg)
        frames.append(merged)
    result = pd.concat(frames, ignore_index=True) if frames else synth_df.copy()
    print(f"[Stage 4] Inserted {total_imputed:,} grid rows.")
    return result, total_imputed


def compute_metrics(before: pd.DataFrame, after: pd.DataFrame, real_df: pd.DataFrame, cfg: DatasetConfig, n_imputed: int) -> Dict[str, Any]:
    key_col = require_key(cfg)
    metrics: Dict[str, Any] = {
        "total_rows_before": int(len(before)),
        "total_rows_after": int(len(after)),
        "rows_removed": int(len(before) - len(after) + n_imputed),
        "imputed_rows": int(n_imputed),
        "imputation_ratio": round(float(n_imputed / len(after)), 6) if len(after) else 0.0,
        "flows_before": int(before[key_col].nunique()) if key_col in before else 0,
        "flows_after": int(after[key_col].nunique()) if key_col in after else 0,
        "real_flows": int(real_df[key_col].nunique()) if key_col in real_df else 0,
    }
    if key_col in before and key_col in after:
        metrics["avg_flow_length_before"] = round(float(before.groupby(key_col).size().mean()), 4)
        metrics["avg_flow_length_after"] = round(float(after.groupby(key_col).size().mean()), 4)
    return metrics


def sequence_integrity_metrics(df: pd.DataFrame, cfg: DatasetConfig) -> Dict[str, Any]:
    if not cfg.sequence_col:
        return {}
    key_col = require_key(cfg)
    seq_numeric = pd.to_numeric(df[cfg.sequence_col], errors="coerce")
    work = df.assign(_seq=seq_numeric).dropna(subset=["_seq"])
    total = 0
    with_gaps = 0
    missing_total = 0
    missing_per_gap: List[int] = []
    for _, group in work.groupby(key_col, sort=False):
        observed_vals = group["_seq"].astype(int).unique()
        total += 1
        if len(observed_vals) == 0:
            continue
        max_seq = int(observed_vals.max())
        if max_seq < 1:
            continue
        missing = set(range(1, max_seq + 1)) - {int(v) for v in observed_vals}
        if missing:
            with_gaps += 1
            missing_total += len(missing)
            missing_per_gap.append(len(missing))
    return {
        "seq_loans_total": int(total),
        "seq_loans_with_gaps": int(with_gaps),
        "seq_loans_complete_ratio": round(1.0 - (with_gaps / total), 6) if total else 1.0,
        "seq_total_missing_numbers": int(missing_total),
        "seq_avg_missing_per_loan": round(float(np.mean(missing_per_gap)), 4) if missing_per_gap else None,
    }


def config_for_args(args: argparse.Namespace) -> DatasetConfig:
    cfg = PRESETS[args.dataset]
    if args.dataset == "google_cluster" and args.table and "," not in args.table:
        if args.table not in GOOGLE_TABLES:
            choices = ", ".join(sorted(GOOGLE_TABLES))
            raise ValueError(f"--table for google_cluster must be one of: {choices}")
        cfg = GOOGLE_TABLES[args.table]
    if args.key_col:
        cfg = replace(cfg, key_col=args.key_col)
    if args.time_col:
        cfg = replace(cfg, time_col=args.time_col)
    if args.row_index_col:
        cfg = replace(cfg, row_index_col=args.row_index_col)
    if args.time_kind:
        cfg = replace(cfg, time_kind=args.time_kind)
    if args.is_regular is not None:
        cfg = replace(cfg, is_regular=args.is_regular)
    if args.grid_freq:
        cfg = replace(cfg, grid_freq=args.grid_freq, grid_kind="date_range")
    if args.grid_step:
        cfg = replace(cfg, grid_step=args.grid_step, grid_kind="numeric_step")
    if args.dedup:
        cfg = replace(cfg, dedup=args.dedup)
    return cfg


def format_template(template: str, model: str, table_file: Optional[str], table_arg: Optional[str]) -> str:
    table_name = table_file or table_arg or "table.csv"
    table_stem = Path(table_name).stem
    table_value = table_stem if "{table}.csv" in template or "{table}.parquet" in template else table_name
    return template.format(model=model, table=table_value, table_stem=table_stem)


def table_files_for(cfg: DatasetConfig, table_arg: Optional[str]) -> Tuple[Optional[str], ...]:
    if cfg.table_files:
        return tuple(table_arg.split(",")) if table_arg else tuple(cfg.table_files)
    if "{table}" in cfg.real or "{table}" in cfg.synth:
        if not table_arg:
            return (None,)
        return tuple(table_arg.split(","))
    return (None,)


def dataset_dir(path_template: str, model: str, table_arg: Optional[str]) -> Optional[Path]:
    if "{table}" not in path_template:
        return None
    marker = "__TABLE__"
    rendered = path_template.format(model=model, table=marker, table_stem=marker)
    before_table = rendered.split(marker, 1)[0].rstrip("/")
    return resolve_path(before_table)


def discover_models(cfg: DatasetConfig, table_arg: Optional[str]) -> List[str]:
    """Return sorted list of all model subdirectories under the synthetic dataset path."""
    if "{model}" not in cfg.synth:
        return []
    prefix = cfg.synth.split("{model}")[0].rstrip("/")
    synth_base = resolve_path(prefix)
    if not synth_base.is_dir():
        return []
    return sorted(d.name for d in synth_base.iterdir() if d.is_dir())


def discover_table_files(cfg: DatasetConfig, model: str, table_arg: Optional[str]) -> Tuple[Optional[str], ...]:
    explicit = table_files_for(cfg, table_arg)
    if explicit != (None,):
        return explicit
    real_dir = dataset_dir(cfg.real, model, table_arg)
    synth_dir = dataset_dir(cfg.synth, model, table_arg)
    if not real_dir or not synth_dir:
        real_dir = resolve_path(format_template(cfg.real, model, None, table_arg)).parent
        synth_dir = resolve_path(format_template(cfg.synth, model, None, table_arg)).parent
    if not real_dir or not synth_dir or not real_dir.is_dir() or not synth_dir.is_dir():
        return explicit
    real_files = {
        path.name for path in real_dir.iterdir()
        if path.suffix.lower() in {".csv", ".parquet", ".pq"} and path.is_file()
    }
    synth_files = {
        path.name for path in synth_dir.iterdir()
        if path.suffix.lower() in {".csv", ".parquet", ".pq"} and path.is_file()
    }
    files = tuple(sorted(real_files & synth_files))
    if table_arg and files:
        requested = {Path(item).name for item in table_arg.split(",")}
        requested |= {f"{Path(item).name}.csv" for item in table_arg.split(",") if not Path(item).suffix}
        files = tuple(file for file in files if file in requested)
    return files or explicit


def path_for_table(template: str, model: str, table_file: Optional[str], table_arg: Optional[str]) -> Path:
    if table_file and "{table}" not in template:
        base = resolve_path(format_template(template, model, None, table_arg)).parent
        return base / table_file
    return resolve_path(format_template(template, model, table_file, table_arg))


def override_path(value: Optional[str], table_file: Optional[str]) -> Optional[Path]:
    if value is None:
        return None
    path = resolve_path(value)
    if table_file and path.is_dir():
        return path / table_file
    return path


def config_for_table(cfg: DatasetConfig, table_file: Optional[str]) -> DatasetConfig:
    if cfg.dataset == "google_cluster" and table_file:
        return GOOGLE_TABLES.get(Path(table_file).stem, cfg)
    if cfg.dataset == "home_credit" and table_file:
        return HOME_CREDIT_TABLES.get(Path(table_file).stem, cfg)
    return cfg


def metadata_path_for(cfg: DatasetConfig, model: str, table_arg: Optional[str]) -> Optional[Path]:
    real_path = path_for_table(cfg.real, model, None, table_arg)
    candidates = [
        real_path.parent / "metadata.json",
        DATA_DIR / "real" / cfg.dataset / "metadata.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_metadata_for(cfg: DatasetConfig, model: str, table_arg: Optional[str]) -> Dict[str, Any]:
    path = metadata_path_for(cfg, model, table_arg)
    if path is None:
        return {}
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"[WARN] Could not read metadata at {path}: {exc}")
        return {}


def root_tables_from_metadata(metadata: Dict[str, Any]) -> set[str]:
    child_tables = {
        rel.get("child_table_name")
        for rel in metadata.get("relationships", [])
        if rel.get("child_table_name")
    }
    parent_tables = {
        rel.get("parent_table_name")
        for rel in metadata.get("relationships", [])
        if rel.get("parent_table_name")
    }
    return {table for table in parent_tables if table not in child_tables}


def is_root_table(table_file: Optional[str], metadata: Dict[str, Any]) -> bool:
    if not table_file:
        return False
    return Path(table_file).stem in root_tables_from_metadata(metadata)


def table_metadata(metadata: Dict[str, Any], table_file: Optional[str]) -> Dict[str, Any]:
    if not table_file:
        return {}
    return metadata.get("tables", {}).get(Path(table_file).stem, {})


def infer_key_col(metadata: Dict[str, Any], table_file: Optional[str], fallback: Optional[str]) -> Optional[str]:
    if not table_file:
        return fallback
    table_name = Path(table_file).stem
    for rel in metadata.get("relationships", []):
        if rel.get("child_table_name") == table_name and rel.get("child_foreign_key"):
            return rel["child_foreign_key"]
    table = table_metadata(metadata, table_file)
    return table.get("primary_key") or fallback


def infer_time_col(metadata: Dict[str, Any], table_file: Optional[str], fallback: str) -> str:
    table = table_metadata(metadata, table_file)
    columns = table.get("columns", {})
    datetime_cols = [
        name
        for name, info in columns.items()
        if str(info.get("sdtype", "")).lower() == "datetime"
    ]
    return datetime_cols[0] if datetime_cols else fallback


def apply_metadata_to_table_config(
    cfg: DatasetConfig,
    table_file: Optional[str],
    metadata: Dict[str, Any],
) -> DatasetConfig:
    if not table_file or not metadata:
        return cfg
    key_col = infer_key_col(metadata, table_file, cfg.key_col)
    time_col = infer_time_col(metadata, table_file, cfg.time_col)
    return replace(cfg, key_col=key_col, time_col=time_col)


def has_required_columns(path: Path, cfg: DatasetConfig) -> Tuple[bool, str]:
    try:
        sample = load_data(path)
        sample = build_date_column(sample, cfg)
    except Exception as exc:
        return False, f"could not read/build date columns: {exc}"
    required = [cfg.key_col, cfg.time_col, cfg.row_index_col, cfg.sequence_col, *cfg.dedup_keys]
    missing = [col for col in required if col and col not in sample.columns]
    if missing:
        return False, f"missing columns: {', '.join(missing)}"
    return True, ""


def process_one(
    cfg: DatasetConfig,
    real_path: Path,
    synth_path: Path,
    output_path: Optional[Path],
    method: str,
    num_cols: Sequence[str],
    cat_cols: Sequence[str],
) -> Path:
    print("\n" + "=" * 72)
    print(f"Temporal post-processing: dataset={cfg.dataset}, method={method}")
    print(f"Real : {real_path}")
    print(f"Synth: {synth_path}")
    print("=" * 72)
    real_df = load_data(real_path)
    synth_df = load_data(synth_path)
    real_df = build_date_column(real_df, cfg)
    synth_df = build_date_column(synth_df, cfg)
    validate_columns(real_df, [cfg.key_col, cfg.time_col, cfg.sequence_col, *cfg.dedup_keys], "real")
    validate_columns(synth_df, [cfg.key_col, cfg.time_col, cfg.row_index_col, cfg.sequence_col, *cfg.dedup_keys], "synth")
    real_df = coerce_time(real_df, cfg, "real")
    synth_df = coerce_time(synth_df, cfg, "synth")
    synth_df = filter_invalid_synth_dates(synth_df, real_df, cfg)
    before = synth_df.copy()
    synth_df = stage1_sort(synth_df, cfg)
    synth_df = stage2_time_handling(synth_df, real_df, cfg)
    synth_df = stage3_dedup(synth_df, cfg)
    synth_df, n_imputed = stage4_grid_imputation(synth_df, real_df, cfg, method, num_cols, cat_cols)
    out_path = output_path or output_path_for(synth_path, method)
    save_data(synth_df, out_path)
    metrics = compute_metrics(before, synth_df, real_df, cfg, n_imputed)
    metrics.update(sequence_integrity_metrics(synth_df, cfg))
    metrics.update({
        "dataset": cfg.dataset,
        "method": method,
        "real_file": str(real_path.relative_to(DATA_DIR)) if real_path.is_relative_to(DATA_DIR) else str(real_path),
        "synth_file": str(synth_path.relative_to(DATA_DIR)) if synth_path.is_relative_to(DATA_DIR) else str(synth_path),
        "output_file": str(out_path.relative_to(DATA_DIR)) if out_path.is_relative_to(DATA_DIR) else str(out_path),
    })
    with metrics_path_for(out_path).open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Done] Saved: {out_path}")
    print(f"[Done] Metrics: {metrics_path_for(out_path)}")
    return out_path


def parse_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected one of true/false/1/0/yes/no.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified temporal post-processing for benchmark datasets.")
    parser.add_argument("--dataset", choices=sorted(PRESETS), required=True)
    parser.add_argument("--model", default="SDV", help="Synthetic model directory used by preset paths.")
    parser.add_argument("--table", default=None, help="Optional table filename/name. Comma-separated to limit multi-table processing.")
    parser.add_argument("--real", default=None, help="Real file path. Relative paths are resolved under benchmark/data.")
    parser.add_argument("--synth", default=None, help="Synthetic file path. Relative paths are resolved under benchmark/data.")
    parser.add_argument("--output", default=None, help="Output path. Only valid for a single processed table.")
    parser.add_argument("--key_col", default=None)
    parser.add_argument("--time_col", default=None)
    parser.add_argument("--row_index_col", default=None)
    parser.add_argument("--time_kind", choices=["datetime", "date_string", "integer", "numeric", "duration"], default=None)
    parser.add_argument("--is_regular", type=parse_bool, default=None)
    parser.add_argument("--grid_freq", default=None)
    parser.add_argument("--grid_step", type=int, default=None)
    parser.add_argument("--dedup", choices=["full_row", "key_time", "none"], default=None)
    parser.add_argument("--num_cols", nargs="*", default=[])
    parser.add_argument("--cat_cols", nargs="*", default=[])
    parser.add_argument("--method", choices=["sparse", "id_fill", "prev_fill", "interp_fill"], default="sparse")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = config_for_args(args)

    if args.model.lower() == "all":
        models = discover_models(cfg, args.table)
        if not models:
            print(f"[Error] No model directories found for dataset={args.dataset}")
            return
        if args.output:
            raise ValueError("--output cannot be used with --model all.")
        print(f"[Info] Found {len(models)} model(s): {', '.join(models)}")
    else:
        models = [args.model]

    total_processed = 0
    total_skipped = 0

    for model in models:
        if len(models) > 1:
            print(f"\n{'#' * 72}")
            print(f"# Model: {model}")
            print(f"{'#' * 72}")

        tables = discover_table_files(cfg, model, args.table)
        if args.output and len(tables) > 1:
            raise ValueError("--output can only be used when processing a single table.")
        metadata = load_metadata_for(cfg, model, args.table)
        processed = 0
        skipped = 0
        for table_file in tables:
            table_cfg = config_for_table(cfg, table_file)
            table_name = table_file or "single-table"
            if is_root_table(table_file, metadata):
                skipped += 1
                print(f"[Skip] {table_name}: root table; temporal imputation is only applied to child/temporal tables.")
                continue
            table_cfg = apply_metadata_to_table_config(table_cfg, table_file, metadata)
            real_path = override_path(args.real, table_file) or path_for_table(table_cfg.real, model, table_file, args.table)
            synth_path = override_path(args.synth, table_file) or path_for_table(table_cfg.synth, model, table_file, args.table)
            if not real_path.exists():
                skipped += 1
                print(f"[Skip] Missing real file: {real_path}")
                continue
            if not synth_path.exists():
                skipped += 1
                print(f"[Skip] Missing synth file: {synth_path}")
                continue
            ok, reason = has_required_columns(synth_path, table_cfg)
            if not ok:
                skipped += 1
                print(f"[Skip] {table_name}: {reason}")
                continue
            output = Path(args.output) if args.output else None
            if output and not output.is_absolute():
                output = resolve_path(str(output))
            process_one(
                cfg=table_cfg,
                real_path=real_path,
                synth_path=synth_path,
                output_path=output,
                method=args.method,
                num_cols=args.num_cols,
                cat_cols=args.cat_cols,
            )
            processed += 1
        print(f"\n[Model: {model}] Processed {processed} file(s); skipped {skipped} file(s).")
        total_processed += processed
        total_skipped += skipped

    if len(models) > 1:
        print(f"\n[Summary] {len(models)} models — total processed: {total_processed}, skipped: {total_skipped}.")
    else:
        print(f"\n[Summary] Processed {total_processed} file(s); skipped {total_skipped} file(s).")


if __name__ == "__main__":
    main()
