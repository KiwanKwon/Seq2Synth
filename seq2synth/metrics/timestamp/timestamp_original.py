#!/usr/bin/env python3
"""
run_timestamp_metrics.py
========================
Batch runner for timestamp_metrics.py across all benchmark datasets.

Usage
-----
  # Run all datasets
  python run_timestamp_metrics.py

  # Run specific datasets
  python run_timestamp_metrics.py --datasets freddiemac fanniemae

  # Print commands without executing
  python run_timestamp_metrics.py --dry_run
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd

BASE = Path(__file__).parent


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SubtableConfig:
    """One CSV table inside a multi-table dataset."""
    filename:        str                    # e.g. "task_events.csv"
    key_col:         str
    time_col:        str
    periodicity:     str                    # "Regular" | "Irregular"
    time_repr:       str                    # "Absolute" | "Relative"
    time_col_format: Optional[str] = None


@dataclass
class DatasetConfig:
    name:               str
    skip_postprocessed: bool = False
    name_fn:            Callable = lambda p: p.parent.name  # (Path) -> str
    # Single-table fields (used when subtables is None)
    real_data:          Optional[Path] = None
    synth_glob:         Optional[str]  = None
    key_col:            Optional[str]  = None
    time_col:           Optional[str]  = None
    periodicity:        Optional[str]  = None      # "Regular" | "Irregular"
    time_repr:          Optional[str]  = None      # "Absolute" | "Relative"
    time_col_format:    Optional[str]  = None
    filter_fn:          Optional[Callable] = None
    preprocess_fn:      Optional[Callable] = None
    # Multi-table fields (mutually exclusive with above)
    real_dir:           Optional[Path] = None      # e.g. data/real/google_cluster
    synth_dir_glob:     Optional[str]  = None      # e.g. data/synthetic/google_cluster/*
    subtables:          Optional[List[SubtableConfig]] = None


# ---------------------------------------------------------------------------
# Dataset configurations
# ---------------------------------------------------------------------------



CONFIGS: List[DatasetConfig] = [
    DatasetConfig(
        name               = "freddiemac",
        real_data          = BASE / "data/real/freddiemac/hist_original.csv",
        synth_glob         = str(BASE / "data/synthetic/freddiemac/*/hist.csv"),
        skip_postprocessed = False,
        name_fn            = lambda p: p.parent.name,
        key_col            = "LOAN SEQUENCE NUMBER",
        time_col           = "MONTHLY REPORTING PERIOD",
        periodicity        = "Regular",
        time_repr          = "Absolute",
    ),
    DatasetConfig(
        name               = "fanniemae",
        real_data          = BASE / "data/real/fanniemae/child.csv",
        synth_glob         = str(BASE / "data/synthetic/fanniemae/*/child.csv"),
        skip_postprocessed = False,  # glob targets child.csv exactly
        name_fn            = lambda p: p.parent.name,
        key_col            = "Loan Identifier",
        time_col           = "Monthly Reporting Period",
        periodicity        = "Regular",
        time_repr          = "Absolute",
    ),
    DatasetConfig(
        name               = "rossmann",
        real_data          = BASE / "data/real/rossmann/historical.csv",
        synth_glob         = str(BASE / "data/synthetic/rossmann/*/historical.csv"),
        skip_postprocessed = False,  # glob targets historical.csv exactly
        name_fn            = lambda p: p.parent.name,
        key_col            = "Store",
        time_col           = "Date",
        periodicity        = "Regular",
        time_repr          = "Absolute",
    ),
    DatasetConfig(
        name               = "rossmann_tabdit",
        real_data          = BASE / "data/real/rossmann_tabdit/historical.csv",
        synth_glob         = str(BASE / "data/synthetic/rossmann_tabdit/*/historical.csv"),
        skip_postprocessed = False,  # glob targets historical.csv exactly
        name_fn            = lambda p: p.parent.name,
        key_col            = "user",
        time_col           = "Date",  # auto-constructed from Date_month + Date_day as "MM-DD"
        time_col_format    = "%m-%d",  # explicit format to avoid NaT when parsing "MM-DD" strings
        periodicity        = "Regular",
        time_repr          = "Absolute",
    ),
    DatasetConfig(
        name               = "walmart",
        real_data          = BASE / "data/real/walmart/features.csv",
        synth_glob         = str(BASE / "data/synthetic/walmart/*/features.csv"),
        skip_postprocessed = False,  # glob targets features.csv exactly
        name_fn            = lambda p: p.parent.name,
        key_col            = "Store",
        time_col           = "Date",
        periodicity        = "Regular",
        time_repr          = "Absolute",
    ),
    DatasetConfig(
        name               = "ptbxl",
        real_data          = BASE / "data/real/ptbxl/records.csv",
        synth_glob         = str(BASE / "data/synthetic/ptbxl/*/records.csv"),
        skip_postprocessed = False,
        name_fn            = lambda p: p.parent.name,
        key_col            = "ecg_id",
        time_col           = "step",
        periodicity        = "Regular",
        time_repr          = "Relative",
    ),
    DatasetConfig(
        name               = "hnm",
        real_data          = BASE / "data/real/hnm/transactions_train.csv",
        synth_glob         = str(BASE / "data/synthetic/hnm/*/transactions_train.csv"),
        skip_postprocessed = False,
        name_fn            = lambda p: p.parent.name,
        key_col            = "customer_id",
        time_col           = "t_dat",
        periodicity        = "Irregular",
        time_repr          = "Absolute",
    ),
    DatasetConfig(
        name               = "coupon",
        real_data          = BASE / "data/real/coupon/coupon_detail.csv",
        synth_glob         = str(BASE / "data/synthetic/coupon/*/coupon_detail.csv"),
        skip_postprocessed = False,
        name_fn            = lambda p: p.parent.name,
        key_col            = "USER_ID",
        time_col           = "I_DATE",
        periodicity        = "Irregular",
        time_repr          = "Absolute",
    ),
    DatasetConfig(
        name               = "coupon_visit",
        real_data          = BASE / "data/real/coupon/coupon_visit.csv",
        synth_glob         = str(BASE / "data/synthetic/coupon/*/coupon_visit.csv"),
        skip_postprocessed = False,
        name_fn            = lambda p: p.parent.name,
        key_col            = "USER_ID",
        time_col           = "I_DATE",
        periodicity        = "Irregular",
        time_repr          = "Absolute",
    ),
    DatasetConfig(
        name               = "airbnb_tabdit",
        real_data          = BASE / "data/real/airbnb_tabdit/child.csv",
        synth_glob         = str(BASE / "data/synthetic/airbnb_tabdit/*/child.csv"),
        skip_postprocessed = False,
        name_fn            = lambda p: p.parent.name,
        key_col            = "user",
        time_col           = "secs_elapsed",
        periodicity        = "Irregular",
        time_repr          = "Relative",
    ),
    DatasetConfig(
        name               = "cmapss",
        real_data          = BASE / "data/real/cmapss/cmapss.csv",
        synth_glob         = str(BASE / "data/synthetic/cmapss/*/cmapss.csv"),
        skip_postprocessed = False,
        name_fn            = lambda p: p.parent.name,
        key_col            = "unit_nr",
        time_col           = "time_cycles",
        periodicity        = "Regular",
        time_repr          = "Relative",
    ),
    DatasetConfig(
        name               = "berka_tabdit",
        real_data          = BASE / "data/real/berka_tabdit/child.csv",
        synth_glob         = str(BASE / "data/synthetic/berka_tabdit/*/child.csv"),
        skip_postprocessed = False,
        name_fn            = lambda p: p.parent.name,
        key_col            = "user",
        time_col           = "Date",
        periodicity        = "Irregular",
        time_repr          = "Absolute",
        preprocess_fn      = lambda df: _berka_tabdit_add_date(df),
    ),
    DatasetConfig(
        name               = "google_cluster",
        skip_postprocessed = True,
        name_fn            = lambda p: p.parent.name,
        real_dir           = BASE / "data/real/google_cluster",
        synth_dir_glob     = str(BASE / "data/synthetic/google_cluster/*"),
        subtables = [
            SubtableConfig("task_events.csv",      "job_id", "time",       "Irregular", "Relative"),
            SubtableConfig("job_events.csv",       "job_id", "time",       "Irregular", "Relative"),
            SubtableConfig("task_constraints.csv", "job_id", "time",       "Irregular", "Relative"),
            SubtableConfig("task_usage.csv",       "job_id", "start_time", "Regular",   "Relative"),
        ],
    ),
    DatasetConfig(
        name               = "citi_bike",
        real_data          = BASE / "data/real/citi_bike/citi_bike.csv",
        synth_glob         = str(BASE / "data/synthetic/citi_bike/*/citi_bike.csv"),
        skip_postprocessed = False,
        name_fn            = lambda p: p.parent.name,
        key_col            = "bikeid",
        time_col           = "starttime",
        periodicity        = "Irregular",
        time_repr          = "Absolute",
    ),
    DatasetConfig(
        name           = "mimic",
        name_fn        = lambda p: p.parent.name,
        real_dir       = BASE / "data/real/mimic",
        synth_dir_glob = str(BASE / "data/synthetic/mimic/*"),
        subtables = [
            SubtableConfig("chartevents_categorical.csv", "subject_id", "charttime", "Irregular", "Absolute"),
            SubtableConfig("chartevents_numeric.csv",     "subject_id", "charttime", "Irregular", "Absolute"),
            SubtableConfig("labevents_categorical.csv",   "subject_id", "charttime", "Irregular", "Absolute"),
            SubtableConfig("labevents_numeric.csv",       "subject_id", "charttime", "Irregular", "Absolute"),
            SubtableConfig("outputevents.csv",            "subject_id", "charttime", "Irregular", "Absolute"),
        ],
    ),
]

CONFIG_MAP = {c.name: c for c in CONFIGS}


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def _berka_tabdit_add_date(df: pd.DataFrame) -> pd.DataFrame:
    """Construct a YYYY-MM-DD 'Date' column from Year (2-digit), Month, Day.
    The original CSV is never modified — this operates on an in-memory copy.
    """
    df = df.copy()
    year  = pd.to_numeric(df['Year'],  errors='coerce').fillna(0).astype(int) + 1900
    month = pd.to_numeric(df['Month'], errors='coerce').fillna(1).astype(int)
    day   = pd.to_numeric(df['Day'],   errors='coerce').fillna(1).astype(int)
    date_str = (year.astype(str).str.zfill(4) + '-' +
                month.astype(str).str.zfill(2) + '-' +
                day.astype(str).str.zfill(2))
    df['Date'] = pd.to_datetime(date_str, errors='coerce').dt.strftime('%Y-%m-%d')
    return df


def _write_temp_csv(df: pd.DataFrame) -> str:
    """Write a DataFrame to a named temp file and return its path.
    Caller is responsible for deleting it with os.unlink().
    """
    tmp = tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w')
    df.to_csv(tmp.name, index=False)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _build_cmd(real_path: str, synth_path: str, key_col: str, time_col: str,
               periodicity: str, time_repr: str, output: str,
               time_col_format: Optional[str] = None) -> List[str]:
    cmd = [
        sys.executable, str(BASE / "timestamp_metrics.py"),
        "--real_data",           real_path,
        "--synth_data",          synth_path,
        "--key_col",             key_col,
        "--time_col",            time_col,
        "--periodicity",         periodicity,
        "--time_representation", time_repr,
        "--output",              output,
    ]
    if time_col_format is not None:
        cmd += ["--time_col_format", time_col_format]
    return cmd


def _aggregate_metrics(per_table):
    """Average numeric metric values across sub-tables. Non-numeric values
    (e.g. 'N/A' sentinels for periodicity-specific metrics) are skipped per
    metric key; if no numeric values exist for a key, output 'N/A'.
    """
    keys = set()
    for m in per_table.values():
        keys.update(m.keys())
    averaged = {}
    for k in keys:
        nums = [float(v) for v in (m.get(k) for m in per_table.values())
                if isinstance(v, (int, float)) and not isinstance(v, bool)]
        averaged[k] = round(sum(nums) / len(nums), 6) if nums else "N/A"
    return averaged


def run_multitable_dataset(cfg: DatasetConfig, dry_run: bool = False,
                           models: Optional[List[str]] = None) -> None:
    assert cfg.subtables is not None and cfg.real_dir is not None and cfg.synth_dir_glob is not None
    model_dirs = sorted(d for d in glob.glob(cfg.synth_dir_glob) if Path(d).is_dir())
    if not model_dirs:
        print(f"[WARN] No model dirs found for '{cfg.name}': {cfg.synth_dir_glob}")
        return

    for model_dir in model_dirs:
        model_name = Path(model_dir).name
        if models is not None and model_name not in models:
            continue

        print(f"========== [{cfg.name}] {model_name} ==========")
        per_table = {}
        per_table_chars = {}
        any_failure = False

        for sub in cfg.subtables:
            real_path  = cfg.real_dir / sub.filename
            synth_path = Path(model_dir) / sub.filename
            if not synth_path.exists():
                print(f"  [SKIP] missing synth file: {synth_path}")
                continue
            if not real_path.exists():
                print(f"  [SKIP] missing real file:  {real_path}")
                continue

            tmp_json = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
            tmp_json.close()
            cmd = _build_cmd(
                str(real_path), str(synth_path),
                sub.key_col, sub.time_col, sub.periodicity, sub.time_repr,
                tmp_json.name, sub.time_col_format,
            )
            print(f"  -- {sub.filename}")
            if dry_run:
                print("     [DRY RUN]", " ".join(f'"{a}"' if " " in a else a for a in cmd))
                os.unlink(tmp_json.name)
                continue
            try:
                result = subprocess.run(cmd, cwd=str(BASE))
                if result.returncode != 0:
                    print(f"     [ERROR] {sub.filename} failed (exit {result.returncode})")
                    any_failure = True
                    continue
                with open(tmp_json.name) as f:
                    sub_out = json.load(f)
                per_table[sub.filename] = sub_out.get("metrics", {})
                per_table_chars[sub.filename] = sub_out.get("data_characteristics", {})
            finally:
                if os.path.exists(tmp_json.name):
                    os.unlink(tmp_json.name)

        if dry_run or not per_table:
            print()
            continue

        averaged = _aggregate_metrics(per_table)
        output_dir = BASE / "results" / cfg.name / model_name / "timestamp"
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / f"{model_name}.json"
        with open(output, "w") as f:
            json.dump({
                "metrics": averaged,
                "per_table": {
                    name: {"data_characteristics": per_table_chars.get(name, {}),
                           "metrics": metrics}
                    for name, metrics in per_table.items()
                },
            }, f, indent=2)
        print(f"  Aggregated {len(per_table)} table(s) -> {output}")
        if any_failure:
            print("  [WARN] some sub-tables failed; averages computed over successful ones")
        print()


def run_dataset(cfg: DatasetConfig, dry_run: bool = False, models: Optional[List[str]] = None) -> None:
    if cfg.subtables is not None:
        run_multitable_dataset(cfg, dry_run=dry_run, models=models)
        return
    synth_files = sorted(glob.glob(cfg.synth_glob))
    if not synth_files:
        print(f"[WARN] No synthetic files found for '{cfg.name}': {cfg.synth_glob}")
        return

    # Preprocess real data once into a temp file (original CSV untouched)
    real_data_path = str(cfg.real_data)
    real_tmp_path = None
    if cfg.preprocess_fn is not None and not dry_run:
        real_df = pd.read_csv(cfg.real_data)
        real_tmp_path = _write_temp_csv(cfg.preprocess_fn(real_df))
        real_data_path = real_tmp_path

    try:
        for synth_path in synth_files:
            p = Path(synth_path)
            if cfg.skip_postprocessed and "_postprocessed" in p.stem:
                continue
            if cfg.filter_fn is not None and not cfg.filter_fn(p):
                continue
            if models is not None and p.parent.name not in models:
                continue

            model_name = cfg.name_fn(p)
            output_dir = BASE / "results" / cfg.name / model_name / "timestamp"
            output_dir.mkdir(parents=True, exist_ok=True)
            output = output_dir / f"{model_name}.json"

            # Preprocess synth data into a temp file (original CSV untouched)
            synth_data_path = str(synth_path)
            synth_tmp_path = None
            if cfg.preprocess_fn is not None and not dry_run:
                synth_df = pd.read_csv(synth_path)
                synth_tmp_path = _write_temp_csv(cfg.preprocess_fn(synth_df))
                synth_data_path = synth_tmp_path

            try:
                print(f"========== [{cfg.name}] {model_name} ==========")
                cmd = [
                    sys.executable, str(BASE / "timestamp_metrics.py"),
                    "--real_data",           real_data_path,
                    "--synth_data",          synth_data_path,
                    "--key_col",             cfg.key_col,
                    "--time_col",            cfg.time_col,
                    "--periodicity",         cfg.periodicity,
                    "--time_representation", cfg.time_repr,
                    "--output",              str(output),
                ]
                if cfg.time_col_format is not None:
                    cmd += ["--time_col_format", cfg.time_col_format]

                if dry_run:
                    print("  [DRY RUN]", " ".join(f'"{a}"' if " " in a else a for a in cmd))
                else:
                    result = subprocess.run(cmd, cwd=str(BASE))
                    if result.returncode != 0:
                        print(f"  [ERROR] {cfg.name}/{model_name} failed (exit {result.returncode})")
                print()
            finally:
                if synth_tmp_path:
                    os.unlink(synth_tmp_path)
    finally:
        if real_tmp_path:
            os.unlink(real_tmp_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch timestamp metrics runner for all benchmark datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        choices=list(CONFIG_MAP.keys()),
        default=None,
        metavar="DATASET",
        help=(
            f"Datasets to evaluate. Choices: {list(CONFIG_MAP.keys())}. "
            "Defaults to all."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="MODEL",
        help="Filter by model folder name (e.g. RDBDIFF SDV). Defaults to all models.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing.",
    )
    args = parser.parse_args()

    targets = args.datasets if args.datasets else list(CONFIG_MAP.keys())

    print(f"Running timestamp metrics for: {targets}")
    if args.models:
        print(f"Filtering models: {args.models}")
    print()
    for name in targets:
        run_dataset(CONFIG_MAP[name], dry_run=args.dry_run, models=args.models)


if __name__ == "__main__":
    main()