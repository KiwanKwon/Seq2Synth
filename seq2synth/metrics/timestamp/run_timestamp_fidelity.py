from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd

# Anchor sys.path to the benchmark root so that configs.py / paths.py are importable
# regardless of where this script is invoked from.
TIMESYNTH_DIR = Path(__file__).parents[2]
BENCHMARK_DIR = TIMESYNTH_DIR.parent
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from configs import (  # noqa: E402
    DatasetConfig as MetadataConfig,
    available_datasets,
    load_dataset_config,
    load_metadata,
)
from paths import RESULTS_DIR, ROOT_DIR  # noqa: E402


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SubtableConfig:
    """One CSV table inside a multi-table dataset."""
    filename:        str
    key_col:         str
    time_col:        str
    periodicity:     str                   # "Regular" | "Irregular"
    time_repr:       str                   # "Absolute" | "Relative"
    time_col_format: Optional[str] = None
    preprocess_fn:   Optional[Callable] = None


@dataclass
class DatasetConfig:
    name:               str
    skip_postprocessed: bool               = False
    name_fn:            Callable           = lambda p: p.parent.name
    # Single-table fields
    real_data:          Optional[Path]     = None
    synth_glob:         Optional[str]      = None
    key_col:            Optional[str]      = None
    time_col:           Optional[str]      = None
    periodicity:        Optional[str]      = None
    time_repr:          Optional[str]      = None
    time_col_format:    Optional[str]      = None
    filter_fn:          Optional[Callable] = None
    preprocess_fn:      Optional[Callable] = None
    # Multi-table fields (mutually exclusive with single-table fields)
    real_dir:           Optional[Path]     = None
    synth_dir_glob:     Optional[str]      = None
    subtables:          Optional[List[SubtableConfig]] = None


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _infer_time_col(table) -> Optional[str]:
    if table.datetime_cols:
        return table.datetime_cols[0]
    preferred = ("time", "start_time", "step", "time_cycles", "secs_elapsed", "charttime")
    for col in preferred:
        if col in table.columns:
            return col
    for col in table.numeric_cols:
        lower = col.lower()
        if "time" in lower or "step" in lower or "cycle" in lower:
            return col
    return None


def _infer_time_repr(table, time_col: str) -> str:
    return "Absolute" if time_col in table.datetime_cols else "Relative"


def _configured_time_repr(dataset_cfg: MetadataConfig, table, time_col: str) -> str:
    if dataset_cfg.time_representation:
        return dataset_cfg.time_representation.strip().title()
    return _infer_time_repr(table, time_col)


def _infer_periodicity(time_col: str, time_repr: str) -> str:
    lower = time_col.lower()
    if time_repr == "Relative" and (
        lower in {"step", "time_cycles", "start_time"} or "cycle" in lower
    ):
        return "Regular"
    if time_repr == "Absolute" and (lower == "date" or "period" in lower):
        return "Regular"
    return "Irregular"


def _configured_periodicity(dataset_cfg: MetadataConfig, time_col: str, time_repr: str) -> str:
    if dataset_cfg.periodicity and dataset_cfg.periodicity.strip().lower() in {"regular", "irregular"}:
        return dataset_cfg.periodicity.strip().title()
    return _infer_periodicity(time_col, time_repr)


def _metadata_subtable_config(
    dataset_cfg: MetadataConfig, table_name: str
) -> Optional[SubtableConfig]:
    table = dataset_cfg.tables[table_name]
    rel = next(
        (r for r in dataset_cfg.relationships if r.get("child_table_name") == table_name),
        {},
    )
    key_col = rel.get("child_foreign_key") or table.primary_key
    if key_col is None and table.id_cols:
        key_col = table.id_cols[0]
    time_col = _infer_time_col(table)
    if key_col is None or time_col is None:
        return None
    time_repr = _configured_time_repr(dataset_cfg, table, time_col)
    return SubtableConfig(
        filename=f"{table_name}.csv",
        key_col=key_col,
        time_col=time_col,
        periodicity=_configured_periodicity(dataset_cfg, time_col, time_repr),
        time_repr=time_repr,
        time_col_format=table.columns.get(time_col, {}).get("datetime_format"),
    )


def _subtables_from_override(ts_list: list) -> List[SubtableConfig]:
    return [
        SubtableConfig(
            filename=f"{ts['table']}.csv",
            key_col=ts["key_col"],
            time_col=ts["time_col"],
            periodicity=ts["periodicity"],
            time_repr=ts["time_repr"],
            time_col_format=ts.get("time_col_format"),
        )
        for ts in ts_list
    ]


def _home_credit_relative_to_elapsed(key_col: str, time_col: str) -> Callable:
    def preprocess(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
        df[time_col] = df[time_col] - df.groupby(key_col)[time_col].transform("min")
        return df

    return preprocess


def _home_credit_config() -> DatasetConfig:
    return DatasetConfig(
        name="home_credit",
        skip_postprocessed=True,
        name_fn=lambda p: p.parent.name,
        real_dir=ROOT_DIR / "data/real/home_credit",
        synth_dir_glob=str(ROOT_DIR / "data/synthetic/home_credit/*"),
        subtables=[
            # All time columns are negative-day or negative-month offsets relative to
            # each client's application date, so time_repr="Relative" for every table.
            # Three *_balance tables share a regular monthly grid (MONTHS_BALANCE);
            # the other three are event-log style (irregular DAYS_* offsets).
            # application.csv is excluded: it is cross-sectional (one row per client),
            # so timestamp metrics are undefined for it.
            #
            # preprocess_fn: shift each flow's time column so its earliest value is 0.
            # This matches metric_temporal_range_compliance's Relative-series
            # assumption that time starts at 0 and grows positive.
            SubtableConfig(
                "bureau.csv",
                "SK_ID_CURR",
                "DAYS_CREDIT",
                "Irregular",
                "Relative",
                preprocess_fn=_home_credit_relative_to_elapsed("SK_ID_CURR", "DAYS_CREDIT"),
            ),
            SubtableConfig(
                "bureau_balance.csv",
                "SK_ID_BUREAU",
                "MONTHS_BALANCE",
                "Regular",
                "Relative",
                preprocess_fn=_home_credit_relative_to_elapsed("SK_ID_BUREAU", "MONTHS_BALANCE"),
            ),
            SubtableConfig(
                "previous_application.csv",
                "SK_ID_CURR",
                "DAYS_DECISION",
                "Irregular",
                "Relative",
                preprocess_fn=_home_credit_relative_to_elapsed("SK_ID_CURR", "DAYS_DECISION"),
            ),
            SubtableConfig(
                "POS_CASH_balance.csv",
                "SK_ID_PREV",
                "MONTHS_BALANCE",
                "Regular",
                "Relative",
                preprocess_fn=_home_credit_relative_to_elapsed("SK_ID_PREV", "MONTHS_BALANCE"),
            ),
            SubtableConfig(
                "installments_payments.csv",
                "SK_ID_PREV",
                "DAYS_INSTALMENT",
                "Irregular",
                "Relative",
                preprocess_fn=_home_credit_relative_to_elapsed("SK_ID_PREV", "DAYS_INSTALMENT"),
            ),
            SubtableConfig(
                "credit_card_balance.csv",
                "SK_ID_PREV",
                "MONTHS_BALANCE",
                "Regular",
                "Relative",
                preprocess_fn=_home_credit_relative_to_elapsed("SK_ID_PREV", "MONTHS_BALANCE"),
            ),
        ],
    )


def build_metadata_configs() -> List[DatasetConfig]:
    configs = []
    for dataset in available_datasets():
        dataset_cfg = load_dataset_config(dataset)
        raw_metadata = load_metadata(dataset)

        # Explicit timestamp_fidelity section in metadata.json takes precedence
        # over structural inference.  Supports both a single dict and a list.
        ts_override = raw_metadata.get("timestamp_fidelity")
        if ts_override is not None:
            if isinstance(ts_override, dict):
                ts_override = [ts_override]
            subtables = _subtables_from_override(ts_override)
            if len(subtables) == 1:
                sub = subtables[0]
                table_name = Path(sub.filename).stem
                real_filename = (
                    dataset_cfg.tables[table_name].filename
                    if table_name in dataset_cfg.tables
                    else sub.filename
                )
                configs.append(DatasetConfig(
                    name=dataset,
                    real_data=dataset_cfg.real_dir / real_filename,
                    synth_glob=str(dataset_cfg.synth_dir / "*" / sub.filename),
                    key_col=sub.key_col,
                    time_col=sub.time_col,
                    periodicity=sub.periodicity,
                    time_repr=sub.time_repr,
                    time_col_format=sub.time_col_format,
                ))
            else:
                configs.append(DatasetConfig(
                    name=dataset,
                    skip_postprocessed=True,
                    real_dir=dataset_cfg.real_dir,
                    synth_dir_glob=str(dataset_cfg.synth_dir / "*"),
                    subtables=subtables,
                ))
            continue

        # Auto-inference from SDV-style metadata structure
        child_tables = dataset_cfg.child_tables or list(dataset_cfg.tables)
        subtables = [
            sub
            for table_name in child_tables
            if table_name in dataset_cfg.tables
            for sub in [_metadata_subtable_config(dataset_cfg, table_name)]
            if sub is not None
        ]
        if not subtables:
            continue
        if len(subtables) == 1:
            sub = subtables[0]
            table_name = Path(sub.filename).stem
            real_filename = dataset_cfg.tables[table_name].filename
            configs.append(DatasetConfig(
                name=dataset,
                real_data=dataset_cfg.real_dir / real_filename,
                synth_glob=str(dataset_cfg.synth_dir / "*" / sub.filename),
                key_col=sub.key_col,
                time_col=sub.time_col,
                periodicity=sub.periodicity,
                time_repr=sub.time_repr,
                time_col_format=sub.time_col_format,
            ))
        else:
            configs.append(DatasetConfig(
                name=dataset,
                skip_postprocessed=True,
                real_dir=dataset_cfg.real_dir,
                synth_dir_glob=str(dataset_cfg.synth_dir / "*"),
                subtables=subtables,
            ))
    return configs


CONFIGS: List[DatasetConfig] = [
    cfg for cfg in build_metadata_configs() if cfg.name != "home_credit"
]
CONFIGS.append(_home_credit_config())
CONFIG_MAP = {c.name: c for c in CONFIGS}

_TIMESTAMP_SCRIPT = Path(__file__).with_name("timestamp_metrics.py")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _build_cmd(
    real_path: str,
    synth_path: str,
    key_col: str,
    time_col: str,
    periodicity: str,
    time_repr: str,
    output: str,
    time_col_format: Optional[str] = None,
) -> List[str]:
    cmd = [
        sys.executable, str(_TIMESTAMP_SCRIPT),
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


def _write_temp_csv(df: pd.DataFrame) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    df.to_csv(tmp.name, index=False)
    tmp.close()
    return tmp.name


def _aggregate_metrics(per_table):
    """Average numeric metric values across sub-tables; non-numeric values yield 'N/A'."""
    keys: set = set()
    for m in per_table.values():
        keys.update(m.keys())
    averaged = {}
    for k in keys:
        nums = [
            float(v) for v in (m.get(k) for m in per_table.values())
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        ]
        averaged[k] = round(sum(nums) / len(nums), 6) if nums else "N/A"
    return averaged


def _get_summary(result: dict) -> dict:
    """Read the current summary key while accepting older metrics outputs."""
    return result.get("summary", result.get("metrics", {}))


def run_multitable_dataset(
    cfg: DatasetConfig,
    dry_run: bool = False,
    models: Optional[List[str]] = None,
) -> None:
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
        per_table: dict = {}
        per_table_chars: dict = {}
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

            real_data_path = str(real_path)
            synth_data_path = str(synth_path)
            real_tmp_path = None
            synth_tmp_path = None
            if sub.preprocess_fn is not None and not dry_run:
                real_df = pd.read_csv(real_path)
                synth_df = pd.read_csv(synth_path)
                real_tmp_path = _write_temp_csv(sub.preprocess_fn(real_df))
                synth_tmp_path = _write_temp_csv(sub.preprocess_fn(synth_df))
                real_data_path = real_tmp_path
                synth_data_path = synth_tmp_path

            tmp_json = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
            tmp_json.close()
            cmd = _build_cmd(
                real_data_path, synth_data_path,
                sub.key_col, sub.time_col, sub.periodicity, sub.time_repr,
                tmp_json.name, sub.time_col_format,
            )
            print(f"  -- {sub.filename}")
            if dry_run:
                print("     [DRY RUN]", " ".join(f'"{a}"' if " " in a else a for a in cmd))
                os.unlink(tmp_json.name)
                continue
            try:
                result = subprocess.run(cmd, cwd=str(ROOT_DIR))
                if result.returncode != 0:
                    print(f"     [ERROR] {sub.filename} failed (exit {result.returncode})")
                    any_failure = True
                    continue
                with open(tmp_json.name) as f:
                    sub_out = json.load(f)
                per_table[sub.filename] = _get_summary(sub_out)
                per_table_chars[sub.filename] = sub_out.get("data_characteristics", {})
            finally:
                if os.path.exists(tmp_json.name):
                    os.unlink(tmp_json.name)
                if real_tmp_path and os.path.exists(real_tmp_path):
                    os.unlink(real_tmp_path)
                if synth_tmp_path and os.path.exists(synth_tmp_path):
                    os.unlink(synth_tmp_path)

        if dry_run or not per_table:
            print()
            continue

        averaged = _aggregate_metrics(per_table)
        output_dir = RESULTS_DIR / cfg.name / model_name / "timestamp"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{model_name}.json"
        with open(output_path, "w") as f:
            json.dump(
                {
                    "summary": averaged,
                    "per_table": {
                        name: {
                            "data_characteristics": per_table_chars.get(name, {}),
                            "summary": metrics,
                        }
                        for name, metrics in per_table.items()
                    },
                },
                f,
                indent=2,
            )
        print(f"  Aggregated {len(per_table)} table(s) -> {output_path}")
        if any_failure:
            print("  [WARN] some sub-tables failed; averages computed over successful ones")
        print()


def run_dataset(
    cfg: DatasetConfig,
    dry_run: bool = False,
    models: Optional[List[str]] = None,
) -> None:
    if cfg.subtables is not None:
        run_multitable_dataset(cfg, dry_run=dry_run, models=models)
        return

    synth_files = sorted(glob.glob(cfg.synth_glob))
    if not synth_files:
        print(f"[WARN] No synthetic files found for '{cfg.name}': {cfg.synth_glob}")
        return

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
            output_dir = RESULTS_DIR / cfg.name / model_name / "timestamp"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{model_name}.json"

            synth_data_path = str(synth_path)
            synth_tmp_path = None
            if cfg.preprocess_fn is not None and not dry_run:
                synth_df = pd.read_csv(synth_path)
                synth_tmp_path = _write_temp_csv(cfg.preprocess_fn(synth_df))
                synth_data_path = synth_tmp_path

            try:
                print(f"========== [{cfg.name}] {model_name} ==========")
                cmd = [
                    sys.executable, str(_TIMESTAMP_SCRIPT),
                    "--real_data",           real_data_path,
                    "--synth_data",          synth_data_path,
                    "--key_col",             cfg.key_col,
                    "--time_col",            cfg.time_col,
                    "--periodicity",         cfg.periodicity,
                    "--time_representation", cfg.time_repr,
                    "--output",              str(output_path),
                ]
                if cfg.time_col_format is not None:
                    cmd += ["--time_col_format", cfg.time_col_format]

                if dry_run:
                    print("  [DRY RUN]", " ".join(f'"{a}"' if " " in a else a for a in cmd))
                else:
                    result = subprocess.run(cmd, cwd=str(ROOT_DIR))
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
        description="Batch timestamp fidelity runner for all benchmark datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        metavar="DATASET",
        help=(
            f"Datasets to evaluate. Available: {sorted(CONFIG_MAP.keys())}. "
            "Defaults to all supported datasets. Unknown datasets are skipped."
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
    args = parser.parse_args()

    targets = args.datasets if args.datasets else list(CONFIG_MAP.keys())
    unknown = [t for t in targets if t not in CONFIG_MAP]
    if unknown:
        print(f"[WARN] No timestamp fidelity config for: {unknown} (skipping)")
        targets = [t for t in targets if t in CONFIG_MAP]

    print(f"Running timestamp fidelity for: {targets}")
    if args.models:
        print(f"Filtering models: {args.models}")
    print()
    worker_count = args.workers if args.workers is not None else args.n_jobs
    worker_count = worker_count or min(32, (os.cpu_count() or 1) + 4)
    if args.parallel and len(targets) > 1:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            list(executor.map(
                lambda name: run_dataset(CONFIG_MAP[name], dry_run=args.dry_run, models=args.models),
                targets,
            ))
    else:
        for name in targets:
            run_dataset(CONFIG_MAP[name], dry_run=args.dry_run, models=args.models)


if __name__ == "__main__":
    main()
