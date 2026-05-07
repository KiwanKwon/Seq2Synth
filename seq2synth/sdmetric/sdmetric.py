from __future__ import annotations

import argparse
import os
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BENCHMARK_DIR = Path(__file__).resolve().parents[2]
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import numpy as np
import pandas as pd
from sdmetrics.single_column import KSComplement, TVComplement

from configs import CATEGORICAL_SDTYPES, NUMERIC_SDTYPES, VARIANT_SUFFIXES
from paths import RESULTS_DIR


def _sniff_format(path: str | Path) -> str:
    p = Path(path)
    ext = p.suffix.lower()
    if ext in (".parquet", ".pq"):
        return "parquet"
    if p.is_file():
        with open(p, "rb") as f:
            if f.read(4) == b"PAR1":
                return "parquet"
    return "csv"


def load_data(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if _sniff_format(p) == "parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def nan_to_none(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [nan_to_none(v) for v in obj]
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def _normalize_categorical_value(value: Any) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, float) and value == int(value):
        return str(int(value))
    return str(value)


def _normalize_categorical_series(series: pd.Series) -> pd.Series:
    return series.map(_normalize_categorical_value)


def compute_tv_complement(real_col: pd.Series, synth_col: pd.Series) -> float:
    try:
        return TVComplement.compute(real_data=real_col, synthetic_data=synth_col)
    except Exception:
        return np.nan


def compute_ks_complement(real_col: pd.Series, synth_col: pd.Series) -> float:
    try:
        return KSComplement.compute(real_data=real_col, synthetic_data=synth_col)
    except Exception:
        return np.nan


def statistic_similarity(
    real_data: pd.DataFrame,
    synth_data: pd.DataFrame,
    value_col: str,
    epsilon: float = 1e-6,
) -> float:
    try:
        real_mean = pd.to_numeric(real_data[value_col], errors="coerce").mean()
        synth_mean = pd.to_numeric(synth_data[value_col], errors="coerce").mean()
        if pd.isna(real_mean) or pd.isna(synth_mean):
            return np.nan
        denom = max(abs(real_mean), abs(synth_mean)) + epsilon
        return float(1 - (abs(real_mean - synth_mean) / denom))
    except Exception:
        return np.nan


def category_coverage(real_data: pd.DataFrame, synth_data: pd.DataFrame, value_col: str) -> float:
    try:
        real_values = set(_normalize_categorical_series(real_data[value_col].dropna()).dropna().unique())
        synth_values = set(_normalize_categorical_series(synth_data[value_col].dropna()).dropna().unique())
        if not real_values:
            return np.nan
        return float(len(real_values & synth_values) / len(real_values))
    except Exception:
        return np.nan


def range_coverage(real_data: pd.DataFrame, synth_data: pd.DataFrame, value_col: str) -> float:
    try:
        real_values = pd.to_numeric(real_data[value_col], errors="coerce").dropna()
        synth_values = pd.to_numeric(synth_data[value_col], errors="coerce").dropna()
        if len(real_values) == 0 or len(synth_values) == 0:
            return np.nan
        return float(synth_values.between(real_values.min(), real_values.max(), inclusive="both").mean())
    except Exception:
        return np.nan


def statistic_msas(
    real_data: pd.DataFrame,
    synth_data: pd.DataFrame,
    id_col: str,
    value_col: str,
    statistic: str = "mean",
) -> float:
    _STAT_FNS = {
        "mean": lambda s: s.mean(),
        "median": lambda s: s.median(),
        "std": lambda s: s.std(),
        "min": lambda s: s.min(),
        "max": lambda s: s.max(),
    }
    try:
        fn = _STAT_FNS.get(statistic)
        if fn is None:
            raise ValueError(f"Unknown statistic '{statistic}'. Choose from: {list(_STAT_FNS)}")
        real_seq = pd.to_numeric(real_data[value_col], errors="coerce")
        synth_seq = pd.to_numeric(synth_data[value_col], errors="coerce")
        real_stats = real_data[[id_col]].assign(_v=real_seq).groupby(id_col)["_v"].agg(fn).dropna()
        synth_stats = synth_data[[id_col]].assign(_v=synth_seq).groupby(id_col)["_v"].agg(fn).dropna()
        if real_stats.empty or synth_stats.empty:
            return np.nan
        return KSComplement.compute(real_data=real_stats, synthetic_data=synth_stats)
    except Exception:
        return np.nan


def correlation_similarity(real_data: pd.DataFrame, synth_data: pd.DataFrame, col1: str, col2: str) -> float:
    try:
        real_pair = real_data[[col1, col2]].apply(pd.to_numeric, errors="coerce").dropna()
        synth_pair = synth_data[[col1, col2]].apply(pd.to_numeric, errors="coerce").dropna()
        rho_real = np.nan
        if len(real_pair) > 1:
            real_std_1 = real_pair[col1].std()
            real_std_2 = real_pair[col2].std()
            if real_std_1 != 0 and real_std_2 != 0:
                rho_real = real_pair[col1].corr(real_pair[col2])

        rho_synth = np.nan
        if len(synth_pair) > 1:
            synth_std_1 = synth_pair[col1].std()
            synth_std_2 = synth_pair[col2].std()
            if synth_std_1 != 0 and synth_std_2 != 0:
                rho_synth = synth_pair[col1].corr(synth_pair[col2])

        if pd.isna(rho_real):
            rho_real = 0.0
        if pd.isna(rho_synth):
            rho_synth = 0.0
        return float(1 - (abs(rho_real - rho_synth) / 2))
    except Exception:
        return np.nan


def contingency_similarity(real_data: pd.DataFrame, synth_data: pd.DataFrame, col_a: str, col_b: str) -> float:
    try:
        real_combined = pd.DataFrame({
            "a": _normalize_categorical_series(real_data[col_a]),
            "b": _normalize_categorical_series(real_data[col_b]),
        })
        synth_combined = pd.DataFrame({
            "a": _normalize_categorical_series(synth_data[col_a]),
            "b": _normalize_categorical_series(synth_data[col_b]),
        })
        p_real = real_combined.value_counts(normalize=True, dropna=False)
        p_synth = synth_combined.value_counts(normalize=True, dropna=False)
        all_keys = p_real.index.append(p_synth.index).drop_duplicates()
        all_keys = p_real.index.union(p_synth.index, sort=False)
        p_real = p_real.reindex(all_keys, fill_value=0.0)
        p_synth = p_synth.reindex(all_keys, fill_value=0.0)
        tvd = 0.5 * np.abs(p_real - p_synth).sum()
        return float(1 - tvd)
    except Exception:
        return np.nan


def select_contingency_pairs(
    real_df: pd.DataFrame,
    cat_cols: list[str],
    max_card_product: int | None = 500,
) -> list[tuple[str, str]]:
    present = [c for c in cat_cols if c in real_df.columns]
    pairs = []
    for i, col_a in enumerate(present):
        card_a = real_df[col_a].nunique()
        for col_b in present[i + 1:]:
            card_b = real_df[col_b].nunique()
            if max_card_product is None or card_a * card_b <= max_card_product:
                pairs.append((col_a, col_b))
    return pairs


def compute_jobs(
    title: str,
    jobs: list[tuple[str, Any, tuple[Any, ...]]],
    parallel: bool = False,
    workers: int | None = None,
    verbose: bool = False,
) -> dict[str, float]:
    if not jobs:
        return {}

    worker_count = workers or min(32, (os.cpu_count() or 1) + 4)
    mode = f" with {worker_count} worker(s)" if parallel else ""
    if verbose:
        print(f"\n[INFO] Computing {title} for {len(jobs)} item(s){mode}...")

    if not parallel or len(jobs) == 1:
        scores = {}
        for key, fn, args in jobs:
            scores[key] = fn(*args)
        return scores

    scores: dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(fn, *args): key
            for key, fn, args in jobs
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                scores[key] = future.result()
            except Exception:
                scores[key] = np.nan

    return {key: scores.get(key, np.nan) for key, _, _ in jobs}


def compute_all_sd_metrics(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    contingency_pairs: list[tuple[str, str]] | None = None,
    sequence_pairs: list[tuple[str, str, str]] | None = None,
    parallel: bool = False,
    workers: int | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    results = {
        "summary": {},
        "per_feature": {},
        "metadata": {
            "n_num_cols": len(num_cols),
            "n_cat_cols": len(cat_cols),
            "num_cols": list(num_cols),
            "cat_cols": list(cat_cols),
            "real_shape": list(real_df.shape),
            "synth_shape": list(synth_df.shape),
        },
    }

    tv_scores = {}
    if cat_cols:
        tv_scores = compute_jobs(
            "TVComplement",
            [(col, compute_tv_complement, (real_df[col], synth_df[col])) for col in cat_cols],
            parallel=parallel,
            workers=workers,
            verbose=verbose,
        )

    ks_scores = {}
    if num_cols:
        ks_scores = compute_jobs(
            "KSComplement",
            [(col, compute_ks_complement, (real_df[col], synth_df[col])) for col in num_cols],
            parallel=parallel,
            workers=workers,
            verbose=verbose,
        )

    stat_scores = {}
    if num_cols:
        stat_scores = compute_jobs(
            "StatisticSimilarity",
            [(col, statistic_similarity, (real_df, synth_df, col)) for col in num_cols],
            parallel=parallel,
            workers=workers,
            verbose=verbose,
        )

    coverage_scores = {}
    if cat_cols:
        coverage_scores = compute_jobs(
            "CategoryCoverage",
            [(col, category_coverage, (real_df, synth_df, col)) for col in cat_cols],
            parallel=parallel,
            workers=workers,
            verbose=verbose,
        )

    range_scores = {}
    if num_cols:
        range_scores = compute_jobs(
            "RangeCoverage",
            [(col, range_coverage, (real_df, synth_df, col)) for col in num_cols],
            parallel=parallel,
            workers=workers,
            verbose=verbose,
        )

    corr_scores = {}
    if len(num_cols) >= 2:
        pairs = [(num_cols[i], num_cols[j]) for i in range(len(num_cols)) for j in range(i + 1, len(num_cols))]
        corr_scores = compute_jobs(
            "CorrelationSim",
            [
                (f"{col1} x {col2}", correlation_similarity, (real_df, synth_df, col1, col2))
                for col1, col2 in pairs
            ],
            parallel=parallel,
            workers=workers,
            verbose=verbose,
        )

    cont_scores = {}
    cat_set = set(cat_cols)
    if contingency_pairs is not None:
        pairs = [(a, b) for a, b in contingency_pairs if a in cat_set and b in cat_set]
    else:
        pairs = [(cat_cols[i], cat_cols[j]) for i in range(len(cat_cols)) for j in range(i + 1, len(cat_cols))]
    if pairs:
        cont_scores = compute_jobs(
            "ContingencySim",
            [
                (f"{col_a} x {col_b}", contingency_similarity, (real_df, synth_df, col_a, col_b))
                for col_a, col_b in pairs
            ],
            parallel=parallel,
            workers=workers,
            verbose=verbose,
        )

    msas_scores = {}
    if sequence_pairs:
        msas_scores = compute_jobs(
            "StatisticMSAS",
            [
                (f"{id_col} x {val_col} [{stat}]", statistic_msas, (real_df, synth_df, id_col, val_col, stat))
                for id_col, val_col, stat in sequence_pairs
                if id_col in real_df.columns and val_col in real_df.columns
                and id_col in synth_df.columns and val_col in synth_df.columns
            ],
            parallel=parallel,
            workers=workers,
            verbose=verbose,
        )

    def safe_mean(scores: dict[str, float]) -> float:
        vals = [v for v in scores.values() if not (isinstance(v, float) and np.isnan(v))]
        return float(np.mean(vals)) if vals else np.nan

    results["summary"] = {
        "KSComplement": safe_mean(ks_scores),
        "TVComplement": safe_mean(tv_scores),
        "StatisticSimilarity": safe_mean(stat_scores),
        "CorrelationSim": safe_mean(corr_scores),
        "RangeCoverage": safe_mean(range_scores),
        "CategoryCoverage": safe_mean(coverage_scores),
        "ContingencySim": safe_mean(cont_scores),
        "StatisticMSAS": safe_mean(msas_scores),
    }
    results["per_feature"] = {
        "TVComplement": tv_scores,
        "KSComplement": ks_scores,
        "StatisticSimilarity": stat_scores,
        "CategoryCoverage": coverage_scores,
        "RangeCoverage": range_scores,
        "CorrelationSim": corr_scores,
        "ContingencySim": cont_scores,
        "StatisticMSAS": msas_scores,
    }
    return results


@dataclass(frozen=True)
class EvalUnit:
    table_name: str
    variants: list[tuple[str, Path]]


def default_output_dir(dataset: str, model_name: str) -> Path:
    return RESULTS_DIR / dataset / model_name / "sd"


def load_metadata(metadata_path: str | Path) -> dict[str, Any]:
    with open(metadata_path, encoding="utf-8") as f:
        return json.load(f)


def table_columns(metadata: dict[str, Any], table_name: str) -> dict[str, dict[str, Any]]:
    return metadata.get("tables", {}).get(table_name, {}).get("columns", {})


def cols_from_metadata(
    metadata: dict[str, Any],
    table_names: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    tables = metadata.get("tables", {})
    selected = table_names or list(tables)
    num_cols: list[str] = []
    cat_cols: list[str] = []

    for table_name in selected:
        for col, info in table_columns(metadata, table_name).items():
            sdtype = info.get("sdtype")
            if sdtype in NUMERIC_SDTYPES and col not in num_cols:
                num_cols.append(col)
            elif sdtype in CATEGORICAL_SDTYPES and col not in cat_cols:
                cat_cols.append(col)
    return num_cols, cat_cols


def real_table_path(real_dir: Path, table_name: str) -> Path:
    if table_name == "hist" and (real_dir / "hist_original.csv").exists():
        return real_dir / "hist_original.csv"
    return real_dir / f"{table_name}.csv"


def synth_table_path(synth_dir: Path, table_name: str) -> Path:
    return synth_dir / f"{table_name}.csv"


def existing_table_path(root_dir: Path, table_name: str, real: bool = False) -> Path:
    candidates = []
    if real:
        candidates.append(real_table_path(root_dir, table_name))
    candidates.extend([
        root_dir / f"{table_name}.csv",
        root_dir / f"{table_name}.parquet",
        root_dir / f"{table_name}.pq",
    ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def discover_variants(synth_dir: Path, table_name: str) -> list[tuple[str, Path]]:
    variants = []
    for variant, suffix in VARIANT_SUFFIXES.items():
        for ext in (".csv", ".parquet", ".pq"):
            path = synth_dir / f"{table_name}{suffix}{ext}"
            if path.exists():
                variants.append((variant, path))
                break
    return variants


def discover_eval_units(metadata: dict[str, Any], synth_dir: Path) -> list[EvalUnit]:
    units = []
    relationships = metadata.get("relationships", [])
    if relationships:
        candidate_tables = list(dict.fromkeys(
            rel["child_table_name"]
            for rel in relationships
            if rel.get("child_table_name") in metadata.get("tables", {})
        ))
    else:
        candidate_tables = list(metadata.get("tables", {}))

    for table_name in candidate_tables:
        variants = discover_variants(synth_dir, table_name)
        if variants:
            units.append(EvalUnit(table_name=table_name, variants=variants))
    return units


def parent_relationships(metadata: dict[str, Any], child_table: str) -> list[dict[str, Any]]:
    return [
        rel
        for rel in metadata.get("relationships", [])
        if rel.get("child_table_name") == child_table
    ]


def parent_source_path(
    table_name: str,
    real_dir: Path,
    synth_dir: Path | None,
    parent_source: str,
) -> Path:
    real_path = existing_table_path(real_dir, table_name, real=True)
    if synth_dir is None or parent_source == "real":
        return real_path

    synth_path = existing_table_path(synth_dir, table_name)
    if parent_source == "synth":
        return synth_path
    return synth_path if synth_path.exists() else real_path


def add_datetime_numeric_cols(
    df: pd.DataFrame,
    metadata: dict[str, Any],
    table_name: str,
) -> pd.DataFrame:
    columns = table_columns(metadata, table_name)
    for col, info in columns.items():
        if info.get("sdtype") != "datetime" or col not in df.columns:
            continue
        numeric_col = f"{col}_numeric"
        if numeric_col in df.columns:
            continue
        df[numeric_col] = pd.to_datetime(
            df[col],
            errors="coerce",
            format=info.get("datetime_format"),
        ).astype("int64") / 1e9
    return df


def load_flat_table(
    metadata: dict[str, Any],
    table_name: str,
    real_dir: Path,
    synth_dir: Path | None = None,
    override_path: Path | None = None,
    parent_source: str = "auto",
    verbose: bool = False,
    _seen: set[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    if _seen is None:
        _seen = set()
    if table_name in _seen:
        raise ValueError(f"Cycle detected while joining metadata relationships: {table_name}")
    _seen = set(_seen)
    _seen.add(table_name)

    path = override_path
    if path is None:
        path = (
            parent_source_path(table_name, real_dir, synth_dir, parent_source)
            if synth_dir is not None
            else existing_table_path(real_dir, table_name, real=True)
        )

    if verbose:
        print(f"[INFO] Loading {table_name} from: {path}")
    df = load_data(path)
    if verbose:
        print(f"[INFO] {table_name} shape: {df.shape}")
    df = add_datetime_numeric_cols(df, metadata, table_name)
    included_tables = [table_name]

    for rel in parent_relationships(metadata, table_name):
        parent_table = rel.get("parent_table_name")
        parent_pk = rel.get("parent_primary_key")
        child_fk = rel.get("child_foreign_key")
        if not parent_table or not parent_pk or not child_fk:
            print(f"[WARN] Incomplete relationship skipped: {rel}")
            continue
        if child_fk not in df.columns:
            print(f"[WARN] Missing child foreign key '{child_fk}' in {table_name}; join skipped")
            continue

        parent_df, parent_tables = load_flat_table(
            metadata,
            parent_table,
            real_dir=real_dir,
            synth_dir=synth_dir,
            parent_source=parent_source,
            verbose=verbose,
            _seen=_seen,
        )
        if parent_pk not in parent_df.columns:
            print(f"[WARN] Missing parent key '{parent_pk}' in {parent_table}; join skipped")
            continue

        df = relationship_join(
            child_df=df,
            parent_df=parent_df,
            parent_table=parent_table,
            parent_pk=parent_pk,
            child_fk=child_fk,
            verbose=verbose,
        )
        included_tables.extend(t for t in parent_tables if t not in included_tables)

    return df, included_tables


def relationship_join(
    child_df: pd.DataFrame,
    parent_df: pd.DataFrame,
    parent_table: str,
    parent_pk: str,
    child_fk: str,
    verbose: bool = False,
) -> pd.DataFrame:
    """Join parent attributes using metadata PK/FK columns.

    Parent and child key columns may have different names. The parent key is
    copied to a temporary merge column so pandas suffixes cannot corrupt a
    same-named non-key column already present in the child frame.
    """
    if child_fk not in child_df.columns:
        print(f"[WARN] Missing child foreign key '{child_fk}'; join skipped")
        return child_df
    if parent_pk not in parent_df.columns:
        print(f"[WARN] Missing parent key '{parent_pk}' in {parent_table}; join skipped")
        return child_df

    merge_key = f"__seq2synth_parent_key__{parent_table}__{parent_pk}"
    while merge_key in child_df.columns or merge_key in parent_df.columns:
        merge_key = f"_{merge_key}"

    parent_subset = parent_df.copy()
    parent_subset[merge_key] = parent_subset[parent_pk]

    keep_cols = [merge_key]
    rename_cols: dict[str, str] = {}
    for col in parent_subset.columns:
        if col in {parent_pk, merge_key}:
            continue
        out_col = col
        if out_col in child_df.columns:
            out_col = f"{parent_table}.{col}"
        keep_cols.append(col)
        if out_col != col:
            rename_cols[col] = out_col

    parent_subset = (
        parent_subset[keep_cols]
        .rename(columns=rename_cols)
        .drop_duplicates(subset=[merge_key])
    )
    before = child_df.shape
    joined = child_df.merge(
        parent_subset,
        left_on=child_fk,
        right_on=merge_key,
        how="left",
    ).drop(columns=[merge_key])
    if verbose:
        print(
            f"[INFO] Joined child FK '{child_fk}' to "
            f"{parent_table}.{parent_pk}: {before} -> {joined.shape}"
        )
    return joined


def build_sequence_pairs(
    metadata: dict[str, Any],
    table_name: str,
    real_df: pd.DataFrame,
    statistics: list[str] | None = None,
) -> list[tuple[str, str, str]]:
    stats = statistics or ["mean"]
    rels = parent_relationships(metadata, table_name)
    if not rels:
        return []
    child_num_cols = [
        col
        for col, info in table_columns(metadata, table_name).items()
        if info.get("sdtype") in NUMERIC_SDTYPES and col in real_df.columns
    ]
    pairs = []
    for rel in rels:
        child_fk = rel.get("child_foreign_key")
        if not child_fk or child_fk not in real_df.columns:
            continue
        for val_col in child_num_cols:
            for stat in stats:
                pairs.append((child_fk, val_col, stat))
    return pairs


def contingency_max_product(args: argparse.Namespace, real_df: pd.DataFrame) -> int | None:
    return 500


def available_cols(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
) -> tuple[list[str], list[str]]:
    available_num = [c for c in num_cols if c in real_df.columns and c in synth_df.columns]
    available_cat = [c for c in cat_cols if c in real_df.columns and c in synth_df.columns]
    missing = [c for c in num_cols + cat_cols if c not in set(available_num + available_cat)]
    if missing:
        print(f"[WARN] Columns missing in one dataset, skipping: {missing}")
    return available_num, available_cat


def print_summary(label: str, results: dict[str, Any]) -> None:
    print(f"\n[AGGREGATED SCORES - {label}]")
    for metric, score in results["summary"].items():
        if isinstance(score, float) and not np.isnan(score):
            print(f"  {metric}: {score:.4f}")
        else:
            print(f"  {metric}: N/A")


def print_final_summaries(summary_records: list[dict[str, Any]]) -> None:
    if not summary_records:
        return

    print("\n" + "=" * 60)
    print("FINAL AGGREGATED SCORES")
    print("=" * 60)
    for record in summary_records:
        print(f"\n[AGGREGATED SCORES - {record['label']}]")
        for metric, score in record["summary"].items():
            if isinstance(score, float) and not np.isnan(score):
                print(f"  {metric}: {score:.4f}")
            else:
                print(f"  {metric}: N/A")
        print(f"  saved_to: {record['output_path']}")


def run_eval_unit(
    args: argparse.Namespace,
    metadata: dict[str, Any],
    unit: EvalUnit,
    output_dir: Path,
    multi_unit: bool,
) -> list[dict[str, Any]]:
    real_dir = Path(args.real_dir)
    synth_dir = Path(args.synth_base_dir)
    model_name = synth_dir.name
    summary_records: list[dict[str, Any]] = []

    print(f"\n{'#' * 60}")
    print(f"TABLE: {unit.table_name}  ({len(unit.variants)} variant(s))")
    print(f"{'#' * 60}")

    print("\n[INFO] Loading real data from metadata relationships...")
    real_df, joined_tables = load_flat_table(metadata, unit.table_name, real_dir)
    num_cols, cat_cols = cols_from_metadata(metadata, joined_tables)
    num_cols.extend(
        f"{col}_numeric"
        for table in joined_tables
        for col, info in table_columns(metadata, table).items()
        if info.get("sdtype") == "datetime"
    )
    contingency_pairs = select_contingency_pairs(
        real_df,
        cat_cols,
        max_card_product=contingency_max_product(args, real_df),
    )
    sequence_pairs = build_sequence_pairs(metadata, unit.table_name, real_df)

    selected_variants = set(args.variants or [])
    variants = [
        (variant, path)
        for variant, path in unit.variants
        if not selected_variants or variant in selected_variants
    ]
    if selected_variants and not variants:
        print(f"[WARN] No requested variants for {unit.table_name}: {sorted(selected_variants)}")
        return summary_records

    for variant, synth_path in variants:
        label = f"{unit.table_name}/{variant}" if multi_unit else variant
        print(f"\n{'=' * 60}")
        print(f"VARIANT: {label}  ({synth_path.name})")
        print(f"{'=' * 60}")

        synth_df, _ = load_flat_table(
            metadata,
            unit.table_name,
            real_dir=real_dir,
            synth_dir=synth_dir,
            override_path=synth_path,
            parent_source=args.parent_source,
        )
        available_num, available_cat = available_cols(real_df, synth_df, num_cols, cat_cols)
        if not available_num and not available_cat:
            print(f"[ERROR] No valid columns for {label}, skipping", file=sys.stderr)
            continue

        print(f"\n{'=' * 60}\nCOMPUTING SD METRICS  [{label}]\n{'=' * 60}")
        results = compute_all_sd_metrics(
            real_df,
            synth_df,
            available_num,
            available_cat,
            contingency_pairs=contingency_pairs,
            sequence_pairs=sequence_pairs,
            parallel=args.parallel,
            workers=args.workers if args.workers is not None else args.n_jobs,
        )
        results["metadata"].update({
            "dataset": args.dataset,
            "model": model_name,
            "variant": variant,
            "table": unit.table_name,
            "joined_tables": joined_tables,
            "parent_source": args.parent_source,
        })

        output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"sd_metrics_{unit.table_name}_{variant}" if multi_unit else f"sd_metrics_{variant}"
        output_path = output_dir / f"{stem}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(nan_to_none(results), f, indent=2, default=str)
        print(f"[INFO] Saved: {output_path}")
        summary_records.append({
            "label": label,
            "output_path": output_path,
            "summary": results["summary"],
        })

    return summary_records


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--real_dir", required=True)
    parser.add_argument("--synth_base_dir", required=True)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument(
        "--parent_source",
        choices=["real", "synth", "auto"],
        default="auto",
        help=(
            "Where joined parent tables are loaded from for synthetic child variants. "
            "auto uses synthetic parents when present and falls back to real parents."
        ),
    )
    parser.add_argument("--cap_strategy", choices=["naive", "kcap", "both"], default="naive")
    parser.add_argument("--kcap_c", type=float, default=14.0)
    parser.add_argument("--n_ref", type=int, default=None)
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Compute independent column and pair metrics concurrently.",
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
    return parser.parse_args(argv)


def run_from_args(args: argparse.Namespace) -> int:
    real_dir = Path(args.real_dir)
    synth_dir = Path(args.synth_base_dir)
    metadata_path = Path(args.metadata) if args.metadata else real_dir / "metadata.json"

    if not metadata_path.exists():
        print(f"[ERROR] metadata.json not found: {metadata_path}", file=sys.stderr)
        return 1
    if not synth_dir.exists():
        print(f"[ERROR] --synth_base_dir does not exist: {synth_dir}", file=sys.stderr)
        return 1

    metadata = load_metadata(metadata_path)
    units = discover_eval_units(metadata, synth_dir)
    if not units:
        print(f"[WARN] No synthetic variant files found in: {synth_dir}", file=sys.stderr)
        return 0

    model_name = synth_dir.name
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args.dataset, model_name)
    print(f"[INFO] Model: {model_name}")
    print(f"[INFO] Detected tables: {[unit.table_name for unit in units]}")
    print(f"[INFO] Dataset: {args.dataset}")
    print(f"[INFO] Real dir: {real_dir}")
    print(f"[INFO] Synth dir: {synth_dir}")
    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] Variants: {args.variants or 'all detected'}")
    print(f"[INFO] Parent source: {args.parent_source}")
    print(f"[INFO] Parallel: {args.parallel}")
    print(f"[INFO] Workers: {args.workers or args.n_jobs or 'auto'}")
    for unit in units:
        print(f"[INFO] {unit.table_name} variants: {[variant for variant, _ in unit.variants]}")

    multi_unit = len(units) > 1
    summary_records: list[dict[str, Any]] = []
    for unit in units:
        summary_records.extend(
            run_eval_unit(args, metadata, unit, output_dir, multi_unit=multi_unit)
        )

    print_final_summaries(summary_records)
    return 0


def main(argv: list[str] | None = None) -> int:
    return run_from_args(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
