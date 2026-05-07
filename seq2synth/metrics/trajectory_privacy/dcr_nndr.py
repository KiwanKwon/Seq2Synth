#!/usr/bin/env python3
"""
Compare general DCR/NNDR vs temporal DCR/NNDR vs N-gram exposure.

General DCR:   For each real row, find distance to closest synthetic row
               across ALL timestamps (no timestamp constraint). Return median.
General NNDR:  For each real row, compute ratio of 1st- to 2nd-nearest
               synthetic distance across ALL timestamps. Return median.
Temporal DCR:  At each shared timestamp t, find each real row's distance to
               closest synthetic row AT THAT SAME t. Average median over T.
Temporal NNDR: At each shared timestamp t, compute NNDR within that timestamp.
               Average median over T.
N-gram:        Sequential pattern-based privacy metric. For each synth entity,
               compute the fraction of consecutive n-row patterns that also
               appear in real data. Numerical columns are equi-distance binned
               (skipped if unique-count <= max_bins). Short trajectories
               (len < n) are compared as a single full-length tuple.

Datasets evaluated: airbnb_tabdit, berka_tabdit, rossmann_tabdit
Models: CLAVADDPM, RCTGAN, RDBDIFF, REALTABFORMER, RelDiff, RGCLD, SDV, TabDiT

Usage:
    python compare_dcr_nndr.py [--datasets ...] [--models ...] [--output results_dcr_nndr.json]
                               [--ngram-n 3] [--ngram-max-bins 10]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def _find_seq2synth_root() -> Path:
    """Find the Seq2Synth project root from either module or script location."""
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / 'data' / 'real').is_dir() and (candidate / 'data' / 'synthetic').is_dir():
            return candidate
    return here.parent


REPO = _find_seq2synth_root()
REAL_BASE = REPO / 'data' / 'real'
SYNTH_BASE = REPO / 'data' / 'synthetic'

# ---------------------------------------------------------------------------
# Low-level helpers (reused from run_trajectory_privacy.py logic)
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in ('.parquet', '.pq'):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _exists(path: Path) -> bool:
    return path.exists() and path.is_file()


def _coerce_numeric(df: pd.DataFrame, num_cols: list) -> pd.DataFrame:
    df = df.copy()
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def _normalize_time_col(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Coerce time_col to canonical form (datetime normalized or string stripped).
    Numeric columns (e.g. row_rank) are left unchanged."""
    if time_col not in df.columns:
        return df
    df = df.copy()
    if pd.api.types.is_numeric_dtype(df[time_col]):
        return df
    try:
        parsed = pd.to_datetime(df[time_col], errors='raise', format='mixed')
        df[time_col] = parsed.dt.normalize()
        return df
    except (ValueError, TypeError, pd.errors.ParserError, OverflowError):
        pass
    df[time_col] = df[time_col].astype(str).str.strip()
    return df


# ---------------------------------------------------------------------------
# Dataset-specific loaders (mirrors run_trajectory_privacy.py)
# ---------------------------------------------------------------------------

def _berka_tabdit_post(df: pd.DataFrame) -> pd.DataFrame:
    if 'date_str' in df.columns:
        return df
    df = df.copy()
    df['date_str'] = (
        ('19' + df['Year'].astype(int).astype(str).str.zfill(2)) + '-'
        + df['Month'].astype(int).astype(str).str.zfill(2) + '-'
        + df['Day'].astype(int).astype(str).str.zfill(2)
    )
    return df


def _airbnb_tabdit_post(df: pd.DataFrame) -> pd.DataFrame:
    if 'row_rank' in df.columns:
        return df
    df = df.sort_values(['user', 'Unnamed: 0']).reset_index(drop=True)
    df['row_rank'] = df.groupby('user').cumcount()
    return df


def _rossmann_tabdit_post(df: pd.DataFrame) -> pd.DataFrame:
    if 'Date' in df.columns:
        return df
    if 'Date_month' in df.columns and 'Date_day' in df.columns:
        df = df.copy()
        df['Date'] = (df['Date_month'].astype(str).str.zfill(2) + '-'
                      + df['Date_day'].astype(str).str.zfill(2))
    return df


def _rossmann_tabdit_merge(hist: pd.DataFrame, store: pd.DataFrame) -> pd.DataFrame:
    hist = hist.copy()
    store = store.copy()
    hist['user'] = pd.to_numeric(hist['user'], errors='coerce').astype('Int64')
    store['user'] = pd.to_numeric(store['user'], errors='coerce').astype('Int64')
    return hist.merge(store, on='user', how='left')


# ---------------------------------------------------------------------------
# Dataset configs
# ---------------------------------------------------------------------------
DATASET_CONFIGS = {
    'airbnb_tabdit': {
        'key_col': 'user',
        'time_col': 'row_rank',
        'num_cols': ['secs_elapsed'],
        # ngram_feature_cols: include categorical+numerical, exclude id/key/time
        'ngram_feature_cols': ['action_type', 'device_type', 'action_detail',
                               'action', 'secs_elapsed'],
        'ngram_num_cols': ['secs_elapsed'],
        'models': ['CLAVADDPM', 'RCTGAN', 'RDBDIFF', 'REALTABFORMER',
                   'RelDiff', 'RGCLD', 'SDV', 'TabDiT'],
        'real_loader': lambda: _airbnb_tabdit_post(
            _read_csv(REAL_BASE / 'airbnb_tabdit' / 'child.csv')),
        'synth_loader': lambda model: (
            (lambda p: _airbnb_tabdit_post(_read_csv(p)) if _exists(p) else None)(
                SYNTH_BASE / 'airbnb_tabdit' / model / 'child_postprocessed.csv')),
    },
    'berka_tabdit': {
        'key_col': 'user',
        'time_col': 'date_str',
        'num_cols': ['amount_trans', 'balance'],
        'ngram_feature_cols': ['type_trans', 'operation', 'k_symbol',
                               'amount_trans', 'balance'],
        'ngram_num_cols': ['amount_trans', 'balance'],
        'models': ['CLAVADDPM', 'RCTGAN', 'RDBDIFF', 'REALTABFORMER',
                   'RelDiff', 'RGCLD', 'SDV', 'TabDiT'],
        'real_loader': lambda: _berka_tabdit_post(
            _read_csv(REAL_BASE / 'berka_tabdit' / 'child.csv')),
        'synth_loader': lambda model: (
            (lambda p: _berka_tabdit_post(_read_csv(p)) if _exists(p) else None)(
                SYNTH_BASE / 'berka_tabdit' / model / 'child_postprocessed.csv')),
    },
    'rossmann_tabdit': {
        'key_col': 'user',
        'time_col': 'Date',
        'num_cols': ['Sales', 'Customers'],
        # historical-side feature columns; store-side static cols are
        # per-entity constants and would dominate hashing, so excluded.
        'ngram_feature_cols': ['Open', 'Promo', 'StateHoliday', 'SchoolHoliday',
                               'DayOfWeek', 'Sales', 'Customers'],
        'ngram_num_cols': ['Sales', 'Customers'],
        'models': ['CLAVADDPM', 'RCTGAN', 'RDBDIFF', 'REALTABFORMER',
                   'RelDiff', 'RGCLD', 'SDV', 'TabDiT'],
        'real_loader': lambda: _rossmann_tabdit_post(_rossmann_tabdit_merge(
            _read_csv(REAL_BASE / 'rossmann_tabdit' / 'historical.csv'),
            _read_csv(REAL_BASE / 'rossmann_tabdit' / 'store.csv'))),
        'synth_loader': lambda model: (
            (lambda hp, sp: _rossmann_tabdit_post(
                _rossmann_tabdit_merge(_read_csv(hp), _read_csv(sp)))
             if _exists(hp) and _exists(sp) else None)(
                SYNTH_BASE / 'rossmann_tabdit' / model / 'historical_postprocessed.csv',
                SYNTH_BASE / 'rossmann_tabdit' / model / 'store.csv')),
    },
}


# ---------------------------------------------------------------------------
# General DCR  (no timestamp constraint)
# ---------------------------------------------------------------------------

def compute_dcr(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    feature_cols: list,
    zero_tol: float = 1e-12,
) -> dict:
    """
    General DCR: for each real row, distance to closest synthetic row
    across ALL rows (no timestamp filtering). Returns the median DCR.

    Lower → higher memorization risk.
    """
    R = real_df[feature_cols].dropna(how='any').to_numpy(dtype=np.float64)
    S = synth_df[feature_cols].dropna(how='any').to_numpy(dtype=np.float64)

    result = {
        'metric': 'dcr',
        'median': float('nan'),
        'mean': float('nan'),
        'n_real_rows': int(len(R)),
        'n_synth_rows': int(len(S)),
        'feature_cols': list(feature_cols),
    }

    if R.shape[0] == 0 or S.shape[0] == 0:
        result['notes'] = 'Empty real or synth after NaN drop.'
        return result

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree',
                            n_jobs=-1, metric='euclidean')
    nbrs.fit(S)
    distances, _ = nbrs.kneighbors(R)
    d = distances[:, 0]

    result['median'] = float(np.median(d))
    result['mean'] = float(np.mean(d))
    result['notes'] = (
        f"General DCR over all {len(R)} real rows vs {len(S)} synth rows."
    )
    return result


# ---------------------------------------------------------------------------
# General NNDR  (no timestamp constraint)
# ---------------------------------------------------------------------------

def compute_nndr(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    feature_cols: list,
    zero_tol: float = 1e-12,
) -> dict:
    """
    General NNDR: for each real row, ratio of distance to 1st vs 2nd
    nearest synthetic row across ALL rows (no timestamp filtering).
    Returns the median NNDR over valid real rows.

    Lower → higher memorization risk.
    """
    R = real_df[feature_cols].dropna(how='any').to_numpy(dtype=np.float64)
    S = synth_df[feature_cols].dropna(how='any').to_numpy(dtype=np.float64)

    result = {
        'metric': 'nndr',
        'median': float('nan'),
        'mean': float('nan'),
        'n_real_rows': int(len(R)),
        'n_synth_rows': int(len(S)),
        'n_excluded_zero': 0,
        'feature_cols': list(feature_cols),
    }

    if R.shape[0] == 0 or S.shape[0] < 2:
        result['notes'] = 'Insufficient rows for NNDR (need >= 2 synth rows).'
        return result

    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree',
                            n_jobs=-1, metric='euclidean')
    nbrs.fit(S)
    distances, _ = nbrs.kneighbors(R)
    d1, d2 = distances[:, 0], distances[:, 1]

    valid = d2 > zero_tol
    n_excl = int((~valid).sum())
    result['n_excluded_zero'] = n_excl

    if not valid.any():
        result['notes'] = 'All d2 <= zero_tol; no valid ratios.'
        return result

    ratios = d1[valid] / d2[valid]
    result['median'] = float(np.median(ratios))
    result['mean'] = float(np.mean(ratios))
    result['notes'] = (
        f"General NNDR over {valid.sum()}/{len(R)} real rows "
        f"(excluded {n_excl} with d2<=zero_tol)."
    )
    return result


# ---------------------------------------------------------------------------
# Temporal DCR  (copied from temporal_privacy_metrics.py)
# ---------------------------------------------------------------------------

def _get_cross_sectional_feature_matrix(grouped, t, feature_cols):
    if t not in grouped.groups:
        return np.zeros((0, len(feature_cols)), dtype=np.float64)
    group = grouped.get_group(t)
    clean = group[feature_cols].dropna(how='any')
    if len(clean) == 0:
        return np.zeros((0, len(feature_cols)), dtype=np.float64)
    return clean.to_numpy(dtype=np.float64)


def _compute_T_eval(real_df, synth_df, time_col):
    real_ts = set(real_df[time_col].dropna().unique().tolist())
    synth_ts = set(synth_df[time_col].dropna().unique().tolist())
    shared = real_ts & synth_ts
    try:
        return sorted(shared)
    except TypeError:
        return list(shared)


def compute_temporal_dcr(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    time_col: str,
    feature_cols: list,
) -> dict:
    """
    Temporal DCR (Paper Eq. 57):
    At each shared timestamp t, compute median distance from each real row
    to its nearest synthetic row at the SAME t. Average medians over T_eval.
    """
    T_eval = _compute_T_eval(real_df, synth_df, time_col)
    n_t_total = len(T_eval)

    result = {
        'metric': 'temporal_dcr',
        'mean_over_T': float('nan'),
        'median_over_T': float('nan'),
        'n_t_total': int(n_t_total),
        'n_t_evaluated': 0,
        'n_t_skipped_empty': 0,
        'feature_cols': list(feature_cols),
    }

    if n_t_total == 0:
        result['notes'] = 'T_eval is empty (no shared timestamps).'
        return result

    real_grp = real_df.groupby(time_col)
    synth_grp = synth_df.groupby(time_col)
    _ = real_grp.groups
    _ = synth_grp.groups

    per_t_values = []
    n_skipped = 0

    for t in T_eval:
        R_t = _get_cross_sectional_feature_matrix(real_grp, t, feature_cols)
        S_t = _get_cross_sectional_feature_matrix(synth_grp, t, feature_cols)
        if R_t.shape[0] == 0 or S_t.shape[0] == 0:
            n_skipped += 1
            continue
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree',
                                n_jobs=1, metric='euclidean')
        nbrs.fit(S_t)
        distances, _ = nbrs.kneighbors(R_t)
        per_t_values.append(float(np.median(distances[:, 0])))

    n_eval = len(per_t_values)
    result['n_t_evaluated'] = int(n_eval)
    result['n_t_skipped_empty'] = int(n_skipped)

    if n_eval == 0:
        result['notes'] = 'No timestamps evaluable (all slices empty).'
        return result

    arr = np.asarray(per_t_values, dtype=np.float64)
    result['mean_over_T'] = float(arr.mean())
    result['median_over_T'] = float(np.median(arr))
    result['notes'] = (
        f"Temporal DCR over {n_eval}/{n_t_total} timestamps "
        f"(skipped {n_skipped} empty)."
    )
    return result


# ---------------------------------------------------------------------------
# Temporal NNDR  (copied from temporal_privacy_metrics.py)
# ---------------------------------------------------------------------------

def compute_temporal_nndr(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    time_col: str,
    feature_cols: list,
    zero_tol: float = 1e-12,
) -> dict:
    """
    Temporal NNDR (Paper Eq. 59):
    At each shared timestamp t, compute median NNDR from each real row
    to its 1st and 2nd nearest synthetic rows at the SAME t. Average over T.
    """
    T_eval = _compute_T_eval(real_df, synth_df, time_col)
    n_t_total = len(T_eval)

    result = {
        'metric': 'temporal_nndr',
        'mean_over_T': float('nan'),
        'median_over_T': float('nan'),
        'n_t_total': int(n_t_total),
        'n_t_evaluated': 0,
        'n_t_skipped': 0,
        'n_queries_excluded_zero': 0,
        'feature_cols': list(feature_cols),
    }

    if n_t_total == 0:
        result['notes'] = 'T_eval is empty (no shared timestamps).'
        return result

    real_grp = real_df.groupby(time_col)
    synth_grp = synth_df.groupby(time_col)
    _ = real_grp.groups
    _ = synth_grp.groups

    per_t_values = []
    n_skipped = 0
    n_excl_total = 0

    for t in T_eval:
        R_t = _get_cross_sectional_feature_matrix(real_grp, t, feature_cols)
        S_t = _get_cross_sectional_feature_matrix(synth_grp, t, feature_cols)
        if R_t.shape[0] == 0 or S_t.shape[0] < 2:
            n_skipped += 1
            continue
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree',
                                n_jobs=1, metric='euclidean')
        nbrs.fit(S_t)
        distances, _ = nbrs.kneighbors(R_t)
        d1, d2 = distances[:, 0], distances[:, 1]
        valid = d2 > zero_tol
        n_excl = int((~valid).sum())
        n_excl_total += n_excl
        if not valid.any():
            n_skipped += 1
            continue
        per_t_values.append(float(np.median(d1[valid] / d2[valid])))

    n_eval = len(per_t_values)
    result['n_t_evaluated'] = int(n_eval)
    result['n_t_skipped'] = int(n_skipped)
    result['n_queries_excluded_zero'] = int(n_excl_total)

    if n_eval == 0:
        result['notes'] = 'No timestamps evaluable.'
        return result

    arr = np.asarray(per_t_values, dtype=np.float64)
    result['mean_over_T'] = float(arr.mean())
    result['median_over_T'] = float(np.median(arr))
    result['notes'] = (
        f"Temporal NNDR over {n_eval}/{n_t_total} timestamps "
        f"(skipped {n_skipped}, excluded_zero_queries {n_excl_total})."
    )
    return result


# ---------------------------------------------------------------------------
# N-gram privacy metric (consecutive n-row pattern matching)
# ---------------------------------------------------------------------------
# Adapted from ngram_privacy_metric_final (1).py:
#   - numerical columns equi-distance binned (bin_edges fit on real)
#   - binning skipped for columns whose unique-count <= max_bins
#   - row → hash on full (categorical + binned-numerical) column set
#   - short trajectories (len < n) compared as a single full-length tuple
#   - hash-based set membership for O(1) lookup

def _fit_bin_edges(real_df: pd.DataFrame,
                   num_cols: list,
                   n_bins: int = 10) -> dict:
    """Equi-distance bin edges fit on real_df. Open at both ends (-inf, +inf).
    Skips columns with unique-count <= n_bins (already categorical-like)."""
    edges: dict = {}
    for col in num_cols:
        if col not in real_df.columns:
            continue
        col_data = real_df[col].dropna()
        if len(col_data) == 0:
            continue
        if col_data.nunique() <= n_bins:
            continue
        _, col_edges = pd.cut(col_data, bins=n_bins, retbins=True)
        edges[col] = [-np.inf] + col_edges[1:-1].tolist() + [np.inf]
    return edges


def _apply_binning(df: pd.DataFrame, bin_edges: dict) -> pd.DataFrame:
    """Replace numerical columns with their bin-label string."""
    df = df.copy()
    for col, ed in bin_edges.items():
        if col not in df.columns:
            continue
        df[col] = pd.cut(df[col], bins=ed, include_lowest=True).astype(str)
    return df


def _normalize_value_for_hash(v):
    """Stable string repr that treats int-valued floats as ints (so e.g. 1.0
    matches 1 across real/synth dtype mismatches), and NaN as '__NA__'."""
    if v is None:
        return '__NA__'
    if isinstance(v, float):
        if np.isnan(v):
            return '__NA__'
        if v.is_integer():
            return str(int(v))
        return repr(v)
    return str(v)


def _build_hash_sequences(df: pd.DataFrame,
                          key_col: str,
                          time_col: str,
                          feature_cols: list) -> dict:
    """Per-entity ordered hash-token list (sorted by time_col).
    Vectorised: cast feature_cols to normalized string then row-tuple-hash."""
    use_cols = [c for c in feature_cols if c in df.columns]
    if not use_cols:
        return {}

    # Sort once globally by (key, time) for determinism + O(N log N).
    df_sorted = df.sort_values([key_col, time_col], kind='mergesort')

    # Normalize values column-wise so int-valued floats match across dtypes.
    feat_cols_data = []
    for c in use_cols:
        col = df_sorted[c]
        if pd.api.types.is_float_dtype(col):
            arr = col.to_numpy()
            out = np.empty(len(arr), dtype=object)
            for i, v in enumerate(arr):
                if np.isnan(v):
                    out[i] = '__NA__'
                elif float(v).is_integer():
                    out[i] = str(int(v))
                else:
                    out[i] = repr(float(v))
            feat_cols_data.append(out)
        else:
            arr = col.where(col.notna(), '__NA__').astype(str).to_numpy()
            feat_cols_data.append(arr)

    # Stack column arrays → row tuples.
    stacked = np.stack(feat_cols_data, axis=1)  # shape (N, C)
    row_tuples = list(map(tuple, stacked))
    row_hashes = np.fromiter((hash(t) for t in row_tuples),
                             dtype=np.int64, count=len(row_tuples))

    keys = df_sorted[key_col].to_numpy()
    sequences: dict = {}
    if len(keys) == 0:
        return sequences
    boundaries = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1], True])
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        sequences[keys[start]] = row_hashes[start:end].tolist()
    return sequences


def _extract_ngrams(hashes: list, n: int) -> list:
    """Sliding n-grams. If len(hashes) < n, return [tuple(hashes)] as one
    unit (so short trajectories still participate)."""
    L = len(hashes)
    if L == 0:
        return []
    if L < n:
        return [tuple(hashes)]
    return [tuple(hashes[i:i + n]) for i in range(L - n + 1)]


def compute_ngram_privacy(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    key_col: str,
    time_col: str,
    feature_cols: list,
    num_cols: list,
    n: int = 3,
    max_bins: int = 10,
) -> dict:
    """
    N-gram exposure privacy score.

    score = mean over synth entities of (matched n-grams / total n-grams).
    Lower → better privacy protection (synth patterns rarely appear in real).
    Higher → synth reproduces real consecutive-row patterns (privacy risk).

    NaN handling: rows where ANY feature column is NaN are dropped before
    building hash sequences. This mirrors the dropna(how='any') behaviour in
    compute_dcr / compute_nndr, and ensures that grid-imputed NaN rows
    inserted by sparse post-processing (e.g. Rossmann) do not participate in
    n-gram matching. The key_col and time_col are preserved so that the
    within-entity ordering remains correct after the drop.
    """
    feature_cols = [c for c in feature_cols
                    if c in real_df.columns and c in synth_df.columns]

    if not feature_cols:
        return {
            'metric': 'ngram_exposure',
            'n': n,
            'max_bins': max_bins,
            'score': float('nan'),
            'notes': 'No overlapping feature columns.',
        }

    bin_num_cols = [c for c in num_cols if c in feature_cols]

    # ── Step 1: filter all-NaN rows BEFORE binning ────────────────────────────
    # Must happen on the original df while NaN values are still actual NaN (not
    # yet converted to the string 'nan' by _apply_binning / pd.cut.astype(str)).
    # Targets sparse-imputation artifacts where EVERY feature column is NaN
    # (e.g. Rossmann grid rows inserted by sparse post-processing).
    # Rows with only partial NaN (e.g. k_symbol=NaN in Berka) are kept, because
    # NaN is a legitimate real-world state for that column.
    n_real_before = len(real_df)
    n_synth_before = len(synth_df)
    real_df_f  = real_df[~real_df[feature_cols].isna().all(axis=1)]
    synth_df_f = synth_df[~synth_df[feature_cols].isna().all(axis=1)]
    n_real_dropped  = n_real_before  - len(real_df_f)
    n_synth_dropped = n_synth_before - len(synth_df_f)
    if n_real_dropped > 0 or n_synth_dropped > 0:
        print(f"    [n-gram drop all-NaN] real: dropped {n_real_dropped}/{n_real_before} rows  "
              f"synth: dropped {n_synth_dropped}/{n_synth_before} rows", flush=True)

    # ── Step 2: fit bin edges and apply binning ────────────────────────────────
    # Fit on the filtered real data so that NaN rows do not affect bin boundaries.
    bin_edges = _fit_bin_edges(real_df_f, bin_num_cols, n_bins=max_bins) if bin_num_cols else {}
    real_b  = _apply_binning(real_df_f,  bin_edges) if bin_edges else real_df_f.copy()
    synth_b = _apply_binning(synth_df_f, bin_edges) if bin_edges else synth_df_f.copy()

    real_seqs = _build_hash_sequences(real_b, key_col, time_col, feature_cols)
    synth_seqs = _build_hash_sequences(synth_b, key_col, time_col, feature_cols)

    # Global real n-gram set for O(1) membership.
    global_real: set = set()
    for hs in real_seqs.values():
        global_real.update(_extract_ngrams(hs, n))

    exposures: list = []
    matched_total = 0
    ngrams_total = 0
    n_synth_entities = len(synth_seqs)
    n_skipped_empty = 0

    for hs in synth_seqs.values():
        s_ngrams = _extract_ngrams(hs, n)
        if not s_ngrams:
            n_skipped_empty += 1
            continue
        m = sum(1 for ng in s_ngrams if ng in global_real)
        exposures.append(m / len(s_ngrams))
        matched_total += m
        ngrams_total += len(s_ngrams)

    score = float(np.mean(exposures)) if exposures else float('nan')
    micro = (matched_total / ngrams_total) if ngrams_total else float('nan')

    return {
        'metric': 'ngram_exposure',
        'n': n,
        'max_bins': max_bins,
        'score': score,                        # macro: avg of per-entity ratios
        'micro_score': float(micro),           # micro: total matched / total ngrams
        'median_exposure': (float(np.median(exposures)) if exposures
                            else float('nan')),
        'feature_cols': list(feature_cols),
        'num_cols_binned': list(bin_edges.keys()),
        'num_cols_skipped_low_cardinality':
            [c for c in bin_num_cols if c not in bin_edges],
        'n_synth_entities': int(n_synth_entities),
        'n_synth_entities_evaluated': int(len(exposures)),
        'n_synth_entities_skipped_empty': int(n_skipped_empty),
        'n_real_entities': int(len(real_seqs)),
        'matched_total': int(matched_total),
        'ngrams_total': int(ngrams_total),
        # rows dropped due to NaN in feature_cols (mirrors DCR/NNDR dropna)
        'n_real_rows_dropped_nan': int(n_real_dropped),
        'n_synth_rows_dropped_nan': int(n_synth_dropped),
        'notes': (
            f"N-gram exposure (n={n}, max_bins={max_bins}). "
            f"Binned numerical: {list(bin_edges.keys())}. "
            f"All-NaN rows filtered BEFORE binning (prevents NaN→'nan' string masking "
            f"by pd.cut): real={n_real_dropped}, synth={n_synth_dropped} dropped. "
            f"Partial-NaN rows (e.g. k_symbol=NaN in Berka) retained as '__NA__'. "
            f"Score=macro mean of per-synth-entity matched/total."
        ),
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_dataset_model(
    dataset: str,
    model: str,
    cfg: dict,
    ngram_n1: int = 1,
    ngram_n2: int = 3,
    ngram_max_bins: int = 10,
) -> Optional[dict]:
    """Load data, compute all four metrics, return result dict or None on error."""
    print(f"\n[{dataset}] [{model}] Loading data...", flush=True)

    real_df_raw = cfg['real_loader']()
    synth_df_raw = cfg['synth_loader'](model)

    if synth_df_raw is None:
        print(f"  SKIP: synthetic files not found for {model}", flush=True)
        return None

    time_col = cfg['time_col']
    num_cols = cfg['num_cols']

    real_df = _normalize_time_col(real_df_raw, time_col)
    synth_df = _normalize_time_col(synth_df_raw, time_col)

    feature_cols = [c for c in num_cols
                    if c in real_df.columns and c in synth_df.columns]
    if not feature_cols:
        print(f"  SKIP: no numerical feature columns overlap", flush=True)
        return None

    real_df = _coerce_numeric(real_df, feature_cols)
    synth_df = _coerce_numeric(synth_df, feature_cols)

    print(f"  real rows: {len(real_df):,}  synth rows: {len(synth_df):,}  "
          f"features: {feature_cols}", flush=True)

    # --- General DCR ---
    print(f"  Computing general DCR...", flush=True)
    dcr_res = compute_dcr(real_df, synth_df, feature_cols)
    print(f"    DCR median={dcr_res['median']:.6f}", flush=True)

    # --- General NNDR ---
    print(f"  Computing general NNDR...", flush=True)
    nndr_res = compute_nndr(real_df, synth_df, feature_cols)
    print(f"    NNDR median={nndr_res['median']:.6f}", flush=True)

    # --- Temporal DCR ---
    print(f"  Computing temporal DCR (time_col={time_col!r})...", flush=True)
    tdcr_res = compute_temporal_dcr(real_df, synth_df, time_col, feature_cols)
    print(f"    Temporal DCR mean_over_T={tdcr_res['mean_over_T']:.6f}  "
          f"n_t={tdcr_res['n_t_evaluated']}/{tdcr_res['n_t_total']}", flush=True)

    # --- Temporal NNDR ---
    print(f"  Computing temporal NNDR...", flush=True)
    tnndr_res = compute_temporal_nndr(real_df, synth_df, time_col, feature_cols)
    print(f"    Temporal NNDR mean_over_T={tnndr_res['mean_over_T']:.6f}  "
          f"n_t={tnndr_res['n_t_evaluated']}/{tnndr_res['n_t_total']}", flush=True)

    # --- N-gram exposure (two sizes for comparison) ---
    key_col = cfg.get('key_col')
    ngram_results: dict = {}
    if key_col is None:
        print(f"  SKIP n-gram: no key_col defined in dataset config.", flush=True)
        ngram_results = None
    else:
        ngram_feature_cols = cfg.get('ngram_feature_cols', feature_cols)
        ngram_num_cols = cfg.get('ngram_num_cols', feature_cols)
        for n_val in (ngram_n1, ngram_n2):
            label = f'n{n_val}'
            print(f"  Computing n-gram exposure (n={n_val}, max_bins={ngram_max_bins}, "
                  f"key_col={key_col!r})...", flush=True)
            ng = compute_ngram_privacy(
                real_df=real_df_raw,
                synth_df=synth_df_raw,
                key_col=key_col,
                time_col=time_col,
                feature_cols=ngram_feature_cols,
                num_cols=ngram_num_cols,
                n=n_val,
                max_bins=ngram_max_bins,
            )
            ngram_results[label] = ng
            print(f"    n={n_val}: macro={ng['score']:.6f}  micro={ng['micro_score']:.6f}  "
                  f"binned={ng['num_cols_binned']}  "
                  f"skipped_low_card={ng['num_cols_skipped_low_cardinality']}",
                  flush=True)

    return {
        'dataset': dataset,
        'model': model,
        'time_col': time_col,
        'key_col': key_col,
        'feature_cols': feature_cols,
        'n_real_rows': int(len(real_df)),
        'n_synth_rows': int(len(synth_df)),
        'dcr': dcr_res,
        'nndr': nndr_res,
        'temporal_dcr': tdcr_res,
        'temporal_nndr': tnndr_res,
        'ngram': ngram_results,
        'ngram_n1': ngram_n1,
        'ngram_n2': ngram_n2,
    }


def print_summary(all_results: list, ngram_n1: int, ngram_n2: int) -> None:
    """Print compact comparison: DCR vs CS-DCR, NNDR vs CS-NNDR, n1-gram vs n2-gram."""
    print("\n" + "=" * 130)
    print(f"SUMMARY: DCR vs CS-DCR | NNDR vs CS-NNDR | "
          f"{ngram_n1}-gram vs {ngram_n2}-gram (macro exposure)")
    print("=" * 130)
    header = (f"{'Dataset':<18} {'Model':<14} "
              f"{'DCR':>11} {'CS-DCR':>12} "
              f"{'NNDR':>10} {'CS-NNDR':>10} "
              f"{f'{ngram_n1}-gram':>11} {f'{ngram_n2}-gram':>11}")
    print(header)
    print("-" * 130)
    for r in all_results:
        if r is None:
            continue
        dcr_gen   = r['dcr']['median']
        dcr_temp  = r['temporal_dcr']['mean_over_T']
        nndr_gen  = r['nndr']['median']
        nndr_temp = r['temporal_nndr']['mean_over_T']
        ng = r.get('ngram') or {}
        ng1 = ng.get(f'n{ngram_n1}', {}).get('score', float('nan'))
        ng2 = ng.get(f'n{ngram_n2}', {}).get('score', float('nan'))

        def fmt(v, w):
            if isinstance(v, float) and np.isnan(v):
                return f"{'nan':>{w}}"
            return f"{v:>{w}.6f}"

        print(f"{r['dataset']:<18} {r['model']:<14} "
              f"{fmt(dcr_gen,11)} {fmt(dcr_temp,12)} "
              f"{fmt(nndr_gen,10)} {fmt(nndr_temp,10)} "
              f"{fmt(ng1,11)} {fmt(ng2,11)}")
    print("=" * 130)


def _nan_to_none(obj):
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_nan_to_none(v) for v in obj]
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if np.isnan(v) else v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets', nargs='+',
                   default=['airbnb_tabdit', 'berka_tabdit', 'rossmann_tabdit'],
                   choices=list(DATASET_CONFIGS.keys()),
                   help='Datasets to evaluate (default: all three)')
    p.add_argument('--models', nargs='+', default=None,
                   help='Models to evaluate (default: all models per dataset)')
    p.add_argument('--output', default='results_dcr_nndr.json',
                   help='Output JSON file (default: results_dcr_nndr.json)')
    p.add_argument('--ngram-n1', type=int, default=1,
                   help='First n-gram size for comparison (default: 1)')
    p.add_argument('--ngram-n2', type=int, default=3,
                   help='Second n-gram size for comparison (default: 3)')
    p.add_argument('--ngram-max-bins', type=int, default=10,
                   help='Max bins for numerical binning in n-gram (default: 10)')
    return p.parse_args()


def main():
    args = parse_args()

    all_results = []

    for dataset in args.datasets:
        cfg = DATASET_CONFIGS[dataset]
        models = args.models if args.models else cfg['models']

        for model in models:
            try:
                res = run_dataset_model(dataset, model, cfg,
                                        ngram_n1=args.ngram_n1,
                                        ngram_n2=args.ngram_n2,
                                        ngram_max_bins=args.ngram_max_bins)
                all_results.append(res)
            except Exception as e:
                print(f"  ERROR [{dataset}][{model}]: {e}", flush=True)
                all_results.append(None)

    valid = [r for r in all_results if r is not None]
    print_summary(valid, ngram_n1=args.ngram_n1, ngram_n2=args.ngram_n2)

    out_path = Path(args.output)
    with open(out_path, 'w') as f:
        json.dump(_nan_to_none(valid), f, indent=2, default=str)
    print(f"\nResults saved to {out_path}", flush=True)


if __name__ == '__main__':
    main()