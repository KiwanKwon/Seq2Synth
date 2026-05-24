"""
Spectral Embedding Infrastructure for Trajectory-Level Privacy Metrics.

This module implements the shared spectral embedding function φ(·) used by
all trajectory-level privacy metrics (DCR, NNDR, NNAA, Hitting Rate, Epsilon
Risk, MIA, Time-ADR) and the TT-Wasserstein fidelity metric.

Pipeline for a multivariate trajectory X ∈ R^{L_k × C}:
    1. Per-trajectory, per-channel z-score normalization.
    2. First-order differencing along time.
    3. Per-channel DFT, retain magnitudes of the first F frequency components.
    4. Concatenate across channels into a flat vector of length F * C.

A min-max normalized variant ~φ (joint min/max over concatenated real+synth
embedding matrix) is also provided for the Spectral MIA hybrid distance.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


# =============================================================================
# Project-convention utilities (verbatim from cross_sectional_metrics_*.py)
# =============================================================================

def _sniff_format(path) -> str:
    """Detect file format from extension or magic bytes."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext in ('.csv', '.tsv', '.txt'):
        return 'csv'
    if ext in ('.parquet', '.pq'):
        return 'parquet'
    if p.is_file():
        with open(p, 'rb') as f:
            header = f.read(4)
        if header == b'PAR1':
            return 'parquet'
    return 'csv'


def load_data(path) -> pd.DataFrame:
    """Load data from CSV or Parquet file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    fmt = _sniff_format(p)
    if fmt == 'parquet':
        return pd.read_parquet(p)
    else:
        return pd.read_csv(p)


def nan_to_none(obj: Any) -> Any:
    """Recursively convert NaN values to None for JSON serialization."""
    if isinstance(obj, dict):
        return {k: nan_to_none(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [nan_to_none(v) for v in obj]
    elif isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def safe_mean(values) -> float:
    """Compute mean, ignoring NaN values. Empty input → nan."""
    if isinstance(values, dict):
        vals = [v for v in values.values() if v is not None and not (isinstance(v, float) and np.isnan(v))]
    else:
        vals = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if vals else float('nan')


# =============================================================================
# Core spectral embedding
# =============================================================================

def spectral_embedding(
    trajectory: np.ndarray,
    F: int = 8,
    epsilon: float = 1e-8,
    precomputed_stats: tuple | None = None,
) -> np.ndarray:
    """
    Compute spectral embedding φ(X) of a single multivariate trajectory.

    Parameters
    ----------
    trajectory : np.ndarray of shape (L_k, C)
        Numerical trajectory data. C must match across all trajectories
        in a dataset.
    F : int
        Number of frequency components to retain. Default 8.
    epsilon : float
        Small constant for z-score numerical stability.
    precomputed_stats : (mu, sigma) or None
        If provided, ``mu`` and ``sigma`` are 1-D arrays of length C giving
        per-channel z-score statistics fit on an external reference set
        (e.g. the union of member + non-member + synth trajectories — see
        prompt §3.2 / §4 step 1). When None, statistics are fit per-trajectory
        (legacy behavior).

    Returns
    -------
    np.ndarray of shape (F * C,)
        Flat spectral embedding (channel-wise concatenation of DFT magnitudes).

    Raises
    ------
    ValueError
        If `trajectory` contains NaN or inf values, or if L_k < 2.

    Notes
    -----
    - If L_k - 1 < F, the differenced signal is zero-padded to length F.
    - Constant channels (σ = 0) produce an all-zero contribution to φ(X).
    """
    X = np.asarray(trajectory, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(
            f"trajectory must be 2D (L_k, C); got shape {X.shape}"
        )
    L_k, C = X.shape

    if L_k < 2:
        raise ValueError(
            f"trajectory too short for first-order differencing: L_k={L_k} < 2"
        )

    if not np.all(np.isfinite(X)):
        raise ValueError(
            "trajectory contains NaN or inf; caller must handle missingness "
            "before calling spectral_embedding()"
        )

    if F < 1:
        raise ValueError(f"F must be >= 1; got F={F}")

    # Step 1: per-channel z-score (per-trajectory or precomputed statistics).
    if precomputed_stats is not None:
        mu_raw, sigma_raw = precomputed_stats
        mu = np.asarray(mu_raw, dtype=np.float64).reshape(1, -1)
        sigma = np.asarray(sigma_raw, dtype=np.float64).reshape(1, -1)
        if mu.shape[1] != C or sigma.shape[1] != C:
            raise ValueError(
                f"precomputed_stats shape mismatch: expected length C={C}, "
                f"got mu={mu.shape}, sigma={sigma.shape}"
            )
    else:
        mu = X.mean(axis=0, keepdims=True)           # (1, C)
        sigma = X.std(axis=0, keepdims=True)         # (1, C)
    X_norm = (X - mu) / (sigma + epsilon)

    # Step 2: first-order differencing along time.
    X_diff = np.diff(X_norm, n=1, axis=0)        # (L_k - 1, C)

    # Step 3: zero-pad (along time axis) if differenced signal shorter than F.
    diff_len = X_diff.shape[0]
    if diff_len < F:
        pad = np.zeros((F - diff_len, C), dtype=np.float64)
        X_diff = np.concatenate([X_diff, pad], axis=0)

    # Step 4: per-channel DFT, magnitudes of first F components.
    spectrum = np.fft.fft(X_diff, axis=0)        # (>=F, C)
    magnitudes = np.abs(spectrum[:F, :])         # (F, C)

    # Flat channel-wise concatenation: [|F(x^1)|_{1:F}, ..., |F(x^C)|_{1:F}].
    # Channel-major flattening => ravel in column-major ('F') order.
    phi = magnitudes.ravel(order='F')            # (F * C,)
    return phi


# =============================================================================
# Batch embedding from DataFrame
# =============================================================================

def _format_offending_keys(keys: list, max_show: int = 5) -> str:
    shown = [str(k) for k in keys[:max_show]]
    extra = len(keys) - max_show
    if extra > 0:
        return ", ".join(shown) + f", ... and {extra} more"
    return ", ".join(shown)


def embed_trajectories(
    df: pd.DataFrame,
    key_col: str,
    time_col: str,
    num_cols: list[str],
    F: int = 8,
    epsilon: float = 1e-8,
    on_nan: str = 'raise',
    sort_by_time: bool = True,
    precomputed_stats: tuple | None = None,
    pad_short_trajectories: bool = False,
) -> tuple[np.ndarray, list, dict]:
    """
    Extract all trajectories from a flat DataFrame and embed them.

    Parameters
    ----------
    df : pd.DataFrame
        Flat (post-join) DataFrame containing key_col, time_col, and num_cols.
    key_col, time_col : str
        Column names for flow identifier and timestamp.
    num_cols : list[str]
        Numerical channels to embed.
    F, epsilon : see spectral_embedding().
    on_nan : {'raise', 'skip'}
        How to handle trajectories with NaN/inf in any num_col.
    sort_by_time : bool
        If True, rows within each group are sorted by time_col before embedding.
    precomputed_stats : (mu, sigma) or None
        If provided, applied to every trajectory (paper §B.4 with global
        statistics — see prompt §3.2 / §4). Both arrays must have length C.
    pad_short_trajectories : bool
        If True, trajectories with L_k < 2 yield a zero-vector embedding
        instead of being dropped (prompt §3.6 — required for orchestrator
        key alignment in the Spectral MIA pipeline).

    Returns
    -------
    embeddings : np.ndarray of shape (N_kept, F * C)
    keys : list
        key_col values for each retained trajectory, same order as `embeddings`.
    report : dict
        Diagnostic metadata.
    """
    if on_nan not in ('raise', 'skip'):
        raise ValueError(
            f"on_nan must be 'raise' or 'skip'; got {on_nan!r}"
        )

    if not num_cols:
        raise ValueError("num_cols is empty; at least one numerical column required")

    missing = [c for c in [key_col, time_col, *num_cols] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Columns missing from df: {missing}"
        )

    C = len(num_cols)
    emb_dim = F * C

    # Validate numeric dtype / coercibility.
    for c in num_cols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise ValueError(
                f"num_col {c!r} has non-numeric dtype {df[c].dtype}; "
                f"clean before calling embed_trajectories()"
            )

    report = {
        'n_trajectories_total': 0,
        'n_skipped_too_short': 0,
        'n_zero_padded_short': 0,
        'n_skipped_nan': 0,
        'n_kept': 0,
        'F': int(F),
        'C': int(C),
        'on_nan': on_nan,
        'precomputed_stats': bool(precomputed_stats is not None),
        'pad_short_trajectories': bool(pad_short_trajectories),
    }

    if len(df) == 0:
        print("[WARN] embed_trajectories: empty DataFrame, returning empty embedding",
              flush=True)
        return np.zeros((0, emb_dim), dtype=np.float64), [], report

    work = df[[key_col, time_col, *num_cols]]
    if sort_by_time:
        work = work.sort_values([key_col, time_col], kind='mergesort')

    embeddings: list[np.ndarray] = []
    keys_kept: list = []
    too_short_keys: list = []
    nan_keys: list = []

    # Preserve first-seen group order.
    for key, group in work.groupby(key_col, sort=False):
        report['n_trajectories_total'] += 1

        traj = group[num_cols].to_numpy(dtype=np.float64, copy=False)
        L_k = traj.shape[0]

        if L_k < 2:
            if pad_short_trajectories:
                report['n_zero_padded_short'] += 1
                embeddings.append(np.zeros(emb_dim, dtype=np.float64))
                keys_kept.append(key)
                continue
            report['n_skipped_too_short'] += 1
            too_short_keys.append(key)
            continue

        if not np.all(np.isfinite(traj)):
            nan_keys.append(key)
            if on_nan == 'raise':
                # Defer raising until we've collected a few offending keys,
                # but don't scan the whole dataset — raise now.
                raise ValueError(
                    f"Trajectory contains NaN or inf in num_cols for key(s): "
                    f"{_format_offending_keys(nan_keys)}. "
                    f"Use on_nan='skip' or impute before embedding."
                )
            # on_nan == 'skip'
            report['n_skipped_nan'] += 1
            continue

        phi = spectral_embedding(
            traj, F=F, epsilon=epsilon,
            precomputed_stats=precomputed_stats,
        )
        embeddings.append(phi)
        keys_kept.append(key)

    if too_short_keys:
        print(
            f"[WARN] embed_trajectories: skipped {len(too_short_keys)} "
            f"trajectory(ies) with L_k < 2",
            flush=True,
        )

    if on_nan == 'skip' and nan_keys:
        print(
            f"[WARN] embed_trajectories: skipped {len(nan_keys)} "
            f"trajectory(ies) containing NaN/inf in num_cols",
            flush=True,
        )

    report['n_kept'] = len(embeddings)

    if embeddings:
        out = np.vstack(embeddings)
    else:
        out = np.zeros((0, emb_dim), dtype=np.float64)

    return out, keys_kept, report


def fit_joint_zscore_stats(
    *dfs: pd.DataFrame,
    num_cols: list,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit per-channel z-score statistics (μ, σ) on the row-wise concatenation
    of the provided DataFrames. Used to ensure the spectral embedding shares
    a common normalization frame across member, non-member, and synthetic
    trajectory sets (prompt §3.2).

    Returns
    -------
    mu : np.ndarray, shape (C,)
    sigma : np.ndarray, shape (C,)
        NaN/Inf entries are ignored (np.nanmean / np.nanstd). Channels whose
        union is entirely NaN return ``mu=0, sigma=0`` and a warning is
        printed; downstream callers must rely on the ``epsilon`` floor in
        ``spectral_embedding`` to avoid division by zero.
    """
    if not num_cols:
        raise ValueError("num_cols must be non-empty")
    arrays = []
    for df in dfs:
        if df is None or len(df) == 0:
            continue
        missing = [c for c in num_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"fit_joint_zscore_stats: columns missing from df: {missing}"
            )
        arrays.append(df[num_cols].to_numpy(dtype=np.float64, copy=False))
    if not arrays:
        # No data — return zero stats; epsilon in spectral_embedding handles div.
        C = len(num_cols)
        return np.zeros(C, dtype=np.float64), np.zeros(C, dtype=np.float64)
    cat = np.vstack(arrays)
    mu = np.nanmean(cat, axis=0)
    sigma = np.nanstd(cat, axis=0)
    # All-NaN columns → mu/sigma become NaN; replace with zeros + warn.
    nan_cols = ~np.isfinite(mu) | ~np.isfinite(sigma)
    if np.any(nan_cols):
        offenders = [c for c, bad in zip(num_cols, nan_cols) if bad]
        print(
            f"[WARN] fit_joint_zscore_stats: all-NaN columns {offenders}; "
            f"mu/sigma set to 0",
            flush=True,
        )
        mu = np.where(nan_cols, 0.0, mu)
        sigma = np.where(nan_cols, 0.0, sigma)
    return mu.astype(np.float64), sigma.astype(np.float64)


# =============================================================================
# Min-max normalization (for Spectral MIA hybrid distance)
# =============================================================================

def minmax_normalize_embeddings(
    *embedding_matrices: np.ndarray,
    epsilon: float = 1e-8,
) -> list[np.ndarray]:
    """
    Per-dimension min-max normalization using statistics computed over
    the concatenated collection of embedding matrices.

    Each dimension is scaled via (x - min) / (max - min + epsilon)
    where min/max are over the concatenated matrix.

    Returns normalized matrices in the SAME order as the inputs, each
    preserving its original shape.
    """
    if len(embedding_matrices) == 0:
        return []

    dims = {M.shape[1] for M in embedding_matrices if M.ndim == 2}
    if len(dims) != 1:
        raise ValueError(
            f"Inconsistent embedding dimensions across inputs: "
            f"shapes={[M.shape for M in embedding_matrices]}"
        )

    non_empty = [M for M in embedding_matrices if M.shape[0] > 0]
    if not non_empty:
        # Everything empty; return inputs unchanged (but as copies).
        return [M.copy() for M in embedding_matrices]

    concat = np.vstack(non_empty)
    mn = concat.min(axis=0, keepdims=True)       # (1, d)
    mx = concat.max(axis=0, keepdims=True)       # (1, d)
    denom = (mx - mn) + epsilon

    out = []
    for M in embedding_matrices:
        if M.shape[0] == 0:
            out.append(M.copy())
        else:
            out.append((M - mn) / denom)
    return out


# =============================================================================
# Pairwise distances
# =============================================================================

def pairwise_l2_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Pairwise ℓ₂ distances between rows of A (n, d) and B (m, d) → (n, m).
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(
            f"A and B must be 2D; got shapes {A.shape} and {B.shape}"
        )
    if A.shape[1] != B.shape[1]:
        raise ValueError(
            f"Second dimension mismatch: A.shape[1]={A.shape[1]} vs "
            f"B.shape[1]={B.shape[1]}"
        )
    return cdist(A, B, metric='euclidean')


# =============================================================================
# Self-tests
# =============================================================================

def _make_df(trajectories: dict, num_cols: list[str],
             key_col: str = 'id', time_col: str = 't') -> pd.DataFrame:
    """Build a flat DataFrame from {key: np.ndarray of shape (L_k, C)}."""
    rows = []
    for key, arr in trajectories.items():
        L_k = arr.shape[0]
        for t in range(L_k):
            row = {key_col: key, time_col: t}
            for j, col in enumerate(num_cols):
                row[col] = arr[t, j]
            rows.append(row)
    return pd.DataFrame(rows)


def _run_self_tests() -> None:
    rng = np.random.default_rng(0)
    num_cols = ['a', 'b', 'c']
    C = len(num_cols)
    F = 8
    failures = 0

    # 1. Shape test
    try:
        trajs = {
            i: rng.standard_normal((rng.integers(5, 21), C))
            for i in range(10)
        }
        df = _make_df(trajs, num_cols)
        emb, keys, rep = embed_trajectories(df, 'id', 't', num_cols, F=F)
        assert emb.shape == (10, F * C), emb.shape
        assert rep['n_kept'] == 10 and rep['n_skipped_too_short'] == 0
        print("[PASS] 1. Shape test")
    except Exception as e:
        failures += 1
        print(f"[FAIL] 1. Shape test: {e}")

    # 2. Too-short skip
    try:
        trajs = {
            0: rng.standard_normal((1, C)),           # L_k=1, skipped
            1: rng.standard_normal((10, C)),
        }
        df = _make_df(trajs, num_cols)
        emb, keys, rep = embed_trajectories(df, 'id', 't', num_cols, F=F)
        assert rep['n_skipped_too_short'] == 1, rep
        assert rep['n_kept'] == 1
        assert keys == [1]
        print("[PASS] 2. Too-short skip")
    except Exception as e:
        failures += 1
        print(f"[FAIL] 2. Too-short skip: {e}")

    # 3. Minimum valid length (L_k=2, diff len=1 < F → zero-padded)
    try:
        trajs = {0: rng.standard_normal((2, C))}
        df = _make_df(trajs, num_cols)
        emb, keys, rep = embed_trajectories(df, 'id', 't', num_cols, F=F)
        assert emb.shape == (1, F * C), emb.shape
        assert np.all(np.isfinite(emb))
        print("[PASS] 3. Minimum valid length with zero-padding")
    except Exception as e:
        failures += 1
        print(f"[FAIL] 3. Minimum valid length: {e}")

    # 4. NaN raise
    try:
        arr = rng.standard_normal((10, C))
        arr[3, 1] = np.nan
        trajs = {42: arr, 43: rng.standard_normal((10, C))}
        df = _make_df(trajs, num_cols)
        try:
            embed_trajectories(df, 'id', 't', num_cols, F=F, on_nan='raise')
        except ValueError as e:
            assert '42' in str(e), f"error message missing offending key: {e}"
            print("[PASS] 4. NaN raise")
        else:
            raise AssertionError("ValueError not raised")
    except Exception as e:
        failures += 1
        print(f"[FAIL] 4. NaN raise: {e}")

    # 5. NaN skip
    try:
        arr = rng.standard_normal((10, C))
        arr[3, 1] = np.nan
        trajs = {42: arr, 43: rng.standard_normal((10, C))}
        df = _make_df(trajs, num_cols)
        emb, keys, rep = embed_trajectories(df, 'id', 't', num_cols,
                                            F=F, on_nan='skip')
        assert rep['n_skipped_nan'] == 1, rep
        assert rep['n_kept'] == 1
        assert keys == [43]
        print("[PASS] 5. NaN skip")
    except Exception as e:
        failures += 1
        print(f"[FAIL] 5. NaN skip: {e}")

    # 6. Min-max normalization
    try:
        a = rng.standard_normal((5, F * C))
        b = rng.standard_normal((7, F * C))
        na, nb = minmax_normalize_embeddings(a, b)
        cat = np.vstack([na, nb])
        tol = 1e-6
        assert cat.min() >= -tol and cat.max() <= 1.0 + tol, (cat.min(), cat.max())
        # Per-dim min should be ~0 and per-dim max ~1 on concatenation
        assert np.allclose(cat.min(axis=0), 0.0, atol=1e-6)
        assert np.allclose(cat.max(axis=0), 1.0, atol=1e-3)
        print("[PASS] 6. Min-max normalization")
    except Exception as e:
        failures += 1
        print(f"[FAIL] 6. Min-max normalization: {e}")

    # 7. Pairwise distance symmetry
    try:
        A = rng.standard_normal((6, F * C))
        D = pairwise_l2_distances(A, A)
        assert D.shape == (6, 6)
        assert np.allclose(D, D.T, atol=1e-10)
        assert np.allclose(np.diag(D), 0.0, atol=1e-10)
        print("[PASS] 7. Pairwise distance symmetry")
    except Exception as e:
        failures += 1
        print(f"[FAIL] 7. Pairwise distance symmetry: {e}")

    # 8. Determinism
    try:
        x = rng.standard_normal((12, C))
        p1 = spectral_embedding(x, F=F)
        p2 = spectral_embedding(x, F=F)
        assert np.array_equal(p1, p2)
        print("[PASS] 8. Determinism")
    except Exception as e:
        failures += 1
        print(f"[FAIL] 8. Determinism: {e}")

    # 9. Variable L_k batch
    try:
        lengths = [3, 5, 8, 15, 30]
        trajs = {i: rng.standard_normal((L, C)) for i, L in enumerate(lengths)}
        df = _make_df(trajs, num_cols)
        emb, keys, rep = embed_trajectories(df, 'id', 't', num_cols, F=F)
        assert emb.shape == (len(lengths), F * C), emb.shape
        assert rep['n_kept'] == len(lengths)
        print("[PASS] 9. Variable L_k batch")
    except Exception as e:
        failures += 1
        print(f"[FAIL] 9. Variable L_k batch: {e}")

    print(f"\n[INFO] Self-tests: {9 - failures}/9 passed", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    _run_self_tests()
