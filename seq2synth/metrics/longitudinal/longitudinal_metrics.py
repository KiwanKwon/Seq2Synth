"""
Longitudinal Metrics for Synthetic Relational Time Series Data

Implements 4 metrics from Seq2Synth paper Section Appendix B.3 :
1. FirstDifference KSComplement (univariate, numerical)
2. Transition Matrix TVDComplement (univariate, categorical)
3. AutoCorrelation Similarity (univariate, numerical)
4. TT-Wasserstein Distance (multivariate, numerical)

These metrics evaluate temporal dynamics and sequential patterns,
as opposed to cross-sectional metrics which compare distributions at fixed time points.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.fft import rfft
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp

_OT_MODULE = None
_OT_IMPORT_ERROR: Exception | None = None
DEFAULT_TT_WASSERSTEIN_NUM_ITER_MAX = 10_000_000
DEFAULT_TT_WASSERSTEIN_MAX_FLOWS = 2000 # 10_000
DEFAULT_TT_WASSERSTEIN_SAMPLE_SEED = 20240506


# =============================================================================
# Data Loading
# =============================================================================

def _sniff_format(path: Path) -> str:
    """Detect file format from extension or magic bytes."""
    ext = path.suffix.lower()
    if ext in ('.csv', '.tsv', '.txt'):
        return 'csv'
    if ext in ('.parquet', '.pq'):
        return 'parquet'
    if path.is_file():
        with open(path, 'rb') as f:
            header = f.read(4)
        if header == b'PAR1':
            return 'parquet'
    return 'csv'


def load_data(path: str) -> pd.DataFrame:
    """Load data from CSV or Parquet file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    fmt = _sniff_format(p)
    if fmt == 'parquet':
        return pd.read_parquet(p)
    else:
        return pd.read_csv(p)


# =============================================================================
# Validation
# =============================================================================

def validate_columns(df: pd.DataFrame, key_col: str, time_col: str,
                     num_cols: list, cat_cols: list, df_name: str) -> None:
    """Validate that required columns exist in the DataFrame."""
    missing = []

    if key_col not in df.columns:
        missing.append(f"key_col '{key_col}'")
    if time_col not in df.columns:
        missing.append(f"time_col '{time_col}'")

    for col in num_cols:
        if col not in df.columns:
            missing.append(f"num_col '{col}'")

    for col in cat_cols:
        if col not in df.columns:
            missing.append(f"cat_col '{col}'")

    if missing:
        raise ValueError(f"[{df_name}] Missing columns: {', '.join(missing)}")


# =============================================================================
# Metric 1: FirstDifference KSComplement
# =============================================================================

def first_difference_ks_complement(real_df: pd.DataFrame, synth_df: pd.DataFrame,
                                   key_col: str, time_col: str, num_col: str) -> float:
    """
    Compute FirstDifference KS Complement for a numerical column.

    Compares the distribution of first differences (Δx = x_j - x_{j-1})
    between real and synthetic data using KS test.

    Returns: 1 - KS_statistic (higher is better, max 1.0)
    """

    def collect_first_diffs(df: pd.DataFrame) -> np.ndarray:
        """Collect all first differences across all flows."""
        all_diffs = []
        for _, group in df.groupby(key_col):
            sorted_vals = group.sort_values(time_col)[num_col].dropna().values
            if len(sorted_vals) >= 2:
                diffs = np.diff(sorted_vals.astype(float))
                all_diffs.extend(diffs)
        return np.array(all_diffs, dtype=float)

    real_diffs = collect_first_diffs(real_df)
    synth_diffs = collect_first_diffs(synth_df)

    if len(real_diffs) < 2 or len(synth_diffs) < 2:
        return np.nan

    # Handle edge case: all differences are identical
    if np.std(real_diffs) == 0 and np.std(synth_diffs) == 0:
        if np.isclose(real_diffs[0], synth_diffs[0]):
            return 1.0

    ks_stat, _ = ks_2samp(real_diffs, synth_diffs)
    return 1.0 - ks_stat


# =============================================================================
# Metric 2: Transition Matrix TVDComplement
# =============================================================================

def transition_matrix_tvd_complement(real_df: pd.DataFrame, synth_df: pd.DataFrame,
                                     key_col: str, time_col: str, cat_col: str) -> float:
    """
    Compute Transition Matrix TVDComplement for a categorical column.

    Compares transition probabilities P(category_j | category_i) at each time step
    using Total Variation Distance (TVD).

    Returns: Average (1 - TVD) across all valid time points (higher is better, max 1.0)
    """

    def build_transition_probs(df: pd.DataFrame) -> tuple[pd.Series, pd.Index]:
        """Build P(to | from, time) as a sparse Series indexed by time/from/to."""
        work = (
            df[[key_col, time_col, cat_col]]
            .dropna(subset=[key_col])
            .sort_values([key_col, time_col])
        )
        if work.empty:
            return pd.Series(dtype=float), pd.Index([])

        next_cat = work.groupby(key_col, sort=False)[cat_col].shift(-1)
        transitions = pd.DataFrame(
            {
                "time": work[time_col].to_numpy(),
                "from": work[cat_col].to_numpy(),
                "to": next_cat.to_numpy(),
            }
        ).dropna(subset=["time", "from", "to"])
        if transitions.empty:
            return pd.Series(dtype=float), pd.Index([])

        categories = pd.Index(pd.unique(transitions[["from", "to"]].to_numpy().ravel()))
        counts = transitions.groupby(["time", "from", "to"], sort=False).size()
        totals = counts.groupby(level=["time", "from"], sort=False).transform("sum")
        return (counts / totals).astype(float), categories

    real_probs, real_categories = build_transition_probs(real_df)
    synth_probs, synth_categories = build_transition_probs(synth_df)

    if real_probs.empty or synth_probs.empty:
        return np.nan

    all_categories = pd.Index(
        pd.unique(np.concatenate([real_categories.to_numpy(), synth_categories.to_numpy()]))
    )

    if len(all_categories) == 0:
        return np.nan

    # T_trans = time points where both real and synth have transitions
    real_times = real_probs.index.get_level_values("time").unique()
    synth_times = synth_probs.index.get_level_values("time").unique()
    T_trans = real_times.intersection(synth_times)

    if len(T_trans) == 0:
        return np.nan

    num_cats = len(all_categories)
    real_common = real_probs[real_probs.index.get_level_values("time").isin(T_trans)]
    synth_common = synth_probs[synth_probs.index.get_level_values("time").isin(T_trans)]
    real_aligned, synth_aligned = real_common.align(synth_common, fill_value=0.0)
    tvd_by_time = (real_aligned - synth_aligned).abs().groupby(level="time").sum()
    scores = 1.0 - (tvd_by_time / (2 * num_cats))
    return float(scores.mean()) if not scores.empty else np.nan


# =============================================================================
# Metric 3: AutoCorrelation Similarity
# =============================================================================

def compute_acf(series: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Compute autocorrelation function for lags 1 to max_lag.

    Returns: Array of ACF values for lags [1, 2, ..., max_lag]
    """
    n = len(series)
    if n < max_lag + 1:
        max_lag = n - 1
    if max_lag <= 0:
        return np.array([])

    mean = np.mean(series)
    var = np.var(series)

    if var == 0:
        return np.zeros(max_lag)

    acf = np.zeros(max_lag)
    for h in range(1, max_lag + 1):
        cov = np.mean((series[:n - h] - mean) * (series[h:] - mean))
        acf[h - 1] = cov / var

    return acf


def autocorrelation_similarity(real_df: pd.DataFrame, synth_df: pd.DataFrame,
                               key_col: str, time_col: str, num_col: str,
                               flow_independence: str, max_lag: int) -> float:
    """
    Compute AutoCorrelation Similarity for a numerical column.

    Mode A (independent): Average ACF across individual flows
    Mode B (dependent): ACF of macro-trajectory (cross-sectional mean at each time)

    Returns: 1 - MAE(ACF_real, ACF_synth) (higher is better, max 1.0)
    """

    if flow_independence == "independent":
        # Mode A: Per-flow ACF averaged across flows
        def avg_acf(df: pd.DataFrame) -> np.ndarray:
            all_acf = []
            for _, group in df.groupby(key_col):
                vals = group.sort_values(time_col)[num_col].dropna().values.astype(float)
                if len(vals) >= max_lag + 1:
                    acf = compute_acf(vals, max_lag)
                    if len(acf) == max_lag:
                        all_acf.append(acf)
            if len(all_acf) == 0:
                return None
            return np.mean(all_acf, axis=0)

        real_acf = avg_acf(real_df)
        synth_acf = avg_acf(synth_df)

    else:
        # Mode B: Macro-trajectory ACF
        def macro_trajectory(df: pd.DataFrame) -> np.ndarray:
            agg = df.groupby(time_col)[num_col].mean().sort_index()
            return agg.values.astype(float)

        real_macro = macro_trajectory(real_df)
        synth_macro = macro_trajectory(synth_df)

        if len(real_macro) < max_lag + 1 or len(synth_macro) < max_lag + 1:
            return np.nan

        real_acf = compute_acf(real_macro, max_lag)
        synth_acf = compute_acf(synth_macro, max_lag)

    if real_acf is None or synth_acf is None:
        return np.nan

    if len(real_acf) == 0 or len(synth_acf) == 0:
        return np.nan

    # Align to shorter length
    H = min(len(real_acf), len(synth_acf))

    mae = np.mean(np.abs(real_acf[:H] - synth_acf[:H]))
    return 1.0 - mae


# =============================================================================
# Metric 4: TT-Wasserstein Distance (Reference Implementation)
# =============================================================================

def _require_ot():
    """Load POT lazily and return the module."""
    global _OT_MODULE, _OT_IMPORT_ERROR

    if _OT_MODULE is None and _OT_IMPORT_ERROR is None:
        try:
            import ot as ot_module
        except ImportError as exc:  # pragma: no cover
            _OT_IMPORT_ERROR = exc
        else:
            _OT_MODULE = ot_module

    if _OT_MODULE is None:
        raise ImportError(
            "POT package is required for TT-Wasserstein. Install with: pip install POT"
        ) from _OT_IMPORT_ERROR
    return _OT_MODULE


def _to_3d_series(x: np.ndarray) -> np.ndarray:
    """Convert (N,T) or (N,T,C) array to (N,T,C)."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 2:
        return x[..., None]
    if x.ndim != 3:
        raise ValueError(f"Expected shape (N,T) or (N,T,C), got {x.shape}")
    return x


def first_order_difference(x: np.ndarray) -> np.ndarray:
    """Compute first-order difference along time axis."""
    return np.diff(x, axis=1)


def l1_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize each row to have L1 norm = 1."""
    denom = np.maximum(x.sum(axis=1, keepdims=True), eps)
    return x / denom


def compute_series_spectrum(
    x: np.ndarray,
    apply_diff: bool = True,
    feature_type: str = "magnitude",
    normalize_spectrum: bool = True,
    channel_reduce: str = "mean",
) -> np.ndarray:
    """
    Compute spectral features from 3D panel data.

    Pipeline: diff -> rfft -> magnitude/power -> channel_reduce -> L1 normalize

    Args:
        x: Array of shape (N, T) or (N, T, C)
        apply_diff: Whether to apply first-order differencing
        feature_type: "magnitude" or "power"
        normalize_spectrum: Whether to L1 normalize rows
        channel_reduce: "none", "mean", "sum", or "median"

    Returns:
        Feature array of shape (N, D) where D depends on settings
    """
    x = _to_3d_series(x)

    if apply_diff:
        x = first_order_difference(x)
    if x.shape[1] < 2:
        raise ValueError("Need time length >= 3 if apply_diff=True, else >= 2.")

    fft_vals = rfft(x, axis=1)
    if feature_type == "magnitude":
        spec = np.abs(fft_vals)
    elif feature_type == "power":
        spec = np.abs(fft_vals) ** 2
    else:
        raise ValueError("feature_type must be 'magnitude' or 'power'")

    if channel_reduce == "none":
        feat = spec.reshape(spec.shape[0], -1)
    elif channel_reduce == "mean":
        feat = spec.mean(axis=2)
    elif channel_reduce == "sum":
        feat = spec.sum(axis=2)
    elif channel_reduce == "median":
        feat = np.median(spec, axis=2)
    else:
        raise ValueError("channel_reduce must be one of {'none','mean','sum','median'}")

    if feat.ndim == 3:
        feat = feat.reshape(feat.shape[0], -1)
    if normalize_spectrum:
        feat = l1_normalize_rows(feat)
    return feat


def make_uniform_weights(n: int) -> np.ndarray:
    """Create uniform probability weights."""
    return np.ones(n, dtype=float) / n


def spectral_wasserstein_distance(
    real_features: np.ndarray,
    syn_features: np.ndarray,
    real_weights: np.ndarray | None = None,
    syn_weights: np.ndarray | None = None,
    cost_metric: str = "euclidean",
    return_transport: bool = True,
    num_iter_max: int = DEFAULT_TT_WASSERSTEIN_NUM_ITER_MAX,
) -> dict:
    """
    Compute Wasserstein distance between spectral feature distributions.

    Uses exact optimal transport via POT library.
    """
    ot_module = _require_ot()

    real_features = np.asarray(real_features, dtype=float)
    syn_features = np.asarray(syn_features, dtype=float)
    n = real_features.shape[0]
    m = syn_features.shape[0]

    if real_weights is None:
        real_weights = make_uniform_weights(n)
    else:
        real_weights = np.asarray(real_weights, dtype=float)
        real_weights = real_weights / real_weights.sum()

    if syn_weights is None:
        syn_weights = make_uniform_weights(m)
    else:
        syn_weights = np.asarray(syn_weights, dtype=float)
        syn_weights = syn_weights / syn_weights.sum()

    cost_matrix = cdist(real_features, syn_features, metric=cost_metric)
    gamma = ot_module.emd(
        real_weights,
        syn_weights,
        cost_matrix,
        numItermax=num_iter_max,
    )
    out = {
        "distance": float(np.sum(gamma * cost_matrix)),
        "cost_matrix": cost_matrix,
        "real_weights": real_weights,
        "syn_weights": syn_weights,
    }
    if return_transport:
        out["transport_plan"] = gamma
    return out


def compute_spectral_wasserstein(
    real_series: np.ndarray,
    synthetic_series: np.ndarray,
    apply_diff: bool = True,
    feature_type: str = "magnitude",
    normalize_spectrum: bool = True,
    channel_reduce: str = "none",
    cost_metric: str = "euclidean",
    real_weights: np.ndarray | None = None,
    syn_weights: np.ndarray | None = None,
    return_features: bool = True,
    return_transport: bool = True,
    num_iter_max: int = DEFAULT_TT_WASSERSTEIN_NUM_ITER_MAX,
) -> dict:
    """
    Compute spectral Wasserstein distance between two time series panels.
    """
    real_feat = compute_series_spectrum(
        real_series,
        apply_diff=apply_diff,
        feature_type=feature_type,
        normalize_spectrum=normalize_spectrum,
        channel_reduce=channel_reduce,
    )
    syn_feat = compute_series_spectrum(
        synthetic_series,
        apply_diff=apply_diff,
        feature_type=feature_type,
        normalize_spectrum=normalize_spectrum,
        channel_reduce=channel_reduce,
    )
    result = spectral_wasserstein_distance(
        real_features=real_feat,
        syn_features=syn_feat,
        real_weights=real_weights,
        syn_weights=syn_weights,
        cost_metric=cost_metric,
        return_transport=return_transport,
        num_iter_max=num_iter_max,
    )
    if return_features:
        result["real_spectral_features"] = real_feat
        result["synthetic_spectral_features"] = syn_feat
        result["config"] = {
            "apply_diff": apply_diff,
            "feature_type": feature_type,
            "normalize_spectrum": normalize_spectrum,
            "channel_reduce": channel_reduce,
            "cost_metric": cost_metric,
        }
    return result


def compute_decomposed_spectral_wasserstein(
    real_series: np.ndarray,
    synthetic_series: np.ndarray,
    feature_cols: list[str],
    apply_diff: bool = True,
    feature_type: str = "magnitude",
    normalize_spectrum: bool = True,
    cost_metric: str = "euclidean",
    real_weights: np.ndarray | None = None,
    syn_weights: np.ndarray | None = None,
    return_transport: bool = True,
    compute_standalone: bool = True,
    num_iter_max: int = DEFAULT_TT_WASSERSTEIN_NUM_ITER_MAX,
) -> dict:
    """
    Compute decomposed spectral Wasserstein distance with per-channel contributions.

    Returns exact channel decomposition with reconstruction verification.
    """
    supported_metrics = {"euclidean", "sqeuclidean", "cityblock"}
    if cost_metric not in supported_metrics:
        raise ValueError(
            f"Exact channel decomposition only supports {sorted(supported_metrics)}, got {cost_metric!r}"
        )

    real_series_3d = _to_3d_series(real_series)
    syn_series_3d = _to_3d_series(synthetic_series)
    c_real = real_series_3d.shape[2]
    c_syn = syn_series_3d.shape[2]

    if c_real != c_syn:
        raise ValueError(f"Channel mismatch: real has {c_real}, synthetic has {c_syn}")
    if len(feature_cols) != c_real:
        raise ValueError(
            f"len(feature_cols)={len(feature_cols)} must match number of channels={c_real}"
        )

    full_result = compute_spectral_wasserstein(
        real_series=real_series_3d,
        synthetic_series=syn_series_3d,
        apply_diff=apply_diff,
        feature_type=feature_type,
        normalize_spectrum=normalize_spectrum,
        channel_reduce="none",
        cost_metric=cost_metric,
        real_weights=real_weights,
        syn_weights=syn_weights,
        return_features=True,
        return_transport=True,
        num_iter_max=num_iter_max,
    )

    gamma = full_result["transport_plan"]
    total_cost = full_result["cost_matrix"]
    real_feat = full_result["real_spectral_features"]
    syn_feat = full_result["synthetic_spectral_features"]

    if real_feat.shape[1] % c_real != 0:
        raise ValueError(
            f"Feature dimension {real_feat.shape[1]} is not divisible by channel count {c_real}"
        )

    spectral_bins = real_feat.shape[1] // c_real
    real_feat_3d = real_feat.reshape(real_feat.shape[0], spectral_bins, c_real)
    syn_feat_3d = syn_feat.reshape(syn_feat.shape[0], spectral_bins, c_real)

    component_cost_matrices: dict[str, np.ndarray] = {}
    component_scores: dict[str, float] = {}
    standalone_component_distances: dict[str, float] = {}

    if cost_metric == "euclidean":
        squared_component_costs: dict[str, np.ndarray] = {}
        for idx, col_name in enumerate(feature_cols):
            real_col_feat = real_feat_3d[:, :, idx]
            syn_col_feat = syn_feat_3d[:, :, idx]
            diff = real_col_feat[:, None, :] - syn_col_feat[None, :, :]
            squared_component_costs[col_name] = np.sum(diff * diff, axis=2)
            if compute_standalone:
                standalone_component_distances[col_name] = spectral_wasserstein_distance(
                    real_features=real_col_feat,
                    syn_features=syn_col_feat,
                    real_weights=full_result["real_weights"],
                    syn_weights=full_result["syn_weights"],
                    cost_metric="euclidean",
                    return_transport=False,
                    num_iter_max=num_iter_max,
                )["distance"]

        safe_total_cost = np.where(total_cost > 0, total_cost, 1.0)
        for col_name in feature_cols:
            component_cost_matrices[col_name] = np.where(
                total_cost > 0,
                squared_component_costs[col_name] / safe_total_cost,
                0.0,
            )
            component_scores[col_name] = float(
                np.sum(gamma * component_cost_matrices[col_name])
            )
    else:
        for idx, col_name in enumerate(feature_cols):
            real_col_feat = real_feat_3d[:, :, idx]
            syn_col_feat = syn_feat_3d[:, :, idx]
            c_col = cdist(real_col_feat, syn_col_feat, metric=cost_metric)
            component_cost_matrices[col_name] = c_col
            component_scores[col_name] = float(np.sum(gamma * c_col))
            if compute_standalone:
                standalone_component_distances[col_name] = spectral_wasserstein_distance(
                    real_features=real_col_feat,
                    syn_features=syn_col_feat,
                    real_weights=full_result["real_weights"],
                    syn_weights=full_result["syn_weights"],
                    cost_metric=cost_metric,
                    return_transport=False,
                    num_iter_max=num_iter_max,
                )["distance"]

    reconstructed_total_cost = np.zeros_like(total_cost)
    for c_col in component_cost_matrices.values():
        reconstructed_total_cost += c_col
    if not np.allclose(reconstructed_total_cost, total_cost, atol=1e-10, rtol=1e-8):
        raise RuntimeError("Component costs do not reconstruct the joint cost matrix")

    total_component_score = sum(component_scores.values())
    if total_component_score > 0:
        relative_contribution = {
            key: value / total_component_score for key, value in component_scores.items()
        }
    else:
        relative_contribution = {key: 0.0 for key in component_scores}

    component_rank = sorted(
        component_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    raw_result = dict(full_result)
    raw_result["component_cost_matrices"] = component_cost_matrices
    raw_result["feature_cols"] = list(feature_cols)
    raw_result["decomposition_metric"] = cost_metric
    if not return_transport:
        raw_result.pop("transport_plan", None)

    return {
        "total_distance": full_result["distance"],
        "component_scores": component_scores,
        "standalone_component_distances": standalone_component_distances,
        "relative_contribution": relative_contribution,
        "component_rank": component_rank,
        "raw_result": raw_result,
    }


def edge_fill_group(group: pd.DataFrame) -> pd.DataFrame:
    filled = group.copy()
    for col in filled.columns:
        series = filled[col]
        first_valid = series.first_valid_index()
        last_valid = series.last_valid_index()
        if first_valid is None or last_valid is None:
            continue

        before_mask = series.index < first_valid
        after_mask = series.index > last_valid
        if before_mask.any():
            series.loc[before_mask] = series.loc[first_valid]
        if after_mask.any():
            series.loc[after_mask] = series.loc[last_valid]
        filled[col] = series
    return filled


def transform_to_3d_array(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    feature_cols: list[str],
    id_index: list | pd.Index | None = None,
    time_index: list | pd.Index | None = None,
    fill_method: str | None = "edge_nearest",
    fill_value: float = 0.0,
    return_metadata: bool = False,
):
    """
    Transform DataFrame to 3D array (N, T, C).

    Args:
        df: Input DataFrame
        id_col: Column name for entity ID
        time_col: Column name for timestamp
        feature_cols: List of feature column names
        id_index: Optional fixed ID index
        time_index: Optional fixed time index
        fill_method: Method for filling missing values
                     (None, 'none', 'edge_nearest', 'interpolate', 'ffill',
                      'tsmean', 'crossmean', 'entitymean', 'zero')
        fill_value: Value to use for 'zero' fill method
        return_metadata: Whether to return metadata dict

    Returns:
        arr_3d: Array of shape (N, T, C)
        metadata: (optional) dict with id_index, time_index, feature_cols
    """
    feature_cols = list(feature_cols)

    work = df[[id_col, time_col, *feature_cols]].copy()
    work = work.drop_duplicates(subset=[id_col, time_col])
    work[feature_cols] = work[feature_cols].apply(pd.to_numeric, errors="coerce").astype(float)
    if pd.api.types.is_datetime64_any_dtype(work[time_col]) or work[time_col].dtype == "O":
        work[time_col] = pd.to_datetime(work[time_col], errors="coerce")

    if id_index is None:
        id_index = pd.Index(sorted(work[id_col].dropna().unique()), name=id_col)
    else:
        id_index = pd.Index(id_index, name=id_col)

    if time_index is None:
        time_values = work[time_col].dropna().unique()
        time_index = pd.Index(sorted(time_values), name=time_col)
    else:
        time_index = pd.Index(time_index, name=time_col)

    full_index = pd.MultiIndex.from_product([id_index, time_index], names=[id_col, time_col])
    aligned = (
        work.set_index([id_col, time_col])[feature_cols]
        .sort_index()
        .reindex(full_index)
    )

    if fill_method not in (None, "none"):
        filled_groups = []
        timestamp_means = None
        if fill_method in {"tsmean", "crossmean"}:
            timestamp_means = aligned.groupby(level=time_col).mean(numeric_only=True)

        for current_id in id_index:
            group = aligned.xs(current_id, level=id_col, drop_level=True).copy()
            if fill_method == "edge_nearest":
                group = edge_fill_group(group)
            elif fill_method == "interpolate":
                method = "time" if isinstance(group.index, pd.DatetimeIndex) else "index"
                group = group.interpolate(method=method, limit_direction="both")
                group = group.ffill().bfill()
            elif fill_method == "ffill":
                group = group.ffill().bfill()
            elif fill_method in {"tsmean", "crossmean"}:
                if timestamp_means is None:
                    raise RuntimeError(
                        "timestamp_means should be initialized for fill_method in {'tsmean', 'crossmean'}"
                    )
                group = group.fillna(timestamp_means.reindex(group.index))
            elif fill_method == "entitymean":
                group = group.fillna(group.mean(numeric_only=True))
            elif fill_method == "zero":
                group = group.fillna(fill_value)
            else:
                raise ValueError(
                    "fill_method must be one of {None, 'none', 'edge_nearest', 'interpolate', "
                    "'ffill', 'tsmean', 'crossmean', 'entitymean', 'zero'}"
                )

            if fill_method != "zero":
                group = group.fillna(fill_value)

            group[id_col] = current_id
            filled_groups.append(group.reset_index())

        aligned = (
            pd.concat(filled_groups, ignore_index=True)
            .set_index([id_col, time_col])[feature_cols]
            .reindex(full_index)
        )

    arr_3d = aligned.to_numpy(dtype=float).reshape(len(id_index), len(time_index), len(feature_cols))
    if return_metadata:
        return arr_3d, {
            "id_index": id_index,
            "time_index": time_index,
            "feature_cols": feature_cols,
        }
    return arr_3d


def standardize_panels(
    real_panel: np.ndarray,
    syn_panel: np.ndarray,
    reference: str = "real",
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply panel-level z-score normalization.

    Args:
        real_panel: Real data array (N, T, C)
        syn_panel: Synthetic data array (M, T, C)
        reference: "real", "pooled", or "none"
        eps: Small value for numerical stability

    Returns:
        real_scaled: Normalized real panel
        syn_scaled: Normalized synthetic panel
        mean: Mean vector used for normalization
        std: Std vector used for normalization
    """
    real_panel = np.asarray(real_panel, dtype=float)
    syn_panel = np.asarray(syn_panel, dtype=float)

    if reference == "none":
        mean = np.zeros(real_panel.shape[-1], dtype=float)
        std = np.ones(real_panel.shape[-1], dtype=float)
        return real_panel.copy(), syn_panel.copy(), mean, std

    if reference == "real":
        fit_block = real_panel.reshape(-1, real_panel.shape[-1])
    elif reference == "pooled":
        fit_block = np.concatenate(
            [
                real_panel.reshape(-1, real_panel.shape[-1]),
                syn_panel.reshape(-1, syn_panel.shape[-1]),
            ],
            axis=0,
        )
    else:
        raise ValueError("reference must be one of {'real', 'pooled', 'none'}")

    mean = fit_block.mean(axis=0)
    std = fit_block.std(axis=0, ddof=0)
    std = np.where(std > eps, std, 1.0)

    real_scaled = (real_panel - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
    syn_scaled = (syn_panel - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
    return real_scaled, syn_scaled, mean, std


def deterministic_panel_sample(
    panel: np.ndarray,
    max_flows: int | None,
    seed: int,
    label: str,
) -> tuple[np.ndarray, bool]:
    """Deterministically sample panel rows for scalable exact OT."""
    if max_flows is None or max_flows <= 0 or panel.shape[0] <= max_flows:
        return panel, False
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(panel.shape[0], size=max_flows, replace=False))
    print(f"    Sampling {label} TT-W flows: {panel.shape[0]} -> {max_flows} (seed={seed})")
    return panel[indices], True


def tt_wasserstein_distance(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    key_col: str,
    time_col: str,
    num_cols: list,
    n_fft_components: int = 10,  # API compatibility (not used in new implementation)
    zscore_reference: str = "real",
    apply_diff: bool = True,
    feature_type: str = "magnitude",
    normalize_spectrum: bool = True,
    cost_metric: str = "euclidean",
    fill_method: str = "edge_nearest",
    compute_standalone: bool = True,
    num_iter_max: int = DEFAULT_TT_WASSERSTEIN_NUM_ITER_MAX,
    max_flows: int | None = DEFAULT_TT_WASSERSTEIN_MAX_FLOWS,
    sample_seed: int = DEFAULT_TT_WASSERSTEIN_SAMPLE_SEED,
) -> tuple:
    """
    Compute TT-Wasserstein Distance between real and synthetic data.

    Wrapper that converts DataFrames to 3D panels and calls the reference
    implementation for decomposed spectral Wasserstein distance.

    Args:
        real_df: Real data DataFrame
        synth_df: Synthetic data DataFrame
        key_col: Column name for flow/entity ID
        time_col: Column name for timestamp
        num_cols: List of numerical feature column names
        n_fft_components: (Deprecated, kept for API compatibility)
        zscore_reference: Z-score reference ("real", "pooled", "none")
        apply_diff: Whether to apply first-order differencing
        feature_type: "magnitude" or "power"
        normalize_spectrum: Whether to L1 normalize spectral features
        cost_metric: Distance metric for OT ("euclidean", "sqeuclidean", "cityblock")
        fill_method: Fill method for missing values
            ("edge_nearest", "ffill", "interpolate", "tsmean", "crossmean",
            "entitymean", "zero")
        compute_standalone: Whether to run per-feature standalone OT distances.
        num_iter_max: POT EMD maximum iterations.
        max_flows: Maximum real/synthetic trajectories per side for TT-W exact OT.
            Set to 0 or None to disable TT-W trajectory sampling.
        sample_seed: Seed used for deterministic trajectory sampling.

    Returns:
        total_distance: float (lower is better)
        component_scores: dict mapping column name to contribution score
        relative_contribution: dict mapping column name to relative contribution ratio
        standalone_distances: dict mapping column name to standalone distance
    """
    feature_cols = list(num_cols)

    if len(feature_cols) == 0:
        return np.nan, {}, {}, {}

    # Step 1: DataFrame -> 3D panel
    # Use real's time_index as reference
    print(f"    Converting DataFrames to 3D panels...")
    try:
        real_panel, real_meta = transform_to_3d_array(
            real_df,
            id_col=key_col,
            time_col=time_col,
            feature_cols=feature_cols,
            fill_method=fill_method,
            return_metadata=True,
        )

        syn_panel, syn_meta = transform_to_3d_array(
            synth_df,
            id_col=key_col,
            time_col=time_col,
            feature_cols=feature_cols,
            time_index=real_meta["time_index"],  # Use real's time_index
            fill_method=fill_method,
            return_metadata=True,
        )
    except Exception as e:
        print(f"    WARNING: Failed to convert to 3D panel: {e}")
        return np.nan, {}, {}, {}

    print(f"    Real panel shape: {real_panel.shape}, Synth panel shape: {syn_panel.shape}")

    # Check minimum time length for differencing
    if apply_diff and real_panel.shape[1] < 3:
        print(f"    WARNING: Time length {real_panel.shape[1]} too short for differencing (need >= 3)")
        return np.nan, {}, {}, {}

    # Step 2: Panel-level z-score normalization
    print(f"    Applying z-score normalization (reference={zscore_reference})...")
    real_scaled, syn_scaled, mean_vec, std_vec = standardize_panels(
        real_panel, syn_panel, reference=zscore_reference
    )
    real_scaled, real_sampled = deterministic_panel_sample(
        real_scaled,
        max_flows=max_flows,
        seed=sample_seed,
        label="real",
    )
    syn_scaled, syn_sampled = deterministic_panel_sample(
        syn_scaled,
        max_flows=max_flows,
        seed=sample_seed + 1,
        label="synthetic",
    )
    if real_sampled or syn_sampled:
        print(
            "    Sampled panel shapes for TT-W: "
            f"Real {real_scaled.shape}, Synth {syn_scaled.shape}"
        )

    # Step 3: Compute decomposed spectral Wasserstein
    print(f"    Computing decomposed spectral Wasserstein (numItermax={num_iter_max})...")
    try:
        result = compute_decomposed_spectral_wasserstein(
            real_scaled,
            syn_scaled,
            feature_cols=feature_cols,
            apply_diff=apply_diff,
            feature_type=feature_type,
            normalize_spectrum=normalize_spectrum,
            cost_metric=cost_metric,
            compute_standalone=compute_standalone,
            num_iter_max=num_iter_max,
        )
    except ImportError as e:
        print(f"    ERROR: {e}")
        return np.nan, {}, {}, {}
    except Exception as e:
        print(f"    WARNING: Failed to compute spectral Wasserstein: {e}")
        return np.nan, {}, {}, {}

    return (
        result["total_distance"],
        result["component_scores"],
        result["relative_contribution"],
        result["standalone_component_distances"],
    )


# =============================================================================
# Aggregation
# =============================================================================

def compute_all_longitudinal_metrics(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    key_col: str,
    time_col: str,
    num_cols: list,
    cat_cols: list,
    flow_independence: str,
    max_lag: int,
    n_fft_components: int,
    zscore_reference: str = "real",
    fill_method: str = "edge_nearest",
    skip_tt_wasserstein: bool = False,
    compute_tt_standalone: bool = True,
    tt_num_iter_max: int = DEFAULT_TT_WASSERSTEIN_NUM_ITER_MAX,
    tt_max_flows: int | None = DEFAULT_TT_WASSERSTEIN_MAX_FLOWS,
    tt_sample_seed: int = DEFAULT_TT_WASSERSTEIN_SAMPLE_SEED,
) -> dict:
    """
    Compute all 4 longitudinal metrics.

    Returns: Dictionary with summary scores, per-feature breakdown, and metadata
    """
    results = {
        'summary': {},
        'per_feature': {},
        'metadata': {
            'flow_independence': flow_independence,
            'max_lag': max_lag,
            'n_fft_components': n_fft_components,
            'zscore_reference': zscore_reference,
            'fill_method': fill_method,
            'compute_tt_standalone': compute_tt_standalone,
            'tt_num_iter_max': tt_num_iter_max,
            'tt_max_flows': tt_max_flows,
            'tt_sample_seed': tt_sample_seed,
            'n_real_flows': int(real_df[key_col].nunique()),
            'n_synth_flows': int(synth_df[key_col].nunique()),
            'n_real_rows': len(real_df),
            'n_synth_rows': len(synth_df),
        }
    }

    # --- FirstDifference KSComplement (per numerical feature) ---
    print("\n[COMPUTING] FirstDifference KSComplement...")
    fd_scores = {}
    for col in num_cols:
        print(f"  Processing: {col}")
        fd_scores[col] = first_difference_ks_complement(
            real_df, synth_df, key_col, time_col, col
        )
    print("  Done.")

    # --- Transition Matrix TVDComplement (per categorical feature) ---
    print("\n[COMPUTING] Transition Matrix TVDComplement...")
    tm_scores = {}
    for col in cat_cols:
        print(f"  Processing: {col}")
        tm_scores[col] = transition_matrix_tvd_complement(
            real_df, synth_df, key_col, time_col, col
        )
    print("  Done.")

    # --- AutoCorrelation Similarity (per numerical feature) ---
    print(f"\n[COMPUTING] AutoCorrelation Similarity (Mode: {flow_independence})...")
    ac_scores = {}
    for col in num_cols:
        print(f"  Processing: {col}")
        ac_scores[col] = autocorrelation_similarity(
            real_df, synth_df, key_col, time_col, col,
            flow_independence, max_lag
        )
    print("  Done.")

    # --- TT-Wasserstein Distance (multivariate, all numerical) ---
    print("\n[COMPUTING] TT-Wasserstein Distance...")
    tt_distance = np.nan
    tt_component = {}
    tt_relative = {}
    tt_standalone = {}
    if not skip_tt_wasserstein and len(num_cols) >= 1:
        tt_distance, tt_component, tt_relative, tt_standalone = tt_wasserstein_distance(
            real_df, synth_df, key_col, time_col, num_cols,
            n_fft_components=n_fft_components,
            zscore_reference=zscore_reference,
            fill_method=fill_method,
            compute_standalone=compute_tt_standalone,
            num_iter_max=tt_num_iter_max,
            max_flows=tt_max_flows,
            sample_seed=tt_sample_seed,
        )
    print("  Done.")

    # --- Aggregation ---
    def safe_mean(d: dict) -> float:
        vals = [v for v in d.values() if not (isinstance(v, float) and np.isnan(v))]
        return float(np.mean(vals)) if vals else np.nan

    results['summary'] = {
        'FirstDifferenceKSComplement': safe_mean(fd_scores),
        'TransitionMatrixTVDComplement': safe_mean(tm_scores),
        'AutoCorrelationSimilarity': safe_mean(ac_scores),
        'TTWassersteinDistance': tt_distance,
    }

    results['per_feature'] = {
        'FirstDifferenceKSComplement': fd_scores,
        'TransitionMatrixTVDComplement': tm_scores,
        'AutoCorrelationSimilarity': ac_scores,
        'TTWasserstein_ComponentScores': tt_component,
        'TTWasserstein_RelativeContribution': tt_relative,
        'TTWasserstein_StandaloneDistances': tt_standalone,
    }

    return results


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Longitudinal Metrics for Synthetic Relational Time Series Data"
    )
    parser.add_argument(
        "--real", type=str, required=True,
        help="Path to real (original) data file"
    )
    parser.add_argument(
        "--synth", type=str, required=True,
        help="Path to post-processed synthetic data file"
    )
    parser.add_argument(
        "--key_col", type=str, required=True,
        help="Flow key column name"
    )
    parser.add_argument(
        "--time_col", type=str, required=True,
        help="Timestamp column name"
    )
    parser.add_argument(
        "--num_cols", type=str, nargs="*", default=[],
        help="List of numerical feature column names"
    )
    parser.add_argument(
        "--cat_cols", type=str, nargs="*", default=[],
        help="List of categorical feature column names"
    )
    parser.add_argument(
        "--flow_independence", type=str, choices=["independent", "dependent"],
        default="dependent",
        help="Flow independence mode for AutoCorrelation (default: dependent)"
    )
    parser.add_argument(
        "--max_lag", type=int, default=12,
        help="Maximum lag for AutoCorrelation (default: 12)"
    )
    parser.add_argument(
        "--n_fft_components", type=int, default=10,
        help="(Deprecated) Number of DFT components (kept for API compatibility)"
    )
    parser.add_argument(
        "--zscore_reference", type=str, choices=["real", "pooled", "none"],
        default="real",
        help="Z-score reference for TT-Wasserstein (default: real)"
    )
    parser.add_argument(
        "--fill_method", type=str, default="ffill",
        help="Fill method for 3D panel construction (ffill, interpolate, zero)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for results (.json)"
    )
    parser.add_argument(
        "--skip_tt_wasserstein", action="store_true",
        help="Skip TT-Wasserstein computation."
    )
    parser.add_argument(
        "--tt_standalone",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Compute per-feature standalone TT-Wasserstein distances. "
            "Default is off to skip extra per-feature OT solves."
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

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # 1. Load data
    print("=" * 60)
    print("LONGITUDINAL METRICS")
    print("=" * 60)

    print(f"\n[LOADING] Real data: {args.real}")
    real_df = load_data(args.real)

    print(f"[LOADING] Synthetic data: {args.synth}")
    synth_df = load_data(args.synth)

    # 2. Validate columns
    print("\n[VALIDATING] Checking columns...")
    validate_columns(real_df, args.key_col, args.time_col,
                     args.num_cols, args.cat_cols, "Real")
    validate_columns(synth_df, args.key_col, args.time_col,
                     args.num_cols, args.cat_cols, "Synth")
    print("  Columns validated.")

    # 3. Parse timestamps
    print("\n[PARSING] Converting timestamps...")
    for df, name in [(real_df, "Real"), (synth_df, "Synth")]:
        try:
            df[args.time_col] = pd.to_datetime(df[args.time_col])
            print(f"  {name}: datetime")
        except (ValueError, TypeError):
            try:
                df[args.time_col] = pd.to_numeric(df[args.time_col])
                print(f"  {name}: numeric")
            except (ValueError, TypeError):
                print(f"  {name}: kept as-is")

    # 4. Print info
    print("\n[INFO] Dataset Overview")
    print(f"  Real: {len(real_df):,} rows, {real_df[args.key_col].nunique():,} flows")
    print(f"  Synth: {len(synth_df):,} rows, {synth_df[args.key_col].nunique():,} flows")
    print(f"  Flow independence: {args.flow_independence}")
    print(f"  Max lag: {args.max_lag}")
    print(f"  Z-score reference: {args.zscore_reference}")
    print(f"  Fill method: {args.fill_method}")
    print(f"  Numerical columns: {args.num_cols}")
    print(f"  Categorical columns: {args.cat_cols}")

    # 5. Compute metrics
    results = compute_all_longitudinal_metrics(
        real_df, synth_df, args.key_col, args.time_col,
        args.num_cols, args.cat_cols, args.flow_independence,
        args.max_lag, args.n_fft_components,
        zscore_reference=args.zscore_reference,
        fill_method=args.fill_method,
        skip_tt_wasserstein=args.skip_tt_wasserstein,
        compute_tt_standalone=args.tt_standalone,
        tt_num_iter_max=args.tt_num_iter_max,
        tt_max_flows=args.tt_max_flows,
        tt_sample_seed=args.tt_sample_seed,
    )

    # 6. Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print("\n[AGGREGATED SCORES]")
    for metric, score in results['summary'].items():
        if isinstance(score, float) and not np.isnan(score):
            if 'Distance' in metric:
                print(f"  {metric}: {score:.6f} (lower=better)")
            else:
                print(f"  {metric}: {score:.4f} (higher=better, max=1.0)")
        else:
            print(f"  {metric}: N/A")

    print("\n[PER-FEATURE BREAKDOWN]")
    for metric, breakdown in results['per_feature'].items():
        if breakdown:
            print(f"\n  {metric}:")
            for feature, score in breakdown.items():
                if isinstance(score, float):
                    if not np.isnan(score):
                        print(f"    {feature}: {score:.4f}")
                    else:
                        print(f"    {feature}: N/A")
                else:
                    print(f"    {feature}: {score}")

    # 7. Save to JSON
    if args.output:
        def nan_to_none(obj):
            """Convert NaN values to None for JSON serialization."""
            if isinstance(obj, dict):
                return {k: nan_to_none(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [nan_to_none(v) for v in obj]
            elif isinstance(obj, float) and np.isnan(obj):
                return None
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            return obj

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(nan_to_none(results), f, indent=2, default=str)
        print(f"\n[INFO] Results saved to: {args.output}")

    print("\n[DONE]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
