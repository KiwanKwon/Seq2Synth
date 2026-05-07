"""
GPU-accelerated longitudinal TT-Wasserstein utilities.

This module intentionally does not modify the reference implementation in
longitudinal_metrics.py.  It mirrors the public TT-Wasserstein entry points but
uses PyTorch tensor ops and entropic Sinkhorn optimal transport, which can run
on CUDA when a CUDA-enabled PyTorch build is installed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from seq2synth.metrics.longitudinal.longitudinal_metrics import (
    autocorrelation_similarity,
    first_difference_ks_complement,
    standardize_panels,
    transition_matrix_tvd_complement,
    transform_to_3d_array,
)


_TORCH_MODULE = None
_TORCH_IMPORT_ERROR: Exception | None = None


def _require_torch():
    """Load PyTorch lazily so importing this file does not require torch."""
    global _TORCH_MODULE, _TORCH_IMPORT_ERROR

    if _TORCH_MODULE is None and _TORCH_IMPORT_ERROR is None:
        try:
            import torch as torch_module
        except ImportError as exc:  # pragma: no cover
            _TORCH_IMPORT_ERROR = exc
        else:
            _TORCH_MODULE = torch_module

    if _TORCH_MODULE is None:
        raise ImportError(
            "PyTorch is required for GPU TT-Wasserstein. Install a CUDA-enabled "
            "torch build to run on GPU."
        ) from _TORCH_IMPORT_ERROR
    return _TORCH_MODULE


def resolve_torch_device(device: str | None = "auto"):
    """Resolve a torch device, preferring CUDA for device='auto'."""
    torch = _require_torch()
    if device in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is False")
    return resolved


def _to_3d_tensor(x: Any, *, device, dtype):
    torch = _require_torch()
    x_t = torch.as_tensor(x, dtype=dtype, device=device)
    if x_t.ndim == 2:
        return x_t.unsqueeze(-1)
    if x_t.ndim != 3:
        raise ValueError(f"Expected shape (N,T) or (N,T,C), got {tuple(x_t.shape)}")
    return x_t


def l1_normalize_rows_torch(x, eps: float = 1e-12):
    return x / x.sum(dim=1, keepdim=True).clamp_min(eps)


def compute_series_spectrum_torch(
    x: Any,
    apply_diff: bool = True,
    feature_type: str = "magnitude",
    normalize_spectrum: bool = True,
    channel_reduce: str = "mean",
    device: str | None = "auto",
    dtype: str = "float64",
):
    """
    Compute spectral features from panel data with torch FFT.

    Args mirror compute_series_spectrum from longitudinal_metrics.py.  Returns a
    torch tensor on the requested device with shape (N, D).
    """
    torch = _require_torch()
    torch_dtype = getattr(torch, dtype)
    resolved_device = resolve_torch_device(device)
    x_t = _to_3d_tensor(x, device=resolved_device, dtype=torch_dtype)

    if apply_diff:
        x_t = torch.diff(x_t, dim=1)
    if x_t.shape[1] < 2:
        raise ValueError("Need time length >= 3 if apply_diff=True, else >= 2.")

    fft_vals = torch.fft.rfft(x_t, dim=1)
    if feature_type == "magnitude":
        spec = torch.abs(fft_vals)
    elif feature_type == "power":
        spec = torch.abs(fft_vals).square()
    else:
        raise ValueError("feature_type must be 'magnitude' or 'power'")

    if channel_reduce == "none":
        feat = spec.reshape(spec.shape[0], -1)
    elif channel_reduce == "mean":
        feat = spec.mean(dim=2)
    elif channel_reduce == "sum":
        feat = spec.sum(dim=2)
    elif channel_reduce == "median":
        feat = spec.median(dim=2).values
    else:
        raise ValueError("channel_reduce must be one of {'none','mean','sum','median'}")

    if feat.ndim == 3:
        feat = feat.reshape(feat.shape[0], -1)
    if normalize_spectrum:
        feat = l1_normalize_rows_torch(feat)
    return feat


def make_uniform_weights_torch(n: int, *, device, dtype):
    torch = _require_torch()
    return torch.full((n,), 1.0 / n, dtype=dtype, device=device)


def pairwise_cost_torch(real_features, syn_features, cost_metric: str = "euclidean"):
    """Compute pairwise ground cost on the active torch device."""
    torch = _require_torch()
    if cost_metric == "euclidean":
        return torch.cdist(real_features, syn_features, p=2)
    if cost_metric == "sqeuclidean":
        return torch.cdist(real_features, syn_features, p=2).square()
    if cost_metric == "cityblock":
        return torch.cdist(real_features, syn_features, p=1)
    raise ValueError("cost_metric must be one of {'euclidean','sqeuclidean','cityblock'}")


def sinkhorn_transport_torch(
    cost_matrix,
    real_weights=None,
    syn_weights=None,
    reg: float = 0.05,
    max_iter: int = 500,
    tol: float = 1e-9,
):
    """
    Entropic Sinkhorn transport in the log domain.

    Returns (transport_plan, distance).  The distance is sum(gamma * C), not the
    entropy-regularized objective value, matching the reporting convention used
    by the reference code.
    """
    torch = _require_torch()
    if reg <= 0:
        raise ValueError("reg must be positive for Sinkhorn OT")

    n, m = cost_matrix.shape
    dtype = cost_matrix.dtype
    device = cost_matrix.device
    a = (
        make_uniform_weights_torch(n, device=device, dtype=dtype)
        if real_weights is None
        else torch.as_tensor(real_weights, dtype=dtype, device=device)
    )
    b = (
        make_uniform_weights_torch(m, device=device, dtype=dtype)
        if syn_weights is None
        else torch.as_tensor(syn_weights, dtype=dtype, device=device)
    )
    a = a / a.sum()
    b = b / b.sum()

    log_a = torch.log(a.clamp_min(torch.finfo(dtype).tiny))
    log_b = torch.log(b.clamp_min(torch.finfo(dtype).tiny))
    log_k = -cost_matrix / reg
    u = torch.zeros_like(a)
    v = torch.zeros_like(b)

    for _ in range(max_iter):
        prev_u = u
        u = log_a - torch.logsumexp(log_k + v.unsqueeze(0), dim=1)
        v = log_b - torch.logsumexp(log_k + u.unsqueeze(1), dim=0)
        if torch.max(torch.abs(u - prev_u)).item() < tol:
            break

    gamma = torch.exp(log_k + u.unsqueeze(1) + v.unsqueeze(0))
    distance = torch.sum(gamma * cost_matrix)
    return gamma, distance, a, b


def spectral_wasserstein_distance_gpu(
    real_features: Any,
    syn_features: Any,
    real_weights: Any | None = None,
    syn_weights: Any | None = None,
    cost_metric: str = "euclidean",
    return_transport: bool = True,
    device: str | None = "auto",
    dtype: str = "float64",
    sinkhorn_reg: float = 0.05,
    sinkhorn_max_iter: int = 500,
    sinkhorn_tol: float = 1e-9,
    return_tensors: bool = False,
) -> dict:
    """Compute approximate Wasserstein distance with torch Sinkhorn OT."""
    torch = _require_torch()
    torch_dtype = getattr(torch, dtype)
    resolved_device = resolve_torch_device(device)
    real_feat = torch.as_tensor(real_features, dtype=torch_dtype, device=resolved_device)
    syn_feat = torch.as_tensor(syn_features, dtype=torch_dtype, device=resolved_device)

    cost_matrix = pairwise_cost_torch(real_feat, syn_feat, cost_metric=cost_metric)
    gamma, distance, a, b = sinkhorn_transport_torch(
        cost_matrix,
        real_weights=real_weights,
        syn_weights=syn_weights,
        reg=sinkhorn_reg,
        max_iter=sinkhorn_max_iter,
        tol=sinkhorn_tol,
    )

    out = {
        "distance": float(distance.detach().cpu().item()),
        "cost_matrix": cost_matrix if return_tensors else cost_matrix.detach().cpu().numpy(),
        "real_weights": a if return_tensors else a.detach().cpu().numpy(),
        "syn_weights": b if return_tensors else b.detach().cpu().numpy(),
        "device": str(resolved_device),
        "ot_solver": "sinkhorn",
        "sinkhorn_reg": sinkhorn_reg,
        "sinkhorn_max_iter": sinkhorn_max_iter,
        "sinkhorn_tol": sinkhorn_tol,
    }
    if return_transport:
        out["transport_plan"] = gamma if return_tensors else gamma.detach().cpu().numpy()
    return out


def compute_spectral_wasserstein_gpu(
    real_series: Any,
    synthetic_series: Any,
    apply_diff: bool = True,
    feature_type: str = "magnitude",
    normalize_spectrum: bool = True,
    channel_reduce: str = "none",
    cost_metric: str = "euclidean",
    real_weights: Any | None = None,
    syn_weights: Any | None = None,
    return_features: bool = True,
    return_transport: bool = True,
    device: str | None = "auto",
    dtype: str = "float64",
    sinkhorn_reg: float = 0.05,
    sinkhorn_max_iter: int = 500,
    sinkhorn_tol: float = 1e-9,
    return_tensors: bool = False,
) -> dict:
    """Compute spectral Sinkhorn-Wasserstein distance from panel arrays."""
    real_feat = compute_series_spectrum_torch(
        real_series,
        apply_diff=apply_diff,
        feature_type=feature_type,
        normalize_spectrum=normalize_spectrum,
        channel_reduce=channel_reduce,
        device=device,
        dtype=dtype,
    )
    syn_feat = compute_series_spectrum_torch(
        synthetic_series,
        apply_diff=apply_diff,
        feature_type=feature_type,
        normalize_spectrum=normalize_spectrum,
        channel_reduce=channel_reduce,
        device=device,
        dtype=dtype,
    )
    result = spectral_wasserstein_distance_gpu(
        real_feat,
        syn_feat,
        real_weights=real_weights,
        syn_weights=syn_weights,
        cost_metric=cost_metric,
        return_transport=return_transport,
        device=str(real_feat.device),
        dtype=dtype,
        sinkhorn_reg=sinkhorn_reg,
        sinkhorn_max_iter=sinkhorn_max_iter,
        sinkhorn_tol=sinkhorn_tol,
        return_tensors=True,
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

    if not return_tensors:
        for key in (
            "cost_matrix",
            "real_weights",
            "syn_weights",
            "transport_plan",
            "real_spectral_features",
            "synthetic_spectral_features",
        ):
            if key in result and hasattr(result[key], "detach"):
                result[key] = result[key].detach().cpu().numpy()
    return result


def compute_decomposed_spectral_wasserstein_gpu(
    real_series: Any,
    synthetic_series: Any,
    feature_cols: list[str],
    apply_diff: bool = True,
    feature_type: str = "magnitude",
    normalize_spectrum: bool = True,
    cost_metric: str = "euclidean",
    real_weights: Any | None = None,
    syn_weights: Any | None = None,
    return_transport: bool = True,
    compute_standalone: bool = True,
    device: str | None = "auto",
    dtype: str = "float64",
    sinkhorn_reg: float = 0.05,
    sinkhorn_max_iter: int = 500,
    sinkhorn_tol: float = 1e-9,
    return_tensors: bool = False,
) -> dict:
    """Compute TT-Wasserstein with per-feature contribution scores on GPU."""
    torch = _require_torch()
    real_series_3d = _to_3d_tensor(
        real_series,
        device=resolve_torch_device(device),
        dtype=getattr(torch, dtype),
    )
    syn_series_3d = _to_3d_tensor(
        synthetic_series,
        device=real_series_3d.device,
        dtype=real_series_3d.dtype,
    )
    c_real = real_series_3d.shape[2]
    c_syn = syn_series_3d.shape[2]

    if c_real != c_syn:
        raise ValueError(f"Channel mismatch: real has {c_real}, synthetic has {c_syn}")
    if len(feature_cols) != c_real:
        raise ValueError(
            f"len(feature_cols)={len(feature_cols)} must match number of channels={c_real}"
        )

    full_result = compute_spectral_wasserstein_gpu(
        real_series_3d,
        syn_series_3d,
        apply_diff=apply_diff,
        feature_type=feature_type,
        normalize_spectrum=normalize_spectrum,
        channel_reduce="none",
        cost_metric=cost_metric,
        real_weights=real_weights,
        syn_weights=syn_weights,
        return_features=True,
        return_transport=True,
        device=str(real_series_3d.device),
        dtype=dtype,
        sinkhorn_reg=sinkhorn_reg,
        sinkhorn_max_iter=sinkhorn_max_iter,
        sinkhorn_tol=sinkhorn_tol,
        return_tensors=True,
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

    component_cost_matrices: dict[str, Any] = {}
    component_scores: dict[str, float] = {}
    standalone_component_distances: dict[str, float] = {}

    if cost_metric == "euclidean":
        squared_component_costs = {}
        safe_total_cost = torch.where(total_cost > 0, total_cost, torch.ones_like(total_cost))
        for idx, col_name in enumerate(feature_cols):
            real_col_feat = real_feat_3d[:, :, idx]
            syn_col_feat = syn_feat_3d[:, :, idx]
            diff = real_col_feat[:, None, :] - syn_col_feat[None, :, :]
            squared_component_costs[col_name] = torch.sum(diff * diff, dim=2)
            component_cost = torch.where(
                total_cost > 0,
                squared_component_costs[col_name] / safe_total_cost,
                torch.zeros_like(total_cost),
            )
            component_cost_matrices[col_name] = component_cost
            component_scores[col_name] = float(torch.sum(gamma * component_cost).detach().cpu().item())

            if compute_standalone:
                standalone_component_distances[col_name] = spectral_wasserstein_distance_gpu(
                    real_col_feat,
                    syn_col_feat,
                    real_weights=full_result["real_weights"],
                    syn_weights=full_result["syn_weights"],
                    cost_metric="euclidean",
                    return_transport=False,
                    device=str(real_col_feat.device),
                    dtype=dtype,
                    sinkhorn_reg=sinkhorn_reg,
                    sinkhorn_max_iter=sinkhorn_max_iter,
                    sinkhorn_tol=sinkhorn_tol,
                )["distance"]
    else:
        for idx, col_name in enumerate(feature_cols):
            real_col_feat = real_feat_3d[:, :, idx]
            syn_col_feat = syn_feat_3d[:, :, idx]
            component_cost = pairwise_cost_torch(real_col_feat, syn_col_feat, cost_metric=cost_metric)
            component_cost_matrices[col_name] = component_cost
            component_scores[col_name] = float(torch.sum(gamma * component_cost).detach().cpu().item())

            if compute_standalone:
                standalone_component_distances[col_name] = spectral_wasserstein_distance_gpu(
                    real_col_feat,
                    syn_col_feat,
                    real_weights=full_result["real_weights"],
                    syn_weights=full_result["syn_weights"],
                    cost_metric=cost_metric,
                    return_transport=False,
                    device=str(real_col_feat.device),
                    dtype=dtype,
                    sinkhorn_reg=sinkhorn_reg,
                    sinkhorn_max_iter=sinkhorn_max_iter,
                    sinkhorn_tol=sinkhorn_tol,
                )["distance"]

    reconstructed_total_cost = torch.zeros_like(total_cost)
    for c_col in component_cost_matrices.values():
        reconstructed_total_cost = reconstructed_total_cost + c_col
    if not torch.allclose(reconstructed_total_cost, total_cost, atol=1e-8, rtol=1e-6):
        raise RuntimeError("Component costs do not reconstruct the joint cost matrix")

    total_component_score = sum(component_scores.values())
    if total_component_score > 0:
        relative_contribution = {
            key: value / total_component_score for key, value in component_scores.items()
        }
    else:
        relative_contribution = {key: 0.0 for key in component_scores}

    component_rank = sorted(component_scores.items(), key=lambda item: item[1], reverse=True)

    raw_result = dict(full_result)
    raw_result["component_cost_matrices"] = component_cost_matrices
    raw_result["feature_cols"] = list(feature_cols)
    raw_result["decomposition_metric"] = cost_metric
    if not return_transport:
        raw_result.pop("transport_plan", None)

    if not return_tensors:
        for key, value in list(raw_result.items()):
            if hasattr(value, "detach"):
                raw_result[key] = value.detach().cpu().numpy()
        raw_result["component_cost_matrices"] = {
            key: value.detach().cpu().numpy() if hasattr(value, "detach") else value
            for key, value in component_cost_matrices.items()
        }

    return {
        "total_distance": full_result["distance"],
        "component_scores": component_scores,
        "standalone_component_distances": standalone_component_distances,
        "relative_contribution": relative_contribution,
        "component_rank": component_rank,
        "raw_result": raw_result,
    }


def tt_wasserstein_distance_gpu(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    key_col: str,
    time_col: str,
    num_cols: list,
    n_fft_components: int = 10,
    zscore_reference: str = "real",
    apply_diff: bool = True,
    feature_type: str = "magnitude",
    normalize_spectrum: bool = True,
    cost_metric: str = "euclidean",
    fill_method: str = "edge_nearest",
    compute_standalone: bool = True,
    device: str | None = "auto",
    dtype: str = "float64",
    sinkhorn_reg: float = 0.05,
    sinkhorn_max_iter: int = 500,
    sinkhorn_tol: float = 1e-9,
) -> tuple:
    """
    GPU-friendly TT-Wasserstein wrapper for DataFrame inputs.

    n_fft_components is kept for API compatibility with the reference wrapper.
    """
    del n_fft_components
    feature_cols = list(num_cols)

    if len(feature_cols) == 0:
        return np.nan, {}, {}, {}

    print("    [GPU] Converting DataFrames to 3D panels...")
    try:
        real_panel, real_meta = transform_to_3d_array(
            real_df,
            id_col=key_col,
            time_col=time_col,
            feature_cols=feature_cols,
            fill_method=fill_method,
            return_metadata=True,
        )
        syn_panel, _ = transform_to_3d_array(
            synth_df,
            id_col=key_col,
            time_col=time_col,
            feature_cols=feature_cols,
            time_index=real_meta["time_index"],
            fill_method=fill_method,
            return_metadata=True,
        )
    except Exception as exc:
        print(f"    WARNING: Failed to convert to 3D panel: {exc}")
        return np.nan, {}, {}, {}

    print(f"    [GPU] Real panel shape: {real_panel.shape}, Synth panel shape: {syn_panel.shape}")
    if apply_diff and real_panel.shape[1] < 3:
        print(f"    WARNING: Time length {real_panel.shape[1]} too short for differencing (need >= 3)")
        return np.nan, {}, {}, {}

    print(f"    [GPU] Applying z-score normalization (reference={zscore_reference})...")
    real_scaled, syn_scaled, _, _ = standardize_panels(
        real_panel,
        syn_panel,
        reference=zscore_reference,
    )

    resolved_device = resolve_torch_device(device)
    print(
        "    [GPU] Computing decomposed spectral Sinkhorn-Wasserstein "
        f"(device={resolved_device}, reg={sinkhorn_reg})..."
    )
    try:
        result = compute_decomposed_spectral_wasserstein_gpu(
            real_scaled,
            syn_scaled,
            feature_cols=feature_cols,
            apply_diff=apply_diff,
            feature_type=feature_type,
            normalize_spectrum=normalize_spectrum,
            cost_metric=cost_metric,
            compute_standalone=compute_standalone,
            device=str(resolved_device),
            dtype=dtype,
            sinkhorn_reg=sinkhorn_reg,
            sinkhorn_max_iter=sinkhorn_max_iter,
            sinkhorn_tol=sinkhorn_tol,
        )
    except ImportError as exc:
        print(f"    ERROR: {exc}")
        return np.nan, {}, {}, {}
    except Exception as exc:
        print(f"    WARNING: Failed to compute GPU spectral Wasserstein: {exc}")
        return np.nan, {}, {}, {}

    return (
        result["total_distance"],
        result["component_scores"],
        result["relative_contribution"],
        result["standalone_component_distances"],
    )


def compute_all_longitudinal_metrics_gpu(
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
    device: str | None = "auto",
    dtype: str = "float64",
    sinkhorn_reg: float = 0.05,
    sinkhorn_max_iter: int = 500,
    sinkhorn_tol: float = 1e-9,
) -> dict:
    """Compute longitudinal metrics while using the GPU TT-Wasserstein path."""
    results = {
        "summary": {},
        "per_feature": {},
        "metadata": {
            "flow_independence": flow_independence,
            "max_lag": max_lag,
            "n_fft_components": n_fft_components,
            "zscore_reference": zscore_reference,
            "fill_method": fill_method,
            "compute_tt_standalone": compute_tt_standalone,
            "tt_wasserstein_solver": "torch_sinkhorn",
            "tt_wasserstein_device": str(resolve_torch_device(device)),
            "tt_wasserstein_sinkhorn_reg": sinkhorn_reg,
            "tt_wasserstein_sinkhorn_max_iter": sinkhorn_max_iter,
            "tt_wasserstein_sinkhorn_tol": sinkhorn_tol,
            "n_real_flows": int(real_df[key_col].nunique()),
            "n_synth_flows": int(synth_df[key_col].nunique()),
            "n_real_rows": len(real_df),
            "n_synth_rows": len(synth_df),
        },
    }

    print("\n[COMPUTING] FirstDifference KSComplement...")
    fd_scores = {}
    for col in num_cols:
        print(f"  Processing: {col}")
        fd_scores[col] = first_difference_ks_complement(
            real_df,
            synth_df,
            key_col,
            time_col,
            col,
        )
    print("  Done.")

    print("\n[COMPUTING] Transition Matrix TVDComplement...")
    tm_scores = {}
    for col in cat_cols:
        print(f"  Processing: {col}")
        tm_scores[col] = transition_matrix_tvd_complement(
            real_df,
            synth_df,
            key_col,
            time_col,
            col,
        )
    print("  Done.")

    print(f"\n[COMPUTING] AutoCorrelation Similarity (Mode: {flow_independence})...")
    ac_scores = {}
    for col in num_cols:
        print(f"  Processing: {col}")
        ac_scores[col] = autocorrelation_similarity(
            real_df,
            synth_df,
            key_col,
            time_col,
            col,
            flow_independence,
            max_lag,
        )
    print("  Done.")

    print("\n[COMPUTING] TT-Wasserstein Distance (GPU Sinkhorn)...")
    tt_distance = np.nan
    tt_component = {}
    tt_relative = {}
    tt_standalone = {}
    if not skip_tt_wasserstein and len(num_cols) >= 1:
        tt_distance, tt_component, tt_relative, tt_standalone = tt_wasserstein_distance_gpu(
            real_df,
            synth_df,
            key_col,
            time_col,
            num_cols,
            n_fft_components=n_fft_components,
            zscore_reference=zscore_reference,
            fill_method=fill_method,
            compute_standalone=compute_tt_standalone,
            device=device,
            dtype=dtype,
            sinkhorn_reg=sinkhorn_reg,
            sinkhorn_max_iter=sinkhorn_max_iter,
            sinkhorn_tol=sinkhorn_tol,
        )
    print("  Done.")

    def safe_mean(d: dict) -> float:
        vals = [v for v in d.values() if not (isinstance(v, float) and np.isnan(v))]
        return float(np.mean(vals)) if vals else np.nan

    results["summary"] = {
        "FirstDifferenceKSComplement": safe_mean(fd_scores),
        "TransitionMatrixTVDComplement": safe_mean(tm_scores),
        "AutoCorrelationSimilarity": safe_mean(ac_scores),
        "TTWassersteinDistance": tt_distance,
    }

    results["per_feature"] = {
        "FirstDifferenceKSComplement": fd_scores,
        "TransitionMatrixTVDComplement": tm_scores,
        "AutoCorrelationSimilarity": ac_scores,
        "TTWasserstein_ComponentScores": tt_component,
        "TTWasserstein_RelativeContribution": tt_relative,
        "TTWasserstein_StandaloneDistances": tt_standalone,
    }
    return results
