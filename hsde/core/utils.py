# ============================================================================
# utils.py - Geometry, Normalization, and Preprocessing Utilities
# ============================================================================
"""
Utility functions combining:
- Lorentz (hyperbolic) geometry from HSDE
- scATAC-seq TF-IDF normalization from HSDE
- Highly variable peak selection from HSDE
- Euclidean distance baseline
"""

import numpy as np
from scipy.sparse import issparse, csr_matrix
from typing import Literal
import torch

EPS = 1e-8
MAX_NORM = 15.0


# ============================================================================
# Lorentz (Hyperbolic) Geometry
# ============================================================================

def lorentzian_product(
    x: torch.Tensor,
    y: torch.Tensor,
    keepdim: bool = False,
    use_double: bool = True,
) -> torch.Tensor:
    """
    Compute Lorentzian inner product: <x, y> = -x₀y₀ + Σᵢ≥₁ xᵢyᵢ.

    Parameters
    ----------
    use_double : bool
        Cast to float64 for numerical stability (recommended for hyperbolic ops).
    """
    orig_dtype = x.dtype
    if use_double and x.dtype != torch.float64:
        x = x.double()
        y = y.double()

    res = -x[..., 0] * y[..., 0] + torch.sum(x[..., 1:] * y[..., 1:], dim=-1)
    res = torch.clamp(res, min=-1e10, max=1e10)

    if use_double and orig_dtype != torch.float64:
        res = res.to(orig_dtype)

    return res.unsqueeze(-1) if keepdim else res


def lorentz_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = EPS,
    use_double: bool = True,
) -> torch.Tensor:
    """
    Hyperbolic distance on Lorentz manifold: d(x, y) = acosh(-<x, y>).
    """
    orig_dtype = x.dtype
    if use_double and x.dtype != torch.float64:
        x = x.double()
        y = y.double()

    xy_inner = lorentzian_product(x, y, use_double=False)
    clamped = torch.clamp(-xy_inner, min=1.0 + eps, max=1e10)

    if torch.isnan(clamped).any() or torch.isinf(clamped).any():
        return torch.zeros_like(clamped).mean()

    dist = torch.where(
        clamped > 1e4,
        torch.log(2 * clamped),
        torch.acosh(clamped),
    )

    if use_double and orig_dtype != torch.float64:
        dist = dist.to(orig_dtype)

    return dist


def exp_map_at_origin(
    v_tangent: torch.Tensor,
    eps: float = EPS,
    use_double: bool = True,
) -> torch.Tensor:
    """
    Exponential map from tangent space at origin to hyperboloid.
    exp₀(v) = (cosh(‖v‖), sinh(‖v‖) · v/‖v‖)
    """
    orig_dtype = v_tangent.dtype
    if use_double and v_tangent.dtype != torch.float64:
        v_tangent = v_tangent.double()

    v_spatial = v_tangent[..., 1:]
    v_norm = torch.clamp(
        torch.norm(v_spatial, p=2, dim=-1, keepdim=True), max=MAX_NORM
    )

    is_zero = v_norm < eps
    v_unit = torch.where(is_zero, torch.zeros_like(v_spatial), v_spatial / (v_norm + eps))

    x_coord = torch.cosh(v_norm)
    y_coords = torch.sinh(v_norm) * v_unit
    result = torch.cat([x_coord, y_coords], dim=-1)

    if torch.isnan(result).any() or torch.isinf(result).any():
        safe_point = torch.zeros_like(result)
        safe_point[..., 0] = 1.0
        result = safe_point

    if use_double and orig_dtype != torch.float64:
        result = result.to(orig_dtype)

    return result


def euclidean_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Euclidean (L2) distance baseline."""
    return torch.norm(x - y, p=2, dim=-1)


# ============================================================================
# TF-IDF Normalization (scATAC-seq)
# ============================================================================

def tfidf_normalization(
    adata,
    scale_factor: float = 1e4,
    log_tf: bool = False,
    log_idf: bool = True,
    inplace: bool = True,
):
    """
    TF-IDF normalization following Signac/SnapATAC2 best practices.
    """
    if not inplace:
        adata = adata.copy()

    print(f"Applying TF-IDF normalization (scale={scale_factor:.0e})...")

    if issparse(adata.X):
        if adata.X.format != "csr":
            adata.X = adata.X.tocsr()
        X = adata.X
    else:
        X = csr_matrix(adata.X)
        adata.X = X

    if not np.issubdtype(X.dtype, np.floating):
        X = X.astype(np.float32)
        adata.X = X
    elif X.dtype != np.float32:
        X = X.astype(np.float32, copy=False)
        adata.X = X

    cell_sums = np.array(X.sum(axis=1)).flatten()
    cell_sums[cell_sums == 0] = 1
    row_indices = np.repeat(np.arange(X.shape[0]), np.diff(X.indptr))
    X.data /= cell_sums[row_indices]

    if log_tf:
        np.log1p(X.data, out=X.data)

    n_cells = adata.n_obs
    n_cells_per_peak = np.asarray((X > 0).sum(axis=0)).ravel()
    n_cells_per_peak[n_cells_per_peak == 0] = 1

    if log_idf:
        idf = np.log1p(n_cells / n_cells_per_peak)
    else:
        idf = n_cells / n_cells_per_peak

    X.data *= idf[X.indices]
    X.data *= scale_factor

    adata.uns["tfidf_params"] = {
        "scale_factor": scale_factor,
        "log_tf": log_tf,
        "log_idf": log_idf,
    }

    print(f"  TF-IDF complete. Value range: [{X.data.min():.2e}, {X.data.max():.2e}]")
    return adata if not inplace else None


# ============================================================================
# Highly Variable Peak Selection (scATAC-seq)
# ============================================================================

def select_highly_variable_peaks(
    adata,
    n_top_peaks: int = 20000,
    min_accessibility: float = 0.01,
    max_accessibility: float = 0.95,
    method: Literal["signac", "snapatac2", "deviance"] = "signac",
    use_raw_counts: bool = True,
    inplace: bool = True,
):
    """
    Select highly variable peaks using variance / VMR / deviance methods.
    """
    if not inplace:
        adata = adata.copy()

    if use_raw_counts and "counts" in adata.layers:
        X = adata.layers["counts"]
    else:
        X = adata.X

    if issparse(X):
        n_cells_per_peak = np.array((X > 0).sum(axis=0)).flatten()
    else:
        n_cells_per_peak = np.sum(X > 0, axis=0)

    accessibility = n_cells_per_peak / adata.n_obs
    adata.var["accessibility"] = accessibility

    accessibility_mask = (accessibility >= min_accessibility) & (accessibility <= max_accessibility)

    if method == "signac":
        X_norm = adata.X
        if issparse(X_norm):
            mean = np.array(X_norm.mean(axis=0)).flatten()
            mean_sq = np.array(X_norm.power(2).mean(axis=0)).flatten()
            variance = mean_sq - mean ** 2
        else:
            variance = np.var(X_norm, axis=0)
        variance[~accessibility_mask] = -np.inf
        adata.var["variance"] = variance
        score = variance

    elif method == "snapatac2":
        X_norm = adata.X
        if issparse(X_norm):
            mean = np.array(X_norm.mean(axis=0)).flatten()
            mean_sq = np.array(X_norm.power(2).mean(axis=0)).flatten()
            variance = mean_sq - mean ** 2
        else:
            mean = np.mean(X_norm, axis=0)
            variance = np.var(X_norm, axis=0)
        vmr = np.zeros_like(variance)
        valid_mask = mean > 0
        vmr[valid_mask] = variance[valid_mask] / mean[valid_mask]
        score = vmr * accessibility
        score[~accessibility_mask] = -np.inf
        adata.var["vmr_weighted"] = score

    elif method == "deviance":
        if use_raw_counts and "counts" in adata.layers:
            X_counts = adata.layers["counts"]
        else:
            X_counts = adata.X
        if issparse(X_counts):
            X_binary = (X_counts > 0).astype(float).toarray()
        else:
            X_binary = (X_counts > 0).astype(float)
        p = np.clip(accessibility.copy(), 1e-10, 1 - 1e-10)
        deviance = -2 * (
            X_binary * np.log(p)[None, :] + (1 - X_binary) * np.log(1 - p)[None, :]
        ).sum(axis=0)
        deviance[~accessibility_mask] = -np.inf
        adata.var["deviance"] = deviance
        score = deviance
    else:
        raise ValueError(f"Unknown method: {method}")

    n_top_peaks = min(n_top_peaks, accessibility_mask.sum())
    top_idx = np.argsort(-score)[:n_top_peaks]

    adata.var["highly_variable"] = False
    adata.var.loc[adata.var.index[top_idx], "highly_variable"] = True

    print(f"Selected {n_top_peaks:,} highly variable peaks")
    return adata if not inplace else None
