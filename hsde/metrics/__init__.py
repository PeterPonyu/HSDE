"""
HSDE Metrics Module
===================
Self-contained metric computation for HSDE benchmarking.
Internalized from MoCoO canonical implementations — no external dependencies.

Primary API: compute_all_metrics(latent, labels, dre_k=15) -> dict

Metric categories:
  1. Clustering: NMI, ARI, ASW, DAV, CAL, COR
  2. DRE (co-ranking): distance_correlation, Q_local, Q_global, overall_quality
  3. LSE (intrinsic): manifold_dimensionality, spectral_decay_rate, participation_ratio,
     anisotropy_score, noise_resilience, core_quality, overall_quality
  4. DREX (extended DR): trustworthiness, continuity, distance_spearman/pearson,
     local_scale_quality, neighborhood_symmetry, overall_quality
  5. LSEX (extended latent): two_hop_connectivity, radial_concentration,
     local_curvature, entropy_stability, overall_quality
  6. Latent diagnostics: norm, std, variance, pairwise distance stats
"""

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr, pearsonr
from sklearn.cluster import KMeans
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.neighbors import NearestNeighbors
import warnings

from .dre import DimensionalityReductionEvaluator
from .lse import SingleCellLatentSpaceEvaluator

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
# Clustering metrics
# ═══════════════════════════════════════════════════════════════════════════════

def _clustering_metrics(latent, labels):
    """NMI, ARI, ASW, DAV, CAL, COR."""
    latent = np.asarray(latent, dtype=float)
    n_clusters = len(np.unique(labels))
    pred = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(latent)

    m = {
        'NMI': normalized_mutual_info_score(labels, pred),
        'ARI': adjusted_rand_score(labels, pred),
    }
    try:
        m['ASW'] = silhouette_score(latent, pred) if len(np.unique(pred)) > 1 else np.nan
    except Exception:
        m['ASW'] = np.nan
    try:
        m['DAV'] = davies_bouldin_score(latent, pred)
    except Exception:
        m['DAV'] = np.nan
    try:
        m['CAL'] = calinski_harabasz_score(latent, pred)
    except Exception:
        m['CAL'] = np.nan
    try:
        acorr = np.abs(np.corrcoef(latent.T))
        m['COR'] = float(acorr.sum(axis=1).mean() - 1)
    except Exception:
        m['COR'] = np.nan
    return m


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _knn_indices(X, k):
    X = np.asarray(X, dtype=float)
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm='auto')
    nn.fit(X)
    return nn.kneighbors(X, return_distance=False)[:, 1:]


def _q_local(knn_source, knn_target, k):
    n = knn_source.shape[0]
    overlap = 0.0
    for i in range(n):
        s = set(knn_source[i, :k])
        t = set(knn_target[i, :k])
        overlap += len(s & t) / k
    return overlap / n


# ═══════════════════════════════════════════════════════════════════════════════
# DRE — uses internal co-ranking evaluator
# ═══════════════════════════════════════════════════════════════════════════════

def _dre_metrics(latent, projection_2d, k=15, prefix="DRE_umap"):
    latent = np.asarray(latent, dtype=float)
    projection_2d = np.asarray(projection_2d, dtype=float)
    m = {}
    try:
        dre = DimensionalityReductionEvaluator(verbose=False)
        results = dre.comprehensive_evaluation(latent, projection_2d, k=k)
        m[f'{prefix}_distance_correlation'] = results['distance_correlation']
        m[f'{prefix}_Q_local'] = results['Q_local']
        m[f'{prefix}_Q_global'] = results['Q_global']
        m[f'{prefix}_overall_quality'] = results['overall_quality']
    except Exception:
        for key in ('distance_correlation', 'Q_local', 'Q_global', 'overall_quality'):
            m[f'{prefix}_{key}'] = np.nan
    return m


# ═══════════════════════════════════════════════════════════════════════════════
# LSE — uses internal evaluator
# ═══════════════════════════════════════════════════════════════════════════════

def _lse_metrics(latent):
    latent = np.asarray(latent, dtype=float)
    m = {}
    try:
        lse = SingleCellLatentSpaceEvaluator(data_type="trajectory", verbose=False)
        results = lse.comprehensive_evaluation(latent)
        m['LSE_manifold_dimensionality'] = results['manifold_dimensionality']
        m['LSE_spectral_decay_rate'] = results['spectral_decay_rate']
        m['LSE_participation_ratio'] = results['participation_ratio']
        m['LSE_anisotropy_score'] = results['anisotropy_score']
        m['LSE_noise_resilience'] = results['noise_resilience']
        m['LSE_core_quality'] = results['core_quality']
        m['LSE_overall_quality'] = results['overall_quality']
    except Exception:
        for key in ('manifold_dimensionality', 'spectral_decay_rate',
                    'participation_ratio', 'anisotropy_score',
                    'noise_resilience', 'core_quality', 'overall_quality'):
            m[f'LSE_{key}'] = np.nan
    return m


# ═══════════════════════════════════════════════════════════════════════════════
# DREX — Extended Dimensionality Reduction metrics
# ═══════════════════════════════════════════════════════════════════════════════

def _trustworthiness(X_high, X_low, k=15):
    from sklearn.manifold import trustworthiness as _tw
    return _tw(np.asarray(X_high, dtype=float), np.asarray(X_low, dtype=float), n_neighbors=k)


def _continuity(X_high, X_low, k=15):
    X_high = np.asarray(X_high, dtype=float)
    X_low = np.asarray(X_low, dtype=float)
    n = X_high.shape[0]
    nn_high = _knn_indices(X_high, k)
    nn_low = _knn_indices(X_low, k)
    cont = 0.0
    for i in range(n):
        low_set = set(nn_low[i])
        for j in nn_high[i]:
            if j not in low_set:
                cont += 1
    max_cont = n * k * (2 * n - 3 * k - 1)
    return 1.0 - (2.0 / max_cont) * cont if max_cont != 0 else 1.0


def _drex_metrics(latent, projection_2d, k=15):
    latent = np.asarray(latent, dtype=float)
    projection_2d = np.asarray(projection_2d, dtype=float)
    m = {}
    try:
        m['DREX_trustworthiness'] = _trustworthiness(latent, projection_2d, k)
        m['DREX_continuity'] = _continuity(latent, projection_2d, k)

        n = min(latent.shape[0], 2000)
        if latent.shape[0] > n:
            idx = np.random.RandomState(42).choice(latent.shape[0], n, replace=False)
            d_h, d_l = pdist(latent[idx]), pdist(projection_2d[idx])
        else:
            d_h, d_l = pdist(latent), pdist(projection_2d)

        m['DREX_distance_spearman'] = float(spearmanr(d_h, d_l).correlation)
        m['DREX_distance_pearson'] = float(pearsonr(d_h, d_l)[0])

        nn_h = NearestNeighbors(n_neighbors=k + 1).fit(latent)
        dists_h = nn_h.kneighbors(latent, return_distance=True)[0][:, 1:]
        nn_l = NearestNeighbors(n_neighbors=k + 1).fit(projection_2d)
        dists_l = nn_l.kneighbors(projection_2d, return_distance=True)[0][:, 1:]
        m['DREX_local_scale_quality'] = float(spearmanr(dists_h.mean(axis=1), dists_l.mean(axis=1)).correlation)

        knn_h = _knn_indices(latent, k)
        knn_l = _knn_indices(projection_2d, k)
        sym = sum(len(set(knn_h[i]) & set(knn_l[i])) / k for i in range(latent.shape[0]))
        m['DREX_neighborhood_symmetry'] = sym / latent.shape[0]

        m['DREX_overall_quality'] = np.mean([
            m['DREX_trustworthiness'], m['DREX_continuity'],
            max(0, m['DREX_distance_spearman']), max(0, m['DREX_distance_pearson']),
            max(0, m['DREX_local_scale_quality']), m['DREX_neighborhood_symmetry'],
        ])
    except Exception:
        for key in ('trustworthiness', 'continuity', 'distance_spearman',
                    'distance_pearson', 'local_scale_quality',
                    'neighborhood_symmetry', 'overall_quality'):
            m[f'DREX_{key}'] = np.nan
    return m


# ═══════════════════════════════════════════════════════════════════════════════
# LSEX — Extended Latent Space metrics
# ═══════════════════════════════════════════════════════════════════════════════

def _lsex_metrics(latent, k=15):
    latent = np.asarray(latent, dtype=float)
    m = {}
    try:
        n = latent.shape[0]
        knn = _knn_indices(latent, k)

        # Two-hop connectivity
        two_hop_unique = 0.0
        for i in range(n):
            one_hop = set(knn[i])
            two_hop = set()
            for j in knn[i]:
                two_hop.update(knn[j])
            two_hop -= one_hop
            two_hop.discard(i)
            two_hop_unique += len(two_hop) / max(1, k * k)
        m['LSEX_two_hop_connectivity'] = two_hop_unique / n

        # Radial concentration
        dists = NearestNeighbors(n_neighbors=k + 1).fit(latent).kneighbors(
            latent, return_distance=True)[0][:, 1:]
        cv = dists.std(axis=1) / (dists.mean(axis=1) + 1e-10)
        m['LSEX_radial_concentration'] = 1.0 - float(cv.mean())

        # Local curvature
        curvature = 0.0
        for i in range(min(n, 2000)):
            nbrs = latent[knn[i]]
            residuals = nbrs - nbrs.mean(axis=0)
            _, s, _ = np.linalg.svd(residuals, full_matrices=False)
            curvature += s[0] / (s.sum() + 1e-10)
        m['LSEX_local_curvature'] = curvature / min(n, 2000)

        # Entropy stability
        q_k = _q_local(knn, knn, k)
        knn_half = _knn_indices(latent, max(k // 2, 3))
        q_half = _q_local(knn_half, _knn_indices(latent, max(k // 2, 3)), max(k // 2, 3))
        m['LSEX_entropy_stability'] = float(np.mean([q_k, q_half]))

        m['LSEX_overall_quality'] = np.mean([
            m['LSEX_two_hop_connectivity'],
            max(0, m['LSEX_radial_concentration']),
            m['LSEX_local_curvature'],
            m['LSEX_entropy_stability'],
        ])
    except Exception:
        for key in ('two_hop_connectivity', 'radial_concentration',
                    'local_curvature', 'entropy_stability', 'overall_quality'):
            m[f'LSEX_{key}'] = np.nan
    return m


# ═══════════════════════════════════════════════════════════════════════════════
# Latent Diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_latent_diagnostics(latent, max_samples=2000):
    z = np.asarray(latent, dtype=float)
    std = z.std(axis=0)
    var = z.var(axis=0)
    n = z.shape[0]
    z_sub = z[np.random.choice(n, min(n, max_samples), replace=False)] if n > max_samples else z
    try:
        dists = pdist(z_sub)
        dist_mean, dist_std = float(np.mean(dists)), float(np.std(dists))
    except Exception:
        dist_mean = dist_std = np.nan
    return {
        'diag_mean_norm': float(np.linalg.norm(z.mean(axis=0))),
        'diag_std_mean': float(std.mean()),
        'diag_std_min': float(std.min()),
        'diag_std_max': float(std.max()),
        'diag_var_mean': float(var.mean()),
        'diag_near_zero_dims': int((std < 1e-3).sum()),
        'diag_pairwise_dist_mean': dist_mean,
        'diag_pairwise_dist_std': dist_std,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 2D projections helper
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_2d_projections(latent):
    latent = np.asarray(latent, dtype=float)
    import scanpy as sc
    umap_2d = tsne_2d = None
    try:
        adata = sc.AnnData(latent.astype(np.float32))
        sc.pp.neighbors(adata, use_rep='X', n_neighbors=15)
        sc.tl.umap(adata)
        umap_2d = adata.obsm['X_umap']
        sc.tl.tsne(adata, use_rep='X')
        tsne_2d = adata.obsm['X_tsne']
    except Exception as e:
        print(f"  Warning: 2D projection failed: {e}")
    return umap_2d, tsne_2d


# ═══════════════════════════════════════════════════════════════════════════════
# Primary API
# ═══════════════════════════════════════════════════════════════════════════════

def compute_all_metrics(latent, labels, dre_k=15):
    """Compute the full metric battery.

    Parameters
    ----------
    latent : np.ndarray (n_cells, latent_dim)
    labels : array-like (n_cells,)
    dre_k : int
        Number of neighbors for DRE/DREX evaluations.

    Returns
    -------
    dict
        All metric values (NaN for any that fail).
    """
    latent = np.asarray(latent, dtype=float)
    labels = np.asarray(labels, dtype=int)
    metrics = {}

    # 1. Clustering
    metrics.update(_clustering_metrics(latent, labels))

    # 2. 2D projections
    umap_2d, tsne_2d = _compute_2d_projections(latent)

    # 3. DRE (UMAP)
    if umap_2d is not None:
        metrics.update(_dre_metrics(latent, umap_2d, dre_k, "DRE_umap"))
    else:
        for k in ('distance_correlation', 'Q_local', 'Q_global', 'overall_quality'):
            metrics[f'DRE_umap_{k}'] = np.nan

    # 4. DRE (tSNE)
    if tsne_2d is not None:
        metrics.update(_dre_metrics(latent, tsne_2d, dre_k, "DRE_tsne"))
    else:
        for k in ('distance_correlation', 'Q_local', 'Q_global', 'overall_quality'):
            metrics[f'DRE_tsne_{k}'] = np.nan

    # 5. LSE
    metrics.update(_lse_metrics(latent))

    # 6. DREX (using UMAP)
    if umap_2d is not None:
        metrics.update(_drex_metrics(latent, umap_2d, dre_k))
    else:
        for k in ('trustworthiness', 'continuity', 'distance_spearman',
                   'distance_pearson', 'local_scale_quality',
                   'neighborhood_symmetry', 'overall_quality'):
            metrics[f'DREX_{k}'] = np.nan

    # 7. LSEX
    metrics.update(_lsex_metrics(latent, dre_k))

    # 8. Latent diagnostics
    metrics.update(compute_latent_diagnostics(latent))

    # 9. Store projections for visualization
    metrics['_umap_2d'] = umap_2d
    metrics['_tsne_2d'] = tsne_2d

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Metric display metadata (for plotting)
# ═══════════════════════════════════════════════════════════════════════════════

CORE_METRICS = [
    ("NMI",                        "NMI ↑",            True),
    ("ARI",                        "ARI ↑",            True),
    ("ASW",                        "ASW ↑",            True),
    ("DAV",                        "DAV ↓",            False),
    ("DRE_umap_overall_quality",   "DRE UMAP ↑",      True),
    ("LSE_overall_quality",        "LSE Overall ↑",    True),
]

EXT_METRICS_CLUSTERING = [
    ("COR",  "Corr ↑",     True),
    ("CAL",  "Cal-H ↑",    True),
]

EXT_METRICS_DRE = [
    ("DRE_umap_distance_correlation", "DRE UMAP DistCorr ↑", True),
    ("DRE_umap_Q_local",             "DRE UMAP Qloc ↑",     True),
    ("DRE_umap_Q_global",            "DRE UMAP Qglob ↑",    True),
    ("DRE_tsne_distance_correlation", "DRE tSNE DistCorr ↑", True),
    ("DRE_tsne_Q_local",             "DRE tSNE Qloc ↑",     True),
    ("DRE_tsne_Q_global",            "DRE tSNE Qglob ↑",    True),
    ("DRE_tsne_overall_quality",      "DRE tSNE Overall ↑",  True),
]

EXT_METRICS_LSE = [
    ("LSE_manifold_dimensionality", "LSE ManDim ↑",   True),
    ("LSE_spectral_decay_rate",     "LSE SpDecay ↑",  True),
    ("LSE_participation_ratio",     "LSE PartRat ↑",  True),
    ("LSE_anisotropy_score",        "LSE Aniso ↓",    False),
    ("LSE_noise_resilience",        "LSE NoiseR ↑",   True),
    ("LSE_core_quality",            "LSE Core ↑",     True),
]

EXT_METRICS_DREX = [
    ("DREX_trustworthiness",      "DREX Trust ↑",    True),
    ("DREX_continuity",           "DREX Cont ↑",     True),
    ("DREX_distance_spearman",    "DREX Spear ↑",    True),
    ("DREX_distance_pearson",     "DREX Pearson ↑",  True),
    ("DREX_local_scale_quality",  "DREX LocScale ↑", True),
    ("DREX_neighborhood_symmetry","DREX NbrSym ↑",   True),
    ("DREX_overall_quality",      "DREX Overall ↑",  True),
]

EXT_METRICS_LSEX = [
    ("LSEX_two_hop_connectivity",  "LSEX 2Hop ↑",     True),
    ("LSEX_radial_concentration",  "LSEX RadConc ↑",   True),
    ("LSEX_local_curvature",       "LSEX LocCurv ↑",   True),
    ("LSEX_entropy_stability",     "LSEX Entropy ↑",   True),
    ("LSEX_overall_quality",       "LSEX Overall ↑",   True),
]

ALL_METRIC_GROUPS = [
    ("Core Clustering", CORE_METRICS),
    ("Extended Clustering", EXT_METRICS_CLUSTERING),
    ("DR Quality (DRE)", EXT_METRICS_DRE),
    ("Latent Structure (LSE)", EXT_METRICS_LSE),
    ("Extended DR (DREX)", EXT_METRICS_DREX),
    ("Extended Latent (LSEX)", EXT_METRICS_LSEX),
]
