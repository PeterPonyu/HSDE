# ============================================================================
# mixin.py - Shared Mixins (HSDE + CCVGAE)
# ============================================================================
"""
Merged mixins from both HSDE and CCVGAE:
- scviMixin: Count-based likelihoods (NB, ZINB, Poisson, ZIP)
- betatcMixin: β-TC-VAE total correlation
- infoMixin: InfoVAE with MMD
- dipMixin: DIP-VAE covariance regularization
- adjMixin: Graph adjacency builder (from CCVGAE)
- envMixin: Clustering evaluation metrics (merged from both)
- SDEMixin: Neural SDE solver
- scMixin: Scanpy-based preprocessing (from CCVGAE)
"""

import torch
import torch.nn as nn
import torchsde
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from scipy.sparse import issparse
from typing import Optional
from anndata import AnnData


# ============================================================================
# Count-Based Likelihoods
# ============================================================================

class scviMixin:
    """NB, ZINB, Poisson, ZIP log-likelihoods + KL divergence."""

    @staticmethod
    def _normal_kl(mu1, lv1, mu2, lv2):
        v1, v2 = torch.exp(lv1), torch.exp(lv2)
        return (lv2 - lv1) / 2.0 + (v1 + (mu1 - mu2) ** 2) / (2.0 * v2) - 0.5

    @staticmethod
    def _log_nb(x, mu, theta, eps=1e-8):
        log_theta_mu = torch.log(theta + mu + eps)
        return (
            theta * (torch.log(theta + eps) - log_theta_mu)
            + x * (torch.log(mu + eps) - log_theta_mu)
            + torch.lgamma(x + theta)
            - torch.lgamma(theta + eps)
            - torch.lgamma(x + 1)
        )

    def _log_zinb(self, x, mu, theta, pi, eps=1e-8):
        pi = torch.sigmoid(pi)
        log_nb = self._log_nb(x, mu, theta, eps)
        case_zero = torch.log(pi + (1 - pi) * torch.exp(log_nb) + eps)
        case_nonzero = torch.log(1 - pi + eps) + log_nb
        return torch.where(x < eps, case_zero, case_nonzero)

    @staticmethod
    def _log_poisson(x, mu, eps=1e-8):
        return x * torch.log(mu + eps) - mu - torch.lgamma(x + 1)

    def _log_zip(self, x, mu, pi, eps=1e-8):
        pi = torch.sigmoid(pi)
        case_zero = torch.log(pi + (1 - pi) * torch.exp(-mu) + eps)
        case_nonzero = torch.log(1 - pi + eps) + self._log_poisson(x, mu, eps)
        return torch.where(x < eps, case_zero, case_nonzero)


# ============================================================================
# Disentanglement Regularizers
# ============================================================================

class betatcMixin:
    """β-TC-VAE total correlation penalty."""

    @staticmethod
    def _betatc_compute_gaussian_log_density(samples, mean, log_var):
        inv_sigma = torch.exp(-log_var)
        tmp = samples - mean
        return -0.5 * (tmp ** 2 * inv_sigma + log_var + np.log(2 * np.pi))

    def _betatc_compute_total_correlation(self, z_sampled, z_mean, z_logvar):
        log_qz_prob = self._betatc_compute_gaussian_log_density(
            z_sampled.unsqueeze(1), z_mean.unsqueeze(0), z_logvar.unsqueeze(0)
        )
        log_qz = log_qz_prob.sum(dim=2).exp().sum(dim=1).log()
        log_qz_product = log_qz_prob.exp().sum(dim=1).log().sum(dim=1)
        return (log_qz - log_qz_product).mean()


class infoMixin:
    """InfoVAE with MMD using RBF kernel."""

    def _compute_mmd(self, z_posterior, z_prior):
        kqq = self._compute_kernel_mean(self._compute_kernel(z_posterior, z_posterior), True)
        kpp = self._compute_kernel_mean(self._compute_kernel(z_prior, z_prior), True)
        kpq = self._compute_kernel_mean(self._compute_kernel(z_prior, z_posterior), False)
        return kpp - 2 * kpq + kqq

    @staticmethod
    def _compute_kernel_mean(kernel, unbiased):
        N = kernel.shape[0]
        if unbiased:
            return (kernel.sum() - kernel.diagonal().sum()) / (N * (N - 1))
        return kernel.mean()

    @staticmethod
    def _compute_kernel(z0, z1):
        z_size = z0.shape[1]
        z0 = z0.unsqueeze(1)
        z1 = z1.unsqueeze(0)
        return torch.exp(-((z0 - z1) ** 2).sum(dim=-1) / (2 * z_size))


class dipMixin:
    """DIP-VAE covariance regularization."""

    def _dip_loss(self, q_m, q_s):
        cov = torch.cov(q_m.T) + torch.diag(torch.exp(q_s).mean(dim=0))
        cov_diag = cov.diagonal()
        cov_off_diag = cov - torch.diag(cov_diag)
        return 10 * ((cov_diag - 1) ** 2).sum() + 5 * (cov_off_diag ** 2).sum()


# ============================================================================
# Graph Adjacency Mixin (from CCVGAE)
# ============================================================================

class adjMixin:
    """Sparse adjacency matrix builder for graph-based training."""

    def _build_adj(self, edge_index, num_nodes, edge_weight=None):
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
        return torch.sparse_coo_tensor(
            edge_index, edge_weight,
            size=(num_nodes, num_nodes),
            device=edge_index.device,
        )


# ============================================================================
# Evaluation Metrics
# ============================================================================

class envMixin:
    """Clustering metrics for latent space evaluation."""

    def _calc_score_with_labels(self, latent, labels):
        """Compute clustering metrics using ground truth labels.

        ARI and NMI compare KMeans-predicted clusters against ground truth.
        ASW, CH, and DB are computed on ground truth labels to measure how
        well the latent space preserves known biological cell types.
        """
        n_clusters = len(np.unique(labels))
        pred = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(latent)
        return (
            adjusted_rand_score(labels, pred),
            normalized_mutual_info_score(labels, pred),
            silhouette_score(latent, labels),
            calinski_harabasz_score(latent, labels),
            davies_bouldin_score(latent, labels),
            self._calc_corr(latent),
        )

    def _calc_score(self, latent):
        """Score using self.labels (CCVGAE-style interface)."""
        labels = self._calc_label(latent)
        return self._metrics(latent, labels)

    def _calc_label(self, latent):
        if not hasattr(self, "labels") or self.labels is None:
            n_clusters = latent.shape[1]
        else:
            n_clusters = len(np.unique(self.labels))
        return KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(latent)

    @staticmethod
    def _calc_corr(latent):
        corr = np.abs(np.corrcoef(latent.T))
        return corr.sum(axis=1).mean() - 1

    def _metrics(self, latent, pred_labels):
        """Compute metrics: ARI/NMI vs ground truth, ASW/CH/DB on ground truth."""
        ARI = adjusted_rand_score(self.labels, pred_labels)
        NMI = normalized_mutual_info_score(self.labels, pred_labels)
        ASW = silhouette_score(latent, self.labels)
        CH = calinski_harabasz_score(latent, self.labels)
        DB = davies_bouldin_score(latent, self.labels)
        PC = self._calc_corr(latent)
        return ARI, NMI, ASW, CH, DB, PC


# ============================================================================
# Neural SDE Solver
# ============================================================================

class SDEMixin:
    """Neural SDE solver using torchsde.sdeint()."""

    def solve_sde(
        self,
        sde_func: nn.Module,
        z0: torch.Tensor,
        t: torch.Tensor,
        method: str = "euler",
        step_size: Optional[float] = None,
    ) -> torch.Tensor:
        if not (hasattr(sde_func, "f") and hasattr(sde_func, "g")):
            raise ValueError("sde_func must have f() and g() methods")

        device = z0.device
        t = t.detach().to(device)
        sde_func = sde_func.to(device)

        if step_size is None or step_size == "auto":
            dt = (t[-1] - t[0]) / (len(t) - 1)
        else:
            dt = float(step_size)

        try:
            pred_z = torchsde.sdeint(sde_func, z0, t, method=method, dt=dt)
        except RuntimeError as e:
            raise RuntimeError(
                f"SDE solving failed on {device} with {method}. "
                f"Try smaller step_size, method='euler', or CPU. Original: {e}"
            )
        return pred_z


# ============================================================================
# Scanpy-Based Preprocessing (from CCVGAE)
# ============================================================================

class scMixin:
    """Scanpy-based preprocessing: normalization, HVG, decomposition, batch correction."""

    def _preprocess(self, adata: AnnData, layer: str, n_var: int) -> None:
        import scanpy as sc

        try:
            if layer not in adata.layers.keys():
                adata.layers[layer] = adata.X.copy()
                print(f"Creating layer: {layer}.")
            if "log1p" not in adata.uns.keys():
                sc.pp.normalize_total(adata, target_sum=1e4)
                sc.pp.log1p(adata)
                print("Performing normalization.")
            if "highly_variable" not in adata.var.keys():
                if n_var:
                    sc.pp.highly_variable_genes(adata, n_top_genes=n_var)
                else:
                    sc.pp.highly_variable_genes(adata)
                print("Selecting highly variable genes.")
        except (ValueError, KeyError, AttributeError) as e:
            print(f"Error during preprocessing: {e}")

    def _decomposition(self, adata: AnnData, tech: str, latent_dim: int) -> None:
        from sklearn.decomposition import PCA, NMF, FastICA, TruncatedSVD, FactorAnalysis

        try:
            decomp_map = {
                "PCA": PCA,
                "NMF": NMF,
                "FastICA": FastICA,
                "TruncatedSVD": TruncatedSVD,
                "FactorAnalysis": FactorAnalysis,
            }
            if tech not in decomp_map:
                raise ValueError(f"Unsupported method: {tech}")

            X_hvg = adata[:, adata.var["highly_variable"]].X
            if issparse(X_hvg):
                X_hvg = X_hvg.toarray()
            latent = decomp_map[tech](n_components=latent_dim).fit_transform(X_hvg)
            adata.obsm[f"X_{tech}"] = latent
            print(f"Stored latent in adata.obsm['X_{tech}'].")
        except (ValueError, KeyError, TypeError) as e:
            print(f"Error during decomposition: {e}")

    def _batchcorrect(self, adata: AnnData, batch_tech: str, tech: str, layer: str) -> None:
        try:
            has_batch = "batch" in adata.obs.columns
            if batch_tech == "harmony":
                if not has_batch:
                    raise ValueError("Harmony requires 'batch' column in adata.obs")
                import scanpy.external as sce
                sce.pp.harmony_integrate(
                    adata, key="batch", basis=f"X_{tech}", adjusted_basis=f"X_harmony_{tech}"
                )
            elif batch_tech == "scvi":
                import scvi

                setup_kwargs = {"layer": layer}
                if has_batch:
                    setup_kwargs["batch_key"] = "batch"
                scvi.model.SCVI.setup_anndata(adata, **setup_kwargs)
                model = scvi.model.SCVI(adata)
                model.train()
                adata.obsm["X_scvi"] = model.get_latent_representation()
        except (ValueError, ImportError, RuntimeError) as e:
            print(f"Error during batch correction: {e}")
