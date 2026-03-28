# ============================================================================
# mixin.py - Shared Mixins for HSDE
# ============================================================================
"""
Mixins:
- scviMixin: Count-based likelihoods (NB, ZINB, Poisson, ZIP)
- betatcMixin: β-TC-VAE total correlation
- infoMixin: InfoVAE with MMD
- dipMixin: DIP-VAE covariance regularization
- envMixin: Clustering evaluation metrics
- SDEMixin: Neural SDE solver
"""

import logging

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
from typing import Optional

logger = logging.getLogger(__name__)


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
# Evaluation Metrics
# ============================================================================

class envMixin:
    """Clustering metrics for latent space evaluation."""

    def _calc_score_with_labels(self, latent, labels):
        """Compute clustering metrics with unsupervised reference labels."""
        n_clusters = len(np.unique(labels))
        if n_clusters <= 1:
            logger.warning("Only %d unique label(s); returning NaN for cluster metrics", n_clusters)
            return (np.nan, np.nan, np.nan, np.nan, np.nan, self._calc_corr(latent))
        pred = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(latent)
        n_pred = len(np.unique(pred))
        return (
            adjusted_rand_score(labels, pred),
            normalized_mutual_info_score(labels, pred),
            silhouette_score(latent, pred) if n_pred > 1 else np.nan,
            calinski_harabasz_score(latent, pred) if n_pred > 1 else np.nan,
            davies_bouldin_score(latent, pred) if n_pred > 1 else np.nan,
            self._calc_corr(latent),
        )

    @staticmethod
    def _calc_corr(latent):
        if latent.shape[1] <= 1:
            return 0.0
        corr = np.abs(np.corrcoef(latent.T))
        return corr.sum(axis=1).mean() - 1


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

        if len(t) <= 1:
            raise ValueError("SDE requires at least 2 unique timepoints, got %d" % len(t))

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
