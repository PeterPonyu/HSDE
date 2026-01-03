
"""SDE-based mixins for single-cell analysis."""

import torch
import torch.nn as nn
import torchsde
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
from scipy.sparse import issparse, csr_matrix
from scipy.stats import norm
from typing import Optional, Tuple
from anndata import AnnData


class scviMixin:
    """Count-based likelihoods for scRNA-seq (NB, ZINB, Poisson, ZIP)."""
    
    @staticmethod
    def _normal_kl(mu1, lv1, mu2, lv2):
        """KL(N(mu1, exp(lv1)) || N(mu2, exp(lv2)))"""
        v1, v2 = torch.exp(lv1), torch.exp(lv2)
        return (lv2 - lv1) / 2.0 + (v1 + (mu1 - mu2) ** 2) / (2.0 * v2) - 0.5
    
    @staticmethod
    def _log_nb(x, mu, theta, eps=1e-8):
        """Negative Binomial log-likelihood."""
        log_theta_mu = torch.log(theta + mu + eps)
        return (
            theta * (torch.log(theta + eps) - log_theta_mu)
            + x * (torch.log(mu + eps) - log_theta_mu)
            + torch.lgamma(x + theta)
            - torch.lgamma(theta + eps)
            - torch.lgamma(x + 1)
        )
    
    def _log_zinb(self, x, mu, theta, pi, eps=1e-8):
        """Zero-Inflated Negative Binomial log-likelihood."""
        pi = torch.sigmoid(pi)
        log_nb = self._log_nb(x, mu, theta, eps)
        case_zero = torch.log(pi + (1 - pi) * torch.exp(log_nb) + eps)
        case_nonzero = torch.log(1 - pi + eps) + log_nb
        return torch.where(x < eps, case_zero, case_nonzero)
    
    @staticmethod
    def _log_poisson(x, mu, eps=1e-8):
        """Poisson log-likelihood."""
        return x * torch.log(mu + eps) - mu - torch.lgamma(x + 1)
    
    def _log_zip(self, x, mu, pi, eps=1e-8):
        """Zero-Inflated Poisson log-likelihood."""
        pi = torch.sigmoid(pi)
        case_zero = torch.log(pi + (1 - pi) * torch.exp(-mu) + eps)
        case_nonzero = torch.log(1 - pi + eps) + self._log_poisson(x, mu, eps)
        return torch.where(x < eps, case_zero, case_nonzero)


class betatcMixin:
    """β-TC-VAE disentanglement via total correlation penalty."""
    
    @staticmethod
    def _betatc_compute_gaussian_log_density(samples, mean, log_var):
        """Log p(samples | mean, log_var) for Gaussian."""
        inv_sigma = torch.exp(-log_var)
        tmp = samples - mean
        return -0.5 * (tmp ** 2 * inv_sigma + log_var + np.log(2 * np.pi))
    
    def _betatc_compute_total_correlation(self, z_sampled, z_mean, z_logvar):
        """TC = KL(q(z) || ∏_j q(z_j)) measures dimension dependence."""
        log_qz_prob = self._betatc_compute_gaussian_log_density(
            z_sampled.unsqueeze(1), z_mean.unsqueeze(0), z_logvar.unsqueeze(0)
        )
        log_qz = log_qz_prob.sum(dim=2).exp().sum(dim=1).log()
        log_qz_product = log_qz_prob.exp().sum(dim=1).log().sum(dim=1)
        return (log_qz - log_qz_product).mean()


class infoMixin:
    """InfoVAE with Maximum Mean Discrepancy (MMD) regularization."""
    
    def _compute_mmd(self, z_posterior, z_prior):
        """MMD²(q(z) || p(z)) using RBF kernel."""
        kqq = self._compute_kernel_mean(self._compute_kernel(z_posterior, z_posterior), True)
        kpp = self._compute_kernel_mean(self._compute_kernel(z_prior, z_prior), True)
        kpq = self._compute_kernel_mean(self._compute_kernel(z_prior, z_posterior), False)
        return kpp - 2 * kpq + kqq
    
    @staticmethod
    def _compute_kernel_mean(kernel, unbiased):
        """Mean of kernel matrix (unbiased removes diagonal)."""
        N = kernel.shape[0]
        if unbiased:
            return (kernel.sum() - kernel.diagonal().sum()) / (N * (N - 1))
        return kernel.mean()
    
    @staticmethod
    def _compute_kernel(z0, z1):
        """RBF kernel: k(x,y) = exp(-||x-y||²/σ²)"""
        z_size = z0.shape[1]
        z0 = z0.unsqueeze(1)
        z1 = z1.unsqueeze(0)
        return torch.exp(-((z0 - z1) ** 2).sum(dim=-1) / (2 * z_size))


class dipMixin:
    """DIP-VAE: Disentanglement via posterior covariance regularization."""
    
    def _dip_loss(self, q_m, q_s):
        """Penalize off-diagonal covariance and deviation from unit variance."""
        cov = torch.cov(q_m.T) + torch.diag(torch.exp(q_s).mean(dim=0))
        cov_diag = cov.diagonal()
        cov_off_diag = cov - torch.diag(cov_diag)
        return 10 * ((cov_diag - 1) ** 2).sum() + 5 * (cov_off_diag ** 2).sum()


class envMixin:
    """Clustering metrics for latent space evaluation."""
    
    def _calc_score_with_labels(self, latent, labels):
        """Compute ARI, NMI, Silhouette, Calinski-Harabasz, Davies-Bouldin, Correlation."""
        n_clusters = len(np.unique(labels))
        pred = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(latent)
        
        return (
            adjusted_rand_score(labels, pred),
            normalized_mutual_info_score(labels, pred),
            silhouette_score(latent, pred),
            calinski_harabasz_score(latent, pred),
            davies_bouldin_score(latent, pred),
            self._calc_corr(latent)
        )
    
    @staticmethod
    def _calc_corr(latent):
        """Mean absolute off-diagonal correlation."""
        corr = np.abs(np.corrcoef(latent.T))
        return corr.sum(axis=1).mean() - 1


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
        """
        Solve dz = f(t,z)dt + g(t,z)dW using torchsde.
        
        Returns: (num_times, batch_size, latent_dim)
        """
        # Validate SDE function interface
        if not (hasattr(sde_func, 'f') and hasattr(sde_func, 'g')):
            raise ValueError("sde_func must have f() and g() methods for torchsde")
        
        # Ensure all tensors on same device
        device = z0.device
        t = t.detach().to(device)
        sde_func = sde_func.to(device)
        
        # Compute step size
        if step_size is None or step_size == "auto":
            dt = (t[-1] - t[0]) / (len(t) - 1)
        else:
            dt = float(step_size)
        
        try:
            pred_z = torchsde.sdeint(sde_func, z0, t, method=method, dt=dt)
        except RuntimeError as e:
            raise RuntimeError(
                f"SDE solving failed on {device} with {method} method. "
                f"Try: (1) smaller step_size, (2) method='euler', (3) CPU mode. "
                f"Original error: {e}"
            )
        
        return pred_z

