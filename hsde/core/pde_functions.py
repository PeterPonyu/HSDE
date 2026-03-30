# ============================================================================
# pde_functions.py - Latent-Space Graph PDE Diffusion
# ============================================================================
"""
Implements a discrete graph heat equation for latent smoothing:
    ∂z/∂t = Δ_G z
where Δ_G is a kNN graph Laplacian built on latent embeddings.
"""

import torch
import torch.nn as nn


class GraphDiffusionPDE(nn.Module):
    """kNN graph diffusion (explicit Euler) in latent space."""

    def __init__(
        self,
        k: int = 15,
        alpha: float = 0.1,
        steps: int = 2,
        tau: float = 1.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.k = max(1, int(k))
        self.alpha = float(alpha)
        self.steps = max(1, int(steps))
        self.tau = float(tau)
        self.eps = float(eps)

    def _knn_weights(self, z: torch.Tensor):
        n = z.shape[0]
        if n <= 1:
            return None, None

        dists = torch.cdist(z, z, p=2)
        k_eff = min(self.k, n - 1)
        knn_dist, knn_idx = torch.topk(dists, k=k_eff + 1, dim=1, largest=False)
        knn_dist = knn_dist[:, 1:]
        knn_idx = knn_idx[:, 1:]

        logits = -(knn_dist ** 2) / max(self.tau, self.eps)
        weights = torch.softmax(logits, dim=1)
        return knn_idx, weights

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2:
            raise ValueError(f"Expected 2D latent tensor, got shape={tuple(z.shape)}")

        # Detach kNN graph construction from encoder gradients:
        # the graph topology should not influence encoder updates.
        knn_idx, weights = self._knn_weights(z.detach())
        if knn_idx is None:
            return z

        z_t = z
        for _ in range(self.steps):
            neigh = z_t[knn_idx]
            z_center = z_t.unsqueeze(1)
            laplace = (weights.unsqueeze(-1) * (neigh - z_center)).sum(dim=1)
            z_t = z_t + self.alpha * laplace

        return z_t
