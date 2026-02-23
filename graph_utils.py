# ============================================================================
# graph_utils.py - Graph Structure Decoders and Adjacency Utilities
# ============================================================================
"""
Graph structure learning utilities from CCVGAE:
- Adjacency-to-edge conversion with sparsification
- Bilinear, inner-product, and MLP structure decoders
- Sparse adjacency matrix builder
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class AdjToEdge:
    """
    Convert adjacency matrix to edge indices and weights.

    Parameters
    ----------
    threshold : float
        Probability threshold for edge existence.
    sparse_threshold : int, optional
        Maximum number of edges to keep per node.
    symmetric : bool
        Ensure symmetric edges.
    add_self_loops : bool
        Whether to add self-loops.
    """

    def __init__(
        self,
        threshold: float = 0,
        sparse_threshold: Optional[int] = None,
        symmetric: bool = True,
        add_self_loops: bool = False,
    ):
        self.threshold = threshold
        self.sparse_threshold = sparse_threshold
        self.symmetric = symmetric
        self.add_self_loops = add_self_loops

    def _sparsify(self, adj: np.ndarray, k: int) -> np.ndarray:
        sparse_adj = np.zeros_like(adj)
        for i in range(adj.shape[0]):
            actual_k = min(k, adj.shape[1])
            if actual_k == 0:
                continue
            top_k_idx = np.argpartition(adj[i], -actual_k)[-actual_k:]
            mask = adj[i, top_k_idx] > self.threshold
            sparse_adj[i, top_k_idx] = adj[i, top_k_idx] * mask
        return sparse_adj

    def _symmetrize(
        self, edge_index: np.ndarray, edge_weight: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if edge_index.size == 0 or edge_weight.size == 0:
            return np.zeros((2, 0), dtype=np.int64), np.array([], dtype=edge_weight.dtype)
        n = max(edge_index[0].max(), edge_index[1].max()) + 1
        adj = np.zeros((n, n))
        adj[edge_index[0], edge_index[1]] = edge_weight
        adj = (adj + adj.T) / 2
        rows, cols = np.nonzero(adj)
        return np.stack([rows, cols]), adj[rows, cols]

    def _add_self_loops(
        self, edge_index: np.ndarray, edge_weight: np.ndarray, num_nodes: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        self_loops = np.arange(num_nodes)
        self_loops = np.stack([self_loops, self_loops])
        edge_index = np.concatenate([edge_index, self_loops], axis=1)
        edge_weight = np.concatenate([edge_weight, np.ones(num_nodes)])
        return edge_index, edge_weight

    def convert(self, adj: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert adjacency matrix to (edge_index, edge_weight)."""
        if self.sparse_threshold is not None:
            adj = self._sparsify(adj, self.sparse_threshold)
        mask = adj > self.threshold
        rows, cols = np.nonzero(mask)
        edge_index = np.stack([rows, cols])
        edge_weight = adj[rows, cols]
        if self.symmetric:
            edge_index, edge_weight = self._symmetrize(edge_index, edge_weight)
        if self.add_self_loops:
            edge_index, edge_weight = self._add_self_loops(edge_index, edge_weight, adj.shape[0])
        return edge_index, edge_weight


# ============================================================================
# Structure Decoders
# ============================================================================

class BilinearDecoder(nn.Module):
    """Bilinear decoder: adj = σ(z W zᵀ)."""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(latent_dim, latent_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(torch.matmul(z @ self.weight, z.t()))


class InnerProductDecoder(nn.Module):
    """Inner product decoder: adj = σ(z zᵀ)."""

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(torch.matmul(z, z.t()))


class MLPDecoder(nn.Module):
    """MLP decoder for pairwise edge prediction."""

    def __init__(self, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = z.size(0)
        row, col = edge_index
        edge_features = torch.cat([z[row], z[col]], dim=1)
        edge_probs = self.mlp(edge_features).squeeze()
        adj_recon = torch.zeros((num_nodes, num_nodes), device=z.device)
        adj_recon[row, col] = edge_probs
        return adj_recon


class GraphStructureDecoder(nn.Module):
    """
    Combined structure decoder with edge converter for graph reconstruction.

    Supports 'bilinear', 'inner_product', and 'mlp' modes.
    """

    def __init__(
        self,
        structure_decoder: str,
        latent_dim: int,
        hidden_dim: Optional[int] = None,
        threshold: float = 0,
        sparse_threshold: Optional[int] = None,
        symmetric: bool = True,
        add_self_loops: bool = False,
    ):
        super().__init__()
        if structure_decoder == "bilinear":
            self.decoder = BilinearDecoder(latent_dim)
        elif structure_decoder == "inner_product":
            self.decoder = InnerProductDecoder()
        elif structure_decoder == "mlp":
            if hidden_dim is None:
                raise ValueError("hidden_dim required for MLP decoder")
            self.decoder = MLPDecoder(latent_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown decoder: {structure_decoder}")

        self.edge_converter = AdjToEdge(
            threshold=threshold,
            sparse_threshold=sparse_threshold,
            symmetric=symmetric,
            add_self_loops=add_self_loops,
        )
        self.structure_decoder = structure_decoder

    def forward(
        self,
        z: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.structure_decoder == "mlp":
            if edge_index is None:
                raise ValueError("edge_index required for MLP decoder")
            adj = self.decoder(z, edge_index)
        else:
            adj = self.decoder(z)

        adj_np = adj.detach().cpu().numpy()
        edge_index_np, edge_weight_np = self.edge_converter.convert(adj_np)

        device = z.device
        edge_index_out = torch.from_numpy(edge_index_np).to(device)
        edge_weight_out = torch.from_numpy(edge_weight_np).float().to(device)
        return adj, edge_index_out, edge_weight_out


def build_adj(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build a sparse adjacency matrix from edge_index and optional weights."""
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    return torch.sparse_coo_tensor(
        edge_index, edge_weight, size=(num_nodes, num_nodes), device=edge_index.device
    )
