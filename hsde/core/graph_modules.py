# ============================================================================
# graph_modules.py - Graph Neural Network Encoder/Decoder from CCVGAE
# ============================================================================
"""
Graph-based encoder and decoder modules using PyTorch Geometric convolutions.
Supports: GAT, GCN, Cheb, SAGE, Graph, TAG, ARMA, Transformer, SG, SSG.

Integrated from PeterPonyu/CCVGAE with adaptations for the unified framework.
"""

from typing import Dict, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

try:
    from torch_geometric.nn import (
        ARMAConv,
        ChebConv,
        GATConv,
        GCNConv,
        GraphConv,
        SAGEConv,
        SGConv,
        SSGConv,
        TAGConv,
        TransformerConv,
    )

    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False


def _require_pyg():
    if not _HAS_PYG:
        raise ImportError(
            "torch_geometric is required for graph-based encoders/decoders. "
            "Install via: pip install torch-geometric"
        )


# ============================================================================
# Base Graph Network
# ============================================================================

class BaseGraphNetwork(nn.Module):
    """
    Base class for graph neural networks with configurable convolution types.
    """

    CONV_LAYERS: Dict[str, Type[nn.Module]] = {}

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        conv_layer_type: str = "GAT",
        hidden_layers: int = 2,
        dropout: float = 0.05,
        Cheb_k: int = 1,
        alpha: float = 0.5,
    ) -> None:
        super().__init__()
        _require_pyg()

        # Populate CONV_LAYERS lazily after import
        if not BaseGraphNetwork.CONV_LAYERS:
            BaseGraphNetwork.CONV_LAYERS = {
                "GCN": GCNConv,
                "Cheb": ChebConv,
                "SAGE": SAGEConv,
                "Graph": GraphConv,
                "TAG": TAGConv,
                "ARMA": ARMAConv,
                "GAT": GATConv,
                "Transformer": TransformerConv,
                "SG": SGConv,
                "SSG": SSGConv,
            }

        if conv_layer_type not in self.CONV_LAYERS:
            raise ValueError(
                f"Unsupported layer type: {conv_layer_type}. "
                f"Choose from {list(self.CONV_LAYERS.keys())}"
            )

        self.conv_layer_type = conv_layer_type
        self.conv_layer = self.CONV_LAYERS[conv_layer_type]
        self.hidden_layers = hidden_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.relu = nn.ReLU()

        # Build layers
        self.convs.append(self._create_conv_layer(input_dim, hidden_dim, Cheb_k, alpha))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.dropouts.append(nn.Dropout(dropout))

        for _ in range(hidden_layers - 1):
            self.convs.append(self._create_conv_layer(hidden_dim, hidden_dim, Cheb_k, alpha))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout))

        self._build_output_layer(hidden_dim, output_dim, Cheb_k, alpha)
        self.apply(self._init_weights)

    def _create_conv_layer(self, in_dim, out_dim, Cheb_k, alpha):
        if self.conv_layer_type == "Transformer":
            return self.conv_layer(in_dim, out_dim, edge_dim=1)
        elif self.conv_layer_type == "Cheb":
            return self.conv_layer(in_dim, out_dim, Cheb_k)
        elif self.conv_layer_type == "SSG":
            return self.conv_layer(in_dim, out_dim, alpha=alpha)
        else:
            return self.conv_layer(in_dim, out_dim)

    def _build_output_layer(self, hidden_dim, output_dim, Cheb_k, alpha):
        raise NotImplementedError

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _process_layer(self, x, conv, edge_index, edge_weight):
        if isinstance(conv, SAGEConv):
            return conv(x, edge_index)
        elif isinstance(conv, TransformerConv):
            return conv(x, edge_index, edge_weight.view(-1, 1) if edge_weight is not None else None)
        else:
            return conv(x, edge_index, edge_weight)


# ============================================================================
# Graph Encoder (Variational)
# ============================================================================

class GraphEncoder(BaseGraphNetwork):
    """
    Graph-based variational encoder producing (q_z, q_m, q_s).

    Uses graph convolution layers to capture cell-cell relationships,
    with optional residual connections and time prediction for SDE mode.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        conv_layer_type: str = "GAT",
        hidden_layers: int = 2,
        dropout: float = 0.05,
        Cheb_k: int = 1,
        alpha: float = 0.5,
        use_sde: bool = False,
    ) -> None:
        self._use_sde = use_sde
        self._hidden_dim = hidden_dim
        super().__init__(
            input_dim, hidden_dim, latent_dim,
            conv_layer_type, hidden_layers, dropout, Cheb_k, alpha,
        )
        if use_sde:
            self.time_encoder = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def _build_output_layer(self, hidden_dim, latent_dim, Cheb_k, alpha):
        self.conv_mean = self._create_conv_layer(hidden_dim, latent_dim, Cheb_k, alpha)
        self.conv_logvar = self._create_conv_layer(hidden_dim, latent_dim, Cheb_k, alpha)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        use_residual: bool = True,
    ):
        """
        Returns
        -------
        Without SDE: (q_z, q_m, q_s)
        With SDE: (q_z, q_m, q_s, Normal, t)
        """
        residual = None
        h = x

        for i, (conv, bn, drop) in enumerate(zip(self.convs, self.bns, self.dropouts)):
            h = self._process_layer(h, conv, edge_index, edge_weight)
            h = bn(h)
            h = self.relu(h)
            h = drop(h)
            if use_residual and i == 0:
                residual = h

        if use_residual and residual is not None:
            h = h + residual

        feature = h  # save for time encoder

        q_m = self._process_layer(h, self.conv_mean, edge_index, edge_weight)
        q_s = self._process_layer(h, self.conv_logvar, edge_index, edge_weight)

        q_m = q_m.clamp(-10, 10)
        q_s = q_s.clamp(-10, 10)
        std = F.softplus(q_s).clamp(1e-6, 5.0)
        dist = Normal(q_m, std)
        q_z = dist.rsample()

        if self._use_sde:
            t = self.time_encoder(feature).squeeze(-1)
            return q_z, q_m, q_s, dist, t

        return q_z, q_m, q_s, dist


# ============================================================================
# Graph Decoder (Feature Reconstruction)
# ============================================================================

class GraphDecoder(BaseGraphNetwork):
    """Graph-based feature decoder with softmax output."""

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        latent_dim: int,
        conv_layer_type: str = "GAT",
        hidden_layers: int = 2,
        dropout: float = 0.05,
        Cheb_k: int = 1,
        alpha: float = 0.5,
        loss_type: str = "nb",
    ):
        self._loss_type = loss_type
        super().__init__(
            latent_dim, hidden_dim, state_dim,
            conv_layer_type, hidden_layers, dropout, Cheb_k, alpha,
        )
        self.disp = nn.Parameter(torch.randn(state_dim))

        if loss_type in ("zinb", "zip"):
            self.dropout_layer = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, state_dim),
            )

    def _build_output_layer(self, hidden_dim, output_dim, Cheb_k, alpha):
        self.output_conv = self._create_conv_layer(hidden_dim, output_dim, Cheb_k, alpha)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        use_residual: bool = True,
        z_for_dropout: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Returns (output, dropout_logits)."""
        residual = None
        h = x

        for i, (conv, bn, drop) in enumerate(zip(self.convs, self.bns, self.dropouts)):
            h = self._process_layer(h, conv, edge_index, edge_weight)
            h = bn(h)
            h = self.relu(h)
            h = drop(h)
            if use_residual and i == 0:
                residual = h

        if use_residual and residual is not None:
            h = h + residual

        output = self._process_layer(h, self.output_conv, edge_index, edge_weight)
        output = F.softmax(output, dim=-1)

        dropout_out = None
        if self._loss_type in ("zinb", "zip"):
            src = z_for_dropout if z_for_dropout is not None else x
            dropout_out = self.dropout_layer(src)

        return output, dropout_out
