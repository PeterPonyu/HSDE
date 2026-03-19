# ============================================================================
# module.py - Unified Neural Network Modules (MLP / Transformer / Graph)
# ============================================================================
"""
Neural network modules combining HSDE and CCVGAE architectures:

Encoders:
  - MLPEncoder: Fully-connected variational encoder (HSDE)
  - TransformerEncoder: Multi-head projection transformer (HSDE)
  - GraphEncoder: Graph attention / convolution encoder (CCVGAE)

Decoders:
  - MLPDecoder: Count-based decoder with NB/ZINB/Poisson/ZIP (HSDE)
  - GraphDecoder: Graph convolution decoder (CCVGAE)

VAE:
  - Unified VAE combining encoder + decoder + SDE + PDE + manifold + graph structure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Optional, Tuple
import dataclasses

from .utils import exp_map_at_origin
from .mixin import SDEMixin
from .sde_functions import create_controlled_sde
from .pde_functions import GraphDiffusionPDE


@dataclasses.dataclass
class ForwardOutput:
    """Structured output from VAE.forward(), replacing fragile positional tuples."""
    # Core VAE outputs (always present)
    q_z: torch.Tensor
    q_m: torch.Tensor
    q_s: torch.Tensor
    pred_x: torch.Tensor
    le: torch.Tensor
    ld: torch.Tensor
    pred_xl: torch.Tensor
    z_manifold: torch.Tensor
    ld_manifold: torch.Tensor
    dropout_x: torch.Tensor
    dropout_xl: torch.Tensor
    # Graph structure decoder (None when not using graph decoder)
    pred_a: Optional[torch.Tensor] = None
    # SDE-specific (None when not using SDE)
    q_z_sde: Optional[torch.Tensor] = None
    pred_x_sde: Optional[torch.Tensor] = None
    dropout_x_sde: Optional[torch.Tensor] = None
    x_sorted: Optional[torch.Tensor] = None
    t: Optional[torch.Tensor] = None
    # PDE-specific (None when not using PDE)
    q_z_pde: Optional[torch.Tensor] = None
    pred_x_pde: Optional[torch.Tensor] = None
    dropout_x_pde: Optional[torch.Tensor] = None


# ============================================================================
# MLP Encoder (from HSDE)
# ============================================================================

class MLPEncoder(nn.Module):
    """Variational encoder: x → q(z|x) with optional time prediction for SDE."""

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        action_dim: int,
        use_layer_norm: bool = True,
        use_sde: bool = False,
    ):
        super().__init__()
        self.use_layer_norm = use_layer_norm
        self.use_sde = use_sde

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim * 2)

        if use_layer_norm:
            self.ln1 = nn.LayerNorm(hidden_dim)
            self.ln2 = nn.LayerNorm(hidden_dim)

        if use_sde:
            self.time_encoder = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

        self.apply(lambda m: nn.init.xavier_normal_(m.weight) if isinstance(m, nn.Linear) else None)

    def forward(self, x):
        h = F.relu(self.ln1(self.fc1(x)) if self.use_layer_norm else self.fc1(x))
        h = F.relu(self.ln2(self.fc2(h)) if self.use_layer_norm else self.fc2(h))
        output = self.fc3(h)
        feature = h

        q_m, q_s = output.chunk(2, dim=-1)
        q_m = q_m.clamp(-10, 10)
        q_s = q_s.clamp(-10, 10)
        s = F.softplus(q_s).clamp(1e-6, 5.0)
        n = Normal(q_m, s)
        q_z = n.rsample()

        if self.use_sde:
            t = self.time_encoder(feature).squeeze(-1)
            return q_z, q_m, q_s, n, t
        return q_z, q_m, q_s, n


# ============================================================================
# Transformer Encoder (from HSDE)
# ============================================================================

class TransformerEncoder(nn.Module):
    """Multi-head projection transformer for gene expression encoding."""

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        action_dim: int,
        attn_embed_dim: int = 64,
        attn_num_heads: int = 4,
        attn_num_layers: int = 2,
        attn_seq_len: int = 8,
        attn_dropout: float = 0.1,
        use_sde: bool = False,
    ):
        super().__init__()
        self.use_sde = use_sde
        self.attn_embed_dim = attn_embed_dim

        self.projection_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim, attn_embed_dim),
                nn.LayerNorm(attn_embed_dim),
                nn.GELU(),
                nn.Dropout(attn_dropout),
            )
            for _ in range(attn_seq_len)
        ])

        self.token_embeddings = nn.Parameter(
            torch.randn(1, attn_seq_len, attn_embed_dim) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=attn_embed_dim,
            nhead=attn_num_heads,
            dim_feedforward=max(attn_embed_dim * 4, 128),
            dropout=attn_dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=attn_num_layers)

        self.aggregation_query = nn.Parameter(torch.randn(1, 1, attn_embed_dim) * 0.02)
        self.cross_attention = nn.MultiheadAttention(
            attn_embed_dim, attn_num_heads, dropout=attn_dropout, batch_first=True
        )

        self.attn_final_norm = nn.LayerNorm(attn_embed_dim)
        self.attn_pool_fc = nn.Linear(attn_embed_dim, action_dim * 2)

        if use_sde:
            self.time_encoder = nn.Sequential(nn.Linear(attn_embed_dim, 1), nn.Sigmoid())

        self.apply(lambda m: nn.init.xavier_normal_(m.weight) if isinstance(m, nn.Linear) else None)

    def forward(self, x):
        tokens = torch.stack([proj(x) for proj in self.projection_heads], dim=1)
        tokens = tokens + self.token_embeddings.expand(x.size(0), -1, -1)
        seq_out = self.transformer(tokens)

        query = self.aggregation_query.expand(x.size(0), -1, -1)
        pooled, _ = self.cross_attention(query, seq_out, seq_out)
        pooled = pooled.squeeze(1)
        pooled = self.attn_final_norm(pooled)

        output = self.attn_pool_fc(pooled)
        feature = pooled

        q_m, q_s = output.chunk(2, dim=-1)
        q_m = q_m.clamp(-10, 10)
        q_s = q_s.clamp(-10, 10)
        s = F.softplus(q_s).clamp(1e-6, 5.0)
        n = Normal(q_m, s)
        q_z = n.rsample()

        if self.use_sde:
            t = self.time_encoder(feature).squeeze(-1)
            return q_z, q_m, q_s, n, t
        return q_z, q_m, q_s, n


# ============================================================================
# MLP Decoder (from HSDE)
# ============================================================================

class Decoder(nn.Module):
    """Generative decoder: z → p(x|z) with count-based likelihoods."""

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        action_dim: int,
        loss_type: str = "nb",
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.use_layer_norm = use_layer_norm

        self.fc1 = nn.Linear(action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, state_dim)

        if use_layer_norm:
            self.ln1 = nn.LayerNorm(hidden_dim)
            self.ln2 = nn.LayerNorm(hidden_dim)

        self.disp = nn.Parameter(torch.randn(state_dim))

        if loss_type in ("zinb", "zip"):
            self.dropout = nn.Sequential(
                nn.Linear(action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, state_dim),
            )

        self.apply(lambda m: nn.init.xavier_normal_(m.weight) if isinstance(m, nn.Linear) else None)

    def forward(self, x):
        h = F.relu(self.ln1(self.fc1(x)) if self.use_layer_norm else self.fc1(x))
        h = F.relu(self.ln2(self.fc2(h)) if self.use_layer_norm else self.fc2(h))
        output = F.softmax(self.fc3(h), dim=-1)
        dropout = self.dropout(x) if self.loss_type in ("zinb", "zip") else None
        return output, dropout


# ============================================================================
# Unified VAE (MLP / Transformer / Graph + SDE + PDE + Manifold + Graph Structure)
# ============================================================================

class VAE(nn.Module, SDEMixin):
    """
    Unified VAE supporting three encoder types and optional:
    - Hyperbolic/Euclidean manifold regularization
    - Neural SDE for continuous dynamics
    - Graph PDE diffusion for latent smoothing
    - Graph structure decoder for adjacency reconstruction (CCVGAE)
    - Count-based likelihoods (NB, ZINB, Poisson, ZIP)
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        action_dim: int,
        i_dim: int,
        use_bottleneck_lorentz: bool = True,
        loss_type: str = "nb",
        use_layer_norm: bool = True,
        use_euclidean_manifold: bool = False,
        use_sde: bool = False,
        use_pde: bool = False,
        device: torch.device = None,
        # Encoder type selection
        encoder_type: str = "mlp",
        # Transformer params
        attn_embed_dim: int = 64,
        attn_num_heads: int = 4,
        attn_num_layers: int = 2,
        attn_seq_len: int = 8,
        attn_dropout: float = 0.1,
        # SDE params
        sde_hidden_dim: Optional[int] = None,
        sde_strategy: str = "scaled",
        sde_time_cond: str = "concat",
        sde_solver_method: str = "euler",
        sde_step_size: Optional[float] = None,
        # PDE params
        pde_k: int = 15,
        pde_alpha: float = 0.1,
        pde_steps: int = 2,
        pde_tau: float = 1.0,
        # Graph encoder params (CCVGAE)
        graph_type: str = "GAT",
        graph_hidden_layers: int = 2,
        graph_dropout: float = 0.05,
        graph_Cheb_k: int = 1,
        graph_alpha: float = 0.5,
        use_residual: bool = True,
        # Graph structure decoder
        use_graph_decoder: bool = False,
        structure_decoder_type: str = "mlp",
        decoder_hidden_dim: int = 128,
        graph_threshold: float = 0,
        graph_sparse_threshold: Optional[int] = None,
        # Feature decoder type
        feature_decoder_type: str = "mlp",
        **kwargs,
    ):
        super().__init__()

        self.encoder_type = encoder_type.lower()
        self.use_bottleneck_lorentz = use_bottleneck_lorentz
        self.use_euclidean_manifold = use_euclidean_manifold
        self.use_sde = use_sde
        self.use_pde = use_pde
        self.use_graph_decoder = use_graph_decoder
        self.use_residual = use_residual
        self.sde_solver_method = sde_solver_method
        self.sde_step_size = sde_step_size
        self.feature_decoder_type = feature_decoder_type.lower()

        # ----- Encoder -----
        if self.encoder_type == "mlp":
            self.encoder = MLPEncoder(
                state_dim, hidden_dim, action_dim, use_layer_norm, use_sde
            ).to(device)
        elif self.encoder_type == "transformer":
            self.encoder = TransformerEncoder(
                state_dim, hidden_dim, action_dim,
                attn_embed_dim, attn_num_heads, attn_num_layers,
                attn_seq_len, attn_dropout, use_sde,
            ).to(device)
        elif self.encoder_type == "graph":
            from .graph_modules import GraphEncoder as GEncoder

            self.encoder = GEncoder(
                state_dim, hidden_dim, action_dim,
                conv_layer_type=graph_type,
                hidden_layers=graph_hidden_layers,
                dropout=graph_dropout,
                Cheb_k=graph_Cheb_k,
                alpha=graph_alpha,
                use_sde=use_sde,
            ).to(device)
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        # ----- Feature Decoder -----
        if self.feature_decoder_type == "mlp":
            self.decoder = Decoder(
                state_dim, hidden_dim, action_dim, loss_type, use_layer_norm
            ).to(device)
        elif self.feature_decoder_type == "graph":
            from .graph_modules import GraphDecoder as GDecoder

            self.decoder = GDecoder(
                state_dim, hidden_dim, action_dim,
                conv_layer_type=graph_type,
                hidden_layers=graph_hidden_layers,
                dropout=graph_dropout,
                Cheb_k=graph_Cheb_k,
                alpha=graph_alpha,
                loss_type=loss_type,
            ).to(device)
        else:
            raise ValueError(f"Unknown feature_decoder_type: {feature_decoder_type}")

        # ----- Graph Structure Decoder (CCVGAE) -----
        if use_graph_decoder:
            from .graph_utils import GraphStructureDecoder

            self.structure_decoder = GraphStructureDecoder(
                structure_decoder=structure_decoder_type,
                latent_dim=action_dim,
                hidden_dim=decoder_hidden_dim,
                threshold=graph_threshold,
                sparse_threshold=graph_sparse_threshold,
                symmetric=True,
                add_self_loops=False,
            ).to(device)

        # ----- Information Bottleneck (Coupling) -----
        self.latent_encoder = nn.Linear(action_dim, i_dim).to(device)
        self.latent_decoder = nn.Linear(i_dim, action_dim).to(device)

        # ----- SDE Solver -----
        if use_sde:
            sde_n_hidden = sde_hidden_dim or hidden_dim
            self.sde_solver = create_controlled_sde(
                strategy=sde_strategy,
                n_latent=action_dim,
                n_hidden=sde_n_hidden,
                time_cond=sde_time_cond,
            ).to(device)

        # ----- PDE Diffusion -----
        if use_pde:
            self.pde_solver = GraphDiffusionPDE(
                k=pde_k, alpha=pde_alpha, steps=pde_steps, tau=pde_tau
            ).to(device)

    # ----- Manifold Mapping -----
    def _map_to_manifold(self, z: torch.Tensor) -> torch.Tensor:
        if self.use_euclidean_manifold:
            return z
        z_clipped = z.clamp(-5, 5)
        z_tangent = F.pad(z_clipped, (1, 0), value=0)
        return exp_map_at_origin(z_tangent)

    # ----- Decoder Wrapper -----
    def _decode(self, z, edge_index=None, edge_weight=None):
        """Unified decode handling both MLP and graph feature decoders."""
        if self.feature_decoder_type == "graph":
            if edge_index is None:
                raise ValueError("edge_index required for graph feature decoder")
            return self.decoder(z, edge_index, edge_weight, self.use_residual, z_for_dropout=z)
        else:
            return self.decoder(z)

    # ----- Forward -----
    def forward(self, x, edge_index=None, edge_weight=None):
        if self.use_sde:
            return self._forward_sde(x, edge_index, edge_weight)
        return self._forward_standard(x, edge_index, edge_weight)

    def _forward_standard(self, x, edge_index=None, edge_weight=None):
        # Encode
        if self.encoder_type == "graph":
            enc_out = self.encoder(x, edge_index, edge_weight, self.use_residual)
        else:
            enc_out = self.encoder(x)

        q_z, q_m, q_s = enc_out[0], enc_out[1], enc_out[2]
        z_manifold = self._map_to_manifold(q_z)

        # Bottleneck (coupling)
        le = self.latent_encoder(q_z)
        ld = self.latent_decoder(le)

        if self.use_bottleneck_lorentz:
            ld_manifold = self._map_to_manifold(ld)
        else:
            n = enc_out[3]
            q_z2 = n.sample()
            ld_manifold = self._map_to_manifold(q_z2)

        # Decode
        pred_x, dropout_x = self._decode(q_z, edge_index, edge_weight)
        pred_xl, dropout_xl = self._decode(ld, edge_index, edge_weight)

        # Optional graph structure decoder
        pred_a = None
        if self.use_graph_decoder:
            pred_a, _, _ = self.structure_decoder(q_z, edge_index)

        # Optional PDE
        if self.use_pde:
            q_z_pde = self.pde_solver(q_z)
            pred_x_pde, dropout_x_pde = self._decode(q_z_pde, edge_index, edge_weight)
            return ForwardOutput(
                q_z=q_z, q_m=q_m, q_s=q_s, pred_x=pred_x, le=le, ld=ld,
                pred_xl=pred_xl, z_manifold=z_manifold, ld_manifold=ld_manifold,
                dropout_x=dropout_x, dropout_xl=dropout_xl, pred_a=pred_a,
                q_z_pde=q_z_pde, pred_x_pde=pred_x_pde, dropout_x_pde=dropout_x_pde,
            )

        return ForwardOutput(
            q_z=q_z, q_m=q_m, q_s=q_s, pred_x=pred_x, le=le, ld=ld,
            pred_xl=pred_xl, z_manifold=z_manifold, ld_manifold=ld_manifold,
            dropout_x=dropout_x, dropout_xl=dropout_xl, pred_a=pred_a,
        )

    def _forward_sde(self, x, edge_index=None, edge_weight=None):
        # Encode
        if self.encoder_type == "graph":
            enc_out = self.encoder(x, edge_index, edge_weight, self.use_residual)
        else:
            enc_out = self.encoder(x)

        q_z, q_m, q_s, n, t = enc_out

        # Sort by pseudotime
        idxs = t.argsort()
        t, q_z, q_m, q_s, x = t[idxs], q_z[idxs], q_m[idxs], q_s[idxs], x[idxs]

        # Remove duplicate time points
        if len(t) > 1:
            unique_mask = torch.cat([torch.tensor([True], device=t.device), t[1:] != t[:-1]])
            t, q_z, q_m, q_s, x = (
                t[unique_mask], q_z[unique_mask], q_m[unique_mask],
                q_s[unique_mask], x[unique_mask],
            )

        # Solve SDE
        z0 = q_z[0].unsqueeze(0)
        q_z_sde = self.solve_sde(
            self.sde_solver, z0, t,
            method=self.sde_solver_method,
            step_size=self.sde_step_size,
        ).squeeze(1)

        # Manifold mappings
        z_manifold = self._map_to_manifold(q_z)
        le = self.latent_encoder(q_z)
        ld = self.latent_decoder(le)

        if self.use_bottleneck_lorentz:
            ld_manifold = self._map_to_manifold(ld)
        else:
            q_z2 = n.sample()
            ld_manifold = self._map_to_manifold(q_z2)

        # Decode all paths
        pred_x, dropout_x = self._decode(q_z, edge_index, edge_weight)
        pred_xl, dropout_xl = self._decode(ld, edge_index, edge_weight)
        pred_x_sde, dropout_x_sde = self._decode(q_z_sde, edge_index, edge_weight)

        # Optional graph structure decoder
        pred_a = None
        if self.use_graph_decoder:
            pred_a, _, _ = self.structure_decoder(q_z, edge_index)

        # Optional PDE
        if self.use_pde:
            q_z_pde = self.pde_solver(q_z)
            pred_x_pde, dropout_x_pde = self._decode(q_z_pde, edge_index, edge_weight)
            return ForwardOutput(
                q_z=q_z, q_m=q_m, q_s=q_s, pred_x=pred_x, le=le, ld=ld,
                pred_xl=pred_xl, z_manifold=z_manifold, ld_manifold=ld_manifold,
                dropout_x=dropout_x, dropout_xl=dropout_xl, pred_a=pred_a,
                q_z_sde=q_z_sde, pred_x_sde=pred_x_sde, dropout_x_sde=dropout_x_sde,
                x_sorted=x, t=t,
                q_z_pde=q_z_pde, pred_x_pde=pred_x_pde, dropout_x_pde=dropout_x_pde,
            )

        return ForwardOutput(
            q_z=q_z, q_m=q_m, q_s=q_s, pred_x=pred_x, le=le, ld=ld,
            pred_xl=pred_xl, z_manifold=z_manifold, ld_manifold=ld_manifold,
            dropout_x=dropout_x, dropout_xl=dropout_xl, pred_a=pred_a,
            q_z_sde=q_z_sde, pred_x_sde=pred_x_sde, dropout_x_sde=dropout_x_sde,
            x_sorted=x, t=t,
        )
