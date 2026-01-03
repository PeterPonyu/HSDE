
"""Neural network modules for HSDE with SDE dynamics."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Optional, Tuple
from .utils import exp_map_at_origin
from .mixin import SDEMixin
from .sde_functions import create_controlled_sde


class Encoder(nn.Module):
    """Variational encoder: x → q(z|x) with optional time prediction."""
    
    def __init__(
        self, 
        state_dim: int, 
        hidden_dim: int, 
        action_dim: int, 
        use_layer_norm: bool = True, 
        use_sde: bool = False,
        encoder_type: str = 'mlp',
        attn_embed_dim: int = 64,
        attn_num_heads: int = 4,
        attn_num_layers: int = 2,
        attn_seq_len: int = 32,
    ):
        super().__init__()
        self.use_layer_norm = use_layer_norm
        self.use_sde = use_sde
        self.encoder_type = encoder_type.lower()
        
        if self.encoder_type == 'mlp':
            self.fc1 = nn.Linear(state_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, action_dim * 2)
            
            if use_layer_norm:
                self.ln1 = nn.LayerNorm(hidden_dim)
                self.ln2 = nn.LayerNorm(hidden_dim)
            
            if use_sde:
                self.time_encoder = nn.Sequential(
                    nn.Linear(hidden_dim, 1),
                    nn.Sigmoid()
                )
        else:
            self.attn_seq_len = attn_seq_len
            self.attn_embed_dim = attn_embed_dim
            self.input_proj = nn.Linear(state_dim, attn_seq_len * attn_embed_dim)
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=attn_embed_dim,
                nhead=attn_num_heads,
                dim_feedforward=max(attn_embed_dim * 4, 128),
                activation='relu',
                batch_first=False,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=attn_num_layers)
            self.attn_pool_fc = nn.Linear(attn_embed_dim, action_dim * 2)
            
            if use_layer_norm:
                self.attn_ln = nn.LayerNorm(attn_embed_dim)
            
            if use_sde:
                self.time_encoder = nn.Sequential(
                    nn.Linear(attn_embed_dim, 1),
                    nn.Sigmoid()
                )
        
        self.apply(lambda m: nn.init.xavier_normal_(m.weight) if isinstance(m, nn.Linear) else None)

    def forward(self, x: torch.Tensor):
        """Returns: (q_z, q_m, q_s, Normal, [t])"""
        if self.encoder_type == 'mlp':
            h = F.relu(self.ln1(self.fc1(x)) if self.use_layer_norm else self.fc1(x))
            h = F.relu(self.ln2(self.fc2(h)) if self.use_layer_norm else self.fc2(h))
            output = self.fc3(h)
            feature = h
        else:
            proj = self.input_proj(x).view(x.size(0), self.attn_seq_len, self.attn_embed_dim).transpose(0, 1)
            seq_out = self.transformer(proj).transpose(0, 1)
            
            if self.use_layer_norm:
                seq_out = self.attn_ln(seq_out)
            
            pooled = seq_out.mean(dim=1)
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


class Decoder(nn.Module):
    """Generative decoder: z → p(x|z) with count-based likelihoods."""
    
    def __init__(
        self, 
        state_dim: int, 
        hidden_dim: int, 
        action_dim: int, 
        loss_type: str = 'nb', 
        use_layer_norm: bool = True
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
        
        if loss_type in ['zinb', 'zip']:
            self.dropout = nn.Sequential(
                nn.Linear(action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, state_dim)
            )
        
        self.apply(lambda m: nn.init.xavier_normal_(m.weight) if isinstance(m, nn.Linear) else None)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Returns: (output, dropout_logits)"""
        h = F.relu(self.ln1(self.fc1(x)) if self.use_layer_norm else self.fc1(x))
        h = F.relu(self.ln2(self.fc2(h)) if self.use_layer_norm else self.fc2(h))
        output = F.softmax(self.fc3(h), dim=-1)
        dropout = self.dropout(x) if self.loss_type in ['zinb', 'zip'] else None
        return output, dropout

class VAE(nn.Module, SDEMixin):
    """
    HSDE: Hyperbolic SDE-VAE for single-cell trajectory inference.
    
    Combines VAE with:
    - Hyperbolic/Euclidean manifold regularization
    - Neural SDE for continuous dynamics (optional)
    - Count-based likelihoods (NB, ZINB, Poisson, ZIP)
    """
    
    def __init__(
        self, 
        state_dim: int, 
        hidden_dim: int, 
        action_dim: int, 
        i_dim: int,
        use_bottleneck_lorentz: bool = True, 
        loss_type: str = 'nb', 
        use_layer_norm: bool = True, 
        use_euclidean_manifold: bool = False, 
        use_sde: bool = False,
        device: torch.device = None,
        encoder_type: str = 'mlp',
        attn_embed_dim: int = 64,
        attn_num_heads: int = 4,
        attn_num_layers: int = 2,
        attn_seq_len: int = 32,
        sde_hidden_dim: Optional[int] = None,
        strategy: str = 'scaled',
        sde_time_cond: str = 'concat',
        sde_solver_method: str = 'euler',
        sde_step_size: Optional[float] = None,
        **kwargs
    ):
        super().__init__()
        
        self.encoder = Encoder(
            state_dim, hidden_dim, action_dim, use_layer_norm, use_sde,
            encoder_type, attn_embed_dim, attn_num_heads, attn_num_layers, attn_seq_len
        ).to(device)
        self.decoder = Decoder(state_dim, hidden_dim, action_dim, loss_type, use_layer_norm).to(device)
        self.latent_encoder = nn.Linear(action_dim, i_dim).to(device)
        self.latent_decoder = nn.Linear(i_dim, action_dim).to(device)
        
        self.use_bottleneck_lorentz = use_bottleneck_lorentz
        self.use_euclidean_manifold = use_euclidean_manifold
        self.use_sde = use_sde
        self.sde_solver_method = sde_solver_method
        self.sde_step_size = sde_step_size
        
        if use_sde:
            sde_n_hidden = sde_hidden_dim or hidden_dim
            self.sde_solver = create_controlled_sde(
                strategy=strategy,
                n_latent=action_dim,
                n_hidden=sde_n_hidden,
                time_cond=sde_time_cond
            ).to(device)
    
    def _map_to_manifold(self, z: torch.Tensor) -> torch.Tensor:
        """Map latent code to Lorentz/Euclidean manifold."""
        if self.use_euclidean_manifold:
            return z
        z_clipped = z.clamp(-5, 5)
        z_tangent = F.pad(z_clipped, (1, 0), value=0)
        return exp_map_at_origin(z_tangent)
    
    def forward(self, x: torch.Tensor):
        """Forward pass with optional SDE dynamics."""
        if self.use_sde:
            return self._forward_sde(x)
        return self._forward_standard(x)
    
    def _forward_standard(self, x: torch.Tensor) -> Tuple:
        """Standard VAE forward: x → z → x̂"""
        q_z, q_m, q_s, n = self.encoder(x)
        z_manifold = self._map_to_manifold(q_z)
        
        # Bottleneck path
        le = self.latent_encoder(q_z)
        ld = self.latent_decoder(le)
        
        if self.use_bottleneck_lorentz:
            ld_manifold = self._map_to_manifold(ld)
        else:
            q_z2 = n.sample()
            ld_manifold = self._map_to_manifold(q_z2)
        
        pred_x, dropout_x = self.decoder(q_z)
        pred_xl, dropout_xl = self.decoder(ld)
        
        return q_z, q_m, q_s, pred_x, le, ld, pred_xl, z_manifold, ld_manifold, dropout_x, dropout_xl
    
    def _forward_sde(self, x: torch.Tensor) -> Tuple:
        """SDE-augmented forward: x → z → SDE trajectory → x̂"""
        q_z, q_m, q_s, n, t = self.encoder(x)
        
        # Sort by pseudotime
        idxs = t.argsort()
        t, q_z, q_m, q_s, x = t[idxs], q_z[idxs], q_m[idxs], q_s[idxs], x[idxs]
        
        # Remove duplicate time points
        if len(t) > 1:
            unique_mask = torch.cat([torch.tensor([True], device=t.device), t[1:] != t[:-1]])
            t, q_z, q_m, q_s, x = t[unique_mask], q_z[unique_mask], q_m[unique_mask], q_s[unique_mask], x[unique_mask]
        
        # Solve SDE: z(t₀) → z(t)
        z0 = q_z[0].unsqueeze(0)
        q_z_sde = self.solve_sde(
            self.sde_solver, z0, t,
            method=self.sde_solver_method,
            step_size=self.sde_step_size
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
        pred_x, dropout_x = self.decoder(q_z)
        pred_xl, dropout_xl = self.decoder(ld)
        pred_x_sde, dropout_x_sde = self.decoder(q_z_sde)
        
        return (q_z, q_m, q_s, pred_x, le, ld, pred_xl, z_manifold, ld_manifold,
                dropout_x, dropout_xl, q_z_sde, pred_x_sde, dropout_x_sde, x, t)
