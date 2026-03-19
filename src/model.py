# ============================================================================
# model.py - Core Model: Loss Computation, Optimization, Latent Extraction
# ============================================================================
"""
Unified HSDE + CCVGAE model combining:
- Multi-objective loss (recon, KL, geometric, SDE, PDE, graph adjacency)
- Count-based likelihoods (NB, ZINB, Poisson, ZIP)
- Support for MLP / Transformer / Graph encoders
- Gradient descent with optional graph structure learning
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Optional
from sklearn.metrics.pairwise import pairwise_distances
from .mixin import scviMixin, dipMixin, betatcMixin, infoMixin, adjMixin
from .module import VAE
from .utils import lorentz_distance


class HSDEModel(scviMixin, dipMixin, betatcMixin, infoMixin, adjMixin):
    """
    Core model merging HSDE + CCVGAE loss objectives.

    Supports all three encoder types (mlp, transformer, graph) and optional
    graph structure reconstruction loss from CCVGAE.
    """

    def __init__(
        self,
        recon, irecon, lorentz, beta, dip, tc, info,
        state_dim, hidden_dim, latent_dim, i_dim,
        lr, device,
        use_bottleneck_lorentz=True,
        loss_type="nb",
        grad_clip=1.0,
        use_layer_norm=True,
        use_euclidean_manifold=False,
        use_sde=False,
        use_pde=False,
        vae_reg=0.5,
        sde_reg=0.5,
        pde_reg=0.2,
        # Encoder selection
        encoder_type="mlp",
        # Transformer
        attn_embed_dim=64, attn_num_heads=4, attn_num_layers=2,
        attn_seq_len=8, attn_dropout=0.1,
        # SDE
        sde_strategy="scaled", sde_type="time_mlp", sde_time_cond="concat",
        sde_hidden_dim=None, sde_solver_method="euler", sde_step_size=None,
        # PDE
        pde_k=15, pde_alpha=0.1, pde_steps=2, pde_tau=1.0,
        # Graph (CCVGAE)
        graph_type="GAT",
        graph_hidden_layers=2,
        graph_dropout=0.05,
        graph_Cheb_k=1,
        graph_alpha=0.5,
        use_residual=True,
        use_graph_decoder=False,
        structure_decoder_type="mlp",
        decoder_hidden_dim=128,
        graph_threshold=0,
        graph_sparse_threshold=None,
        feature_decoder_type="mlp",
        # Graph loss weights (CCVGAE)
        w_adj=0.0,
        graph_loss_weight=1.0,
        **kwargs,
    ):
        self.recon = recon
        self.irecon = irecon
        self.lorentz = lorentz
        self.beta = beta
        self.dip = dip
        self.tc = tc
        self.info = info
        self.loss_type = loss_type
        self.grad_clip = grad_clip
        self.use_euclidean_manifold = use_euclidean_manifold
        self.use_sde = use_sde
        self.use_pde = use_pde
        self.vae_reg = vae_reg
        self.sde_reg = sde_reg
        self.pde_reg = pde_reg
        self.device = device
        self.encoder_type = encoder_type.lower()
        self.use_graph_decoder = use_graph_decoder
        self.w_adj = w_adj
        self.graph_loss_weight = graph_loss_weight

        self.nn = VAE(
            state_dim, hidden_dim, latent_dim, i_dim,
            use_bottleneck_lorentz=use_bottleneck_lorentz,
            loss_type=loss_type,
            use_layer_norm=use_layer_norm,
            use_euclidean_manifold=use_euclidean_manifold,
            use_sde=use_sde,
            use_pde=use_pde,
            device=device,
            encoder_type=encoder_type,
            attn_embed_dim=attn_embed_dim,
            attn_num_heads=attn_num_heads,
            attn_num_layers=attn_num_layers,
            attn_seq_len=attn_seq_len,
            attn_dropout=attn_dropout,
            sde_hidden_dim=sde_hidden_dim,
            sde_strategy=sde_strategy,
            sde_time_cond=sde_time_cond,
            sde_solver_method=sde_solver_method,
            sde_step_size=sde_step_size,
            pde_k=pde_k,
            pde_alpha=pde_alpha,
            pde_steps=pde_steps,
            pde_tau=pde_tau,
            graph_type=graph_type,
            graph_hidden_layers=graph_hidden_layers,
            graph_dropout=graph_dropout,
            graph_Cheb_k=graph_Cheb_k,
            graph_alpha=graph_alpha,
            use_residual=use_residual,
            use_graph_decoder=use_graph_decoder,
            structure_decoder_type=structure_decoder_type,
            decoder_hidden_dim=decoder_hidden_dim,
            graph_threshold=graph_threshold,
            graph_sparse_threshold=graph_sparse_threshold,
            feature_decoder_type=feature_decoder_type,
        )

        self.nn_optimizer = optim.Adam(self.nn.parameters(), lr=lr)
        self.loss = []

    # ========================================================================
    # Reconstruction loss
    # ========================================================================

    def _compute_reconstruction_loss(self, x_raw, pred_x, dropout_x):
        lib_size = torch.clamp(x_raw.sum(dim=-1, keepdim=True), min=1.0)
        pred_x = pred_x * lib_size

        if self.loss_type == "nb":
            disp = torch.exp(self.nn.decoder.disp)
            return -self._log_nb(x_raw, pred_x, disp).sum(dim=-1).mean()
        elif self.loss_type == "zinb":
            disp = torch.exp(self.nn.decoder.disp)
            return -self._log_zinb(x_raw, pred_x, disp, dropout_x).sum(dim=-1).mean()
        elif self.loss_type == "poisson":
            return -self._log_poisson(x_raw, pred_x).sum(dim=-1).mean()
        elif self.loss_type == "zip":
            return -self._log_zip(x_raw, pred_x, dropout_x).sum(dim=-1).mean()
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

    # ========================================================================
    # Adjacency loss (CCVGAE)
    # ========================================================================

    def _compute_adj_loss(self, pred_a, edge_index, num_nodes, edge_weight=None):
        """Binary cross-entropy adjacency reconstruction (from CCVGAE)."""
        if pred_a is None:
            return torch.tensor(0.0, device=self.device)
        adj = self._build_adj(edge_index, num_nodes, edge_weight).to_dense()
        return self.graph_loss_weight * F.binary_cross_entropy_with_logits(pred_a, adj)

    # ========================================================================
    # Latent extraction
    # ========================================================================

    @torch.no_grad()
    @torch.no_grad()
    def take_latent(self, state, edge_index=None, edge_weight=None):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        ei = torch.tensor(edge_index, dtype=torch.long).to(self.device) if edge_index is not None else None
        ew = torch.tensor(edge_weight, dtype=torch.float32).to(self.device) if edge_weight is not None else None

        if self.use_sde:
            if self.encoder_type == "graph":
                enc_out = self.nn.encoder(state, ei, ew, self.nn.use_residual)
            else:
                enc_out = self.nn.encoder(state)
            q_z, q_m, q_s, n, t = enc_out

            t_cpu = t.cpu().numpy()
            t_sorted, sort_idx, sort_idxr = np.unique(t_cpu, return_index=True, return_inverse=True)
            t_sorted = torch.tensor(t_sorted, dtype=torch.float32)
            q_z_sorted = q_z[sort_idx]
            z0 = q_z_sorted[0].unsqueeze(0)

            q_z_sde = self.nn.solve_sde(
                self.nn.sde_solver, z0, t_sorted,
                method=self.nn.sde_solver_method,
                step_size=self.nn.sde_step_size,
            ).squeeze(1)
            q_z_sde = q_z_sde[sort_idxr]

            combined = self.vae_reg * q_z + self.sde_reg * q_z_sde
            return combined.cpu().numpy()
        else:
            if self.encoder_type == "graph":
                enc_out = self.nn.encoder(state, ei, ew, self.nn.use_residual)
            else:
                enc_out = self.nn.encoder(state)
            q_z = enc_out[0]
            return q_z.cpu().numpy()

    @torch.no_grad()
    def take_centroid(self, state, edge_index=None, edge_weight=None):
        """Extract deterministic posterior mean (CCVGAE Centroid Inference)."""
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        ei = torch.tensor(edge_index, dtype=torch.long).to(self.device) if edge_index is not None else None
        ew = torch.tensor(edge_weight, dtype=torch.float32).to(self.device) if edge_weight is not None else None

        if self.encoder_type == "graph":
            enc_out = self.nn.encoder(state, ei, ew, self.nn.use_residual)
        else:
            enc_out = self.nn.encoder(state)
        q_m = enc_out[1]
        return q_m.cpu().numpy()

    @torch.no_grad()
    def take_time(self, state, edge_index=None, edge_weight=None):
        if not self.use_sde:
            raise ValueError("take_time() requires use_sde=True")
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        ei = torch.tensor(edge_index, dtype=torch.long).to(self.device) if edge_index is not None else None
        ew = torch.tensor(edge_weight, dtype=torch.float32).to(self.device) if edge_weight is not None else None

        if self.encoder_type == "graph":
            output = self.nn.encoder(state, ei, ew, self.nn.use_residual)
        else:
            output = self.nn.encoder(state)
        t = output[-1]
        return t.cpu().numpy()

    @torch.no_grad()
    def take_grad(self, state, edge_index=None, edge_weight=None):
        if not self.use_sde:
            raise ValueError("take_grad() requires use_sde=True")
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        ei = torch.tensor(edge_index, dtype=torch.long).to(self.device) if edge_index is not None else None
        ew = torch.tensor(edge_weight, dtype=torch.float32).to(self.device) if edge_weight is not None else None

        if self.encoder_type == "graph":
            enc_out = self.nn.encoder(state, ei, ew, self.nn.use_residual)
        else:
            enc_out = self.nn.encoder(state)
        q_z, _, _, _, t = enc_out
        drift = self.nn.sde_solver.f(t, q_z)
        return drift.cpu().numpy()

    @torch.no_grad()
    def take_transition(self, state, edge_index=None, edge_weight=None, top_k=30):
        if not self.use_sde:
            raise ValueError("take_transition() requires use_sde=True")
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        ei = torch.tensor(edge_index, dtype=torch.long).to(self.device) if edge_index is not None else None
        ew = torch.tensor(edge_weight, dtype=torch.float32).to(self.device) if edge_weight is not None else None

        if self.encoder_type == "graph":
            enc_out = self.nn.encoder(state, ei, ew, self.nn.use_residual)
        else:
            enc_out = self.nn.encoder(state)
        q_z, _, _, _, t = enc_out

        drift = self.nn.sde_solver.f(t, q_z).cpu().numpy()
        z_latent = q_z.cpu().numpy()
        z_future = z_latent + 1e-2 * drift

        distances = pairwise_distances(z_latent, z_future)
        sigma = np.median(distances)
        similarity = np.exp(-(distances ** 2) / (2 * sigma ** 2))
        transition_matrix = similarity / similarity.sum(axis=1, keepdims=True)

        n_cells = transition_matrix.shape[0]
        sparse_trans = np.zeros_like(transition_matrix)
        for i in range(n_cells):
            top_indices = np.argsort(transition_matrix[i])[::-1][:top_k]
            sparse_trans[i, top_indices] = transition_matrix[i, top_indices]
            sparse_trans[i] /= sparse_trans[i].sum()

        return sparse_trans

    @torch.no_grad()
    def take_pde_latent(self, state, edge_index=None, edge_weight=None):
        if not self.use_pde:
            raise ValueError("take_pde_latent() requires use_pde=True")
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        ei = torch.tensor(edge_index, dtype=torch.long).to(self.device) if edge_index is not None else None
        ew = torch.tensor(edge_weight, dtype=torch.float32).to(self.device) if edge_weight is not None else None

        if self.encoder_type == "graph":
            enc_out = self.nn.encoder(state, ei, ew, self.nn.use_residual)
        else:
            enc_out = self.nn.encoder(state)

        q_z = enc_out[0]
        q_z_pde = self.nn.pde_solver(q_z)
        return q_z_pde.cpu().numpy()

    # ========================================================================
    # Training update step
    # ========================================================================

    def update(self, states_norm, states_raw, edge_index=None, edge_weight=None):
        """One gradient descent step with full multi-objective loss."""
        states_norm = torch.tensor(states_norm, dtype=torch.float32).to(self.device)
        states_raw = torch.tensor(states_raw, dtype=torch.float32).to(self.device)
        ei = torch.tensor(edge_index, dtype=torch.long).to(self.device) if edge_index is not None else None
        ew = torch.tensor(edge_weight, dtype=torch.float32).to(self.device) if edge_weight is not None else None

        if torch.isnan(states_norm).any() or torch.isinf(states_norm).any():
            return

        outputs = self.nn(states_norm, ei, ew)

        # Unpack outputs based on mode
        if self.use_sde:
            if self.use_pde:
                (q_z, q_m, q_s, pred_x, le, ld, pred_xl, z_manifold, ld_manifold,
                 dropout_x, dropout_xl, q_z_sde, pred_x_sde, dropout_x_sde,
                 x_sorted, t, q_z_pde, pred_x_pde, dropout_x_pde, pred_a) = outputs
            else:
                (q_z, q_m, q_s, pred_x, le, ld, pred_xl, z_manifold, ld_manifold,
                 dropout_x, dropout_xl, q_z_sde, pred_x_sde, dropout_x_sde,
                 x_sorted, t, pred_a) = outputs

            qz_div = F.mse_loss(q_z, q_z_sde, reduction="none").sum(-1).mean()
            target_raw = x_sorted  # SDE reorders input

            recon_loss = self.recon * self._compute_reconstruction_loss(target_raw, pred_x, dropout_x)
            recon_loss += self.recon * self._compute_reconstruction_loss(target_raw, pred_x_sde, dropout_x_sde)

            if self.use_pde:
                qz_div += self.pde_reg * F.mse_loss(q_z, q_z_pde, reduction="none").sum(-1).mean()
                recon_loss += self.pde_reg * self.recon * self._compute_reconstruction_loss(
                    target_raw, pred_x_pde, dropout_x_pde
                )

            irecon_loss = torch.tensor(0.0, device=self.device)
            if self.irecon > 0:
                irecon_loss = self.irecon * self._compute_reconstruction_loss(target_raw, pred_xl, dropout_xl)
        else:
            if self.use_pde:
                (q_z, q_m, q_s, pred_x, le, ld, pred_xl, z_manifold, ld_manifold,
                 dropout_x, dropout_xl, q_z_pde, pred_x_pde, dropout_x_pde, pred_a) = outputs
            else:
                (q_z, q_m, q_s, pred_x, le, ld, pred_xl, z_manifold, ld_manifold,
                 dropout_x, dropout_xl, pred_a) = outputs

            qz_div = torch.tensor(0.0, device=self.device)
            target_raw = states_raw

            recon_loss = self.recon * self._compute_reconstruction_loss(target_raw, pred_x, dropout_x)

            if self.use_pde:
                qz_div += self.pde_reg * F.mse_loss(q_z, q_z_pde, reduction="none").sum(-1).mean()
                recon_loss += self.pde_reg * self.recon * self._compute_reconstruction_loss(
                    target_raw, pred_x_pde, dropout_x_pde
                )

            irecon_loss = torch.tensor(0.0, device=self.device)
            if self.irecon > 0:
                irecon_loss = self.irecon * self._compute_reconstruction_loss(target_raw, pred_xl, dropout_xl)

        # Geometric (manifold) loss
        geometric_loss = torch.tensor(0.0, device=self.device)
        if self.lorentz > 0:
            if not (torch.isnan(z_manifold).any() or torch.isnan(ld_manifold).any()):
                if self.use_euclidean_manifold:
                    from .utils import euclidean_distance
                    dist = euclidean_distance(z_manifold, ld_manifold)
                else:
                    dist = lorentz_distance(z_manifold, ld_manifold)
                if not torch.isnan(dist).any():
                    geometric_loss = self.lorentz * dist.mean()

        if torch.isnan(q_m).any() or torch.isnan(q_s).any():
            return

        # KL divergence
        kl_div = self.beta * self._normal_kl(
            q_m, q_s, torch.zeros_like(q_m), torch.zeros_like(q_s)
        ).sum(dim=-1).mean()

        # Additional regularizers
        dip_loss = self.dip * self._dip_loss(q_m, q_s) if self.dip > 0 else torch.tensor(0.0, device=self.device)
        tc_loss = self.tc * self._betatc_compute_total_correlation(q_z, q_m, q_s) if self.tc > 0 else torch.tensor(0.0, device=self.device)
        mmd_loss = self.info * self._compute_mmd(q_z, torch.randn_like(q_z)) if self.info > 0 else torch.tensor(0.0, device=self.device)

        # Graph adjacency loss (CCVGAE)
        adj_loss = torch.tensor(0.0, device=self.device)
        if self.use_graph_decoder and self.w_adj > 0 and pred_a is not None and ei is not None:
            adj_loss = self.w_adj * self._compute_adj_loss(pred_a, ei, states_norm.size(0), ew)

        total_loss = recon_loss + irecon_loss + geometric_loss + qz_div + kl_div + dip_loss + tc_loss + mmd_loss + adj_loss

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            return

        self.nn_optimizer.zero_grad()
        total_loss.backward()

        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.nn.parameters(), self.grad_clip)

        self.nn_optimizer.step()

        self.loss.append((
            total_loss.item(), recon_loss.item(), irecon_loss.item(),
            geometric_loss.item(), qz_div.item(), kl_div.item(),
            dip_loss.item(), tc_loss.item(), mmd_loss.item(), adj_loss.item(),
        ))
