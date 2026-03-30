# ============================================================================
# model.py - Core Model: Loss Computation, Optimization, Latent Extraction
# ============================================================================
"""
HSDE model combining:
- Multi-objective loss (recon, KL, geometric, SDE, PDE)
- Count-based likelihoods (NB, ZINB, Poisson, ZIP)
- Support for MLP / Transformer encoders
- Gradient descent with mixed precision
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import warnings
from typing import Optional
from sklearn.metrics.pairwise import pairwise_distances
from .mixin import scviMixin, dipMixin, betatcMixin, infoMixin
from .module import VAE
from .utils import lorentz_distance


class HSDEModel(scviMixin, dipMixin, betatcMixin, infoMixin):
    """
    Core model with multi-objective loss.

    Supports MLP and Transformer encoder types with optional
    SDE trajectory inference and PDE latent diffusion.
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
        self._nan_skip_count = 0

        # Architectural ablation: derive from loss weights.
        # Bottleneck is only built when irecon > 0 (used) or lorentz > 0
        # with use_bottleneck_lorentz (manifold measured on bottleneck).
        use_bottleneck = (irecon > 0) or (lorentz > 0 and use_bottleneck_lorentz)
        # Manifold module is only needed when lorentz > 0.
        use_manifold = (lorentz > 0)

        self.nn = VAE(
            state_dim, hidden_dim, latent_dim, i_dim,
            use_bottleneck_lorentz=use_bottleneck_lorentz,
            loss_type=loss_type,
            use_layer_norm=use_layer_norm,
            use_euclidean_manifold=use_euclidean_manifold,
            use_sde=use_sde,
            use_pde=use_pde,
            device=device,
            use_bottleneck=use_bottleneck,
            use_manifold=use_manifold,
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
        )

        # AMP: enable mixed precision on CUDA for ~2x throughput
        self._use_amp = (device is not None and "cuda" in str(device))

        # Fused Adam: runs the entire optimizer step in a single CUDA kernel
        self.nn_optimizer = optim.Adam(
            self.nn.parameters(), lr=lr, fused=self._use_amp,
        )
        self.loss = []

        self._scaler = torch.amp.GradScaler("cuda", enabled=self._use_amp)

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
    # Latent extraction
    # ========================================================================

    @torch.no_grad()
    def take_latent(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state, dtype=torch.float32)
        state = state.to(self.device, non_blocking=True)

        if self.use_sde:
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
            enc_out = self.nn.encoder(state)
            q_z = enc_out[0]
            return q_z.cpu().numpy()

    @torch.no_grad()
    def take_time(self, state):
        if not self.use_sde:
            raise ValueError("take_time() requires use_sde=True")
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        output = self.nn.encoder(state)
        t = output[-1]
        return t.cpu().numpy()

    @torch.no_grad()
    def take_grad(self, state):
        if not self.use_sde:
            raise ValueError("take_grad() requires use_sde=True")
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        enc_out = self.nn.encoder(state)
        q_z, _, _, _, t = enc_out
        drift = self.nn.sde_solver.f(t, q_z)
        return drift.cpu().numpy()

    @torch.no_grad()
    def take_transition(self, state, top_k=30):
        if not self.use_sde:
            raise ValueError("take_transition() requires use_sde=True")
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
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
    def take_pde_latent(self, state):
        if not self.use_pde:
            raise ValueError("take_pde_latent() requires use_pde=True")
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        enc_out = self.nn.encoder(state)
        q_z = enc_out[0]
        q_z_pde = self.nn.pde_solver(q_z)
        return q_z_pde.cpu().numpy()

    # ========================================================================
    # NaN escalation
    # ========================================================================

    _NAN_SKIP_LIMIT = 50

    def _nan_escalate(self, reason: str):
        """Track consecutive NaN skips and escalate if threshold is exceeded."""
        self._nan_skip_count += 1
        warnings.warn(
            f"Skipping update ({self._nan_skip_count}/{self._NAN_SKIP_LIMIT}): {reason}",
            RuntimeWarning,
        )
        if self._nan_skip_count >= self._NAN_SKIP_LIMIT:
            raise RuntimeError(
                f"Training diverged: {self._nan_skip_count} consecutive NaN/Inf updates. "
                "Consider lowering learning rate or checking input data."
            )

    # ========================================================================
    # Training update step
    # ========================================================================

    def update(self, states_norm, states_raw):
        """One gradient descent step with full multi-objective loss."""
        if not isinstance(states_norm, torch.Tensor):
            states_norm = torch.as_tensor(states_norm, dtype=torch.float32)
        states_norm = states_norm.to(self.device, non_blocking=True)
        if not isinstance(states_raw, torch.Tensor):
            states_raw = torch.as_tensor(states_raw, dtype=torch.float32)
        states_raw = states_raw.to(self.device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=self._use_amp):
            outputs = self.nn(states_norm, x_raw=states_raw)

            # Access outputs via named attributes (ForwardOutput dataclass)
            q_z, q_m, q_s = outputs.q_z, outputs.q_m, outputs.q_s
            pred_x, dropout_x = outputs.pred_x, outputs.dropout_x
            le, ld, pred_xl, dropout_xl = outputs.le, outputs.ld, outputs.pred_xl, outputs.dropout_xl
            z_manifold, ld_manifold = outputs.z_manifold, outputs.ld_manifold

            if self.use_sde:
                qz_div = F.mse_loss(q_z, outputs.q_z_sde, reduction="none").sum(-1).mean()
                # Use the properly reordered raw counts for SDE reconstruction
                target_raw = outputs.x_raw_sorted if outputs.x_raw_sorted is not None else outputs.x_sorted

                recon_loss = self.recon * self._compute_reconstruction_loss(target_raw, pred_x, dropout_x)
                recon_loss += self.recon * self._compute_reconstruction_loss(
                    target_raw, outputs.pred_x_sde, outputs.dropout_x_sde
                )
            else:
                qz_div = torch.tensor(0.0, device=self.device)
                target_raw = states_raw

                recon_loss = self.recon * self._compute_reconstruction_loss(target_raw, pred_x, dropout_x)

            if self.use_pde:
                qz_div += self.pde_reg * F.mse_loss(q_z, outputs.q_z_pde, reduction="none").sum(-1).mean()
                recon_loss += self.pde_reg * self.recon * self._compute_reconstruction_loss(
                    target_raw, outputs.pred_x_pde, outputs.dropout_x_pde
                )

            # Bottleneck reconstruction loss — only when bottleneck is active
            irecon_loss = torch.tensor(0.0, device=self.device)
            if self.irecon > 0 and self.nn.use_bottleneck:
                irecon_loss = self.irecon * self._compute_reconstruction_loss(target_raw, pred_xl, dropout_xl)

            # Geometric (manifold) loss — only when manifold is active
            geometric_loss = torch.tensor(0.0, device=self.device)
            if self.lorentz > 0 and self.nn.use_manifold:
                if self.use_euclidean_manifold:
                    from .utils import euclidean_distance
                    dist = euclidean_distance(z_manifold, ld_manifold)
                else:
                    dist = lorentz_distance(z_manifold, ld_manifold)
                geometric_loss = self.lorentz * dist.mean()

            # KL divergence
            kl_div = self.beta * self._normal_kl(
                q_m, q_s, torch.zeros_like(q_m), torch.zeros_like(q_s)
            ).sum(dim=-1).mean()

            # Additional regularizers
            dip_loss = self.dip * self._dip_loss(q_m, q_s) if self.dip > 0 else torch.tensor(0.0, device=self.device)
            tc_loss = self.tc * self._betatc_compute_total_correlation(q_z, q_m, q_s) if self.tc > 0 else torch.tensor(0.0, device=self.device)
            mmd_loss = self.info * self._compute_mmd(q_z, torch.randn_like(q_z)) if self.info > 0 else torch.tensor(0.0, device=self.device)

            total_loss = recon_loss + irecon_loss + geometric_loss + qz_div + kl_div + dip_loss + tc_loss + mmd_loss

        # Single GPU→CPU sync: get total_loss value, then check NaN/Inf on CPU
        total_val = total_loss.item()
        if math.isnan(total_val) or math.isinf(total_val):
            status = "NaN" if math.isnan(total_val) else "Inf"
            self._nan_escalate(f"total_loss is {status}")
            return

        self.nn_optimizer.zero_grad()
        self._scaler.scale(total_loss).backward()

        if self.grad_clip is not None:
            self._scaler.unscale_(self.nn_optimizer)
            torch.nn.utils.clip_grad_norm_(self.nn.parameters(), self.grad_clip)

        self._scaler.step(self.nn_optimizer)
        self._scaler.update()
        self._nan_skip_count = 0  # Reset on successful step

        self.loss.append((
            total_val, recon_loss.item(), irecon_loss.item(),
            geometric_loss.item(), qz_div.item(), kl_div.item(),
            dip_loss.item(), tc_loss.item(), mmd_loss.item(),
        ))
