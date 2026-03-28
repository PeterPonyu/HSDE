# ============================================================================
# environment.py - Data Loading, Preprocessing, and Training Loop
# ============================================================================
"""
Environment for HSDE data handling:
- Raw count preprocessing with adaptive normalization
- DataLoader-based batch training for MLP/Transformer encoders
- Validation with clustering metrics and early stopping
"""

import logging
import os

from .model import HSDEModel
from .mixin import envMixin
import numpy as np
from scipy.sparse import issparse
from sklearn.cluster import KMeans
import torch
from torch.utils.data import DataLoader, TensorDataset


# Limit CPU thread over-subscription: without this, each process spawns
# threads equal to the total CPU count (e.g. 24), causing severe contention
# when multiple experiments run in parallel.
_MAX_THREADS = min(os.cpu_count() or 4, 8)
torch.set_num_threads(_MAX_THREADS)
os.environ.setdefault("OMP_NUM_THREADS", str(_MAX_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(_MAX_THREADS))

logger = logging.getLogger(__name__)


def is_raw_counts(X, threshold=0.5):
    """Heuristically determine if data contains raw integer counts."""
    if issparse(X):
        sample_data = X.data[:min(10000, len(X.data))]
    else:
        flat_data = X.flatten()
        sample_data = flat_data[np.random.choice(len(flat_data), min(10000, len(flat_data)), replace=False)]

    sample_data = sample_data[sample_data > 0]
    if len(sample_data) == 0:
        return False
    if np.mean((sample_data > 0) & (sample_data < 1)) > 0.1:
        return False
    if np.any(sample_data < 0):
        return False

    integer_like = np.abs(sample_data - np.round(sample_data)) < 1e-6
    return np.mean(integer_like) >= threshold


def compute_dataset_stats(X):
    X_dense = X.toarray() if issparse(X) else X
    return {
        "sparsity": np.mean(X_dense == 0),
        "lib_size_mean": X_dense.sum(axis=1).mean(),
        "lib_size_std": X_dense.sum(axis=1).std(),
        "max_val": X_dense.max(),
    }


class Env(HSDEModel, envMixin):
    """
    Environment supporting batch-based (MLP/Transformer) encoder training.

    Handles data registration, preprocessing, train/val/test splitting,
    and the training loop with early stopping.
    """

    def __init__(
        self,
        adata, layer, recon, irecon, lorentz, beta, dip, tc, info,
        hidden_dim, latent_dim, i_dim, lr,
        use_bottleneck_lorentz, loss_type, device,
        grad_clip=1.0, adaptive_norm=True, use_layer_norm=True,
        use_euclidean_manifold=False, use_sde=False, use_pde=False,
        vae_reg=0.5, sde_reg=0.5, pde_reg=0.2,
        train_size=0.7, val_size=0.15, test_size=0.15,
        batch_size=128, random_seed=42,
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
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.loss_type = loss_type
        self.adaptive_norm = adaptive_norm
        self.encoder_type = encoder_type.lower()
        self._init_device = device

        # Register data
        self._register_anndata(adata, layer, latent_dim)

        super().__init__(
            recon=recon, irecon=irecon, lorentz=lorentz, beta=beta,
            dip=dip, tc=tc, info=info,
            state_dim=self.n_var, hidden_dim=hidden_dim,
            latent_dim=latent_dim, i_dim=i_dim, lr=lr,
            use_bottleneck_lorentz=use_bottleneck_lorentz,
            loss_type=loss_type, device=device,
            grad_clip=grad_clip, use_layer_norm=use_layer_norm,
            use_euclidean_manifold=use_euclidean_manifold,
            use_sde=use_sde, use_pde=use_pde,
            vae_reg=vae_reg, sde_reg=sde_reg, pde_reg=pde_reg,
            encoder_type=encoder_type,
            attn_embed_dim=attn_embed_dim, attn_num_heads=attn_num_heads,
            attn_num_layers=attn_num_layers, attn_seq_len=attn_seq_len,
            attn_dropout=attn_dropout,
            sde_strategy=sde_strategy, sde_type=sde_type,
            sde_time_cond=sde_time_cond, sde_hidden_dim=sde_hidden_dim,
            sde_solver_method=sde_solver_method, sde_step_size=sde_step_size,
            pde_k=pde_k, pde_alpha=pde_alpha, pde_steps=pde_steps, pde_tau=pde_tau,
            **kwargs,
        )

        self.train_losses = []
        self.val_losses = []
        self.val_scores = []
        self.best_val_loss = float("inf")
        self.best_model_state = None
        self.patience_counter = 0

    # ========================================================================
    # Data Registration
    # ========================================================================

    def _register_anndata(self, adata, layer, latent_dim):
        X = adata.layers[layer]
        if not is_raw_counts(X):
            raise ValueError(f"Layer '{layer}' does not contain raw counts.")

        X = X.toarray() if issparse(X) else np.asarray(X)
        X_raw = X.astype(np.float32)

        stats = compute_dataset_stats(X)
        logger.info("Dataset statistics:")
        logger.info(f"  Cells: {X.shape[0]:,}, Genes: {X.shape[1]:,}")
        logger.info(f"  Sparsity: {stats['sparsity']:.2%}, "
              f"Lib size: {stats['lib_size_mean']:.0f}±{stats['lib_size_std']:.0f}")

        X_log = np.log1p(X)

        if self.adaptive_norm:
            if stats["sparsity"] > 0.95:
                X_norm = np.clip(X_log, -5, 5).astype(np.float32)
            elif stats["lib_size_std"] / stats["lib_size_mean"] > 2.0:
                cell_means = X_log.mean(axis=1, keepdims=True)
                cell_stds = X_log.std(axis=1, keepdims=True) + 1e-6
                X_norm = np.clip((X_log - cell_means) / cell_stds, -10, 10).astype(np.float32)
            elif stats["max_val"] > 10000:
                scale = min(1.0, 10.0 / X_log.max())
                X_norm = np.clip(X_log * scale, -10, 10).astype(np.float32)
            else:
                X_norm = np.clip(X_log, -10, 10).astype(np.float32)
        else:
            X_norm = np.clip(X_log, -10, 10).astype(np.float32)

        self.n_obs, self.n_var = adata.shape

        try:
            self.labels = KMeans(
                n_clusters=min(latent_dim, self.n_obs - 1),
                n_init=10, random_state=self.random_seed,
            ).fit_predict(X_norm)
        except Exception:
            self.labels = np.random.default_rng(self.random_seed).integers(
                0, latent_dim, size=self.n_obs
            )

        rng = np.random.default_rng(self.random_seed)
        indices = rng.permutation(self.n_obs)
        n_train = int(self.train_size * self.n_obs)
        n_val = int(self.val_size * self.n_obs)

        self.train_idx = indices[:n_train]
        self.val_idx = indices[n_train:n_train + n_val]
        self.test_idx = indices[n_train + n_val:]

        self.X_train_norm = X_norm[self.train_idx]
        self.X_train_raw = X_raw[self.train_idx]
        self.X_val_norm = X_norm[self.val_idx]
        self.X_val_raw = X_raw[self.val_idx]
        self.X_test_norm = X_norm[self.test_idx]
        self.X_test_raw = X_raw[self.test_idx]
        self.X_norm = X_norm
        self.X_raw = X_raw

        self.labels_train = self.labels[self.train_idx]
        self.labels_val = self.labels[self.val_idx]
        self.labels_test = self.labels[self.test_idx]

        self._create_dataloaders()

    # ========================================================================
    # DataLoaders
    # ========================================================================

    def _create_dataloaders(self):
        train_ds = TensorDataset(torch.FloatTensor(self.X_train_norm), torch.FloatTensor(self.X_train_raw))
        val_ds = TensorDataset(torch.FloatTensor(self.X_val_norm), torch.FloatTensor(self.X_val_raw))
        test_ds = TensorDataset(torch.FloatTensor(self.X_test_norm), torch.FloatTensor(self.X_test_raw))

        _dev = getattr(self, 'device', getattr(self, '_init_device', 'cpu'))
        use_cuda = _dev.type == "cuda" if hasattr(_dev, "type") else ("cuda" in str(_dev))
        loader_kw = dict(
            pin_memory=use_cuda,
            num_workers=2 if use_cuda else 0,
            persistent_workers=True if use_cuda else False,
        )
        self.train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, drop_last=True, **loader_kw)
        self.val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, drop_last=False, **loader_kw)
        self.test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False, drop_last=False, **loader_kw)

    # ========================================================================
    # Training
    # ========================================================================

    def train_epoch(self):
        """One training epoch with mini-batch gradient descent."""
        self.nn.train()
        epoch_losses = []

        for batch_norm, batch_raw in self.train_loader:
            self.update(batch_norm, batch_raw)
            if len(self.loss) > 0:
                epoch_losses.append(self.loss[-1][0])

        avg = np.mean(epoch_losses) if epoch_losses else 0.0
        self.train_losses.append(avg)
        return avg

    def validate(self):
        """Evaluate on validation set with clustering metrics."""
        self.nn.eval()
        all_latents = []

        with torch.no_grad():
            for batch_norm, _ in self.val_loader:
                latent = self.take_latent(batch_norm)
                all_latents.append(latent)

        all_latents = np.concatenate(all_latents, axis=0)
        val_score = self._calc_score_with_labels(all_latents, self.labels_val)
        self.val_scores.append(val_score)

        # Approximate val loss
        avg_val_loss = -val_score[2]  # Negative silhouette as proxy
        self.val_losses.append(avg_val_loss)
        return avg_val_loss, val_score

    def validate_loss(self):
        """Fast validation: use recent training loss trend as early stopping signal."""
        if len(self.train_losses) < 2:
            val_loss = self.train_losses[-1] if self.train_losses else 0.0
        else:
            window = min(5, len(self.train_losses))
            recent = self.train_losses[-window:]
            alpha = 0.4
            ema = recent[0]
            for v in recent[1:]:
                ema = alpha * v + (1 - alpha) * ema
            val_loss = ema

        self.val_losses.append(val_loss)
        return val_loss

    def check_early_stopping(self, val_loss, patience=25):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_model_state = {k: v.cpu().clone() for k, v in self.nn.state_dict().items()}
            self.patience_counter = 0
            return False, True
        else:
            self.patience_counter += 1
            return self.patience_counter >= patience, False

    def load_best_model(self):
        if self.best_model_state is not None:
            self.nn.load_state_dict(self.best_model_state)
            logger.info(f"Loaded best model (val_loss={self.best_val_loss:.4f})")
