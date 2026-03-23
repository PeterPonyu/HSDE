# ============================================================================
# environment.py - Data Loading, Preprocessing, and Training Loop
# ============================================================================
"""
Unified environment merging HSDE and CCVGAE data handling:
- HSDE-style: raw count preprocessing with adaptive normalization, DataLoader
- CCVGAE-style: graph construction via scanpy, subgraph sampling
- Supports both MLP/Transformer (batch-based) and Graph (graph-based) training
"""

import logging

from .model import HSDEModel
from .mixin import envMixin, scMixin
import numpy as np
from scipy.sparse import issparse
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional

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


class Env(HSDEModel, envMixin, scMixin):
    """
    Unified environment supporting both batch-based (MLP/Transformer) and
    graph-based (GAT/GCN/etc.) encoder training.

    For graph encoders, constructs a cell-cell graph from the data and
    supports subgraph sampling for scalability (CCVGAE-style).
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
        # Encoder/Decoder selection
        encoder_type="mlp",
        feature_decoder_type="mlp",
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
        w_adj=0.0,
        graph_loss_weight=1.0,
        # Graph data construction
        n_neighbors=15,
        n_var=None,
        tech="PCA",
        batch_tech=None,
        all_feat=False,
        subgraph_size=512,
        num_subgraphs_per_epoch=10,
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

        # Graph-specific storage
        self.edge_index = None
        self.edge_weight = None
        self.n_neighbors = n_neighbors
        self.subgraph_size = subgraph_size
        self.num_subgraphs_per_epoch = num_subgraphs_per_epoch

        # Register data
        if self.encoder_type == "graph":
            self._register_anndata_graph(
                adata, layer, latent_dim, n_var, tech, n_neighbors, batch_tech, all_feat
            )
        else:
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
            feature_decoder_type=feature_decoder_type,
            attn_embed_dim=attn_embed_dim, attn_num_heads=attn_num_heads,
            attn_num_layers=attn_num_layers, attn_seq_len=attn_seq_len,
            attn_dropout=attn_dropout,
            sde_strategy=sde_strategy, sde_type=sde_type,
            sde_time_cond=sde_time_cond, sde_hidden_dim=sde_hidden_dim,
            sde_solver_method=sde_solver_method, sde_step_size=sde_step_size,
            pde_k=pde_k, pde_alpha=pde_alpha, pde_steps=pde_steps, pde_tau=pde_tau,
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
            w_adj=w_adj,
            graph_loss_weight=graph_loss_weight,
            **kwargs,
        )

        self.train_losses = []
        self.val_losses = []
        self.val_scores = []
        self.best_val_loss = float("inf")
        self.best_model_state = None
        self.patience_counter = 0

    # ========================================================================
    # Data Registration (HSDE-style for MLP/Transformer)
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
    # Data Registration (CCVGAE-style for Graph encoder)
    # ========================================================================

    def _register_anndata_graph(self, adata, layer, latent_dim, n_var, tech, n_neighbors, batch_tech, all_feat):
        """CCVGAE-style preprocessing: normalize, HVG, decompose, build graph."""
        import scanpy as sc

        # Preprocessing
        self._preprocess(adata, layer, n_var)
        self._decomposition(adata, tech, latent_dim)

        if batch_tech:
            self._batchcorrect(adata, batch_tech, tech, layer)

        if batch_tech == "harmony":
            use_rep = f"X_harmony_{tech}"
        elif batch_tech == "scvi":
            use_rep = "X_scvi"
        else:
            use_rep = f"X_{tech}"

        sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=use_rep)

        # Extract features
        if all_feat:
            X = adata.layers[layer]
            X = X.toarray() if issparse(X) else np.asarray(X)
            self.X_norm = np.log1p(X).astype(np.float32)
        else:
            X = adata[:, adata.var["highly_variable"]].X
            X = X.toarray() if issparse(X) else np.asarray(X)
            self.X_norm = X.astype(np.float32)

        self.n_obs, self.n_var = self.X_norm.shape

        # Raw counts for reconstruction
        X_raw = adata.layers[layer]
        X_raw = X_raw.toarray() if issparse(X_raw) else np.asarray(X_raw)
        if all_feat:
            self.X_raw = X_raw.astype(np.float32)
        else:
            # Use HVG subset of raw counts
            hvg_mask = adata.var["highly_variable"].values
            self.X_raw = X_raw[:, hvg_mask].astype(np.float32)

        # Labels — Leiden on the neighbor graph (unsupervised, no cell_type)
        _leiden_key = '_hsde_val_leiden'
        sc.tl.leiden(adata, resolution=1.0, key_added=_leiden_key)
        self.labels = LabelEncoder().fit_transform(adata.obs[_leiden_key].values)

        # Graph connectivity
        coo = adata.obsp["connectivities"].tocoo()
        self.edge_index = np.array([coo.row, coo.col])
        self.edge_weight = coo.data.astype(np.float32)

        # Simple splits (full graph always available)
        self.y = np.arange(self.n_obs)
        self.idx = np.arange(self.n_obs)

        rng = np.random.default_rng(self.random_seed)
        indices = rng.permutation(self.n_obs)
        n_train = int(self.train_size * self.n_obs)
        n_val = int(self.val_size * self.n_obs)
        self.train_idx = indices[:n_train]
        self.val_idx = indices[n_train:n_train + n_val]
        self.test_idx = indices[n_train + n_val:]

        self.X_train_norm = self.X_norm[self.train_idx]
        self.X_train_raw = self.X_raw[self.train_idx]
        self.X_val_norm = self.X_norm[self.val_idx]
        self.X_val_raw = self.X_raw[self.val_idx]
        self.X_test_norm = self.X_norm[self.test_idx]
        self.X_test_raw = self.X_raw[self.test_idx]

        self.labels_train = self.labels[self.train_idx]
        self.labels_val = self.labels[self.val_idx]
        self.labels_test = self.labels[self.test_idx]

        logger.info(f"Graph constructed: {self.n_obs} nodes, {len(coo.data)} edges")
        logger.info(f"Data split: Train={len(self.train_idx)}, Val={len(self.val_idx)}, Test={len(self.test_idx)}")

    # ========================================================================
    # DataLoaders (for MLP/Transformer)
    # ========================================================================

    def _create_dataloaders(self):
        train_ds = TensorDataset(torch.FloatTensor(self.X_train_norm), torch.FloatTensor(self.X_train_raw))
        val_ds = TensorDataset(torch.FloatTensor(self.X_val_norm), torch.FloatTensor(self.X_val_raw))
        test_ds = TensorDataset(torch.FloatTensor(self.X_test_norm), torch.FloatTensor(self.X_test_raw))

        self.train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, drop_last=False)
        self.test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False, drop_last=False)

    # ========================================================================
    # Training
    # ========================================================================

    def train_epoch(self):
        """One training epoch (batch-based for MLP/Transformer, full-graph for Graph)."""
        self.nn.train()
        epoch_losses = []

        if self.encoder_type == "graph":
            # Full-graph training pass
            self.update(self.X_norm, self.X_raw, self.edge_index, self.edge_weight)
            if len(self.loss) > 0:
                epoch_losses.append(self.loss[-1][0])
        else:
            # Mini-batch training
            for batch_norm, batch_raw in self.train_loader:
                batch_norm = batch_norm.to(self.device)
                batch_raw = batch_raw.to(self.device)
                self.update(batch_norm.cpu().numpy(), batch_raw.cpu().numpy())
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
            if self.encoder_type == "graph":
                # Graph encoder needs full graph; extract val indices afterwards
                full_latent = self.take_latent(self.X_norm, self.edge_index, self.edge_weight)
                all_latents.append(full_latent[self.val_idx])
            else:
                for batch_norm, _ in self.val_loader:
                    latent = self.take_latent(batch_norm.numpy())
                    all_latents.append(latent)

        all_latents = np.concatenate(all_latents, axis=0)
        val_score = self._calc_score_with_labels(all_latents, self.labels_val)
        self.val_scores.append(val_score)

        # Approximate val loss
        avg_val_loss = -val_score[2]  # Negative silhouette as proxy
        self.val_losses.append(avg_val_loss)
        return avg_val_loss, val_score

    def validate_loss(self):
        """Fast validation: use recent training loss trend as early stopping signal.

        Avoids the expensive clustering metric computation while still
        providing a reasonable signal for early stopping. Uses an exponential
        moving average of the training loss.
        """
        if len(self.train_losses) < 2:
            val_loss = self.train_losses[-1] if self.train_losses else 0.0
        else:
            # Smoothed training loss (EMA over last 5 epochs)
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
