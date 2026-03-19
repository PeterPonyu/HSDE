"""
HSDE: Hyperbolic SDE-Regularized VAE
=======================================================================

A unified deep learning framework for single-cell omics analysis combining:
- Variational Autoencoder (VAE) for dimensionality reduction
- Lorentz geometric regularization for hierarchical structure
- Dual-path information bottleneck for coordinated biological programs
- Neural SDE regularization for trajectory inference
- Transformer-based attention mechanisms for long-range dependencies
- Multiple count-based likelihood functions (NB, ZINB, Poisson, ZIP)
- Graph neural network encoders (GAT, GCN, ChebConv, SAGE, etc.)
- Graph structure decoders for adjacency learning
- Subgraph-aware training for scalability

Supports scRNA-seq and scATAC-seq modalities without architectural modification.
"""

# ============================================================================
# agent.py - Main User Interface
# ============================================================================
from .vectorfield import VectorFieldMixin
from .environment import Env
from anndata import AnnData
from typing import Optional
import logging
import torch
import tqdm
import time
import numpy as np

logger = logging.getLogger(__name__)


class HSDE(Env, VectorFieldMixin):
    """
    HSDE: Hyperbolic SDE-Regularized VAE

    A unified framework for single-cell omics analysis (scRNA-seq and scATAC-seq)
    that learns low-dimensional representations while preserving both local
    cell-state structure and global hierarchical organization through Lorentz
    geometric regularization, information bottleneck architecture, neural
    SDE-based trajectory inference, and optional graph neural network encoders.

    Architecture Overview
    ---------------------
    1. **Encoder**: MLP, Transformer, or Graph (GAT/GCN/etc.) backbone
    2. **Information Bottleneck**: Optional compression layer for hierarchical features
    3. **Manifold Regularization**: Lorentz or Euclidean distance constraints
    4. **Feature Decoder**: MLP or Graph decoder for expression reconstruction
    5. **Structure Decoder** (optional): Adjacency reconstruction (CCVGAE-style)
    6. **SDE Solver** (optional): Stochastic cell state trajectory modeling
    7. **PDE Solver** (optional): Graph Laplacian latent diffusion

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with raw count data in ``adata.layers[layer]``
    layer : str, default='counts'
        Layer name containing raw unnormalized count data
    recon : float, default=1.0
        Weight for reconstruction loss (primary objective)
    irecon : float, default=0.0
        Weight for information bottleneck reconstruction loss
    lorentz : float, default=0.0
        Weight for geometric manifold regularization
    beta : float, default=1.0
        Weight for KL divergence (beta-VAE); >1 encourages disentanglement
    dip : float, default=0.0
        Weight for Disentangled Inferred Prior (DIP-VAE) loss
    tc : float, default=0.0
        Weight for Total Correlation (beta-TC-VAE) loss
    info : float, default=0.0
        Weight for Maximum Mean Discrepancy (InfoVAE) loss
    hidden_dim : int, default=128
        Hidden layer dimension in encoder/decoder
    latent_dim : int, default=10
        Primary latent space dimensionality
    i_dim : int, default=2
        Information bottleneck dimension (should be < latent_dim)
    lr : float, default=1e-4
        Learning rate for Adam optimizer
    use_bottleneck_lorentz : bool, default=True
        If True, compute manifold distance on bottleneck; else on resampled latents
    loss_type : str, default='nb'
        Count likelihood model: 'nb', 'zinb', 'poisson', or 'zip'
    grad_clip : float, default=1.0
        Gradient clipping threshold for training stability
    adaptive_norm : bool, default=True
        Use dataset-specific normalization heuristics
    use_layer_norm : bool, default=True
        Apply layer normalization in encoder/decoder
    use_euclidean_manifold : bool, default=False
        Use Euclidean distance instead of Lorentz (hyperbolic) distance
    use_sde : bool, default=False
        Enable Neural SDE regularization for trajectory inference
    use_pde : bool, default=False
        Enable graph PDE latent diffusion regularization
    vae_reg : float, default=0.5
        Weight for VAE path in SDE mode
    sde_reg : float, default=0.5
        Weight for SDE path in SDE mode
    pde_reg : float, default=0.2
        Weight for PDE reconstruction/consistency regularization
    train_size : float, default=0.7
        Proportion of cells for training set
    val_size : float, default=0.15
        Proportion of cells for validation set
    test_size : float, default=0.15
        Proportion of cells for test set
    batch_size : int, default=128
        Mini-batch size for stochastic gradient descent
    random_seed : int, default=42
        Random seed for reproducibility
    device : torch.device, optional
        Computation device (auto-detects CUDA if available)

    Encoder Selection
    -----------------
    encoder_type : str, default='mlp'
        Encoder backbone:
        - 'mlp': Standard MLP with LayerNorm (default)
        - 'transformer': Multi-head attention encoder
        - 'graph': Graph neural network encoder (requires torch_geometric)

    Transformer Parameters (encoder_type='transformer')
    ---------------------------------------------------
    attn_embed_dim : int, default=64
    attn_num_heads : int, default=4
    attn_num_layers : int, default=2
    attn_seq_len : int, default=32

    SDE Parameters (use_sde=True)
    ----------------------------
    sde_strategy : str, default='scaled'
        SDE strategy: 'scaled', 'constant', 'annealed', 'clipped'
    sde_type : str, default='time_mlp'
        SDE function type: 'time_mlp' or 'gru'
    sde_time_cond : str, default='concat'
        Time conditioning: 'concat', 'film', or 'add'
    sde_hidden_dim : int, optional
    sde_solver_method : str, default='euler'
    sde_step_size : float, optional

    PDE Parameters (use_pde=True)
    ----------------------------
    pde_k : int, default=15
    pde_alpha : float, default=0.1
    pde_steps : int, default=2
    pde_tau : float, default=1.0

    Graph Parameters (encoder_type='graph')
    ----------------------------------------
    graph_type : str, default='GAT'
        Graph convolution type: 'GAT', 'GCN', 'Cheb', 'SAGE', 'SSG',
        'Transformer', 'GIN', 'Linear', 'GeneralConv', 'FILM'
    graph_hidden_layers : int, default=2
    graph_dropout : float, default=0.05
    graph_Cheb_k : int, default=1
    graph_alpha : float, default=0.5
    use_residual : bool, default=True
    use_graph_decoder : bool, default=False
        Enable adjacency reconstruction (structure decoder)
    structure_decoder_type : str, default='mlp'
        Structure decoder type: 'bilinear', 'inner_product', 'mlp'
    decoder_hidden_dim : int, default=128
    graph_threshold : float, default=0
    graph_sparse_threshold : float, optional
    w_adj : float, default=0.0
        Weight for adjacency BCE loss
    graph_loss_weight : float, default=1.0
    n_neighbors : int, default=15
        KNN neighbors for constructing cell-cell graph
    n_var : int, optional
        Number of highly variable genes to select
    tech : str, default='PCA'
        Decomposition method: 'PCA', 'NMF', 'ICA', 'FA'
    batch_tech : str, optional
        Batch correction method: 'harmony', 'scvi', or None
    all_feat : bool, default=False
        Whether to use all features or only HVGs
    subgraph_size : int, default=512
        Subgraph size for scalable graph training
    num_subgraphs_per_epoch : int, default=10
        Subgraphs sampled per epoch

    Examples
    --------
    >>> import scanpy as sc
    >>> from hsde import HSDE
    >>>
    >>> # Load data
    >>> adata = sc.read_h5ad('data.h5ad')
    >>>
    >>> # Standard MLP encoder
    >>> model = HSDE(adata, layer='counts')
    >>> model.fit(epochs=100)
    >>> latent = model.get_latent()
    >>>
    >>> # With Lorentz regularization and SDE trajectory inference
    >>> model = HSDE(adata, lorentz=5.0, use_sde=True, latent_dim=10, i_dim=2)
    >>> model.fit(epochs=400, patience=25)
    >>>
    >>> # Graph encoder (requires torch_geometric)
    >>> model = HSDE(
    ...     adata, encoder_type='graph', graph_type='GAT',
    ...     use_graph_decoder=True, w_adj=0.1
    ... )
    >>> model.fit(epochs=200)
    >>> centroids = model.get_centroid()  # CCVGAE-style centroid inference
    """

    def __init__(
        self,
        adata: AnnData,
        layer: str = "counts",
        recon: float = 1.0,
        irecon: float = 0.0,
        lorentz: float = 0.0,
        beta: float = 1.0,
        dip: float = 0.0,
        tc: float = 0.0,
        info: float = 0.0,
        hidden_dim: int = 128,
        latent_dim: int = 10,
        i_dim: int = 2,
        lr: float = 1e-4,
        use_bottleneck_lorentz: bool = True,
        loss_type: str = "nb",
        grad_clip: float = 1.0,
        adaptive_norm: bool = True,
        use_layer_norm: bool = True,
        use_euclidean_manifold: bool = False,
        use_sde: bool = False,
        use_pde: bool = False,
        vae_reg: float = 0.5,
        sde_reg: float = 0.5,
        pde_reg: float = 0.2,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        batch_size: int = 128,
        random_seed: int = 42,
        device: torch.device = None,
        # Encoder/Decoder selection
        encoder_type: str = "mlp",
        feature_decoder_type: str = "mlp",
        # Transformer
        attn_embed_dim: int = 64,
        attn_num_heads: int = 4,
        attn_num_layers: int = 2,
        attn_seq_len: int = 32,
        attn_dropout: float = 0.1,
        # SDE
        sde_strategy: str = "scaled",
        sde_type: str = "time_mlp",
        sde_time_cond: str = "concat",
        sde_hidden_dim: Optional[int] = None,
        sde_solver_method: str = "euler",
        sde_step_size: Optional[float] = None,
        # PDE
        pde_k: int = 15,
        pde_alpha: float = 0.1,
        pde_steps: int = 2,
        pde_tau: float = 1.0,
        # Graph (CCVGAE)
        graph_type: str = "GAT",
        graph_hidden_layers: int = 2,
        graph_dropout: float = 0.05,
        graph_Cheb_k: int = 1,
        graph_alpha: float = 0.5,
        use_residual: bool = True,
        use_graph_decoder: bool = False,
        structure_decoder_type: str = "mlp",
        decoder_hidden_dim: int = 128,
        graph_threshold: float = 0,
        graph_sparse_threshold: Optional[float] = None,
        w_adj: float = 0.0,
        graph_loss_weight: float = 1.0,
        # Graph data construction
        n_neighbors: int = 15,
        n_var: Optional[int] = None,
        tech: str = "PCA",
        batch_tech: Optional[str] = None,
        all_feat: bool = False,
        subgraph_size: int = 512,
        num_subgraphs_per_epoch: int = 10,
    ):
        # Auto-detect device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Validate parameters
        if not (0.99 <= train_size + val_size + test_size <= 1.01):
            raise ValueError(
                f"Split sizes must sum to 1.0, got {train_size + val_size + test_size}"
            )

        if train_size < 0 or val_size < 0 or test_size < 0:
            raise ValueError("Split sizes must be non-negative")

        if use_sde and not (0.99 <= vae_reg + sde_reg <= 1.01):
            raise ValueError(
                f"SDE weights must sum to 1.0, got {vae_reg + sde_reg}"
            )

        if i_dim >= latent_dim:
            raise ValueError(
                f"Information bottleneck dimension ({i_dim}) must be < latent dimension ({latent_dim})"
            )

        # Validate enum-like parameters
        _VALID_LOSS = {"nb", "zinb", "poisson", "zip"}
        _VALID_ENCODER = {"mlp", "transformer", "graph"}
        if loss_type not in _VALID_LOSS:
            raise ValueError(f"loss_type={loss_type!r} not in {_VALID_LOSS}")
        if encoder_type not in _VALID_ENCODER:
            raise ValueError(f"encoder_type={encoder_type!r} not in {_VALID_ENCODER}")

        # Validate positive numeric parameters
        for name, val in [("hidden_dim", hidden_dim), ("latent_dim", latent_dim),
                          ("batch_size", batch_size), ("lr", lr), ("epochs", epochs)]:
            if val <= 0:
                raise ValueError(f"{name} must be positive, got {val}")

        # Initialize parent environment
        super().__init__(
            adata=adata,
            layer=layer,
            recon=recon,
            irecon=irecon,
            lorentz=lorentz,
            beta=beta,
            dip=dip,
            tc=tc,
            info=info,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            i_dim=i_dim,
            lr=lr,
            use_bottleneck_lorentz=use_bottleneck_lorentz,
            loss_type=loss_type,
            grad_clip=grad_clip,
            adaptive_norm=adaptive_norm,
            use_layer_norm=use_layer_norm,
            use_euclidean_manifold=use_euclidean_manifold,
            use_sde=use_sde,
            use_pde=use_pde,
            vae_reg=vae_reg,
            sde_reg=sde_reg,
            pde_reg=pde_reg,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            batch_size=batch_size,
            random_seed=random_seed,
            device=device,
            encoder_type=encoder_type,
            feature_decoder_type=feature_decoder_type,
            attn_embed_dim=attn_embed_dim,
            attn_num_heads=attn_num_heads,
            attn_num_layers=attn_num_layers,
            attn_seq_len=attn_seq_len,
            attn_dropout=attn_dropout,
            sde_strategy=sde_strategy,
            sde_type=sde_type,
            sde_time_cond=sde_time_cond,
            sde_hidden_dim=sde_hidden_dim,
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
            w_adj=w_adj,
            graph_loss_weight=graph_loss_weight,
            n_neighbors=n_neighbors,
            n_var=n_var,
            tech=tech,
            batch_tech=batch_tech,
            all_feat=all_feat,
            subgraph_size=subgraph_size,
            num_subgraphs_per_epoch=num_subgraphs_per_epoch,
        )

        # Resource tracking
        self.train_time = 0.0
        self.peak_memory_gb = 0.0
        self.actual_epochs = 0

    # ====================================================================
    # Training
    # ====================================================================

    def fit(
        self,
        epochs: int = 400,
        patience: int = 25,
        val_every: int = 5,
        early_stop: bool = True,
        compute_metrics: bool = True,
    ):
        """
        Train the HSDE model.

        Parameters
        ----------
        epochs : int, default=400
            Maximum number of training epochs
        patience : int, default=25
            Early stopping patience
        val_every : int, default=5
            Validation frequency (every N epochs)
        early_stop : bool, default=True
            Enable early stopping mechanism
        compute_metrics : bool, default=True
            If True, compute clustering metrics (ARI, NMI, etc.) during
            validation. Set to False for faster training when intermediate
            metrics are not needed — early stopping will use reconstruction
            loss on the validation set instead.

        Returns
        -------
        self : HSDE
        """
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.reset_peak_memory_stats()
        start_time = time.time()

        with tqdm.tqdm(total=epochs, desc="Training", ncols=200) as pbar:
            for epoch in range(epochs):
                train_loss = self.train_epoch()

                if (epoch + 1) % val_every == 0 or epoch == 0:
                    if compute_metrics:
                        val_loss, val_score = self.validate()
                    else:
                        val_loss = self.validate_loss()
                        val_score = None

                    if early_stop:
                        should_stop, improved = self.check_early_stopping(
                            val_loss, patience
                        )

                        postfix = {
                            "Train": f"{train_loss:.2f}",
                            "Val": f"{val_loss:.2f}",
                            "Best": f"{self.best_val_loss:.2f}",
                            "Pat": f"{self.patience_counter}/{patience}",
                            "Imp": "Y" if improved else "N",
                        }
                        if val_score is not None:
                            postfix.update({
                                "ARI": f"{val_score[0]:.2f}",
                                "NMI": f"{val_score[1]:.2f}",
                                "ASW": f"{val_score[2]:.2f}",
                                "CAL": f"{val_score[3]:.2f}",
                                "DAV": f"{val_score[4]:.2f}",
                                "COR": f"{val_score[5]:.2f}",
                            })
                        pbar.set_postfix(postfix)

                        if should_stop:
                            self.actual_epochs = epoch + 1
                            logger.info("Early stopping at epoch %d", epoch + 1)
                            logger.info("Best validation loss: %.4f", self.best_val_loss)
                            self.load_best_model()
                            break
                    else:
                        postfix = {
                            "Train": f"{train_loss:.2f}",
                            "Val": f"{val_loss:.2f}",
                        }
                        if val_score is not None:
                            postfix.update({
                                "ARI": f"{val_score[0]:.2f}",
                                "NMI": f"{val_score[1]:.2f}",
                                "ASW": f"{val_score[2]:.2f}",
                                "CAL": f"{val_score[3]:.2f}",
                                "DAV": f"{val_score[4]:.2f}",
                                "COR": f"{val_score[5]:.2f}",
                            })
                        pbar.set_postfix(postfix)

                pbar.update(1)
            else:
                self.actual_epochs = epochs

        self.train_time = time.time() - start_time
        self.peak_memory_gb = (
            torch.cuda.max_memory_allocated() / 1e9 if use_cuda else 0.0
        )
        return self

    # ====================================================================
    # Inference
    # ====================================================================

    def get_latent(self):
        """
        Extract latent representations for all cells.

        Returns
        -------
        latent : ndarray of shape (n_cells, latent_dim)
        """
        return self.take_latent(
            self.X_norm,
            edge_index=self.edge_index,
            edge_weight=self.edge_weight,
        )

    def get_centroid(self):
        """
        Extract centroid (deterministic mean) for all cells (graph encoder only).

        This returns the q_m output from the graph encoder, which is the
        centroid-based representation from CCVGAE's coupling mechanism.

        Returns
        -------
        centroid : ndarray of shape (n_cells, latent_dim)
        """
        return self.take_centroid(
            self.X_norm,
            edge_index=self.edge_index,
            edge_weight=self.edge_weight,
        )

    def get_time(self):
        """
        Extract pseudotime for all cells (SDE mode only).

        Returns
        -------
        pseudotime : ndarray of shape (n_cells,)
        """
        return self.take_time(
            self.X_norm,
            edge_index=self.edge_index,
            edge_weight=self.edge_weight,
        )

    def get_transition(self):
        """
        Extract transition probabilities for all cells (SDE mode only).

        Returns
        -------
        transition : ndarray of shape (n_cells, n_cells)
        """
        return self.take_transition(
            self.X_norm,
            edge_index=self.edge_index,
            edge_weight=self.edge_weight,
        )

    def get_test_latent(self):
        """
        Extract latent representations for test set only.

        Returns
        -------
        latent : ndarray of shape (n_test, latent_dim)
        """
        return self.take_latent(
            self.X_test_norm,
            edge_index=self.edge_index,
            edge_weight=self.edge_weight,
        )

    def get_pde_latent(self):
        """
        Extract PDE-smoothed latent representations (PDE mode only).

        Returns
        -------
        latent_pde : ndarray of shape (n_cells, latent_dim)
        """
        return self.take_pde_latent(
            self.X_norm,
            edge_index=self.edge_index,
            edge_weight=self.edge_weight,
        )

    def get_bottleneck(self):
        """
        Extract information bottleneck representations.

        Returns
        -------
        bottleneck : ndarray of shape (n_cells, i_dim)
        """
        x = torch.tensor(self.X_norm, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            if self.encoder_type == "graph":
                ei = torch.LongTensor(self.edge_index).to(self.device) if self.edge_index is not None else None
                ew = torch.FloatTensor(self.edge_weight).to(self.device) if self.edge_weight is not None else None
                outputs = self.nn(x, edge_index=ei, edge_weight=ew)
            else:
                outputs = self.nn(x)
            le = outputs.le  # Information bottleneck encoding
        return le.cpu().numpy()

    def get_resource_metrics(self):
        """
        Get training resource usage metrics.

        Returns
        -------
        metrics : dict
            Dictionary with 'train_time', 'peak_memory_gb', 'actual_epochs'
        """
        return {
            "train_time": self.train_time,
            "peak_memory_gb": self.peak_memory_gb,
            "actual_epochs": self.actual_epochs,
        }

    def __repr__(self):
        n_params = sum(p.numel() for p in self.nn.parameters())
        parts = [
            f"HSDE(encoder={self.encoder_type!r}",
            f"latent_dim={self.nn.latent_dim}",
            f"n_cells={self.n_obs}",
            f"n_genes={self.n_var}",
            f"params={n_params:,}",
        ]
        if self.encoder_type == "graph":
            parts.append(f"graph={self.nn.encoder.conv_type}")
        if self.nn.use_sde:
            parts.append("sde=True")
        if self.nn.use_pde:
            parts.append("pde=True")
        return ", ".join(parts) + ")"

    def summary_dict(self):
        """Return a dictionary of the model configuration.

        Returns
        -------
        config : dict
            Dictionary with model configuration keys.
        """
        n_params = sum(p.numel() for p in self.nn.parameters())
        d = {
            "encoder_type": self.encoder_type,
            "parameters": n_params,
            "latent_dim": self.nn.latent_dim,
            "bottleneck_dim": self.nn.i_dim,
            "input_dim": self.n_var,
            "n_cells": self.n_obs,
            "loss_type": self.loss_type,
            "device": str(self.device),
            "use_sde": self.nn.use_sde,
            "use_pde": self.nn.use_pde,
        }
        if self.encoder_type == "graph":
            d["graph_type"] = self.nn.encoder.conv_type
            d["graph_decoder"] = self.nn.use_graph_decoder
            d["n_edges"] = len(self.edge_weight) if self.edge_weight is not None else 0
        return d

    def summary(self):
        """Print a summary of the model configuration."""
        d = self.summary_dict()
        logger.info("=" * 60)
        logger.info("HSDE Model Summary")
        logger.info("=" * 60)
        for key, val in d.items():
            label = key.replace("_", " ").title()
            logger.info("  %-18s %s", label, f"{val:,}" if isinstance(val, int) and key == "parameters" else val)
        logger.info("=" * 60)
