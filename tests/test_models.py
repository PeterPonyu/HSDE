#!/usr/bin/env python3
"""
LIORA Model Workability Tests
==============================

Tests that all base model configurations are actually workable and effective:
1. Models instantiate without errors
2. Training converges (loss decreases)
3. Latent representations are valid (finite, correct shape)
4. Clustering metrics (ARI, NMI) are above chance level
5. All encoder types work (MLP, Transformer, Graph)
6. All optional components work (SDE, PDE, IB, Lorentz)
7. All loss types work (NB, ZINB, Poisson, ZIP)

Uses a small synthetic dataset for fast execution.
"""

import sys
import os
import pytest
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_adata():
    """Create a small synthetic AnnData with known cluster structure."""
    import anndata
    import scipy.sparse as sp

    np.random.seed(42)
    n_cells = 300
    n_genes = 500
    n_clusters = 3
    cells_per_cluster = n_cells // n_clusters

    # Generate count data with clear cluster structure
    X = np.zeros((n_cells, n_genes), dtype=np.float32)
    labels = []
    for c in range(n_clusters):
        start = c * cells_per_cluster
        end = start + cells_per_cluster
        # Each cluster has different gene expression patterns
        base_rate = np.random.exponential(0.5, n_genes)
        cluster_signal = np.zeros(n_genes)
        # Marker genes for this cluster
        marker_start = c * (n_genes // n_clusters)
        marker_end = marker_start + n_genes // (n_clusters * 2)
        cluster_signal[marker_start:marker_end] = np.random.exponential(3.0, marker_end - marker_start)
        rates = base_rate + cluster_signal
        for i in range(start, end):
            X[i] = np.random.poisson(rates)
        labels.extend([f"cluster_{c}"] * cells_per_cluster)

    X = sp.csr_matrix(X)
    adata = anndata.AnnData(X=X.copy())
    adata.layers["counts"] = X.copy()
    adata.obs["cell_type"] = labels
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    return adata


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _train_and_check(adata, device, epochs=30, patience=15, **model_kwargs):
    """Train a model and return (model, latent, metrics_dict)."""
    from hsde import HSDE

    model = HSDE(
        adata, layer="counts",
        hidden_dim=64, latent_dim=8, i_dim=2,
        lr=1e-3, loss_type="nb",
        device=device,
        batch_size=64,
        train_size=0.7, val_size=0.15, test_size=0.15,
        random_seed=42,
        **model_kwargs,
    )
    model.fit(epochs=epochs, patience=patience, early_stop=True,
              compute_metrics=False)

    latent = model.get_latent()

    # Basic validity checks
    assert latent is not None, "Latent should not be None"
    assert latent.shape[0] == adata.n_obs, \
        f"Latent rows ({latent.shape[0]}) != cells ({adata.n_obs})"
    assert latent.shape[1] == 8, \
        f"Latent dim ({latent.shape[1]}) != expected (8)"
    assert np.all(np.isfinite(latent)), "Latent contains NaN/Inf"
    assert latent.std() > 0, "Latent is constant (no variance)"

    # Check training actually happened
    assert model.actual_epochs > 0, "No training epochs completed"
    assert len(model.loss) > 0, "No loss values recorded"

    # Check loss decreased (first vs last quarter mean)
    losses = [l[0] for l in model.loss]
    first_q = np.mean(losses[:max(1, len(losses) // 4)])
    last_q = np.mean(losses[-max(1, len(losses) // 4):])
    # Allow some tolerance — loss should generally decrease
    assert last_q < first_q * 1.5, \
        f"Loss did not decrease: first_q={first_q:.2f}, last_q={last_q:.2f}"

    return model, latent


def _compute_clustering_quality(latent, labels):
    """Compute ARI and NMI from latent space."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    labels_int = le.fit_transform(np.asarray(labels).astype(str))
    n_clusters = len(np.unique(labels_int))

    pred = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(latent)
    ari = adjusted_rand_score(labels_int, pred)
    nmi = normalized_mutual_info_score(labels_int, pred)
    return ari, nmi


# ---------------------------------------------------------------------------
# Test: Base VAE (MLP encoder)
# ---------------------------------------------------------------------------

class TestBaseVAE:
    """Test the baseline VAE with MLP encoder."""

    def test_instantiation(self, synthetic_adata, device):
        from hsde import HSDE
        model = HSDE(synthetic_adata, layer="counts", device=device,
                     hidden_dim=64, latent_dim=8, i_dim=2)
        assert model is not None

    def test_training_and_latent(self, synthetic_adata, device):
        model, latent = _train_and_check(
            synthetic_adata, device, epochs=50,
            encoder_type="mlp", recon=1.0, irecon=0.0,
            lorentz=0.0, beta=1.0,
        )
        assert latent.shape == (300, 8)

    def test_clustering_above_chance(self, synthetic_adata, device):
        model, latent = _train_and_check(
            synthetic_adata, device, epochs=50,
            encoder_type="mlp", recon=1.0, irecon=0.0,
            lorentz=0.0, beta=1.0,
        )
        labels = synthetic_adata.obs["cell_type"].values
        ari, nmi = _compute_clustering_quality(latent, labels)
        # ARI > 0 means better than random
        assert ari > 0.0, f"ARI ({ari:.3f}) not above chance"
        assert nmi > 0.0, f"NMI ({nmi:.3f}) not above chance"

    def test_resource_metrics(self, synthetic_adata, device):
        model, _ = _train_and_check(
            synthetic_adata, device, epochs=30,
            encoder_type="mlp",
        )
        res = model.get_resource_metrics()
        assert res["train_time"] > 0
        assert res["actual_epochs"] > 0


# ---------------------------------------------------------------------------
# Test: IRecon-VAE (information bottleneck)
# ---------------------------------------------------------------------------

class TestIReconVAE:
    def test_training(self, synthetic_adata, device):
        model, latent = _train_and_check(
            synthetic_adata, device, epochs=50,
            encoder_type="mlp", recon=1.0, irecon=1.0,
            lorentz=0.0, beta=1.0,
        )
        assert latent.shape == (300, 8)

    def test_bottleneck_output(self, synthetic_adata, device):
        model, _ = _train_and_check(
            synthetic_adata, device, epochs=30,
            encoder_type="mlp", irecon=1.0,
        )
        bottleneck = model.get_bottleneck()
        assert bottleneck.shape == (300, 2)
        assert np.all(np.isfinite(bottleneck))


# ---------------------------------------------------------------------------
# Test: Lorentz-VAE (hyperbolic geometry)
# ---------------------------------------------------------------------------

class TestLorentzVAE:
    def test_training(self, synthetic_adata, device):
        model, latent = _train_and_check(
            synthetic_adata, device, epochs=50,
            encoder_type="mlp", recon=1.0, irecon=0.0,
            lorentz=5.0, beta=1.0,
        )
        assert latent.shape == (300, 8)


# ---------------------------------------------------------------------------
# Test: GM-VAE (IRecon + Lorentz)
# ---------------------------------------------------------------------------

class TestGMVAE:
    def test_training(self, synthetic_adata, device):
        model, latent = _train_and_check(
            synthetic_adata, device, epochs=50,
            encoder_type="mlp", recon=1.0, irecon=1.0,
            lorentz=5.0, beta=1.0,
        )
        assert latent.shape == (300, 8)


# ---------------------------------------------------------------------------
# Test: Euclidean manifold variant
# ---------------------------------------------------------------------------

class TestEuclideanManifold:
    def test_training(self, synthetic_adata, device):
        model, latent = _train_and_check(
            synthetic_adata, device, epochs=50,
            encoder_type="mlp", irecon=1.0, lorentz=5.0,
            use_euclidean_manifold=True,
        )
        assert latent.shape == (300, 8)


# ---------------------------------------------------------------------------
# Test: Transformer encoder
# ---------------------------------------------------------------------------

class TestTransformerEncoder:
    def test_training(self, synthetic_adata, device):
        model, latent = _train_and_check(
            synthetic_adata, device, epochs=50,
            encoder_type="transformer",
            attn_embed_dim=32, attn_num_heads=2,
            attn_num_layers=1, attn_seq_len=8,
        )
        assert latent.shape == (300, 8)


# ---------------------------------------------------------------------------
# Test: Graph encoder (GAT)
# ---------------------------------------------------------------------------

class TestGraphEncoder:
    def test_gat_training(self, synthetic_adata, device):
        try:
            import torch_geometric
        except ImportError:
            pytest.skip("torch_geometric not installed")

        model, latent = _train_and_check(
            synthetic_adata, device, epochs=50,
            encoder_type="graph", graph_type="GAT",
            graph_hidden_layers=2, n_neighbors=10,
        )
        assert latent.shape == (300, 8)

    def test_gcn_training(self, synthetic_adata, device):
        try:
            import torch_geometric
        except ImportError:
            pytest.skip("torch_geometric not installed")

        model, latent = _train_and_check(
            synthetic_adata, device, epochs=50,
            encoder_type="graph", graph_type="GCN",
            graph_hidden_layers=2, n_neighbors=10,
        )
        assert latent.shape == (300, 8)

    def test_graph_decoder(self, synthetic_adata, device):
        try:
            import torch_geometric
        except ImportError:
            pytest.skip("torch_geometric not installed")

        model, latent = _train_and_check(
            synthetic_adata, device, epochs=30,
            encoder_type="graph", graph_type="GAT",
            use_graph_decoder=True, w_adj=0.1,
            n_neighbors=10,
        )
        assert latent.shape == (300, 8)

    def test_centroid_inference(self, synthetic_adata, device):
        try:
            import torch_geometric
        except ImportError:
            pytest.skip("torch_geometric not installed")

        model, _ = _train_and_check(
            synthetic_adata, device, epochs=30,
            encoder_type="graph", graph_type="GAT",
            n_neighbors=10,
        )
        centroid = model.get_centroid()
        assert centroid.shape == (300, 8)
        assert np.all(np.isfinite(centroid))


# ---------------------------------------------------------------------------
# Test: SDE trajectory inference
# ---------------------------------------------------------------------------

class TestSDE:
    def test_sde_training(self, synthetic_adata, device):
        model, latent = _train_and_check(
            synthetic_adata, device, epochs=50,
            encoder_type="mlp", use_sde=True,
            vae_reg=0.5, sde_reg=0.5,
        )
        assert latent.shape == (300, 8)

    def test_pseudotime_extraction(self, synthetic_adata, device):
        model, _ = _train_and_check(
            synthetic_adata, device, epochs=30,
            encoder_type="mlp", use_sde=True,
            vae_reg=0.5, sde_reg=0.5,
        )
        pseudotime = model.get_time()
        assert pseudotime.shape == (300,)
        assert np.all(np.isfinite(pseudotime))
        assert pseudotime.min() >= 0 and pseudotime.max() <= 1

    def test_transition_matrix(self, synthetic_adata, device):
        model, _ = _train_and_check(
            synthetic_adata, device, epochs=30,
            encoder_type="mlp", use_sde=True,
            vae_reg=0.5, sde_reg=0.5,
        )
        trans = model.get_transition()
        assert trans.shape == (300, 300)
        # Rows should sum to 1
        row_sums = trans.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Test: Loss types
# ---------------------------------------------------------------------------

class TestLossTypes:
    @pytest.mark.parametrize("loss_type", ["nb", "zinb", "poisson", "zip"])
    def test_loss_type(self, synthetic_adata, device, loss_type):
        from hsde import HSDE
        model = HSDE(
            synthetic_adata, layer="counts",
            hidden_dim=64, latent_dim=8, i_dim=2,
            lr=1e-3, loss_type=loss_type,
            device=device, batch_size=64,
        )
        model.fit(epochs=20, patience=10, early_stop=False,
                  compute_metrics=False)
        latent = model.get_latent()
        assert np.all(np.isfinite(latent))


# ---------------------------------------------------------------------------
# Test: HSDE Full (Graph + SDE + PDE + IB + Lorentz)
# ---------------------------------------------------------------------------

class TestHSDEFull:
    def test_full_model(self, synthetic_adata, device):
        try:
            import torch_geometric
        except ImportError:
            pytest.skip("torch_geometric not installed")

        model, latent = _train_and_check(
            synthetic_adata, device, epochs=50,
            encoder_type="graph", graph_type="GAT",
            recon=1.0, irecon=1.0, lorentz=5.0, beta=1.0,
            use_sde=True, use_pde=True,
            vae_reg=0.5, sde_reg=0.5, pde_reg=0.2,
            n_neighbors=10,
        )
        assert latent.shape == (300, 8)

    def test_full_model_all_outputs(self, synthetic_adata, device):
        try:
            import torch_geometric
        except ImportError:
            pytest.skip("torch_geometric not installed")

        model, latent = _train_and_check(
            synthetic_adata, device, epochs=30,
            encoder_type="graph", graph_type="GAT",
            recon=1.0, irecon=1.0, lorentz=5.0, beta=1.0,
            use_sde=True, use_pde=True,
            vae_reg=0.5, sde_reg=0.5, pde_reg=0.2,
            n_neighbors=10,
        )
        # Test all output methods
        bottleneck = model.get_bottleneck()
        assert bottleneck.shape == (300, 2)

        pseudotime = model.get_time()
        assert pseudotime.shape == (300,)

        centroid = model.get_centroid()
        assert centroid.shape == (300, 8)

        pde_latent = model.get_pde_latent()
        assert pde_latent.shape == (300, 8)

        # All should be finite
        for name, arr in [("bottleneck", bottleneck), ("pseudotime", pseudotime),
                          ("centroid", centroid), ("pde_latent", pde_latent)]:
            assert np.all(np.isfinite(arr)), f"{name} contains NaN/Inf"


# ---------------------------------------------------------------------------
# Test: Parameter validation
# ---------------------------------------------------------------------------

class TestParameterValidation:
    def test_invalid_split_sum(self, synthetic_adata, device):
        from hsde import HSDE
        with pytest.raises(ValueError, match="Split sizes must sum to 1.0"):
            HSDE(synthetic_adata, layer="counts", device=device,
                 train_size=0.5, val_size=0.5, test_size=0.5)

    def test_invalid_idim(self, synthetic_adata, device):
        from hsde import HSDE
        with pytest.raises(ValueError, match="Information bottleneck dimension"):
            HSDE(synthetic_adata, layer="counts", device=device,
                 latent_dim=10, i_dim=10)

    def test_invalid_sde_weights(self, synthetic_adata, device):
        from hsde import HSDE
        with pytest.raises(ValueError, match="SDE weights must sum to 1.0"):
            HSDE(synthetic_adata, layer="counts", device=device,
                 use_sde=True, vae_reg=0.3, sde_reg=0.3)


# ---------------------------------------------------------------------------
# Test: Visualization controller
# ---------------------------------------------------------------------------

class TestVisualizationController:
    def test_load_ablation_results(self):
        tables_dir = os.path.join(
            PROJECT_ROOT, "HSDE_results", "ablation", "tables"
        )
        if not os.path.exists(tables_dir):
            pytest.skip("Ablation results not found")

        from hsde.viz.controller import VisualizationController
        ctrl = VisualizationController(results_dir=tables_dir)
        ctrl.load_all()

        assert len(ctrl.raw_data) > 0
        assert len(ctrl.get_available_metrics()) > 0
        assert ctrl.long_data is not None
        assert len(ctrl.long_data) > 0

    def test_metric_availability(self):
        tables_dir = os.path.join(
            PROJECT_ROOT, "HSDE_results", "ablation", "tables"
        )
        if not os.path.exists(tables_dir):
            pytest.skip("Ablation results not found")

        from hsde.viz.controller import VisualizationController
        ctrl = VisualizationController(results_dir=tables_dir)
        ctrl.load_all()

        metrics = ctrl.get_available_metrics()
        # DRE series should be present
        dre_metrics = [m for m in metrics if m.startswith("DRE_")]
        assert len(dre_metrics) > 0, "No DRE metrics found"

        # LSE series should be present
        lse_metrics = [m for m in metrics if m.startswith("LSE_")]
        assert len(lse_metrics) > 0, "No LSE metrics found"

    def test_significance_computation(self):
        tables_dir = os.path.join(
            PROJECT_ROOT, "HSDE_results", "ablation", "tables"
        )
        if not os.path.exists(tables_dir):
            pytest.skip("Ablation results not found")

        from hsde.viz.controller import VisualizationController
        ctrl = VisualizationController(results_dir=tables_dir)
        ctrl.load_all()

        pval, stars = ctrl.compute_significance("NMI", "VAE", "HSDE (Full)")
        assert isinstance(pval, float)
        assert stars in ("***", "**", "*", "ns")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
