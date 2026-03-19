# HSDE

**Hyperbolic SDE-Regularised VAE with Graph Attention for Single-Cell Omics**

A PyTorch framework that combines variational autoencoders with graph neural networks, hyperbolic geometry, neural SDEs, and graph PDEs for single-cell RNA-seq analysis — including dimensionality reduction, clustering, trajectory inference, and vector field estimation.

---

## Features

- **Multi-encoder architecture**: MLP, Transformer (multi-head attention), Graph (GAT, GCN, ChebConv, GraphSAGE, SSG, Transformer, ARMA, and more via PyTorch Geometric)
- **Flexible likelihood**: Negative Binomial (NB), Zero-Inflated NB (ZINB), Poisson, Zero-Inflated Poisson (ZIP)
- **Information Bottleneck**: optional secondary reconstruction objective (`irecon`) for structured latent compression
- **Manifold geometry**: Lorentz (hyperbolic) and Euclidean manifold losses
- **Neural SDE**: stochastic trajectory inference in latent space
- **Graph PDE**: diffusion-based message passing for temporal dynamics
- **Structural decoder**: adjacency reconstruction via inner product, bilinear, or MLP-based decoders
- **Advanced VAE regularisers**: β-VAE, DIP-VAE, β-TC-VAE, InfoVAE

## Project Structure

```
├── hsde/                          # Main package
│   ├── __init__.py                # Package root — exports HSDE class
│   ├── core/                      # Core framework
│   │   ├── agent.py               # HSDE — main user-facing API
│   │   ├── environment.py         # Data loading, preprocessing, training loop
│   │   ├── model.py               # Multi-objective loss computation, latent extraction
│   │   ├── module.py              # Neural network modules (encoders, decoders, VAE)
│   │   ├── graph_modules.py       # Graph encoder/decoder with 10+ conv types
│   │   ├── graph_utils.py         # Adjacency-to-edge, structural decoders
│   │   ├── mixin.py               # Loss mixins (scVI, β-TC, Info, DIP, SDE, ...)
│   │   ├── utils.py               # Lorentz geometry, TF-IDF, utilities
│   │   ├── vectorfield.py         # Vector field analysis & visualisation
│   │   ├── sde_functions.py       # SDE strategies (scaled, constant, annealed, clipped)
│   │   └── pde_functions.py       # Graph diffusion PDE
│   └── viz/                       # Visualization tools
│       ├── style.py               # Publication figure styling (IEEE J-BHI)
│       ├── controller.py          # Automated benchmark visualization
│       └── run_all_visualizations.py
│
├── experiments/                   # Evaluation & ablation scripts
│   ├── exp_utils.py               # Shared experiment utilities
│   ├── run_ablation.py            # Ablation study (5 variants × 12 datasets)
│   ├── run_disentanglement.py     # Disentanglement regularization comparison
│   ├── run_gmvae_benchmark.py     # GM-VAE geometric distribution benchmark
│   ├── downstream_analysis.py     # Full downstream analysis pipeline
│   └── visualize_studies.py       # Study result visualization
│
├── tests/                         # Integration tests
│   ├── conftest.py
│   └── test_models.py
│
├── data/                          # Datasets (not tracked)
├── HSDE_results/                  # Experiment outputs (not tracked)
├── pyproject.toml                 # Package configuration & dependencies
├── STUDY_REPORT.md                # Full experimental report
├── LICENSE
└── README.md
```

## Installation

```bash
# Core only
pip install -e .

# With all optional dependencies
pip install -e ".[all]"

# Development (includes testing)
pip install -e ".[dev]"
```

### Requirements

- Python ≥ 3.9
- PyTorch ≥ 1.12
- See `pyproject.toml` for full dependency list

## Quick Start

```python
from hsde import HSDE
import scanpy as sc

# Load data
adata = sc.read_h5ad("data/BoneMarrow/human_cd34_bone_marrow.h5ad")

# Standard MLP encoder
model = HSDE(adata, layer="counts", latent_dim=10, i_dim=2)
model.fit(epochs=100, patience=25)
latent = model.get_latent()

# Full model: Graph + Lorentz + SDE + PDE
model = HSDE(
    adata, layer="counts",
    encoder_type="graph", graph_type="GAT",
    irecon=1.0, lorentz=5.0,
    use_sde=True, use_pde=True,
    vae_reg=0.5, sde_reg=0.5, pde_reg=0.2,
    latent_dim=10, i_dim=2,
)
model.fit(epochs=400, patience=25)

latent = model.get_latent()
pseudotime = model.get_time()
centroids = model.get_centroid()
```

### Run Experiments

```bash
python experiments/run_ablation.py
python experiments/run_disentanglement.py
python experiments/run_gmvae_benchmark.py
python experiments/downstream_analysis.py
```

## Design Rule

> **Geometry loss REQUIRES Information Bottleneck.**
>
> The Lorentz/Euclidean geometry loss computes manifold distance between `z_manifold` and `ld_manifold`. Without the Information Bottleneck (`irecon = 0`), `ld` is untrained, making the distance meaningless. The framework enforces this constraint automatically.

## Key Results

From the unified study on Setty Bone Marrow (3 000 cells, 2 000 HVGs, ≤ 100 epochs):

| Finding | Detail |
|---------|--------|
| **Best ARI** | 0.5902 — GAT Baseline (recon + β-KL only) |
| **Best NMI** | 0.7060 — Lorentz → Euclidean (IB + Euclidean geometry) |
| **Best embedding quality** | DRE UMAP 0.6997 — Full GAT (IB + Lorentz + β) |
| **Best latent structure** | LSE 0.4939 — MLP (minimal regularisation) |
| **Best efficiency** | 0.1135 ARI/s — GAT Baseline (5.2 s training) |
| **Graph vs MLP** | GAT: +0.139 ARI, +0.164 DRE UMAP, 2.6× faster |

See [STUDY_REPORT.md](STUDY_REPORT.md) for full analysis, tables, and conclusions.

## License

See [LICENSE](LICENSE) for details.
