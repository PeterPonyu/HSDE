# HSDE

**Hyperbolic SDE-Regularised VAE for Single-Cell Omics**

A PyTorch framework that combines variational autoencoders with hyperbolic geometry, neural SDEs, and graph PDEs for single-cell RNA-seq analysis — including dimensionality reduction, clustering, trajectory inference, and vector field estimation.

---

## Features

- **Multi-encoder architecture**: MLP and Transformer (multi-head attention)
- **Flexible likelihood**: Negative Binomial (NB), Zero-Inflated NB (ZINB), Poisson, Zero-Inflated Poisson (ZIP)
- **Information Bottleneck**: optional secondary reconstruction objective (`irecon`) for structured latent compression
- **Manifold geometry**: Lorentz (hyperbolic) and Euclidean manifold losses
- **Neural SDE**: stochastic trajectory inference in latent space
- **Graph PDE**: kNN-based diffusion for latent smoothing
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
│   │   ├── mixin.py               # Loss mixins (scVI, β-TC, Info, DIP, SDE)
│   │   ├── utils.py               # Lorentz geometry, utilities
│   │   ├── vectorfield.py         # Vector field analysis & visualisation
│   │   ├── sde_functions.py       # SDE strategies (scaled, constant, annealed, clipped)
│   │   └── pde_functions.py       # Graph diffusion PDE (kNN Laplacian)
│   └── metrics/                   # Internal evaluation metrics
│       ├── dre.py                 # Dimensionality Reduction Error
│       └── lse.py                 # Latent Structure Ensemble
│
├── tests/                         # Integration tests
│   ├── conftest.py
│   └── test_models.py
│
├── data/                          # Datasets (not tracked)
├── pyproject.toml                 # Package configuration & dependencies
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
- torchsde (for SDE trajectory inference)
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

# Full model: Lorentz + IB + SDE + PDE
model = HSDE(
    adata, layer="counts",
    irecon=1.0, lorentz=5.0,
    use_sde=True, use_pde=True,
    vae_reg=0.5, sde_reg=0.5, pde_reg=0.2,
    latent_dim=10, i_dim=2,
)
model.fit(epochs=400, patience=25)

latent = model.get_latent()
pseudotime = model.get_time()
```

## Design Rule

> **Geometry loss REQUIRES Information Bottleneck.**
>
> The Lorentz/Euclidean geometry loss computes manifold distance between `z_manifold` and `ld_manifold`. Without the Information Bottleneck (`irecon = 0`), `ld` is untrained, making the distance meaningless. The framework enforces this constraint automatically.

## License

See [LICENSE](LICENSE) for details.
