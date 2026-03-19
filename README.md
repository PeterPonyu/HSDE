# HSDE-Graph

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
├── src/                        # Core framework
│   ├── agent.py                # HSDE — main user-facing API
│   ├── environment.py          # Data loading, preprocessing, training loop
│   ├── model.py                # Multi-objective loss computation, latent extraction
│   ├── module.py               # Neural network modules (encoders, decoders, VAE)
│   ├── graph_modules.py        # Graph encoder/decoder with 10+ conv types
│   ├── graph_utils.py          # Adjacency-to-edge, structural decoders
│   ├── mixin.py                # Loss mixins (scVI, β-TC, Info, DIP, SDE, ...)
│   ├── utils.py                # Lorentz geometry, TF-IDF, utilities
│   ├── vectorfield.py          # Vector field analysis & visualisation
│   ├── sde_functions.py        # SDE strategies (scaled, constant, annealed, clipped)
│   └── pde_functions.py        # Graph diffusion PDE
│
├── experiments/                # Evaluation & ablation scripts
│   ├── ablation_skill.py       # Reusable evaluation harness (metrics, LSE, DRE)
│   └── run_study.py            # Unified ablation & component efficiency study (v2)
│
├── data/                       # Datasets
│   └── BoneMarrow/
│       └── human_cd34_bone_marrow.h5ad
│
├── results/                    # Saved experiment outputs (CSV, JSON, logs)
│
├── STUDY_REPORT.md             # Full experimental report with analysis
├── LICENSE
└── README.md
```

## Quick Start

### Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.12
- PyTorch Geometric
- scanpy, scvelo, anndata
- scikit-learn, scipy, numpy, pandas

### Basic Usage

```python
from src.agent import HSDE

model = HSDE()

# Load and preprocess data
model.load_data("bone_marrow")  # or provide an AnnData object
model.preprocess(n_top_genes=2000)

# Configure and train
model.setup(
    encoder_type="graph",          # "mlp" | "transformer" | "graph"
    graph_conv_type="GAT",         # GAT, GCN, SAGE, ChebConv, ...
    latent_dim=10,
    information_bottleneck=True,   # enable IB
    irecon=0.5,                    # IB reconstruction weight
    geometry="lorentz",            # "lorentz" | "euclidean" | None
    beta=0.1,                      # KL weight
)

model.fit(max_epochs=100, patience=20)

# Extract results
latent = model.get_latent()        # latent representations
centroids = model.get_centroid()   # deterministic centroids (graph only)
```

### Run the Ablation Study

```bash
python experiments/run_study.py --epochs 100 --n_cells 3000 --n_genes 2000 --patience 20 --part all
```

Parts can be run individually: `--part encoder`, `--part component`, `--part ablation`.

Results are saved to `results/study_*.csv`, `results/study_*.json`, and `results/study_full_log.txt`.

## Design Rule

> **Geometry loss REQUIRES Information Bottleneck.**
>
> The Lorentz/Euclidean geometry loss computes manifold distance between `z_manifold` and `ld_manifold`. Without the Information Bottleneck (`irecon = 0`), `ld` is untrained, making the distance meaningless. The framework enforces this constraint automatically.

## Key Results

From the unified study (v2) on Setty Bone Marrow (3 000 cells, 2 000 HVGs, ≤ 100 epochs):

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
