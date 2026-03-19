# HSDE: Lorentz Information-bottleneck Omics Representation Architecture
"""
HSDE: Unified Single-Cell Omics Analysis Framework
=====================================================

Project Structure:
- src/       : Core framework source code
- experiments/ : Ablation studies and component analysis scripts
- data/      : Datasets
- results/   : Experiment outputs

Integrates complementary architectural components:
- **Lorentz Geometry**: Hyperbolic manifold regularization for hierarchical structure
- **Information Bottleneck**: Dual-path coupling for coordinated biological programs
- **Neural SDE**: Stochastic differential equation trajectory inference
- **Graph PDE**: Latent-space graph diffusion regularization
- **Count-based VAE**: NB/ZINB/Poisson/ZIP likelihood functions
- **Multi-encoder**: MLP, Transformer, and Graph (GAT/GCN/etc.) backbones
- **Disentanglement**: β-VAE, DIP-VAE, TC-VAE, InfoVAE regularizers
"""

from .src.agent import HSDE

__all__ = ["HSDE"]

__version__ = "2.0.0"
