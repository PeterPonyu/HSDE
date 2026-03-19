"""
HSDE Core: Unified Single-Cell Omics Analysis Framework
========================================================

Integrates:
- HSDE: Hyperbolic SDE-Regularized VAE with Lorentz geometry, neural SDE
  trajectory inference, PDE graph diffusion, and count-based likelihoods.
- CCVGAE: Graph Attention Network encoders/decoders, graph structure learning,
  subgraph sampling, and centroid inference.

Encoder options: 'mlp', 'transformer', 'graph' (GAT, GCN, ChebConv, SAGE, etc.)
Decoder options: 'mlp' (with NB/ZINB/Poisson/ZIP likelihoods), 'graph'
Regularization: beta-VAE, DIP-VAE, beta-TC-VAE, InfoVAE, Lorentz/Euclidean manifold
Dynamics: Neural SDE trajectory inference, Graph PDE diffusion
"""

from .agent import HSDE

__all__ = [
    "HSDE",
    "agent", "environment", "model", "module",
    "graph_modules", "graph_utils", "mixin",
    "sde_functions", "pde_functions", "utils", "vectorfield",
]

__version__ = "2.0.0"
