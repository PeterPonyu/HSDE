"""
LIORA: Lorentz Information-bottleneck Omics Representation Architecture
=======================================================================

A unified deep learning framework for single-cell omics analysis combining:
- Variational Autoencoder (VAE) with count-based likelihoods (NB, ZINB, Poisson, ZIP)
- Lorentz (hyperbolic) geometric regularization for hierarchical structure
- Dual-path information bottleneck for coordinated biological programs
- Neural SDE regularization for trajectory inference
- Graph PDE diffusion for latent smoothing
- Graph neural network encoders (GAT, GCN, ChebConv, SAGE, etc.)
- Graph structure decoders for adjacency learning (CCVGAE)
- Transformer-based attention mechanisms

Supports scRNA-seq and scATAC-seq modalities.

Version: 2.0.0
Author: Zeyu Fu (School of Computer Science, University of Birmingham)
License: MIT
"""

__version__ = "2.0.0"
__author__ = "Zeyu Fu"
__project__ = "LIORA"
__full_name__ = "Lorentz Information-bottleneck Omics Representation Architecture"
