"""
HSDE: Hyperbolic SDE-Regularized VAE
=====================================

A deep learning framework for single-cell omics analysis combining:
- Variational Autoencoder (VAE) with count-based likelihoods (NB, ZINB, Poisson, ZIP)
- Lorentz (hyperbolic) geometric regularization for hierarchical structure
- Dual-path information bottleneck for coordinated biological programs
- Neural SDE regularization for trajectory inference
- Graph PDE diffusion for latent smoothing
- Multi-encoder: MLP and Transformer backbones
- Disentanglement: beta-VAE, DIP-VAE, TC-VAE, InfoVAE regularizers
"""

import logging

logging.getLogger("hsde").addHandler(logging.NullHandler())

from .core.agent import HSDE

__all__ = ["HSDE"]

__version__ = "2.0.0"
__author__ = "Zeyu Fu"
__project__ = "HSDE"
__full_name__ = "Hyperbolic SDE-Regularized VAE"
