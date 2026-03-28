"""
HSDE Core: Hyperbolic SDE-Regularized VAE Framework
====================================================

Integrates:
- VAE with count-based likelihoods (NB, ZINB, Poisson, ZIP)
- Lorentz (hyperbolic) and Euclidean manifold regularization
- Dual-path information bottleneck
- Neural SDE trajectory inference
- Graph PDE latent diffusion

Encoder options: 'mlp', 'transformer'
Decoder: MLP with NB/ZINB/Poisson/ZIP likelihoods
Regularization: beta-VAE, DIP-VAE, beta-TC-VAE, InfoVAE, Lorentz/Euclidean manifold
Dynamics: Neural SDE trajectory inference, Graph PDE diffusion
"""

from .agent import HSDE

__all__ = ["HSDE"]

__version__ = "2.0.0"
