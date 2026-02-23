# ============================================================================
# sde_functions.py - Controlled SDE Diffusion Strategies
# ============================================================================
"""
SDE diffusion term strategies for stable learning (from HSDE):
- ScaledDiffusion: learnable global scaling (recommended starting point)
- ConstantDiffusion: constant small diffusion (near-ODE)
- AnnealedDiffusion: annealing during training
- ClippedDiffusion: state-dependent but bounded
"""

import torch
import torch.nn as nn


class BaseControlledSDE(nn.Module):
    """Base class for controlled SDE models."""

    def __init__(self, n_latent: int, n_hidden: int, time_cond: str):
        super().__init__()
        self.n_latent = n_latent
        self.noise_type = "diagonal"
        self.sde_type = "ito"
        self.time_cond = time_cond

    def _build_network(self, name: str, n_latent: int, n_hidden: int, time_cond: str):
        if time_cond == "concat":
            setattr(self, f"{name}_fc1", nn.Linear(n_latent + 1, n_hidden))
        elif time_cond == "film":
            setattr(self, f"{name}_fc1", nn.Linear(n_latent, n_hidden))
            setattr(self, f"{name}_time_scale", nn.Linear(1, n_hidden))
            setattr(self, f"{name}_time_shift", nn.Linear(1, n_hidden))
        else:
            setattr(self, f"{name}_fc1", nn.Linear(n_latent, n_hidden))
            setattr(self, f"{name}_time_embed", nn.Linear(1, n_hidden))
        setattr(self, f"{name}_fc2", nn.Linear(n_hidden, n_latent))

    def _apply_time_cond(self, x, t, network):
        if self.time_cond == "concat":
            h = torch.cat([x, t], dim=-1)
            h = getattr(self, f"{network}_fc1")(h)
        elif self.time_cond == "film":
            h = getattr(self, f"{network}_fc1")(x)
            scale = getattr(self, f"{network}_time_scale")(t)
            shift = getattr(self, f"{network}_time_shift")(t)
            h = scale * h + shift
        else:
            h = getattr(self, f"{network}_fc1")(x) + getattr(self, f"{network}_time_embed")(t)
        return h

    def _broadcast_time(self, t, batch_size):
        if t.dim() == 0:
            return t.view(1, 1).expand(batch_size, 1)
        elif t.numel() == 1:
            return t.view(1, 1).expand(batch_size, 1)
        return t.view(-1, 1)

    def f(self, t, x):
        t = self._broadcast_time(t, x.shape[0])
        h = self._apply_time_cond(x, t, "drift")
        h = torch.nn.functional.elu(h)
        return self.drift_fc2(h)

    def forward(self, t, x):
        return self.f(t, x)


class ScaledDiffusionSDE(BaseControlledSDE):
    """SDE with learnable global diffusion scale."""

    def __init__(self, n_latent=10, n_hidden=25, initial_scale=0.1, time_cond="concat"):
        super().__init__(n_latent, n_hidden, time_cond)
        self._build_network("drift", n_latent, n_hidden, time_cond)
        self._build_network("diffusion", n_latent, n_hidden, time_cond)
        self.log_diffusion_scale = nn.Parameter(torch.tensor(initial_scale).log())
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def g(self, t, x):
        t = self._broadcast_time(t, x.shape[0])
        h = self._apply_time_cond(x, t, "diffusion")
        h = torch.nn.functional.elu(h)
        diffusion = self.diffusion_fc2(h)
        scale = self.log_diffusion_scale.exp()
        return scale * torch.nn.functional.softplus(diffusion)


class ConstantDiffusionSDE(BaseControlledSDE):
    """SDE with constant small diffusion (near-ODE)."""

    def __init__(self, n_latent=10, n_hidden=25, diffusion_const=0.01, time_cond="concat"):
        super().__init__(n_latent, n_hidden, time_cond)
        self._build_network("drift", n_latent, n_hidden, time_cond)
        self.register_buffer("diffusion_const", torch.tensor(diffusion_const))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def g(self, t, x):
        return torch.full_like(x, self.diffusion_const.item())


class AnnealedDiffusionSDE(BaseControlledSDE):
    """SDE with diffusion that anneals during training."""

    def __init__(self, n_latent=10, n_hidden=25, initial_scale=0.5, final_scale=0.01, time_cond="concat"):
        super().__init__(n_latent, n_hidden, time_cond)
        self._build_network("drift", n_latent, n_hidden, time_cond)
        self._build_network("diffusion", n_latent, n_hidden, time_cond)
        self.register_buffer("initial_scale", torch.tensor(initial_scale))
        self.register_buffer("final_scale", torch.tensor(final_scale))
        self.register_buffer("current_scale", torch.tensor(initial_scale))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def set_diffusion_scale(self, progress: float):
        new_scale = self.initial_scale + progress * (self.final_scale - self.initial_scale)
        self.current_scale.fill_(new_scale.item())

    def g(self, t, x):
        t = self._broadcast_time(t, x.shape[0])
        h = self._apply_time_cond(x, t, "diffusion")
        h = torch.nn.functional.elu(h)
        diffusion = self.diffusion_fc2(h)
        return self.current_scale * torch.nn.functional.softplus(diffusion)


class ClippedDiffusionSDE(BaseControlledSDE):
    """SDE with state-dependent but bounded diffusion."""

    def __init__(self, n_latent=10, n_hidden=25, min_diffusion=0.001, max_diffusion=0.1, time_cond="concat"):
        super().__init__(n_latent, n_hidden, time_cond)
        self.register_buffer("min_diffusion", torch.tensor(min_diffusion))
        self.register_buffer("max_diffusion", torch.tensor(max_diffusion))
        self._build_network("drift", n_latent, n_hidden, time_cond)
        self._build_network("diffusion", n_latent, n_hidden, time_cond)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def g(self, t, x):
        t = self._broadcast_time(t, x.shape[0])
        h = self._apply_time_cond(x, t, "diffusion")
        h = torch.nn.functional.elu(h)
        diffusion = self.diffusion_fc2(h)
        diffusion = torch.sigmoid(diffusion)
        return self.min_diffusion + diffusion * (self.max_diffusion - self.min_diffusion)


def create_controlled_sde(
    strategy: str = "scaled",
    n_latent: int = 10,
    n_hidden: int = 25,
    time_cond: str = "concat",
    **kwargs,
):
    """
    Factory for controlled SDE diffusion strategies.

    Parameters
    ----------
    strategy : str
        'scaled', 'constant', 'annealed', or 'clipped'
    """
    if strategy == "scaled":
        return ScaledDiffusionSDE(n_latent, n_hidden, kwargs.get("initial_scale", 0.1), time_cond)
    elif strategy == "constant":
        return ConstantDiffusionSDE(n_latent, n_hidden, kwargs.get("diffusion_const", 0.01), time_cond)
    elif strategy == "annealed":
        return AnnealedDiffusionSDE(
            n_latent, n_hidden,
            kwargs.get("initial_scale", 0.5),
            kwargs.get("final_scale", 0.01),
            time_cond,
        )
    elif strategy == "clipped":
        return ClippedDiffusionSDE(
            n_latent, n_hidden,
            kwargs.get("min_diffusion", 0.001),
            kwargs.get("max_diffusion", 0.1),
            time_cond,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Choose: scaled, constant, annealed, clipped")
