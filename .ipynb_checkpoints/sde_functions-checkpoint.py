
"""
Strategies to control SDE diffusion term for stable learning.
"""

import torch
import torch.nn as nn


# ============================================================================
# Strategy 1: Learnable Global Scaling (Recommended Starting Point)
# ============================================================================

class ScaledDiffusionSDE(nn.Module):
    """SDE with learnable global diffusion scale."""
    
    def __init__(self, n_latent: int = 10, n_hidden: int = 25, 
                 initial_scale: float = 0.1, time_cond: str = 'concat'):
        super().__init__()
        self.n_latent = n_latent
        self.noise_type = 'diagonal'
        self.sde_type = 'ito'
        self.time_cond = time_cond
        
        # Drift network (same as before)
        self._build_network('drift', n_latent, n_hidden, time_cond)
        
        # Diffusion network
        self._build_network('diffusion', n_latent, n_hidden, time_cond)
        
        # **Key addition: Learnable global scale**
        self.log_diffusion_scale = nn.Parameter(
            torch.tensor(initial_scale).log()
        )
        
        self.apply(lambda m: nn.init.xavier_normal_(m.weight) 
                   if isinstance(m, nn.Linear) else None)
    
    def _build_network(self, name: str, n_latent: int, n_hidden: int, time_cond: str):
        if time_cond == 'concat':
            setattr(self, f'{name}_fc1', nn.Linear(n_latent + 1, n_hidden))
        elif time_cond == 'film':
            setattr(self, f'{name}_fc1', nn.Linear(n_latent, n_hidden))
            setattr(self, f'{name}_time_scale', nn.Linear(1, n_hidden))
            setattr(self, f'{name}_time_shift', nn.Linear(1, n_hidden))
        else:  # 'add'
            setattr(self, f'{name}_fc1', nn.Linear(n_latent, n_hidden))
            setattr(self, f'{name}_time_embed', nn.Linear(1, n_hidden))
        
        setattr(self, f'{name}_fc2', nn.Linear(n_hidden, n_latent))
    
    def _apply_time_cond(self, x: torch.Tensor, t: torch.Tensor, network: str):
        if self.time_cond == 'concat':
            h = torch.cat([x, t], dim=-1)
            h = getattr(self, f'{network}_fc1')(h)
        elif self.time_cond == 'film':
            h = getattr(self, f'{network}_fc1')(x)
            scale = getattr(self, f'{network}_time_scale')(t)
            shift = getattr(self, f'{network}_time_shift')(t)
            h = scale * h + shift
        else:
            h = getattr(self, f'{network}_fc1')(x) + getattr(self, f'{network}_time_embed')(t)
        return h
    
    def _broadcast_time(self, t: torch.Tensor, batch_size: int):
        if t.dim() == 0:
            return t.expand(batch_size, 1)
        return t.view(-1, 1).expand(batch_size, 1)
    
    def f(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Drift term."""
        t = self._broadcast_time(t, x.shape[0])
        h = self._apply_time_cond(x, t, 'drift')
        h = torch.nn.functional.elu(h)
        return self.drift_fc2(h)
    
    def g(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Diffusion term with learnable scaling."""
        t = self._broadcast_time(t, x.shape[0])
        h = self._apply_time_cond(x, t, 'diffusion')
        h = torch.nn.functional.elu(h)
        diffusion = self.diffusion_fc2(h)
        
        # Apply learnable global scale
        scale = self.log_diffusion_scale.exp()
        return scale * torch.nn.functional.softplus(diffusion)
    
    def forward(self, t: torch.Tensor, x: torch.Tensor):
        return self.f(t, x)


# ============================================================================
# Strategy 2: Constant Small Diffusion (Most Conservative)
# ============================================================================

class ConstantDiffusionSDE(nn.Module):
    """SDE with constant small diffusion (near-ODE behavior)."""
    
    def __init__(self, n_latent: int = 10, n_hidden: int = 25,
                 diffusion_const: float = 0.01, time_cond: str = 'concat'):
        super().__init__()
        self.n_latent = n_latent
        self.noise_type = 'diagonal'
        self.sde_type = 'ito'
        self.time_cond = time_cond
        
        # Only drift network needed
        self._build_network('drift', n_latent, n_hidden, time_cond)
        
        # Constant diffusion
        self.diffusion_const = diffusion_const
        
        self.apply(lambda m: nn.init.xavier_normal_(m.weight) 
                   if isinstance(m, nn.Linear) else None)
    
    def _build_network(self, name: str, n_latent: int, n_hidden: int, time_cond: str):
        if time_cond == 'concat':
            setattr(self, f'{name}_fc1', nn.Linear(n_latent + 1, n_hidden))
        elif time_cond == 'film':
            setattr(self, f'{name}_fc1', nn.Linear(n_latent, n_hidden))
            setattr(self, f'{name}_time_scale', nn.Linear(1, n_hidden))
            setattr(self, f'{name}_time_shift', nn.Linear(1, n_hidden))
        else:
            setattr(self, f'{name}_fc1', nn.Linear(n_latent, n_hidden))
            setattr(self, f'{name}_time_embed', nn.Linear(1, n_hidden))
        
        setattr(self, f'{name}_fc2', nn.Linear(n_hidden, n_latent))
    
    def _apply_time_cond(self, x: torch.Tensor, t: torch.Tensor, network: str):
        if self.time_cond == 'concat':
            h = torch.cat([x, t], dim=-1)
            h = getattr(self, f'{network}_fc1')(h)
        elif self.time_cond == 'film':
            h = getattr(self, f'{network}_fc1')(x)
            scale = getattr(self, f'{network}_time_scale')(t)
            shift = getattr(self, f'{network}_time_shift')(t)
            h = scale * h + shift
        else:
            h = getattr(self, f'{network}_fc1')(x) + getattr(self, f'{network}_time_embed')(t)
        return h
    
    def _broadcast_time(self, t: torch.Tensor, batch_size: int):
        if t.dim() == 0:
            return t.expand(batch_size, 1)
        return t.view(-1, 1).expand(batch_size, 1)
    
    def f(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Drift term."""
        t = self._broadcast_time(t, x.shape[0])
        h = self._apply_time_cond(x, t, 'drift')
        h = torch.nn.functional.elu(h)
        return self.drift_fc2(h)
    
    def g(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Constant diffusion."""
        return torch.full_like(x, self.diffusion_const)
    
    def forward(self, t: torch.Tensor, x: torch.Tensor):
        return self.f(t, x)


# ============================================================================
# Strategy 3: Annealed Diffusion (Training Stability)
# ============================================================================

class AnnealedDiffusionSDE(nn.Module):
    """SDE with diffusion that anneals during training."""
    
    def __init__(self, n_latent: int = 10, n_hidden: int = 25,
                 initial_scale: float = 0.5, final_scale: float = 0.01,
                 time_cond: str = 'concat'):
        super().__init__()
        self.n_latent = n_latent
        self.noise_type = 'diagonal'
        self.sde_type = 'ito'
        self.time_cond = time_cond
        
        # Networks
        self._build_network('drift', n_latent, n_hidden, time_cond)
        self._build_network('diffusion', n_latent, n_hidden, time_cond)
        
        # Annealing parameters
        self.initial_scale = initial_scale
        self.final_scale = final_scale
        self.current_scale = initial_scale
        
        self.apply(lambda m: nn.init.xavier_normal_(m.weight) 
                   if isinstance(m, nn.Linear) else None)
    
    def _build_network(self, name: str, n_latent: int, n_hidden: int, time_cond: str):
        if time_cond == 'concat':
            setattr(self, f'{name}_fc1', nn.Linear(n_latent + 1, n_hidden))
        elif time_cond == 'film':
            setattr(self, f'{name}_fc1', nn.Linear(n_latent, n_hidden))
            setattr(self, f'{name}_time_scale', nn.Linear(1, n_hidden))
            setattr(self, f'{name}_time_shift', nn.Linear(1, n_hidden))
        else:
            setattr(self, f'{name}_fc1', nn.Linear(n_latent, n_hidden))
            setattr(self, f'{name}_time_embed', nn.Linear(1, n_hidden))
        
        setattr(self, f'{name}_fc2', nn.Linear(n_hidden, n_latent))
    
    def _apply_time_cond(self, x: torch.Tensor, t: torch.Tensor, network: str):
        if self.time_cond == 'concat':
            h = torch.cat([x, t], dim=-1)
            h = getattr(self, f'{network}_fc1')(h)
        elif self.time_cond == 'film':
            h = getattr(self, f'{network}_fc1')(x)
            scale = getattr(self, f'{network}_time_scale')(t)
            shift = getattr(self, f'{network}_time_shift')(t)
            h = scale * h + shift
        else:
            h = getattr(self, f'{network}_fc1')(x) + getattr(self, f'{network}_time_embed')(t)
        return h
    
    def _broadcast_time(self, t: torch.Tensor, batch_size: int):
        if t.dim() == 0:
            return t.expand(batch_size, 1)
        return t.view(-1, 1).expand(batch_size, 1)
    
    def set_diffusion_scale(self, progress: float):
        """
        Update diffusion scale based on training progress.
        
        Parameters
        ----------
        progress : float in [0, 1]
            Training progress (0 = start, 1 = end)
        """
        self.current_scale = self.initial_scale + progress * (
            self.final_scale - self.initial_scale
        )
    
    def f(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Drift term."""
        t = self._broadcast_time(t, x.shape[0])
        h = self._apply_time_cond(x, t, 'drift')
        h = torch.nn.functional.elu(h)
        return self.drift_fc2(h)
    
    def g(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Annealed diffusion."""
        t = self._broadcast_time(t, x.shape[0])
        h = self._apply_time_cond(x, t, 'diffusion')
        h = torch.nn.functional.elu(h)
        diffusion = self.diffusion_fc2(h)
        
        return self.current_scale * torch.nn.functional.softplus(diffusion)
    
    def forward(self, t: torch.Tensor, x: torch.Tensor):
        return self.f(t, x)


# ============================================================================
# Strategy 4: State-Dependent Diffusion with Clipping
# ============================================================================

class ClippedDiffusionSDE(nn.Module):
    """SDE with state-dependent but bounded diffusion."""
    
    def __init__(self, n_latent: int = 10, n_hidden: int = 25,
                 min_diffusion: float = 0.001, max_diffusion: float = 0.1,
                 time_cond: str = 'concat'):
        super().__init__()
        self.n_latent = n_latent
        self.noise_type = 'diagonal'
        self.sde_type = 'ito'
        self.time_cond = time_cond
        
        self.min_diffusion = min_diffusion
        self.max_diffusion = max_diffusion
        
        # Networks
        self._build_network('drift', n_latent, n_hidden, time_cond)
        self._build_network('diffusion', n_latent, n_hidden, time_cond)
        
        self.apply(lambda m: nn.init.xavier_normal_(m.weight) 
                   if isinstance(m, nn.Linear) else None)
    
    def _build_network(self, name: str, n_latent: int, n_hidden: int, time_cond: str):
        if time_cond == 'concat':
            setattr(self, f'{name}_fc1', nn.Linear(n_latent + 1, n_hidden))
        elif time_cond == 'film':
            setattr(self, f'{name}_fc1', nn.Linear(n_latent, n_hidden))
            setattr(self, f'{name}_time_scale', nn.Linear(1, n_hidden))
            setattr(self, f'{name}_time_shift', nn.Linear(1, n_hidden))
        else:
            setattr(self, f'{name}_fc1', nn.Linear(n_latent, n_hidden))
            setattr(self, f'{name}_time_embed', nn.Linear(1, n_hidden))
        
        setattr(self, f'{name}_fc2', nn.Linear(n_hidden, n_latent))
    
    def _apply_time_cond(self, x: torch.Tensor, t: torch.Tensor, network: str):
        if self.time_cond == 'concat':
            h = torch.cat([x, t], dim=-1)
            h = getattr(self, f'{network}_fc1')(h)
        elif self.time_cond == 'film':
            h = getattr(self, f'{network}_fc1')(x)
            scale = getattr(self, f'{network}_time_scale')(t)
            shift = getattr(self, f'{network}_time_shift')(t)
            h = scale * h + shift
        else:
            h = getattr(self, f'{network}_fc1')(x) + getattr(self, f'{network}_time_embed')(t)
        return h
    
    def _broadcast_time(self, t: torch.Tensor, batch_size: int):
        if t.dim() == 0:
            return t.expand(batch_size, 1)
        return t.view(-1, 1).expand(batch_size, 1)
    
    def f(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Drift term."""
        t = self._broadcast_time(t, x.shape[0])
        h = self._apply_time_cond(x, t, 'drift')
        h = torch.nn.functional.elu(h)
        return self.drift_fc2(h)
    
    def g(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Bounded diffusion."""
        t = self._broadcast_time(t, x.shape[0])
        h = self._apply_time_cond(x, t, 'diffusion')
        h = torch.nn.functional.elu(h)
        diffusion = self.diffusion_fc2(h)
        
        # Clip to reasonable range
        diffusion = torch.nn.functional.softplus(diffusion)
        diffusion = torch.clamp(diffusion, self.min_diffusion, self.max_diffusion)
        
        return diffusion
    
    def forward(self, t: torch.Tensor, x: torch.Tensor):
        return self.f(t, x)


# ============================================================================
# Factory Function
# ============================================================================

def create_controlled_sde(
    strategy: str = 'scaled',
    n_latent: int = 10,
    n_hidden: int = 25,
    time_cond: str = 'concat',
    **kwargs
):
    """
    Create SDE with controlled diffusion.
    
    Parameters
    ----------
    strategy : str
        Control strategy: 'scaled', 'constant', 'annealed', or 'clipped'
    n_latent : int
        Latent dimension
    n_hidden : int
        Hidden layer size
    time_cond : str
        Time conditioning: 'concat', 'film', or 'add'
    **kwargs
        Strategy-specific parameters
    
    Returns
    -------
    sde_func : nn.Module
    
    Examples
    --------
    >>> # Start with learnable scaling
    >>> sde = create_controlled_sde('scaled', n_latent=10, initial_scale=0.1)
    
    >>> # Very conservative (near-ODE)
    >>> sde = create_controlled_sde('constant', n_latent=10, diffusion_const=0.01)
    
    >>> # Annealing during training
    >>> sde = create_controlled_sde('annealed', n_latent=10, 
    ...                             initial_scale=0.5, final_scale=0.01)
    
    >>> # State-dependent but bounded
    >>> sde = create_controlled_sde('clipped', n_latent=10, 
    ...                             min_diffusion=0.001, max_diffusion=0.1)
    """
    if strategy == 'scaled':
        return ScaledDiffusionSDE(
            n_latent, n_hidden, 
            initial_scale=kwargs.get('initial_scale', 0.1),
            time_cond=time_cond
        )
    elif strategy == 'constant':
        return ConstantDiffusionSDE(
            n_latent, n_hidden,
            diffusion_const=kwargs.get('diffusion_const', 0.01),
            time_cond=time_cond
        )
    elif strategy == 'annealed':
        return AnnealedDiffusionSDE(
            n_latent, n_hidden,
            initial_scale=kwargs.get('initial_scale', 0.5),
            final_scale=kwargs.get('final_scale', 0.01),
            time_cond=time_cond
        )
    elif strategy == 'clipped':
        return ClippedDiffusionSDE(
            n_latent, n_hidden,
            min_diffusion=kwargs.get('min_diffusion', 0.001),
            max_diffusion=kwargs.get('max_diffusion', 0.1),
            time_cond=time_cond
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
