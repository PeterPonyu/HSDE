import numpy as np
from anndata import AnnData
from scipy.sparse import csr_matrix, issparse
from sklearn.neighbors import NearestNeighbors
from scipy.stats import norm
from typing import Optional, Tuple, Union
import matplotlib.pyplot as plt
import warnings
import scanpy as sc


def quiver_autoscale(E: np.ndarray, V: np.ndarray) -> float:
    """Compute autoscale using matplotlib's quiver rendering."""
    fig, ax = plt.subplots()
    scale_factor = np.abs(E).max()
    
    if scale_factor == 0:
        scale_factor = 1.0

    Q = ax.quiver(
        E[:, 0] / scale_factor,
        E[:, 1] / scale_factor,
        V[:, 0],
        V[:, 1],
        angles="xy",
        scale=None,
        scale_units="xy",
    )
    
    try:
        fig.canvas.draw()
        quiver_scale = Q.scale if Q.scale is not None else 1.0
    except Exception:
        quiver_scale = 1.0
    finally:
        plt.close(fig)

    return quiver_scale / scale_factor


def l2_norm(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute L2 norm."""
    if issparse(x):
        return np.sqrt(x.multiply(x).sum(axis=axis).A1)
    else:
        return np.sqrt(np.sum(x * x, axis=axis))


class VectorFieldMixin:
    """Vector field analysis with automatic drift correction."""
    
    def get_vfres(
        self,
        adata: AnnData,
        zs_key: str = 'X_latent',
        E_key: str = 'X_umap',
        vf_key: str = "X_vf",
        T_key: str = "cosine_similarity",
        dv_key: str = "X_dv",
        t_key: str = 'pseudotime',
        n_neigh: int = 30,
        scale: int = 10,
        smooth: float = 0.5,
        stream: bool = False,
        density: float = 1.0,
        auto_reverse: bool = True,
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Compute vector field with automatic drift direction correction.
        
        Parameters
        ----------
        adata : AnnData
            Annotated data object
        zs_key : str
            Key in adata.obsm for latent space
        E_key : str
            Key in adata.obsm for embedding (UMAP)
        vf_key : str
            Key to store velocity field in adata.obsm
        T_key : str
            Key to store transition matrix in adata.obsp
        dv_key : str
            Key to store projected velocities in adata.obsm
        t_key : str
            Key in adata.obs for pseudotime
        n_neigh : int
            Number of neighbors for KNN
        scale : int
            Exponential scaling factor for transitions
        smooth : float
            Gaussian kernel bandwidth for grid interpolation
        stream : bool
            Return streamplot format (True) or quiver format (False)
        density : float
            Grid density multiplier
        auto_reverse : bool
            Automatically detect and correct drift direction
            
        Returns
        -------
        If stream=True: (x_1d, y_1d, U_2d, V_2d)
        If stream=False: (E_grid, V_grid)
        """
        
        self._validate_inputs(adata, zs_key, E_key, t_key)
        
        # 1. Compute raw drift
        grads = self.take_grad(self.X_norm)
        adata.obsm[vf_key] = grads
        
        # 2. Auto-detect and correct direction (uses BOTH alignment + magnitude)
        if auto_reverse:
            reverse = self._auto_detect_direction(adata, zs_key, vf_key, t_key)
            if reverse:
                adata.obsm[vf_key] = -adata.obsm[vf_key]
                print(f"[Auto-corrected] Reversed drift direction")
        
        # 3. Time-directed transition matrix
        adata.obsp[T_key] = self._get_similarity_time_directed(adata, zs_key, vf_key, t_key, n_neigh)
        
        # 4. Project to embedding
        adata.obsm[dv_key] = self._get_vf(adata, T_key, E_key, scale)
        
        # 5. Generate grid
        E = np.asarray(adata.obsm[E_key])
        V = np.asarray(adata.obsm[dv_key])
        return self._get_vfgrid(E, V, smooth, stream, density)
    
    def _validate_inputs(self, adata: AnnData, zs_key: str, E_key: str, t_key: str):
        """Validate required data exists."""
        if zs_key not in adata.obsm:
            raise KeyError(f"'{zs_key}' not in adata.obsm")
        if E_key not in adata.obsm:
            raise KeyError(f"'{E_key}' not in adata.obsm")
        if t_key not in adata.obs:
            raise KeyError(f"'{t_key}' not in adata.obs")
    
    def _auto_detect_direction(
        self, 
        adata: AnnData, 
        zs_key: str, 
        vf_key: str, 
        t_key: str,
        n_sample: int = 200
    ) -> bool:
        """
        Auto-detect if drift needs reversal using BOTH alignment and magnitude.
        
        Strategy:
        - Primary: Check if drift aligns with trajectory direction
        - Secondary: Check if drift magnitude increases over time
        
        Returns
        -------
        bool
            True if reversal needed
        """
        Z = adata.obsm[zs_key]
        V = adata.obsm[vf_key]
        time = adata.obs[t_key].values
        
        # ========== 1. ALIGNMENT TEST ==========
        n_sample = min(n_sample, len(time))
        sample_idx = np.random.choice(len(time), n_sample, replace=False)
        
        alignments = []
        for i in sample_idx:
            future_mask = time > time[i]
            if future_mask.sum() < 5:
                continue
            
            Z_future_mean = Z[future_mask].mean(axis=0)
            traj_direction = Z_future_mean - Z[i]
            traj_norm = traj_direction / (np.linalg.norm(traj_direction) + 1e-8)
            
            drift_norm = V[i] / (np.linalg.norm(V[i]) + 1e-8)
            alignment = np.dot(drift_norm, traj_norm)
            alignments.append(alignment)
        
        mean_alignment = np.mean(alignments)
        
        # ========== 2. MAGNITUDE TEST ==========
        V_mag = np.sqrt((V**2).sum(axis=1))
        t_sorted = np.argsort(time)
        
        n_edge = max(10, int(len(time) * 0.1))
        early_mag = V_mag[t_sorted[:n_edge]].mean()
        late_mag = V_mag[t_sorted[-n_edge:]].mean()
        
        mag_ratio = late_mag / (early_mag + 1e-8)
        
        # ========== 3. COMBINED DECISION ==========
        print(f"[Auto-detect]")
        print(f"  Alignment: {mean_alignment:.3f} {'→' if mean_alignment > 0 else '←'}")
        print(f"  Magnitude ratio (late/early): {mag_ratio:.2f} {'↑' if mag_ratio > 1 else '↓'}")
        
        # Reverse if EITHER:
        # A. Alignment is clearly negative
        # B. Magnitude decreases significantly (backward SDE)
        reverse_alignment = mean_alignment < -0.1
        reverse_magnitude = mag_ratio < 0.7  # Late < 70% of early
        
        if reverse_alignment and reverse_magnitude:
            print(f"  Decision: REVERSE (both signals agree)")
            return True
        elif reverse_alignment:
            print(f"  Decision: REVERSE (alignment negative)")
            return True
        elif reverse_magnitude:
            print(f"  Decision: REVERSE (magnitude decreasing)")
            return True
        else:
            print(f"  Decision: OK (forward)")
            return False
    
    def _get_similarity_time_directed(
        self,
        adata: AnnData,
        zs_key: str,
        vf_key: str,
        t_key: str,
        n_neigh: int,
    ) -> csr_matrix:
        """Compute time-directed similarity matrix (hard time masking)."""
        Z = np.array(adata.obsm[zs_key])
        V = np.array(adata.obsm[vf_key])
        time = np.array(adata.obs[t_key].values)
        
        nn = NearestNeighbors(n_neighbors=n_neigh, n_jobs=-1).fit(Z)
        _, indices = nn.kneighbors(Z)
        V_norm = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
        
        rows, cols, data = [], [], []
        
        for i in range(adata.n_obs):
            idx = indices[i]
            dZ = Z[idx] - Z[i]
            dZ_norm = dZ / (np.linalg.norm(dZ, axis=1, keepdims=True) + 1e-12)
            cos_sim = np.sum(V_norm[i] * dZ_norm, axis=1)
            
            time_mask = (time[idx] > time[i]).astype(float)
            weight = np.maximum(0, cos_sim * time_mask)
            
            valid = weight > 1e-6
            if np.sum(valid) > 0:
                rows.extend([i] * np.sum(valid))
                cols.extend(idx[valid])
                data.extend(weight[valid])
        
        T = csr_matrix((data, (rows, cols)), shape=(adata.n_obs, adata.n_obs))
        denom = np.array(T.sum(1)).flatten()
        denom[denom == 0] = 1.0
        T = T.multiply(csr_matrix(1.0 / denom[:, np.newaxis]))
        
        return T

    def _get_vf(self, adata: AnnData, T_key: str, E_key: str, scale: int) -> np.ndarray:
        """Project velocity field onto embedding space."""
        T = adata.obsp[T_key].copy()
        
        if issparse(T):
            T.data = np.sign(T.data) * np.expm1(np.abs(T.data) * scale)
        else:
            T = np.sign(T) * np.expm1(np.abs(T) * scale)
        
        if issparse(T):
            denom = np.array(np.abs(T).sum(1)).flatten()
            denom = np.maximum(denom, 1e-12)
            T = T.multiply(csr_matrix(1.0 / denom[:, np.newaxis]))
        else:
            denom = np.maximum(np.abs(T).sum(1, keepdims=True), 1e-12)
            T = T / denom
        
        E = np.array(adata.obsm[E_key])
        V = np.zeros(E.shape)
        
        for i in range(adata.n_obs):
            if issparse(T):
                idx, w = T[i].indices, T[i].data
            else:
                idx = np.where(T[i] != 0)[0]
                w = T[i, idx]
            
            if len(idx) > 0:
                dE = E[idx] - E[i]
                V[i] = np.sum(w[:, None] * dE, axis=0)
        
        V /= (3 * quiver_autoscale(E, V))
        
        return V

    def _get_vfgrid(
        self,
        E: np.ndarray,
        V: np.ndarray,
        smooth: float,
        stream: bool,
        density: float,
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Interpolate velocity field onto regular grid."""
        
        grs = []
        for i in range(E.shape[1]):
            m, M = np.min(E[:, i]), np.max(E[:, i])
            diff = M - m
            m, M = m - 0.01 * diff, M + 0.01 * diff
            grs.append(np.linspace(m, M, int(50 * density)))

        meshes = np.meshgrid(*grs)
        E_grid_points = np.vstack([i.flat for i in meshes]).T
        
        n_neigh = max(1, int(E.shape[0] / 50))
        nn = NearestNeighbors(n_neighbors=n_neigh, n_jobs=-1).fit(E)
        dists, neighs = nn.kneighbors(E_grid_points)
        
        scale = np.mean([g[1] - g[0] for g in grs]) * smooth
        weight = norm.pdf(x=dists, scale=scale)
        weight_sum = weight.sum(1)
        
        V_grid = (V[neighs] * weight[:, :, None]).sum(1)
        V_grid /= np.maximum(1, weight_sum[:, None])

        if stream:
            ns = len(grs[0])
            V_grid_2d = V_grid.T.reshape(2, ns, ns)
            U_grid = V_grid_2d[0]
            V_grid = V_grid_2d[1]
            
            mass = np.sqrt(U_grid**2 + V_grid**2)
            min_mass = np.percentile(mass, 99) * 0.01
            cutoff1 = mass < min_mass
            
            length = np.sum(
                np.mean(np.abs(V[neighs]), axis=1), axis=1
            ).reshape(ns, ns)
            cutoff2 = length < np.percentile(length, 5)
            
            cutoff = cutoff1 | cutoff2
            U_grid[cutoff] = np.nan
            V_grid[cutoff] = np.nan
            
            return grs[0], grs[1], U_grid, V_grid
        
        else:
            min_weight = np.percentile(weight_sum, 99) * 0.01
            mask = weight_sum > min_weight
            E_grid = E_grid_points[mask]
            V_grid = V_grid[mask]
            
            return E_grid, V_grid

    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    
    def plot_streamplot(
        self,
        adata: AnnData,
        zs_key: str = 'X_latent',
        E_key: str = 'X_umap',
        t_key: str = 'pseudotime',
        figsize: Tuple[int, int] = (7, 5),
        density: float = 1.5,
        linewidth: float = 1.2,
        arrowsize: float = 1.2,
        title: str = 'SDE Vector Field',
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Visualize vector field as streamlines.
        
        Parameters
        ----------
        adata : AnnData
            Annotated data object
        zs_key : str
            Key in adata.obsm for latent space
        E_key : str
            Key in adata.obsm for embedding (UMAP)
        t_key : str
            Key in adata.obs for pseudotime
        figsize : tuple
            Figure size (width, height)
        density : float
            Streamline density
        linewidth : float
            Streamline width
        arrowsize : float
            Arrow size
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
            
        Returns
        -------
        plt.Figure
            Matplotlib figure object
        """
        
        x, y, U, V = self.get_vfres(
            adata, zs_key=zs_key, E_key=E_key, t_key=t_key,
            stream=True, density=1.0
        )
        
        fig, ax = plt.subplots(figsize=figsize)
        
        scatter = ax.scatter(
            adata.obsm[E_key][:, 0], adata.obsm[E_key][:, 1],
            c=adata.obs[t_key], cmap='RdBu_r',
            s=15, alpha=0.5, rasterized=True, zorder=1, edgecolors='none'
        )
        
        ax.streamplot(
            x, y, U, V, 
            color='black', 
            density=density, 
            linewidth=linewidth,
            arrowsize=arrowsize, 
            arrowstyle='->', 
            zorder=2
        )
        
        plt.colorbar(scatter, ax=ax, label='Pseudotime', fraction=0.046, pad=0.04)
        ax.set_xlabel('UMAP 1', fontsize=10)
        ax.set_ylabel('UMAP 2', fontsize=10)
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_quiver(
        self,
        adata: AnnData,
        zs_key: str = 'X_latent',
        E_key: str = 'X_umap',
        t_key: str = 'pseudotime',
        figsize: Tuple[int, int] = (7, 5),
        density: float = 1.0,
        arrow_width: float = 0.003,
        title: str = 'SDE Vector Field',
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Visualize vector field as discrete arrows.
        
        Parameters
        ----------
        adata : AnnData
            Annotated data object
        zs_key : str
            Key in adata.obsm for latent space
        E_key : str
            Key in adata.obsm for embedding (UMAP)
        t_key : str
            Key in adata.obs for pseudotime
        figsize : tuple
            Figure size (width, height)
        density : float
            Grid density
        arrow_width : float
            Arrow shaft width
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
            
        Returns
        -------
        plt.Figure
            Matplotlib figure object
        """
        
        E_grid, V_grid = self.get_vfres(
            adata, zs_key=zs_key, E_key=E_key, t_key=t_key,
            stream=False, density=density
        )
        
        fig, ax = plt.subplots(figsize=figsize)
        
        scatter = ax.scatter(
            adata.obsm[E_key][:, 0], adata.obsm[E_key][:, 1],
            c=adata.obs[t_key], cmap='RdBu_r',
            s=15, alpha=0.5, rasterized=True, zorder=1, edgecolors='none'
        )
        
        ax.quiver(
            E_grid[:, 0], E_grid[:, 1], V_grid[:, 0], V_grid[:, 1],
            color='black', alpha=0.8, width=arrow_width,
            headwidth=4, headlength=5, scale_units='xy', angles='xy',
            zorder=2
        )
        
        plt.colorbar(scatter, ax=ax, label='Pseudotime', fraction=0.046, pad=0.04)
        ax.set_xlabel('UMAP 1', fontsize=10)
        ax.set_ylabel('UMAP 2', fontsize=10)
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

    # ========================================================================
    # DIAGNOSTICS
    # ========================================================================
    
    def diagnose_vector_field(
        self,
        adata: AnnData,
        vf_key: str = 'X_vf',
        zs_key: str = 'X_latent',
        t_key: str = 'pseudotime',
    ) -> dict:
        """
        Comprehensive vector field diagnostics.
        
        Parameters
        ----------
        adata : AnnData
            Annotated data object
        vf_key : str
            Key in adata.obsm for velocity field
        zs_key : str
            Key in adata.obsm for latent space
        t_key : str
            Key in adata.obs for pseudotime
            
        Returns
        -------
        dict
            Diagnostic metrics
        """
        
        vf_key = self._resolve_key(adata.obsm, [vf_key, 'X_vf_latent'])
        zs_key = self._resolve_key(adata.obsm, [zs_key, 'X_emb'])
        
        t = adata.obs[t_key].values
        V = adata.obsm[vf_key]
        Z = adata.obsm[zs_key]
        
        print("\n" + "="*70)
        print("VECTOR FIELD DIAGNOSTIC")
        print("="*70)
        
        alignment = []
        for i in range(0, len(t), max(1, len(t)//100)):
            future = t > t[i]
            if future.sum() < 5:
                continue
            traj = Z[future].mean(axis=0) - Z[i]
            traj /= (np.linalg.norm(traj) + 1e-8)
            drift = V[i] / (np.linalg.norm(V[i]) + 1e-8)
            alignment.append(np.dot(drift, traj))
        
        align_mean = np.mean(alignment)
        print(f"\n[Drift-Trajectory Alignment]")
        print(f"  Cosine: {align_mean:.3f}")
        print(f"  Status: {'✅ Forward' if align_mean > 0.3 else '❌ Misaligned'}")
        
        ratio = 0
        if 'cosine_similarity' in adata.obsp:
            T = adata.obsp['cosine_similarity']
            n_forward = sum((t[T[i].indices] > t[i]).sum() for i in range(adata.n_obs))
            n_total = T.nnz
            ratio = n_forward / (n_total + 1e-8)
            
            print(f"\n[Forward Transitions]")
            print(f"  Ratio: {ratio:.1%}")
            print(f"  Status: {'✅ Good' if ratio > 0.85 else '⚠️ Weak'}")
        
        V_mag = np.sqrt((V**2).sum(axis=1))
        print(f"\n[Velocity Magnitude]")
        print(f"  Mean: {V_mag.mean():.3f}")
        print(f"  Status: {'✅' if V_mag.mean() > 0.1 else '⚠️ Weak'}")
        
        print("="*70 + "\n")
        
        return {
            'alignment': align_mean,
            'forward_ratio': ratio,
            'velocity_mean': V_mag.mean(),
        }
    
    def _resolve_key(self, mapping, candidates):
        """Find first matching key from candidates."""
        for c in candidates:
            if c in mapping:
                return c
        raise KeyError(f"None of {candidates} found in {list(mapping.keys())}")