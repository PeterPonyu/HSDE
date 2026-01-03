
import numpy as np
from anndata import AnnData
from scipy.sparse import csr_matrix, issparse
from sklearn.neighbors import NearestNeighbors
from scipy.stats import norm
from typing import Optional, Tuple, Union
import matplotlib.pyplot as plt
import warnings


def quiver_autoscale(E: np.ndarray, V: np.ndarray) -> float:
    """Compute autoscale from matplotlib's quiver."""
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
    """Vector field analysis with automatic drift direction correction."""
    
    def get_vfres(
        self,
        adata: AnnData,
        zs_key: str = "X_latent",
        E_key: str = "X_umap",
        vf_key: str = "X_vf",
        T_key: str = "cosine_similarity",
        dv_key: str = "X_dv",
        t_key: str = "pseudotime",
        n_neigh: int = 30,
        scale: int = 10,
        smooth: float = 0.5,
        stream: bool = False,
        density: float = 1.0,
        auto_reverse: bool = True,
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Compute vector field with automatic drift direction correction.
        
        Args:
            adata: Annotated data object
            zs_key: Latent space key in adata.obsm
            E_key: Embedding (UMAP) key in adata.obsm
            vf_key: Output velocity field key
            T_key: Output transition matrix key
            dv_key: Output projected velocities key
            t_key: Pseudotime key in adata.obs
            n_neigh: KNN neighbors
            scale: Exponential scaling for transitions
            smooth: Gaussian kernel bandwidth
            stream: Return streamplot format (vs quiver)
            density: Grid density multiplier
            auto_reverse: Auto-detect and correct drift direction
        
        Returns:
            If stream=True: (x_1d, y_1d, U_2d, V_2d)
            If stream=False: (E_grid, V_grid)
        """
        self._validate_inputs(adata, zs_key, E_key, t_key)
        
        # Compute raw drift
        grads = self.take_grad(self.X_norm)
        adata.obsm[vf_key] = grads
        
        # Auto-correct if needed
        if auto_reverse:
            reverse = self._auto_detect_direction(adata, zs_key, vf_key, t_key)
            if reverse:
                adata.obsm[vf_key] *= -1
                adata.obs[t_key] *= -1
                print("[Auto-corrected] Reversed drift and time direction")
        
        # Build time-directed transition matrix
        adata.obsp[T_key] = self._get_similarity_time_directed(
            adata, zs_key, vf_key, t_key, n_neigh
        )
        
        # Project to embedding
        adata.obsm[dv_key] = self._get_vf(adata, T_key, E_key, scale)
        
        # Generate grid
        E = np.asarray(adata.obsm[E_key])
        V = np.asarray(adata.obsm[dv_key])
        return self._get_vfgrid(E, V, smooth, stream, density)
    
    def _validate_inputs(self, adata: AnnData, zs_key: str, E_key: str, t_key: str) -> None:
        """Validate required keys exist."""
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
        n_sample: int = 200,
    ) -> bool:
        """
        Auto-detect if drift needs reversal.
        
        Checks:
        1. Alignment: Does drift point toward future trajectory?
        2. Magnitude: Does magnitude increase over time (forward SDE)?
        
        Returns True if reversal needed.
        """
        Z = np.asarray(adata.obsm[zs_key])
        V = np.asarray(adata.obsm[vf_key])
        time = np.asarray(adata.obs[t_key].values)
        
        # Normalize time for clarity
        time = (time - time.min()) / (time.max() - time.min() + 1e-8)
        
        # ===== Alignment check =====
        n_sample = min(n_sample, len(time))
        sample_idx = np.random.choice(len(time), n_sample, replace=False)
        
        alignments = []
        for i in sample_idx:
            # Find cells at later timepoints
            future_mask = time > time[i]
            if future_mask.sum() < 5:
                continue
            
            # Expected trajectory direction
            Z_future_mean = Z[future_mask].mean(axis=0)
            traj_direction = Z_future_mean - Z[i]
            traj_norm = np.linalg.norm(traj_direction)
            
            if traj_norm < 1e-8:
                continue
            
            traj_direction /= traj_norm
            
            # Drift direction
            drift_norm = np.linalg.norm(V[i])
            if drift_norm < 1e-8:
                continue
            
            drift_direction = V[i] / drift_norm
            
            # Cosine similarity
            alignment = np.dot(drift_direction, traj_direction)
            alignments.append(alignment)
        
        mean_alignment = np.mean(alignments) if alignments else 0.0
        
        # ===== Magnitude check =====
        V_mag = np.linalg.norm(V, axis=1)
        t_sorted_idx = np.argsort(time)
        
        n_edge = max(10, int(len(time) * 0.1))
        early_mag = V_mag[t_sorted_idx[:n_edge]].mean()
        late_mag = V_mag[t_sorted_idx[-n_edge:]].mean()
        
        # Avoid division by zero
        mag_ratio = late_mag / (early_mag + 1e-8)
        
        # ===== Decision logic =====
        print("[Auto-detect Direction]")
        print(f"  Alignment (drift→future): {mean_alignment:.3f}")
        print(f"  Magnitude ratio (late/early): {mag_ratio:.3f}")
        
        # Reverse if alignment is clearly negative OR magnitude decreases
        reverse_alignment = mean_alignment < -0.15
        reverse_magnitude = mag_ratio < 0.65
        
        reverse = reverse_alignment or reverse_magnitude
        print(f"  Decision: {'REVERSE' if reverse else 'OK'}")
        
        return reverse
    
    def _get_similarity_time_directed(
        self,
        adata: AnnData,
        zs_key: str,
        vf_key: str,
        t_key: str,
        n_neigh: int,
    ) -> csr_matrix:
        """
        Time-directed transition matrix.
        Only connect to cells at later times if cosine similarity > 0.
        """
        Z = np.asarray(adata.obsm[zs_key])
        V = np.asarray(adata.obsm[vf_key])
        time = np.asarray(adata.obs[t_key].values)
        
        nn = NearestNeighbors(n_neighbors=n_neigh, n_jobs=-1).fit(Z)
        _, indices = nn.kneighbors(Z)
        
        # Normalize drift
        V_norm = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
        
        rows, cols, data = [], [], []
        
        for i in range(adata.n_obs):
            neighbors = indices[i]
            
            # Direction to neighbors
            dZ = Z[neighbors] - Z[i]
            dZ_norm = dZ / (np.linalg.norm(dZ, axis=1, keepdims=True) + 1e-12)
            
            # Cosine similarity with drift
            cos_sim = np.sum(V_norm[i] * dZ_norm, axis=1)
            
            # Only positive alignment
            cos_sim = np.maximum(0, cos_sim)
            
            # Only forward in time
            time_mask = (time[neighbors] > time[i]).astype(float)
            weight = cos_sim * time_mask
            
            # Keep non-trivial weights
            valid = weight > 1e-6
            if valid.sum() > 0:
                rows.extend([i] * valid.sum())
                cols.extend(neighbors[valid])
                data.extend(weight[valid])
        
        # Build sparse matrix and normalize
        T = csr_matrix((data, (rows, cols)), shape=(adata.n_obs, adata.n_obs))
        row_sums = np.array(T.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0
        T = T.multiply(csr_matrix(1.0 / row_sums[:, np.newaxis]))
        
        return T
    
    def _get_vf(
        self,
        adata: AnnData,
        T_key: str,
        E_key: str,
        scale: int,
    ) -> np.ndarray:
        """Project velocity field to embedding space."""
        T = adata.obsp[T_key].copy()
        
        # Exponential scaling
        if issparse(T):
            T.data = np.sign(T.data) * np.expm1(np.abs(T.data) * scale)
        else:
            T = np.sign(T) * np.expm1(np.abs(T) * scale)
        
        # Renormalize
        if issparse(T):
            row_sums = np.array(np.abs(T).sum(axis=1)).flatten()
            row_sums = np.maximum(row_sums, 1e-12)
            T = T.multiply(csr_matrix(1.0 / row_sums[:, np.newaxis]))
        else:
            row_sums = np.maximum(np.abs(T).sum(axis=1, keepdims=True), 1e-12)
            T = T / row_sums
        
        E = np.asarray(adata.obsm[E_key])
        V = np.zeros(E.shape)
        
        # Compute weighted displacement
        for i in range(adata.n_obs):
            if issparse(T):
                idx, w = T[i].indices, T[i].data
            else:
                idx = np.where(T[i] != 0)[0]
                w = T[i, idx]
            
            if len(idx) > 0:
                dE = E[idx] - E[i]
                V[i] = np.sum(w[:, None] * dE, axis=0)
        
        # Autoscale
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
        
        # Build grid
        grids = []
        for i in range(E.shape[1]):
            m, M = np.min(E[:, i]), np.max(E[:, i])
            diff = M - m
            m, M = m - 0.01 * diff, M + 0.01 * diff
            grids.append(np.linspace(m, M, int(50 * density)))
        
        meshes = np.meshgrid(*grids)
        E_grid_points = np.vstack([m.flat for m in meshes]).T
        
        # Interpolate via Gaussian kernel
        n_neigh = max(1, int(E.shape[0] / 50))
        nn = NearestNeighbors(n_neighbors=n_neigh, n_jobs=-1).fit(E)
        dists, neighs = nn.kneighbors(E_grid_points)
        
        scale = np.mean([g[1] - g[0] for g in grids]) * smooth
        weight = norm.pdf(x=dists, scale=scale)
        weight_sum = weight.sum(axis=1, keepdims=True)
        
        V_grid = (V[neighs] * weight[:, :, None]).sum(axis=1)
        V_grid /= np.maximum(1, weight_sum)
        
        if stream:
            # Format for streamplot
            n = len(grids[0])
            V_2d = V_grid.T.reshape(2, n, n)
            U, V_out = V_2d[0], V_2d[1]
            
            # Mask low-signal regions
            magnitude = np.sqrt(U**2 + V_out**2)
            min_mag = np.percentile(magnitude, 99) * 0.01
            U[magnitude < min_mag] = np.nan
            V_out[magnitude < min_mag] = np.nan
            
            return grids[0], grids[1], U, V_out
        else:
            # Mask low-weight regions
            min_weight = np.percentile(weight_sum, 99) * 0.01
            mask = weight_sum.flatten() > min_weight
            return E_grid_points[mask], V_grid[mask]
    
    def plot_streamplot(
        self,
        adata: AnnData,
        zs_key: str = "X_latent",
        E_key: str = "X_umap",
        t_key: str = "pseudotime",
        figsize: Tuple[int, int] = (7, 5),
        density: float = 1.5,
        linewidth: float = 1.2,
        arrowsize: float = 1.2,
        title: str = "SDE Vector Field",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Streamline visualization of vector field."""
        
        x, y, U, V = self.get_vfres(
            adata, zs_key=zs_key, E_key=E_key, t_key=t_key,
            stream=True, density=1.0
        )
        
        fig, ax = plt.subplots(figsize=figsize)
        
        scatter = ax.scatter(
            adata.obsm[E_key][:, 0], adata.obsm[E_key][:, 1],
            c=adata.obs[t_key], cmap="RdBu_r",
            s=15, alpha=0.5, rasterized=True, zorder=1, edgecolors="none"
        )
        
        ax.streamplot(
            x, y, U, V,
            color="black",
            density=density,
            linewidth=linewidth,
            arrowsize=arrowsize,
            arrowstyle="->",
            zorder=2,
        )
        
        plt.colorbar(scatter, ax=ax, label="Pseudotime", fraction=0.046, pad=0.04)
        ax.set_xlabel("UMAP 1", fontsize=10)
        ax.set_ylabel("UMAP 2", fontsize=10)
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        return fig
    
    def plot_quiver(
        self,
        adata: AnnData,
        zs_key: str = "X_latent",
        E_key: str = "X_umap",
        t_key: str = "pseudotime",
        figsize: Tuple[int, int] = (7, 5),
        density: float = 1.0,
        arrow_width: float = 0.003,
        title: str = "SDE Vector Field",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Quiver visualization of vector field."""
        
        E_grid, V_grid = self.get_vfres(
            adata, zs_key=zs_key, E_key=E_key, t_key=t_key,
            stream=False, density=density
        )
        
        fig, ax = plt.subplots(figsize=figsize)
        
        scatter = ax.scatter(
            adata.obsm[E_key][:, 0], adata.obsm[E_key][:, 1],
            c=adata.obs[t_key], cmap="RdBu_r",
            s=15, alpha=0.5, rasterized=True, zorder=1, edgecolors="none"
        )
        
        ax.quiver(
            E_grid[:, 0], E_grid[:, 1], V_grid[:, 0], V_grid[:, 1],
            color="black", alpha=0.8, width=arrow_width,
            headwidth=4, headlength=5, scale_units="xy", angles="xy",
            zorder=2,
        )
        
        plt.colorbar(scatter, ax=ax, label="Pseudotime", fraction=0.046, pad=0.04)
        ax.set_xlabel("UMAP 1", fontsize=10)
        ax.set_ylabel("UMAP 2", fontsize=10)
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        return fig
    
    def diagnose_vector_field(
        self,
        adata: AnnData,
        vf_key: str = "X_vf",
        zs_key: str = "X_latent",
        t_key: str = "pseudotime",
    ) -> dict:
        """
        Comprehensive vector field diagnostics.
        
        Returns dict with alignment, forward_ratio, velocity_mean metrics.
        """
        vf_key = self._resolve_key(adata.obsm, [vf_key, "X_vf_latent"])
        zs_key = self._resolve_key(adata.obsm, [zs_key, "X_emb"])
        
        t = np.asarray(adata.obs[t_key].values)
        V = np.asarray(adata.obsm[vf_key])
        Z = np.asarray(adata.obsm[zs_key])
        
        print("\n" + "=" * 70)
        print("VECTOR FIELD DIAGNOSTIC")
        print("=" * 70)
        
        # Alignment
        alignments = []
        for i in range(0, len(t), max(1, len(t) // 100)):
            future = t > t[i]
            if future.sum() < 5:
                continue
            traj = Z[future].mean(axis=0) - Z[i]
            traj_norm = np.linalg.norm(traj)
            if traj_norm < 1e-8:
                continue
            traj /= traj_norm
            
            drift_norm = np.linalg.norm(V[i])
            if drift_norm < 1e-8:
                continue
            drift = V[i] / drift_norm
            
            alignments.append(np.dot(drift, traj))
        
        align_mean = np.mean(alignments) if alignments else 0.0
        print(f"\n[Drift-Trajectory Alignment]")
        print(f"  Cosine: {align_mean:.3f}")
        print(f"  Status: {'✅ Forward' if align_mean > 0.3 else '⚠️ Misaligned'}")
        
        # Forward transitions
        forward_ratio = 0.0
        if "cosine_similarity" in adata.obsp:
            T = adata.obsp["cosine_similarity"]
            n_forward = sum(
                (t[T[i].indices] > t[i]).sum() if issparse(T) else 
                (t[np.where(T[i] != 0)[0]] > t[i]).sum()
                for i in range(adata.n_obs)
            )
            n_total = T.nnz if issparse(T) else (T != 0).sum()
            forward_ratio = n_forward / (n_total + 1e-8)
            
            print(f"\n[Forward Transitions]")
            print(f"  Ratio: {forward_ratio:.1%}")
            print(f"  Status: {'✅ Good' if forward_ratio > 0.85 else '⚠️ Weak'}")
        
        # Velocity magnitude
        V_mag = np.linalg.norm(V, axis=1)
        vel_mean = V_mag.mean()
        print(f"\n[Velocity Magnitude]")
        print(f"  Mean: {vel_mean:.3f}")
        print(f"  Status: {'✅' if vel_mean > 0.1 else '⚠️ Weak'}")
        
        print("=" * 70 + "\n")
        
        return {
            "alignment": align_mean,
            "forward_ratio": forward_ratio,
            "velocity_mean": vel_mean,
        }
    
    def _resolve_key(self, mapping: dict, candidates: list) -> str:
        """Find first matching key from candidates."""
        for c in candidates:
            if c in mapping:
                return c
        raise KeyError(f"None of {candidates} found")
