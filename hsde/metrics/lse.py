"""
Single-Cell Latent Space Evaluator (LSE).

Internalized from MoCoO canonical implementation. Evaluates intrinsic quality
of latent-space representations for single-cell data.

Metrics:
  - manifold_dimensionality: PCA-based ensemble dimensional efficiency
  - spectral_decay_rate: Eigenvalue concentration
  - participation_ratio: Information concentration score
  - anisotropy_score: 6-method composite anisotropy
  - trajectory_directionality: PC1 dominance
  - noise_resilience: SNR-based (signal in top-2 PCs vs rest)
  - core_quality, overall_quality: Aggregates
"""

import numpy as np
from sklearn.decomposition import PCA
from scipy.linalg import svd
import warnings


class SingleCellLatentSpaceEvaluator:
    """Latent-space quality evaluator tailored for single-cell data."""

    def __init__(self, data_type: str = "trajectory", verbose: bool = False):
        self.data_type = data_type
        self.verbose = verbose
        if data_type == "trajectory":
            self.isotropy_preference = "low"
            self.participation_preference = "low"
        else:
            self.isotropy_preference = "high"
            self.participation_preference = "high"

    def manifold_dimensionality_score_v2(self, latent_space: np.ndarray) -> float:
        """4-method ensemble dimensional efficiency score."""
        try:
            if latent_space.shape[1] == 1:
                return 1.0
            centered = latent_space - np.mean(latent_space, axis=0)
            pca = PCA().fit(centered)
            evr = pca.explained_variance_ratio_
            ev = pca.explained_variance_

            scores = []
            # Method 1: Multi-threshold
            for thr in [0.8, 0.9, 0.95]:
                cumsum = np.cumsum(evr)
                idx = np.where(cumsum >= thr)[0]
                if len(idx) > 0:
                    eff = 1.0 - (idx[0]) / (latent_space.shape[1] - 1)
                    scores.append(eff)

            # Method 2: Kaiser criterion
            norm_ev = ev / np.mean(ev)
            kaiser_dim = np.sum(norm_ev > 1.0)
            scores.append(1.0 - (kaiser_dim - 1) / (latent_space.shape[1] - 1))

            # Method 3: Elbow
            if len(ev) > 2:
                ratios = ev[:-1] / ev[1:]
                elbow_dim = np.argmax(ratios) + 1
                scores.append(1.0 - (elbow_dim - 1) / (latent_space.shape[1] - 1))
            else:
                scores.append(1.0)

            # Method 4: Spectral decay
            if len(ev) > 1:
                log_ev = np.log(ev + 1e-10)
                slope = np.polyfit(np.arange(len(log_ev)), log_ev, 1)[0]
                scores.append(1.0 / (1.0 + np.exp(slope)))
            else:
                scores.append(0.5)

            return float(np.clip(np.mean(scores), 0.0, 1.0))
        except Exception as e:
            warnings.warn(f"Error in manifold_dimensionality: {e}")
            return 0.5

    def spectral_decay_rate(self, latent_space: np.ndarray) -> float:
        """Spectral decay rate — higher = better dimensional concentration."""
        try:
            centered = latent_space - np.mean(latent_space, axis=0)
            _, s, _ = svd(centered, full_matrices=False)
            eigenvalues = s ** 2 / (len(latent_space) - 1)
            if len(eigenvalues) < 2:
                return 1.0
            log_ev = np.log(eigenvalues + 1e-10)
            slope, _ = np.polyfit(np.arange(len(log_ev)), log_ev, 1)
            normalized_decay = 1.0 / (1.0 + np.exp(slope))
            concentration = eigenvalues[0] / np.sum(eigenvalues)
            return float(np.clip(0.6 * normalized_decay + 0.4 * concentration, 0.0, 1.0))
        except Exception as e:
            warnings.warn(f"Error in spectral_decay_rate: {e}")
            return 0.5

    def participation_ratio_score(self, latent_space: np.ndarray) -> float:
        """Participation ratio — adjusted for trajectory vs steady-state."""
        try:
            centered = latent_space - np.mean(latent_space, axis=0)
            cov = np.cov(centered.T)
            eigenvalues = np.real(np.linalg.eigvals(cov))
            eigenvalues = eigenvalues[eigenvalues > 1e-10]
            if len(eigenvalues) == 0:
                return 0.0
            s1 = np.sum(eigenvalues)
            s2 = np.sum(eigenvalues ** 2)
            if s2 > 0:
                pr = s1 ** 2 / s2
                normalized_pr = pr / len(eigenvalues)
            else:
                normalized_pr = 0.0
            score = (1.0 - normalized_pr) if self.participation_preference == "low" else normalized_pr
            return float(np.clip(score, 0.0, 1.0))
        except Exception as e:
            warnings.warn(f"Error in participation_ratio: {e}")
            return 0.5

    def isotropy_anisotropy_score(self, latent_space: np.ndarray) -> float:
        """6-method composite anisotropy score."""
        try:
            centered = latent_space - np.mean(latent_space, axis=0)
            cov = np.cov(centered.T)
            eigenvalues = np.real(np.linalg.eigvals(cov))
            eigenvalues = eigenvalues[eigenvalues > 1e-12]
            if len(eigenvalues) < 2:
                return 1.0
            eigenvalues = np.sort(eigenvalues)[::-1]

            # Method 1: Log-ellipticity
            log_ell = np.log(eigenvalues[0]) - np.log(eigenvalues[-1] + 1e-12)
            m1 = np.tanh(log_ell / 4.0)

            # Method 2: Multi-level condition number
            cond_ratios = [np.log(eigenvalues[i] / (eigenvalues[i + 1] + 1e-12))
                           for i in range(len(eigenvalues) - 1)]
            m2 = np.tanh(np.mean(cond_ratios) / 2.0)

            # Method 3: Ratio-variance
            ratios = eigenvalues[:-1] / (eigenvalues[1:] + 1e-12)
            m3 = np.tanh(np.var(np.log(ratios)))

            # Method 4: Entropy-based
            probs = eigenvalues / np.sum(eigenvalues)
            ent = -np.sum(probs * np.log(probs + 1e-12))
            max_ent = np.log(len(eigenvalues))
            m4 = 1.0 - (ent / max_ent if max_ent > 0 else 0)

            # Method 5: Primary dominance
            dom = eigenvalues[0] / (np.sum(eigenvalues[1:]) + 1e-12)
            m5 = np.tanh(np.log(dom + 1) / 2.0)

            # Method 6: Inverse effective dimensionality
            pr = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)
            m6 = 1.0 - (pr / len(eigenvalues))

            anisotropy = m1 * 0.25 + m2 * 0.25 + m3 * 0.20 + m4 * 0.15 + m5 * 0.10 + m6 * 0.05
            score = anisotropy if self.isotropy_preference == "low" else (1.0 - anisotropy)
            return float(np.clip(score, 0.0, 1.0))
        except Exception as e:
            warnings.warn(f"Error in isotropy_anisotropy: {e}")
            return 0.5

    def trajectory_directionality_score(self, latent_space: np.ndarray) -> float:
        """PC1 dominance as trajectory directionality."""
        try:
            pca = PCA().fit(latent_space)
            ev = pca.explained_variance_ratio_
            if len(ev) >= 2:
                other = np.sum(ev[1:])
                if other > 1e-10:
                    ratio = ev[0] / other
                    return float(np.clip(ratio / (1.0 + ratio), 0.0, 1.0))
            return 1.0
        except Exception as e:
            warnings.warn(f"Error in trajectory_directionality: {e}")
            return 0.5

    def noise_resilience_score(self, latent_space: np.ndarray) -> float:
        """SNR-based noise resilience (signal = top-2 PCs)."""
        try:
            pca = PCA().fit(latent_space)
            ev = pca.explained_variance_
            if len(ev) > 2:
                signal = np.sum(ev[:2])
                noise = np.sum(ev[2:])
                if noise > 1e-10:
                    return float(np.clip(min(signal / noise / 10.0, 1.0), 0.0, 1.0))
            return 1.0
        except Exception as e:
            warnings.warn(f"Error in noise_resilience: {e}")
            return 0.5

    def comprehensive_evaluation(self, latent_space: np.ndarray) -> dict:
        """Full LSE evaluation returning all metrics + aggregates."""
        latent_space = np.asarray(latent_space, dtype=float)
        results = {}
        results['manifold_dimensionality'] = self.manifold_dimensionality_score_v2(latent_space)
        results['spectral_decay_rate'] = self.spectral_decay_rate(latent_space)
        results['participation_ratio'] = self.participation_ratio_score(latent_space)
        results['anisotropy_score'] = self.isotropy_anisotropy_score(latent_space)
        results['trajectory_directionality'] = self.trajectory_directionality_score(latent_space)
        results['noise_resilience'] = self.noise_resilience_score(latent_space)

        core = [results['manifold_dimensionality'], results['spectral_decay_rate'],
                results['participation_ratio'], results['anisotropy_score']]
        results['core_quality'] = float(np.mean(core))

        if self.data_type == "trajectory":
            results['overall_quality'] = (
                results['core_quality'] * 0.5
                + results['trajectory_directionality'] * 0.3
                + results['noise_resilience'] * 0.2
            )
        else:
            results['overall_quality'] = (
                results['core_quality'] * 0.7
                + results['noise_resilience'] * 0.3
            )
        return results
