"""
Dimensionality Reduction Evaluator (DRE).

Internalized from MoCoO canonical implementation. Evaluates how well a
dimensionality-reduction method preserves the structure of the original
high-dimensional data using co-ranking matrix analysis.

Metrics:
  - distance_correlation: Spearman correlation of pairwise distances
  - Q_local: Local quality index (co-ranking, LCMC boundary)
  - Q_global: Global quality index (co-ranking, LCMC boundary)
"""

import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr
import warnings
from typing import Dict, Tuple


class DimensionalityReductionEvaluator:
    """Co-ranking matrix based DR quality evaluator."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def distance_correlation_score(self, X_high: np.ndarray, X_low: np.ndarray) -> float:
        """Spearman correlation between pairwise distances."""
        try:
            D_high = pairwise_distances(X_high)
            D_low = pairwise_distances(X_low)
            corr, _ = spearmanr(D_high.flatten(), D_low.flatten())
            return corr if not np.isnan(corr) else 0.0
        except Exception as e:
            warnings.warn(f"Error computing distance correlation: {e}")
            return 0.0

    def get_ranking_matrix(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Compute ranking matrix from pairwise distances."""
        try:
            n = len(distance_matrix)
            sorted_indices = np.argsort(distance_matrix, axis=1)
            ranking_matrix = np.zeros((n, n), dtype=np.int32)
            for i in range(n):
                ranking_matrix[i, sorted_indices[i]] = np.arange(n)
            # Exclude self
            mask = np.eye(n, dtype=bool)
            ranking_matrix[~mask] = ranking_matrix[~mask] - 1
            ranking_matrix[mask] = 0
            return ranking_matrix
        except Exception as e:
            warnings.warn(f"Error computing ranking matrix: {e}")
            return np.zeros((len(distance_matrix), len(distance_matrix)), dtype=np.int32)

    def get_coranking_matrix(self, rank_high: np.ndarray, rank_low: np.ndarray) -> np.ndarray:
        """Compute co-ranking matrix from high-D and low-D ranking matrices."""
        try:
            n = len(rank_high)
            corank = np.zeros((n - 1, n - 1), dtype=np.int32)
            mask = (rank_high > 0) & (rank_low > 0)
            valid_high = rank_high[mask] - 1
            valid_low = rank_low[mask] - 1
            valid_mask = (valid_high < n - 1) & (valid_low < n - 1)
            valid_high = valid_high[valid_mask]
            valid_low = valid_low[valid_mask]
            np.add.at(corank, (valid_high, valid_low), 1)
            return corank
        except Exception as e:
            warnings.warn(f"Error computing co-ranking matrix: {e}")
            return np.zeros((len(rank_high) - 1, len(rank_high) - 1), dtype=np.int32)

    def compute_qnx_series(self, corank: np.ndarray) -> np.ndarray:
        """Compute Q_NX series from co-ranking matrix."""
        try:
            n = corank.shape[0] + 1
            qnx_values = []
            Qnx_cum = 0
            for K in range(1, n - 1):
                if K - 1 < corank.shape[0]:
                    intrusions = np.sum(corank[:K, K - 1]) if K - 1 < corank.shape[1] else 0
                    extrusions = np.sum(corank[K - 1, :K]) if K - 1 < corank.shape[0] else 0
                    diagonal = corank[K - 1, K - 1] if K - 1 < min(corank.shape) else 0
                    Qnx_cum += intrusions + extrusions - diagonal
                    qnx_values.append(Qnx_cum / (K * n))
            return np.array(qnx_values)
        except Exception as e:
            warnings.warn(f"Error computing Q_NX series: {e}")
            return np.array([0.0])

    def get_q_local_global(self, qnx_values: np.ndarray) -> Tuple[float, float, int]:
        """Compute Q_local and Q_global via LCMC boundary."""
        try:
            if len(qnx_values) == 0:
                return 0.0, 0.0, 1
            lcmc = np.copy(qnx_values)
            N = len(qnx_values)
            for j in range(N):
                lcmc[j] = lcmc[j] - j / N
            K_max = np.argmax(lcmc) + 1
            Q_local = np.mean(qnx_values[:K_max]) if K_max > 0 else (qnx_values[0] if len(qnx_values) > 0 else 0.0)
            Q_global = np.mean(qnx_values[K_max:]) if K_max < len(qnx_values) else (qnx_values[-1] if len(qnx_values) > 0 else 0.0)
            return Q_local, Q_global, K_max
        except Exception as e:
            warnings.warn(f"Error computing Q metrics: {e}")
            return 0.0, 0.0, 1

    def comprehensive_evaluation(self, X_high: np.ndarray, X_low: np.ndarray, k: int = 10) -> Dict:
        """Full DRE evaluation: distance_correlation, Q_local, Q_global, overall_quality."""
        X_high = np.asarray(X_high, dtype=float)
        X_low = np.asarray(X_low, dtype=float)

        if X_high.shape[0] != X_low.shape[0]:
            raise ValueError(f"Sample count mismatch: {X_high.shape[0]} vs {X_low.shape[0]}")
        if k >= X_high.shape[0]:
            raise ValueError(f"k ({k}) must be < n_samples ({X_high.shape[0]})")

        results = {}
        results['distance_correlation'] = self.distance_correlation_score(X_high, X_low)

        D_high = pairwise_distances(X_high)
        D_low = pairwise_distances(X_low)
        rank_high = self.get_ranking_matrix(D_high)
        rank_low = self.get_ranking_matrix(D_low)
        corank = self.get_coranking_matrix(rank_high, rank_low)
        qnx_values = self.compute_qnx_series(corank)
        Q_local, Q_global, K_max = self.get_q_local_global(qnx_values)

        results['Q_local'] = Q_local
        results['Q_global'] = Q_global
        results['K_max'] = K_max
        results['overall_quality'] = np.mean([
            results['distance_correlation'], Q_local, Q_global
        ])
        return results
