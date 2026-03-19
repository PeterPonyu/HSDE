"""
HSDE Automatic Visualization Controller
==========================================

Fully automatic system for loading, computing, and visualizing metrics from
the HSDE benchmark results. Integrates the DRE (Dimensionality Reduction
Evaluator) and LSE (Latent Space Evaluator) series from the MoCoO evaluation
framework.

Usage
-----
    from hsde.viz.controller import VisualizationController

    ctrl = VisualizationController(results_dir="HSDE_results/ablation/tables")
    ctrl.load_all()
    ctrl.generate_all_figures(output_dir="figures/ablation")
"""
from __future__ import annotations

import os
import glob
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy import stats as scipy_stats

from . import style as S


# ---------------------------------------------------------------------------
# Metric group definitions for figure generation
# ---------------------------------------------------------------------------
METRIC_GROUPS = {
    "clustering": {
        "metrics": S.METRICS_CLUSTERING,
        "title": "Clustering Metrics",
        "ncols": 6,
        "figsize": (19.2, 3.8),
    },
    "dre_umap": {
        "metrics": S.METRICS_DRE_UMAP,
        "title": "DRE Series (UMAP)",
        "ncols": 4,
        "figsize": (12.8, 3.8),
    },
    "dre_tsne": {
        "metrics": S.METRICS_DRE_TSNE,
        "title": "DRE Series (t-SNE)",
        "ncols": 4,
        "figsize": (12.8, 3.8),
    },
    "lse": {
        "metrics": S.METRICS_LSE,
        "title": "LSE Series (Intrinsic Latent Quality)",
        "ncols": 7,
        "figsize": (22.4, 3.8),
    },
    "drex": {
        "metrics": S.METRICS_DREX,
        "title": "DREX Series (Extended DR Evaluation)",
        "ncols": 7,
        "figsize": (22.4, 3.8),
    },
    "lsex": {
        "metrics": S.METRICS_LSEX,
        "title": "LSEX Series (Extended Latent Evaluation)",
        "ncols": 5,
        "figsize": (16.0, 3.8),
    },
    "proposed": {
        "metrics": S.PROPOSED_METRICS,
        "title": "Proposed Summary Metrics",
        "ncols": 8,
        "figsize": (25.6, 3.8),
    },
}


class VisualizationController:
    """Fully automatic visualization system for HSDE benchmark results.

    Loads CSV result tables from a directory, extracts DRE and LSE series
    metrics, and generates publication-quality figures matching the MoCoO
    article style (IEEE J-BHI, 17x21cm, 300 DPI, Arial).

    Parameters
    ----------
    results_dir : str or Path
        Directory containing per-dataset CSV files (e.g., ablation_dentate_df.csv)
    method_names : list of str, optional
        Names of methods in the CSV index. Auto-detected if None.
    method_order : list of str, optional
        Display order. Defaults to method_names order.
    palette : dict or list, optional
        Color mapping. Defaults to HSDE palette.
    """

    def __init__(
        self,
        results_dir: str | Path,
        method_names: Optional[List[str]] = None,
        method_order: Optional[List[str]] = None,
        palette: Optional[dict | list] = None,
    ):
        self.results_dir = Path(results_dir)
        self.method_names = method_names
        self.method_order = method_order
        self.palette = palette

        # Data storage
        self.raw_data: Dict[str, pd.DataFrame] = {}  # dataset_name -> DataFrame
        self.long_data: Optional[pd.DataFrame] = None  # melted long-form
        self.summary: Optional[pd.DataFrame] = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_all(self) -> "VisualizationController":
        """Scan results_dir for CSV files and load all metrics data."""
        S.apply_style()

        csv_files = sorted(glob.glob(str(self.results_dir / "*.csv")))
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in {self.results_dir}"
            )

        for fpath in csv_files:
            fname = os.path.basename(fpath)
            # Extract dataset name from filename patterns like:
            #   ablation_dentate_df.csv -> dentate
            #   benchmark_BoneMarrow_df.csv -> BoneMarrow
            name = fname
            for prefix in ["ablation_", "benchmark_", "gmvae_", "disent_"]:
                if name.startswith(prefix):
                    name = name[len(prefix):]
                    break
            name = name.replace("_df.csv", "").replace(".csv", "")

            try:
                df = pd.read_csv(fpath, index_col=0)
                # Clean NaN with column medians
                for col in df.columns:
                    if df[col].isna().any():
                        median_val = df[col].median()
                        df[col] = df[col].fillna(
                            median_val if not pd.isna(median_val) else 0
                        )
                self.raw_data[name] = df
            except Exception as e:
                warnings.warn(f"Failed to load {fpath}: {e}")

        if not self.raw_data:
            raise ValueError("No valid CSV data loaded")

        # Auto-detect methods
        first_df = next(iter(self.raw_data.values()))
        if self.method_names is None:
            self.method_names = list(first_df.index)
        if self.method_order is None:
            self.method_order = list(self.method_names)
        if self.palette is None:
            config_colors = S.get_config_colors()
            self.palette = {
                m: config_colors.get(m, S._PALETTE[i % len(S._PALETTE)])
                for i, m in enumerate(self.method_names)
            }

        # Build long-form DataFrame for plotting
        self._build_long_data()
        self._build_summary()
        self._loaded = True

        print(f"Loaded {len(self.raw_data)} datasets, "
              f"{len(self.method_names)} methods")
        return self

    def _build_long_data(self):
        """Melt raw DataFrames into a single long-form DataFrame."""
        records = []
        for dataset, df in self.raw_data.items():
            for method in df.index:
                for metric in df.columns:
                    val = df.loc[method, metric]
                    if np.isscalar(val) and not pd.isna(val):
                        records.append({
                            "dataset": dataset,
                            "method": str(method),
                            "metric": metric,
                            "value": float(val),
                        })
        self.long_data = pd.DataFrame(records)

    def _build_summary(self):
        """Compute per-method mean +/- std across datasets."""
        if self.long_data is None or self.long_data.empty:
            return

        summary = (
            self.long_data
            .groupby(["method", "metric"])["value"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        self.summary = summary

    # ------------------------------------------------------------------
    # Statistical tests
    # ------------------------------------------------------------------

    def compute_significance(
        self,
        metric: str,
        method_a: str,
        method_b: str,
    ) -> Tuple[float, str]:
        """Compute Wilcoxon signed-rank test between two methods on a metric."""
        if self.long_data is None:
            return 1.0, "ns"

        vals_a = []
        vals_b = []
        for dataset in self.raw_data:
            df = self.raw_data[dataset]
            if method_a in df.index and method_b in df.index and metric in df.columns:
                va = df.loc[method_a, metric]
                vb = df.loc[method_b, metric]
                if np.isfinite(va) and np.isfinite(vb):
                    vals_a.append(va)
                    vals_b.append(vb)

        if len(vals_a) < 3:
            return 1.0, "ns"

        try:
            stat, pval = scipy_stats.wilcoxon(vals_a, vals_b)
        except Exception:
            return 1.0, "ns"

        if pval < 0.001:
            return pval, "***"
        elif pval < 0.01:
            return pval, "**"
        elif pval < 0.05:
            return pval, "*"
        return pval, "ns"

    # ------------------------------------------------------------------
    # Figure generation
    # ------------------------------------------------------------------

    def generate_all_figures(
        self,
        output_dir: str | Path,
        sig_pairs: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[str, str]:
        """Generate all metric group figures and save to output_dir.

        Returns a dict mapping group name to output file path.
        """
        self._ensure_loaded()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if sig_pairs is None and len(self.method_names) >= 2:
            # Default: compare everything against the last method (full model)
            full = self.method_names[-1]
            sig_pairs = [(m, full) for m in self.method_names[:-1]]

        results = {}
        for group_name, group_cfg in METRIC_GROUPS.items():
            # Check which metrics are available
            available = [
                m for m in group_cfg["metrics"]
                if m in self.long_data["metric"].values
            ]
            if not available:
                print(f"  Skipping {group_name}: no metrics available")
                continue

            try:
                outpath = output_dir / f"fig_{group_name}.pdf"
                fig = self._plot_metric_group(
                    metrics=available,
                    title=group_cfg["title"],
                    ncols=min(len(available), group_cfg["ncols"]),
                    figsize=group_cfg.get("figsize"),
                    sig_pairs=sig_pairs,
                )
                S.save_figure(fig, str(outpath))
                plt.close(fig)
                results[group_name] = str(outpath)
                print(f"  Generated: {outpath}")
            except Exception as e:
                warnings.warn(f"Failed to generate {group_name}: {e}")
                import traceback
                traceback.print_exc()

        # Summary heatmap
        try:
            outpath = output_dir / "fig_summary_heatmap.pdf"
            fig = self._plot_summary_heatmap()
            S.save_figure(fig, str(outpath))
            plt.close(fig)
            results["summary_heatmap"] = str(outpath)
            print(f"  Generated: {outpath}")
        except Exception as e:
            warnings.warn(f"Failed to generate summary heatmap: {e}")

        # Mean +/- std table
        try:
            table_path = output_dir / "mean_std_table.csv"
            self._save_summary_table(table_path)
            results["summary_table"] = str(table_path)
            print(f"  Generated: {table_path}")
        except Exception as e:
            warnings.warn(f"Failed to save summary table: {e}")

        # LaTeX table
        try:
            tex_path = output_dir / "mean_std_table.tex"
            self._save_latex_table(tex_path)
            results["latex_table"] = str(tex_path)
            print(f"  Generated: {tex_path}")
        except Exception as e:
            warnings.warn(f"Failed to save LaTeX table: {e}")

        # Statistical significance summary
        if sig_pairs:
            try:
                sig_path = output_dir / "statistical_summary.csv"
                self._save_significance_table(sig_path, sig_pairs)
                results["significance"] = str(sig_path)
                print(f"  Generated: {sig_path}")
            except Exception as e:
                warnings.warn(f"Failed to save significance: {e}")

        return results

    def _plot_metric_group(
        self,
        metrics: List[str],
        title: str,
        ncols: int,
        figsize: Optional[Tuple[float, float]] = None,
        sig_pairs: Optional[List[Tuple[str, str]]] = None,
    ):
        """Plot a group of metrics as side-by-side boxplots."""
        import seaborn as sns

        n_metrics = len(metrics)
        if figsize is None:
            figsize = (3.2 * n_metrics, 3.8)

        fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
        if n_metrics == 1:
            axes = [axes]

        for i, metric in enumerate(metrics):
            ax = axes[i]
            plot_data = self.long_data[
                self.long_data["metric"] == metric
            ].copy()

            if plot_data.empty:
                ax.set_title(S.metric_title(metric), fontsize=S.FS_TITLE)
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center")
                continue

            # Filter to known methods and enforce order
            plot_data = plot_data[
                plot_data["method"].isin(self.method_order)
            ]
            plot_data["method"] = pd.Categorical(
                plot_data["method"],
                categories=self.method_order,
                ordered=True,
            )

            # Boxplot + stripplot
            palette_list = [self.palette.get(m, "#999999")
                            for m in self.method_order
                            if m in plot_data["method"].values]
            methods_present = [m for m in self.method_order
                               if m in plot_data["method"].values]

            sns.boxplot(
                data=plot_data, x="method", y="value",
                order=methods_present,
                palette=palette_list,
                width=0.6, linewidth=0.8, fliersize=2,
                ax=ax,
            )
            sns.stripplot(
                data=plot_data, x="method", y="value",
                order=methods_present,
                color="black", size=2, alpha=0.5,
                jitter=True, ax=ax,
            )

            ax.set_title(S.metric_title(metric), fontsize=S.FS_TITLE)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(axis="x", rotation=35, labelsize=S.FS_TICK)
            ax.tick_params(axis="y", labelsize=S.FS_TICK)

            # Significance brackets
            if sig_pairs:
                self._add_significance_brackets(ax, metric, methods_present,
                                                sig_pairs, plot_data)

        fig.suptitle(title, fontsize=S.FS_TITLE + 2, fontweight="bold", y=1.02)
        plt.tight_layout()
        return fig

    def _add_significance_brackets(self, ax, metric, methods, sig_pairs, data):
        """Add significance brackets between pairs."""
        ymin, ymax = ax.get_ylim()
        y_range = ymax - ymin
        bracket_y = ymax + y_range * 0.02
        step = y_range * 0.08

        bracket_count = 0
        for ma, mb in sig_pairs:
            if ma not in methods or mb not in methods:
                continue
            pval, stars = self.compute_significance(metric, ma, mb)
            if stars == "ns":
                continue

            x1 = methods.index(ma)
            x2 = methods.index(mb)
            y = bracket_y + bracket_count * step

            ax.plot([x1, x1, x2, x2], [y, y + step * 0.3, y + step * 0.3, y],
                    lw=1.0, c="black")
            ax.text((x1 + x2) / 2, y + step * 0.35, stars,
                    ha="center", va="bottom", fontsize=S.FS_SMALL,
                    fontweight="bold")
            bracket_count += 1

        if bracket_count > 0:
            ax.set_ylim(ymin, bracket_y + (bracket_count + 0.5) * step)

    def _plot_summary_heatmap(self):
        """Generate a summary heatmap of mean metric values across methods."""
        if self.summary is None:
            raise ValueError("No summary data available")

        # Use proposed metrics
        proposed = [m for m in S.PROPOSED_METRICS
                    if m in self.long_data["metric"].values]
        if not proposed:
            proposed = [m for m in S.METRICS_CLUSTERING
                        if m in self.long_data["metric"].values]

        # Build matrix
        methods = self.method_order
        matrix = np.full((len(methods), len(proposed)), np.nan)

        for j, metric in enumerate(proposed):
            for i, method in enumerate(methods):
                mask = (
                    (self.summary["method"] == method) &
                    (self.summary["metric"] == metric)
                )
                vals = self.summary.loc[mask, "mean"]
                if not vals.empty:
                    matrix[i, j] = vals.values[0]

        fig, ax = plt.subplots(figsize=(len(proposed) * 1.5 + 2, len(methods) * 0.8 + 1.5))

        # Normalize per column for color mapping
        norm_matrix = np.copy(matrix)
        for j in range(matrix.shape[1]):
            col = matrix[:, j]
            valid = col[~np.isnan(col)]
            if len(valid) > 0:
                vmin, vmax = valid.min(), valid.max()
                if vmax > vmin:
                    norm_matrix[:, j] = (col - vmin) / (vmax - vmin)
                    # Invert for DAV (lower is better)
                    metric_name = proposed[j]
                    if metric_name in S.METRIC_DIRECTION and not S.METRIC_DIRECTION[metric_name]:
                        norm_matrix[:, j] = 1.0 - norm_matrix[:, j]

        im = ax.imshow(norm_matrix, cmap=S.HEATMAP_CMAP, aspect="auto",
                       vmin=0, vmax=1)

        # Annotate cells with actual values
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                if np.isnan(val):
                    continue
                nval = norm_matrix[i, j]
                color = "white" if nval > S.HEATMAP_DARK_THRESHOLD else "black"

                fmt = S.FMT_LARGE if abs(val) > 10 else S.FMT_SCORE_SHORT
                ax.text(j, i, f"{val:{fmt}}", ha="center", va="center",
                        fontsize=S.FS_SMALL, color=color, fontweight="bold")

        # Labels
        display_labels = [S.METRIC_DISPLAY.get(m, m) for m in proposed]
        ax.set_xticks(range(len(proposed)))
        ax.set_xticklabels(display_labels, rotation=45, ha="right",
                           fontsize=S.FS_TICK)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(
            [S.get_display_name(m) for m in methods],
            fontsize=S.FS_TICK,
        )
        ax.set_title("HSDE Benchmark Summary", fontsize=S.FS_TITLE + 1,
                      fontweight="bold", pad=10)

        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("Normalized score", fontsize=S.FS_AXIS)

        plt.tight_layout()
        return fig

    def _save_summary_table(self, path: Path):
        """Save mean +/- std table as CSV."""
        if self.summary is None:
            return

        pivot_mean = self.summary.pivot(
            index="method", columns="metric", values="mean"
        )
        pivot_std = self.summary.pivot(
            index="method", columns="metric", values="std"
        )

        # Combine into "mean +/- std" format
        combined = pivot_mean.copy()
        for col in combined.columns:
            if col in pivot_std.columns:
                combined[col] = combined[col].apply(
                    lambda x: f"{x:.3f}" if pd.notna(x) else ""
                ) + " +/- " + pivot_std[col].apply(
                    lambda x: f"{x:.3f}" if pd.notna(x) else ""
                )

        combined.to_csv(path)

    def _save_latex_table(self, path: Path):
        """Save summary as LaTeX table."""
        if self.summary is None:
            return

        # Use proposed metrics for the table
        proposed = [m for m in S.PROPOSED_METRICS
                    if m in self.summary["metric"].values]
        if not proposed:
            proposed = [m for m in S.METRICS_CLUSTERING
                        if m in self.summary["metric"].values]

        methods = self.method_order

        lines = []
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append("\\caption{HSDE benchmark results (mean $\\pm$ std across datasets)}")
        lines.append("\\label{tab:hsde_benchmark}")

        col_spec = "l" + "c" * len(proposed)
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append("\\toprule")

        header = "Method & " + " & ".join(
            S.METRIC_DISPLAY.get(m, m) for m in proposed
        ) + " \\\\"
        lines.append(header)
        lines.append("\\midrule")

        for method in methods:
            cells = [S.get_display_name(method)]
            for metric in proposed:
                mask = (
                    (self.summary["method"] == method) &
                    (self.summary["metric"] == metric)
                )
                row = self.summary.loc[mask]
                if not row.empty:
                    mean_val = row["mean"].values[0]
                    std_val = row["std"].values[0]
                    if pd.notna(mean_val):
                        cells.append(f"${mean_val:.3f} \\pm {std_val:.3f}$")
                    else:
                        cells.append("--")
                else:
                    cells.append("--")
            lines.append(" & ".join(cells) + " \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")

        with open(path, "w") as f:
            f.write("\n".join(lines))

    def _save_significance_table(self, path: Path, sig_pairs):
        """Save statistical significance results."""
        records = []
        proposed = [m for m in S.PROPOSED_METRICS
                    if m in self.long_data["metric"].values]
        if not proposed:
            proposed = [m for m in S.METRICS_CLUSTERING
                        if m in self.long_data["metric"].values]

        for ma, mb in sig_pairs:
            for metric in proposed:
                pval, stars = self.compute_significance(metric, ma, mb)
                records.append({
                    "method_a": ma,
                    "method_b": mb,
                    "metric": metric,
                    "p_value": pval,
                    "significance": stars,
                })

        pd.DataFrame(records).to_csv(path, index=False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self):
        if not self._loaded:
            self.load_all()

    def get_metric_summary(self, metric: str) -> pd.DataFrame:
        """Get mean/std for a single metric across all methods."""
        self._ensure_loaded()
        return self.summary[self.summary["metric"] == metric].copy()

    def get_available_metrics(self) -> List[str]:
        """Return list of all metrics found in the data."""
        self._ensure_loaded()
        return sorted(self.long_data["metric"].unique().tolist())

    def get_available_datasets(self) -> List[str]:
        """Return list of all loaded datasets."""
        return sorted(self.raw_data.keys())

    def __repr__(self):
        status = "loaded" if self._loaded else "not loaded"
        return (
            f"VisualizationController(results_dir={str(self.results_dir)!r}, "
            f"status={status}, datasets={len(self.raw_data)}, "
            f"methods={self.method_names})"
        )
