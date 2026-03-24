#!/usr/bin/env python3
"""
HSDE: Run All Visualizations
==============================

Fully automatic script that discovers all HSDE benchmark results and
generates publication-quality figures for every experiment series.

Covers:
  1. Ablation study (VAE -> IRecon-VAE -> Lorentz-VAE -> GM-VAE -> HSDE)
  2. GM-VAE geometric benchmark (5 external GM-VAE + HSDE)
  3. Disentanglement regularization (VAE, beta-VAE, DIP, TC, Info, HSDE)
  4. Cross-dataset benchmark (MLP, GAT, GAT+IB, GAT+IB+Lor, etc.)

For each series, generates:
  - Clustering boxplots (NMI, ARI, ASW, DAV, CAL, COR)
  - DRE UMAP series (distance_correlation, Q_local, Q_global, overall)
  - DRE t-SNE series
  - LSE intrinsic series (7 metrics)
  - DREX extended DR series (7 metrics)
  - LSEX extended latent series (5 metrics)
  - Summary heatmap
  - Mean +/- std tables (CSV + LaTeX)
  - Statistical significance tables

Usage:
    python -m hsde.viz.run_all_visualizations
"""

import os
import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hsde.viz.controller import VisualizationController
from hsde.viz import style as S


def run_experiment_visualization(
    tables_dir, output_dir, method_names, method_order=None,
    palette=None, experiment_name="Experiment"
):
    """Run visualization for one experiment series."""
    tables_path = Path(tables_dir)
    if not tables_path.exists():
        print(f"  Skipping {experiment_name}: {tables_path} not found")
        return None

    csv_count = len(list(tables_path.glob("*.csv")))
    if csv_count == 0:
        print(f"  Skipping {experiment_name}: no CSV files in {tables_path}")
        return None

    print(f"\n{'='*70}")
    print(f"  {experiment_name}")
    print(f"  Tables dir: {tables_dir}")
    print(f"  Methods: {method_names}")
    print(f"  Datasets: {csv_count} CSV files")
    print(f"{'='*70}")

    # Default significance pairs: each method vs the last (full model)
    sig_pairs = [(m, method_names[-1]) for m in method_names[:-1]]

    ctrl = VisualizationController(
        results_dir=tables_dir,
        method_names=method_names,
        method_order=method_order or method_names,
        palette=palette,
    )
    ctrl.load_all()

    print(f"\n  Available metrics: {len(ctrl.get_available_metrics())}")
    print(f"  Available datasets: {ctrl.get_available_datasets()}")

    results = ctrl.generate_all_figures(
        output_dir=output_dir,
        sig_pairs=sig_pairs,
    )

    print(f"\n  Generated {len(results)} outputs for {experiment_name}")
    return ctrl


def main():
    results_base = PROJECT_ROOT / "HSDE_results"
    benchmark_base = PROJECT_ROOT / "benchmark_results"

    print(f"\n{'#'*70}")
    print(f"  HSDE: Automatic Visualization System")
    print(f"  Project: {PROJECT_ROOT}")
    print(f"{'#'*70}")

    # ══════════════════════════════════════════════
    # Experiment 1: Ablation (5 methods)
    # ══════════════════════════════════════════════
    ablation_methods = ["VAE", "IRecon-VAE", "Lorentz-VAE", "GM-VAE", "HSDE"]
    ablation_palette = dict(zip(ablation_methods, [
        "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"
    ]))
    run_experiment_visualization(
        tables_dir=results_base / "ablation" / "tables",
        output_dir=results_base / "ablation" / "figures",
        method_names=ablation_methods,
        palette=ablation_palette,
        experiment_name="Ablation Study",
    )

    # ══════════════════════════════════════════════
    # Experiment 2: GM-VAE Benchmark (6 methods)
    # ══════════════════════════════════════════════
    gmvae_methods = [
        "GM-VAE (Eucl.)", "GM-VAE (Poinc.)", "GM-VAE (PGM)",
        "GM-VAE (L-PGM)", "GM-VAE (HW)", "HSDE"
    ]
    gmvae_palette = dict(zip(gmvae_methods, [
        "#0072B2", "#56B4E9", "#E69F00", "#009E73", "#CC79A7", "#D55E00"
    ]))
    run_experiment_visualization(
        tables_dir=results_base / "gmvae_benchmark" / "tables",
        output_dir=results_base / "gmvae_benchmark" / "figures",
        method_names=gmvae_methods,
        palette=gmvae_palette,
        experiment_name="GM-VAE Geometric Benchmark",
    )

    # ══════════════════════════════════════════════
    # Experiment 3: Disentanglement (6 methods)
    # ══════════════════════════════════════════════
    disent_methods = ["VAE", "beta-VAE", "DIP-VAE", "TC-VAE", "InfoVAE", "HSDE"]
    disent_palette = dict(zip(disent_methods, [
        "#0072B2", "#56B4E9", "#E69F00", "#009E73", "#CC79A7", "#D55E00"
    ]))
    run_experiment_visualization(
        tables_dir=results_base / "disentanglement" / "tables",
        output_dir=results_base / "disentanglement" / "figures",
        method_names=disent_methods,
        palette=disent_palette,
        experiment_name="Disentanglement Regularization",
    )

    # ══════════════════════════════════════════════
    # Experiment 4: Cross-dataset Benchmark
    # ══════════════════════════════════════════════
    if (benchmark_base / "tables").exists():
        # Detect method names from first CSV
        import pandas as pd
        csv_files = list((benchmark_base / "tables").glob("*.csv"))
        if csv_files:
            first_df = pd.read_csv(csv_files[0], index_col=0)
            bench_methods = list(first_df.index)
            bench_palette = dict(zip(bench_methods, [
                "#0072B2", "#56B4E9", "#E69F00", "#009E73",
                "#CC79A7", "#D55E00", "#F0E442", "#999999",
            ][:len(bench_methods)]))

            run_experiment_visualization(
                tables_dir=benchmark_base / "tables",
                output_dir=benchmark_base / "figures_auto",
                method_names=bench_methods,
                palette=bench_palette,
                experiment_name="Cross-Dataset Benchmark",
            )

    print(f"\n{'#'*70}")
    print(f"  ALL VISUALIZATIONS COMPLETE")
    print(f"{'#'*70}")

    # Print output locations
    for d in [results_base / "ablation" / "figures",
              results_base / "gmvae_benchmark" / "figures",
              results_base / "disentanglement" / "figures",
              benchmark_base / "figures_auto"]:
        if d.exists():
            n_files = len(list(d.glob("*")))
            print(f"  {d}: {n_files} files")


if __name__ == "__main__":
    main()
