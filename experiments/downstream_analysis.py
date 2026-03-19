#!/usr/bin/env python3
"""
HSDE Downstream Analysis
===========================

Comprehensive downstream analysis pipeline that:
1. Runs all 5 ablation variants on available local datasets
2. Computes full DRE + LSE + clustering metrics via MoCoO evaluators
3. Generates cross-dataset comparison tables
4. Produces statistical significance tests (Wilcoxon signed-rank)
5. Generates publication-quality figures via the HSDE visualization controller
6. Computes per-component effectiveness analysis
7. Produces a final report

Datasets used (from data/ directory):
  - Pancreas (endocrinogenesis_day15.h5ad)
  - DentateGyrus (10X43_1.h5ad)
  - Gastrulation (erythroid_lineage.h5ad)
  - BoneMarrow (human_cd34_bone_marrow.h5ad)

Usage:
    python experiments/downstream_analysis.py
"""

import sys
import os
import gc
import json
import traceback
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import scanpy as sc
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hsde import HSDE

# ---------------------------------------------------------------------------
# MoCoO evaluators (DRE + LSE series)
# ---------------------------------------------------------------------------
MOCOO_EVAL_DIR = Path(os.environ.get(
    "HSDE_MOCOO_DIR",
    str(Path.home() / "Desktop" / "MoCoO" / "mocoo" / "evaluation"),
))
sys.path.insert(0, str(MOCOO_EVAL_DIR.parent.parent))

try:
    from mocoo.evaluation.dre import DimensionalityReductionEvaluator
    from mocoo.evaluation.lse import SingleCellLatentSpaceEvaluator
    HAS_MOCOO = True
except (ImportError, ModuleNotFoundError):
    HAS_MOCOO = False
    DimensionalityReductionEvaluator = None
    SingleCellLatentSpaceEvaluator = None

# Extended evaluators
try:
    from mocoo.evaluation.drex import evaluate_extended_dr
    HAS_DREX = True
except ImportError:
    HAS_DREX = False

try:
    from mocoo.evaluation.lsex import evaluate_extended_latent
    HAS_LSEX = True
except ImportError:
    HAS_LSEX = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_CELLS = 3000
N_HVG = 2000
SEED = 42
EPOCHS = 200
PATIENCE = 30

RESULTS_DIR = PROJECT_ROOT / "HSDE_results" / "downstream"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

# Local datasets
LOCAL_DATASETS = {
    "Pancreas": PROJECT_ROOT / "data" / "Pancreas" / "endocrinogenesis_day15.h5ad",
    "DentateGyrus": PROJECT_ROOT / "data" / "DentateGyrus" / "10X43_1.h5ad",
    "Gastrulation": PROJECT_ROOT / "data" / "Gastrulation" / "erythroid_lineage.h5ad",
    "BoneMarrow": PROJECT_ROOT / "data" / "BoneMarrow" / "human_cd34_bone_marrow.h5ad",
}

# Model variants
VARIANTS = {
    "VAE": dict(
        recon=1.0, irecon=0.0, lorentz=0.0, beta=1.0,
        encoder_type="mlp",
    ),
    "IRecon-VAE": dict(
        recon=1.0, irecon=1.0, lorentz=0.0, beta=1.0,
        encoder_type="mlp",
    ),
    "Lorentz-VAE": dict(
        recon=1.0, irecon=0.0, lorentz=5.0, beta=1.0,
        encoder_type="mlp",
    ),
    "GM-VAE": dict(
        recon=1.0, irecon=1.0, lorentz=5.0, beta=1.0,
        encoder_type="mlp",
    ),
    "HSDE (Full)": dict(
        recon=1.0, irecon=1.0, lorentz=5.0, beta=1.0,
        encoder_type="graph", graph_type="GAT",
        use_sde=True, use_pde=True,
        vae_reg=0.5, sde_reg=0.5, pde_reg=0.2,
        n_neighbors=15,
    ),
}


# ---------------------------------------------------------------------------
# Data loading (matching exp_utils.py pipeline)
# ---------------------------------------------------------------------------

def get_labels(adata):
    """Extract cell type labels from adata.obs."""
    for col in ["cell_type", "celltype", "CellType", "cell_types",
                "cluster", "clusters", "louvain", "leiden",
                "annotation", "label", "labels", "Group", "group"]:
        if col in adata.obs.columns:
            labels = adata.obs[col].values
            if hasattr(labels, "cat"):
                labels = labels.astype(str)
            return labels
    print("  No labels found, computing leiden clusters...")
    sc.pp.neighbors(adata, use_rep="X_pca" if "X_pca" in adata.obsm else None)
    sc.tl.leiden(adata, resolution=1.0)
    return adata.obs["leiden"].values


def load_and_preprocess(filepath):
    """Load and preprocess dataset (same pipeline as exp_utils)."""
    adata = sc.read(str(filepath))
    adata.obs_names_make_unique()
    adata.var_names_make_unique()

    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)

    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG)

    rng = np.random.default_rng(SEED)
    if adata.shape[0] > MAX_CELLS:
        idxs = rng.choice(adata.shape[0], MAX_CELLS, replace=False)
    else:
        idxs = rng.permutation(adata.shape[0])

    adata1 = adata[idxs, adata.var["highly_variable"]].copy()
    print(f"  Preprocessed: {adata.n_obs} -> {adata1.n_obs} cells, "
          f"{adata.n_vars} -> {adata1.n_vars} HVGs")
    return adata1


# ---------------------------------------------------------------------------
# Metrics computation using MoCoO evaluators
# ---------------------------------------------------------------------------

def compute_clustering_metrics(latent, labels):
    """Compute clustering metrics (NMI, ARI, ASW, DAV, CAL, COR)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import (
        normalized_mutual_info_score, adjusted_rand_score,
        silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    )
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    labels_int = le.fit_transform(np.asarray(labels).astype(str))
    n_clusters = len(np.unique(labels_int))

    pred = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(latent)

    metrics = {
        "NMI": normalized_mutual_info_score(labels_int, pred),
        "ARI": adjusted_rand_score(labels_int, pred),
    }

    try:
        metrics["ASW"] = silhouette_score(latent, pred) if len(np.unique(pred)) > 1 else 0.0
    except Exception:
        metrics["ASW"] = 0.0
    try:
        metrics["DAV"] = davies_bouldin_score(latent, pred)
    except Exception:
        metrics["DAV"] = float("nan")
    try:
        metrics["CAL"] = calinski_harabasz_score(latent, pred)
    except Exception:
        metrics["CAL"] = 0.0
    try:
        acorr = np.abs(np.corrcoef(latent.T))
        metrics["COR"] = float(acorr.sum(axis=1).mean() - 1)
    except Exception:
        metrics["COR"] = 0.0

    return metrics


def compute_dre_metrics(latent, verbose=False):
    """Compute DRE series using MoCoO DimensionalityReductionEvaluator."""
    import umap
    from sklearn.manifold import TSNE

    metrics = {}

    # UMAP projection
    try:
        reducer = umap.UMAP(n_components=2, random_state=42)
        X_umap = reducer.fit_transform(latent)
        dre = DimensionalityReductionEvaluator(verbose=verbose)
        dre_results = dre.comprehensive_evaluation(latent, X_umap, k=15)
        metrics["DRE_umap_distance_correlation"] = dre_results["distance_correlation"]
        metrics["DRE_umap_Q_local"] = dre_results["Q_local"]
        metrics["DRE_umap_Q_global"] = dre_results["Q_global"]
        metrics["DRE_umap_overall_quality"] = dre_results["overall_quality"]
    except Exception as e:
        print(f"    DRE UMAP failed: {e}")
        for k in ["DRE_umap_distance_correlation", "DRE_umap_Q_local",
                   "DRE_umap_Q_global", "DRE_umap_overall_quality"]:
            metrics[k] = 0.0

    # t-SNE projection
    try:
        X_tsne = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(latent)
        dre = DimensionalityReductionEvaluator(verbose=verbose)
        dre_results = dre.comprehensive_evaluation(latent, X_tsne, k=15)
        metrics["DRE_tsne_distance_correlation"] = dre_results["distance_correlation"]
        metrics["DRE_tsne_Q_local"] = dre_results["Q_local"]
        metrics["DRE_tsne_Q_global"] = dre_results["Q_global"]
        metrics["DRE_tsne_overall_quality"] = dre_results["overall_quality"]
    except Exception as e:
        print(f"    DRE t-SNE failed: {e}")
        for k in ["DRE_tsne_distance_correlation", "DRE_tsne_Q_local",
                   "DRE_tsne_Q_global", "DRE_tsne_overall_quality"]:
            metrics[k] = 0.0

    return metrics


def compute_lse_metrics(latent, verbose=False):
    """Compute LSE series using MoCoO SingleCellLatentSpaceEvaluator."""
    metrics = {}
    try:
        lse = SingleCellLatentSpaceEvaluator(data_type="trajectory", verbose=verbose)
        lse_results = lse.comprehensive_evaluation(latent)
        metrics["LSE_manifold_dimensionality"] = lse_results["manifold_dimensionality"]
        metrics["LSE_spectral_decay_rate"] = lse_results["spectral_decay_rate"]
        metrics["LSE_participation_ratio"] = lse_results["participation_ratio"]
        metrics["LSE_anisotropy_score"] = lse_results["anisotropy_score"]
        metrics["LSE_noise_resilience"] = lse_results["noise_resilience"]
        metrics["LSE_core_quality"] = lse_results["core_quality"]
        metrics["LSE_overall_quality"] = lse_results["overall_quality"]
    except Exception as e:
        print(f"    LSE failed: {e}")
        for k in ["LSE_manifold_dimensionality", "LSE_spectral_decay_rate",
                   "LSE_participation_ratio", "LSE_anisotropy_score",
                   "LSE_noise_resilience", "LSE_core_quality", "LSE_overall_quality"]:
            metrics[k] = 0.0

    return metrics


def compute_all_metrics(latent, labels):
    """Compute full metric battery: clustering + DRE + LSE."""
    metrics = {}
    metrics.update(compute_clustering_metrics(latent, labels))
    metrics.update(compute_dre_metrics(latent))
    metrics.update(compute_lse_metrics(latent))
    return metrics


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_variant(adata1, variant_name, params, dataset_name):
    """Train one HSDE variant and return metrics."""
    print(f"  Training {variant_name} on {dataset_name}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = HSDE(
            adata1, layer="counts",
            hidden_dim=128, latent_dim=10, i_dim=2,
            lr=1e-4, loss_type="nb",
            device=device,
            **params,
        )
        model.fit(epochs=EPOCHS, patience=PATIENCE, early_stop=True,
                  compute_metrics=False)

        latent = model.get_latent()
        labels = get_labels(adata1)
        metrics = compute_all_metrics(latent, labels)

        res = model.get_resource_metrics()
        metrics["train_time"] = res["train_time"]
        metrics["peak_memory_gb"] = res["peak_memory_gb"]
        metrics["actual_epochs"] = res["actual_epochs"]

        print(f"    {variant_name}: ARI={metrics.get('ARI', 0):.3f}, "
              f"NMI={metrics.get('NMI', 0):.3f}, "
              f"DRE={metrics.get('DRE_umap_overall_quality', 0):.3f}, "
              f"LSE={metrics.get('LSE_overall_quality', 0):.3f}, "
              f"time={res['train_time']:.1f}s")

        del model
        torch.cuda.empty_cache()
        return metrics

    except Exception as e:
        print(f"    {variant_name} FAILED: {e}")
        traceback.print_exc()
        torch.cuda.empty_cache()
        return None


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def compute_component_effectiveness(all_results):
    """Analyze the contribution of each architectural component."""
    records = []
    proposed = ["NMI", "ARI", "ASW", "DRE_umap_overall_quality", "LSE_overall_quality"]

    for dataset, df in all_results.items():
        for metric in proposed:
            if metric not in df.columns:
                continue

            base_val = df.loc["VAE", metric] if "VAE" in df.index else 0

            for variant in ["IRecon-VAE", "Lorentz-VAE", "GM-VAE", "HSDE (Full)"]:
                if variant not in df.index:
                    continue
                var_val = df.loc[variant, metric]
                delta = var_val - base_val

                records.append({
                    "dataset": dataset,
                    "variant": variant,
                    "metric": metric,
                    "value": var_val,
                    "delta_vs_vae": delta,
                    "relative_change": delta / abs(base_val) if abs(base_val) > 1e-10 else 0,
                })

    return pd.DataFrame(records)


def generate_report(all_results, effectiveness_df, output_dir):
    """Generate a comprehensive analysis report."""
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("HSDE DOWNSTREAM ANALYSIS REPORT")
    report_lines.append("=" * 70)
    report_lines.append("")

    # Summary statistics
    report_lines.append("1. CROSS-DATASET SUMMARY")
    report_lines.append("-" * 40)

    for variant in VARIANTS.keys():
        aris, nmis, dres, lses = [], [], [], []
        for dataset, df in all_results.items():
            if variant in df.index:
                aris.append(df.loc[variant, "ARI"])
                nmis.append(df.loc[variant, "NMI"])
                if "DRE_umap_overall_quality" in df.columns:
                    dres.append(df.loc[variant, "DRE_umap_overall_quality"])
                if "LSE_overall_quality" in df.columns:
                    lses.append(df.loc[variant, "LSE_overall_quality"])

        if aris:
            report_lines.append(
                f"  {variant:20s}: "
                f"ARI={np.mean(aris):.3f}+/-{np.std(aris):.3f}  "
                f"NMI={np.mean(nmis):.3f}+/-{np.std(nmis):.3f}  "
                f"DRE={np.mean(dres):.3f}+/-{np.std(dres):.3f}  "
                f"LSE={np.mean(lses):.3f}+/-{np.std(lses):.3f}"
            )

    # Component effectiveness
    report_lines.append("")
    report_lines.append("2. COMPONENT EFFECTIVENESS (vs VAE baseline)")
    report_lines.append("-" * 40)

    if not effectiveness_df.empty:
        for variant in ["IRecon-VAE", "Lorentz-VAE", "GM-VAE", "HSDE (Full)"]:
            vdf = effectiveness_df[effectiveness_df["variant"] == variant]
            if vdf.empty:
                continue
            mean_delta = vdf.groupby("metric")["delta_vs_vae"].mean()
            report_lines.append(f"  {variant}:")
            for metric, delta in mean_delta.items():
                direction = "+" if delta > 0 else ""
                report_lines.append(f"    {metric:30s}: {direction}{delta:.4f}")

    # Significance tests
    report_lines.append("")
    report_lines.append("3. STATISTICAL SIGNIFICANCE (Wilcoxon signed-rank)")
    report_lines.append("-" * 40)

    from scipy.stats import wilcoxon
    full_model = "HSDE (Full)"
    for variant in ["VAE", "IRecon-VAE", "Lorentz-VAE", "GM-VAE"]:
        for metric in ["ARI", "NMI"]:
            vals_a, vals_b = [], []
            for dataset, df in all_results.items():
                if variant in df.index and full_model in df.index and metric in df.columns:
                    vals_a.append(df.loc[variant, metric])
                    vals_b.append(df.loc[full_model, metric])

            if len(vals_a) >= 3:
                try:
                    stat, pval = wilcoxon(vals_a, vals_b)
                    stars = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
                    report_lines.append(
                        f"  {variant:20s} vs {full_model:15s} ({metric}): "
                        f"p={pval:.4f} {stars}"
                    )
                except Exception:
                    report_lines.append(
                        f"  {variant:20s} vs {full_model:15s} ({metric}): "
                        f"insufficient data"
                    )

    report_lines.append("")
    report_lines.append("=" * 70)

    report_text = "\n".join(report_lines)
    print(report_text)

    report_path = output_dir / "downstream_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\nReport saved: {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not HAS_MOCOO:
        print("ERROR: MoCoO evaluation package not found.")
        print(f"  Expected at: {MOCOO_EVAL_DIR}")
        print("  Set HSDE_MOCOO_DIR env var to the correct evaluation directory.")
        sys.exit(1)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    method_names = list(VARIANTS.keys())

    # Check which datasets are available
    available = {name: path for name, path in LOCAL_DATASETS.items()
                 if path.exists()}

    if not available:
        print("No local datasets found. Exiting.")
        return

    print(f"\n{'='*70}")
    print(f"HSDE DOWNSTREAM ANALYSIS")
    print(f"Variants: {method_names}")
    print(f"Datasets: {list(available.keys())}")
    print(f"Epochs: {EPOCHS}, Patience: {PATIENCE}")
    print(f"{'='*70}\n")

    # Check which datasets are already done
    done = set()
    for f in TABLES_DIR.glob("downstream_*_df.csv"):
        name = f.stem.replace("downstream_", "").replace("_df", "")
        done.add(name)

    all_results = {}

    for dataset_name, filepath in available.items():
        if dataset_name in done:
            print(f"  Loading cached: {dataset_name}")
            df = pd.read_csv(TABLES_DIR / f"downstream_{dataset_name}_df.csv",
                             index_col=0)
            all_results[dataset_name] = df
            continue

        print(f"\n{'─'*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'─'*60}")

        try:
            adata1 = load_and_preprocess(filepath)
        except Exception as e:
            print(f"  Failed to preprocess: {e}")
            traceback.print_exc()
            continue

        all_metrics = []
        for variant_name, params in VARIANTS.items():
            metrics = train_variant(adata1, variant_name, params, dataset_name)
            all_metrics.append(metrics if metrics else {})

        df = pd.DataFrame(all_metrics, index=method_names)
        csv_path = TABLES_DIR / f"downstream_{dataset_name}_df.csv"
        df.to_csv(csv_path, index_label="method")
        print(f"  Saved: {csv_path}")

        all_results[dataset_name] = df

        del adata1
        gc.collect()
        torch.cuda.empty_cache()

    # ── Component effectiveness analysis ──
    print(f"\n{'='*70}")
    print("COMPONENT EFFECTIVENESS ANALYSIS")
    print(f"{'='*70}")
    effectiveness_df = compute_component_effectiveness(all_results)
    if not effectiveness_df.empty:
        effectiveness_path = RESULTS_DIR / "component_effectiveness.csv"
        effectiveness_df.to_csv(effectiveness_path, index=False)
        print(f"  Saved: {effectiveness_path}")

    # ── Generate report ──
    generate_report(all_results, effectiveness_df, RESULTS_DIR)

    # ── Generate figures via HSDE visualization controller ──
    print(f"\n{'='*70}")
    print("GENERATING FIGURES")
    print(f"{'='*70}")

    try:
        from hsde.viz.controller import VisualizationController

        ctrl = VisualizationController(
            results_dir=TABLES_DIR,
            method_names=method_names,
            method_order=method_names,
            palette=dict(zip(method_names, [
                "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"
            ])),
        )
        ctrl.load_all()
        fig_results = ctrl.generate_all_figures(
            output_dir=FIGURES_DIR,
            sig_pairs=[(m, "HSDE (Full)") for m in method_names[:-1]],
        )
        print(f"\n  Generated {len(fig_results)} figure outputs")
    except Exception as e:
        print(f"  Visualization failed: {e}")
        traceback.print_exc()

    # ── Save combined results ──
    combined_path = RESULTS_DIR / "combined_results.json"
    combined = {}
    for dataset, df in all_results.items():
        combined[dataset] = df.to_dict(orient="index")
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    print(f"  Combined results: {combined_path}")

    print(f"\n{'='*70}")
    print("DOWNSTREAM ANALYSIS COMPLETE")
    print(f"  Tables:  {TABLES_DIR}")
    print(f"  Figures: {FIGURES_DIR}")
    print(f"  Report:  {RESULTS_DIR / 'downstream_report.txt'}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
