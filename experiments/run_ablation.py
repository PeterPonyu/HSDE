#!/usr/bin/env python3
"""
Experiment 1: HSDE Ablation Study
==================================
Additive ablation of HSDE core architecture components.
Full model = Graph GAT + IRecon + Lorentz (β=0.1).

Variants (5) in logical architecture order:
  1. Base VAE    — MLP base VAE
  2. VAE+IB      — + information bottleneck
  3. VAE+Hyp     — + hyperbolic geometry
  4. VAE+IB+Hyp  — + both IB + hyperbolic
  5. HSDE        — Graph GAT + IB + Hyp (β=0.1)

Preprocessing: normalize, log1p, 2000 HVGs, subsample 3000 cells.
Each variant: 200 epochs x 12 datasets.
"""

import sys, os, gc, traceback
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from experiments.exp_utils import (
    discover_datasets, get_labels, load_and_preprocess,
    evaluate_latent, get_done_datasets
)
from hsde import HSDE

# ── Configuration ──
EPOCHS = 200
EXPERIMENT = 'ablation'
PREFIX = 'ablation'
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'HSDE_results', EXPERIMENT)
TABLES_DIR = os.path.join(RESULTS_DIR, 'tables')
os.makedirs(TABLES_DIR, exist_ok=True)

# Variants in logical architecture order (base -> additive -> full)
VARIANTS = {
    'Base VAE': dict(
        recon=1.0, irecon=0.0, lorentz=0.0, beta=1.0,
        encoder_type='mlp',
    ),
    'VAE+IB': dict(
        recon=1.0, irecon=1.0, lorentz=0.0, beta=1.0,
        encoder_type='mlp',
    ),
    'VAE+Hyp': dict(
        recon=1.0, irecon=0.0, lorentz=5.0, beta=1.0,
        encoder_type='mlp',
    ),
    'VAE+IB+Hyp': dict(
        recon=1.0, irecon=1.0, lorentz=5.0, beta=1.0,
        encoder_type='mlp',
    ),
    'HSDE': dict(
        recon=1.0, irecon=0.5, lorentz=5.0, beta=0.1,
        encoder_type='graph', graph_type='GAT',
    ),
}


def run_single(adata1, variant_name, params, dataset_name):
    """Train one HSDE variant on preprocessed adata1."""
    print(f"  Training {variant_name} on {dataset_name}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        model = HSDE(
            adata1, layer='counts',
            hidden_dim=128, latent_dim=10, i_dim=2,
            lr=1e-4, loss_type='nb',
            device=device,
            **params
        )
        model.fit(epochs=EPOCHS, patience=30, early_stop=True,
                  compute_metrics=False)
        latent = model.get_latent()
        labels, _ = get_labels(adata1)
        metrics = evaluate_latent(latent, labels)

        res = model.get_resource_metrics()
        metrics['train_time'] = res['train_time']
        metrics['peak_memory_gb'] = res['peak_memory_gb']
        metrics['actual_epochs'] = res['actual_epochs']

        print(f"    ✓ {variant_name}: ARI={metrics.get('ARI', 0):.3f}, "
              f"NMI={metrics.get('NMI', 0):.3f}, time={res['train_time']:.1f}s")

        del model
        torch.cuda.empty_cache()
        return metrics

    except Exception as e:
        print(f"    ✗ {variant_name} FAILED: {e}")
        traceback.print_exc()
        torch.cuda.empty_cache()
        return None


def main():
    datasets = discover_datasets()
    done = get_done_datasets(TABLES_DIR, PREFIX)
    method_names = list(VARIANTS.keys())

    print(f"\n{'='*70}")
    print(f"HSDE ABLATION STUDY")
    print(f"Variants: {method_names}")
    print(f"Datasets: {len(datasets)} ({len(done)} already done)")
    print(f"Epochs: {EPOCHS}, compute_metrics=False")
    print(f"Preprocessing: {2000} HVGs, max {3000} cells")
    print(f"{'='*70}\n")

    for filepath in datasets:
        dataset_name = os.path.basename(filepath).replace('.h5ad', '')
        if dataset_name in done:
            print(f"  Skipping {dataset_name} (already done)")
            continue

        print(f"\n{'─'*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'─'*60}")

        try:
            adata1 = load_and_preprocess(filepath)
        except Exception as e:
            print(f"  ✗ Failed to preprocess: {e}")
            traceback.print_exc()
            continue

        all_metrics = []
        for variant_name, params in VARIANTS.items():
            metrics = run_single(adata1, variant_name, params, dataset_name)
            all_metrics.append(metrics if metrics else {})

        df = pd.DataFrame(all_metrics, index=method_names)
        csv_path = os.path.join(TABLES_DIR, f'{PREFIX}_{dataset_name}_df.csv')
        df.to_csv(csv_path, index_label='method')
        print(f"  Saved: {csv_path}")

        del adata1
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print(f"ABLATION COMPLETE — Results in: {TABLES_DIR}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
