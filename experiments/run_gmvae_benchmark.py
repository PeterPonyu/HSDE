#!/usr/bin/env python3
"""
Experiment 2: GM-VAE Geometric Distribution Benchmark
======================================================
External benchmark using the unified GM-VAE model from external-benchmarker
skill with 5 geometric distributions, compared against the Full HSDE model.

Uses the SAME preprocessed data (2000 HVGs, 3000 cells) for all models.

Variants (6) in order: external baselines -> proposed model:
  1. GM-VAE (Euclidean)      — standard Euclidean latent space
  2. GM-VAE (Poincare)       — Poincare ball (hyperbolic)
  3. GM-VAE (PGM)            — Product Gaussian Manifold
  4. GM-VAE (LearnablePGM)   — PGM with learnable curvature
  5. GM-VAE (HW)             — Hyperboloid Wrapped Normal (Lorentz)
  6. HSDE (Full)             — Full HSDE (Graph GAT + all components)

Each variant: 200 epochs x 12 datasets.
"""

import sys, os, gc, time, traceback, types
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from experiments.exp_utils import (
    discover_datasets, get_labels, load_and_preprocess,
    get_dense_X, evaluate_latent, get_done_datasets
)
from src.agent import HSDE

# External benchmarker models (package import via module alias)
BENCHMARKER_DIR = os.path.expanduser('~/.copilot/skills/external-benchmarker')
sys.path.insert(0, os.path.dirname(BENCHMARKER_DIR))
_pkg = types.ModuleType('external_benchmarker')
_pkg.__path__ = [BENCHMARKER_DIR]
_pkg.__package__ = 'external_benchmarker'
sys.modules['external_benchmarker'] = _pkg
from external_benchmarker.unified_models import create_gmvae_model

# ── Configuration ──
EPOCHS = 200
EXPERIMENT = 'gmvae_benchmark'
PREFIX = 'gmvae'
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'HSDE_results', EXPERIMENT)
TABLES_DIR = os.path.join(RESULTS_DIR, 'tables')
os.makedirs(TABLES_DIR, exist_ok=True)

# GM-VAE geometric distributions
GMVAE_DISTRIBUTIONS = ['euclidean', 'poincare', 'pgm', 'learnable_pgm', 'hw']

# Full HSDE config (the proposed model to compare against)
HSDE_FULL_CONFIG = dict(
    recon=1.0, irecon=1.0, lorentz=5.0, beta=1.0,
    encoder_type='graph', graph_type='GAT',
    use_sde=True, use_pde=True,
    vae_reg=0.5, sde_reg=0.5, pde_reg=0.2,
)


def train_gmvae(X_dense, distribution, epochs, batch_size=128):
    """Train an external GM-VAE model on preprocessed normalized HVG data."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = X_dense.shape[1]
    latent_dim = 10

    model = create_gmvae_model(
        input_dim=input_dim, latent_dim=latent_dim,
        distribution=distribution, hidden_dims=[512, 256],
    )
    model = model.to(device)

    X_t = torch.FloatTensor(X_dense)
    dataset = TensorDataset(X_t)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, drop_last=True)

    model.fit(loader, epochs=epochs, lr=1e-3)

    model.eval()
    with torch.no_grad():
        z = model.encode(X_t.to(device)).cpu().numpy()

    del model
    torch.cuda.empty_cache()
    return z


def run_gmvae_variant(X_dense, distribution, dataset_name):
    """Train one GM-VAE distribution variant."""
    label = f"GM-VAE ({distribution.replace('_', '-')})"
    print(f"  Training {label}...")

    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()

        latent = train_gmvae(X_dense, distribution, EPOCHS)

        train_time = time.time() - t0
        peak_mem = (torch.cuda.max_memory_allocated() / 1e9
                    if torch.cuda.is_available() else 0)

        print(f"    ✓ {label}: time={train_time:.1f}s")
        return latent, train_time, peak_mem

    except Exception as e:
        print(f"    ✗ {label} FAILED: {e}")
        traceback.print_exc()
        torch.cuda.empty_cache()
        return None, 0, 0


def run_hsde_full(adata1, dataset_name):
    """Train the full HSDE model on preprocessed adata1."""
    print(f"  Training HSDE (Full)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        model = HSDE(
            adata1, layer='counts',
            hidden_dim=128, latent_dim=10, i_dim=2,
            lr=1e-4, loss_type='nb', device=device,
            **HSDE_FULL_CONFIG
        )
        model.fit(epochs=EPOCHS, patience=30, early_stop=True,
                  compute_metrics=False)
        latent = model.get_latent()
        res = model.get_resource_metrics()

        print(f"    ✓ HSDE (Full): time={res['train_time']:.1f}s")

        del model
        torch.cuda.empty_cache()
        return latent, res['train_time'], res['peak_memory_gb']

    except Exception as e:
        print(f"    ✗ HSDE (Full) FAILED: {e}")
        traceback.print_exc()
        torch.cuda.empty_cache()
        return None, 0, 0


def main():
    datasets = discover_datasets()
    done = get_done_datasets(TABLES_DIR, PREFIX)

    method_names = ([f"GM-VAE ({d.replace('_', '-')})"
                     for d in GMVAE_DISTRIBUTIONS] +
                    ['HSDE (Full)'])

    print(f"\n{'='*70}")
    print(f"GM-VAE GEOMETRIC DISTRIBUTION BENCHMARK")
    print(f"Variants: {method_names}")
    print(f"Datasets: {len(datasets)} ({len(done)} already done)")
    print(f"Epochs: {EPOCHS}")
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

        labels = get_labels(adata1)
        X_dense = get_dense_X(adata1)

        all_metrics = []

        # 1. GM-VAE with each distribution (external baselines)
        for dist in GMVAE_DISTRIBUTIONS:
            latent, t, mem = run_gmvae_variant(X_dense, dist, dataset_name)
            if latent is not None:
                metrics = evaluate_latent(latent, labels)
                metrics['train_time'] = t
                metrics['peak_memory_gb'] = mem
                all_metrics.append(metrics)
                print(f"    → ARI={metrics.get('ARI', 0):.3f}, "
                      f"NMI={metrics.get('NMI', 0):.3f}")
            else:
                all_metrics.append({})

        # 2. Full HSDE (proposed model — last row)
        latent, t, mem = run_hsde_full(adata1, dataset_name)
        if latent is not None:
            metrics = evaluate_latent(latent, labels)
            metrics['train_time'] = t
            metrics['peak_memory_gb'] = mem
            all_metrics.append(metrics)
            print(f"    → ARI={metrics.get('ARI', 0):.3f}, "
                  f"NMI={metrics.get('NMI', 0):.3f}")
        else:
            all_metrics.append({})

        df = pd.DataFrame(all_metrics, index=method_names)
        csv_path = os.path.join(TABLES_DIR, f'{PREFIX}_{dataset_name}_df.csv')
        df.to_csv(csv_path, index_label='method')
        print(f"  Saved: {csv_path}")

        del adata1
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print(f"GM-VAE BENCHMARK COMPLETE — Results in: {TABLES_DIR}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
