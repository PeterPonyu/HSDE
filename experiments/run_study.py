#!/usr/bin/env python3
"""
Single-Dataset Deep Study: Ablation & Component Efficiency
===========================================================
16 configurations in 3 parts on Setty Bone Marrow hematopoiesis data.

Part 1: Encoder Architecture Comparison (5 encoders, identical loss)
Part 2: Component Effectiveness (6 configs, additive from minimal GAT)
Part 3: Ablation Study (5 configs, subtractive from full GAT)

All evaluation uses unsupervised Leiden clustering as the reference
partition — no ground-truth cell type annotations.

Usage:
    python experiments/run_study.py --part all
    python experiments/run_study.py --part 1 --epochs 50
"""

import sys, os, gc, time, json, argparse, traceback
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from experiments.exp_utils import (
    load_and_preprocess, get_labels, evaluate_latent
)
from hsde import HSDE

# ── Defaults ──
DEFAULT_DATASET = os.path.join(
    PROJECT_ROOT, 'data', 'BoneMarrow', 'human_cd34_bone_marrow.h5ad'
)
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Part 1: Encoder Architecture Comparison ──
# All use identical loss: recon + β=0.1 KL, no IB, no geometry
PART1_VARIANTS = {
    '1.1 MLP': dict(
        encoder_type='mlp', recon=1.0, beta=0.1, irecon=0.0, lorentz=0.0,
    ),
    '1.2 Transformer': dict(
        encoder_type='transformer', recon=1.0, beta=0.1, irecon=0.0, lorentz=0.0,
    ),
    '1.3 Graph GAT': dict(
        encoder_type='graph', graph_type='GAT',
        recon=1.0, beta=0.1, irecon=0.0, lorentz=0.0,
    ),
    '1.4 Graph GCN': dict(
        encoder_type='graph', graph_type='GCN',
        recon=1.0, beta=0.1, irecon=0.0, lorentz=0.0,
    ),
    '1.5 Graph SAGE': dict(
        encoder_type='graph', graph_type='SAGE',
        recon=1.0, beta=0.1, irecon=0.0, lorentz=0.0,
    ),
}

# ── Part 2: Component Effectiveness (additive from minimal GAT) ──
PART2_VARIANTS = {
    '2.1 GAT Baseline': dict(
        encoder_type='graph', graph_type='GAT',
        recon=1.0, beta=0.1, irecon=0.0, lorentz=0.0,
    ),
    '2.2 + Low beta': dict(
        encoder_type='graph', graph_type='GAT',
        recon=1.0, beta=0.01, irecon=0.0, lorentz=0.0,
    ),
    '2.3 + IB': dict(
        encoder_type='graph', graph_type='GAT',
        recon=1.0, beta=0.1, irecon=0.5, lorentz=0.0,
    ),
    '2.4 + IB + Lorentz': dict(
        encoder_type='graph', graph_type='GAT',
        recon=1.0, beta=0.1, irecon=0.5, lorentz=5.0,
    ),
    '2.5 + IB + Euclidean': dict(
        encoder_type='graph', graph_type='GAT',
        recon=1.0, beta=0.1, irecon=0.5, lorentz=5.0,
        use_euclidean_manifold=True,
    ),
    '2.6 + Adj Decoder': dict(
        encoder_type='graph', graph_type='GAT',
        recon=1.0, beta=0.1, irecon=0.0, lorentz=0.0,
        use_graph_decoder=True, structure_decoder_type='mlp', w_adj=0.1,
    ),
}

# ── Part 3: Ablation (subtractive from full: IB + Lorentz + β=0.1) ──
PART3_VARIANTS = {
    '3.1 Full (IB+Lor+beta)': dict(
        encoder_type='graph', graph_type='GAT',
        recon=1.0, beta=0.1, irecon=0.5, lorentz=5.0,
    ),
    '3.2 - IB (-> - Geo)': dict(
        encoder_type='graph', graph_type='GAT',
        recon=1.0, beta=0.1, irecon=0.0, lorentz=0.0,
    ),
    '3.3 - Geometry': dict(
        encoder_type='graph', graph_type='GAT',
        recon=1.0, beta=0.1, irecon=0.5, lorentz=0.0,
    ),
    '3.4 Lor -> Euclid': dict(
        encoder_type='graph', graph_type='GAT',
        recon=1.0, beta=0.1, irecon=0.5, lorentz=5.0,
        use_euclidean_manifold=True,
    ),
    '3.5 - KL (beta=0)': dict(
        encoder_type='graph', graph_type='GAT',
        recon=1.0, beta=0.0, irecon=0.5, lorentz=5.0,
    ),
}

PARTS = {
    1: ('study_encoder_comparison', PART1_VARIANTS),
    2: ('study_component_effectiveness', PART2_VARIANTS),
    3: ('study_ablation', PART3_VARIANTS),
}


# Column renames: metrics_expanded names -> STUDY_REPORT short names
_METRIC_RENAMES = {
    'CAL': 'CH',
    'DAV': 'DB',
    'COR': 'Corr',
    'DRE_umap_overall_quality': 'DRE_umap',
    'DRE_umap_distance_correlation': 'DRE_umap_distcorr',
    'DRE_umap_Q_local': 'DRE_umap_Qloc',
    'DRE_umap_Q_global': 'DRE_umap_Qglob',
    'DRE_tsne_overall_quality': 'DRE_tsne',
    'DRE_tsne_distance_correlation': 'DRE_tsne_distcorr',
    'DRE_tsne_Q_local': 'DRE_tsne_Qloc',
    'DRE_tsne_Q_global': 'DRE_tsne_Qglob',
    'LSE_overall_quality': 'LSE_overall',
    'LSE_manifold_dimensionality': 'LSE_manifold_dim',
    'LSE_noise_resilience': 'LSE_noise_resil',
    'LSE_spectral_decay_rate': 'LSE_spectral_decay',
}


def run_single(adata1, config_name, params, labels, epochs, patience, seed):
    """Train one HSDE configuration and return metrics dict."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        model = HSDE(
            adata1, layer='counts',
            hidden_dim=128, latent_dim=10, i_dim=2,
            lr=1e-4, loss_type='nb',
            random_seed=seed,
            device=device,
            **params
        )
        # val_every=1 so patience=N means N epochs without improvement
        model.fit(epochs=epochs, patience=patience, early_stop=True,
                  val_every=1, compute_metrics=False)
        latent = model.get_latent()
        raw_metrics = evaluate_latent(latent, labels)

        # Rename to STUDY_REPORT convention
        metrics = {_METRIC_RENAMES.get(k, k): v for k, v in raw_metrics.items()}

        res = model.get_resource_metrics()
        metrics['train_time_s'] = round(res['train_time'], 1)
        metrics['best_val_loss'] = round(getattr(model, 'best_val_loss', float('nan')), 4)
        metrics['actual_epochs'] = res['actual_epochs']
        metrics['n_params'] = sum(p.numel() for p in model.nn.parameters())
        metrics['peak_memory_gb'] = round(res['peak_memory_gb'], 3)
        metrics['status'] = 'OK'

        print(f"    OK  {config_name}: ARI={metrics.get('ARI', 0):.4f}, "
              f"NMI={metrics.get('NMI', 0):.4f}, "
              f"time={metrics['train_time_s']}s, "
              f"epochs={metrics['actual_epochs']}")

        del model
        torch.cuda.empty_cache()
        return metrics

    except Exception as e:
        print(f"    FAIL  {config_name}: {e}")
        traceback.print_exc()
        torch.cuda.empty_cache()
        return {'status': f'FAIL: {e}'}


def run_part(adata1, part_num, labels, epochs, patience, seed):
    """Run all configs for one part, return DataFrame."""
    csv_name, variants = PARTS[part_num]
    config_names = list(variants.keys())

    print(f"\n{'='*60}")
    print(f"PART {part_num}: {csv_name}")
    print(f"Configs: {config_names}")
    print(f"{'='*60}")

    all_metrics = []
    for config_name, params in variants.items():
        metrics = run_single(adata1, config_name, params, labels,
                             epochs, patience, seed)
        all_metrics.append(metrics)

    df = pd.DataFrame(all_metrics, index=config_names)
    csv_path = os.path.join(RESULTS_DIR, f'{csv_name}.csv')
    df.to_csv(csv_path, index_label='config')
    print(f"  Saved: {csv_path}")

    # Also save checkpoint
    cp_path = os.path.join(RESULTS_DIR, f'{csv_name}_checkpoint.csv')
    df.to_csv(cp_path, index_label='config')

    return df


def main():
    parser = argparse.ArgumentParser(description='HSDE Single-Dataset Deep Study')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_cells', type=int, default=3000)
    parser.add_argument('--n_genes', type=int, default=2000)
    parser.add_argument('--part', type=str, default='all',
                        choices=['all', '1', '2', '3'])
    parser.add_argument('--dataset', type=str, default=DEFAULT_DATASET)
    args = parser.parse_args()

    # Determine which parts to run
    if args.part == 'all':
        parts_to_run = [1, 2, 3]
    else:
        parts_to_run = [int(args.part)]

    total_configs = sum(len(PARTS[p][1]) for p in parts_to_run)

    print(f"\n{'#'*60}")
    print(f"HSDE SINGLE-DATASET DEEP STUDY (v3)")
    print(f"{'#'*60}")
    print(f"Dataset: {args.dataset}")
    print(f"Parts: {parts_to_run} ({total_configs} configurations)")
    print(f"Epochs: {args.epochs}, Patience: {args.patience}, Seed: {args.seed}")
    print(f"Preprocessing: {args.n_genes} HVGs, max {args.n_cells} cells")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"{'#'*60}\n")

    # Load and preprocess dataset
    t0 = time.time()
    adata1 = load_and_preprocess(args.dataset)
    print(f"Preprocessing took {time.time() - t0:.1f}s")

    # Compute Leiden labels once (shared across all configs)
    labels, n_clusters = get_labels(adata1)
    print(f"Leiden labels: {n_clusters} clusters\n")

    # Open log file
    log_path = os.path.join(RESULTS_DIR, 'study_full_log.txt')
    log_lines = []

    # Run each part
    all_dfs = []
    study_start = time.time()
    for part_num in parts_to_run:
        df = run_part(adata1, part_num, labels, args.epochs,
                      args.patience, args.seed)
        all_dfs.append(df)
        log_lines.append(f"Part {part_num}: {len(df)} configs completed")

    total_time = time.time() - study_start

    # Save combined results
    if len(all_dfs) == 3:
        combined = pd.concat(all_dfs)
        combined_path = os.path.join(RESULTS_DIR, 'study_combined_results.csv')
        combined.to_csv(combined_path, index_label='config')
        print(f"\nCombined: {combined_path}")

        # Also save as JSON
        json_path = os.path.join(RESULTS_DIR, 'study_combined_results.json')
        combined_dict = combined.to_dict(orient='index')
        with open(json_path, 'w') as f:
            json.dump(combined_dict, f, indent=2, default=str)
        print(f"JSON: {json_path}")

    # Write log
    log_lines.append(f"\nTotal runtime: {total_time:.1f}s ({total_configs} configurations)")
    log_lines.append(f"Seed: {args.seed}, Epochs: {args.epochs}, Patience: {args.patience}")
    with open(log_path, 'w') as f:
        f.write('\n'.join(log_lines))

    print(f"\n{'='*60}")
    print(f"STUDY COMPLETE — Total: {total_time:.1f}s")
    print(f"Results: {RESULTS_DIR}")
    print(f"{'='*60}")

    del adata1
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
