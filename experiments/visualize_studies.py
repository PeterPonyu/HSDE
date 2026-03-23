#!/usr/bin/env python3
"""
Visualize results from all 3 HSDE experiment series using REA.py.
=================================================================
Generates publication-quality figures for:
  1. Ablation study (5 methods)
  2. GM-VAE geometric benchmark (6 methods)
  3. Disentanglement regularization (6 methods)

Each experiment produces:
  - clustering.pdf     (6 metrics: NMI, ARI, ASW, CAL, DAV, COR)
  - dr_umap.pdf        (4 metrics)
  - dr_tsne.pdf        (4 metrics)
  - intrinsic.pdf      (7 metrics)
  - all_metrics.pdf    (merged vertical stack)
"""

import sys, os
import numpy as np
from PIL import Image
from pathlib import Path

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# REA.py (configurable via HSDE_VISUALIZER_DIR env var)
SKILL_DIR = os.environ.get(
    "HSDE_VISUALIZER_DIR",
    os.path.expanduser("~/.copilot/skills/results-visualizer"),
)
sys.path.insert(0, SKILL_DIR)
from REA import RigorousExperimentalAnalyzer, create_publication_figure, SERIES_PALETTES

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── Metric definitions ──
# Group 1: Clustering (6)
METRICS_CLUSTERING = ['NMI', 'ARI', 'ASW', 'CAL', 'DAV', 'COR']
DISPLAY_CLUSTERING = {'NMI': 'NMI', 'ARI': 'ARI', 'ASW': 'ASW',
                       'CAL': 'CAL', 'DAV': 'DAV', 'COR': 'COR'}

# Group 2: DR quality UMAP (4)
METRICS_DR_UMAP = [
    'DRE_umap_distance_correlation', 'DRE_umap_Q_local',
    'DRE_umap_Q_global', 'DRE_umap_overall_quality',
]
DISPLAY_DR_UMAP = {
    'DRE_umap_distance_correlation': 'DC (umap)',
    'DRE_umap_Q_local': 'QL (umap)',
    'DRE_umap_Q_global': 'QG (umap)',
    'DRE_umap_overall_quality': 'OV (umap)',
}

# Group 3: DR quality t-SNE (4)
METRICS_DR_TSNE = [
    'DRE_tsne_distance_correlation', 'DRE_tsne_Q_local',
    'DRE_tsne_Q_global', 'DRE_tsne_overall_quality',
]
DISPLAY_DR_TSNE = {
    'DRE_tsne_distance_correlation': 'DC (tsne)',
    'DRE_tsne_Q_local': 'QL (tsne)',
    'DRE_tsne_Q_global': 'QG (tsne)',
    'DRE_tsne_overall_quality': 'OV (tsne)',
}

# Group 4: Intrinsic manifold (7)
METRICS_INTRINSIC = [
    'LSE_manifold_dimensionality', 'LSE_spectral_decay_rate',
    'LSE_participation_ratio', 'LSE_anisotropy_score',
    'LSE_noise_resilience', 'LSE_core_quality', 'LSE_overall_quality',
]
DISPLAY_INTRINSIC = {
    'LSE_manifold_dimensionality': 'Man. dim.',
    'LSE_spectral_decay_rate': 'Spec. decay',
    'LSE_participation_ratio': 'Part. ratio',
    'LSE_anisotropy_score': 'Anisotropy',
    'LSE_noise_resilience': 'Noise res.',
    'LSE_core_quality': 'Core qual.',
    'LSE_overall_quality': 'Overall qual.',
}

# Combined display names
ALL_DISPLAY = {**DISPLAY_CLUSTERING, **DISPLAY_DR_UMAP,
               **DISPLAY_DR_TSNE, **DISPLAY_INTRINSIC}


def merge_pdfs_as_images(pdf_paths, output_path, dpi=300):
    """Merge multiple single-row PDFs into one vertical-stacked PDF via matplotlib."""
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.image as mpimg
    import subprocess, tempfile

    images = []
    tmp_files = []
    for p in pdf_paths:
        if not os.path.exists(p):
            continue
        # Convert PDF to PNG using pdftoppm (poppler-utils)
        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        tmp.close()
        tmp_files.append(tmp.name)
        ret = subprocess.run(
            ['pdftoppm', '-png', '-r', str(dpi), '-singlefile', p, tmp.name.replace('.png', '')],
            capture_output=True
        )
        if ret.returncode == 0 and os.path.exists(tmp.name):
            img = Image.open(tmp.name)
            images.append(img)
        else:
            # Fallback: try to use matplotlib's pdf backend
            print(f"  ⚠ Could not convert {p}")

    if not images:
        # Cleanup
        for t in tmp_files:
            if os.path.exists(t):
                os.unlink(t)
        return

    # Stack vertically
    total_height = sum(img.height for img in images)
    max_width = max(img.width for img in images)

    merged = Image.new('RGB', (max_width, total_height), 'white')
    y_offset = 0
    for img in images:
        merged.paste(img, (0, y_offset))
        y_offset += img.height

    merged.save(output_path, 'PDF', resolution=dpi)
    print(f"  Merged → {output_path}")

    # Cleanup
    for t in tmp_files:
        if os.path.exists(t):
            os.unlink(t)


def generate_experiment_figures(
    data_folder, method_names, method_order,
    palette, sig_pairs, experiment_name, output_dir
):
    """Generate all metric group figures for one experiment series."""
    import shutil, tempfile
    os.makedirs(output_dir, exist_ok=True)

    n_methods = len(method_names)
    fig_h = 3.5 if n_methods >= 8 else 3.8
    xtick_fs = 8 if n_methods >= 8 else (9 if n_methods >= 6 else 11)
    x_rot = 45 if n_methods >= 8 else (35 if n_methods >= 6 else 30)

    # Clean data: fill NaN with column medians (handles crashed models)
    import pandas as pd, glob
    clean_dir = tempfile.mkdtemp(prefix='hsde_clean_')
    csv_files = sorted(glob.glob(os.path.join(data_folder, '*.csv')))
    for f in csv_files:
        df = pd.read_csv(f, index_col=0)
        if df.isna().any().any():
            # Fill NaN with column median (robust to outliers)
            for col in df.columns:
                if df[col].isna().any():
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val if not pd.isna(median_val) else 0)
        df.to_csv(os.path.join(clean_dir, os.path.basename(f)), index_label='method')
    actual_data_folder = clean_dir if csv_files else data_folder

    common_kwargs = dict(
        plot_type='boxplot',
        show_significance=True,
        show_significance_pairs=sig_pairs,
        palette=palette,
        font_family='Arial',
        show_legend=False,
        significance_line_width=1.5,
        significance_marker_offset=-0.065,
        bar_strip_size=2,
        bar_strip_alpha=0.5,
        title_fontsize=11,
        title_fontweight='normal',
        axis_label_fontsize=11,
        tick_label_fontsize=11,
        significance_fontsize=13,
        ns_fontsize=10,
        ns_offset=0,
        xlabel_rotation=x_rot,
        dpi=300,
        bottom=0.12,
    )

    print(f"\n{'='*60}")
    print(f"Generating figures for: {experiment_name}")
    print(f"  Methods: {method_names}")
    print(f"  Significance pairs: {sig_pairs}")
    print(f"{'='*60}")

    # Initialize analyzer
    analyzer = RigorousExperimentalAnalyzer(
        data_folder_path=actual_data_folder,
        method_names=method_names,
        method_order=method_order,
        verbose=True,
    )
    analyzer.load_experimental_data()
    analyzer.preprocess_data()

    # Print summary
    try:
        analyzer.print_comprehensive_summary()
    except Exception as e:
        print(f"  ⚠ Summary failed: {e}")

    pdf_paths = []

    # ─── Group 1: Clustering (6) ───
    save_path = os.path.join(output_dir, 'clustering.pdf')
    try:
        fig, axs = create_publication_figure(
            analyzer, METRICS_CLUSTERING,
            metric_display_names=DISPLAY_CLUSTERING,
            figsize=(19.2, fig_h), ncols=6,
            panel_labels=False,
            save_path=save_path,
            **common_kwargs,
        )
        plt.close(fig)
        pdf_paths.append(save_path)
        print(f"  ✓ Clustering → {save_path}")
    except Exception as e:
        print(f"  ✗ Clustering FAILED: {e}")
        import traceback; traceback.print_exc()

    # ─── Group 2: DR UMAP (4) ───
    save_path = os.path.join(output_dir, 'dr_umap.pdf')
    try:
        fig, axs = create_publication_figure(
            analyzer, METRICS_DR_UMAP,
            metric_display_names=DISPLAY_DR_UMAP,
            figsize=(12.8, fig_h), ncols=4,
            panel_labels=False,
            save_path=save_path,
            **common_kwargs,
        )
        plt.close(fig)
        pdf_paths.append(save_path)
        print(f"  ✓ DR UMAP → {save_path}")
    except Exception as e:
        print(f"  ✗ DR UMAP FAILED: {e}")
        import traceback; traceback.print_exc()

    # ─── Group 3: DR t-SNE (4) ───
    save_path = os.path.join(output_dir, 'dr_tsne.pdf')
    try:
        fig, axs = create_publication_figure(
            analyzer, METRICS_DR_TSNE,
            metric_display_names=DISPLAY_DR_TSNE,
            figsize=(12.8, fig_h), ncols=4,
            panel_labels=False,
            save_path=save_path,
            **common_kwargs,
        )
        plt.close(fig)
        pdf_paths.append(save_path)
        print(f"  ✓ DR t-SNE → {save_path}")
    except Exception as e:
        print(f"  ✗ DR t-SNE FAILED: {e}")
        import traceback; traceback.print_exc()

    # ─── Group 4: Intrinsic (7) ───
    save_path = os.path.join(output_dir, 'intrinsic.pdf')
    try:
        fig, axs = create_publication_figure(
            analyzer, METRICS_INTRINSIC,
            metric_display_names=DISPLAY_INTRINSIC,
            figsize=(22.4, fig_h), ncols=7,
            panel_labels=False,
            save_path=save_path,
            **common_kwargs,
        )
        plt.close(fig)
        pdf_paths.append(save_path)
        print(f"  ✓ Intrinsic → {save_path}")
    except Exception as e:
        print(f"  ⚠ Intrinsic with significance failed: {e}")
        # Fallback: plot individual metrics with manual subplot grid
        try:
            import seaborn as sns
            n_intr = len(METRICS_INTRINSIC)
            fig, axes = plt.subplots(1, n_intr, figsize=(22.4, fig_h))
            if n_intr == 1:
                axes = [axes]

            for i, metric in enumerate(METRICS_INTRINSIC):
                ax = axes[i]
                try:
                    analyzer.create_metric_comparison_plot(
                        metric, ax=ax,
                        plot_type='boxplot',
                        show_significance=False,
                        palette=palette,
                        font_family='Arial',
                        show_legend=False,
                        title=DISPLAY_INTRINSIC.get(metric, metric),
                        title_fontsize=11,
                        title_fontweight='normal',
                        tick_label_fontsize=11,
                        axis_label_fontsize=11,
                        xlabel_rotation=x_rot,
                        stat_test_display='none',
                    )
                except Exception:
                    # Ultimate fallback: seaborn boxplot
                    plot_data = analyzer.processed_data[
                        analyzer.processed_data['metric'] == metric
                    ].copy()
                    if len(plot_data) > 0:
                        sns.boxplot(data=plot_data, x='method', y='value',
                                    palette=palette, ax=ax, width=0.7)
                        ax.set_title(DISPLAY_INTRINSIC.get(metric, metric),
                                     fontsize=11)
                        ax.set_xlabel('')
                        ax.tick_params(axis='x', rotation=x_rot)
                    else:
                        ax.set_title(DISPLAY_INTRINSIC.get(metric, metric))
                        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                                ha='center')

            plt.tight_layout()
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            pdf_paths.append(save_path)
            print(f"  ✓ Intrinsic (fallback) → {save_path}")
        except Exception as e2:
            print(f"  ✗ Intrinsic FAILED completely: {e2}")
            import traceback; traceback.print_exc()

    # ─── Merge all into one PDF ───
    merged_path = os.path.join(output_dir, 'all_metrics.pdf')
    try:
        merge_pdfs_as_images(pdf_paths, merged_path)
    except Exception as e:
        print(f"  ⚠ Merge failed: {e} (individual PDFs still available)")

    # Cleanup temp dir
    try:
        shutil.rmtree(clean_dir)
    except Exception:
        pass

    return analyzer


def main():
    results_base = os.path.join(PROJECT_ROOT, 'HSDE_results')

    # ══════════════════════════════════════════════
    # Experiment 1: Ablation (5 methods)
    # ══════════════════════════════════════════════
    ablation_methods = ['Base VAE', 'VAE+IB', 'VAE+Hyp', 'VAE+IB+Hyp', 'HSDE']
    ablation_sig = [
        ('Base VAE', 'HSDE'),
        ('VAE+IB', 'HSDE'),
        ('VAE+Hyp', 'HSDE'),
        ('VAE+IB+Hyp', 'HSDE'),
    ]
    generate_experiment_figures(
        data_folder=os.path.join(results_base, 'ablation', 'tables'),
        method_names=ablation_methods,
        method_order=ablation_methods,
        palette=SERIES_PALETTES['ablation'],
        sig_pairs=ablation_sig,
        experiment_name='Ablation Study',
        output_dir=os.path.join(results_base, 'ablation', 'figures'),
    )

    # ══════════════════════════════════════════════
    # Experiment 2: GM-VAE Benchmark (6 methods)
    # ══════════════════════════════════════════════
    gmvae_methods = [
        'GM-VAE (Eucl.)', 'GM-VAE (Poinc.)', 'GM-VAE (PGM)',
        'GM-VAE (L-PGM)', 'GM-VAE (HW)', 'HSDE'
    ]
    gmvae_sig = [
        ('GM-VAE (Eucl.)', 'HSDE'),
        ('GM-VAE (Poinc.)', 'HSDE'),
        ('GM-VAE (PGM)', 'HSDE'),
        ('GM-VAE (L-PGM)', 'HSDE'),
        ('GM-VAE (HW)', 'HSDE'),
    ]
    generate_experiment_figures(
        data_folder=os.path.join(results_base, 'gmvae_benchmark', 'tables'),
        method_names=gmvae_methods,
        method_order=gmvae_methods,
        palette=SERIES_PALETTES['gmvae_benchmark'],
        sig_pairs=gmvae_sig,
        experiment_name='GM-VAE Geometric Benchmark',
        output_dir=os.path.join(results_base, 'gmvae_benchmark', 'figures'),
    )

    # ══════════════════════════════════════════════
    # Experiment 3: Disentanglement (6 methods)
    # ══════════════════════════════════════════════
    disent_methods = ['Base VAE', 'beta-VAE', 'DIP-VAE', 'TC-VAE', 'InfoVAE', 'HSDE']
    disent_sig = [
        ('Base VAE', 'HSDE'),
        ('beta-VAE', 'HSDE'),
        ('DIP-VAE', 'HSDE'),
        ('TC-VAE', 'HSDE'),
        ('InfoVAE', 'HSDE'),
    ]
    generate_experiment_figures(
        data_folder=os.path.join(results_base, 'disentanglement', 'tables'),
        method_names=disent_methods,
        method_order=disent_methods,
        palette=SERIES_PALETTES['disentanglement'],
        sig_pairs=disent_sig,
        experiment_name='Disentanglement Regularization',
        output_dir=os.path.join(results_base, 'disentanglement', 'figures'),
    )

    print(f"\n{'='*60}")
    print("ALL VISUALIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Ablation:        {results_base}/ablation/figures/")
    print(f"  GM-VAE:          {results_base}/gmvae_benchmark/figures/")
    print(f"  Disentanglement: {results_base}/disentanglement/figures/")


if __name__ == '__main__':
    main()
