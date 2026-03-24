#!/usr/bin/env python3
"""
Visualize Single-Dataset Deep Study Results
============================================
Reads the CSVs produced by run_study.py and generates publication figures
using absolute-geometry layout from hsde.viz.style.

Outputs (to results/figures/):
  - part1_encoder_comparison.pdf
  - part2_component_effectiveness.pdf
  - part3_ablation.pdf
  - summary_heatmap.pdf
  - efficiency_scatter.pdf
  - delta_part2.pdf, delta_part3.pdf
"""

import sys, os
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Load style module directly (avoids pulling in heavy hsde.core dependencies)
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    'style', os.path.join(PROJECT_ROOT, 'hsde', 'viz', 'style.py'))
_style = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_style)

apply_style = _style.apply_style
save_figure = _style.save_figure
row_of_axes = _style.row_of_axes
place_axes = _style.place_axes
grid_of_axes = _style.grid_of_axes
RECT_BOXPLOT_ROW = _style.RECT_BOXPLOT_ROW
RECT_HEATMAP = _style.RECT_HEATMAP
RECT_TITLE_Y = _style.RECT_TITLE_Y
GAP_BOXPLOT = _style.GAP_BOXPLOT
FS_TITLE = _style.FS_TITLE
FS_AXIS = _style.FS_AXIS
FS_TICK = _style.FS_TICK
FS_SMALL = _style.FS_SMALL
FS_LABEL = _style.FS_LABEL
HEATMAP_CMAP = _style.HEATMAP_CMAP
HEATMAP_DARK_THRESHOLD = _style.HEATMAP_DARK_THRESHOLD
ACCENT_POSITIVE = _style.ACCENT_POSITIVE
ACCENT_NEGATIVE = _style.ACCENT_NEGATIVE
ACCENT_BEST = _style.ACCENT_BEST
FIG_WIDTH_IN = _style.FIG_WIDTH_IN
DPI = _style.DPI
SAVEFIG_KW = _style.SAVEFIG_KW

# ── Configuration ──
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# Key metrics for the bar charts (matching STUDY_REPORT.md Table 3)
# These are the renamed column names from run_study.py
DISPLAY_METRICS = ['ARI', 'NMI', 'ASW', 'CH', 'DB', 'DRE_umap', 'DRE_tsne', 'LSE_overall']
METRIC_LABELS = {
    'ARI': 'ARI',
    'NMI': 'NMI',
    'ASW': 'ASW',
    'CH': 'CH',
    'DB': 'DB',
    'DRE_umap': 'DRE\nUMAP',
    'DRE_tsne': 'DRE\nt-SNE',
    'LSE_overall': 'LSE',
    'train_time_s': 'Time (s)',
    # Legacy fallbacks (metrics_expanded raw names)
    'CAL': 'CH',
    'DAV': 'DB',
}

# Lower is better for these metrics
LOWER_BETTER = {'DB', 'DAV'}

# Palette for study configs (Wong colorblind-safe)
STUDY_PALETTE = [
    '#0072B2', '#E69F00', '#009E73', '#CC79A7', '#D55E00',
    '#56B4E9', '#F0E442', '#999999', '#000000', '#8B0000',
    '#4B0082', '#2F4F4F', '#FF6347', '#228B22', '#DAA520', '#708090',
]


def _load_csv(name):
    """Load a study CSV from results/."""
    path = os.path.join(RESULTS_DIR, f'{name}.csv')
    if not os.path.exists(path):
        print(f"  Missing: {path}")
        return None
    df = pd.read_csv(path, index_col='config')
    return df


def _remap_metric_names(df):
    """Map the metrics_expanded output column names to the short names used in STUDY_REPORT.

    run_study.py already renames these columns, but this handles legacy CSVs or
    CSVs from other experiment scripts that use the raw metrics_expanded names.
    """
    renames = {
        'CAL': 'CH',
        'DAV': 'DB',
        'COR': 'Corr',
        'DRE_umap_overall_quality': 'DRE_umap',
        'DRE_tsne_overall_quality': 'DRE_tsne',
        'LSE_overall_quality': 'LSE_overall',
        'DRE_umap_distcorr': 'DRE_umap_distcorr',
        'DRE_umap_Qloc': 'DRE_umap_Qloc',
        'DRE_umap_Qglob': 'DRE_umap_Qglob',
        'DRE_tsne_distcorr': 'DRE_tsne_distcorr',
        'DRE_tsne_Qloc': 'DRE_tsne_Qloc',
        'DRE_tsne_Qglob': 'DRE_tsne_Qglob',
        'LSE_manifold_dimensionality': 'LSE_manifold_dim',
        'LSE_noise_resilience': 'LSE_noise_resil',
        'LSE_spectral_decay_rate': 'LSE_spectral_decay',
    }
    existing = {k: v for k, v in renames.items() if k in df.columns}
    if existing:
        df = df.rename(columns=existing)
    return df


def plot_part_bars(df, title, filename, metrics=None):
    """Grouped bar chart for one study part."""
    if df is None:
        return
    df = _remap_metric_names(df)
    if metrics is None:
        metrics = [m for m in DISPLAY_METRICS if m in df.columns]
    if not metrics:
        print(f"  No metrics found for {filename}")
        return

    n_configs = len(df)
    n_metrics = len(metrics)

    fig_w = max(FIG_WIDTH_IN, n_metrics * 1.8)
    fig_h = 5.0
    fig = plt.figure(figsize=(fig_w, fig_h))

    # Use row_of_axes for proper layout – extra bottom margin for rotated labels
    rect = [0.06, 0.32, 0.92, 0.56]
    axes = row_of_axes(fig, n_metrics, rect, gap=0.03)

    colors = STUDY_PALETTE[:n_configs]
    configs = list(df.index)

    for ax, metric in zip(axes, metrics):
        vals = df[metric].values.astype(float)
        x = np.arange(n_configs)
        bars = ax.bar(x, vals, color=colors, edgecolor='white', linewidth=0.5)

        # Highlight best
        if metric in LOWER_BETTER:
            best_idx = np.nanargmin(vals)
        else:
            best_idx = np.nanargmax(vals)
        bars[best_idx].set_edgecolor(ACCENT_BEST)
        bars[best_idx].set_linewidth(1.5)

        label = METRIC_LABELS.get(metric, metric)
        direction = ' ↓' if metric in LOWER_BETTER else ' ↑'
        ax.set_title(f"{label}{direction}", fontsize=FS_TITLE, fontweight='bold')
        ax.set_xticks(x)
        # Strip leading number prefix for cleaner labels (e.g. "1.1 MLP" -> "MLP")
        short_labels = [c.split(' ', 1)[-1] if ' ' in c else c for c in configs]
        ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=FS_SMALL + 1)
        ax.tick_params(axis='y', labelsize=FS_TICK)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.text(0.5, RECT_TITLE_Y + 0.02, title,
             fontsize=FS_TITLE + 3, fontweight='bold', ha='center', va='bottom')

    path = os.path.join(FIGURES_DIR, filename)
    save_figure(fig, path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_summary_heatmap(combined_df, filename='summary_heatmap'):
    """Z-scored heatmap of all 16 configs x key metrics."""
    if combined_df is None:
        return
    combined_df = _remap_metric_names(combined_df)
    metrics = [m for m in DISPLAY_METRICS if m in combined_df.columns]
    if not metrics:
        return

    data = combined_df[metrics].astype(float)

    # Z-score each column, flip DB direction
    z = data.copy()
    for col in z.columns:
        mean, std = z[col].mean(), z[col].std()
        if std > 0:
            z[col] = (z[col] - mean) / std
        else:
            z[col] = 0.0
        if col in LOWER_BETTER:
            z[col] = -z[col]  # flip so higher = better

    n_rows = len(z)
    n_cols = len(metrics)
    fig_h = max(5.0, 0.35 * n_rows + 2.0)
    fig_w = max(FIG_WIDTH_IN, n_cols * 0.9 + 3.0)

    fig = plt.figure(figsize=(fig_w, fig_h))
    rect = [0.22, 0.14, 0.60, 0.74]
    ax = place_axes(fig, rect)

    im = ax.imshow(z.values, cmap=HEATMAP_CMAP, aspect='auto',
                   vmin=-2.0, vmax=2.0)

    # Labels
    display_cols = [METRIC_LABELS.get(m, m).replace('\n', ' ') for m in metrics]
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(display_cols, rotation=40, ha='right', fontsize=FS_TICK)
    ax.set_yticks(np.arange(n_rows))
    # Strip leading number prefix for cleaner y-labels
    y_labels = [n.split(' ', 1)[-1] if ' ' in n else n for n in z.index]
    ax.set_yticklabels(y_labels, fontsize=FS_TICK)

    # Cell annotations
    for i in range(n_rows):
        for j in range(n_cols):
            raw_val = data.iloc[i, j]
            z_val = z.iloc[i, j]
            norm_val = (z_val + 2.0) / 4.0  # map [-2,2] to [0,1]
            color = 'white' if norm_val > HEATMAP_DARK_THRESHOLD else 'black'
            txt = f'{raw_val:.3f}' if abs(raw_val) < 10 else f'{raw_val:.0f}'
            ax.text(j, i, txt, ha='center', va='center',
                    fontsize=FS_SMALL, color=color)

    # Colorbar
    cbar_rect = [rect[0] + rect[2] + 0.02, rect[1], 0.025, rect[3]]
    cax = fig.add_axes(cbar_rect)
    fig.colorbar(im, cax=cax, label='z-score (↑ = better)')

    fig.text(0.5, RECT_TITLE_Y, 'Summary Heatmap (z-scored, all parts)',
             fontsize=FS_TITLE + 2, fontweight='bold', ha='center', va='bottom')

    path = os.path.join(FIGURES_DIR, filename)
    save_figure(fig, path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_efficiency_scatter(combined_df, filename='efficiency_scatter'):
    """ARI vs training time scatter plot."""
    if combined_df is None:
        return
    combined_df = _remap_metric_names(combined_df)
    if 'ARI' not in combined_df.columns or 'train_time_s' not in combined_df.columns:
        print("  Missing ARI or train_time_s columns")
        return

    fig = plt.figure(figsize=(FIG_WIDTH_IN + 1.5, 5.5))
    rect = [0.10, 0.14, 0.82, 0.74]
    ax = place_axes(fig, rect)

    configs = list(combined_df.index)
    ari = combined_df['ARI'].values.astype(float)
    time_s = combined_df['train_time_s'].values.astype(float)
    colors = STUDY_PALETTE[:len(configs)]

    ax.scatter(time_s, ari, c=colors, s=80, edgecolors='black',
               linewidths=0.5, zorder=5)

    # Label each point with overlap-aware placement
    # Collect label positions and nudge overlapping ones
    labels_xy = []
    for i, name in enumerate(configs):
        short = name.split(' ', 1)[-1] if ' ' in name else name
        labels_xy.append((time_s[i], ari[i], short))

    # Sort by y to assign alternating offsets for close points
    labels_xy.sort(key=lambda t: (t[0], t[1]))
    placed = []
    for tx, ty, short in labels_xy:
        ox, oy = 6, 5
        # Check for nearby already-placed labels and alternate offset
        for px, py, _, _ in placed:
            if abs(tx - px) < 2.0 and abs(ty - py) < 0.06:
                oy = -12  # push below
                break
        placed.append((tx, ty, ox, oy))
        ax.annotate(short, (tx, ty),
                    textcoords='offset points', xytext=(ox, oy),
                    fontsize=FS_SMALL, ha='left',
                    arrowprops=dict(arrowstyle='-', color='grey',
                                   lw=0.4, shrinkA=0, shrinkB=3))

    # Add margins so edge labels aren't clipped
    x_margin = (time_s.max() - time_s.min()) * 0.08
    y_margin = (ari.max() - ari.min()) * 0.06
    ax.set_xlim(time_s.min() - x_margin, time_s.max() + x_margin * 2)
    ax.set_ylim(ari.min() - y_margin, ari.max() + y_margin)

    ax.set_xlabel('Training Time (s)', fontsize=FS_AXIS)
    ax.set_ylabel('ARI', fontsize=FS_AXIS)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.text(0.5, RECT_TITLE_Y, 'Efficiency: ARI vs Training Time',
             fontsize=FS_TITLE + 2, fontweight='bold', ha='center', va='bottom')

    path = os.path.join(FIGURES_DIR, filename)
    save_figure(fig, path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_delta_chart(df, baseline_name, title, filename):
    """Bar chart of Δ from baseline for each metric."""
    if df is None:
        return
    df = _remap_metric_names(df)
    if baseline_name not in df.index:
        print(f"  Baseline '{baseline_name}' not in DataFrame")
        return

    metrics = [m for m in DISPLAY_METRICS if m in df.columns]
    baseline = df.loc[baseline_name, metrics].astype(float)
    others = df.drop(baseline_name)

    n_others = len(others)
    n_metrics = len(metrics)
    if n_others == 0 or n_metrics == 0:
        return

    fig_w = max(FIG_WIDTH_IN, n_metrics * 1.6)
    fig_h = 5.0
    fig = plt.figure(figsize=(fig_w, fig_h))
    rect = [0.06, 0.32, 0.92, 0.56]
    axes = row_of_axes(fig, n_metrics, rect, gap=0.03)

    colors = STUDY_PALETTE[1:n_others + 1]
    other_names = list(others.index)

    bar_width = 0.8 / n_others
    for ax, metric in zip(axes, metrics):
        x = np.arange(n_others)
        deltas = others[metric].values.astype(float) - baseline[metric]

        # For DB, positive delta is bad (lower is better)
        bar_colors = []
        for d in deltas:
            if metric in LOWER_BETTER:
                bar_colors.append(ACCENT_POSITIVE if d < 0 else ACCENT_NEGATIVE)
            else:
                bar_colors.append(ACCENT_POSITIVE if d > 0 else ACCENT_NEGATIVE)

        ax.bar(x, deltas, color=bar_colors, edgecolor='white', linewidth=0.5)
        ax.axhline(0, color='black', linewidth=0.5, linestyle='-')

        label = METRIC_LABELS.get(metric, metric)
        direction = ' ↓' if metric in LOWER_BETTER else ' ↑'
        ax.set_title(f"Δ{label}{direction}", fontsize=FS_TITLE, fontweight='bold')
        ax.set_xticks(x)
        # Strip leading number prefix for cleaner labels
        short_names = [n.split(' ', 1)[-1] if ' ' in n else n for n in other_names]
        ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=FS_SMALL + 1)
        ax.tick_params(axis='y', labelsize=FS_TICK)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.text(0.5, RECT_TITLE_Y + 0.02, title,
             fontsize=FS_TITLE + 3, fontweight='bold', ha='center', va='bottom')

    path = os.path.join(FIGURES_DIR, filename)
    save_figure(fig, path)
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    apply_style()

    print(f"\n{'='*60}")
    print("VISUALIZE SINGLE-DATASET DEEP STUDY")
    print(f"{'='*60}")
    print(f"Results: {RESULTS_DIR}")
    print(f"Figures: {FIGURES_DIR}\n")

    # Load CSVs
    df1 = _load_csv('study_encoder_comparison')
    df2 = _load_csv('study_component_effectiveness')
    df3 = _load_csv('study_ablation')
    combined = _load_csv('study_combined_results')

    # Part bar charts
    print("Part 1: Encoder comparison bars...")
    plot_part_bars(df1, 'Part 1: Encoder Architecture Comparison',
                   'part1_encoder_comparison')

    print("Part 2: Component effectiveness bars...")
    plot_part_bars(df2, 'Part 2: Component Effectiveness (Additive)',
                   'part2_component_effectiveness')

    print("Part 3: Ablation bars...")
    plot_part_bars(df3, 'Part 3: Ablation Study (Subtractive)',
                   'part3_ablation')

    # Summary heatmap
    print("Summary heatmap...")
    plot_summary_heatmap(combined)

    # Efficiency scatter
    print("Efficiency scatter...")
    plot_efficiency_scatter(combined)

    # Delta charts
    print("Delta chart (Part 2 vs GAT Baseline)...")
    plot_delta_chart(df2, '2.1 GAT Baseline',
                     'Part 2: Δ from GAT Baseline', 'delta_part2')

    print("Delta chart (Part 3 vs Full)...")
    plot_delta_chart(df3, '3.1 Full (IB+Lor+beta)',
                     'Part 3: Δ from Full Model', 'delta_part3')

    print(f"\nDone. All figures in: {FIGURES_DIR}")


if __name__ == '__main__':
    main()
