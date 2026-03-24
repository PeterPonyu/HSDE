#!/usr/bin/env python3
"""
Visualize Single-Dataset Deep Study Results
============================================
Reads the CSVs produced by run_study.py and generates composite publication
figures organized by logical flow.

Outputs (to results/figures/):
  - fig1_encoder_comparison.pdf    Part 1: encoder bars (standalone)
  - fig2_component_analysis.pdf    Part 2: bars + delta (two-row composite)
  - fig3_ablation_analysis.pdf     Part 3: bars + delta (two-row composite)
  - fig4_overview.pdf              Summary heatmap + efficiency scatter
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
add_panel_label = _style.add_panel_label
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
    """Map metrics_expanded column names to the short names used in STUDY_REPORT."""
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


def _short_config(name):
    """Strip leading number prefix for cleaner labels (e.g. '1.1 MLP' -> 'MLP')."""
    return name.split(' ', 1)[-1] if ' ' in name else name


def _draw_bars_on_axes(axes, df, metrics, palette):
    """Draw bar charts across a row of axes. Returns axes for chaining."""
    configs = list(df.index)
    n_configs = len(configs)
    colors = palette[:n_configs]

    for ax, metric in zip(axes, metrics):
        vals = df[metric].values.astype(float)
        x = np.arange(n_configs)
        bars = ax.bar(x, vals, color=colors, edgecolor='white', linewidth=0.5)

        if metric in LOWER_BETTER:
            best_idx = np.nanargmin(vals)
        else:
            best_idx = np.nanargmax(vals)
        bars[best_idx].set_edgecolor(ACCENT_BEST)
        bars[best_idx].set_linewidth(1.5)

        label = METRIC_LABELS.get(metric, metric)
        direction = ' \u2193' if metric in LOWER_BETTER else ' \u2191'
        ax.set_title(f"{label}{direction}", fontsize=FS_TITLE, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([_short_config(c) for c in configs],
                           rotation=45, ha='right', fontsize=FS_SMALL + 1)
        ax.tick_params(axis='y', labelsize=FS_TICK)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    return axes


def _draw_deltas_on_axes(axes, df, baseline_name, metrics):
    """Draw delta bar charts across a row of axes."""
    baseline = df.loc[baseline_name, metrics].astype(float)
    others = df.drop(baseline_name)
    other_names = list(others.index)

    for ax, metric in zip(axes, metrics):
        x = np.arange(len(others))
        deltas = others[metric].values.astype(float) - baseline[metric]

        bar_colors = []
        for d in deltas:
            if metric in LOWER_BETTER:
                bar_colors.append(ACCENT_POSITIVE if d < 0 else ACCENT_NEGATIVE)
            else:
                bar_colors.append(ACCENT_POSITIVE if d > 0 else ACCENT_NEGATIVE)

        ax.bar(x, deltas, color=bar_colors, edgecolor='white', linewidth=0.5)
        ax.axhline(0, color='black', linewidth=0.5, linestyle='-')

        label = METRIC_LABELS.get(metric, metric)
        direction = ' \u2193' if metric in LOWER_BETTER else ' \u2191'
        ax.set_title(f"\u0394{label}{direction}", fontsize=FS_TITLE, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([_short_config(n) for n in other_names],
                           rotation=45, ha='right', fontsize=FS_SMALL + 1)
        ax.tick_params(axis='y', labelsize=FS_TICK)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    return axes


# ── Composite Figure Generators ──

def plot_fig1_encoder(df1, filename='fig1_encoder_comparison'):
    """Figure 1: Part 1 encoder architecture comparison (standalone bars)."""
    if df1 is None:
        return
    df1 = _remap_metric_names(df1)
    metrics = [m for m in DISPLAY_METRICS if m in df1.columns]
    if not metrics:
        return

    n_metrics = len(metrics)
    fig_w = max(FIG_WIDTH_IN, n_metrics * 1.8)
    fig = plt.figure(figsize=(fig_w, 5.0))

    rect = [0.06, 0.28, 0.92, 0.58]
    axes = row_of_axes(fig, n_metrics, rect, gap=0.03)
    _draw_bars_on_axes(axes, df1, metrics, STUDY_PALETTE)

    fig.text(0.5, 0.94, 'Part 1: Encoder Architecture Comparison',
             fontsize=FS_TITLE + 3, fontweight='bold', ha='center', va='bottom')

    path = os.path.join(FIGURES_DIR, filename)
    save_figure(fig, path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_fig2_component(df2, baseline_name='2.1 GAT Baseline',
                        filename='fig2_component_analysis'):
    """Figure 2: Part 2 component effectiveness — bars (top) + deltas (bottom)."""
    if df2 is None:
        return
    df2 = _remap_metric_names(df2)
    metrics = [m for m in DISPLAY_METRICS if m in df2.columns]
    if not metrics or baseline_name not in df2.index:
        return

    n_metrics = len(metrics)
    fig_w = max(FIG_WIDTH_IN, n_metrics * 1.8)
    fig = plt.figure(figsize=(fig_w, 9.0))

    # Top row: absolute bars
    rect_top = [0.06, 0.56, 0.92, 0.34]
    axes_top = row_of_axes(fig, n_metrics, rect_top, gap=0.03)
    _draw_bars_on_axes(axes_top, df2, metrics, STUDY_PALETTE)
    add_panel_label(axes_top[0], 'a', x=-0.15, y=1.08)

    # Bottom row: delta bars
    rect_bot = [0.06, 0.10, 0.92, 0.34]
    axes_bot = row_of_axes(fig, n_metrics, rect_bot, gap=0.03)
    _draw_deltas_on_axes(axes_bot, df2, baseline_name, metrics)
    add_panel_label(axes_bot[0], 'b', x=-0.15, y=1.08)

    fig.text(0.5, 0.96, 'Part 2: Component Effectiveness (Additive)',
             fontsize=FS_TITLE + 3, fontweight='bold', ha='center', va='bottom')
    fig.text(0.02, 0.73, '(a) Absolute', fontsize=FS_AXIS, rotation=90,
             va='center', ha='center', fontstyle='italic')
    fig.text(0.02, 0.27, '(b) \u0394 from Baseline', fontsize=FS_AXIS, rotation=90,
             va='center', ha='center', fontstyle='italic')

    path = os.path.join(FIGURES_DIR, filename)
    save_figure(fig, path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_fig3_ablation(df3, baseline_name='3.1 Full (IB+Lor+beta)',
                       filename='fig3_ablation_analysis'):
    """Figure 3: Part 3 ablation study — bars (top) + deltas (bottom)."""
    if df3 is None:
        return
    df3 = _remap_metric_names(df3)
    metrics = [m for m in DISPLAY_METRICS if m in df3.columns]
    if not metrics or baseline_name not in df3.index:
        return

    n_metrics = len(metrics)
    fig_w = max(FIG_WIDTH_IN, n_metrics * 1.8)
    fig = plt.figure(figsize=(fig_w, 9.0))

    # Top row: absolute bars
    rect_top = [0.06, 0.56, 0.92, 0.34]
    axes_top = row_of_axes(fig, n_metrics, rect_top, gap=0.03)
    _draw_bars_on_axes(axes_top, df3, metrics, STUDY_PALETTE)
    add_panel_label(axes_top[0], 'a', x=-0.15, y=1.08)

    # Bottom row: delta bars
    rect_bot = [0.06, 0.10, 0.92, 0.34]
    axes_bot = row_of_axes(fig, n_metrics, rect_bot, gap=0.03)
    _draw_deltas_on_axes(axes_bot, df3, baseline_name, metrics)
    add_panel_label(axes_bot[0], 'b', x=-0.15, y=1.08)

    fig.text(0.5, 0.96, 'Part 3: Ablation Study (Subtractive)',
             fontsize=FS_TITLE + 3, fontweight='bold', ha='center', va='bottom')
    fig.text(0.02, 0.73, '(a) Absolute', fontsize=FS_AXIS, rotation=90,
             va='center', ha='center', fontstyle='italic')
    fig.text(0.02, 0.27, '(b) \u0394 from Full Model', fontsize=FS_AXIS, rotation=90,
             va='center', ha='center', fontstyle='italic')

    path = os.path.join(FIGURES_DIR, filename)
    save_figure(fig, path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_fig4_overview(combined_df, filename='fig4_overview'):
    """Figure 4: Overview — heatmap (left) + efficiency scatter (right)."""
    if combined_df is None:
        return
    combined_df = _remap_metric_names(combined_df)
    metrics = [m for m in DISPLAY_METRICS if m in combined_df.columns]
    has_ari = 'ARI' in combined_df.columns and 'train_time_s' in combined_df.columns
    if not metrics and not has_ari:
        return

    n_rows = len(combined_df)
    n_cols = len(metrics)
    fig_w = max(FIG_WIDTH_IN + 5.0, n_cols * 0.9 + 8.0)
    fig_h = max(6.5, 0.35 * n_rows + 2.5)
    fig = plt.figure(figsize=(fig_w, fig_h))

    # ── Left panel: heatmap ──
    hm_rect = [0.14, 0.14, 0.40, 0.74]
    ax_hm = place_axes(fig, hm_rect)

    data = combined_df[metrics].astype(float)
    z = data.copy()
    for col in z.columns:
        mean, std = z[col].mean(), z[col].std()
        if std > 0:
            z[col] = (z[col] - mean) / std
        else:
            z[col] = 0.0
        if col in LOWER_BETTER:
            z[col] = -z[col]

    im = ax_hm.imshow(z.values, cmap=HEATMAP_CMAP, aspect='auto',
                       vmin=-2.0, vmax=2.0)

    display_cols = [METRIC_LABELS.get(m, m).replace('\n', ' ') for m in metrics]
    ax_hm.set_xticks(np.arange(n_cols))
    ax_hm.set_xticklabels(display_cols, rotation=40, ha='right', fontsize=FS_TICK)
    ax_hm.set_yticks(np.arange(n_rows))
    y_labels = [_short_config(n) for n in z.index]
    ax_hm.set_yticklabels(y_labels, fontsize=FS_TICK)

    for i in range(n_rows):
        for j in range(n_cols):
            raw_val = data.iloc[i, j]
            z_val = z.iloc[i, j]
            norm_val = (z_val + 2.0) / 4.0
            color = 'white' if norm_val > HEATMAP_DARK_THRESHOLD else 'black'
            txt = f'{raw_val:.3f}' if abs(raw_val) < 10 else f'{raw_val:.0f}'
            ax_hm.text(j, i, txt, ha='center', va='center',
                       fontsize=FS_SMALL - 1, color=color)

    cbar_rect = [hm_rect[0] + hm_rect[2] + 0.01, hm_rect[1], 0.015, hm_rect[3]]
    cax = fig.add_axes(cbar_rect)
    fig.colorbar(im, cax=cax, label='z-score (\u2191 = better)')
    add_panel_label(ax_hm, 'a', x=-0.32, y=1.04)

    # ── Right panel: efficiency scatter ──
    if has_ari:
        sc_rect = [0.64, 0.14, 0.33, 0.74]
        ax_sc = place_axes(fig, sc_rect)

        configs = list(combined_df.index)
        ari = combined_df['ARI'].values.astype(float)
        time_s = combined_df['train_time_s'].values.astype(float)
        colors = STUDY_PALETTE[:len(configs)]

        ax_sc.scatter(time_s, ari, c=colors, s=70, edgecolors='black',
                      linewidths=0.5, zorder=5)

        labels_xy = []
        for i, name in enumerate(configs):
            labels_xy.append((time_s[i], ari[i], _short_config(name)))

        labels_xy.sort(key=lambda t: (t[0], t[1]))
        placed = []
        for tx, ty, short in labels_xy:
            ox, oy = 6, 5
            for px, py, _, _ in placed:
                if abs(tx - px) < 2.0 and abs(ty - py) < 0.06:
                    oy = -12
                    break
            placed.append((tx, ty, ox, oy))
            ax_sc.annotate(short, (tx, ty),
                           textcoords='offset points', xytext=(ox, oy),
                           fontsize=FS_SMALL, ha='left',
                           arrowprops=dict(arrowstyle='-', color='grey',
                                          lw=0.4, shrinkA=0, shrinkB=3))

        x_margin = (time_s.max() - time_s.min()) * 0.08
        y_margin = (ari.max() - ari.min()) * 0.06
        ax_sc.set_xlim(time_s.min() - x_margin, time_s.max() + x_margin * 2)
        ax_sc.set_ylim(ari.min() - y_margin, ari.max() + y_margin)

        ax_sc.set_xlabel('Training Time (s)', fontsize=FS_AXIS)
        ax_sc.set_ylabel('ARI', fontsize=FS_AXIS)
        ax_sc.spines['top'].set_visible(False)
        ax_sc.spines['right'].set_visible(False)
        add_panel_label(ax_sc, 'b', x=-0.15, y=1.04)

    fig.text(0.5, RECT_TITLE_Y + 0.01, 'Study Overview: All Configurations',
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

    # Figure 1: Encoder comparison (standalone)
    print("Figure 1: Encoder comparison...")
    plot_fig1_encoder(df1)

    # Figure 2: Component analysis (bars + deltas composite)
    print("Figure 2: Component analysis (bars + deltas)...")
    plot_fig2_component(df2)

    # Figure 3: Ablation analysis (bars + deltas composite)
    print("Figure 3: Ablation analysis (bars + deltas)...")
    plot_fig3_ablation(df3)

    # Figure 4: Overview (heatmap + efficiency scatter)
    print("Figure 4: Overview (heatmap + scatter)...")
    plot_fig4_overview(combined)

    print(f"\nDone. 4 composite figures in: {FIGURES_DIR}")


if __name__ == '__main__':
    main()
