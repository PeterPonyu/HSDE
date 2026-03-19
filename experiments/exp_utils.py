"""
Shared utilities for HSDE experiment scripts.
============================================
Follows the training-pipeline skill Phase 2 preprocessing exactly:
  1. Save raw counts in layers['counts']
  2. Normalize (target_sum=1e4) + log1p
  3. Select highly variable genes (n_top_genes=2000)
  4. Subsample to max_cells=3000
  5. Subset to (subsampled cells) × (HVG genes) and .copy()

The resulting adata1 is used by ALL models:
  - HSDE variants:  HSDE(adata1, layer='counts', ...)
  - External models: X = adata1.X.toarray()  (normalized HVG)
  - Labels:          get_labels(adata1)
"""

import sys, os, glob
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

# Add project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Add evaluator (configurable via HSDE_EVALUATOR_DIR env var)
_EVALUATOR_DIR = os.environ.get(
    "HSDE_EVALUATOR_DIR",
    os.path.expanduser("~/.copilot/skills/latent-space-evaluator"),
)
if os.path.isdir(_EVALUATOR_DIR):
    sys.path.insert(0, _EVALUATOR_DIR)
try:
    from metrics_expanded import compute_all_metrics
except ImportError:
    raise ImportError(
        f"Cannot import metrics_expanded. Set HSDE_EVALUATOR_DIR env var "
        f"or install the latent-space-evaluator package. Checked: {_EVALUATOR_DIR}"
    )

# ── Constants ──
MAX_CELLS = 3000
N_HVG = 2000
SEED = 42

EXCLUDE = ['GSE120575_melanomaHmCancer', 'GSE225948_bloodMmStrokeDev']

SELECTED_DATASETS = [
    # Cancer (6)
    'GSE98638_TcellLiverHmCancer',
    'GSE117988_MCCTumorCancer',
    'GSE155109_bcECHmCancer',
    'GSE149655_CAHmCancer',
    'GSE283205_hepatoblastomaCancer',
    'GSE168181_BreastHmCancer',
    # Development (6)
    'endo',
    'GSE142653pitHmDev',
    'setty',
    'hESC_GSE144024',
    'GSE130148_LungHmDev',
    'dentate',
]


def discover_datasets():
    """Find all h5ad files and filter to selected 12.

    Dataset directories are configurable via the HSDE_DATASET_DIRS environment
    variable (colon-separated list of paths). Falls back to ~/Downloads/ defaults.
    """
    _dataset_dirs_env = os.environ.get("HSDE_DATASET_DIRS", "")
    if _dataset_dirs_env:
        search_dirs = _dataset_dirs_env.split(os.pathsep)
    else:
        search_dirs = [
            os.path.expanduser("~/Downloads/CancerDatasets2"),
            os.path.expanduser("~/Downloads/CancerDatasets"),
            os.path.expanduser("~/Downloads/DevelopmentDatasets2"),
            os.path.expanduser("~/Downloads/DevelopmentDatasets"),
        ]
    all_files = []
    for d in search_dirs:
        all_files.extend(glob.glob(os.path.join(d, "*.h5ad")))
    all_files = [f for f in all_files if not any(e in f for e in EXCLUDE)]

    selected = []
    for name in SELECTED_DATASETS:
        matches = [f for f in all_files
                   if name in os.path.basename(f).replace('.h5ad', '')]
        if matches:
            selected.append(matches[0])
        else:
            print(f"⚠ Dataset not found: {name}")
    return selected


def get_labels(adata):
    """Extract cell type labels from adata.obs."""
    for col in ['cell_type', 'celltype', 'CellType', 'cell_types',
                'cluster', 'clusters', 'louvain', 'leiden',
                'annotation', 'label', 'labels', 'Group', 'group']:
        if col in adata.obs.columns:
            labels = adata.obs[col].values
            if hasattr(labels, 'cat'):
                labels = labels.astype(str)
            return labels
    # Fallback: compute leiden clustering
    print("  No labels found, computing leiden clusters...")
    sc.pp.neighbors(adata, use_rep='X_pca' if 'X_pca' in adata.obsm else None)
    sc.tl.leiden(adata, resolution=1.0)
    return adata.obs['leiden'].values


def load_and_preprocess(filepath):
    """Load h5ad and apply the MANDATORY preprocessing pipeline.

    Returns
    -------
    adata1 : AnnData
        Preprocessed data with:
        - adata1.X = normalized log-transformed HVG expression
        - adata1.layers['counts'] = raw integer counts (HVG subset)
        - adata1.n_obs <= MAX_CELLS (3000)
        - adata1.n_vars = N_HVG (2000)
    """
    adata = sc.read_h5ad(filepath)
    adata.obs_names_make_unique()
    adata.var_names_make_unique()

    # 1. Ensure sparse
    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)

    # 2. Save raw counts BEFORE normalization
    adata.layers['counts'] = adata.X.copy()

    # 3. Normalize + log-transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # 4. Select highly variable genes
    sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG)

    # 5. Subsample cells to fixed size
    rng = np.random.default_rng(SEED)
    if adata.shape[0] > MAX_CELLS:
        idxs = rng.choice(adata.shape[0], MAX_CELLS, replace=False)
    else:
        idxs = rng.permutation(adata.shape[0])

    # 6. Subset to (subsampled cells) x (HVG genes) and COPY
    adata1 = adata[idxs, adata.var['highly_variable']].copy()

    print(f"  Preprocessed: {adata.n_obs} cells -> {adata1.n_obs} cells, "
          f"{adata.n_vars} genes -> {adata1.n_vars} HVGs")

    return adata1


def get_dense_X(adata1):
    """Get dense normalized HVG matrix from preprocessed adata1.
    Use this as input for external models (GM-VAE, etc.)."""
    X = adata1.X
    if sp.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def evaluate_latent(latent, labels, dre_k=15):
    """Compute all metrics for a latent embedding."""
    # Encode string labels to integers (compute_all_metrics expects int)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    labels_int = le.fit_transform(np.asarray(labels).astype(str))
    raw = compute_all_metrics(latent, labels_int, dre_k=dre_k)
    return {k: v for k, v in raw.items()
            if not k.startswith('_') and np.isscalar(v)}


def get_done_datasets(tables_dir, prefix):
    """Check which datasets have already been processed (for resume)."""
    done = set()
    for f in glob.glob(os.path.join(tables_dir, f'{prefix}_*_df.csv')):
        name = os.path.basename(f).replace(f'{prefix}_', '').replace('_df.csv', '')
        done.add(name)
    return done
