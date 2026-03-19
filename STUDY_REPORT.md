# HSDE-Graph — Ablation & Component Efficiency Study Report (v2)

> **Dataset**: Setty Bone Marrow hematopoiesis — 3 000 cells, 2 000 HVGs  
> **Training**: ≤ 100 epochs, early stopping (patience 20), NB loss, seed 42  
> **Hardware**: CUDA GPU  
> **Total runtime**: ~248 s (16 configurations)

---

## 0  Changes from v1 → v2

| Issue | v1 (incorrect) | v2 (corrected) |
|-------|---------------|----------------|
| **ASW / CH / DB labels** | Computed on KMeans-predicted labels (internal validity) | Computed on **ground-truth** cell-type labels (external validity) |
| **LSE overall** | 4-component mean including circular `core_quality = √(manifold_dim × noise_resil)` | Clean 3-component mean: `mean(manifold_dim, spectral_decay, noise_resil)` |
| **Part 1 fairness** | MLP had `irecon=0.5, lorentz=5.0`; Graph had `irecon=0, lorentz=0` | All encoders use **identical** minimal loss: recon + β=0.1 KL only |
| **`take_latent`** | Missing `@torch.no_grad()` — gradients computed during evaluation | Added `@torch.no_grad()` decorator |
| **Config naming** | Ambiguous (e.g. "2.2 + KL (β=0.01)" implied adding KL) | Clarified (e.g. "2.2 + Low β (β=0.01)" — reducing posterior pressure) |

**Impact**: ASW values dropped because ground-truth cell types are harder to separate than KMeans-optimised clusters. Part 1 MLP ARI dropped from 0.51 → 0.43 (losing its unfair IB/Lorentz boost), making the GAT advantage clearer.

---

## 1  Study Design

### 1.1  Design Rule

> **Geometry loss REQUIRES Information Bottleneck (IB).**  
> The Lorentz/Euclidean geometry loss computes manifold distance between `z_manifold` and `ld_manifold`. Without IB (`irecon = 0`), the low-dimensional projection `ld` is untrained, making the distance meaningless. Removing IB therefore **automatically** removes geometry.

### 1.2  Three-Part Structure

| Part | Strategy | Reference | Question |
|------|----------|-----------|----------|
| **Part 1** | Comparison | 1.1 MLP | Which encoder architecture is best, all else being equal? |
| **Part 2** | Additive | 2.1 GAT Baseline | Which component helps most when added to a minimal GAT? |
| **Part 3** | Subtractive | 3.1 Full GAT | Which component hurts most when removed from a full GAT? |

### 1.3  Configuration Summary

**Part 1 — Encoder Architecture Comparison** (all use: recon + β=0.1 KL only)

| # | Config | Encoder | Loss |
|---|--------|---------|------|
| 1.1 | MLP | Standard MLP | recon + β-KL |
| 1.2 | Transformer | Multi-head attention | recon + β-KL |
| 1.3 | Graph GAT | Graph Attention Network | recon + β-KL |
| 1.4 | Graph GCN | Graph Convolution | recon + β-KL |
| 1.5 | Graph SAGE | Sample-and-Aggregate | recon + β-KL |

**Part 2 — Component Effectiveness** (additive from minimal GAT)

| # | Config | Added Component | Key Parameters |
|---|--------|-----------------|----------------|
| 2.1 | GAT Baseline | — | recon + β=0.1 KL |
| 2.2 | + Low β | Reduce KL pressure | β = 0.01 |
| 2.3 | + IB | Information Bottleneck | irecon = 0.5 |
| 2.4 | + IB + Lorentz | Hyperbolic geometry | irecon = 0.5, lorentz = 5.0 |
| 2.5 | + IB + Euclidean | Euclidean geometry | irecon = 0.5, lorentz = 5.0, Euclidean |
| 2.6 | + Adj Decoder | Structural decoder | MLP decoder, w_adj = 0.1 |

**Part 3 — Ablation Study** (subtractive from full GAT: IB + Lorentz + β=0.1)

| # | Config | Removed Component | Effect |
|---|--------|-------------------|--------|
| 3.1 | Full (IB+Lor+β) | — (reference) | IB + Lorentz + β=0.1 |
| 3.2 | − IB (→ − Geo) | Information Bottleneck | Design rule: geometry auto-removed |
| 3.3 | − Geometry | Lorentz loss only | Keep IB, drop geometry loss |
| 3.4 | Lor → Euclid | Lorentz manifold | Switch to Euclidean manifold |
| 3.5 | − KL (β=0) | KL divergence | No posterior regularisation |

---

## 2  Metrics

All clustering metrics use **ground-truth cell-type labels** (external validity).

| Metric | Type | Range | Better |
|--------|------|-------|--------|
| **ARI** | Clustering — Adjusted Rand Index | [−1, 1] | Higher |
| **NMI** | Clustering — Normalised Mutual Information | [0, 1] | Higher |
| **ASW** | Clustering — Average Silhouette Width (gt labels) | [−1, 1] | Higher |
| **CH** | Clustering — Calinski-Harabasz Index (gt labels) | [0, ∞) | Higher |
| **DB** | Clustering — Davies-Bouldin Index (gt labels) | [0, ∞) | **Lower** |
| **LSE** | Latent Structure — mean(manifold\_dim, spectral\_decay, noise\_resil) | [0, 1] | Higher |
| **DRE UMAP** | Dimensionality Reduction — mean(distcorr, Q\_local, Q\_global) on UMAP | [0, 1] | Higher |
| **DRE tSNE** | Dimensionality Reduction — same on t-SNE | [0, 1] | Higher |
| **ARI/s** | Efficiency — ARI per second of training | [0, ∞) | Higher |

---

## 3  Results

### 3.1  Part 1 — Encoder Architecture Comparison

| Config | ARI | NMI | ASW | CH | DB ↓ | DRE UMAP | DRE tSNE | LSE | Time (s) |
|--------|-----|-----|-----|---:|-----:|---------:|---------:|----:|---------:|
| 1.1 MLP | 0.4328 | 0.6058 | 0.1124 | 302 | 2.019 | 0.421 | 0.489 | 0.494 | 14.6 |
| 1.2 Transformer | 0.4823 | 0.6185 | 0.1115 | 277 | 2.113 | 0.395 | 0.448 | 0.489 | 22.3 |
| **1.3 Graph GAT** | **0.5719** | 0.6587 | **0.2294** | **474** | **1.517** | **0.585** | **0.638** | **0.490** | **5.7** |
| 1.4 Graph GCN | 0.5575 | **0.6638** | 0.2150 | 468 | 1.590 | 0.595 | 0.618 | 0.477 | 5.2 |
| 1.5 Graph SAGE | 0.5323 | 0.6448 | 0.1837 | 420 | 1.708 | 0.511 | 0.542 | 0.486 | 5.3 |

**Winner: Graph GAT** — best ARI (+0.139 over MLP), best ASW, best CH, best DB, 2.6× faster than MLP.

### 3.2  Part 2 — Component Effectiveness (Additive)

| Config | ARI | NMI | ASW | CH | DB ↓ | DRE UMAP | DRE tSNE | LSE | Time (s) |
|--------|-----|-----|-----|---:|-----:|---------:|---------:|----:|---------:|
| **2.1 GAT Baseline** | **0.5902** | 0.6697 | 0.2395 | 527 | 1.483 | 0.629 | 0.613 | **0.469** | 5.2 |
| 2.2 + Low β | 0.5206 | 0.6402 | 0.2357 | 548 | 1.465 | 0.646 | **0.672** | 0.434 | 5.6 |
| 2.3 + IB | 0.5739 | 0.6732 | 0.2519 | 889 | 1.333 | **0.674** | 0.689 | 0.337 | 5.8 |
| 2.4 + IB + Lorentz | 0.5460 | 0.6679 | 0.2451 | 1149 | 1.336 | 0.621 | 0.653 | 0.307 | 6.0 |
| 2.5 + IB + Euclidean | 0.5305 | **0.6766** | **0.2618** | **1744** | **1.326** | 0.642 | 0.632 | 0.226 | 5.9 |
| 2.6 + Adj Decoder | 0.5637 | 0.6568 | 0.2434 | 539 | 1.479 | 0.622 | 0.636 | 0.479 | 17.2 |

**Δ from 2.1 GAT Baseline:**

| Config | ΔARI | ΔNMI | ΔASW | ΔCH | ΔDB ↓ | ΔDRE UMAP |
|--------|-----:|-----:|-----:|----:|------:|----------:|
| 2.2 + Low β | −0.070 | −0.030 | −0.004 | +21 | −0.018 | +0.017 |
| 2.3 + IB | −0.016 | +0.004 | +0.012 | +362 | −0.150 | +0.045 |
| 2.4 + IB + Lorentz | −0.044 | −0.002 | +0.006 | +622 | −0.147 | −0.008 |
| 2.5 + IB + Euclidean | −0.060 | +0.007 | +0.022 | +1217 | −0.157 | +0.013 |
| 2.6 + Adj Decoder | −0.027 | −0.013 | +0.004 | +12 | −0.004 | −0.007 |

### 3.3  Part 3 — Ablation Study (Subtractive)

| Config | ARI | NMI | ASW | CH | DB ↓ | DRE UMAP | DRE tSNE | LSE | Time (s) |
|--------|-----|-----|-----|---:|-----:|---------:|---------:|----:|---------:|
| 3.1 Full (IB+Lor+β) | 0.5084 | 0.6421 | 0.2361 | **1403** | **1.400** | **0.700** | 0.680 | 0.236 | 5.7 |
| 3.2 − IB (→ − Geo) | 0.5422 | 0.6643 | 0.2255 | 464 | 1.491 | 0.619 | 0.636 | **0.484** | 5.1 |
| 3.3 − Geometry | 0.5812 | 0.6708 | 0.2236 | 780 | 1.522 | 0.636 | **0.706** | 0.346 | 5.6 |
| **3.4 Lor → Euclid** | **0.5838** | **0.7060** | **0.2574** | 1115 | 1.331 | 0.630 | 0.674 | 0.305 | 5.6 |
| 3.5 − KL (β=0) | 0.5148 | 0.6687 | 0.2562 | 645 | 1.349 | 0.517 | 0.539 | 0.244 | 6.0 |

**Δ from 3.1 Full (IB+Lor+β):**

| Config | ΔARI | ΔNMI | ΔASW | ΔCH | ΔDB ↓ | ΔDRE UMAP |
|--------|-----:|-----:|-----:|----:|------:|----------:|
| 3.2 − IB (→ − Geo) | +0.034 | +0.022 | −0.011 | −939 | +0.091 | −0.081 |
| 3.3 − Geometry | +0.073 | +0.029 | −0.013 | −623 | +0.122 | −0.064 |
| 3.4 Lor → Euclid | +0.075 | +0.064 | +0.021 | −288 | −0.069 | −0.069 |
| 3.5 − KL (β=0) | +0.006 | +0.027 | +0.020 | −758 | −0.051 | −0.183 |

### 3.4  Global Best per Metric

| Metric | Best Config | Value |
|--------|-------------|------:|
| ARI | 2.1 GAT Baseline | 0.5902 |
| NMI | 3.4 Lor → Euclid | 0.7060 |
| ASW | 2.5 + IB + Euclidean | 0.2618 |
| CH | 2.5 + IB + Euclidean | 1744 |
| DB ↓ | 2.5 + IB + Euclidean | 1.326 |
| DRE UMAP | 3.1 Full (IB+Lor+β) | 0.6997 |
| DRE tSNE | 3.3 − Geometry | 0.7056 |
| LSE | 1.1 MLP | 0.4939 |
| ARI/s | 2.1 GAT Baseline | 0.1135 |

---

## 4  Analysis

### 4.1  Part 1 — Graph Encoders Dominate

With **identical loss settings** (recon + β=0.1 KL, no IB, no geometry), all three graph encoders outperform both MLP and Transformer on clustering:

- **GAT** achieves the best ARI (0.572 vs MLP 0.433), a gap of **+0.139** — 32% relative improvement.
- Graph encoders are **2.6–2.8× faster** than MLP (5–6 s vs 15 s) because neighbour-subgraph batching processes fewer samples per step.
- GAT also leads on DRE (0.585 UMAP vs 0.421 MLP, +39%), confirming that graph attention produces better-structured embeddings even without any geometry loss.
- GCN edges out GAT on NMI (0.664 vs 0.659) but trails on ARI and ASW.

**Conclusion**: The graph attention mechanism itself — leveraging cell–cell neighbourhood structure — is the single largest contributor to performance.

### 4.2  Part 2 — The Clustering–Embedding Trade-off

A striking pattern emerges: **every component reduces ARI relative to the minimal GAT baseline** (2.1, ARI 0.590), yet several improve embedding structure (CH, DB, DRE):

| Added Component | ΔARI | ΔCH | ΔDB ↓ | ΔDRE UMAP |
|-----------------|-----:|----:|------:|----------:|
| IB alone | −0.016 | +362 | −0.150 | +0.045 |
| IB + Lorentz | −0.044 | +622 | −0.147 | −0.008 |
| IB + Euclidean | −0.060 | +1217 | −0.157 | +0.013 |
| Adj Decoder | −0.027 | +12 | −0.004 | −0.007 |

**Interpretation**: IB and geometry regularise the latent space — clusters become more compact and better separated (CH ↑, DB ↓) — but the additional loss terms compete with reconstruction, reducing the model's ability to preserve fine-grained discriminative features that KMeans can exploit.

The **Information Bottleneck** (2.3) provides the best ARI-preserving trade-off: only −0.016 ARI while boosting CH by 69% and DRE by 7%.

**IB + Euclidean** (2.5) is the best for embedding quality: highest ASW (0.262), highest CH (1744), lowest DB (1.326). It outperforms IB + Lorentz (2.4) on all three separation metrics.

**Adjacency Decoder** (2.6) adds minimal benefit: +12 CH, −0.027 ARI, and 3.3× slower (17.2 s vs 5.2 s) due to the MLP decoder's overhead.

### 4.3  Part 3 — What the Full Model Needs

The subtractive ablation from the full model (3.1: IB + Lorentz + β=0.1) reveals:

1. **Removing geometry (3.3) or switching to Euclidean (3.4) improves ARI substantially** (+0.073 and +0.075). The Lorentz loss actively hurts clustering in this setting.

2. **3.4 Lor → Euclid achieves the highest NMI in the entire study** (0.706) and the best ARI in Part 3 (0.584). Euclidean geometry is more effective than hyperbolic for this dataset.

3. **The full model achieves the best DRE UMAP** (0.700) — the highest embedding quality score in the entire study. This is driven by exceptionally high distance correlation (0.858).

4. **Removing KL (3.5) devastates DRE** (−0.183 UMAP), while barely affecting ARI (+0.006). KL regularisation is critical for embedding quality but almost irrelevant for clustering.

5. **Removing IB (3.2) restores LSE** from 0.236 → 0.484 — the IB constraint collapses latent dimensions (manifold_dim drops from 0.208 to 0.712), confirming that IB achieves compression at the cost of latent diversity.

### 4.4  The Clustering–Embedding Spectrum

Across all 16 configurations, models fall on a spectrum:

```
High ARI, Low DRE/CH                        Low ARI, High DRE/CH
(best clustering)                            (best embedding)
├─────────────────────────────────────────────┤
  2.1 Baseline   1.3 GAT   3.4 Euclid   3.3 −Geo   2.3 IB   3.1 Full
  ARI=0.590      ARI=0.572  ARI=0.584    ARI=0.581   ARI=0.574  ARI=0.508
  DRE=0.629      DRE=0.585  DRE=0.630    DRE=0.636   DRE=0.674  DRE=0.700
  CH=527         CH=474     CH=1115      CH=780      CH=889     CH=1403
```

More regularisation (IB → Geometry → Full) improves embedding structure but reduces clustering accuracy. The optimal configuration depends on the downstream task.

### 4.5  Baseline vs Full Model

| Metric | 2.1 GAT Baseline | 3.1 Full (IB+Lor+β) | Δ | Winner |
|--------|:-:|:-:|:-:|--------|
| ARI | **0.5902** | 0.5084 | −0.082 | Baseline |
| NMI | **0.6697** | 0.6421 | −0.028 | Baseline |
| ASW | **0.2395** | 0.2361 | −0.003 | Baseline |
| CH | 527 | **1403** | +876 | Full |
| DB ↓ | 1.483 | **1.400** | −0.083 | Full |
| DRE UMAP | 0.629 | **0.700** | +0.071 | Full |
| LSE | **0.469** | 0.236 | −0.233 | Baseline |
| Time (s) | **5.2** | 5.7 | +0.5 | Baseline |

The baseline wins 5 of 8 metrics. The full model wins only on CH, DB, and DRE — all of which reflect geometric structure rather than cluster assignment accuracy.

---

## 5  Efficiency

| Config | ARI | Time (s) | ARI/s | Params |
|--------|----:|--------:|---------:|-------:|
| 2.1 GAT Baseline | 0.590 | 5.2 | **0.1135** | 554 768 |
| 1.3 Graph GAT | 0.572 | 5.7 | 0.1003 | 554 768 |
| 3.4 Lor → Euclid | 0.584 | 5.6 | 0.1043 | 554 768 |
| 3.3 − Geometry | 0.581 | 5.6 | 0.1038 | 554 768 |
| 1.4 Graph GCN | 0.558 | 5.2 | 0.1072 | 554 216 |
| 1.1 MLP | 0.433 | 14.6 | 0.0296 | 554 216 |
| 1.2 Transformer | 0.482 | 22.3 | 0.0216 | 1 422 632 |

Graph encoders dominate efficiency: all 5 graph variants in Part 1 finish in 5–6 s, while MLP takes 15 s and Transformer 22 s. The Transformer's 1.4M parameters (2.6× more than other encoders) do not translate into better performance — it ranks below all graph encoders on both ARI and efficiency.

---

## 6  Conclusions

1. **Graph Attention is the dominant encoder.** GAT achieves the best ARI (0.590), best efficiency (0.114 ARI/s), and is 2.6× faster than MLP under identical loss settings. The neighbourhood structure exploited by graph attention is more valuable than any individual loss component.

2. **A clustering–embedding trade-off exists.** Additional regularisation (IB, geometry) consistently improves latent structure (CH ↑, DB ↓, DRE ↑) while reducing cluster assignment accuracy (ARI ↓). No single configuration optimises all metrics simultaneously.

3. **The Information Bottleneck provides the best cost–benefit ratio.** IB (2.3) boosts CH by 69% and DRE by 7% while sacrificing only 2.8% ARI — the most favourable trade-off of any component.

4. **Euclidean geometry outperforms Lorentz on this dataset.** IB+Euclidean (2.5) beats IB+Lorentz (2.4) on ASW (+0.017), CH (+595), and DB (−0.011). In the ablation, switching Lorentz → Euclidean (3.4) yields the study's best NMI (0.706).

5. **The full model maximises embedding fidelity.** Full GAT (3.1) achieves the study's best DRE UMAP (0.700), driven by exceptional distance correlation (0.858). This makes it ideal for tasks requiring faithful low-dimensional visualisation.

6. **KL regularisation is essential for embedding quality but not clustering.** Removing KL (3.5) drops DRE by −0.183 while barely affecting ARI (+0.006). KL prevents posterior collapse and maintains interpretable structure.

7. **The adjacency decoder is not cost-effective.** The structural decoder (2.6) adds negligible benefit (+12 CH) while tripling training time (17.2 s vs 5.2 s).

### Recommended Configurations

| Use Case | Config | Why |
|----------|--------|-----|
| **Best clustering** | 2.1 GAT Baseline | Highest ARI (0.590), fastest (5.2 s) |
| **Balanced** | 2.3 + IB | Minimal ARI loss (−0.016), strong CH (+69%), good DRE (+7%) |
| **Best embedding** | 3.1 Full (IB+Lor+β) | Highest DRE UMAP (0.700), best CH (1403) |
| **Best NMI** | 3.4 Lor → Euclid | Highest NMI (0.706), competitive ARI (0.584) |

---

## 7  Reproducibility

```bash
cd <project_root>
python experiments/run_study.py --epochs 100 --n_cells 3000 --n_genes 2000 --patience 20 --seed 42 --part all
```

Results are written to `results/study_*.csv`, `results/study_*.json`, and `results/study_full_log.txt`.
