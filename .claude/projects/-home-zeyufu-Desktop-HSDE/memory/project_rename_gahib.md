---
name: HSDE renamed to GAHIB
description: Model renamed from HSDE (Hyperbolic SDE) to GAHIB (Graph Attention Hyperbolic Information Bottleneck) — all code, paper, and repo updated
type: project
---

Model renamed HSDE → GAHIB on 2026-03-27.

**Why:** "HSDE" stood for "Hyperbolic SDE-Regularized VAE" but the current architecture has no SDE component — it's GAT + Information Bottleneck + Lorentz Hyperbolic geometry. The acronym was misleading.

**How to apply:**
- Python package: `gahib/` (was `hsde/`), class `GAHIB` (was `HSDE`)
- Results dir: `GAHIB_results/` (was `HSDE_results/`)
- Paper file: `paper/gahib_paper.tex` (was `hsde_paper.tex`)
- GitHub repo: `PeterPonyu/GAHIB` (old `PeterPonyu/HSDE` abandoned)
- Paper assets excluded from remote repo via `.gitignore`
- `torchsde` external dependency was NOT renamed (false positive caught and fixed)
