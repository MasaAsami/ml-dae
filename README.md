# DEA meets Machine Learning
## Orthogonalized DEA for Focal Input Efficiency
- [preprint](https://papers.ssrn.com/abstract=5216241)

## Tree-Based DEA for Focal Input Efficiency
- [preprint](https://papers.ssrn.com/abstract=5216670)

# 🛠️ WIP: Refactoring in Progress

## Reproducibility (Tree-Based DEA)

All figures, tables, and reported numbers in the *Tree-Based DEA* manuscript can
be regenerated from this repository.

**Environment.** Python 3.9+ with `numpy`, `scipy`, `pandas`, `scikit-learn`,
`seaborn`, `matplotlib`, and `tqdm`. (`lightgbm` and `mapie` are only needed for
the separate Orthogonalized-DEA notebook.)

**Contents.**
- `model/dae.py` — DEA efficiency solver, CCR (constant-returns) ratio form
  (`fmin_slsqp`); returns a score in `(0, 1]`.
- `notebooks/tree_based_DEA.ipynb` — simulation study, within-leaf correlation
  analysis, and the Japanese-firm empirical illustration.
- `notebooks/revision_analysis.py` — Monte Carlo **bias/MSE** of the focal-input
  slope `τ⁽ᵐ⁾` (global vs leaf-wise) and the **stability / hyperparameter**
  sensitivity analyses.
- `notebooks/make_figures.py` — regenerates every manuscript figure as **vector
  PDF**.
- `data/gbizinfo_df_japan2023.csv` — firm-level dataset (METI gBizINFO, FY2023).

**Commands** (from the repository root):
```bash
# Bias/MSE + stability + hyperparameter tables (prints tables; writes CSVs)
python notebooks/revision_analysis.py
#   -> notebooks/out_bias_mse.csv, out_stability_realdata.csv, out_hyperparam_sweep.csv

# All manuscript figures as vector PDF
PYTHONPATH=notebooks python notebooks/make_figures.py
#   -> sim_regplot.pdf, correlation-hist.pdf, stability-hist.pdf,
#      multi-leaf.pdf, single-leaf.pdf (written into the LaTeX source directory)
```

The figure outputs are written into the manuscript's LaTeX source tree, which is
kept local (git-ignored); the CSV outputs land alongside the scripts in
`notebooks/`.
