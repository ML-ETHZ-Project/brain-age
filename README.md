# Brain Age

Kaggle competition: [eth-fdd-competition](https://www.kaggle.com/competitions/eth-fdd-competition/overview) — brain age prediction. Team project, ML-ETHZ-Project org.

## Team

- Wojciech Bentkowski — wbentkowski@student.ethz.ch
- André Waser — wasera@student.ethz.ch
- Philippe Haas — haasph@student.ethz.ch

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Register the Jupyter kernel for this venv so notebooks can pick it up:

```bash
python -m ipykernel install --user --name=brain-age --display-name="Python (brain-age)"
```

### Kaggle API

1. Get an API token: Kaggle → your profile → *Settings* → *API* → *Create New Token*. This downloads `kaggle.json`.
2. Place it at `~/.kaggle/kaggle.json` and lock down permissions:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```
3. Never commit `kaggle.json` — it's gitignored, but double-check before pushing.

Download the competition data into `data/raw/`:

```bash
scripts/download_data.sh
```

Submit a predictions file:

```bash
scripts/submit.sh submissions/my_submission.csv "short description of the approach"
```

## Structure

- `data/` — raw/processed data (gitignored, never commit datasets)
- `notebooks/` — exploration notebooks, one per person/experiment
- `src/` — reusable code (features, models, utils) once stable
- `scripts/` — Kaggle CLI helpers (download data, submit predictions)
- `submissions/` — local submission files (gitignored)

## Team workflow

- Don't commit directly to `main`. Work on a branch per person/experiment (e.g. `andre/xgboost-baseline`), open a PR, and get at least one teammate to review before merging.
- Keep `main` in a working state — anything merged should run end to end.
- One notebook per person/experiment under `notebooks/`; once an approach is worth reusing, move the code into `src/`.
- After a training run that's worth recording, add a row to the experiment log below (in the same PR).

## Experiment log

| Date | Who | Approach | CV score | Public LB | Notes |
|------|-----|----------|----------|-----------|-------|
| 2026-07-12 | André (w/ Claude) | ~~Median impute -> RobustScaler -> VarianceThreshold -> SelectKBest(f_regression, k=100) -> IsolationForest outlier removal -> GradientBoostingRegressor~~ | ~~R²=0.57 (5-fold CV)~~ | not submitted | **Superseded 2026-07-14**: the 0.57 came from a leaky evaluation (preprocessing + outlier detector fit on all training data before the CV split). See the 07-14 outlier-removal row below. |
| 2026-07-14 | André (w/ Claude) | Subtask 0 ablation: imputer choice (mean/median/most_frequent/KNN/iterative), rest of pipeline fixed | median best: R²=0.5065; mean 0.5028; KNN 0.4968; iterative 0.4948; most_frequent 0.4639 | n/a (ablation) | See `notebooks/impute_comparison.py`. Confirms median impute in `src/baseline.py` is the right call; fancier imputers don't pay off at n=1212 with ~300 noisy/irrelevant columns. |
| 2026-07-14 | André (w/ Claude) | Subtask 1 ablation, redone leak-free (detector fit inside each CV fold, not on full training set before splitting): IsolationForest (contamination 0.02-0.12) and LOF vs no removal | no removal wins: R²=0.5065; IsolationForest 0.44-0.48 across contaminations; LOF(c=0.05) 0.4928 | n/a (ablation) | See `notebooks/outlier_comparison.py`. Outlier removal actively hurts once leakage is fixed — GradientBoostingRegressor is already robust to outliers and dropping rows just loses signal. `src/baseline.py` no longer filters training rows; it still emits `data/processed/outlier_labels.csv` as the required classification artifact. |
| 2026-07-16 | Wojciech (w/ Claude) | Subtask 2 ablation: SelectKBest score_func, f_regression vs mutual_info_regression (k=100, rest of pipeline fixed) | f_regression wins: R²=0.5061; mutual_info R²=0.4854 | n/a (ablation) | See `notebooks/feature_selection_comparison.py`. Mutual information (KL divergence between joint and product-of-marginals) scores any dependency, not just linear ones, but its k-NN density estimate adds noise at n=1212 and doesn't beat a plain linear-correlation F-test here. Age vs. brain-volume features are apparently linear enough that f_regression already captures the signal; keep it as the SelectKBest score_func in `src/baseline.py`. |
| 2026-07-16 | Wojciech (w/ Claude) | Subtask 2 ablation: SelectKBest(k=100) vs. new `CorrelationPrunedKBest` (greedy correlation-cutoff pruning down to 25 features, thresholds 0.8/0.9/0.95), rest of pipeline fixed | SelectKBest(k=100) wins: R²=0.5061; corr-pruned 25 features: 0.4475-0.4581 across thresholds | n/a (ablation) | See `notebooks/feature_selection_corr_cutoff.py` and `src/feature_selection.py`. `feature_visualization.ipynb` found some top age-correlated features are highly inter-correlated (e.g. x133/x334/x465, |r|>0.9), so pruning redundancy seemed promising — but cutting to 25 features loses more signal than the redundancy removal recovers, regardless of cutoff strictness. Superseded by the same-day n=100 follow-up below. |
| 2026-07-16 | Wojciech (w/ Claude) | Subtask 2 follow-up: same `CorrelationPrunedKBest`, but n_features=100 (matching SelectKBest's k, so width is held fixed and only redundant-vs-distinct picks differ), thresholds 0.8/0.9/0.95 | corr_threshold=0.9 wins: R²=0.5141; thresh=0.8: 0.5073; thresh=0.95: 0.5052; vs SelectKBest(k=100) baseline 0.5061 | n/a (ablation) | See `notebooks/feature_selection_corr_cutoff.py`. Keeping the feature count at 100 (not cutting to 25) while swapping redundant top-ranked features for the next-best distinct ones improves CV R^2 by ~0.008 over plain SelectKBest — the earlier n=25 attempt lost signal from having too few features, not from the redundancy pruning itself. **Adopted**: `src/baseline.py` now uses `CorrelationPrunedKBest(n_features=100, corr_threshold=0.9)` in place of `SelectKBest`. |
| 2026-07-16 | Wojciech (w/ Claude) | Data-leakage sanity check: swept Pearson + Spearman correlation with age, `id`-vs-age ordering, per-feature unique-value counts, and best single-feature linear-fit residuals, looking for any feature suspiciously over-correlated with the target | no leak found | n/a (sanity check, no CV run) | Ad hoc check, no notebook artifact. Top correlations decay smoothly (0.42-0.46 Pearson, 0.5-0.6 Spearman) with no cliff between rank 1 and the rest (a real leak typically shows one feature far above the pack); `id` vs age correlation is -0.029 (no row-order leak from sorting-then-splitting); top features have >1100/1212 unique values (continuous measurements, not a binned/encoded leak); best single-feature linear fit still leaves 89% of age's variance in the residual. Correlations in the 0.4-0.6 range match published brain-age literature — consistent with genuine anatomical signal, not injected leakage. |
| 2026-07-16 | Wojciech (w/ Claude) | Feature skewness: decided against log-transforming or dropping highly-skewed features (e.g. the injected noise columns flagged in `feature_visualization.ipynb`) | n/a (design decision, no CV run) | n/a | `GradientBoostingRegressor` splits on rank order, so it's invariant to monotonic transforms like `log` — skew doesn't affect it. Removal is also redundant: `CorrelationPrunedKBest` already drops low-signal features by correlation with age regardless of their skew. Revisit only if a linear/distance-based model (Ridge, KNN, SVR) re-enters the candidate pool, where skew would start to matter. |
| 2026-07-16 | Wojciech (w/ Claude) | Experiment: unsupervised sparse autoencoder bottleneck (hidden=32/64/128, sparsity weight 1e-3/1e-2, L1 penalty on hidden activations) vs. `CorrelationPrunedKBest(n=100)`, rest of pipeline fixed | `CorrelationPrunedKBest` wins by a wide margin: R²=0.5141 vs. sparse AE 0.30-0.34 across all hidden sizes/sparsity weights tried | n/a (ablation) | See `notebooks/sparse_autoencoder_experiment.py` and `src/sparse_autoencoder.py` (adds `torch` to `requirements.txt`). The autoencoder is unsupervised (trained to reconstruct X, never sees age), and with only ~970 training rows per fold against 800+ noisy input columns, its bottleneck can't reliably prioritize the age-relevant subset of variance the way the supervised correlation-based selector does. Not adopted; `src/baseline.py` is unchanged. |
| 2026-07-16 | Wojciech (w/ Claude) | Experiment: plain PCA (n_components=10/25/50/100/200) vs. `CorrelationPrunedKBest(n=100)`, rest of pipeline fixed | `CorrelationPrunedKBest` wins by a wide margin: R²=0.5141 vs. PCA 0.39-0.40 across all component counts tried | n/a (ablation) | See `notebooks/pca_experiment.py`. Same story as the sparse-autoencoder experiment above: PCA is unsupervised (maximizes variance, never sees age), and with ~300+ near-irrelevant/noisy input columns (per `CLAUDE.md`/`feature_visualization.ipynb`), the top variance directions don't reliably align with age-predictive signal the way the supervised correlation-based selector does. Result is flat across component counts, confirming it's not just a wrong-k issue. Not adopted; `src/baseline.py` is unchanged. |

## Current best

- Approach: GradientBoostingRegressor on `CorrelationPrunedKBest`-100 features (correlation-cutoff pruning, threshold 0.9, of the same f_regression ranking SelectKBest uses), median impute, no outlier removal (`src/baseline.py`)
- CV score: R²=0.5141 (5-fold)
- Public LB score: not yet submitted
