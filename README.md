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
| 2026-07-16 | Wojciech (w/ Claude) | Experiment: small feedforward NN regressor (MLP, hidden sizes (32,)/(64,32)/(128,64), dropout 0.3-0.5, weight decay 1e-3/1e-2) vs. `GradientBoostingRegressor`, same preprocessing (`CorrelationPrunedKBest(n=100)`) | GradientBoosting still wins: R²=0.5141; best MLP config (128,64 hidden, dropout=0.3) got 0.4770; smaller/more-regularized MLPs did worse (down to 0.3255 for a single 32-unit layer) | n/a (ablation) | See `notebooks/nn_regressor_experiment.py` and `src/neural_net_regressor.py`. Larger hidden layers consistently beat smaller ones, suggesting the gap could narrow further with more tuning (architecture, epochs, learning-rate schedule) — but at ~970 training rows per fold and 100 features, a from-scratch MLP just doesn't have the data volume to out-learn a tree ensemble's inductive bias for tabular data. Not adopted; `src/baseline.py` keeps `GradientBoostingRegressor`. |
| 2026-07-16 | Wojciech (w/ Claude) | NN follow-up: more epochs (600 vs 300) + cosine-annealing LR schedule, a bigger (256,128) architecture, and a new `EnsembleMLPRegressor` (averages 5 identically-configured MLPs trained from different seeds) vs. the round-1 best single MLP(128,64, e=300)=0.4770 | more epochs+schedule alone slightly hurt (128,64 e=600 cosine: 0.4666); bigger architecture helped modestly (256,128 e=600 cosine: 0.4804); 5-seed ensembling of MLP(128,64) gave the best NN result yet: 0.4862 — still below GradientBoosting's 0.5141 | n/a (ablation) | See `notebooks/nn_regressor_experiment.py` and `src/neural_net_regressor.py` (`EnsembleMLPRegressor`, `TorchMLPRegressor`'s new `lr_schedule` option). Of the three levers tried, seed-ensembling gave the clearest lift, consistent with a from-scratch NN on ~970 rows being far more init/shuffle-sensitive than a tree ensemble (which is itself already an ensemble) — more epochs and a decaying LR didn't help on their own. Gap to GradientBoosting narrowed (0.5141 - 0.4862 = 0.028, down from 0.5141 - 0.4770 = 0.037) but not closed. Not adopted; `src/baseline.py` keeps `GradientBoostingRegressor`. Untried: ensembling the bigger (256,128) architecture, which combines the two levers that each helped independently. |
| 2026-07-16 | Wojciech (w/ Claude) | Cross-branch combination: swapped philippe/ensembling's `StackingRegressor` (GBR + Ridge(α=10) + KNN(k=15) → Ridge meta-learner, `notebooks/stacking_comparison.py` on that branch) to use this branch's `CorrelationPrunedKBest(n=100, thresh=0.9)` instead of its original `SelectKBest(k=100)` | **new best: R²=0.5229** — beats both individual improvements (`CorrelationPrunedKBest`+GBR alone: 0.5141; `SelectKBest`+stack, reproduced here: 0.5175 ≈ philippe/ensembling's reported 0.5179) | n/a (ablation) | See `notebooks/stacking_with_corr_pruned.py`. feature-section and philippe/ensembling independently improved different, non-overlapping pipeline stages (feature selection vs. modeling) off the same fork point (`main` @ `5e833eb`) — this combines them. Gain isn't fully additive (naive addition would predict ~0.5255): both improvements partly address the same redundant/noisy-feature problem, so their benefits overlap somewhat. Not yet adopted into `src/baseline.py` pending a decision on merging with philippe/ensembling's branch (avoid duplicating that work) — see conversation for options. |

## Current best

- Approach: GradientBoostingRegressor on `CorrelationPrunedKBest`-100 features (correlation-cutoff pruning, threshold 0.9, of the same f_regression ranking SelectKBest uses), median impute, no outlier removal (`src/baseline.py`)
- CV score: R²=0.5141 (5-fold)
- Public LB score: R²=0.6382 (submitted 2026-07-16)
