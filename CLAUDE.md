# Brain Age — Kaggle competition

3-person team project (Wojciech Bentkowski, André Waser, Philippe Haas) for the
[eth-fdd-competition](https://www.kaggle.com/competitions/eth-fdd-competition/overview) Kaggle
competition — predict a subject's age from 832 anatomical brain features extracted from MRI via
FreeSurfer (feature extraction is already done; we work from the CSVs, not raw images).

- Metric: **R²** (`sklearn.metrics.r2_score`), maximize. Public leaderboard baseline to pass: **0.5**.
- Deadline: 2026-08-11 23:59. Team names must be alphanumeric. Public/private LB split — don't
  overfit to the public leaderboard.
- Full brief: `Project week 1.pdf` in repo root.

## On startup

- Check that `.venv/` exists and `requirements.txt` is installed into it; create/install if not.
- Check that a Jupyter kernel is registered for `.venv` (`jupyter kernelspec list`); register it if missing.

## Data files

`data/raw/` (gitignored, pull with `scripts/download_data.sh`):

- `X_train.csv` (1212 rows × 833 cols: `id` + `x0..x831`), `y_train.csv` (`id`, `y`=age, range 42-97)
- `X_test.csv` (776 rows × 833 cols, same schema, age unknown)
- `sample.csv` — submission format template: columns `id,y`, `id` 0-indexed matching `X_test` row order.

## Data quirks (confirmed by EDA, see `notebooks/eda.py`)

The organizers deliberately corrupted the FreeSurfer features in three ways — handle all three
before regressing:

1. **Missing values**: ~7.6% of all cells are NaN, spread across every column and every row (no
   row or column is NaN-free). Must impute (do NOT drop rows/cols). `notebooks/impute_comparison.py`
   benchmarks mean/median/most_frequent/KNN/iterative imputation with the rest of the pipeline
   held fixed: median wins (CV R²=0.5065), narrowly ahead of mean (0.5028); KNN (0.4968) and
   iterative/MICE (0.4948) don't pay for their extra complexity at this sample size (1212 rows)
   and with ~300 near-irrelevant columns; most_frequent is worst (0.4639), as expected for
   continuous measurements. Stick with median unless the dataset or downstream pipeline changes.
2. **Irrelevant/noise features**: a handful of columns (e.g. `x665`, `x173`, `x596` in the original
   EDA) have insane magnitudes (std up to ~1e22) — clearly injected noise, not real anatomical
   measurements. A few columns are also constant (zero variance, e.g. `x104`, `x129`, `x489`,
   `x530`). ~300+ of the 832 columns show negligible correlation with age. Use `RobustScaler`
   (median/IQR-based, so huge-scale noise columns don't blow up distance/gradient-based models)
   and a feature-selection step (`VarianceThreshold` + `SelectKBest`) before modeling.
3. **Outliers**: some training rows are outlier subjects, not just outlier feature values. Task
   spec requires classifying each training row as outlier/not — but see below, this classification
   should NOT be used to filter the regressor's training data.

Column indices/names above may shift if the dataset is regenerated — re-run `notebooks/eda.py`
rather than trusting hardcoded names blindly.

## Pipeline stages (matches the assignment's required subtasks)

1. Impute missing values (train+test, fit imputer on train only).
2. Outlier detection on training rows → classify as a standalone artifact (subtask deliverable),
   but do NOT use it to filter the regressor's training data — see below, it hurts R².
3. Feature selection → label features selected/unselected (drop irrelevant + redundant).
4. Regression → predict age, evaluate with R² via cross-validation on held-out folds.

Always fit preprocessing (imputer, scaler, selector, outlier detector) on **training data only**,
then `.transform()` the test set — never fit on test data or leak test rows into fitting. This
also means: when validating whether a step (outlier removal, an imputer choice, etc.) helps,
fit that step *inside* each CV fold on that fold's training rows only. Fitting it on the full
training set before splitting into folds leaks validation-row information into the decision and
inflates the reported R² — this bit us once already (see outlier removal below) so double-check
new ablations don't repeat it.

Current baseline (`src/baseline.py`): median impute → RobustScaler → VarianceThreshold →
SelectKBest(f_regression, k=100) → GradientBoostingRegressor, chosen by 5-fold CV against
Ridge/RandomForest. ~0.5065 CV R².

**Outlier removal was tried and rejected**: an earlier version filtered IsolationForest-flagged
training rows before fitting the regressor, reporting ~0.57 CV R² — but that evaluation fit the
preprocessor (including SelectKBest, which looks at `y`) and the outlier detector on the *entire*
training set before splitting into CV folds, leaking validation-fold information into both feature
selection and outlier detection. `notebooks/outlier_comparison.py` redid this leak-free (detector
fit inside each fold, on that fold's training rows only) and found removal *hurts* at every
contamination level tried (0.44-0.49 vs 0.5065 with no removal) — GradientBoostingRegressor is
already robust to outliers, and dropping rows just loses training signal. The subtask still
requires classifying training rows as outlier/not, so `src/baseline.py` produces that
classification as a standalone artifact (`data/processed/outlier_labels.csv`, IsolationForest,
contamination=0.05) without using it to filter what the regressor trains on.

## Working conventions

- Never commit real data (`data/`), submission files (`submissions/`), or Kaggle credentials
  (`kaggle.json`) — all gitignored, double-check before staging.
- Don't commit directly to `main`; use a branch per person/experiment and a PR.
- Notebooks live under `notebooks/`, one per person/experiment. Code worth reusing across
  notebooks moves into `src/`.
- Keep experiments reproducible: fixed `random_state=42` everywhere, k-fold CV (not a single
  train/val split) for reporting R² before submitting — the dataset is small (1212 rows).
- When a training run produces a result worth keeping, add a row to the experiment log table in
  `README.md`.
- Use `scripts/download_data.sh` and `scripts/submit.sh` for Kaggle CLI interactions rather than
  ad hoc `kaggle` commands.
