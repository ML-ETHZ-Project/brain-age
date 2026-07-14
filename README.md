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
| 2026-07-14 | Philippe (w/ Claude) | Stacking ensemble (leak-free 5-fold CV): GradientBoosting + Ridge(α=10) + KNN(k=15) base learners → Ridge meta-learner (StackingRegressor, internal 5-fold cross-fit) | **STACK R²=0.5179**; base learners: gbr 0.5065, knn 0.4471, ridge 0.3223 | not submitted | See `notebooks/stacking_comparison.py`. Stacking beats the best single model (gbr) by +0.0114. Adds a robust ±15-IQR clip after RobustScaler so the injected-noise columns (std ~1e22) don't overflow f_regression. Real but small gain; Ridge is weak on this noisy data and contributes diversity more than accuracy. |
| 2026-07-14 | Philippe (w/ Claude) | Tuned models folded into the stack: RandomizedSearchCV-tuned HistGradientBoosting (k=200) + GradientBoosting (k=150) as base learners + Ridge + KNN → Ridge meta-learner | **STACK R²=0.5240**; tuned singles: GBR 0.5201, HistGBM 0.5182; tuned-trees-only stack 0.5183 | not submitted | See `notebooks/stacking_tuned.py`. Tuning lifts a single GBR 0.5065→0.5201; diverse stack of tuned models reaches 0.5240. ⚠️ Mildly optimistic: tree hyperparameters selected on full train (selection bias) — nested CV is the unbiased follow-up. The two boosters are correlated (tuned-trees-only stack ≈ best single tree); Ridge/KNN diversity is what moves the stack. |

## Current best

- Approach: stacking ensemble of **tuned** models (tuned HistGradientBoosting + tuned GradientBoosting + Ridge + KNN → Ridge meta-learner), median impute, ±15-IQR clip, no outlier removal (`notebooks/stacking_tuned.py`)
- CV score: R²=0.5240 (5-fold) — ⚠️ mildly optimistic (tree hyperparameters selected on full train; nested CV pending for an unbiased number)
- Public LB score: not yet submitted
- Unbiased reference points: default-hyperparameter stack R²=0.5179 (`notebooks/stacking_comparison.py`); single-model baseline R²=0.5065 (`src/baseline.py`)
