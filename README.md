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
| 2026-07-21 | Claude (instructions.rtf sprint) | Lane A — outlier retry, leak-free (manual per-fold loop): EllipticEnvelope/OneClassSVM hard removal, soft downweighting (detector-flagged rows get `sample_weight=0.3` instead of being dropped), and `GradientBoostingRegressor(loss="huber")` with no removal | Hard removal 0.4728–0.4939 (worse, again); huber loss 0.5077 (marginal); soft downweight (IsolationForest c=0.05) best at R²=0.5146 | not submitted | See `notebooks/outlier_retry.py`. Hard removal keeps losing regardless of detector geometry (Mahalanobis- or SVM-based), consistent with the 07-14 finding. Soft downweighting keeps all rows and gave the best of the four retried ideas, but was superseded by the larger gain from lane E below. |
| 2026-07-21 | Claude (instructions.rtf sprint) | Lane B — imputation refinement: `TopCorrMixedImputer` (`IterativeImputer(BayesianRidge)` on the `n_fancy` columns most `|corr(x,y)|`-ranked on training rows only, median on the rest), swept `n_fancy` ∈ {30,40,50}; also tried median+missingness-indicator | `n_fancy=40`: R²=0.5121 (best, and lowest fold-to-fold std of any imputer variant tried: 0.0253 vs 0.0346 for blanket median); missingness-indicator identical to blanket median (no added value) | not submitted | See `notebooks/impute_refine.py`. Folded into the final combined pipeline (`src/advanced_pipeline.py`) as the imputer. |
| 2026-07-21 | Claude (instructions.rtf sprint) | Lane C — feature selection refinement: scorer (`f_regression` vs `mutual_info_regression`), `SelectFromModel(Lasso)` (4 alphas) and `SelectFromModel(GBR-importance)`, then a `k` sweep for the winning scorer | `SelectKBest(f_regression, k=200)`: R²=0.5119 (only configuration to beat baseline); `mutual_info_regression`, both `SelectFromModel` variants, and `k∈{50,150,300,all}` all underperformed (0.464–0.495) | not submitted | See `notebooks/feature_selection_refine.py`. `k=200` (double the baseline's `k=100`) folded into the final combined pipeline. Diagnostic per-fold OLS Wald-test pass (reporting-only) flagged `x641,x819,x458,x233,x325` as significant in 5/5 folds. |
| 2026-07-21 | Claude (instructions.rtf sprint) | Lane D — distributional regression: age binned into Gaussian-kernel soft targets, small PyTorch MLP trained with `KLDivLoss`, point prediction decoded as the probability-weighted mean of bin centers; swept `n_bins` ∈ {8,10,11,12,15} | R²=0.4375–0.4412 across all bin counts (not sensitive to bin count — not cherry-picked) | not submitted | See `notebooks/distributional_regression.py`. Underperforms baseline by ~0.07 R² consistently: discretizing age throws away within-bin precision that a direct regression loss (squared error) keeps natively. Not used. |
| 2026-07-21 | Claude (instructions.rtf sprint) | Lane E — Bayesian-optimized boosting: Optuna-tuned XGBoost/LightGBM/CatBoost, hyperparameters chosen on an 80% tuning-train split (internal 3-fold CV) kept disjoint from the 5-fold reporting CV on all 1212 rows | XGBoost R²=0.5398 (winner, by a wide margin); CatBoost 0.5126; LightGBM 0.5113 — all three beat baseline | not submitted | See `notebooks/boosting_bayesopt.py`. XGBoost's tuned hyperparameters are the single largest lane-level gain of the sprint (a properly tuned GBM implementation materially outperforms sklearn's default-tuned `GradientBoostingRegressor` here); folded into the final pipeline. |
| 2026-07-21 | Claude (instructions.rtf sprint) | Lane F — lightweight Bayesian NN via MC-Dropout (100→128→64→1, dropout 0.3, 50 stochastic passes at inference) | R²=0.4371 (underperforms, as expected for a small MLP on 1212 tabular rows) | not submitted | See `notebooks/bnn_experiment.py`. Not competitive on point-prediction accuracy, but the MC-Dropout uncertainty estimate is reasonably well-calibrated: pooled `corr(\|error\|, predicted std)=0.3058`. Not used for the final pipeline. |
| 2026-07-21 | Claude (instructions.rtf sprint) | **SYNTHESIS**: combined lane B (`TopCorrMixedImputer`, n_fancy=40) + lane C (`SelectKBest`, k=200) + lane E (Optuna-tuned XGBoost); also leak-free-tested `BaggingRegressor(XGBoost)`, `AdaBoostRegressor(loss="exponential")`, `StackingRegressor(XGBoost+CatBoost+GBR)→Ridge`, and a weighted committee (XGBoost 0.7 + CatBoost 0.3) as alternative combination strategies | **WINNER: R²=0.5518 ± 0.0225** (5-fold CV) — beats every ensembling wrapper tried (Bagging 0.5395, AdaBoost 0.4839 — worse than baseline, Stacking 0.5398, weighted committee 0.5423) and every individual lane, while also having the lowest fold-to-fold std of any candidate. Bootstrap (100 resamples, refit-on-resample/score-on-OOB-rows): mean R²=0.4938, 90% CI [0.4185, 0.5620] — the OOB estimate reads lower than the 5-fold CV number because each bootstrap resample's in-bag training set is only ~63% unique rows (vs ~80% for a 5-fold training split), a known bootstrap-OOB pessimism, not a contradiction; the CV point estimate (0.5518) sits just above the CI's upper end. | not submitted | See `src/advanced_pipeline.py`. Outlier removal is **still** not used to filter training data — retried leak-free in `notebooks/outlier_retry.py`, hard removal still loses at every setting tried, so `data/processed/outlier_labels.csv` remains classify-only, consistent with the 07-14 decision (not reversed). Submissions: `submissions/advanced_submission.csv` (winner) and `submissions/xgboost_tuned_baseline_preproc_submission.csv` (simpler runner-up, R²=0.5398, in case the fancier preprocessing overfits the public LB). |
| 2026-07-22 | Claude (max-R² sprint) | Data-forensics investigation: near-duplicate train/test rows (NN distance), id/row-order leakage, hidden signal in "insane magnitude" noise columns, duplicate columns (all 832-choose-2 pairs), age-correlated KMeans/GMM clusters | Found **no exploitable structure**: test↔train NN distances match the train-train reference distribution (no shared rows); corr(id,y)=-0.029, p=0.32 (no order leakage); noise columns' signal (\|corr\| 0.13-0.23) already survives `RobustScaler` unchanged; 0/346k column pairs redundant (\|corr\|>0.999); best cluster feature moves R²by only +0.0048, inside fold-to-fold noise | n/a (recon) | See `notebooks/leakage_investigation.py`. Run in response to the user citing a ~0.7533 result elsewhere — this rules out a data-structure shortcut as the explanation for that gap. |
| 2026-07-22 | Claude (max-R² sprint) | Lane 1 — feature engineering: pairwise products/ratios + row mean/std/max of top-20 age-corr columns + PCA(5-15), appended to the winning preprocessing's output, with the existing tuned `XGB_PARAMS` (no retune) | R²=0.5567 (beats 0.5518); a fresh 60-trial wider-space Optuna retune on these features scored *worse* (0.5353) — overfits the 969-row/3-fold tuning signal | not submitted | See `notebooks/feature_engineering_v2.py`. Features help; retuning around them doesn't. |
| 2026-07-22 | Claude (max-R² sprint) | Lane 2 — diverse nested stacking: 10 base learners (Ridge/ElasticNet/SVR/KNN/RandomForest/ExtraTrees/HistGB/XGBoost/LightGBM/CatBoost), proper nested leak-free OOF stacking, RidgeCV/ElasticNetCV meta-learners | ElasticNetCV meta: R²=0.5555 (+0.0037 over single tuned XGBoost, RidgeCV meta 0.5548) — real but noise-scale, at ~10x model-maintenance cost | not submitted | See `notebooks/stacking_ensemble_v2.py`. |
| 2026-07-22 | Claude (max-R² sprint) | Lane 3 — target transforms: log/sqrt/Box-Cox/Yeo-Johnson/quantile-normal via `TransformedTargetRegressor`, inverse-transformed before scoring | Yeo-Johnson best at R²=0.5554, but with 40% higher fold-to-fold std (0.0316 vs 0.0225) — not a real win; age's fairly symmetric 42-97 range leaves little for a transform to fix | not submitted | See `notebooks/target_transform_v2.py`. Not adopted. |
| 2026-07-22 | Claude (max-R² sprint) | Lane 4 — transductive/pseudo-labeling: (a) unsupervised preprocessing stats fit on train+test combined vs train-only; (b) confidence-filtered pseudo-labeling, validated via simulated 80/20 holdouts (5 random splits) rather than trusting real-test predictions blindly | (a) exactly 0.0000 delta (theoretical no-op, confirmed); (b) naive pseudo-labeling hurts (-0.0179); confidence-filtered (top 25-50%) turns positive on average (+0.002-0.003) but wins only 3/5 simulated splits — smaller than split-to-split noise | not submitted | See `notebooks/pseudo_labeling_v2.py`. Honest null result — not adopted. |
| 2026-07-22 | Claude (max-R² sprint) | Lane 5 — multi-seed bagging: average 20 XGBoost refits (seeds 0-19, last 10 also bootstrap-resampled) of the exact winning pipeline, preprocessor fit once per fold | **R²=0.5580 ± 0.0259** (+0.0062 over 0.5518) — the single largest confirmed gain of the sprint, though fold-to-fold std did not shrink as hoped | not submitted | See `notebooks/multiseed_bagging_v2.py`. **Adopted as final** (see synthesis row below). |
| 2026-07-22 | Claude (max-R² sprint) | Lane 6 — aggressive retuning: 80-trial-per-library Optuna (XGBoost/LightGBM/CatBoost) with native early stopping (up to 3000 rounds) and a wider search space, on the winning preprocessing | Best (LightGBM/CatBoost tied) R²=0.5326 — **worse** than the existing tuned XGBoost (0.5518); XGBoost itself only completed 26/80 trials under a safety-net timeout and scored 0.5137 | not submitted | See `notebooks/boosting_bayesopt_v2.py`. Confirms (again, like lane 1) that retuning doesn't beat the existing `XGB_PARAMS` here. Not adopted. |
| 2026-07-22 | Claude (max-R² sprint) | **SYNTHESIS**: combined lane 1 (feature engineering) + lane 5 (20-seed bagging) on the theory that input-representation and prediction-variance gains would stack | R²=0.5531 ± 0.0252 — **worse than lane 5 alone** (0.5580); the two gains did not compound (likely overlapping variance reduction), a genuine negative synthesis result | not submitted | See `src/max_r2_pipeline.py`'s docstring. Automated synthesis attempt separately hit a ~2hr runaway bootstrap (killed) and then a session usage limit; redone by hand afterward, which is how this combination result and the negative finding were caught. |
| 2026-07-22 | Claude (max-R² sprint) | **ADOPTED**: lane 5 alone (20-seed bagging, no feature-engineering layer) — simplest option with the best validated number | **R²=0.5580 ± 0.0259** (5-fold CV) | not submitted | See `src/max_r2_pipeline.py`. User asked whether ~0.7533 was reachable (cited from elsewhere) — after a dedicated leak/structure investigation plus 6 modeling lanes plus this synthesis, the honest ceiling for this feature set with these techniques is **R²≈0.55-0.58**, not 0.75; no exploitable data-structure shortcut was found to explain the gap. Submission: `submissions/max_r2_submission.csv`. |

## Current best

- Approach: 20-seed bagged, Optuna-tuned XGBoost on `TopCorrMixedImputer`(n_fancy=40) + `SelectKBest`(f_regression, k=200) features, no outlier removal, no feature-engineering layer (`src/max_r2_pipeline.py`) — supersedes `src/advanced_pipeline.py` below, which remains in the repo as the simpler single-seed reference.
- CV score: R²=0.5580 ± 0.0259 (5-fold)
- Public LB score: not yet submitted
- **Honest ceiling assessment**: after a dedicated data-forensics investigation (no exploitable leakage/structure found) plus six parallel modeling lanes (feature engineering, diverse stacking, target transforms, pseudo-labeling, multi-seed bagging, aggressive retuning) plus a synthesis attempt, results cluster in R²≈0.53-0.58 — a ~0.7533 result cited from elsewhere was not reproducible with any technique tried here and looks like it would require a fundamentally different approach or data source, not a refinement of this pipeline.
- Previous best (kept as reference): single-seed Optuna-tuned XGBoost on the same features (`src/advanced_pipeline.py`), R²=0.5518 ± 0.0225; 90% bootstrap CI (100 resamples, OOB scoring) = [0.4185, 0.5620]
- Runner-up: plain Optuna-tuned XGBoost on the original baseline preprocessing (median impute, k=100), R²=0.5398, submitted as `submissions/xgboost_tuned_baseline_preproc_submission.csv` as a hedge
- Original baseline (kept as reference/fallback): GradientBoostingRegressor on SelectKBest-100 features, median impute, no outlier removal (`src/baseline.py`), R²=0.5065 (5-fold)
