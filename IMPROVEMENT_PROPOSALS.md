# Improvement proposals (2026-07-17)

Status snapshot before these proposals: `src/baseline.py` is a `StackingRegressor`
(GradientBoosting + LightGBM + CatBoost + Ridge(α=10) + KNN(k=15, distance) → Ridge
meta-learner) on `CorrelationPrunedKBest(n=100, thresh=0.9)` features, median impute,
no outlier removal. CV R²≈0.539 (5-fold). See `README.md`'s experiment log for the
full history of what's already been tried and rejected (KNN/iterative imputers,
mutual-info feature scoring, n=25/250 feature counts, outlier removal, sparse/denoising
autoencoders, plain PCA, from-scratch NN regressors, XGBoost family reasoning).

Every model/hyperparameter choice logged so far was picked from a small (2-5 config)
grid, not systematically tuned — that's the biggest lever left untouched. Proposals
below, ranked by expected value:

## 1. Hyperparameter tuning of CatBoost / LightGBM / GBR (highest priority)

CatBoost's `depth=6, l2_leaf_reg=3` beat GBR from just 3 configs tried in
`notebooks/catboost_experiment.py` — there's very likely more headroom in
`learning_rate` / `n_estimators` / `subsample` that a `RandomizedSearchCV` nested
inside each outer CV fold would find. Likely the single largest R² gain still
available, but nested tuning is compute-heavy (search × 5 outer folds ×
`StackingRegressor`'s internal 5-fold OOF cross-fitting).

## 2. Per-base-learner feature width for KNN (cheap, plausible win)

KNN suffers from the curse of dimensionality more than tree models. Giving it a
narrower feature set (e.g. top 20-30 by correlation) while keeping the tree learners
at 100 could help KNN's contribution to the stack without touching what already works
for GBR/LightGBM/CatBoost. Targeted, unlike the earlier rejected blanket-PCA
experiment which replaced feature selection everywhere.

## 3. Meta-learner alternatives (cheap)

Try tuning the meta-learner Ridge's α, or swapping it for ElasticNet. The current
α=1.0 has never itself been ablated — it was inherited from Philippe's original
stacking setup.

## 4. SVR as a 6th, more different base learner (moderate effort, uncertain payoff)

Kernel-based, structurally unlike anything currently in the stack — could add
diversity the way CatBoost/LightGBM did. Small-n SVR tuning (C, gamma) needs its own
nested search, so this overlaps with proposal #1's compute cost.

## 5. Domain-informed ratio/interaction features (higher effort, unclear payoff)

FreeSurfer volumes have known anatomical ratios (e.g. white/gray matter, hemisphere
asymmetry) that raw per-column correlation selection can't construct. Blocked on not
knowing which `x*` column maps to which anatomical structure (columns are
anonymized) — would need a mapping from the organizers or reverse-engineering from
typical FreeSurfer output ordering before this is actionable.

## 6. Stacking `cv` fold count (cheap, low expected value)

Bump `StackingRegressor`'s internal OOF cv from 5 to 10 for less noisy meta-features
at this sample size (1212 rows). Cheap to test, probably marginal.

## Recommendation

Start with **#1** (CatBoost/LightGBM tuning) since it's the highest-value lever and
directly extends the pattern that already worked twice (LightGBM, then CatBoost,
each added as new base learners). #2/#3/#6 are cheap enough to try in parallel or as
quick follow-ups regardless of #1's outcome.
