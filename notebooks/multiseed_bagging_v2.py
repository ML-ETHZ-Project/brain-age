"""LANE 5 -- Multi-seed bagging of the current-best pipeline.

Current best (src/advanced_pipeline.py): TopCorrMixedImputer(n_fancy=40) ->
RobustScaler -> VarianceThreshold(1e-8) -> SelectKBest(f_regression, k=200) ->
XGBRegressor(**XGB_PARAMS), 5-fold CV R^2 = 0.5518 +/- 0.0225
(KFold(5, shuffle=True, random_state=42)).

This script asks: does averaging predictions from 20 differently-seeded
copies of the *exact same tuned pipeline* (varying only XGBRegressor's
random_state 0..19, plus bootstrap-resampling the fold-training rows for a
subset of the 20, for a bagging flavor) reduce variance and/or lift the mean,
relative to the single-seed (random_state=42) baseline?

Leak-free design: for each of the 5 outer CV folds (same KFold(5,
shuffle=True, random_state=42) as the reported baseline), the *preprocessor*
(TopCorrMixedImputer -> RobustScaler -> VarianceThreshold -> SelectKBest) is
fit exactly ONCE per fold, on that fold's training rows only -- it does not
depend on the XGBoost seed, so refitting it 20x per fold would be wasted
compute, not a leak-free requirement. Each of the 20 seeded (optionally
bootstrap-resampled) XGBRegressors is then fit on that same preprocessed
fold-training data and scored/predicts on the fold's held-out validation
rows, which none of the 20 models nor the preprocessor ever saw. The 20
models' validation-row predictions are averaged (simple mean) before scoring
R^2 for that fold -- mirroring exactly how the single-seed baseline is
scored, so the comparison is apples-to-apples.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from advanced_pipeline import (  # noqa: E402
    N_SPLITS,
    RANDOM_STATE,
    XGB_PARAMS,
    build_preprocessor,
    load_data,
)

N_SEEDS = 20
N_BOOTSTRAP_SEEDS = 10  # last N_BOOTSTRAP_SEEDS of the 20 seeds also bootstrap-resample fold-train rows


def fit_predict_one_seed(X_tr, y_tr, X_val, seed, bootstrap):
    """Fit one XGBRegressor (tuned params, given seed) and return its
    predictions on X_val. If bootstrap, first resample (X_tr, y_tr) with
    replacement (seed-determined, in-bag only -- never touches X_val)."""
    if bootstrap:
        rng = np.random.RandomState(RANDOM_STATE + seed)
        in_bag = rng.randint(0, len(y_tr), size=len(y_tr))
        X_fit, y_fit = X_tr[in_bag], y_tr[in_bag]
    else:
        X_fit, y_fit = X_tr, y_tr

    params = dict(XGB_PARAMS)
    params["random_state"] = seed
    model = XGBRegressor(**params)
    model.fit(X_fit, y_fit)
    return model.predict(X_val)


def main():
    t_start = time.time()
    X_train, y_train, train_ids, X_test, test_ids = load_data()
    print(f"Loaded: X_train {X_train.shape}, X_test {X_test.shape}")
    print(f"Ensemble: {N_SEEDS} seeds (0..{N_SEEDS - 1}), "
          f"last {N_BOOTSTRAP_SEEDS} of which also bootstrap-resample fold-train rows")

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    single_seed_scores = []
    ensemble_scores = []

    for fold_i, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        t_fold = time.time()
        X_tr_raw, X_val_raw = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        # Preprocessor fit ONCE per fold, on this fold's training rows only.
        prep = build_preprocessor()
        X_tr = prep.fit_transform(X_tr_raw, y_tr)
        X_val = prep.transform(X_val_raw)

        # Single-seed baseline (random_state=42, the exact winning config),
        # scored on this same fold split for an apples-to-apples comparison.
        base_params = dict(XGB_PARAMS)  # already random_state=42
        base_model = XGBRegressor(**base_params)
        base_model.fit(X_tr, y_tr)
        base_pred = base_model.predict(X_val)
        base_r2 = r2_score(y_val, base_pred)
        single_seed_scores.append(base_r2)

        # 20-seed ensemble.
        preds = []
        for seed in range(N_SEEDS):
            bootstrap = seed >= (N_SEEDS - N_BOOTSTRAP_SEEDS)
            preds.append(fit_predict_one_seed(X_tr, y_tr, X_val, seed, bootstrap))
        ensemble_pred = np.mean(preds, axis=0)
        ensemble_r2 = r2_score(y_val, ensemble_pred)
        ensemble_scores.append(ensemble_r2)

        print(f"  Fold {fold_i}: single-seed R^2 = {base_r2:.4f}  |  "
              f"20-seed ensemble R^2 = {ensemble_r2:.4f}  ({time.time() - t_fold:.1f}s)")

    single_seed_scores = np.array(single_seed_scores)
    ensemble_scores = np.array(ensemble_scores)

    print("\n=== Summary ===")
    print(f"  Single-seed (random_state=42) baseline, this run : "
          f"R^2 = {single_seed_scores.mean():.4f} +/- {single_seed_scores.std():.4f}  "
          f"(folds: {np.round(single_seed_scores, 4)})")
    print(f"  Reference (src/advanced_pipeline.py, full run)   : R^2 = 0.5518 +/- 0.0225")
    print(f"  20-seed bagged ensemble                          : "
          f"R^2 = {ensemble_scores.mean():.4f} +/- {ensemble_scores.std():.4f}  "
          f"(folds: {np.round(ensemble_scores, 4)})")
    print(f"  Delta (ensemble mean - single-seed mean)         : "
          f"{ensemble_scores.mean() - single_seed_scores.mean():+.4f}")
    print(f"  Delta (ensemble std - single-seed std)           : "
          f"{ensemble_scores.std() - single_seed_scores.std():+.4f}  "
          f"(negative = ensemble is more stable fold-to-fold)")
    print(f"\nTotal runtime: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
