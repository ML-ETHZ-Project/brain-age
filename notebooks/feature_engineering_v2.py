"""Lane 1: feature engineering on top of the winning preprocessor + wider-space
Optuna-tuned XGBoost (notebooks/feature_engineering_v2.py).

Current best to beat (src/advanced_pipeline.py): TopCorrMixedImputer(n_fancy=40)
-> RobustScaler -> VarianceThreshold(1e-8) -> SelectKBest(f_regression, k=200)
-> XGBoost(tuned). 5-fold CV R^2 = 0.5518 +/- 0.0225 (KFold(5, shuffle=True,
random_state=42)). TopCorrMixedImputer and XGB_PARAMS are IMPORTED from
src/advanced_pipeline.py (not re-derived) per project convention.

This lane builds engineered features on top of that winning imputer+scaler
stack, then re-tunes XGBoost (with a wider hyperparameter search space) and
SelectKBest's k and the engineered PCA size, all via the same leak-free
two-stage Optuna scheme as notebooks/boosting_bayesopt.py:

  Pipeline: TopCorrMixedImputer(n_fancy=40) -> RobustScaler -> VarianceThreshold
            -> FeatureEngineer -> SelectKBest(k) -> XGBoost(tuned)

  FeatureEngineer (fit on that split's training rows' X, y only -- safe to
  drop into a Pipeline scored by cross_val_score, and refit per CV fold):
    (a) pairwise products AND ratios of the top ~20 |corr(x_j, y)| columns
        (ranked on training rows only), computed on the already-imputed,
        RobustScaler'd, variance-thresholded matrix (not the pre-scaling raw
        values) for two reasons: (i) semantic consistency with the rest of
        the winning pipeline, which the task frames as "on top of the
        TopCorrMixedImputer+RobustScaler output", and (ii) it lets every
        engineered column live on comparable robust-scaled units before
        SelectKBest scores them. Ratios guard against near-zero denominators
        (RobustScaler centers each column at its median, so exact/near-zero
        values do occur) via an eps=0.1 floor on |denominator| and a
        clip to [-50, 50] so no inf/extreme values reach SelectKBest/XGBoost.
    (b) PCA (n_components, tuned in [5, 15]) fit on the same (scaled,
        variance-thresholded) matrix.
    (c) per-row summary stats (mean/std/max) across the same top ~20 columns.
  All new columns are appended to (not a replacement of) the original
  variance-thresholded columns; SelectKBest then chooses among the full
  augmented set.

  Stage 1 (tuning): train_test_split(test_size=0.2, random_state=42) sets
  aside a 20% tuning-holdout untouched by any objective/fit in this stage.
  Within the 80% tuning-train, an Optuna TPE study (60 trials, seed=42) tunes:
  SelectKBest's k in {100, 200, 300, 400}, PCA's n_components in [5, 15], and
  XGBoost's n_estimators/learning_rate/max_depth/min_child_weight/subsample/
  colsample_bytree (same ranges as notebooks/boosting_bayesopt.py) PLUS the
  wider-space additions reg_alpha, reg_lambda, gamma. Each trial is scored by
  an internal 3-fold KFold(random_state=42) CV *within the tuning-train only*.

  SPEED OPTIMIZATION (mathematically identical to, not a relaxation of, the
  leak-free discipline): the imputer/scaler/var-threshold/ratio-product-stats
  part of the pipeline does not depend on any Optuna-suggested hyperparameter,
  so instead of refitting it from scratch on every one of the 60 trials x 3
  inner folds (180 redundant, identical fits of an IterativeImputer), it is
  fit ONCE per inner fold (on that fold's training rows only, exactly as
  Pipeline+cross_val_score would) and the transformed arrays are cached and
  reused across trials. Only the parts that actually vary per trial (PCA
  size, SelectKBest's k, XGBoost's hyperparameters) are refit every trial.
  This changes nothing about what gets fit on what rows -- it only avoids
  recomputing the same deterministic fit 60 times -- so it introduces no
  leakage; it is what makes ~60 trials tractable in the time budget.

  Stage 2 (reporting): the single best trial's hyperparameters are plugged
  into a plain sklearn Pipeline (all steps, including TopCorrMixedImputer and
  FeatureEngineer) scored via cross_val_score over the standard 5-fold
  KFold(shuffle=True, random_state=42) on the FULL 1212-row training set --
  the number directly comparable to 0.5518, and leak-free the same way
  src/advanced_pipeline.py's own reporting is (Pipeline refits every step,
  including the imputer and the correlation-ranking-based feature engineer,
  on each fold's training rows only).
"""
import itertools
import os
import sys
import time
import warnings

import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from src.advanced_pipeline import TopCorrMixedImputer, N_FANCY  # noqa: E402  (winner's imputer, imported not re-derived)

RANDOM_STATE = 42
N_SPLITS = 5
N_TUNE_SPLITS = 3
N_TRIALS = 60
N_TOP_ENG = 20  # top age-correlated cols used for ratios/products/summary stats
K_CHOICES = [100, 200, 300, 400]
CURRENT_BEST_R2 = 0.5518
CURRENT_BEST_STD = 0.0225

DATA_DIR = os.path.join(REPO_ROOT, "data", "raw")


def load_data():
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"))
    feat_cols = [c for c in X_train.columns if c != "id"]
    return X_train[feat_cols].values, y_train["y"].values


def _abs_corr(x_col, y):
    """abs(Pearson corr) between a column and y; 0 for degenerate (constant) cases."""
    if np.std(x_col) == 0 or np.std(y) == 0:
        return 0.0
    c = np.corrcoef(x_col, y)[0, 1]
    return 0.0 if np.isnan(c) else abs(c)


class RatioProductStats(BaseEstimator, TransformerMixin):
    """(a) + (c): pairwise products & ratios, and per-row mean/std/max, of the
    top `n_top` |corr(x_j, y)| columns (ranked on this fit's training rows
    only). Appends new columns; does not drop any original column."""

    def __init__(self, n_top=N_TOP_ENG):
        self.n_top = n_top

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_cols = X.shape[1]
        n_top = min(self.n_top, n_cols)
        corrs = np.array([_abs_corr(X[:, j], y) for j in range(n_cols)])
        self.top_idx_ = np.argsort(-corrs)[:n_top]
        self.pairs_ = list(itertools.combinations(range(n_top), 2))
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        top = X[:, self.top_idx_]
        n = X.shape[0]
        n_pairs = len(self.pairs_)
        prod = np.empty((n, n_pairs))
        ratio = np.empty((n, n_pairs))
        eps = 0.1
        for k, (i, j) in enumerate(self.pairs_):
            a, b = top[:, i], top[:, j]
            prod[:, k] = a * b
            b_sign = np.where(b < 0, -1.0, 1.0)  # b==0 treated as positive floor
            b_safe = np.where(np.abs(b) < eps, b_sign * eps, b)
            ratio[:, k] = np.clip(a / b_safe, -50.0, 50.0)
        stats = np.column_stack([top.mean(axis=1), top.std(axis=1), top.max(axis=1)])
        out = np.hstack([X, prod, ratio, stats])
        return np.nan_to_num(out, nan=0.0, posinf=50.0, neginf=-50.0)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Full lane-1 engineered-feature step for a plain sklearn Pipeline: (a)+(c)
    via RatioProductStats, plus (b) PCA (n_components, tuned), all fit on this
    fit's training rows only."""

    def __init__(self, n_top=N_TOP_ENG, n_components=10):
        self.n_top = n_top
        self.n_components = n_components

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.rps_ = RatioProductStats(n_top=self.n_top)
        self.rps_.fit(X, y)
        n_comp = max(1, min(self.n_components, X.shape[0] - 1, X.shape[1]))
        self.pca_ = PCA(n_components=n_comp, random_state=RANDOM_STATE)
        self.pca_.fit(X)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        base = self.rps_.transform(X)
        pca_feats = self.pca_.transform(X)
        return np.hstack([base, pca_feats])


def build_prefix_pipeline():
    """impute -> scale -> var_thresh (everything upstream of the engineered
    features that does NOT depend on any Optuna-tuned hyperparameter)."""
    return Pipeline([
        ("impute", TopCorrMixedImputer(n_fancy=N_FANCY)),
        ("scale", RobustScaler()),
        ("var_thresh", VarianceThreshold(threshold=1e-8)),
    ])


def build_full_pipeline(k, n_components, xgb_params):
    return Pipeline([
        ("impute", TopCorrMixedImputer(n_fancy=N_FANCY)),
        ("scale", RobustScaler()),
        ("var_thresh", VarianceThreshold(threshold=1e-8)),
        ("engineer", FeatureEngineer(n_top=N_TOP_ENG, n_components=n_components)),
        ("select", SelectKBest(f_regression, k=k)),
        ("model", XGBRegressor(**xgb_params)),
    ])


def build_cached_folds(X, y, splits):
    """Fit the trial-invariant prefix (impute/scale/var_thresh) + the (a)/(c)
    ratio-product-stats step ONCE per fold (training rows only), cache the
    transformed train/val arrays. See module docstring's "SPEED OPTIMIZATION"
    note: this is exactly what Pipeline+cross_val_score would produce per
    trial, just not redundantly recomputed 60 times over."""
    cached = []
    for tr_idx, va_idx in splits:
        Xtr_raw, Xva_raw = X[tr_idx], X[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]

        prefix = build_prefix_pipeline()
        Xtr_v = prefix.fit_transform(Xtr_raw, ytr)
        Xva_v = prefix.transform(Xva_raw)

        rps = RatioProductStats(n_top=N_TOP_ENG)
        Xtr_r = rps.fit_transform(Xtr_v, ytr)
        Xva_r = rps.transform(Xva_v)

        # Xtr_v/Xva_v (pre-ratio/product, post var_thresh) also cached so PCA
        # (which is tuned per trial) can be fit on the *same* matrix that
        # FeatureEngineer.fit would see -- consistent with build_full_pipeline.
        cached.append((Xtr_v, Xtr_r, ytr, Xva_v, Xva_r, yva))
    return cached


def make_objective(cached_folds):
    def objective(trial):
        k = trial.suggest_categorical("k", K_CHOICES)
        n_components = trial.suggest_int("n_components", 5, 15)
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 600),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 2, 8),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            gamma=trial.suggest_float("gamma", 1e-8, 5.0, log=True),
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )

        scores = []
        for Xtr_v, Xtr_r, ytr, Xva_v, Xva_r, yva in cached_folds:
            n_comp = max(1, min(n_components, Xtr_v.shape[0] - 1, Xtr_v.shape[1]))
            pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
            pca_tr = pca.fit_transform(Xtr_v)
            pca_va = pca.transform(Xva_v)
            Xtr_full = np.hstack([Xtr_r, pca_tr])
            Xva_full = np.hstack([Xva_r, pca_va])

            k_eff = min(k, Xtr_full.shape[1])
            skb = SelectKBest(f_regression, k=k_eff)
            Xtr_sel = skb.fit_transform(Xtr_full, ytr)
            Xva_sel = skb.transform(Xva_full)

            model = XGBRegressor(**params)
            model.fit(Xtr_sel, ytr)
            pred = model.predict(Xva_sel)
            scores.append(r2_score(yva, pred))
        return float(np.mean(scores))

    return objective


def evaluate_reporting_stage(name, k, n_components, xgb_params, X, y):
    pipe = build_full_pipeline(k, n_components, xgb_params)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipe, X, y, cv=kf, scoring="r2")
    print(f"  {name}: 5-fold CV R^2 = {scores.mean():.4f} +/- {scores.std():.4f}  "
          f"(folds: {np.round(scores, 4)})")
    return scores


def main():
    t_start = time.time()
    X, y = load_data()
    print(f"Loaded X {X.shape}, y {y.shape}")
    print(f"Current best to beat (src/advanced_pipeline.py): R^2 = {CURRENT_BEST_R2} +/- {CURRENT_BEST_STD}")

    # ------------------------------------------------------------------
    # Quick diagnostic (not the headline number): engineered features with
    # the ALREADY-tuned XGB_PARAMS from src/advanced_pipeline.py (no
    # retuning), default k=200/n_components=10, on the standard 5-fold
    # reporting CV. Answers "do engineered features help even before
    # retuning the model to them?"
    # ------------------------------------------------------------------
    from src.advanced_pipeline import XGB_PARAMS
    print("\n--- Diagnostic: engineered features + OLD tuned XGB_PARAMS (k=200, n_components=10) ---")
    diag_scores = evaluate_reporting_stage(
        "Engineered + old XGB_PARAMS", 200, 10, XGB_PARAMS, X, y
    )

    # ------------------------------------------------------------------
    # Stage 1: Optuna tuning on an 80% tuning-train split, internal 3-fold
    # CV, 20% tuning-holdout set aside and never touched (disjoint from the
    # stage-2 reporting CV below).
    # ------------------------------------------------------------------
    X_tune_train, X_tune_holdout, y_tune_train, y_tune_holdout = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"\n--- Stage 1: Optuna tuning (80% tuning-train = {X_tune_train.shape[0]} rows, "
          f"20% tuning-holdout = {X_tune_holdout.shape[0]} rows set aside/unused) ---")

    inner_kf = KFold(n_splits=N_TUNE_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    inner_splits = list(inner_kf.split(X_tune_train))

    t0 = time.time()
    cached_folds = build_cached_folds(X_tune_train, y_tune_train, inner_splits)
    print(f"  Cached {len(cached_folds)} inner-fold prefix fits (impute/scale/var_thresh/"
          f"ratio-product-stats) in {time.time() - t0:.1f}s -- reused across all {N_TRIALS} trials")

    def progress_cb(study, trial):
        if (trial.number + 1) % 10 == 0 or trial.number == 0:
            print(f"    trial {trial.number + 1}/{N_TRIALS} done, best so far R^2={study.best_value:.4f}")

    t0 = time.time()
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(make_objective(cached_folds), n_trials=N_TRIALS, callbacks=[progress_cb])
    print(f"  Optuna search done: {N_TRIALS} trials in {time.time() - t0:.1f}s, "
          f"best inner-3fold R^2 = {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")

    best = study.best_params
    best_k = best["k"]
    best_n_components = best["n_components"]
    best_xgb_params = dict(
        n_estimators=best["n_estimators"],
        learning_rate=best["learning_rate"],
        max_depth=best["max_depth"],
        min_child_weight=best["min_child_weight"],
        subsample=best["subsample"],
        colsample_bytree=best["colsample_bytree"],
        reg_alpha=best["reg_alpha"],
        reg_lambda=best["reg_lambda"],
        gamma=best["gamma"],
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )

    # ------------------------------------------------------------------
    # Stage 2: reporting CV, 5-fold KFold on the FULL 1212-row training set,
    # hyperparameters FIXED from stage 1. This is the number comparable to
    # 0.5518.
    # ------------------------------------------------------------------
    print("\n--- Stage 2: reporting CV (5-fold KFold on FULL 1212-row training set, "
          "leak-free Pipeline+cross_val_score, hyperparams FIXED from stage 1) ---")
    final_scores = evaluate_reporting_stage(
        "Feature-engineered + Optuna-tuned XGBoost (wider space)",
        best_k, best_n_components, best_xgb_params, X, y,
    )

    print("\n=== Summary ===")
    print(f"  Current best (src/advanced_pipeline.py):                      R^2 = {CURRENT_BEST_R2:.4f} "
          f"+/- {CURRENT_BEST_STD:.4f}")
    print(f"  Diagnostic (engineered feats + OLD XGB_PARAMS, k=200, pca=10): R^2 = {diag_scores.mean():.4f} "
          f"+/- {diag_scores.std():.4f}")
    print(f"  LANE 1 RESULT (engineered feats + wider-space Optuna XGBoost): R^2 = {final_scores.mean():.4f} "
          f"+/- {final_scores.std():.4f}")
    print(f"  Best hyperparameters (k={best_k}, n_components={best_n_components}): {best_xgb_params}")
    diff = final_scores.mean() - CURRENT_BEST_R2
    if diff > 0:
        print(f"  -> BEATS current best by {diff:+.4f}")
    else:
        print(f"  -> does NOT beat current best ({diff:+.4f})")
    print(f"\nTotal runtime: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
