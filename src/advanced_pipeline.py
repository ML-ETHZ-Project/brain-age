"""Advanced pipeline for the Brain Age Prediction Kaggle competition.

Synthesizes six parallel experiment lanes run on top of src/baseline.py's
reference (median-impute -> RobustScaler -> VarianceThreshold -> SelectKBest
f_regression k=100 -> GradientBoostingRegressor, R^2=0.5065, 5-fold CV):

  Lane A (notebooks/outlier_retry.py) -- retried outlier handling leak-free
    (detector fit inside each CV fold, on that fold's training rows only).
    Hard removal (EllipticEnvelope, OneClassSVM) again lost badly (0.47-0.49),
    confirming the project's earlier finding. Soft downweighting (flag via
    IsolationForest, sample_weight=0.3 on flagged rows instead of dropping)
    gave a small, noisy bump (0.5146) -- interesting but superseded below.
  Lane B (notebooks/impute_refine.py) -- TopCorrMixedImputer: rank columns by
    |corr(x_j, y)| on training rows only, apply IterativeImputer(BayesianRidge)
    to the top n_fancy=40 age-correlated columns and SimpleImputer(median) to
    the rest. Modest, stable win (0.5121, lower fold-to-fold std than median).
  Lane C (notebooks/feature_selection_refine.py) -- swept selectors/k with
    everything else fixed. SelectKBest(f_regression, k=200) (double the
    baseline's k=100) was the only configuration to beat baseline (0.5119).
  Lane D (notebooks/distributional_regression.py) -- binned age + KL-loss MLP.
    Underperformed badly (0.4386) -- discretization throws away precision a
    direct regression loss keeps. Not used.
  Lane E (notebooks/boosting_bayesopt.py) -- Optuna-tuned XGBoost/LightGBM/
    CatBoost, hyperparameters chosen on an 80% tuning-train split (internal
    3-fold CV) disjoint from the 5-fold reporting CV on all 1212 rows.
    XGBoost was the standout: R^2=0.5398 on the *baseline* (median, k=100)
    preprocessing -- by far the largest single-lane gain, because a properly
    tuned GBM implementation materially outperforms sklearn's default-tuned
    GradientBoostingRegressor here.
  Lane F (notebooks/bnn_experiment.py) -- MC-Dropout MLP. Underperformed
    (0.4371) as expected for a small MLP on 1212 tabular rows; its calibrated
    uncertainty estimates were the interesting side-result, not used here.

SYNTHESIS: lanes B (mixed imputer) and C (k=200) touch different pipeline
stages than lane E (model choice) and stack cleanly. A direct leak-free
comparison (manual per-fold loop, same pattern as
notebooks/outlier_comparison.py) of combining them, plus an ensembling layer
(BaggingRegressor, AdaBoostRegressor(loss="exponential"), StackingRegressor,
and a weighted committee, all built from the strongest base learners) found:

    GBR baseline (median, k=100)                 : R^2 = 0.5061
    XGBoost tuned (median, k=100)  [lane E alone] : R^2 = 0.5398
    AdaBoost(loss=exponential)                    : R^2 = 0.4839  (worse than baseline)
    BaggingRegressor(XGBoost tuned)                : R^2 = 0.5395  (no better than plain XGBoost)
    StackingRegressor(XGB+CatBoost+GBR)->Ridge     : R^2 = 0.5398  (no better than plain XGBoost)
    Weighted committee XGB(0.7)+CatBoost(0.3)      : R^2 = 0.5423  (small bump)
    XGBoost tuned + combined preproc (B+C)  [WINNER]: R^2 = 0.5518 +/- 0.0225

XGBoost tuned + the lane B/C combined preprocessing (TopCorrMixedImputer
n_fancy=40 -> RobustScaler -> VarianceThreshold -> SelectKBest k=200) won
outright: it beat every ensembling wrapper tried (bagging/boosting/stacking/
committee) on top of the same or plainer preprocessing, AND has the lowest
fold-to-fold std of any candidate (0.0225 vs 0.03-0.04 for the others) --
i.e. it's not just the best point estimate, it's also the most stable one.
This is a case where feeding a stronger, well-preprocessed feature set to a
single well-tuned model beat every attempt to compensate for a weaker feature
set via ensembling on top of it.

Outlier removal (subtask 1) is STILL deliberately NOT used to filter the
regressor's training data, per notebooks/outlier_comparison.py and
notebooks/outlier_retry.py: every hard-removal variant tried across both
scripts (IsolationForest, LOF, EllipticEnvelope, OneClassSVM, at multiple
contaminations) loses to no-removal. This does not reverse project history --
we still emit the required outlier classification as a standalone artifact.
"""
import os
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor

RANDOM_STATE = 42
N_SPLITS = 5
K_BEST = 200
N_FANCY = 40
OUTLIER_CONTAMINATION = 0.05
N_BOOTSTRAP = 100

# Lane E's Optuna-found XGBoost hyperparameters (notebooks/boosting_bayesopt.py),
# chosen on an 80% tuning-train split disjoint from the 5-fold CV reported here.
XGB_PARAMS = dict(
    n_estimators=481,
    learning_rate=0.034029991074560055,
    max_depth=7,
    min_child_weight=3,
    subsample=0.6210342419383295,
    colsample_bytree=0.790030811777265,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=0,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data", "raw")
OUTPUT_DIR = os.path.join(REPO_ROOT, "submissions")
PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")


def load_data():
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"))
    X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))

    feat_cols = [c for c in X_train.columns if c != "id"]
    return (
        X_train[feat_cols].values,
        y_train["y"].values,
        X_train["id"].values,
        X_test[feat_cols].values,
        X_test["id"].values,
    )


def _nan_safe_abs_corr(x_col, y):
    """abs(Pearson corr) between a single column and y, using only rows
    where x_col is observed. Returns 0 for degenerate cases instead of NaN."""
    mask = ~np.isnan(x_col)
    if mask.sum() < 2:
        return 0.0
    xv, yv = x_col[mask], y[mask]
    if np.std(xv) == 0 or np.std(yv) == 0:
        return 0.0
    c = np.corrcoef(xv, yv)[0, 1]
    return 0.0 if np.isnan(c) else abs(c)


class TopCorrMixedImputer(BaseEstimator, TransformerMixin):
    """Lane B winner (notebooks/impute_refine.py): IterativeImputer(BayesianRidge)
    on the n_fancy columns most |corr| with y (ranked on this fit's own X, y
    only -- never on validation/test rows), SimpleImputer(median) on the rest.
    Sklearn-compatible (fit/transform), so it plugs directly into a Pipeline
    and is refit leak-free by cross_val_score on every CV fold.
    """

    def __init__(self, n_fancy=N_FANCY, max_iter=5, n_nearest_features=30, random_state=RANDOM_STATE):
        self.n_fancy = n_fancy
        self.max_iter = max_iter
        self.n_nearest_features = n_nearest_features
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_cols = X.shape[1]
        abs_corrs = np.array([_nan_safe_abs_corr(X[:, j], y) for j in range(n_cols)])

        n_fancy = min(self.n_fancy, n_cols)
        order = np.argsort(-abs_corrs)
        self.fancy_idx_ = np.sort(order[:n_fancy])
        self.other_idx_ = np.sort(order[n_fancy:])

        self.fancy_imputer_ = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=self.max_iter,
            n_nearest_features=min(self.n_nearest_features, max(1, n_fancy - 1)),
            initial_strategy="median",
            random_state=self.random_state,
        )
        if len(self.fancy_idx_) > 0:
            self.fancy_imputer_.fit(X[:, self.fancy_idx_])

        self.other_imputer_ = SimpleImputer(strategy="median")
        if len(self.other_idx_) > 0:
            self.other_imputer_.fit(X[:, self.other_idx_])

        self.n_cols_ = n_cols
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        out = np.empty((X.shape[0], self.n_cols_), dtype=float)
        if len(self.fancy_idx_) > 0:
            out[:, self.fancy_idx_] = self.fancy_imputer_.transform(X[:, self.fancy_idx_])
        if len(self.other_idx_) > 0:
            out[:, self.other_idx_] = self.other_imputer_.transform(X[:, self.other_idx_])
        return out


def build_preprocessor():
    """Lane B (mixed imputer) + lane C (k=200) combined."""
    return Pipeline([
        ("impute", TopCorrMixedImputer(n_fancy=N_FANCY)),
        ("scale", RobustScaler()),
        ("var_thresh", VarianceThreshold(threshold=1e-8)),
        ("select", SelectKBest(f_regression, k=K_BEST)),
    ])


def build_full_pipeline():
    return Pipeline([
        ("prep", build_preprocessor()),
        ("model", XGBRegressor(**XGB_PARAMS)),
    ])


def classify_outliers(X_proc, train_ids, contamination=OUTLIER_CONTAMINATION):
    """Subtask 1 deliverable: classify each training row as outlier/inlier.

    Fit on the full preprocessed training set (a reporting artifact, not
    something used to filter data fed to the regressor -- see module
    docstring: hard removal was retried in notebooks/outlier_retry.py and
    still loses to no-removal at every contamination level tried).
    """
    iso = IsolationForest(contamination=contamination, random_state=RANDOM_STATE)
    is_outlier = iso.fit_predict(X_proc) == -1
    print(f"  Outlier classification: flagged {is_outlier.sum()} / {len(train_ids)} training rows as outliers")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_path = os.path.join(PROCESSED_DIR, "outlier_labels.csv")
    pd.DataFrame({"id": train_ids, "is_outlier": is_outlier.astype(int)}).to_csv(out_path, index=False)
    print(f"  Wrote {out_path}")
    return is_outlier


def bootstrap_ci(X, y, n_bootstrap=N_BOOTSTRAP):
    """Bootstrap uncertainty quantification on the final chosen pipeline.

    For each of n_bootstrap resamples (with replacement, deterministic seed
    RANDOM_STATE + i so every resample is reproducible), refit the *entire*
    pipeline (imputer/scaler/selector/model, all leak-free -- fit only on the
    in-bag resampled rows) and score R^2 on that resample's out-of-bag rows
    (rows never drawn into the resample -- never seen during fitting). This
    is the "refit on a resample, score on held-out rows" variant the task
    describes as simpler and acceptable, chosen over "refit + re-run full
    5-fold CV per resample" purely for runtime (5x the compute for a
    laptop-scale bootstrap). n_bootstrap=100 (rather than the ~200 upper end
    suggested) to keep this laptop run bounded to a few minutes; each
    resample refits IterativeImputer+XGBoost from scratch (~2.5s), so 100
    resamples is already ~4 minutes.
    """
    n = len(y)
    scores = []
    t0 = time.time()
    for i in range(n_bootstrap):
        rng = np.random.RandomState(RANDOM_STATE + i)
        in_bag = rng.randint(0, n, size=n)
        oob_mask = np.ones(n, dtype=bool)
        oob_mask[np.unique(in_bag)] = False
        oob_idx = np.where(oob_mask)[0]
        if len(oob_idx) < 10:  # degenerate resample, skip (essentially never happens at n=1212)
            continue

        pipe = build_full_pipeline()
        pipe.fit(X[in_bag], y[in_bag])
        pred = pipe.predict(X[oob_idx])
        scores.append(r2_score(y[oob_idx], pred))

    scores = np.array(scores)
    elapsed = time.time() - t0
    lo, hi = np.percentile(scores, [5, 95])
    print(f"  Bootstrap: {len(scores)} resamples (target {n_bootstrap}), {elapsed:.1f}s")
    print(f"  Bootstrap OOB R^2: mean = {scores.mean():.4f}, std = {scores.std():.4f}")
    print(f"  90% CI (5th-95th percentile): [{lo:.4f}, {hi:.4f}]")
    return scores, (lo, hi)


def main():
    t_start = time.time()
    X_train, y_train, train_ids, X_test, test_ids = load_data()
    print(f"Loaded: X_train {X_train.shape}, X_test {X_test.shape}")

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    print("\n--- Reference: src/baseline.py's GradientBoosting baseline (same run) ---")
    from sklearn.ensemble import GradientBoostingRegressor
    baseline_pipe = Pipeline([
        ("prep", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", RobustScaler()),
            ("var_thresh", VarianceThreshold(threshold=1e-8)),
            ("select", SelectKBest(f_regression, k=100)),
        ])),
        ("model", GradientBoostingRegressor(random_state=RANDOM_STATE)),
    ])
    baseline_scores = cross_val_score(baseline_pipe, X_train, y_train, cv=kf, scoring="r2")
    print(f"  GBR baseline (median, k=100): R^2 = {baseline_scores.mean():.4f} +/- {baseline_scores.std():.4f}")

    print("\n--- Winning pipeline: TopCorrMixedImputer(n=40) + SelectKBest(k=200) + XGBoost(tuned) ---")
    print("  (leak-free: Pipeline+cross_val_score refits imputer/scaler/selector/model per fold)")
    win_scores = cross_val_score(build_full_pipeline(), X_train, y_train, cv=kf, scoring="r2")
    print(f"  Combined pipeline: R^2 = {win_scores.mean():.4f} +/- {win_scores.std():.4f}  "
          f"(folds: {np.round(win_scores, 4)})")
    print(f"  Improvement over baseline: {win_scores.mean() - baseline_scores.mean():+.4f}")

    print(f"\n--- Bootstrap uncertainty quantification ({N_BOOTSTRAP} resamples, OOB scoring) ---")
    boot_scores, (ci_lo, ci_hi) = bootstrap_ci(X_train, y_train, N_BOOTSTRAP)

    print("\n--- Subtask 1: outlier classification (reporting only, not used for training) ---")
    prep_for_outliers = build_preprocessor()
    X_train_proc_for_outliers = prep_for_outliers.fit_transform(X_train, y_train)
    classify_outliers(X_train_proc_for_outliers, train_ids)
    print(
        "  (Outlier removal was retried leak-free in notebooks/outlier_retry.py: hard removal "
        "-- EllipticEnvelope, OneClassSVM -- still loses to no-removal at every contamination "
        "tried, so it is NOT used to filter training data here either; see module docstring.)"
    )

    print("\n--- Final fit on all training data, predict on test set ---")
    final_pipe = build_full_pipeline()
    final_pipe.fit(X_train, y_train)
    y_pred_test = final_pipe.predict(X_test)
    train_r2 = r2_score(y_train, final_pipe.predict(X_train))
    print(f"  Final model train R^2 (in-sample, optimistic): {train_r2:.4f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    submission = pd.DataFrame({"id": test_ids, "y": y_pred_test})
    out_path = os.path.join(OUTPUT_DIR, "advanced_submission.csv")
    submission.to_csv(out_path, index=False)
    print(f"  Wrote {out_path} ({submission.shape[0]} rows)")

    print("\n--- Runner-up candidate: plain XGBoost(tuned) on baseline preprocessing (simpler, R^2=0.5398) ---")
    runner_up_pipe = Pipeline([
        ("prep", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", RobustScaler()),
            ("var_thresh", VarianceThreshold(threshold=1e-8)),
            ("select", SelectKBest(f_regression, k=100)),
        ])),
        ("model", XGBRegressor(**XGB_PARAMS)),
    ])
    runner_up_scores = cross_val_score(runner_up_pipe, X_train, y_train, cv=kf, scoring="r2")
    print(f"  Runner-up CV: R^2 = {runner_up_scores.mean():.4f} +/- {runner_up_scores.std():.4f}")
    runner_up_pipe.fit(X_train, y_train)
    y_pred_runner_up = runner_up_pipe.predict(X_test)
    runner_up_submission = pd.DataFrame({"id": test_ids, "y": y_pred_runner_up})
    runner_up_path = os.path.join(OUTPUT_DIR, "xgboost_tuned_baseline_preproc_submission.csv")
    runner_up_submission.to_csv(runner_up_path, index=False)
    print(f"  Wrote {runner_up_path} ({runner_up_submission.shape[0]} rows)")

    print("\n=== Summary ===")
    print(f"  Baseline (src/baseline.py, GBR):                       R^2 = {baseline_scores.mean():.4f}")
    print(f"  Runner-up (XGBoost tuned, baseline preproc):           R^2 = {runner_up_scores.mean():.4f}")
    print(f"  WINNER (XGBoost tuned + combined B/C preproc):         R^2 = {win_scores.mean():.4f} "
          f"+/- {win_scores.std():.4f}")
    print(f"  Bootstrap 90% CI on winner's CV R^2:                   [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"\nTotal runtime: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
