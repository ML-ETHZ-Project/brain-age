"""Lane B: does a MIXED per-column imputation strategy beat blanket median?

notebooks/impute_comparison.py already established that median imputation,
applied as a single blanket strategy to every column, beats mean / KNN /
iterative / most_frequent (R^2 = 0.5065). The hypothesis here is narrower:
maybe fancy (iterative/MICE) imputation only pays off on the handful of
columns that actually carry age-signal, while blindly applying it to all
832 columns (many of which are near-pure noise, per notebooks/eda.py) just
adds estimation variance that hurts more than it helps.

Two variants are tried, both plugged into the SAME downstream pipeline as
impute_comparison.py (RobustScaler -> VarianceThreshold(1e-8) ->
SelectKBest(f_regression, k=100) -> GradientBoostingRegressor), evaluated
with sklearn Pipeline + cross_val_score so every fitted piece -- including
which columns count as "top-correlated" -- is refit per CV fold on that
fold's training rows only (no leakage of validation-row correlations):

1. TopCorrMixedImputer: a custom column-selecting transformer. In fit(X, y)
   it ranks all 832 columns by |corr(x_j, y)| computed pairwise-NaN-safe on
   the training fold only, picks the top `n_fancy` columns, fits
   IterativeImputer(BayesianRidge()) on just those, and SimpleImputer
   (median) on the rest, then reassembles columns in original order. Tried
   at n_fancy in {30, 40, 50} per the task's suggested range.
2. SimpleImputer(strategy="median", add_indicator=True): median-impute as
   before, but append a binary "was this cell missing" indicator column per
   originally-missing feature, in case missingness itself is informative
   (rather than just noise to fill in).

Order matches impute_comparison.py convention of putting the imputer first
(mirrors src/baseline.py's build_preprocessor: impute -> scale ->
var_thresh -> select -> model) so RobustScaler's median/IQR (NaN-tolerant)
still runs after imputation is complete, downstream of both experimental
imputers.
"""
import os

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

RANDOM_STATE = 42
N_SPLITS = 5
K_BEST = 100
BASELINE_R2 = 0.5065

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data", "raw")


def load_data():
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"))
    feat_cols = [c for c in X_train.columns if c != "id"]
    return X_train[feat_cols].values, y_train["y"].values


def _nan_safe_abs_corr(x_col, y):
    """abs(Pearson corr) between a single column and y, using only rows
    where x_col is observed. Returns 0 for degenerate cases (too few
    observed rows, or zero variance) instead of NaN."""
    mask = ~np.isnan(x_col)
    if mask.sum() < 2:
        return 0.0
    xv = x_col[mask]
    yv = y[mask]
    if np.std(xv) == 0 or np.std(yv) == 0:
        return 0.0
    c = np.corrcoef(xv, yv)[0, 1]
    return 0.0 if np.isnan(c) else abs(c)


class TopCorrMixedImputer(BaseEstimator, TransformerMixin):
    """Iterative (MICE) imputation on the n_fancy columns most correlated
    with the target (ranked on this fit's own X, y only), median imputation
    on the remaining columns. Column ranking + both imputers are refit from
    scratch on whatever rows are passed to fit() -- safe to drop straight
    into a Pipeline used with cross_val_score.
    """

    def __init__(self, n_fancy=40, max_iter=5, n_nearest_features=30, random_state=RANDOM_STATE):
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
        order = np.argsort(-abs_corrs)  # descending
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


def build_pipeline(imputer, model_random_state=RANDOM_STATE):
    return Pipeline([
        ("impute", imputer),
        ("scale", RobustScaler()),
        ("var_thresh", VarianceThreshold(threshold=1e-8)),
        ("select", SelectKBest(f_regression, k=K_BEST)),
        ("model", GradientBoostingRegressor(random_state=model_random_state)),
    ])


def evaluate(name, imputer, X, y, kf):
    pipe = build_pipeline(imputer)
    scores = cross_val_score(pipe, X, y, cv=kf, scoring="r2")
    print(f"{name:>28}: R^2 = {scores.mean():.4f} +/- {scores.std():.4f}  "
          f"(folds: {np.round(scores, 3)})")
    return scores


def main():
    X_train, y_train = load_data()
    print(f"X_train {X_train.shape}, missing cells: {np.isnan(X_train).sum()} "
          f"({np.isnan(X_train).mean():.2%})\n")

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    print("--- Reference: blanket median (src/baseline.py) ---")
    results["median (blanket, reference)"] = evaluate(
        "median (blanket, reference)", SimpleImputer(strategy="median"), X_train, y_train, kf
    )

    print("\n--- Variant 1: mixed iterative-on-top-corr-cols + median-on-rest ---")
    for n_fancy in (30, 40, 50):
        name = f"mixed(top{n_fancy} iterative + median)"
        results[name] = evaluate(name, TopCorrMixedImputer(n_fancy=n_fancy), X_train, y_train, kf)

    print("\n--- Variant 2: median + missingness indicator ---")
    results["median + missing-indicator"] = evaluate(
        "median + missing-indicator",
        SimpleImputer(strategy="median", add_indicator=True),
        X_train, y_train, kf,
    )

    print("\n--- Summary ---")
    for name, scores in results.items():
        print(f"{name:>35}: R^2 = {scores.mean():.4f} +/- {scores.std():.4f}")

    best_name = max(results, key=lambda k: results[k].mean())
    best_mean = results[best_name].mean()
    print(f"\nBest: {best_name} (R^2 = {best_mean:.4f})")
    print(f"Baseline to beat (blanket median, full pipeline w/ GBR): {BASELINE_R2:.4f}")
    if best_mean > BASELINE_R2:
        print(f"-> Mixed/indicator imputation BEATS baseline by {best_mean - BASELINE_R2:+.4f}")
    else:
        print(f"-> Mixed/indicator imputation does NOT beat baseline ({best_mean - BASELINE_R2:+.4f})")


if __name__ == "__main__":
    main()
