"""Subtask 0: compare missing-value imputation strategies.

Fixes the rest of the pipeline (scale -> variance filter -> select-K-best ->
GradientBoostingRegressor, matching src/baseline.py's winning model) and
swaps only the imputer, to isolate how much the imputation strategy itself
moves CV R^2.

RobustScaler runs BEFORE imputation (it's NaN-tolerant: computes median/IQR
ignoring NaNs and passes NaNs through in transform) so that KNN/iterative
imputers -- which rely on inter-column distances/regressions -- aren't
dominated by the injected noise columns that have huge raw scale (see
CLAUDE.md). Mean/median/most-frequent are per-column and unaffected by
this ordering.
"""
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

RANDOM_STATE = 42
N_SPLITS = 5
K_BEST = 100

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data", "raw")


def load_data():
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"))
    feat_cols = [c for c in X_train.columns if c != "id"]
    return X_train[feat_cols].values, y_train["y"].values


IMPUTERS = {
    "mean": SimpleImputer(strategy="mean"),
    "median": SimpleImputer(strategy="median"),
    "most_frequent": SimpleImputer(strategy="most_frequent"),
    "knn(k=5)": KNNImputer(n_neighbors=5),
    "iterative": IterativeImputer(
        max_iter=5, n_nearest_features=30, initial_strategy="median", random_state=RANDOM_STATE
    ),
}


def build_pipeline(imputer):
    return Pipeline([
        ("scale", RobustScaler()),
        ("impute", imputer),
        ("var_thresh", VarianceThreshold(threshold=1e-8)),
        ("select", SelectKBest(f_regression, k=K_BEST)),
        ("model", GradientBoostingRegressor(random_state=RANDOM_STATE)),
    ])


def main():
    X_train, y_train = load_data()
    print(f"X_train {X_train.shape}, missing cells: {np.isnan(X_train).sum()} "
          f"({np.isnan(X_train).mean():.2%})\n")

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for name, imputer in IMPUTERS.items():
        pipe = build_pipeline(imputer)
        scores = cross_val_score(pipe, X_train, y_train, cv=kf, scoring="r2")
        results[name] = scores
        print(f"{name:>15}: R^2 = {scores.mean():.4f} +/- {scores.std():.4f}  "
              f"(folds: {np.round(scores, 3)})")

    best = max(results, key=lambda k: results[k].mean())
    print(f"\nBest strategy: {best} (R^2 = {results[best].mean():.4f})")


if __name__ == "__main__":
    main()
