"""Compare feature-selection scoring functions for SelectKBest.

Fixes the rest of the pipeline (median impute -> RobustScaler -> variance
filter -> SelectKBest(k=100) -> GradientBoostingRegressor, matching
src/baseline.py's winning model) and swaps only the SelectKBest score_func,
to isolate how much the selection criterion itself moves CV R^2.

Baseline uses f_regression, which scores each feature by its linear
correlation with age (F-test on Pearson r). This notebook adds
mutual_info_regression as an alternative: mutual information between a
feature and age is the KL divergence between their joint distribution
P(x, y) and the independence assumption P(x)P(y), so it scores ANY
statistical dependency (linear or not), not just linear ones. Given
~300+ near-irrelevant columns and some genuinely non-linear anatomical
relationships, MI-based selection could keep signal that f_regression's
linear lens misses -- or it could just add k-NN density-estimation noise
at this sample size (1212 rows). Measure, don't assume.

SelectKBest is refit inside each CV fold (it's part of the sklearn
Pipeline handed to cross_val_score), so no y-leakage into the reported
scores -- same leak-free discipline as notebooks/outlier_comparison.py.
"""
import os
from functools import partial

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression, mutual_info_regression
from sklearn.impute import SimpleImputer
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


SCORE_FUNCS = {
    "f_regression": f_regression,
    "mutual_info": partial(mutual_info_regression, random_state=RANDOM_STATE, n_neighbors=3),
}


def build_pipeline(score_func):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", RobustScaler()),
        ("var_thresh", VarianceThreshold(threshold=1e-8)),
        ("select", SelectKBest(score_func, k=K_BEST)),
        ("model", GradientBoostingRegressor(random_state=RANDOM_STATE)),
    ])


def main():
    X_train, y_train = load_data()
    print(f"X_train {X_train.shape}\n")

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for name, score_func in SCORE_FUNCS.items():
        pipe = build_pipeline(score_func)
        scores = cross_val_score(pipe, X_train, y_train, cv=kf, scoring="r2")
        results[name] = scores
        print(f"{name:>15}: R^2 = {scores.mean():.4f} +/- {scores.std():.4f}  "
              f"(folds: {np.round(scores, 3)})")

    best = max(results, key=lambda k: results[k].mean())
    print(f"\nBest score_func: {best} (R^2 = {results[best].mean():.4f})")


if __name__ == "__main__":
    main()
