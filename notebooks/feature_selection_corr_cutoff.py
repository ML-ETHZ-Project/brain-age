"""Subtask 2 ablation: SelectKBest(k=100) vs. CorrelationPrunedKBest at n_features=25 and 100.

notebooks/feature_visualization.ipynb flagged that several of the top age-correlated
features are highly inter-correlated (e.g. x133/x334/x465, all pairwise |r| > 0.9),
so SelectKBest's independent per-feature ranking can spend part of its k=100 budget
on near-duplicates. src/feature_selection.py's CorrelationPrunedKBest walks the same
f_regression ranking but skips a feature if it's too correlated with one already
kept.

First tried n_features=25 (trading redundancy for a much smaller set) -- that lost
more signal than the pruning recovered (R^2 ~0.45-0.46, see README experiment log).
This version also tries n_features=100 -- same output width as the SelectKBest
baseline, so any R^2 change isolates the effect of swapping out redundant picks for
the next-best distinct ones, without also shrinking the feature count.
corr_threshold is swept at both widths since the cutoff choice is somewhat
arbitrary and the result should be robust to it, not cherry-picked.

Selector is refit inside each CV fold (part of the sklearn Pipeline handed to
cross_val_score), so no y-leakage into the reported scores -- same leak-free
discipline as notebooks/outlier_comparison.py and feature_selection_comparison.py.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from src.feature_selection import CorrelationPrunedKBest  # noqa: E402

RANDOM_STATE = 42
N_SPLITS = 5
K_BEST = 100
N_FEATURES_PRUNED = [25, 100]
CORR_THRESHOLDS = [0.8, 0.9, 0.95]

DATA_DIR = os.path.join(REPO_ROOT, "data", "raw")


def load_data():
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"))
    feat_cols = [c for c in X_train.columns if c != "id"]
    return X_train[feat_cols].values, y_train["y"].values


def build_pipeline(selector):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", RobustScaler()),
        ("var_thresh", VarianceThreshold(threshold=1e-8)),
        ("select", selector),
        ("model", GradientBoostingRegressor(random_state=RANDOM_STATE)),
    ])


def main():
    X_train, y_train = load_data()
    print(f"X_train {X_train.shape}\n")

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    selectors = {
        f"SelectKBest(k={K_BEST})": SelectKBest(f_regression, k=K_BEST),
    }
    for n in N_FEATURES_PRUNED:
        for t in CORR_THRESHOLDS:
            selectors[f"CorrPrunedKBest(n={n}, thresh={t})"] = CorrelationPrunedKBest(
                n_features=n, corr_threshold=t,
            )

    results = {}
    for name, selector in selectors.items():
        pipe = build_pipeline(selector)
        scores = cross_val_score(pipe, X_train, y_train, cv=kf, scoring="r2")
        results[name] = scores
        print(f"{name:>40}: R^2 = {scores.mean():.4f} +/- {scores.std():.4f}  "
              f"(folds: {np.round(scores, 3)})")

    best = max(results, key=lambda k: results[k].mean())
    print(f"\nBest selector: {best} (R^2 = {results[best].mean():.4f})")


if __name__ == "__main__":
    main()
