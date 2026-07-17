"""Experiment: replace CorrelationPrunedKBest with plain PCA.

Another unsupervised alternative to src/baseline.py's supervised
correlation-based selection (src/feature_selection.py) -- PCA finds the
directions of maximum variance in the imputed/scaled features, with no
knowledge of age, and keeps the top n_components as the new feature set.

Fixes the rest of the pipeline (median impute -> RobustScaler ->
VarianceThreshold -> [feature step] -> GradientBoostingRegressor, matching
src/baseline.py) and swaps only the feature step, sweeping n_components, to
see whether variance-maximizing directions happen to align with
age-predictive ones as well as the earlier sparse-autoencoder bottleneck did
(notebooks/sparse_autoencoder_experiment.py) or as well as the supervised
CorrelationPrunedKBest(n=100) baseline does.

PCA is refit inside each CV fold (part of the sklearn Pipeline handed to
cross_val_score), so no y-leakage into the reported scores -- same
leak-free discipline as the other notebooks/*_comparison.py scripts.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import VarianceThreshold
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
CORR_THRESHOLD = 0.9
N_COMPONENTS = [10, 25, 50, 100, 200]

DATA_DIR = os.path.join(REPO_ROOT, "data", "raw")


def load_data():
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"))
    feat_cols = [c for c in X_train.columns if c != "id"]
    return X_train[feat_cols].values, y_train["y"].values


def build_pipeline(feature_step):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", RobustScaler()),
        ("var_thresh", VarianceThreshold(threshold=1e-8)),
        ("features", feature_step),
        ("model", GradientBoostingRegressor(random_state=RANDOM_STATE)),
    ])


def main():
    X_train, y_train = load_data()
    print(f"X_train {X_train.shape}\n")

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    feature_steps = {
        f"CorrPrunedKBest(n={K_BEST}, thresh={CORR_THRESHOLD})": CorrelationPrunedKBest(
            n_features=K_BEST, corr_threshold=CORR_THRESHOLD,
        ),
    }
    for n in N_COMPONENTS:
        feature_steps[f"PCA(n_components={n})"] = PCA(n_components=n, random_state=RANDOM_STATE)

    results = {}
    for name, feature_step in feature_steps.items():
        pipe = build_pipeline(feature_step)
        scores = cross_val_score(pipe, X_train, y_train, cv=kf, scoring="r2")
        results[name] = scores
        print(f"{name:>45}: R^2 = {scores.mean():.4f} +/- {scores.std():.4f}  "
              f"(folds: {np.round(scores, 3)})")

    best = max(results, key=lambda k: results[k].mean())
    print(f"\nBest feature step: {best} (R^2 = {results[best].mean():.4f})")


if __name__ == "__main__":
    main()
