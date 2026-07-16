"""Experiment: replace CorrelationPrunedKBest with a sparse autoencoder bottleneck.

src/baseline.py currently selects features by ranking + correlation-pruning
the original 832 columns against age (supervised, see src/feature_selection.py).
This tries an unsupervised alternative: train a sparse autoencoder
(src/sparse_autoencoder.py) on the imputed/scaled features and use its
hidden-layer activations as the feature set instead, at a few hidden-layer
widths and sparsity weights.

Fixes the rest of the pipeline (median impute -> RobustScaler ->
VarianceThreshold -> [feature step] -> GradientBoostingRegressor, matching
src/baseline.py) and swaps only the feature step, to isolate whether a
learned unsupervised bottleneck helps or hurts CV R^2 relative to the
current CorrelationPrunedKBest(n=100, thresh=0.9) baseline.

The autoencoder is refit inside each CV fold (part of the sklearn Pipeline
handed to cross_val_score), so no y-leakage into the reported scores -- same
leak-free discipline as the other notebooks/*_comparison.py scripts. (y isn't
used to fit the autoencoder itself, but the *rest* of the pipeline downstream
still must not see validation rows during fitting.)
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from src.feature_selection import CorrelationPrunedKBest  # noqa: E402
from src.sparse_autoencoder import SparseAutoencoderFeatures  # noqa: E402

RANDOM_STATE = 42
N_SPLITS = 5
K_BEST = 100
CORR_THRESHOLD = 0.9
HIDDEN_SIZES = [32, 64, 128]
SPARSITY_WEIGHTS = [1e-3, 1e-2]

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
    for n_hidden in HIDDEN_SIZES:
        for sparsity_weight in SPARSITY_WEIGHTS:
            name = f"SparseAE(hidden={n_hidden}, sparsity={sparsity_weight})"
            feature_steps[name] = SparseAutoencoderFeatures(
                n_hidden=n_hidden, sparsity_weight=sparsity_weight, random_state=RANDOM_STATE,
            )

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
