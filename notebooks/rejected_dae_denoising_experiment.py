"""Experiment: insert a denoising autoencoder (DAE) ahead of feature selection.

Treats the premise "the data may be noisy" literally: src/denoising_autoencoder.py
trains an autoencoder to reconstruct clean rows from randomly-corrupted
versions of themselves (a la Vincent et al.'s stacked DAEs), then replaces
each row with its denoised reconstruction before the rest of the pipeline
ever sees it. Unlike notebooks/sparse_autoencoder_experiment.py (an
unsupervised bottleneck meant to *replace* CorrelationPrunedKBest, which lost
badly), this DAE keeps the full feature count and sits *ahead* of the
existing, already-adopted CorrelationPrunedKBest + StackingRegressor combo
(notebooks/stacking_with_corr_pruned.py, CV R^2=0.5229) -- it's an extra
step, not a replacement, so the question is purely "does denoising help or
hurt on top of what already works."

Pipeline layout (why DAE is a SHARED step, not duplicated per base learner
like CorrelationPrunedKBest is): the DAE is unsupervised -- fit(X) never
touches y -- so it's safe to fit once per outer CV fold, ahead of
StackingRegressor, without leaking outer-test-fold information or biasing
the meta-learner's out-of-fold predictions. CorrelationPrunedKBest, being
supervised (uses y for f_regression ranking), stays duplicated inside each
base learner's own sub-pipeline so StackingRegressor's internal cv=5
cross-fit still refits it per internal fold -- same leak-free discipline as
notebooks/stacking_with_corr_pruned.py. Sharing the DAE instead of
duplicating it inside all three base learners' pipelines cuts DAE training
calls roughly 15x (5 outer folds instead of ~90 internal+outer fits),
keeping this tractable.

Keeps the +/-15-IQR post-scale clip (guards both f_regression inside the
selector and the DAE's MSE loss against the injected noise columns, std
~1e22 even after RobustScaler).
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from src.rejected_denoising_autoencoder import DenoisingAutoencoderFeatures  # noqa: E402
from src.feature_selection import CorrelationPrunedKBest  # noqa: E402

RANDOM_STATE = 42
N_SPLITS = 5
K_BEST = 100
CORR_THRESHOLD = 0.9
CLIP_IQR = 15.0

DATA_DIR = os.path.join(REPO_ROOT, "data", "raw")


def load_data():
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"))
    feat_cols = [c for c in X_train.columns if c != "id"]
    return X_train[feat_cols].values, y_train["y"].values


def build_shared_prep(dae):
    steps = [
        ("impute", SimpleImputer(strategy="median")),
        ("scale", RobustScaler()),
        ("clip", FunctionTransformer(np.clip, kw_args={"a_min": -CLIP_IQR, "a_max": CLIP_IQR})),
        ("var_thresh", VarianceThreshold(threshold=1e-8)),
    ]
    if dae is not None:
        steps.append(("dae", dae))
    return Pipeline(steps)


def build_base_estimator(model):
    return Pipeline([
        ("select", CorrelationPrunedKBest(n_features=K_BEST, corr_threshold=CORR_THRESHOLD)),
        ("model", model),
    ])


def build_stack():
    estimators = [
        ("gbr", build_base_estimator(GradientBoostingRegressor(random_state=RANDOM_STATE))),
        ("ridge", build_base_estimator(Ridge(alpha=10.0))),
        ("knn", build_base_estimator(KNeighborsRegressor(n_neighbors=15, weights="distance"))),
    ]
    return StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=1.0), cv=N_SPLITS, n_jobs=-1)


def build_full_pipeline(dae):
    return Pipeline([("prep", build_shared_prep(dae)), ("stack", build_stack())])


def main():
    X, y = load_data()
    print(f"X {X.shape}   to beat: CorrPrunedKBest+stack, no DAE = R^2 0.5229\n")

    configs = {
        "no DAE (control)": None,
        "DAE(hidden=256, corrupt=0.1)": DenoisingAutoencoderFeatures(
            hidden_size=256, corruption_frac=0.1, random_state=RANDOM_STATE),
        "DAE(hidden=256, corrupt=0.3)": DenoisingAutoencoderFeatures(
            hidden_size=256, corruption_frac=0.3, random_state=RANDOM_STATE),
        "DAE(hidden=256, corrupt=0.5)": DenoisingAutoencoderFeatures(
            hidden_size=256, corruption_frac=0.5, random_state=RANDOM_STATE),
        "DAE(hidden=512, corrupt=0.3)": DenoisingAutoencoderFeatures(
            hidden_size=512, corruption_frac=0.3, random_state=RANDOM_STATE),
    }

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for name, dae in configs.items():
        pipe = build_full_pipeline(dae)
        scores = cross_val_score(pipe, X, y, cv=kf, scoring="r2", n_jobs=1)
        results[name] = scores
        print(f"{name:>32}: R^2 = {scores.mean():.4f} +/- {scores.std():.4f}  "
              f"(folds: {np.round(scores, 3)})")

    best = max(results, key=lambda k: results[k].mean())
    print(f"\nBest config: {best} (R^2 = {results[best].mean():.4f})")


if __name__ == "__main__":
    main()
