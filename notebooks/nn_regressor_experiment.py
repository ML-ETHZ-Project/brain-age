"""Experiment: feedforward neural-network regressor vs. GradientBoostingRegressor.

src/baseline.py picks GradientBoostingRegressor over Ridge/RandomForest by
5-fold CV. This tries a small MLP (src/neural_net_regressor.py) as a fourth
candidate, on top of the same preprocessing (median impute -> RobustScaler
-> VarianceThreshold -> CorrelationPrunedKBest(n=100, thresh=0.9)) so only
the final model changes.

First pass swept architecture width/regularization at 300 epochs; the best
MLP (128,64 hidden) got R^2=0.4770 vs GradientBoosting's 0.5141, with a
clear "bigger hidden layers do better" trend. This second pass follows up
on that trend with three more levers: more epochs, a cosine-annealing LR
schedule (decay the learning rate over training instead of a constant
rate, so late epochs take smaller, more precise steps rather than
overshooting), and EnsembleMLPRegressor (src/neural_net_regressor.py) --
averaging predictions across several identically-configured MLPs trained
from different random seeds, to cancel out some of each individual
network's training-variance noise (a from-scratch NN on ~970 rows is far
more seed-sensitive than a tree ensemble, which is itself already an
ensemble).

At n=1212 rows / 100 features per fold (~970 training rows after the CV
split), a from-scratch MLP has far less data than deep learning typically
wants -- dropout and weight decay are there to fight overfitting, but
tree ensembles are usually hard to beat in this regime. Measure, don't
assume.

Model is refit inside each CV fold (part of the sklearn Pipeline handed to
cross_val_score), so no y-leakage into the reported scores -- same
leak-free discipline as the other notebooks/*_comparison.py scripts.
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
from src.neural_net_regressor import EnsembleMLPRegressor, TorchMLPRegressor  # noqa: E402

RANDOM_STATE = 42
N_SPLITS = 5
K_BEST = 100
CORR_THRESHOLD = 0.9

DATA_DIR = os.path.join(REPO_ROOT, "data", "raw")


def load_data():
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"))
    feat_cols = [c for c in X_train.columns if c != "id"]
    return X_train[feat_cols].values, y_train["y"].values


def build_preprocessor():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", RobustScaler()),
        ("var_thresh", VarianceThreshold(threshold=1e-8)),
        ("select", CorrelationPrunedKBest(n_features=K_BEST, corr_threshold=CORR_THRESHOLD)),
    ])


def main():
    X_train, y_train = load_data()
    print(f"X_train {X_train.shape}\n")

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    models = {
        "GradientBoosting (current best)": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "MLP(128,64) e=300 (round-1 best)": TorchMLPRegressor(
            hidden_sizes=(128, 64), dropout=0.3, weight_decay=1e-3,
            n_epochs=300, random_state=RANDOM_STATE,
        ),
        "MLP(128,64) e=600 cosine-lr": TorchMLPRegressor(
            hidden_sizes=(128, 64), dropout=0.3, weight_decay=1e-3,
            n_epochs=600, lr_schedule=True, random_state=RANDOM_STATE,
        ),
        "MLP(256,128) e=600 cosine-lr": TorchMLPRegressor(
            hidden_sizes=(256, 128), dropout=0.3, weight_decay=1e-3,
            n_epochs=600, lr_schedule=True, random_state=RANDOM_STATE,
        ),
        "Ensemble(5 seeds) MLP(128,64) e=600 cosine-lr": EnsembleMLPRegressor(
            n_models=5, hidden_sizes=(128, 64), dropout=0.3, weight_decay=1e-3,
            n_epochs=600, lr_schedule=True, random_state=RANDOM_STATE,
        ),
    }

    results = {}
    for name, model in models.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
        scores = cross_val_score(pipe, X_train, y_train, cv=kf, scoring="r2")
        results[name] = scores
        print(f"{name:>35}: R^2 = {scores.mean():.4f} +/- {scores.std():.4f}  "
              f"(folds: {np.round(scores, 3)})")

    best = max(results, key=lambda k: results[k].mean())
    print(f"\nBest model: {best} (R^2 = {results[best].mean():.4f})")


if __name__ == "__main__":
    main()
