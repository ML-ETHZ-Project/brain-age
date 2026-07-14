"""Ensembling / stacking vs the single-model baseline (R^2 = 0.5065).

Idea: blend learners with different inductive biases so the ensemble catches
signal no single model does -- a gradient-boosted tree (nonlinear, feature
interactions), a regularized linear model (Ridge), and a distance-based
model (KNN) -- via sklearn's StackingRegressor with a Ridge meta-learner.

Leakage discipline (CLAUDE.md, and the outlier-removal incident): every step
that looks at the data lives inside the pipeline, so cross_val_score refits
imputer/scaler/clip/variance/SelectKBest AND every base model on each outer
fold's training rows only. StackingRegressor additionally cross-fits its base
learners internally (cv=5) to build out-of-fold meta-features, so the
meta-learner never trains on a base model's in-sample predictions. Nothing
sees held-out rows during fitting.

Preprocessing note: the deliberately injected noise columns (std ~1e22, see
EDA) overflow float64 inside f_regression even after RobustScaler, polluting
the F-scores. A robust clip (+/-15 IQR units) after scaling caps them
losslessly for the real features -- the same fix used elsewhere.

Reports leak-free 5-fold CV R^2 for each base model alone and for the stack,
all through the identical pipeline, so the comparison is like-for-like.
"""
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

RANDOM_STATE = 42
N_SPLITS = 5
K_BEST = 100
CLIP_IQR = 15.0

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data", "raw")


def load_data():
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"))
    feat_cols = [c for c in X_train.columns if c != "id"]
    return X_train[feat_cols].values, y_train["y"].values


def build_preprocessor():
    """Shared, leak-free preprocessing: median-impute -> robust-scale ->
    clip injected-noise spikes -> drop constants -> univariate top-K."""
    return [
        ("impute", SimpleImputer(strategy="median")),
        ("scale", RobustScaler()),
        ("clip", FunctionTransformer(
            np.clip, kw_args={"a_min": -CLIP_IQR, "a_max": CLIP_IQR})),
        ("var_thresh", VarianceThreshold(threshold=1e-8)),
        ("select", SelectKBest(f_regression, k=K_BEST)),
    ]


def make_pipeline(model):
    return Pipeline(build_preprocessor() + [("model", model)])


def base_estimators():
    """Three diverse learners for the stack. Names are reused when scoring
    each one in isolation, so the standalone and stacked numbers line up."""
    return [
        ("gbr", GradientBoostingRegressor(random_state=RANDOM_STATE)),
        ("ridge", Ridge(alpha=10.0)),
        ("knn", KNeighborsRegressor(n_neighbors=15, weights="distance")),
    ]


def evaluate(name, model, X, y):
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(make_pipeline(model), X, y, cv=kf, scoring="r2", n_jobs=-1)
    print(f"{name:>22}: R^2 = {scores.mean():.4f} +/- {scores.std():.4f}  "
          f"(folds: {np.round(scores, 3)})")
    return scores.mean()


def main():
    X, y = load_data()
    print(f"X {X.shape}   baseline to beat: R^2 = 0.5065\n")

    results = {}
    print("--- base learners alone (leak-free 5-fold CV) ---")
    for name, model in base_estimators():
        results[name] = evaluate(name, model, X, y)

    print("\n--- stacked ensemble (Ridge meta-learner, 5-fold internal cross-fit) ---")
    stack = StackingRegressor(
        estimators=base_estimators(),
        final_estimator=Ridge(alpha=1.0),
        cv=N_SPLITS,
        n_jobs=-1,
    )
    results["STACK"] = evaluate("STACK", stack, X, y)

    best = max(results, key=results.get)
    print(f"\nBest: {best} (R^2 = {results[best]:.4f})")
    print(f"Best base learner: "
          f"{max((k for k in results if k != 'STACK'), key=results.get)} "
          f"({max(v for k, v in results.items() if k != 'STACK'):.4f})")
    delta = results["STACK"] - max(v for k, v in results.items() if k != "STACK")
    print(f"Stacking vs best base learner: {delta:+.4f} R^2")


if __name__ == "__main__":
    main()
