"""Combine this branch's CorrelationPrunedKBest with philippe/ensembling's
stacking ensemble -- two independent improvements to different, non-
overlapping stages of the same src/baseline.py fork point:

  - feature-section (this branch): SelectKBest -> CorrelationPrunedKBest
    (redundancy-pruned selection), single GradientBoostingRegressor.
    CV R^2 0.5065 -> 0.5141.
  - philippe/ensembling (teammate's branch, unmerged): kept SelectKBest,
    but replaced the single model with a StackingRegressor blending
    GradientBoosting + Ridge(alpha=10) + KNN(k=15, distance) via a Ridge
    meta-learner (notebooks/stacking_comparison.py on that branch).
    CV R^2 0.5065 -> 0.5179 (default hyperparameters; a further-tuned
    variant reached 0.5240 but is flagged there as mildly optimistic from
    hyperparameter-selection bias, so 0.5179 is the clean reference point).

This swaps stacking_comparison.py's SelectKBest(k=100) for
CorrelationPrunedKBest(n=100, thresh=0.9) and re-measures, to see whether
the two gains stack. Keeps stacking_comparison.py's +/-15-IQR post-scale
clip (guards f_regression against the injected noise columns, which have
std ~1e22 even after RobustScaler) for both variants, so the ONLY thing
that differs between the two stacks is the selector.

Leak-free: StackingRegressor cross-fits its base learners internally
(cv=5) so the meta-learner never trains on in-sample base predictions, and
the outer cross_val_score refits the entire pipeline -- imputer, scaler,
selector, and every base learner -- on each fold's training rows only.
Same discipline as every other notebooks/*_comparison.py script here.
"""
import os
import sys
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

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
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


def build_shared_steps():
    return [
        ("impute", SimpleImputer(strategy="median")),
        ("scale", RobustScaler()),
        ("clip", FunctionTransformer(np.clip, kw_args={"a_min": -CLIP_IQR, "a_max": CLIP_IQR})),
        ("var_thresh", VarianceThreshold(threshold=1e-8)),
    ]


def make_selector(kind):
    if kind == "selectkbest":
        return SelectKBest(f_regression, k=K_BEST)
    if kind == "corrpruned":
        return CorrelationPrunedKBest(n_features=K_BEST, corr_threshold=CORR_THRESHOLD)
    raise ValueError(kind)


def make_base_pipeline(kind, model):
    return Pipeline(build_shared_steps() + [("select", make_selector(kind)), ("model", model)])


def base_estimators(kind):
    return [
        ("gbr", make_base_pipeline(kind, GradientBoostingRegressor(random_state=RANDOM_STATE))),
        ("ridge", make_base_pipeline(kind, Ridge(alpha=10.0))),
        ("knn", make_base_pipeline(kind, KNeighborsRegressor(n_neighbors=15, weights="distance"))),
    ]


def evaluate(name, estimator, X, y):
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(estimator, X, y, cv=kf, scoring="r2", n_jobs=-1)
    print(f"{name:>40}: R^2 = {scores.mean():.4f} +/- {scores.std():.4f}  "
          f"(folds: {np.round(scores, 3)})")
    return scores.mean()


def main():
    X, y = load_data()
    print(f"X {X.shape}\n")
    print("Reference points: SelectKBest+GBR=0.5065, CorrPrunedKBest+GBR=0.5141, "
          "SelectKBest-stack (philippe/ensembling)=0.5179\n")

    results = {}
    for kind in ["selectkbest", "corrpruned"]:
        print(f"--- selector = {kind} ---")
        for est_name, model_pipe in base_estimators(kind):
            results[f"{kind}/{est_name}"] = evaluate(f"{kind}/{est_name}", model_pipe, X, y)

        stack = StackingRegressor(
            estimators=base_estimators(kind),
            final_estimator=Ridge(alpha=1.0),
            cv=N_SPLITS,
            n_jobs=-1,
        )
        results[f"{kind}/STACK"] = evaluate(f"{kind}/STACK", stack, X, y)
        print()

    best = max(results, key=results.get)
    print(f"Best overall: {best} (R^2 = {results[best]:.4f})")
    print(f"corrpruned/STACK vs selectkbest/STACK: "
          f"{results['corrpruned/STACK'] - results['selectkbest/STACK']:+.4f}")


if __name__ == "__main__":
    main()
