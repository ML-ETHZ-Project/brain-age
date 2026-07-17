"""Experiment: LightGBM as an alternative/addition to GradientBoostingRegressor.

LightGBM grows trees leaf-wise (best-first, picks the leaf with the largest
loss reduction next) rather than GradientBoostingRegressor's level-wise
growth (expands every leaf at the current depth before going deeper) --
often a better fit for small-to-medium tabular datasets since it reaches
useful splits with fewer total splits, though it can overfit faster without
regularization (num_leaves, min_child_samples), which is why those are swept
here rather than left at LightGBM's larger defaults.

Two questions on top of the same preprocessing (median impute -> RobustScaler
-> +/-15-IQR clip -> VarianceThreshold -> CorrelationPrunedKBest(n=100,
thresh=0.9)):
  1. Solo: does LightGBM beat GradientBoostingRegressor as a single model?
  2. Stacked: does adding LightGBM as a 4th base learner (alongside
     GradientBoosting + Ridge + KNN) improve the existing stacking ensemble
     (notebooks/stacking_with_corr_pruned.py, CV R^2=0.5229)?

Leak-free: every candidate is scored via cross_val_score with the selector
refit inside each fold (or, for the stack, inside StackingRegressor's
internal cv=5 per base learner) -- same discipline as every other
notebooks/*_comparison.py script here.

NOTE: LightGBM's compiled extension needs the OpenMP runtime (libomp), which
isn't installed system-wide on this machine (no Homebrew). scikit-learn
happens to bundle its own libomp.dylib; dyld only reads DYLD_LIBRARY_PATH at
process launch (not if set from within an already-running process), so the
block below re-execs this same script with that variable set, once, before
anything imports lightgbm. `python notebooks/lightgbm_experiment.py` works
standalone -- no manual env var, no Homebrew install.
"""
import os
import sys
import warnings

if not os.environ.get("_LIGHTGBM_LIBOMP_REEXEC"):
    _venv_lib = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".venv", "lib",
    )
    _libomp = None
    if os.path.isdir(_venv_lib):
        for _pyver in os.listdir(_venv_lib):
            _candidate = os.path.join(_venv_lib, _pyver, "site-packages", "sklearn", ".dylibs", "libomp.dylib")
            if os.path.isfile(_candidate):
                _libomp = os.path.dirname(_candidate)
                break
    if _libomp is not None:
        env = os.environ.copy()
        env["DYLD_LIBRARY_PATH"] = _libomp
        env["DYLD_FALLBACK_LIBRARY_PATH"] = _libomp
        env["_LIGHTGBM_LIBOMP_REEXEC"] = "1"
        os.execve(sys.executable, [sys.executable] + sys.argv, env)

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
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


def build_shared_prep():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", RobustScaler()),
        ("clip", FunctionTransformer(np.clip, kw_args={"a_min": -CLIP_IQR, "a_max": CLIP_IQR})),
        ("var_thresh", VarianceThreshold(threshold=1e-8)),
    ])


def build_base_estimator(model):
    return Pipeline([
        ("select", CorrelationPrunedKBest(n_features=K_BEST, corr_threshold=CORR_THRESHOLD)),
        ("model", model),
    ])


def lgbm(num_leaves, min_child_samples):
    return LGBMRegressor(
        num_leaves=num_leaves, min_child_samples=min_child_samples,
        random_state=RANDOM_STATE, verbosity=-1,
    )


def build_stack(extra_estimator=None):
    estimators = [
        ("gbr", build_base_estimator(GradientBoostingRegressor(random_state=RANDOM_STATE))),
        ("ridge", build_base_estimator(Ridge(alpha=10.0))),
        ("knn", build_base_estimator(KNeighborsRegressor(n_neighbors=15, weights="distance"))),
    ]
    if extra_estimator is not None:
        estimators.append(("lgbm", build_base_estimator(extra_estimator)))
    return StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=1.0), cv=N_SPLITS, n_jobs=-1)


def main():
    X, y = load_data()
    print(f"X {X.shape}   to beat: GBR solo=0.5141, current stack (GBR+Ridge+KNN)=0.5229\n")

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    print("--- solo models ---")
    solo_models = {
        "GBR solo (reference)": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "LightGBM(leaves=31, min_child=20)": lgbm(31, 20),
        "LightGBM(leaves=15, min_child=30)": lgbm(15, 30),
        "LightGBM(leaves=63, min_child=10)": lgbm(63, 10),
    }
    for name, model in solo_models.items():
        pipe = Pipeline([("prep", build_shared_prep()), ("select_model", build_base_estimator(model))])
        scores = cross_val_score(pipe, X, y, cv=kf, scoring="r2")
        print(f"{name:>36}: R^2 = {scores.mean():.4f} +/- {scores.std():.4f}  (folds: {np.round(scores, 3)})")

    print("\n--- stacks ---")
    stack_configs = {
        "GBR+Ridge+KNN (current stack)": None,
        "+ LightGBM(leaves=31, min_child=20)": lgbm(31, 20),
        "+ LightGBM(leaves=15, min_child=30)": lgbm(15, 30),
    }
    stack_results = {}
    for name, extra in stack_configs.items():
        pipe = Pipeline([("prep", build_shared_prep()), ("stack", build_stack(extra))])
        scores = cross_val_score(pipe, X, y, cv=kf, scoring="r2", n_jobs=1)
        stack_results[name] = scores.mean()
        print(f"{name:>40}: R^2 = {scores.mean():.4f} +/- {scores.std():.4f}  (folds: {np.round(scores, 3)})")

    best = max(stack_results, key=stack_results.get)
    print(f"\nBest stack: {best} (R^2 = {stack_results[best]:.4f})")


if __name__ == "__main__":
    main()
