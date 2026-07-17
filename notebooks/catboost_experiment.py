"""Experiment: CatBoost as an alternative/addition to the current stack.

CatBoost uses ordered boosting (a permutation-driven scheme that avoids the
target leakage plain gradient boosting can suffer from on small datasets) and
grows oblivious (symmetric) trees, rather than GradientBoostingRegressor's
level-wise or LightGBM's leaf-wise growth. Structurally the most different
booster from what's already in the stack, so -- following the same
"diversity beats raw individual accuracy in a stack" pattern that motivated
adding LightGBM (notebooks/lightgbm_experiment.py, CV 0.5229 -> 0.5291) --
it's the next candidate worth testing.

Two questions on top of the same preprocessing (median impute -> RobustScaler
-> +/-15-IQR clip -> VarianceThreshold -> CorrelationPrunedKBest(n=100,
thresh=0.9)):
  1. Solo: does CatBoost beat GradientBoostingRegressor / LightGBM as a
     single model?
  2. Stacked: does adding CatBoost as a 5th base learner (alongside
     GradientBoosting + LightGBM + Ridge + KNN) improve the current stack
     (src/baseline.py, CV R^2=0.5291)?

Leak-free: every candidate is scored via cross_val_score with the selector
refit inside each fold (or, for the stack, inside StackingRegressor's
internal cv=5 per base learner) -- same discipline as every other
notebooks/*_comparison.py script here.

Needs `catboost` installed (added to requirements.txt but not yet installed
in .venv as of writing -- run `pip install -r requirements.txt` first) and
the OpenMP runtime for LightGBM -- see src/baseline.py's module docstring
for why this script self-re-execs with DYLD_LIBRARY_PATH set.
"""
import os
import sys

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

import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
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


def lgbm():
    return LGBMRegressor(
        num_leaves=15, min_child_samples=30, random_state=RANDOM_STATE, verbosity=-1,
    )


def catboost(depth, l2_leaf_reg):
    return CatBoostRegressor(
        depth=depth, l2_leaf_reg=l2_leaf_reg, random_state=RANDOM_STATE,
        verbose=False, allow_writing_files=False,
    )


def build_stack(extra_estimator=None):
    estimators = [
        ("gbr", build_base_estimator(GradientBoostingRegressor(random_state=RANDOM_STATE))),
        ("ridge", build_base_estimator(Ridge(alpha=10.0))),
        ("knn", build_base_estimator(KNeighborsRegressor(n_neighbors=15, weights="distance"))),
        ("lgbm", build_base_estimator(lgbm())),
    ]
    if extra_estimator is not None:
        estimators.append(("catboost", build_base_estimator(extra_estimator)))
    return StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=1.0), cv=N_SPLITS, n_jobs=-1)


def main():
    X, y = load_data()
    print(f"X {X.shape}   to beat: GBR solo=0.5141, LightGBM solo=0.5106, current stack (4 learners)=0.5291\n")

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    print("--- solo models ---")
    solo_models = {
        "GBR solo (reference)": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "LightGBM solo (reference)": lgbm(),
        "CatBoost(depth=6, l2=3)": catboost(6, 3.0),
        "CatBoost(depth=4, l2=3)": catboost(4, 3.0),
        "CatBoost(depth=6, l2=10)": catboost(6, 10.0),
    }
    for name, model in solo_models.items():
        pipe = Pipeline([("prep", build_shared_prep()), ("select_model", build_base_estimator(model))])
        scores = cross_val_score(pipe, X, y, cv=kf, scoring="r2")
        print(f"{name:>36}: R^2 = {scores.mean():.4f} +/- {scores.std():.4f}  (folds: {np.round(scores, 3)})")

    print("\n--- stacks ---")
    stack_configs = {
        "GBR+LightGBM+Ridge+KNN (current stack)": None,
        "+ CatBoost(depth=6, l2=3)": catboost(6, 3.0),
        "+ CatBoost(depth=4, l2=3)": catboost(4, 3.0),
    }
    stack_results = {}
    for name, extra in stack_configs.items():
        pipe = Pipeline([("prep", build_shared_prep()), ("stack", build_stack(extra))])
        scores = cross_val_score(pipe, X, y, cv=kf, scoring="r2", n_jobs=1)
        stack_results[name] = scores.mean()
        print(f"{name:>42}: R^2 = {scores.mean():.4f} +/- {scores.std():.4f}  (folds: {np.round(scores, 3)})")

    best = max(stack_results, key=stack_results.get)
    print(f"\nBest stack: {best} (R^2 = {stack_results[best]:.4f})")


if __name__ == "__main__":
    main()
