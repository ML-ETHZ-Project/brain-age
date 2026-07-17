"""Leak-free CV score for the blend submitted as submissions/submission_blend.csv.

That submission averaged two already-fit models' *test-set* predictions
(our stack x Philippe's submission_stack.csv) with no CV of the blend
itself -- we only ever saw its public LB score (0.6730). This fills that
gap: a proper leak-free CV estimate of "blend of our stack + Philippe's
stack," using our stack pre-LightGBM (CorrelationPrunedKBest + GBR+Ridge+
KNN -> Ridge meta, R^2=0.5229, the one actually used in that blend) and
Philippe's *default* stacking_comparison.py stack (SelectKBest(100) +
GBR+Ridge+KNN -> Ridge meta, R^2=0.5179 as he reported, 0.5175 as
reproduced here) -- not his RandomizedSearchCV-tuned variant, since tuning
would need to rerun inside every outer fold to stay leak-free, which is
expensive; the default and tuned stacks are close in CV (0.5179 vs 0.5222)
so this is a fair stand-in, not a real gap in the comparison.

Both full pipelines (preprocessing + stack) are fit independently on each
outer fold's training rows only, predictions are averaged 50/50 on that
fold's held-out rows, and R^2 is scored there -- so, unlike the actual
submission, this number is a genuine leak-free CV estimate of the blend,
comparable to every other CV number in the README log.
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
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
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


# --- our stack (pre-LightGBM): CorrelationPrunedKBest + GBR/Ridge/KNN -> Ridge ---

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


def build_ours():
    estimators = [
        ("gbr", build_base_estimator(GradientBoostingRegressor(random_state=RANDOM_STATE))),
        ("ridge", build_base_estimator(Ridge(alpha=10.0))),
        ("knn", build_base_estimator(KNeighborsRegressor(n_neighbors=15, weights="distance"))),
    ]
    stack = StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=1.0), cv=N_SPLITS, n_jobs=-1)
    return Pipeline([("prep", build_shared_prep()), ("stack", stack)])


# --- Philippe's default stack (stacking_comparison.py on philippe/ensembling): SelectKBest + GBR/Ridge/KNN -> Ridge ---

def build_philippe_default():
    def philippe_steps():
        return [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", RobustScaler()),
            ("clip", FunctionTransformer(np.clip, kw_args={"a_min": -CLIP_IQR, "a_max": CLIP_IQR})),
            ("var_thresh", VarianceThreshold(threshold=1e-8)),
            ("select", SelectKBest(f_regression, k=K_BEST)),
        ]

    estimators = [
        ("gbr", Pipeline(philippe_steps() + [("model", GradientBoostingRegressor(random_state=RANDOM_STATE))])),
        ("ridge", Pipeline(philippe_steps() + [("model", Ridge(alpha=10.0))])),
        ("knn", Pipeline(philippe_steps() + [("model", KNeighborsRegressor(n_neighbors=15, weights="distance"))])),
    ]
    return StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=1.0), cv=N_SPLITS, n_jobs=-1)


def main():
    X, y = load_data()
    print(f"X {X.shape}\n")

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    ours_scores, philippe_scores, blend_scores = [], [], []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        ours = build_ours()
        ours.fit(X_tr, y_tr)
        pred_ours = ours.predict(X_val)

        philippe = build_philippe_default()
        philippe.fit(X_tr, y_tr)
        pred_philippe = philippe.predict(X_val)

        pred_blend = (pred_ours + pred_philippe) / 2

        r2_ours = r2_score(y_val, pred_ours)
        r2_philippe = r2_score(y_val, pred_philippe)
        r2_blend = r2_score(y_val, pred_blend)

        ours_scores.append(r2_ours)
        philippe_scores.append(r2_philippe)
        blend_scores.append(r2_blend)
        print(f"fold {fold_idx}: ours={r2_ours:.4f}  philippe={r2_philippe:.4f}  blend={r2_blend:.4f}")

    def summarize(name, scores):
        scores = np.array(scores)
        print(f"{name:>12}: R^2 = {scores.mean():.4f} +/- {scores.std():.4f}  (folds: {np.round(scores, 3)})")

    print()
    summarize("ours", ours_scores)
    summarize("philippe", philippe_scores)
    summarize("blend", blend_scores)


if __name__ == "__main__":
    main()
