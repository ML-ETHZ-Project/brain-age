"""Baseline pipeline for the Brain Age Prediction Kaggle competition.

Stages: median-impute -> robust-scale -> +/-15-IQR clip -> variance filter
-> correlation-pruned K-best feature selection -> StackingRegressor. Reports
5-fold CV R^2 (solo GradientBoosting vs. the stack), refits the stack on the
full training set, and writes submissions/submission.csv.

Outlier removal (subtask 1) is deliberately NOT used to filter the
regressor's training data: notebooks/outlier_comparison.py validated it
leak-free (fit the detector inside each CV fold, on training rows only)
and found it *hurts* R^2 at every contamination level tried (0.44-0.49 vs
0.5065 with no removal) -- tree-based models are already robust to
outliers, and dropping rows just loses training signal. We still produce
the required outlier classification as a standalone artifact
(data/processed/outlier_labels.csv) since the subtask asks for a
classification of training samples, not that they must be dropped.

Feature selection uses CorrelationPrunedKBest instead of a plain
SelectKBest: notebooks/feature_visualization.ipynb found several of the
top age-correlated features are highly inter-correlated (e.g. x133/x334/
x465, all |r| > 0.9), so SelectKBest's independent per-feature ranking
spends part of its k=100 budget on near-duplicates.
notebooks/feature_selection_corr_cutoff.py validated leak-free (selector
refit inside each CV fold) that swapping those redundant picks for the
next-best distinct ones -- while keeping the feature count at 100, not
cutting it -- improves CV R^2 from 0.5061 to 0.5141.

The model is a StackingRegressor (GradientBoosting + LightGBM + CatBoost +
Ridge(alpha=10) + KNN(k=15, distance) -> Ridge meta-learner). This combines
several independent, leak-free-validated improvements:
  - CorrelationPrunedKBest (this repo) instead of SelectKBest.
  - Stacking GBR+Ridge+KNN (teammate Philippe's philippe/ensembling branch,
    notebooks/stacking_comparison.py there) instead of a single model --
    swapping that stack's SelectKBest for CorrelationPrunedKBest
    (notebooks/stacking_with_corr_pruned.py) reached CV R^2=0.5229.
  - Adding LightGBM as a 4th base learner (notebooks/lightgbm_experiment.py):
    LightGBM's leaf-wise tree growth isn't individually stronger than GBR's
    level-wise growth here (solo LightGBM: 0.5106 vs. solo GBR: 0.5141), but
    it's different enough to add real diversity to the stack, lifting CV
    R^2 to 0.5291, confirmed on the public leaderboard (0.6645, up from
    0.6528 for the 3-learner stack, moving in the same direction as the CV
    gain).
  - Adding CatBoost as a 5th base learner (notebooks/catboost_experiment.py):
    CatBoost's ordered boosting + oblivious trees is individually the
    strongest single model tried yet (solo CatBoost(depth=6, l2_leaf_reg=3):
    0.5391 vs. solo GBR: 0.5141), and adding it to the stack lifts CV R^2
    to 0.5379 -- the current best.
Each base learner is its own full pipeline (impute/scale/clip/variance/
select/model) and StackingRegressor cross-fits them internally (cv=5) to
build out-of-fold meta-features, so the meta-learner never trains on a
base learner's in-sample predictions; the outer cross_val_score refits
everything (including every base learner's preprocessing) per fold.
The +/-15-IQR post-scale clip guards f_regression/correlation computations
inside CorrelationPrunedKBest against the injected noise columns (std
~1e22 even after RobustScaler).

Several other approaches were tried and rejected (see the README experiment
log for details, all leak-free CV-validated): an unsupervised sparse
autoencoder bottleneck and a denoising autoencoder pre-processing step
(both src/rejected_sparse_autoencoder.py / src/rejected_denoising_autoencoder.py) lost badly
to CorrelationPrunedKBest's supervised ranking; plain PCA likewise;
a from-scratch neural-network regressor (src/rejected_neural_net_regressor.py)
never caught up to GradientBoosting even after seed-ensembling; blending
our stack's predictions with a weaker external stack didn't beat the
stronger model alone. None of those are used here.

Needs the OpenMP runtime (libomp) for LightGBM's compiled extension, which
isn't installed system-wide on this machine (no Homebrew available).
scikit-learn happens to bundle its own libomp.dylib; the block below
re-execs this script once with DYLD_LIBRARY_PATH pointed at that bundled
copy before anything imports lightgbm, so `python src/baseline.py` works
standalone with no manual setup or Homebrew install required.
"""
import os
import sys

if not os.environ.get("_LIGHTGBM_LIBOMP_REEXEC"):
    _venv_lib = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".venv", "lib")
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
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest, StackingRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import RobustScaler

from feature_selection import CorrelationPrunedKBest

RANDOM_STATE = 42
N_SPLITS = 5
K_BEST = 100
CORR_THRESHOLD = 0.9
CLIP_IQR = 15.0
OUTLIER_CONTAMINATION = 0.05

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data", "raw")
OUTPUT_DIR = os.path.join(REPO_ROOT, "submissions")
PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")


def load_data():
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"))
    X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))

    feat_cols = [c for c in X_train.columns if c != "id"]
    return (
        X_train[feat_cols].values,
        y_train["y"].values,
        X_train["id"].values,
        X_test[feat_cols].values,
        X_test["id"].values,
    )


def build_preprocessing_steps():
    return [
        ("impute", SimpleImputer(strategy="median")),
        ("scale", RobustScaler()),
        ("clip", FunctionTransformer(np.clip, kw_args={"a_min": -CLIP_IQR, "a_max": CLIP_IQR})),
        ("var_thresh", VarianceThreshold(threshold=1e-8)),  # drop constant/near-constant features
        ("select", CorrelationPrunedKBest(n_features=K_BEST, corr_threshold=CORR_THRESHOLD)),
    ]


def build_preprocessor():
    return Pipeline(build_preprocessing_steps())


def build_base_pipeline(model):
    return Pipeline(build_preprocessing_steps() + [("model", model)])


def build_stack():
    estimators = [
        ("gbr", build_base_pipeline(GradientBoostingRegressor(random_state=RANDOM_STATE))),
        ("ridge", build_base_pipeline(Ridge(alpha=10.0))),
        ("knn", build_base_pipeline(KNeighborsRegressor(n_neighbors=15, weights="distance"))),
        ("lgbm", build_base_pipeline(
            LGBMRegressor(num_leaves=15, min_child_samples=30, random_state=RANDOM_STATE, verbosity=-1))),
        ("catboost", build_base_pipeline(
            CatBoostRegressor(depth=6, l2_leaf_reg=3.0, random_state=RANDOM_STATE,
                               verbose=False, allow_writing_files=False))),
    ]
    return StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=1.0), cv=N_SPLITS, n_jobs=-1)


def classify_outliers(X_proc, train_ids, contamination=OUTLIER_CONTAMINATION):
    """Subtask 1 deliverable: classify each training row as outlier/inlier.

    Fit on the full preprocessed training set (fine here since this is a
    reporting artifact, not something used to filter data fed to the
    regressor -- see module docstring for why removal isn't used).
    """
    iso = IsolationForest(contamination=contamination, random_state=RANDOM_STATE)
    is_outlier = iso.fit_predict(X_proc) == -1
    print(f"  Outlier classification: flagged {is_outlier.sum()} / {len(train_ids)} training rows as outliers")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_path = os.path.join(PROCESSED_DIR, "outlier_labels.csv")
    pd.DataFrame({"id": train_ids, "is_outlier": is_outlier.astype(int)}).to_csv(out_path, index=False)
    print(f"  Wrote {out_path}")
    return is_outlier


def main():
    X_train, y_train, train_ids, X_test, test_ids = load_data()
    print(f"Loaded: X_train {X_train.shape}, X_test {X_test.shape}")

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    print("\n--- Cross-validated R^2 (leak-free: preprocessing + base learners fit per-fold) ---")
    gbr_solo = build_base_pipeline(GradientBoostingRegressor(random_state=RANDOM_STATE))
    solo_scores = cross_val_score(gbr_solo, X_train, y_train, cv=kf, scoring="r2")
    print(f"  GradientBoosting solo: R^2 = {solo_scores.mean():.4f} +/- {solo_scores.std():.4f}  "
          f"(folds: {np.round(solo_scores, 3)})")

    stack_scores = cross_val_score(build_stack(), X_train, y_train, cv=kf, scoring="r2", n_jobs=1)
    print(f"  StackingRegressor (GBR+LightGBM+CatBoost+Ridge+KNN -> Ridge): R^2 = {stack_scores.mean():.4f} "
          f"+/- {stack_scores.std():.4f}  (folds: {np.round(stack_scores, 3)})")
    print(
        "(Combines this branch's CorrelationPrunedKBest, philippe/ensembling's stacking ensemble, "
        "LightGBM as a 4th base learner, and CatBoost as a 5th -- see module docstring and the README "
        "experiment log for the leak-free ablations behind each choice. Outlier removal was tried and "
        "validated leak-free in notebooks/outlier_comparison.py: it hurts R^2 at every contamination "
        "level, so it is NOT used to filter training data here.)"
    )

    # Subtask 1 deliverable: classify (not remove) training-row outliers, using the same
    # preprocessing (fit on the full training set -- fine for a reporting artifact).
    print("\n--- Subtask 1: outlier classification (reporting only, not used for training) ---")
    X_train_preprocessed = build_preprocessor().fit_transform(X_train, y_train)
    classify_outliers(X_train_preprocessed, train_ids)

    # Final fit on all available training data (each base learner handles its own
    # preprocessing internally), predict on the raw test set.
    final_model = build_stack()
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)

    train_r2 = r2_score(y_train, final_model.predict(X_train))
    print(f"\nFinal model train R^2 (in-sample, optimistic): {train_r2:.4f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    submission = pd.DataFrame({"id": test_ids, "y": y_pred})
    out_path = os.path.join(OUTPUT_DIR, "submission.csv")
    submission.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({submission.shape[0]} rows)")


if __name__ == "__main__":
    main()
