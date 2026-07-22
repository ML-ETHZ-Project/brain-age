"""Lane 2: diverse stacking ensemble, proper nested leak-free stacking.

Uses the winning preprocessing from src/advanced_pipeline.py (imported, not
copied/re-derived): TopCorrMixedImputer(n_fancy=40) -> RobustScaler ->
VarianceThreshold -> SelectKBest(f_regression, k=200).

On top of that shared preprocessing, trains 10 diverse base learners (Ridge,
ElasticNet, SVR-rbf, KNN, RandomForest, ExtraTrees, HistGradientBoosting,
XGBoost (existing tuned XGB_PARAMS), LightGBM, CatBoost) and combines them via
a meta-learner (RidgeCV and ElasticNetCV both tried).

NESTED LEAK-FREE STACKING RECIPE (manual per-fold loop, not
Pipeline+cross_val_score, because the nested two-level CV + extracting each
base learner's own held-out score is genuinely custom logic that doesn't fit
as plain Pipeline steps -- see CLAUDE.md's "use a manual per-fold loop ... when
you need custom/nested logic"):

  Outer loop: KFold(5, shuffle=True, random_state=42) on the full 1212 rows.
  For each outer fold (outer-train / outer-test):
    1. Inner loop: KFold(5, shuffle=True, random_state=42) restricted to the
       outer-train rows ONLY. For each inner fold, fit the preprocessing
       pipeline on the inner-train rows, transform inner-train/inner-val,
       fit all 10 base learners on inner-train, predict inner-val. Stitching
       the 5 inner folds' predictions back together gives an out-of-fold
       (OOF) prediction matrix covering every outer-train row exactly once,
       where no row is ever predicted by a model that trained on it.
    2. Fit RidgeCV and ElasticNetCV meta-learners on (OOF matrix, y_outer_train).
       This is the only step that "sees" the outer-train rows' predictions,
       and it never sees outer-test rows at all.
    3. Refit the preprocessing pipeline and all 10 base learners on the FULL
       outer-train rows (still never touching outer-test), predict outer-test
       with each base learner -> test meta-feature matrix.
    4. Feed the test meta-feature matrix through the step-2 meta-learners to
       get the stacked prediction; score R^2 against outer-test y. Also score
       each base learner's own outer-test prediction (from its full-outer-train
       refit in step 3) for the "best single base learner" comparison -- this
       is computed over the identical 5 outer folds as the stacked score, and
       for the XGB_tuned learner is mathematically the same computation as
       src/advanced_pipeline.py's winning pipeline (same KFold seed, same
       preprocessing, same XGB_PARAMS), so it doubles as an internal sanity
       check that should reproduce ~0.5518.

Reusing one inner-OOF pass per outer fold to fit BOTH meta-learners (rather
than e.g. wrapping sklearn's StackingRegressor twice, once per final_estimator
choice) avoids paying for the expensive inner 5-fold x 10-model computation
twice -- it's already the dominant cost of this script.
"""
import os
import sys
import time
import warnings

import numpy as np
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, ElasticNetCV, Ridge, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from advanced_pipeline import load_data, build_preprocessor, XGB_PARAMS  # noqa: E402

RANDOM_STATE = 42
N_OUTER = 5
N_INNER = 5
CURRENT_BEST_R2 = 0.5518


def make_base_learners():
    """Fresh, unfitted estimator instances -- called anew for every fold so no
    fitted state leaks across folds. Ridge/ElasticNet/KNN use simple, fixed
    "reasonable default" hyperparameters (not tuned); RandomForest/ExtraTrees/
    LightGBM/CatBoost use n_estimators/iterations bumped a bit past library
    defaults for stability while keeping per-fit cost bounded (~10-15 min
    total budget across 5 outer x (5 inner + 1 full-refit) x 10 models = 300
    model fits); XGBoost uses the project's existing tuned XGB_PARAMS as-is."""
    return {
        "Ridge": lambda: Ridge(alpha=10.0, random_state=RANDOM_STATE),
        "ElasticNet": lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000, random_state=RANDOM_STATE),
        "SVR_rbf": lambda: SVR(kernel="rbf"),
        "KNN": lambda: KNeighborsRegressor(n_neighbors=10),
        "RandomForest": lambda: RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "ExtraTrees": lambda: ExtraTreesRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "HistGB": lambda: HistGradientBoostingRegressor(random_state=RANDOM_STATE),
        "XGB_tuned": lambda: XGBRegressor(**XGB_PARAMS),
        "LGBM": lambda: LGBMRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
        "CatBoost": lambda: CatBoostRegressor(
            iterations=300, depth=6, learning_rate=0.05, random_state=RANDOM_STATE,
            verbose=False, thread_count=-1,
        ),
    }


def generate_inner_oof(X_ot_raw, y_ot, learner_names, learner_factories):
    """Inner 5-fold CV restricted to this outer fold's training rows only.
    Returns an (n_outer_train, n_learners) OOF prediction matrix -- every row
    predicted exactly once, by a model that never trained on it."""
    n_ot = X_ot_raw.shape[0]
    n_models = len(learner_names)
    oof = np.full((n_ot, n_models), np.nan)

    inner_kf = KFold(n_splits=N_INNER, shuffle=True, random_state=RANDOM_STATE)
    for inner_train_idx, inner_val_idx in inner_kf.split(X_ot_raw):
        X_it_raw, y_it = X_ot_raw[inner_train_idx], y_ot[inner_train_idx]
        X_iv_raw = X_ot_raw[inner_val_idx]

        prep = build_preprocessor()
        X_it = prep.fit_transform(X_it_raw, y_it)
        X_iv = prep.transform(X_iv_raw)

        for j, name in enumerate(learner_names):
            model = learner_factories[name]()
            model.fit(X_it, y_it)
            oof[inner_val_idx, j] = model.predict(X_iv)

    assert not np.isnan(oof).any(), "every outer-train row must get exactly one inner-fold OOF prediction"
    return oof


def refit_on_full_outer_train(X_ot_raw, y_ot, X_te_raw, learner_names, learner_factories):
    """Refit preprocessing + all base learners on the FULL outer-train rows
    (never touching outer-test), predict outer-test with each -> test
    meta-feature matrix, shape (n_outer_test, n_learners)."""
    prep = build_preprocessor()
    X_ot = prep.fit_transform(X_ot_raw, y_ot)
    X_te = prep.transform(X_te_raw)

    n_te = X_te_raw.shape[0]
    n_models = len(learner_names)
    test_matrix = np.empty((n_te, n_models))
    for j, name in enumerate(learner_names):
        model = learner_factories[name]()
        model.fit(X_ot, y_ot)
        test_matrix[:, j] = model.predict(X_te)
    return test_matrix


def main():
    t_start = time.time()
    X, y, train_ids, X_test, test_ids = load_data()
    print(f"Loaded X {X.shape}")
    print(f"Current best to beat: R^2 = {CURRENT_BEST_R2}")

    learners = make_base_learners()
    learner_names = list(learners.keys())
    n_models = len(learner_names)
    print(f"Base learners ({n_models}): {learner_names}\n")

    outer_kf = KFold(n_splits=N_OUTER, shuffle=True, random_state=RANDOM_STATE)

    base_learner_scores = {name: [] for name in learner_names}
    ridge_stack_scores = []
    en_stack_scores = []

    for fold_i, (outer_train_idx, outer_test_idx) in enumerate(outer_kf.split(X)):
        t_fold = time.time()
        X_ot_raw, y_ot = X[outer_train_idx], y[outer_train_idx]
        X_te_raw, y_te = X[outer_test_idx], y[outer_test_idx]

        # Step 1: leak-free inner-OOF predictions on outer-train rows only.
        oof_matrix = generate_inner_oof(X_ot_raw, y_ot, learner_names, learners)

        # Step 2: fit meta-learners on the OOF matrix (outer-train only).
        ridge_meta = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(oof_matrix, y_ot)
        en_meta = ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0],
            cv=5, max_iter=5000, random_state=RANDOM_STATE,
        ).fit(oof_matrix, y_ot)

        # Step 3: refit base learners on FULL outer-train, predict outer-test.
        test_matrix = refit_on_full_outer_train(X_ot_raw, y_ot, X_te_raw, learner_names, learners)

        # Step 4: score the stack, and each base learner standalone, on outer-test.
        ridge_pred = ridge_meta.predict(test_matrix)
        en_pred = en_meta.predict(test_matrix)
        ridge_r2 = r2_score(y_te, ridge_pred)
        en_r2 = r2_score(y_te, en_pred)
        ridge_stack_scores.append(ridge_r2)
        en_stack_scores.append(en_r2)

        fold_base_r2 = {}
        for j, name in enumerate(learner_names):
            r2 = r2_score(y_te, test_matrix[:, j])
            base_learner_scores[name].append(r2)
            fold_base_r2[name] = r2

        elapsed = time.time() - t_fold
        best_base_this_fold = max(fold_base_r2, key=fold_base_r2.get)
        print(f"Outer fold {fold_i + 1}/{N_OUTER} ({elapsed:.1f}s): "
              f"stack(Ridge)={ridge_r2:.4f}  stack(ElasticNet)={en_r2:.4f}  "
              f"best base learner={best_base_this_fold}({fold_base_r2[best_base_this_fold]:.4f})")
        print(f"    per-base-learner R^2: " +
              ", ".join(f"{name}={fold_base_r2[name]:.4f}" for name in learner_names))

    print(f"\nTotal fold loop time: {time.time() - t_start:.1f}s\n")

    print("=== Summary: base learners (mean +/- std over 5 outer folds) ===")
    mean_base = {}
    for name in learner_names:
        scores = np.array(base_learner_scores[name])
        mean_base[name] = scores.mean()
        print(f"  {name:>14}: R^2 = {scores.mean():.4f} +/- {scores.std():.4f}  (folds: {np.round(scores, 4)})")

    best_base_name = max(mean_base, key=mean_base.get)
    best_base_r2 = mean_base[best_base_name]

    ridge_stack_arr = np.array(ridge_stack_scores)
    en_stack_arr = np.array(en_stack_scores)
    print("\n=== Summary: stacked ensemble (mean +/- std over 5 outer folds) ===")
    print(f"  Stack + RidgeCV meta:       R^2 = {ridge_stack_arr.mean():.4f} +/- {ridge_stack_arr.std():.4f}  "
          f"(folds: {np.round(ridge_stack_arr, 4)})")
    print(f"  Stack + ElasticNetCV meta:  R^2 = {en_stack_arr.mean():.4f} +/- {en_stack_arr.std():.4f}  "
          f"(folds: {np.round(en_stack_arr, 4)})")

    best_stack_name = "RidgeCV" if ridge_stack_arr.mean() >= en_stack_arr.mean() else "ElasticNetCV"
    best_stack_r2 = max(ridge_stack_arr.mean(), en_stack_arr.mean())
    best_stack_std = ridge_stack_arr.std() if best_stack_name == "RidgeCV" else en_stack_arr.std()

    print("\n=== Final comparison ===")
    print(f"  Current best (src/advanced_pipeline.py, single XGBoost + combined preproc): R^2 = {CURRENT_BEST_R2}")
    print(f"  Best single base learner here ({best_base_name}):                            R^2 = {best_base_r2:.4f}")
    print(f"  Best stacked ensemble (meta={best_stack_name}):                                R^2 = {best_stack_r2:.4f} +/- {best_stack_std:.4f}")
    print(f"  Note: XGB_tuned base learner here uses identical KFold(5, shuffle, seed=42), "
          f"preprocessing, and XGB_PARAMS as the current-best pipeline, so its score "
          f"({mean_base['XGB_tuned']:.4f}) is an internal sanity check -- it should closely "
          f"reproduce {CURRENT_BEST_R2}.")

    stack_vs_current = best_stack_r2 - CURRENT_BEST_R2
    stack_vs_best_base = best_stack_r2 - best_base_r2
    print(f"\n  Stacked ensemble vs current best:        {stack_vs_current:+.4f}  "
          f"({'BEATS' if stack_vs_current > 0 else 'does not beat'} current best)")
    print(f"  Stacked ensemble vs best single base learner: {stack_vs_best_base:+.4f}  "
          f"({'BEATS' if stack_vs_best_base > 0 else 'does not beat'} best single base learner)")

    print(f"\nTotal runtime: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
