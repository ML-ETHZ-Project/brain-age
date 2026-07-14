"""Fold tuned tree models into the stacking ensemble (builds on
notebooks/stacking_comparison.py, which stacked *default* base learners for
R^2=0.5179).

Plan: tune the two gradient-boosting families (HistGradientBoosting, the
baseline's GradientBoosting) -- including k, the SelectKBest size -- then use
the tuned pipelines as base learners in a StackingRegressor alongside Ridge
and KNN, and measure the stack leak-free.

Each base learner is its OWN full pipeline (impute -> scale -> clip ->
variance -> SelectKBest -> model), so the stack cross-fits complete pipelines
and the outer cross_val_score refits everything per fold -> model weights are
leak-free. HONESTY CAVEAT: the tree hyperparameters are selected once on the
full training set (RandomizedSearchCV below), so those *values* saw every row.
The final stack CV number is therefore mildly optimistic from
hyperparameter-selection bias; a nested CV (tuning inside each outer fold)
would be the fully unbiased confirmation and is the natural follow-up. We use
a different fold seed for the final stack evaluation than for tuning to at
least avoid rewarding fold-boundary coincidences.
"""
import warnings

import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    StackingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsRegressor

from stacking_comparison import (
    K_BEST,
    N_SPLITS,
    RANDOM_STATE,
    load_data,
    make_pipeline,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

K_CHOICES = [50, 75, 100, 150, 200, "all"]


def hist_space():
    return {
        "select__k": K_CHOICES,
        "model__learning_rate": loguniform(1e-2, 3e-1),
        "model__max_iter": randint(200, 600),
        "model__max_leaf_nodes": [15, 31, 63],
        "model__max_depth": [None, 3, 5, 8],
        "model__min_samples_leaf": randint(10, 50),
        "model__l2_regularization": loguniform(1e-3, 1e1),
    }


def gbr_space():
    return {
        "select__k": K_CHOICES,
        "model__learning_rate": loguniform(1e-2, 2e-1),
        "model__n_estimators": randint(150, 400),
        "model__max_depth": [2, 3, 4],
        "model__subsample": uniform(0.6, 0.4),   # -> [0.6, 1.0)
        "model__min_samples_leaf": randint(1, 30),
        "model__max_features": uniform(0.2, 0.6),  # -> [0.2, 0.8)
    }


def tune(name, model, space, n_iter, X, y):
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        make_pipeline(model),
        param_distributions=space,
        n_iter=n_iter,
        scoring="r2",
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        error_score="raise",
    )
    search.fit(X, y)
    print(f"  {name}: tuned CV R^2 = {search.best_score_:.4f}  "
          f"(k={search.best_params_['select__k']}, "
          f"{n_iter} configs)")
    return search.best_estimator_, search.best_score_


def score_stack(name, stack, X, y):
    # Different seed than tuning so we don't reward fold-boundary luck.
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE + 1)
    scores = cross_val_score(stack, X, y, cv=cv, scoring="r2", n_jobs=-1)
    print(f"{name:>26}: R^2 = {scores.mean():.4f} +/- {scores.std():.4f}  "
          f"(folds: {np.round(scores, 3)})")
    return scores.mean()


def main():
    X, y = load_data()
    print(f"X {X.shape}   to beat: default-stack R^2 = 0.5179, single-model 0.5065\n")

    print("--- tuning tree base learners (leak-free per-fold scoring) ---")
    tuned_hist, hist_cv = tune("HistGradientBoosting", HistGradientBoostingRegressor(
        random_state=RANDOM_STATE), hist_space(), n_iter=30, X=X, y=y)
    tuned_gbr, gbr_cv = tune("GradientBoosting", GradientBoostingRegressor(
        random_state=RANDOM_STATE), gbr_space(), n_iter=20, X=X, y=y)

    ridge_pipe = make_pipeline(Ridge(alpha=10.0))
    knn_pipe = make_pipeline(KNeighborsRegressor(n_neighbors=15, weights="distance"))

    print("\n--- stacks (Ridge meta-learner; hyperparams fixed from full-train tuning) ---")
    results = {}

    # Tuned trees only (two strong but correlated learners).
    results["stack[tuned trees]"] = score_stack(
        "stack[tuned trees]",
        StackingRegressor(
            estimators=[("histgbm", tuned_hist), ("gbr", tuned_gbr)],
            final_estimator=Ridge(alpha=1.0), cv=N_SPLITS, n_jobs=-1),
        X, y)

    # Tuned trees + diversity (Ridge linear, KNN distance-based).
    results["stack[tuned+ridge+knn]"] = score_stack(
        "stack[tuned+ridge+knn]",
        StackingRegressor(
            estimators=[("histgbm", tuned_hist), ("gbr", tuned_gbr),
                        ("ridge", ridge_pipe), ("knn", knn_pipe)],
            final_estimator=Ridge(alpha=1.0), cv=N_SPLITS, n_jobs=-1),
        X, y)

    print(f"\nTuned single models: HistGBM {hist_cv:.4f}, GBR {gbr_cv:.4f}")
    best = max(results, key=results.get)
    print(f"Best stack: {best} (R^2 = {results[best]:.4f})")
    print("(Reminder: stack R^2 is mildly optimistic -- tree hyperparameters were "
          "selected on the full training set; nested CV is the unbiased follow-up.)")


if __name__ == "__main__":
    main()
