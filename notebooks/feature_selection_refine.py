"""Lane C: feature selection refinement, validated leak-free.

Baseline (src/baseline.py) uses SelectKBest(f_regression, k=100) inside the
pipeline impute -> RobustScaler -> VarianceThreshold(1e-8) -> SelectKBest ->
GradientBoostingRegressor, scoring 0.5065 (5-fold CV, cross_val_score).

Everything here goes through sklearn Pipeline + cross_val_score with the
rest of the pipeline held fixed and only the "select" step swapped -- this
is automatically leak-free because cross_val_score refits the *entire*
Pipeline (including the selector) on each fold's training rows only, then
transforms/predicts on the held-out fold. No selector or hyperparameter
ever sees validation-fold rows during fitting.

Compared:
  1. SelectKBest(f_regression, k=100)               [current baseline]
  2. SelectKBest(mutual_info_regression, k=100)      [nonlinear relevance]
  3. SelectFromModel(Lasso(alpha=...))                [sparse linear model], alpha swept
  4. SelectFromModel(GradientBoostingRegressor(...))  [tree importance]
  5. k in {50, 100, 150, 200, 300, "all"} for whichever of f_regression /
     mutual_info_regression wins among (1)/(2)

Diagnostic only (does NOT feed back into model fitting or the comparison
above): for the single best-scoring configuration found, re-run a manual
leak-free KFold loop, and within each fold fit
statsmodels.api.OLS(y_fold_train, sm.add_constant(X_fold_train_selected))
on that fold's selector-chosen features (after the same impute/scale/
var_thresh preprocessing, fit on training rows only). This gives Wald
z/t-test p-values per coefficient (model.pvalues). Because the selector is
refit per fold, the *set* of selected features can differ fold to fold;
a feature is reported as "majority significant" if, among the folds in
which it was selected at all, it had p < 0.05 in more than half of them.
This is purely a reporting/diagnostic exercise on feature significance --
it never changes what gets fed to the regressor.
"""
import os
from functools import partial

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import (
    SelectFromModel,
    SelectKBest,
    VarianceThreshold,
    f_regression,
    mutual_info_regression,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

RANDOM_STATE = 42
N_SPLITS = 5
BASELINE_R2 = 0.5065

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data", "raw")


def load_data():
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"))
    feat_cols = [c for c in X_train.columns if c != "id"]
    return X_train[feat_cols].values, y_train["y"].values, np.array(feat_cols)


def build_pipeline(selector):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", RobustScaler()),
        ("var_thresh", VarianceThreshold(threshold=1e-8)),
        ("select", selector),
        ("model", GradientBoostingRegressor(random_state=RANDOM_STATE)),
    ])


def evaluate(name, selector, X, y, results):
    pipe = build_pipeline(selector)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipe, X, y, cv=kf, scoring="r2", n_jobs=1)
    print(f"  {name:>42}: R^2 = {scores.mean():.4f} +/- {scores.std():.4f}  (folds: {np.round(scores, 3)})")
    results[name] = scores.mean()
    return scores.mean()


def main():
    X, y, feat_names = load_data()
    print(f"X {X.shape}, y {y.shape}\n")
    results = {}

    # ------------------------------------------------------------------
    # 1. current baseline vs. mutual_info_regression, both k=100
    # ------------------------------------------------------------------
    print("--- Step 1: scoring function, k=100 fixed ---")
    evaluate("SelectKBest(f_regression, k=100) [baseline]",
              SelectKBest(f_regression, k=100), X, y, results)
    mi_scorer = partial(mutual_info_regression, random_state=RANDOM_STATE)
    evaluate("SelectKBest(mutual_info_regression, k=100)",
              SelectKBest(mi_scorer, k=100), X, y, results)

    winning_scorer_name = ("f_regression"
                            if results["SelectKBest(f_regression, k=100) [baseline]"]
                            >= results["SelectKBest(mutual_info_regression, k=100)"]
                            else "mutual_info_regression")
    print(f"  -> winning scorer for the k-sweep: {winning_scorer_name}\n")

    # ------------------------------------------------------------------
    # 2. model-based selectors: Lasso (alpha swept) and GBR importance
    # ------------------------------------------------------------------
    print("--- Step 2: model-based selectors ---")
    for alpha in [0.001, 0.01, 0.1, 1.0]:
        evaluate(f"SelectFromModel(Lasso(alpha={alpha}))",
                  SelectFromModel(Lasso(alpha=alpha, random_state=RANDOM_STATE, max_iter=20000)),
                  X, y, results)
    evaluate("SelectFromModel(GBR.feature_importances_)",
              SelectFromModel(GradientBoostingRegressor(random_state=RANDOM_STATE)),
              X, y, results)
    print()

    # ------------------------------------------------------------------
    # 3. k sweep for the winning scorer
    # ------------------------------------------------------------------
    print(f"--- Step 3: k sweep for {winning_scorer_name} ---")
    score_func = f_regression if winning_scorer_name == "f_regression" else mi_scorer
    for k in [50, 100, 150, 200, 300, "all"]:
        name = f"SelectKBest({winning_scorer_name}, k={k})"
        if name in results:
            continue  # already evaluated in step 1 (k=100 baseline case)
        evaluate(name, SelectKBest(score_func, k=k), X, y, results)
    print()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("--- Summary (all configurations, sorted) ---")
    for name, r2 in sorted(results.items(), key=lambda kv: -kv[1]):
        flag = "  <-- BEATS baseline" if r2 > BASELINE_R2 else ""
        print(f"  {name:>42}: R^2 = {r2:.4f}{flag}")

    best_name = max(results, key=results.get)
    best_r2 = results[best_name]
    print(f"\nBest overall: {best_name} (R^2 = {best_r2:.4f}); baseline = {BASELINE_R2:.4f}")

    # ------------------------------------------------------------------
    # Diagnostic (not part of the CV comparison): Wald-test p-values from
    # per-fold OLS fits on the best configuration's selected features.
    # ------------------------------------------------------------------
    print("\n--- Diagnostic: per-fold OLS Wald-test p-values on the best selector's features ---")
    best_selector_factory = _selector_factory_for(best_name, score_func, winning_scorer_name)
    run_ols_diagnostic(best_name, best_selector_factory, X, y, feat_names)


def _selector_factory_for(name, score_func, winning_scorer_name):
    """Rebuild a fresh selector instance matching `name`, for the diagnostic loop."""
    name = name.split(" [")[0]  # strip any " [baseline]"-style annotation before parsing
    if name.startswith("SelectKBest(f_regression") or name.startswith("SelectKBest(mutual_info_regression"):
        k_str = name.split("k=")[1].rstrip(")")
        k = k_str if k_str == "all" else int(k_str)
        sf = f_regression if "f_regression" in name else score_func
        return lambda: SelectKBest(sf, k=k)
    if name.startswith("SelectFromModel(Lasso"):
        alpha = float(name.split("alpha=")[1].rstrip("))"))
        return lambda: SelectFromModel(Lasso(alpha=alpha, random_state=RANDOM_STATE, max_iter=20000))
    if name.startswith("SelectFromModel(GBR"):
        return lambda: SelectFromModel(GradientBoostingRegressor(random_state=RANDOM_STATE))
    raise ValueError(f"unrecognized selector name: {name}")


def run_ols_diagnostic(name, selector_factory, X, y, feat_names):
    """Manual leak-free KFold loop: fit impute/scale/var_thresh/selector on
    each fold's training rows only, then fit OLS on the selected features
    to collect Wald p-values. Reporting only -- does not affect any score
    reported above."""
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    pvals_by_feature = {}  # feature name -> list of p-values across folds where selected

    for fold_i, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr_raw, y_tr = X[train_idx], y[train_idx]

        imputer = SimpleImputer(strategy="median")
        X_tr = imputer.fit_transform(X_tr_raw)
        scaler = RobustScaler()
        X_tr = scaler.fit_transform(X_tr)
        vt = VarianceThreshold(threshold=1e-8)
        X_tr = vt.fit_transform(X_tr)
        names_after_vt = feat_names[vt.get_support()]

        selector = selector_factory()
        X_tr_sel = selector.fit_transform(X_tr, y_tr)
        sel_support = selector.get_support()
        selected_names = names_after_vt[sel_support]

        ols = sm.OLS(y_tr, sm.add_constant(X_tr_sel)).fit()
        pvals = ols.pvalues[1:]  # drop intercept
        assert len(pvals) == len(selected_names)
        for fname, p in zip(selected_names, pvals):
            pvals_by_feature.setdefault(fname, []).append(p)

        n_sig = int((pvals < 0.05).sum())
        print(f"  fold {fold_i}: {len(selected_names)} features selected, {n_sig} significant (p<0.05)")

    # majority-significant: appears in >=1 fold, and p<0.05 in more folds than not
    rows = []
    for fname, plist in pvals_by_feature.items():
        n_folds = len(plist)
        n_sig = sum(p < 0.05 for p in plist)
        rows.append((fname, n_folds, n_sig, np.mean(plist)))
    rows.sort(key=lambda r: (-r[2], -r[1], r[3]))

    majority_sig = [r for r in rows if r[2] > r[1] / 2]
    print(f"\n  Selector used for diagnostic: {name}")
    print(f"  {len(pvals_by_feature)} distinct features selected across the 5 folds; "
          f"{len(majority_sig)} are Wald-significant (p<0.05) in a majority of the folds they appeared in.")
    print("\n  Top significant features (by #folds significant, then #folds selected):")
    print(f"  {'feature':>10} {'folds_selected':>15} {'folds_significant':>18} {'mean_p':>10}")
    for fname, n_folds, n_sig, mean_p in rows[:20]:
        tag = "  *" if n_sig > n_folds / 2 else ""
        print(f"  {fname:>10} {n_folds:>15} {n_sig:>18} {mean_p:>10.4f}{tag}")


if __name__ == "__main__":
    main()
