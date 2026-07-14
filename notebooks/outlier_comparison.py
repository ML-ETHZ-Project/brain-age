"""Subtask 1: outlier detection, validated leak-free.

src/baseline.py's original comparison fit the preprocessor (including
SelectKBest, which looks at y) and the IsolationForest on the *entire*
training set before splitting into CV folds -- that leaks validation-fold
information into feature selection and outlier detection, inflating the
"with outlier removal" score relative to the "without" baseline (which
correctly fits everything inside cross_val_score per fold).

This script instead runs a manual K-fold loop where, for every fold:
  1. impute/scale/variance-filter/select-K-best is fit on the fold's
     training rows only, then applied to both training and validation rows
  2. the outlier detector is fit on the fold's (preprocessed) training rows
     only, and only training rows are filtered -- validation rows are
     always scored in full, since in the real test set we don't get to
     drop "hard" subjects
  3. the regressor is fit on the filtered training rows and scored (R^2)
     against the untouched validation rows

Sweeps IsolationForest contamination and a LocalOutlierFactor baseline
against a no-removal control, all through the same leak-free loop.
"""
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

RANDOM_STATE = 42
N_SPLITS = 5
K_BEST = 100

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
        ("select", SelectKBest(f_regression, k=K_BEST)),
    ])


def make_detector_none(_X):
    return None


def make_detector_iforest(contamination):
    def _fit(X):
        det = IsolationForest(contamination=contamination, random_state=RANDOM_STATE)
        return det.fit_predict(X) == 1  # True = inlier
    return _fit


def make_detector_lof(contamination, n_neighbors=20):
    def _fit(X):
        det = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        return det.fit_predict(X) == 1
    return _fit


def evaluate(name, detector_fn, X, y):
    """Leak-free nested CV: preprocessing, outlier detection, and model
    fitting all happen inside each fold using only that fold's training rows."""
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_scores = []
    n_flagged_total, n_train_total = 0, 0

    for train_idx, val_idx in kf.split(X):
        X_tr_raw, X_val_raw = X[train_idx], X[val_idx]
        y_tr_raw, y_val = y[train_idx], y[val_idx]

        prep = build_preprocessor()
        X_tr = prep.fit_transform(X_tr_raw, y_tr_raw)
        X_val = prep.transform(X_val_raw)

        if detector_fn is None:
            X_tr_clean, y_tr_clean = X_tr, y_tr_raw
        else:
            is_inlier = detector_fn(X_tr)
            X_tr_clean, y_tr_clean = X_tr[is_inlier], y_tr_raw[is_inlier]
            n_flagged_total += (~is_inlier).sum()
            n_train_total += len(is_inlier)

        model = GradientBoostingRegressor(random_state=RANDOM_STATE)
        model.fit(X_tr_clean, y_tr_clean)
        y_pred = model.predict(X_val)
        fold_scores.append(r2_score(y_val, y_pred))

    fold_scores = np.array(fold_scores)
    flag_info = f"  (flagged {n_flagged_total}/{n_train_total} train rows across folds)" if detector_fn else ""
    print(f"{name:>28}: R^2 = {fold_scores.mean():.4f} +/- {fold_scores.std():.4f}  "
          f"(folds: {np.round(fold_scores, 3)}){flag_info}")
    return fold_scores.mean()


def main():
    X, y = load_data()
    print(f"X {X.shape}\n")

    results = {}
    results["no outlier removal"] = evaluate("no outlier removal", None, X, y)
    for c in [0.02, 0.05, 0.08, 0.12]:
        results[f"IsolationForest(c={c})"] = evaluate(
            f"IsolationForest(c={c})", make_detector_iforest(c), X, y
        )
    results["LOF(c=0.05, k=20)"] = evaluate(
        "LOF(c=0.05, k=20)", make_detector_lof(0.05), X, y
    )

    best = max(results, key=results.get)
    print(f"\nBest: {best} (R^2 = {results[best]:.4f})")
    print(f"(no-removal control: R^2 = {results['no outlier removal']:.4f})")


if __name__ == "__main__":
    main()
