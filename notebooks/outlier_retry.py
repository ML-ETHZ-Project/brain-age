"""Lane A: outlier retry -- methods not yet tested in notebooks/outlier_comparison.py.

Prior leak-free comparison (notebooks/outlier_comparison.py) established:
  no removal:                R^2 = 0.5065  (control)
  IsolationForest (c=0.02-0.12): R^2 = 0.44-0.48  (worse)
  LOF (c=0.05):               R^2 = 0.4928  (worse)
-> hard outlier removal has lost every time it's been tried.

This script tries methods/strategies not yet tested, still leak-free (same
manual per-fold pattern as outlier_comparison.py: preprocessing, outlier
detector, and hyperparameter choice are all fit on fold-train rows only,
then applied/scored on the untouched fold-val rows):

  1. EllipticEnvelope (Mahalanobis-distance outlier detector -- may suit
     correlated anatomical features better than IsolationForest's
     axis-aligned splits).
  2. OneClassSVM (nu in {0.05, 0.1}).
  3. SOFT handling: fit a detector on fold-train, but instead of dropping
     flagged rows, downweight them via sample_weight=0.3 (vs 1.0 for
     inliers) when fitting GradientBoostingRegressor -- keeps the signal,
     reduces the flagged rows' influence.
  4. No removal at all, but a robust loss: GradientBoostingRegressor
     (loss="huber") instead of the default squared-error loss.

All compared against a same-run no-removal control (GBR default loss, no
detector) for an apples-to-apples number alongside the historical 0.5065.
"""
import os

import numpy as np
import pandas as pd
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM

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


# --- detector factories: each returns a fit(X_tr) -> boolean is_inlier array ---

def make_detector_elliptic(contamination):
    def _fit(X):
        det = EllipticEnvelope(contamination=contamination, random_state=RANDOM_STATE)
        return det.fit_predict(X) == 1
    return _fit


def make_detector_ocsvm(nu):
    def _fit(X):
        det = OneClassSVM(nu=nu, kernel="rbf", gamma="scale")
        return det.fit_predict(X) == 1
    return _fit


def evaluate_removal(name, detector_fn, X, y, loss="squared_error"):
    """Hard removal (or no-removal control if detector_fn is None): leak-free
    manual per-fold loop, mirroring outlier_comparison.py."""
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

        model = GradientBoostingRegressor(loss=loss, random_state=RANDOM_STATE)
        model.fit(X_tr_clean, y_tr_clean)
        y_pred = model.predict(X_val)
        fold_scores.append(r2_score(y_val, y_pred))

    fold_scores = np.array(fold_scores)
    flag_info = (f"  (flagged {n_flagged_total}/{n_train_total} train rows across folds)"
                 if detector_fn else "")
    print(f"{name:>32}: R^2 = {fold_scores.mean():.4f} +/- {fold_scores.std():.4f}  "
          f"(folds: {np.round(fold_scores, 3)}){flag_info}")
    return fold_scores.mean(), fold_scores.std()


def evaluate_soft_downweight(name, detector_fn, X, y, flagged_weight=0.3):
    """Soft handling: detector flags rows, but instead of dropping them,
    they're downweighted via sample_weight when fitting GBR. All rows stay
    in training; flagged fold-train rows just count less."""
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_scores = []
    n_flagged_total, n_train_total = 0, 0

    for train_idx, val_idx in kf.split(X):
        X_tr_raw, X_val_raw = X[train_idx], X[val_idx]
        y_tr_raw, y_val = y[train_idx], y[val_idx]

        prep = build_preprocessor()
        X_tr = prep.fit_transform(X_tr_raw, y_tr_raw)
        X_val = prep.transform(X_val_raw)

        is_inlier = detector_fn(X_tr)
        sample_weight = np.where(is_inlier, 1.0, flagged_weight)
        n_flagged_total += (~is_inlier).sum()
        n_train_total += len(is_inlier)

        model = GradientBoostingRegressor(random_state=RANDOM_STATE)
        model.fit(X_tr, y_tr_raw, sample_weight=sample_weight)
        y_pred = model.predict(X_val)
        fold_scores.append(r2_score(y_val, y_pred))

    fold_scores = np.array(fold_scores)
    print(f"{name:>32}: R^2 = {fold_scores.mean():.4f} +/- {fold_scores.std():.4f}  "
          f"(folds: {np.round(fold_scores, 3)})  (flagged {n_flagged_total}/{n_train_total} train rows across folds)")
    return fold_scores.mean(), fold_scores.std()


def main():
    X, y = load_data()
    print(f"X {X.shape}\n")

    results = {}

    print("--- Control (no removal, same run) ---")
    results["no removal (control)"] = evaluate_removal("no removal (control)", None, X, y)

    print("\n--- 1. EllipticEnvelope (Mahalanobis distance) ---")
    for c in [0.05, 0.1]:
        results[f"EllipticEnvelope(c={c})"] = evaluate_removal(
            f"EllipticEnvelope(c={c})", make_detector_elliptic(c), X, y
        )

    print("\n--- 2. OneClassSVM ---")
    for nu in [0.05, 0.1]:
        results[f"OneClassSVM(nu={nu})"] = evaluate_removal(
            f"OneClassSVM(nu={nu})", make_detector_ocsvm(nu), X, y
        )

    print("\n--- 3. Soft downweighting (flagged rows get sample_weight=0.3) ---")
    def _iforest_detector(Xtr):
        det = IsolationForest(contamination=0.05, random_state=RANDOM_STATE)
        return det.fit_predict(Xtr) == 1

    results["soft: IsolationForest(c=0.05) dw=0.3"] = evaluate_soft_downweight(
        "soft: IForest(c=0.05) dw=0.3", _iforest_detector, X, y,
    )
    results["soft: EllipticEnvelope(c=0.05) dw=0.3"] = evaluate_soft_downweight(
        "soft: Elliptic(c=0.05) dw=0.3", make_detector_elliptic(0.05), X, y
    )

    print("\n--- 4. No removal, robust Huber loss ---")
    results["GBR(loss=huber), no removal"] = evaluate_removal(
        "GBR(loss=huber), no removal", None, X, y, loss="huber"
    )

    print("\n=== Summary ===")
    control_r2 = results["no removal (control)"][0]
    for name, (mean, std) in sorted(results.items(), key=lambda kv: -kv[1][0]):
        beats = " <-- beats control" if mean > control_r2 else ""
        print(f"{name:>40}: R^2 = {mean:.4f} +/- {std:.4f}{beats}")

    best_name = max(results, key=lambda k: results[k][0])
    best_mean = results[best_name][0]
    print(f"\nBest: {best_name} (R^2 = {best_mean:.4f})")
    print(f"Control (no removal): R^2 = {control_r2:.4f}")
    print(f"Historical baseline (src/baseline.py): R^2 = 0.5065")
    if best_mean > 0.5065:
        print("-> Beats the 0.5065 baseline.")
    else:
        print("-> Does NOT beat the 0.5065 baseline. Consistent with prior finding: "
              "outlier removal/downweighting/robust-loss does not help on this dataset; "
              "GradientBoostingRegressor with no removal remains the best simple choice.")


if __name__ == "__main__":
    main()
