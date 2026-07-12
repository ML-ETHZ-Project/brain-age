"""Baseline pipeline for the Brain Age Prediction Kaggle competition.

Stages: median-impute -> robust-scale -> variance/K-best feature selection
-> IsolationForest outlier removal (train only) -> regressor.
Reports 5-fold CV R^2 for a few candidate regressors, picks the best, refits
on the full (outlier-filtered) training set, and writes submissions/submission.csv.
"""
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

RANDOM_STATE = 42
N_SPLITS = 5
K_BEST = 100

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data", "raw")
OUTPUT_DIR = os.path.join(REPO_ROOT, "submissions")


def load_data():
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"))
    X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))

    feat_cols = [c for c in X_train.columns if c != "id"]
    return X_train[feat_cols].values, y_train["y"].values, X_test[feat_cols].values, X_test["id"].values


def build_preprocessor():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", RobustScaler()),
        ("var_thresh", VarianceThreshold(threshold=1e-8)),  # drop constant/near-constant features
        ("select", SelectKBest(f_regression, k=K_BEST)),
    ])


def remove_outliers(X, y, contamination=0.05):
    """Flag outlier rows in the training set with IsolationForest; return filtered X, y."""
    iso = IsolationForest(contamination=contamination, random_state=RANDOM_STATE)
    is_inlier = iso.fit_predict(X) == 1
    print(f"  Outlier detection: flagged {(~is_inlier).sum()} / {len(y)} rows as outliers")
    return X[is_inlier], y[is_inlier]


def evaluate_model(name, model, X, y):
    pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipe, X, y, cv=kf, scoring="r2")
    print(f"  {name}: R^2 = {scores.mean():.4f} +/- {scores.std():.4f}  (folds: {np.round(scores, 3)})")
    return scores.mean()


def main():
    X_train, y_train, X_test, test_ids = load_data()
    print(f"Loaded: X_train {X_train.shape}, X_test {X_test.shape}")

    print("\n--- Cross-validated R^2 WITHOUT outlier removal ---")
    candidates = {
        "Ridge(alpha=10)": Ridge(alpha=10.0, random_state=RANDOM_STATE),
        "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }
    scores = {name: evaluate_model(name, model, X_train, y_train) for name, model in candidates.items()}

    best_name = max(scores, key=scores.get)
    print(f"\nBest model by CV R^2: {best_name} ({scores[best_name]:.4f})")

    # Outlier removal happens on preprocessed features, then re-evaluate the winner.
    print("\n--- Re-checking best model WITH training-set outlier removal ---")
    prep = build_preprocessor()
    X_train_proc = prep.fit_transform(X_train, y_train)
    X_clean_proc, y_clean = remove_outliers(X_train_proc, y_train)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    clean_scores = cross_val_score(candidates[best_name], X_clean_proc, y_clean, cv=kf, scoring="r2")
    print(f"  {best_name} (outliers removed): R^2 = {clean_scores.mean():.4f} +/- {clean_scores.std():.4f}")

    use_outlier_removal = clean_scores.mean() > scores[best_name]
    print(f"\nUsing outlier removal for final fit: {use_outlier_removal}")

    # Final fit on all available training data, predict on test set.
    final_prep = build_preprocessor()
    X_train_final = final_prep.fit_transform(X_train, y_train)
    y_train_final = y_train
    if use_outlier_removal:
        X_train_final, y_train_final = remove_outliers(X_train_final, y_train_final)

    final_model = candidates[best_name]
    final_model.fit(X_train_final, y_train_final)

    X_test_final = final_prep.transform(X_test)
    y_pred = final_model.predict(X_test_final)

    train_r2 = r2_score(y_train_final, final_model.predict(X_train_final))
    print(f"\nFinal model train R^2 (in-sample, optimistic): {train_r2:.4f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    submission = pd.DataFrame({"id": test_ids, "y": y_pred})
    out_path = os.path.join(OUTPUT_DIR, "submission.csv")
    submission.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({submission.shape[0]} rows)")


if __name__ == "__main__":
    main()
