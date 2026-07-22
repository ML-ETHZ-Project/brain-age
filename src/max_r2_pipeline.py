"""Max-R2 sprint: adopts the single strongest confirmed gain from six parallel
experiment lanes run on top of src/advanced_pipeline.py's reference (mixed
imputer + k=200 SelectKBest + tuned XGBoost, R^2=0.5518, 5-fold CV):

  Investigation (notebooks/leakage_investigation.py) -- a 5-part recon pass
    (train/test near-duplicates, id/row-order leakage, hidden signal in the
    "insane magnitude" noise columns, duplicate columns, age-correlated
    clusters) found NO exploitable structure. The 0.5518 -> ~0.75 gap the user
    asked about is not explained by any data-structure quirk; it likely
    reflects an inherently noisy/hard-to-beat feature set given the
    organizers' deliberate corruption, not a leakage shortcut.
  Lane 1 (notebooks/feature_engineering_v2.py) -- pairwise products/ratios and
    per-row summary stats of the top-20 age-correlated columns, plus PCA,
    appended to the winning preprocessing's output, with the SAME already-
    tuned XGB_PARAMS (no retuning): R^2=0.5567. A fresh wider-space Optuna
    retune on top of these features did WORSE (0.5353) -- overfits the
    969-row/3-fold tuning signal.
  Lane 2 (notebooks/stacking_ensemble_v2.py) -- proper nested leak-free
    stacking of 10 diverse base learners: best (ElasticNetCV meta) R^2=0.5555.
  Lane 3 (notebooks/target_transform_v2.py) -- target transforms: best
    (Yeo-Johnson) R^2=0.5554, but with 40% higher fold-to-fold std -- not a
    real win.
  Lane 4 (notebooks/pseudo_labeling_v2.py) -- transductive unsupervised
    refitting: exactly 0.0000 delta (a validated no-op). Confidence-filtered
    pseudo-labeling: inconclusive (wins 3/5 simulated splits, delta smaller
    than split-to-split noise). Not adopted.
  Lane 5 (notebooks/multiseed_bagging_v2.py) -- averaging 20 differently-
    seeded (10 of them also bootstrap-resampled) refits of the exact same
    tuned XGBoost: R^2=0.5580, the single largest confirmed lane gain.
  Lane 6 (notebooks/boosting_bayesopt_v2.py) -- aggressive 80-trial-per-
    library Optuna retuning WITH native early stopping, on the winning
    preprocessing: best (LightGBM/CatBoost tied) R^2=0.5326 -- WORSE than the
    existing tuned XGBoost. Not adopted.

SYNTHESIS ATTEMPT (and an honest negative result worth recording): lane 1
(feature engineering) and lane 5 (multi-seed bagging) look complementary on
paper -- one changes the input representation, the other averages out
fit-to-fit noise -- so this file originally combined them (FeatureEngineer ->
SelectKBest(k=200) -> 20-seed-bagged XGBoost). That combination was actually
RUN (not just estimated) and scored R^2=0.5531 +/- 0.0252 -- WORSE than lane 5
alone (0.5580) and no better than lane 1 alone (0.5567). The two gains do not
compound; they likely capture overlapping variance (both are, in different
ways, ways of not overfitting to one particular sample of rows/columns/seeds),
so stacking them regresses toward the baseline rather than adding up.

ADOPTED: lane 5 alone (20-seed bagging on the existing, unmodified winning
preprocessing -- no feature engineering layer), since it is both the single
best confirmed number of the whole sprint (R^2=0.5580) AND the simplest
option (no ~1200-column engineered feature space, no PCA, no added leak
surface). Simplicity + best validated result beats complexity + a worse one.

Honest bottom line for the ~0.75 the user asked about: not reachable with any
technique tried in this sprint (six modeling lanes, a dedicated leak/structure
investigation, and this synthesis attempt all land in the R^2=0.53-0.58
band). The investigation lane found no exploitable data-structure explaining
the gap, so it looks like a genuine ceiling for this corrupted, ~832-column
tabular feature set at n=1212 with these techniques, not a shortcut we missed.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from src.advanced_pipeline import RANDOM_STATE, N_SPLITS, XGB_PARAMS, build_preprocessor  # noqa: E402

N_SEEDS = 20
N_BOOTSTRAP_SEEDS = 10
OUTLIER_CONTAMINATION = 0.05

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


def fit_predict_bagged(X_tr, y_tr, X_val, n_seeds=N_SEEDS, n_bootstrap_seeds=N_BOOTSTRAP_SEEDS):
    """20-seed bagged prediction (lane 5's recipe): seeds 0..9 are plain refits
    on (X_tr, y_tr); seeds 10..19 additionally bootstrap-resample the rows
    first (in-bag only, never touching X_val). Predictions are averaged."""
    preds = []
    for seed in range(n_seeds):
        if seed >= (n_seeds - n_bootstrap_seeds):
            rng = np.random.RandomState(RANDOM_STATE + seed)
            in_bag = rng.randint(0, len(y_tr), size=len(y_tr))
            X_fit, y_fit = X_tr[in_bag], y_tr[in_bag]
        else:
            X_fit, y_fit = X_tr, y_tr
        params = dict(XGB_PARAMS)
        params["random_state"] = seed
        model = XGBRegressor(**params)
        model.fit(X_fit, y_fit)
        preds.append(model.predict(X_val))
    return np.mean(preds, axis=0)


def evaluate_bagged_cv(X, y):
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for fold_i, (tr_idx, val_idx) in enumerate(kf.split(X)):
        t0 = time.time()
        X_tr_raw, X_val_raw = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        prep = build_preprocessor()  # fit ONCE per fold, on training rows only
        X_tr = prep.fit_transform(X_tr_raw, y_tr)
        X_val = prep.transform(X_val_raw)

        pred = fit_predict_bagged(X_tr, y_tr, X_val)
        r2 = r2_score(y_val, pred)
        scores.append(r2)
        print(f"  Fold {fold_i}: R^2 = {r2:.4f}  ({time.time() - t0:.1f}s)")
    return np.array(scores)


def classify_outliers(X_proc, train_ids, contamination=OUTLIER_CONTAMINATION):
    iso = IsolationForest(contamination=contamination, random_state=RANDOM_STATE)
    is_outlier = iso.fit_predict(X_proc) == -1
    print(f"  Outlier classification: flagged {is_outlier.sum()} / {len(train_ids)} training rows as outliers")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_path = os.path.join(PROCESSED_DIR, "outlier_labels.csv")
    pd.DataFrame({"id": train_ids, "is_outlier": is_outlier.astype(int)}).to_csv(out_path, index=False)
    print(f"  Wrote {out_path}")


def main():
    t_start = time.time()
    X_train, y_train, train_ids, X_test, test_ids = load_data()
    print(f"Loaded: X_train {X_train.shape}, X_test {X_test.shape}")
    print("Current best (src/advanced_pipeline.py, single-seed): R^2 = 0.5518 +/- 0.0225")

    print("\n--- Winning combo (Lane 5 adopted alone): 20-seed bagging, unmodified winning preprocessing ---")
    bagged_scores = evaluate_bagged_cv(X_train, y_train)
    print(f"  Bagged: R^2 = {bagged_scores.mean():.4f} +/- {bagged_scores.std():.4f}  "
          f"(folds: {np.round(bagged_scores, 4)})")
    print(f"  Improvement over 0.5518: {bagged_scores.mean() - 0.5518:+.4f}")
    print(
        "  (Note: this project's own src/advanced_pipeline.py already reports a bootstrap 90% CI of "
        "[0.4185, 0.5620] for the single-seed version of this same preprocessing+model -- bootstrapping "
        "the 20-seed bagged variant would cost ~20x that runtime, which this sprint's earlier synthesis "
        "attempt found impractical (a single-seed 30-resample bootstrap already took 67.5s; a full "
        "100-resample bagged bootstrap ran over 2 hours before being aborted), so we rely on the 5-fold "
        "CV std above as the primary uncertainty estimate for the bagged model specifically.)"
    )

    print("\n--- Subtask 1: outlier classification (reporting only, not used for training) ---")
    prep_for_outliers = build_preprocessor()
    X_proc_for_outliers = prep_for_outliers.fit_transform(X_train, y_train)
    classify_outliers(X_proc_for_outliers, train_ids)

    print("\n--- Final fit on all training data (20-seed bagged), predict on test set ---")
    prep_final = build_preprocessor()
    X_train_final = prep_final.fit_transform(X_train, y_train)
    X_test_final = prep_final.transform(X_test)
    y_pred_test = fit_predict_bagged(X_train_final, y_train, X_test_final)
    y_pred_train = fit_predict_bagged(X_train_final, y_train, X_train_final)
    train_r2 = r2_score(y_train, y_pred_train)
    print(f"  Final bagged model train R^2 (in-sample, optimistic): {train_r2:.4f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    submission = pd.DataFrame({"id": test_ids, "y": y_pred_test})
    out_path = os.path.join(OUTPUT_DIR, "max_r2_submission.csv")
    submission.to_csv(out_path, index=False)
    print(f"  Wrote {out_path} ({submission.shape[0]} rows)")

    print("\n=== Summary ===")
    print(f"  Baseline (src/baseline.py):                          R^2 = 0.5065")
    print(f"  Current best (src/advanced_pipeline.py, single-seed): R^2 = 0.5518")
    print(f"  WINNER (20-seed bagging, this file):                 R^2 = {bagged_scores.mean():.4f} "
          f"+/- {bagged_scores.std():.4f}")
    print(f"\nTotal runtime: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
