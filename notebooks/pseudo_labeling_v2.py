"""Lane 4: Transductive / pseudo-labeling experiments.

Two ablations on top of src/advanced_pipeline.py's current-best pipeline
(TopCorrMixedImputer(n_fancy=40) -> RobustScaler -> VarianceThreshold(1e-8) ->
SelectKBest(f_regression, k=200) -> XGBRegressor(tuned), 5-fold CV R^2 =
0.5518 +/- 0.0225). Both are validated leak-free; this is explicitly the
highest-risk-of-fooling-yourself lane so every step is checked against what
information source (y? test rows' features? CV validation rows?) it is
allowed to touch.

(a) Unsupervised-only transduction
    -----------------------------------------------------------------------
    RobustScaler's median/IQR and VarianceThreshold's per-column variance are
    UNSUPERVISED statistics (no y involved) -- so it is legitimate to fit them
    on the fold's training rows *combined with the real (unlabeled) X_test
    rows*, i.e. genuinely transductive. The correlation ranking inside
    TopCorrMixedImputer explicitly uses y (it ranks columns by |corr(x_j, y)|)
    and SelectKBest's f_regression explicitly uses y -- both of those stay
    strictly fit on the fold's training rows only, exactly as in the current
    pipeline. Only RobustScaler and VarianceThreshold get the transductive
    treatment. Critically, the CV *validation* fold is NEVER part of the
    "test" pool used for this -- only the real, separately-loaded X_test.csv
    (776 unlabeled competition rows) is combined with each fold's training
    rows; the validation fold remains completely untouched until scoring, so
    this stays a fair leak-free comparison of train-only vs train+test-fit
    unsupervised statistics.

    A manual per-fold loop is required here (this transductive combination
    isn't expressible as a single sklearn Pipeline step), mirroring the
    pattern in notebooks/outlier_comparison.py and notebooks/boosting_bayesopt.py.
    The train-only variant is cross-checked against src/advanced_pipeline.py's
    own cross_val_score(build_full_pipeline(), ...) number as a parity check
    that the manual loop is implemented correctly before trusting the delta.

(b) Pseudo-labeling
    -----------------------------------------------------------------------
    Fit the current-best pipeline on labeled data, predict on an unlabeled
    pool, use a small bootstrap ensemble (8 refits, in the "5-10" range the
    task suggests) to measure per-row prediction agreement (std across the
    ensemble), keep only the most-agreeing subset as pseudo-labels (using the
    ensemble mean as the pseudo-label value), add those rows to the training
    set, and refit. We have no true test labels to check this against, so we
    SIMULATE it: repeatedly carve a random 20% of the REAL training data into
    a stand-in "fake test" pool (labels hidden during fitting, revealed only
    for final scoring), run the exact procedure using the other 80% as
    "train", and score the pseudo-label-augmented model against the fake
    test's TRUE labels -- compared to a plain model fit on the same 80% with
    no pseudo-labeling. Repeated across 5 random 80/20 splits for a stable
    read, and swept across 3 confidence fractions (keep the most-confident
    100%/50%/25% of the unlabeled pool) to see whether confidence-based
    selection actually matters versus blindly adding every pseudo-label.
"""
import os
import sys
import time
import warnings

import numpy as np
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from advanced_pipeline import (  # noqa: E402
    K_BEST,
    N_FANCY,
    XGB_PARAMS,
    TopCorrMixedImputer,
    build_full_pipeline,
    build_preprocessor,
    load_data,
)

RANDOM_STATE = 42
N_SPLITS = 5
CURRENT_BEST_R2 = 0.5518
CURRENT_BEST_STD = 0.0225

# Pseudo-labeling knobs
N_ENSEMBLE = 8
CONFIDENCE_FRACS = [1.0, 0.5, 0.25]  # 1.0 = naive (no filtering), then moderate, then aggressive
SIM_RANDOM_STATES = [42, 43, 44, 45, 46]


# ===========================================================================
# Experiment (a): unsupervised-only transduction
# ===========================================================================

def eval_train_only_manual(X, y, kf):
    """Manual-loop reimplementation of the current pipeline (everything fit
    on the fold's training rows only). Used purely as a parity check against
    build_full_pipeline() + cross_val_score -- if these two don't (nearly)
    match, the manual loop below has a bug and its transductive comparison
    can't be trusted."""
    scores = []
    for tr_idx, va_idx in kf.split(X):
        X_tr_raw, X_va_raw = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        imputer = TopCorrMixedImputer(n_fancy=N_FANCY)
        imputer.fit(X_tr_raw, y_tr)
        X_tr = imputer.transform(X_tr_raw)
        X_va = imputer.transform(X_va_raw)

        scaler = RobustScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)

        vt = VarianceThreshold(threshold=1e-8)
        X_tr_v = vt.fit_transform(X_tr_s)
        X_va_v = vt.transform(X_va_s)

        sel = SelectKBest(f_regression, k=K_BEST)
        X_tr_k = sel.fit_transform(X_tr_v, y_tr)
        X_va_k = sel.transform(X_va_v)

        model = XGBRegressor(**XGB_PARAMS)
        model.fit(X_tr_k, y_tr)
        pred = model.predict(X_va_k)
        scores.append(r2_score(y_va, pred))
    return np.array(scores)


def eval_transductive_manual(X, y, X_test_raw, kf):
    """RobustScaler + VarianceThreshold fit on (fold-train + real X_test)
    combined, unsupervised only. TopCorrMixedImputer's correlation ranking
    and SelectKBest both stay fit on the fold's training rows only, since
    both use y. The CV validation fold is never part of the combined pool."""
    scores = []
    for tr_idx, va_idx in kf.split(X):
        X_tr_raw, X_va_raw = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        imputer = TopCorrMixedImputer(n_fancy=N_FANCY)
        imputer.fit(X_tr_raw, y_tr)  # uses y -> train-fold rows only, correct
        X_tr = imputer.transform(X_tr_raw)
        X_va = imputer.transform(X_va_raw)
        X_te = imputer.transform(X_test_raw)  # transform only -- imputer never fit on test

        combined = np.vstack([X_tr, X_te])  # fold-train + real unlabeled test, no labels used below

        scaler = RobustScaler()
        scaler.fit(combined)  # unsupervised -> legitimate to include real test rows
        X_tr_s = scaler.transform(X_tr)
        X_va_s = scaler.transform(X_va)
        combined_s = scaler.transform(combined)

        vt = VarianceThreshold(threshold=1e-8)
        vt.fit(combined_s)  # unsupervised -> legitimate to include real test rows
        X_tr_v = vt.transform(X_tr_s)
        X_va_v = vt.transform(X_va_s)

        sel = SelectKBest(f_regression, k=K_BEST)
        X_tr_k = sel.fit_transform(X_tr_v, y_tr)  # uses y -> train-fold rows only, correct
        X_va_k = sel.transform(X_va_v)

        model = XGBRegressor(**XGB_PARAMS)
        model.fit(X_tr_k, y_tr)
        pred = model.predict(X_va_k)
        scores.append(r2_score(y_va, pred))
    return np.array(scores)


def run_experiment_a(X_train, y_train, X_test):
    print("=" * 78)
    print("EXPERIMENT (a): unsupervised-only transduction (RobustScaler + VarianceThreshold")
    print("fit on train+real-test combined, vs train-only)")
    print("=" * 78)

    kf_official = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    t0 = time.time()
    official_scores = cross_val_score(build_full_pipeline(), X_train, y_train, cv=kf_official, scoring="r2")
    print(f"\n[Reference] src/advanced_pipeline.py's own Pipeline+cross_val_score: "
          f"R^2 = {official_scores.mean():.4f} +/- {official_scores.std():.4f}  "
          f"({time.time() - t0:.1f}s)")

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    t0 = time.time()
    manual_train_only = eval_train_only_manual(X_train, y_train, kf)
    print(f"[Parity check] manual train-only loop:                        "
          f"R^2 = {manual_train_only.mean():.4f} +/- {manual_train_only.std():.4f}  "
          f"({time.time() - t0:.1f}s)")
    parity_gap = abs(manual_train_only.mean() - official_scores.mean())
    print(f"  Gap vs official cross_val_score: {parity_gap:.4f} "
          f"({'OK, manual loop matches' if parity_gap < 0.01 else 'WARNING: manual loop diverges, investigate before trusting (b)'})")

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    t0 = time.time()
    manual_transductive = eval_transductive_manual(X_train, y_train, X_test, kf)
    print(f"[Transductive] combined-fit RobustScaler+VarianceThreshold:    "
          f"R^2 = {manual_transductive.mean():.4f} +/- {manual_transductive.std():.4f}  "
          f"({time.time() - t0:.1f}s)")

    delta = manual_transductive.mean() - manual_train_only.mean()
    print(f"\nDelta (transductive - train-only), manual-loop apples-to-apples: {delta:+.4f}")
    print(f"Per-fold scores train-only:   {np.round(manual_train_only, 4)}")
    print(f"Per-fold scores transductive: {np.round(manual_transductive, 4)}")
    print(
        "\nNote: RobustScaler is a per-column affine transform (positive scale) and "
        "SelectKBest's f_regression F-statistic ranking (hence which columns get kept) is "
        "invariant to positive-scale affine rescaling of a column, and tree-based models' "
        "learned splits are invariant to monotonic per-feature transforms in exact/greedy "
        "split search -- so a near-zero delta here would be the theoretically expected "
        "result, not evidence of a bug. Any nonzero residual is most plausibly attributable "
        "to XGBoost's histogram-based split-finding discretizing differently at different "
        "absolute feature scales (a second-order numerical effect), not real transductive signal."
    )
    return {
        "official_cross_val_score_r2": official_scores.mean(),
        "manual_train_only_r2": manual_train_only.mean(),
        "manual_train_only_std": manual_train_only.std(),
        "manual_transductive_r2": manual_transductive.mean(),
        "manual_transductive_std": manual_transductive.std(),
        "delta": delta,
        "parity_gap": parity_gap,
    }


# ===========================================================================
# Experiment (b): pseudo-labeling, validated via simulation on real labels
# ===========================================================================

def run_pseudo_labeling_one_split(X_full, y_full, random_state, n_ensemble=N_ENSEMBLE,
                                   confidence_fracs=CONFIDENCE_FRACS):
    """One simulated 80/20 split: X20/y20 stands in for the unlabeled test
    pool (y20 hidden from every fitting step, used only at the very end to
    score). Returns plain R^2 and, for each confidence fraction, the
    pseudo-label-augmented R^2 -- all scored on the FULL X20 (not just the
    pseudo-labeled subset), since in production we'd need predictions for
    the whole test set regardless of which rows we trusted enough to
    pseudo-label."""
    X80, X20, y80, y20 = train_test_split(X_full, y_full, test_size=0.2, random_state=random_state)

    # Plain model: single deterministic fit on the labeled 80%, no pseudo-labeling.
    plain_pipe = build_full_pipeline()
    plain_pipe.fit(X80, y80)
    plain_pred = plain_pipe.predict(X20)
    plain_r2 = r2_score(y20, plain_pred)

    # Bootstrap ensemble (in-bag resampling of the labeled 80% pool only) for
    # per-row agreement/confidence on the unlabeled 20% pool.
    n = len(y80)
    ensemble_preds = np.zeros((n_ensemble, len(y20)))
    for i in range(n_ensemble):
        rng = np.random.RandomState(random_state * 1000 + i)
        boot_idx = rng.randint(0, n, size=n)
        pipe = build_full_pipeline()
        pipe.fit(X80[boot_idx], y80[boot_idx])
        ensemble_preds[i] = pipe.predict(X20)

    pseudo_mean = ensemble_preds.mean(axis=0)
    pseudo_std = ensemble_preds.std(axis=0)

    aug_results = {}
    for frac in confidence_fracs:
        n_confident = max(1, int(len(y20) * frac))
        confident_idx = np.argsort(pseudo_std)[:n_confident]  # lowest std = most agreement

        X_pseudo = X20[confident_idx]
        y_pseudo = pseudo_mean[confident_idx]

        X_aug = np.vstack([X80, X_pseudo])
        y_aug = np.concatenate([y80, y_pseudo])

        aug_pipe = build_full_pipeline()
        aug_pipe.fit(X_aug, y_aug)
        aug_pred = aug_pipe.predict(X20)
        aug_r2 = r2_score(y20, aug_pred)
        aug_results[frac] = aug_r2

    return {
        "plain_r2": plain_r2,
        "aug_r2_by_frac": aug_results,
        "pseudo_std_mean": pseudo_std.mean(),
        "pseudo_std_median": np.median(pseudo_std),
    }


def run_experiment_b(X_train, y_train):
    print("\n" + "=" * 78)
    print("EXPERIMENT (b): pseudo-labeling, validated via simulated 80/20 holdouts")
    print("=" * 78)
    print(f"Ensemble size: {N_ENSEMBLE} bootstrap refits. Confidence fractions tested: {CONFIDENCE_FRACS}")
    print(f"Simulated splits (random_states): {SIM_RANDOM_STATES}\n")

    all_plain = []
    all_aug = {frac: [] for frac in CONFIDENCE_FRACS}
    t0 = time.time()

    for rs in SIM_RANDOM_STATES:
        t_split = time.time()
        result = run_pseudo_labeling_one_split(X_train, y_train, rs)
        all_plain.append(result["plain_r2"])
        line = f"  split rs={rs}: plain R^2={result['plain_r2']:.4f}"
        for frac in CONFIDENCE_FRACS:
            aug_r2 = result["aug_r2_by_frac"][frac]
            all_aug[frac].append(aug_r2)
            line += f" | frac={frac:.2f} aug R^2={aug_r2:.4f} (d={aug_r2 - result['plain_r2']:+.4f})"
        line += f"  [pseudo std mean={result['pseudo_std_mean']:.3f}]  ({time.time() - t_split:.1f}s)"
        print(line)

    print(f"\nTotal experiment (b) runtime: {time.time() - t0:.1f}s")

    all_plain = np.array(all_plain)
    print(f"\n--- Summary across {len(SIM_RANDOM_STATES)} simulated splits ---")
    print(f"Plain model (no pseudo-labeling):  R^2 = {all_plain.mean():.4f} +/- {all_plain.std():.4f}")

    summary_by_frac = {}
    for frac in CONFIDENCE_FRACS:
        arr = np.array(all_aug[frac])
        delta = arr - all_plain
        label = "naive (all rows, no filtering)" if frac == 1.0 else f"top {frac:.0%} most-confident"
        print(f"Augmented, {label:32s}: R^2 = {arr.mean():.4f} +/- {arr.std():.4f}   "
              f"mean delta = {delta.mean():+.4f}  (wins {int((delta > 0).sum())}/{len(delta)} splits)")
        summary_by_frac[frac] = {
            "aug_r2_mean": arr.mean(),
            "aug_r2_std": arr.std(),
            "mean_delta": delta.mean(),
            "wins": int((delta > 0).sum()),
            "n_splits": len(delta),
        }

    best_frac = max(summary_by_frac, key=lambda f: summary_by_frac[f]["aug_r2_mean"])
    best_delta = summary_by_frac[best_frac]["mean_delta"]
    print(f"\nBest confidence fraction: {best_frac} (mean delta {best_delta:+.4f} vs plain)")
    if best_delta > 0.003 and summary_by_frac[best_frac]["wins"] >= 4:
        verdict = "Pseudo-labeling shows a real, consistent improvement -- worth adopting."
    elif best_delta > 0 and summary_by_frac[best_frac]["wins"] >= 3:
        verdict = "Pseudo-labeling shows a small, inconsistent improvement -- marginal, likely within noise."
    else:
        verdict = "Pseudo-labeling does NOT help (flat or negative) in this simulation -- do not adopt."
    print(f"Verdict: {verdict}")

    return {
        "plain_r2_mean": all_plain.mean(),
        "plain_r2_std": all_plain.std(),
        "summary_by_frac": summary_by_frac,
        "best_frac": best_frac,
        "verdict": verdict,
    }


def main():
    t_start = time.time()
    X_train, y_train, train_ids, X_test, test_ids = load_data()
    print(f"Loaded: X_train {X_train.shape}, y_train {y_train.shape}, X_test {X_test.shape}")
    print(f"Current best (src/advanced_pipeline.py): R^2 = {CURRENT_BEST_R2} +/- {CURRENT_BEST_STD}\n")

    result_a = run_experiment_a(X_train, y_train, X_test)
    result_b = run_experiment_b(X_train, y_train)

    print("\n" + "=" * 78)
    print("OVERALL SUMMARY")
    print("=" * 78)
    print(f"Current best (reference, 5-fold CV on full train): R^2 = {CURRENT_BEST_R2} +/- {CURRENT_BEST_STD}")
    print(f"(a) train-only (manual-loop parity check):         R^2 = {result_a['manual_train_only_r2']:.4f} "
          f"+/- {result_a['manual_train_only_std']:.4f}")
    print(f"(a) transductive (train+real-test unsupervised fit): R^2 = {result_a['manual_transductive_r2']:.4f} "
          f"+/- {result_a['manual_transductive_std']:.4f}  (delta {result_a['delta']:+.4f})")
    print(f"(b) plain model (simulated 80/20, avg of 5 splits):  R^2 = {result_b['plain_r2_mean']:.4f} "
          f"+/- {result_b['plain_r2_std']:.4f}")
    for frac in CONFIDENCE_FRACS:
        s = result_b["summary_by_frac"][frac]
        print(f"(b) pseudo-labeled, frac={frac:.2f}:                    R^2 = {s['aug_r2_mean']:.4f} "
              f"+/- {s['aug_r2_std']:.4f}  (delta {s['mean_delta']:+.4f}, wins {s['wins']}/{s['n_splits']})")
    print(f"\n(b) verdict: {result_b['verdict']}")
    print(f"\nTotal runtime: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
