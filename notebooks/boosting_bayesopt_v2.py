"""Lane 6: bigger, wider Optuna re-tune of XGBoost/LightGBM/CatBoost on the
WINNING preprocessing, using native early stopping instead of tuning
n_estimators/iterations as a fixed hyperparameter.

notebooks/boosting_bayesopt.py already Optuna-tuned these 3 libraries
(~27 trials each, TPE), but on the OLD baseline preprocessing (median impute,
SelectKBest k=100) -- that tuning has never been redone on top of the current
winning preprocessing (src/advanced_pipeline.py's TopCorrMixedImputer
n_fancy=40 + SelectKBest k=200, R^2=0.5518, the number this script compares
against). This script redoes it, bigger and wider:

  * ~80 trials per model (TPE sampler, seed=42) instead of ~27.
  * WIDER search spaces: XGBoost gains gamma/reg_alpha/reg_lambda; LightGBM
    gains num_leaves/reg_alpha/reg_lambda (and fixes a latent bug in the v1
    script -- LightGBM's `subsample` has NO effect unless `subsample_freq`
    is also set > 0, so this version sets subsample_freq=1 to make the
    tuned subsample value actually take effect); CatBoost's l2_leaf_reg
    range widens from [1, 10] to [1e-2, 30] (log-scale). CatBoost's depth
    is deliberately kept at v1's [3, 8] rather than widened further, and
    boosting_type is fixed to "Plain": a pre-run timing probe found depth
    9-10 combined with a low learning_rate + early stopping could take
    CatBoost 40-90s for a SINGLE fit (vs ~9s at depth<=8) because its
    oblivious trees get much more expensive per round as depth grows and
    the overfitting detector needed hundreds of extra rounds to trigger --
    unbounded, that risked blowing the runtime budget across ~80 trials.
    XGBoost/LightGBM showed no such blowup even at their widened depth
    ranges (8.7s/9.2s worst-case single-fit probes at depth=10/12), so
    only CatBoost's depth was constrained.
  * NATIVE EARLY STOPPING (EarlyStoppingRegressor below) replaces tuning
    n_estimators/iterations as a discrete hyperparameter: every fit carves
    its own eval-set out of whatever training rows it receives
    (train_test_split, eval_frac=0.15, fixed random_state), trains up to
    max_rounds=3000 boosting iterations, and stops after
    early_stopping_rounds=50 rounds without eval-set improvement. This lets
    the data pick the right number of rounds per hyperparameter combination
    (and per fold size) rather than Optuna searching a fixed n_estimators
    value that may not transfer across fold sizes.
  * Preprocessing is the WINNING one throughout, imported directly from
    src/advanced_pipeline.py (TopCorrMixedImputer + SelectKBest k=200)
    rather than re-derived -- both the tuning stage and the final reporting
    stage use it, so retuned hyperparameters are chosen against the actual
    feature set the reported R^2 is measured on.

Same leak-free two-stage scheme as notebooks/boosting_bayesopt.py:

  1. TUNING STAGE -- an 80/20 train_test_split(random_state=42) sets aside
     20% of the training data as a tuning-holdout that NO objective, model,
     or preprocessor fit in this stage ever touches. Within the 80%
     tuning-train portion, an internal KFold(n_splits=3, shuffle=True,
     random_state=42) provides 3 inner folds. The WINNING preprocessor is
     fit on each inner fold's training rows ONCE (precompute_inner_folds)
     and the resulting transformed arrays are reused, unmodified, across
     every one of the ~80 trials x 3 libraries -- preprocessing doesn't
     depend on any boosting hyperparameter, so refitting per trial would be
     ~240x wasted, identical work. Each Optuna trial's objective fits an
     EarlyStoppingRegressor on each inner fold's (preprocessed) training
     rows -- which itself further carves an early-stopping eval-set out of
     ONLY those training rows, never touching the fold's own validation
     rows -- and scores R^2 against the fold's held-out validation rows.
  2. REPORTING STAGE -- each library's best-found hyperparameters (fixed,
     not re-tuned) are plugged into an EarlyStoppingRegressor inside a
     Pipeline(prep=WINNING preprocessing, model=EarlyStoppingRegressor),
     scored via the standard leak-free 5-fold KFold(shuffle=True,
     random_state=42) + cross_val_score on the FULL 1212-row training set --
     exactly the sklearn-Pipeline auto-leak-free pattern src/advanced_
     pipeline.py itself uses. Every outer fold refits the imputer/scaler/
     selector AND re-carves-and-refits the early-stopping eval split, all
     on that fold's training rows only. This is the number compared against
     CURRENT_BEST_R2=0.5518.

Hyperparameters are chosen on a split (stage 1's 80/20) disjoint from the
row partition used for the reported CV (stage 2's 5-fold KFold on all 1212
rows), so the reporting-stage numbers are not tuning-contaminated.
"""
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import optuna
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

optuna.logging.set_verbosity(optuna.logging.WARNING)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Winning preprocessing (Lane B mixed imputer + Lane C k=200), imported
# rather than re-derived -- see src/advanced_pipeline.py's module docstring.
from src.advanced_pipeline import build_preprocessor, load_data  # noqa: E402

import lightgbm as lgb  # noqa: E402
from catboost import CatBoostRegressor  # noqa: E402
from lightgbm import LGBMRegressor  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

RANDOM_STATE = 42
N_SPLITS = 5            # outer, reporting-stage CV (comparable to CURRENT_BEST_R2)
N_TUNE_SPLITS = 3        # inner, tuning-stage CV (within the 80% tuning-train split)
N_TRIALS = 80            # per library -- upper end of the requested 70-80 range
# Per-library safety-net timeout (seconds) so a slow-converging region of the search
# space can't blow the ~10-15 min overall runtime budget -- Optuna's `timeout` stops
# issuing new trials once elapsed time in a study crosses this, so a library may
# complete fewer than N_TRIALS trials (reported honestly via len(study.trials)).
# CatBoost gets a larger allowance: a pre-run timing probe found it can need
# ~15-20s per trial (3 inner folds) even after capping its depth at 8, because its
# oblivious trees are inherently more expensive per boosting round than XGBoost's/
# LightGBM's at comparable depth -- both of which comfortably finish all 80 trials
# in a couple of minutes and never approach their 240s cap.
STUDY_TIMEOUT_S = {"xgboost": 240, "lightgbm": 240, "catboost": 300}
MAX_ROUNDS = 3000        # cap on boosting iterations; early stopping decides the actual count
MAX_ROUNDS = 3000        # cap on boosting iterations; early stopping decides the actual count
EARLY_STOPPING_ROUNDS = 50
EVAL_FRAC = 0.15         # fraction of whatever rows .fit() receives, carved out purely
                         # for early stopping's own eval_set
CURRENT_BEST_R2 = 0.5518  # src/advanced_pipeline.py's winning 5-fold CV R^2, the number to beat

LIBRARIES = ["xgboost", "lightgbm", "catboost"]


def precompute_inner_folds(X_tune_train, y_tune_train):
    """Fit the WINNING preprocessor once per inner CV fold (on that fold's
    training rows only) and cache the transformed arrays for reuse across
    every Optuna trial and every library's study -- preprocessing doesn't
    depend on boosting hyperparameters, so this is ~240x fewer imputer
    refits than doing it inside each trial, with an identical result.
    """
    kf = KFold(n_splits=N_TUNE_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    folds = []
    for tr_idx, va_idx in kf.split(X_tune_train):
        prep = build_preprocessor()
        X_tr = prep.fit_transform(X_tune_train[tr_idx], y_tune_train[tr_idx])
        X_va = prep.transform(X_tune_train[va_idx])
        folds.append((X_tr, y_tune_train[tr_idx], X_va, y_tune_train[va_idx]))
    return folds


class EarlyStoppingRegressor(BaseEstimator, RegressorMixin):
    """Sklearn-compatible wrapper giving XGBoost/LightGBM/CatBoost native
    early stopping inside a Pipeline / cross_val_score / manual CV loop.

    At fit time it carves its OWN eval-set out of whatever rows it is
    handed (train_test_split(test_size=eval_frac, random_state=random_state)),
    trains with early_stopping_rounds against that eval-set up to
    max_rounds boosting iterations, then discards the split -- predict()
    just calls the underlying fitted model (which, for all 3 libraries,
    already restricts itself to the best/early-stopped iteration by
    default; confirmed empirically for this xgboost/lightgbm/catboost
    version combination before writing this class).

    Because the eval-set carve-out happens inside .fit(), this plugs into
    Pipeline/cross_val_score exactly like any other regressor and is
    refit + early-stopped fresh, leak-free, on whatever training rows it's
    handed -- it never sees validation/test rows, only re-splits its own
    training rows.
    """

    def __init__(self, library, params=None, max_rounds=MAX_ROUNDS,
                 early_stopping_rounds=EARLY_STOPPING_ROUNDS, eval_frac=EVAL_FRAC,
                 random_state=RANDOM_STATE):
        self.library = library
        self.params = params
        self.max_rounds = max_rounds
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_frac = eval_frac
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        params = dict(self.params or {})
        X_fit, X_ev, y_fit, y_ev = train_test_split(
            X, y, test_size=self.eval_frac, random_state=self.random_state
        )

        if self.library == "xgboost":
            model = XGBRegressor(
                n_estimators=self.max_rounds,
                early_stopping_rounds=self.early_stopping_rounds,
                random_state=self.random_state, n_jobs=-1, verbosity=0,
                **params,
            )
            model.fit(X_fit, y_fit, eval_set=[(X_ev, y_ev)], verbose=False)
            self.best_iteration_ = int(model.best_iteration)

        elif self.library == "lightgbm":
            model = LGBMRegressor(
                n_estimators=self.max_rounds,
                random_state=self.random_state, n_jobs=-1, verbose=-1,
                **params,
            )
            model.fit(
                X_fit, y_fit, eval_set=[(X_ev, y_ev)],
                callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)],
            )
            self.best_iteration_ = int(model.best_iteration_)

        elif self.library == "catboost":
            model = CatBoostRegressor(
                iterations=self.max_rounds,
                random_state=self.random_state, verbose=False, thread_count=-1,
                use_best_model=True,
                **params,
            )
            model.fit(
                X_fit, y_fit, eval_set=(X_ev, y_ev),
                early_stopping_rounds=self.early_stopping_rounds, verbose=False,
            )
            self.best_iteration_ = int(model.get_best_iteration())

        else:
            raise ValueError(f"unknown library {self.library!r}")

        self.model_ = model
        return self

    def predict(self, X):
        return self.model_.predict(np.asarray(X, dtype=float))


def cv_r2_inner(library, params, inner_folds):
    """Mean R^2 of an EarlyStoppingRegressor(library, params) across the
    precomputed inner tuning folds -- the quantity each Optuna trial
    maximizes."""
    scores = []
    for X_tr, y_tr, X_va, y_va in inner_folds:
        model = EarlyStoppingRegressor(library, params)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_va)
        scores.append(r2_score(y_va, pred))
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Per-library search spaces -- wider than notebooks/boosting_bayesopt.py:
# reg_alpha/reg_lambda/gamma added for XGBoost, num_leaves/reg_alpha/
# reg_lambda added for LightGBM (+ subsample_freq fix so `subsample` isn't
# silently ignored), l2_leaf_reg range widened for CatBoost. n_estimators/
# iterations are NOT tuned here -- native early stopping (EarlyStoppingRegressor)
# picks the round count instead.
# ---------------------------------------------------------------------------

def suggest_xgb_params(trial):
    return dict(
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        max_depth=trial.suggest_int("max_depth", 2, 10),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        gamma=trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    )


def suggest_lgbm_params(trial):
    return dict(
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        max_depth=trial.suggest_int("max_depth", 2, 12),
        num_leaves=trial.suggest_int("num_leaves", 7, 255),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        subsample_freq=1,  # required or LightGBM silently ignores `subsample` (bagging_freq=0 default)
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    )


def suggest_catboost_params(trial):
    return dict(
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        # NOTE: capped at 8, not widened to 10 -- diagnosed empirically before the full
        # run (see notebooks/boosting_bayesopt_v2.py commit history / dev notes): depth
        # 9-10 combined with a low learning_rate and early stopping can take CatBoost
        # 40-90s for a SINGLE fit here (vs ~9s at depth<=8), because its symmetric/
        # oblivious trees get dramatically more expensive per round as depth grows and
        # the overfitting detector takes hundreds of extra rounds to trigger. depth<=8
        # (256 leaves) is already generous for ~650-970-row folds; keeps runtime bounded
        # without giving up real modeling capacity.
        depth=trial.suggest_int("depth", 3, 8),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-2, 30.0, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bylevel=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        bootstrap_type="Bernoulli",
        # Plain (not the Auto-selected Ordered) boosting: Ordered's main benefit is
        # reducing prediction shift on LARGE datasets; on ~650-970-row folds it's mostly
        # extra cost. Also empirically ~25% faster than Ordered at the same (deep,
        # low-lr) worst case tested above.
        boosting_type="Plain",
    )


SUGGEST_FNS = {
    "xgboost": suggest_xgb_params,
    "lightgbm": suggest_lgbm_params,
    "catboost": suggest_catboost_params,
}


def make_objective(library, inner_folds):
    suggest_fn = SUGGEST_FNS[library]

    def objective(trial):
        params = suggest_fn(trial)
        return cv_r2_inner(library, params, inner_folds)

    return objective


def _progress_callback(library):
    def _cb(study, trial):
        if (trial.number + 1) % 20 == 0:
            print(f"    [{library}] trial {trial.number + 1}: best inner R^2 so far = {study.best_value:.4f}")
    return _cb


def run_study(library, inner_folds, n_trials=N_TRIALS):
    t0 = time.time()
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        make_objective(library, inner_folds),
        n_trials=n_trials,
        timeout=STUDY_TIMEOUT_S[library],
        callbacks=[_progress_callback(library)],
        show_progress_bar=False,
    )
    elapsed = time.time() - t0
    n_done = len(study.trials)
    capped_note = f" (safety-net timeout hit, short of the requested {n_trials})" if n_done < n_trials else ""
    print(f"  [{library}] {n_done} trials in {elapsed:.1f}s, "
          f"best inner-3fold R^2 = {study.best_value:.4f}{capped_note}")
    print(f"  [{library}] best params: {study.best_params}")
    return study.best_params, study.best_value


def finalize_params(library, best_params):
    """Translate Optuna's study.best_params into the actual kwargs the final
    model needs. Two things study.best_params does NOT capture, both of
    which bit this script during development (mirrors notebooks/
    boosting_bayesopt.py's build_final_model, which exists for the same
    reason):

    1. study.best_params is keyed by whatever name was passed as the first
       argument to trial.suggest_* -- for CatBoost's colsample_bylevel
       (named "colsample_bytree" in the suggest call for readability/parity
       with the other two libraries), that name does NOT match CatBoost's
       actual constructor parameter, so it must be renamed back here.
    2. Fixed (non-tuned) values baked directly into a suggest_*_params()
       return dict -- LightGBM's subsample_freq=1, CatBoost's
       bootstrap_type/boosting_type -- never came from a trial.suggest_*
       call, so Optuna never records them in study.best_params either;
       they must be reattached here for the final model to match what was
       actually scored during tuning.
    """
    params = dict(best_params)
    if library == "lightgbm":
        params["subsample_freq"] = 1
    elif library == "catboost":
        params["colsample_bylevel"] = params.pop("colsample_bytree")
        params["bootstrap_type"] = "Bernoulli"
        params["boosting_type"] = "Plain"
    return params


def evaluate_reporting_stage(library, best_params, X, y):
    """Stage 2: fixed (not re-tuned) hyperparameters, WINNING preprocessing,
    native early stopping, standard leak-free 5-fold KFold + cross_val_score
    on the full 1212-row training set. Comparable to CURRENT_BEST_R2."""
    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("model", EarlyStoppingRegressor(library, best_params)),
    ])
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipe, X, y, cv=kf, scoring="r2")
    print(f"  {library}: 5-fold CV R^2 = {scores.mean():.4f} +/- {scores.std():.4f}  "
          f"(folds: {np.round(scores, 4)})")
    return scores.mean(), scores.std()


def main():
    t_start = time.time()
    X, y, train_ids, X_test, test_ids = load_data()
    print(f"Loaded X {X.shape}, y {y.shape}")
    print(f"Current best to beat (src/advanced_pipeline.py): R^2 = {CURRENT_BEST_R2}")
    print("Preprocessing: WINNING (TopCorrMixedImputer n_fancy=40 + SelectKBest k=200), "
          "imported from src/advanced_pipeline.py")
    print(f"Native early stopping: max_rounds={MAX_ROUNDS}, "
          f"early_stopping_rounds={EARLY_STOPPING_ROUNDS}, eval_frac={EVAL_FRAC}")

    # ---------------- Stage 1: tuning ----------------
    X_tune_train, X_tune_holdout, y_tune_train, y_tune_holdout = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"\n--- Stage 1: Optuna tuning (80% tuning-train = {X_tune_train.shape[0]} rows, "
          f"20% tuning-holdout = {X_tune_holdout.shape[0]} rows set aside/unused) ---")

    t0 = time.time()
    inner_folds = precompute_inner_folds(X_tune_train, y_tune_train)
    print(f"  Precomputed {N_TUNE_SPLITS} inner CV folds w/ winning preprocessing in "
          f"{time.time() - t0:.1f}s (shared across all 3 libraries' studies)")

    best_params_by_lib = {}
    best_inner_r2_by_lib = {}
    for lib in LIBRARIES:
        print(f"\nTuning {lib} (up to {N_TRIALS} trials, TPE, seed={RANDOM_STATE}, "
              f"{STUDY_TIMEOUT_S[lib]}s safety-net timeout)...")
        raw_best_params, best_val = run_study(lib, inner_folds, N_TRIALS)
        best_params_by_lib[lib] = finalize_params(lib, raw_best_params)
        best_inner_r2_by_lib[lib] = best_val

    # ---------------- Stage 2: reporting ----------------
    print("\n--- Stage 2: reporting CV (5-fold KFold on FULL 1212-row training set, "
          "leak-free Pipeline+cross_val_score, hyperparams FIXED from stage 1, "
          "native early stopping refit per outer fold) ---")
    reporting_r2 = {}
    reporting_std = {}
    for lib in LIBRARIES:
        mean_r2, std_r2 = evaluate_reporting_stage(lib, best_params_by_lib[lib], X, y)
        reporting_r2[lib] = mean_r2
        reporting_std[lib] = std_r2

    best_lib = max(reporting_r2, key=reporting_r2.get)
    print("\n=== Summary ===")
    print(f"Current best to beat (src/advanced_pipeline.py, winning preproc + tuned "
          f"XGBoost, fixed n_estimators): R^2 = {CURRENT_BEST_R2}")
    for lib in LIBRARIES:
        beats = "BEATS" if reporting_r2[lib] > CURRENT_BEST_R2 else "does not beat"
        print(f"  {lib}: 5-fold CV R^2 = {reporting_r2[lib]:.4f} +/- {reporting_std[lib]:.4f}  "
              f"({beats} {CURRENT_BEST_R2})")
        print(f"    inner-tuning best R^2 (stage 1, 80% split, 3-fold): {best_inner_r2_by_lib[lib]:.4f}")
        print(f"    best hyperparams: {best_params_by_lib[lib]}")
    print(f"\nBest of the three retuned libraries: {best_lib} with R^2 = {reporting_r2[best_lib]:.4f}")
    print(f"Improvement over current best ({CURRENT_BEST_R2}): "
          f"{reporting_r2[best_lib] - CURRENT_BEST_R2:+.4f}")
    print(
        "\nNote: hyperparameters for all 3 models were chosen via Optuna studies whose "
        "objective only ever saw an 80% tuning-train split (internal 3-fold CV within it, "
        "with the WINNING preprocessing refit per inner fold and native early stopping "
        "carving its own eval-set out of each inner fold's training rows); the 20% "
        "tuning-holdout was set aside and unused. The stage-2 reporting numbers above come "
        "from a completely separate 5-fold CV over all 1212 rows with those hyperparameters "
        "held fixed, so the reporting CV is not tuning-contaminated."
    )
    print(f"\nTotal runtime: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
