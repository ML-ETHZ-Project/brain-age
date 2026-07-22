"""Lane 3: target-transform experiments on top of the winning preprocessing +
tuned XGBoost (src/advanced_pipeline.py's TopCorrMixedImputer(n=40) ->
RobustScaler -> VarianceThreshold -> SelectKBest(k=200) -> XGBRegressor(tuned),
CV R^2 = 0.5518 +/- 0.0225).

Question: does predicting a transformed target (log, sqrt, Box-Cox,
Yeo-Johnson, quantile-normal) instead of raw age improve R^2, scored back on
the ORIGINAL age scale?

Leak-free method: sklearn.compose.TransformedTargetRegressor wraps the
*entire* winning Pipeline (imputer/scaler/selector/model) as its `regressor`.
TransformedTargetRegressor.fit(X, y) fits the target transformer on the y it
is given and the wrapped regressor on transform(y); .predict() inverse-
transforms the wrapped regressor's predictions back to the original age
scale. Run through cross_val_score(cv=KFold(5, shuffle=True,
random_state=42)) exactly like every other reported number in this project:
cross_val_score only ever hands each fold's *training* rows to .fit(), so
the target transformer -- like the imputer/scaler/selector inside it -- is
refit fresh per fold on that fold's training y only, never touching that
fold's validation rows. R^2 is computed by cross_val_score on the inverse-
transformed predictions vs the original (never-transformed) y_val, so all
five variants are directly comparable to the untransformed 0.5518 baseline.

Variants tried:
  - identity (sanity check -- should reproduce 0.5518)
  - log(y)                          func=np.log,    inverse=np.exp
  - sqrt(y)                         func=np.sqrt,   inverse=np.square
  - PowerTransformer(box-cox)       fit per fold on y_train only (age > 0, so Box-Cox is valid)
  - PowerTransformer(yeo-johnson)   fit per fold on y_train only
  - QuantileTransformer(normal)     fit per fold on y_train only (n_quantiles capped to fold size)
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import PowerTransformer, QuantileTransformer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from advanced_pipeline import build_full_pipeline, load_data, RANDOM_STATE, N_SPLITS  # noqa: E402

CURRENT_BEST = 0.5518


def make_variant(name):
    """Return a TransformedTargetRegressor for the named target transform,
    wrapping a *fresh* copy of the winning full pipeline (prep + tuned XGB)."""
    base_pipe = build_full_pipeline()

    if name == "identity":
        return base_pipe  # no wrapper -- sanity check reproduces 0.5518

    if name == "log":
        return TransformedTargetRegressor(regressor=base_pipe, func=np.log, inverse_func=np.exp, check_inverse=False)

    if name == "sqrt":
        return TransformedTargetRegressor(regressor=base_pipe, func=np.sqrt, inverse_func=np.square, check_inverse=False)

    if name == "boxcox":
        return TransformedTargetRegressor(
            regressor=base_pipe,
            transformer=PowerTransformer(method="box-cox", standardize=True),
        )

    if name == "yeo-johnson":
        return TransformedTargetRegressor(
            regressor=base_pipe,
            transformer=PowerTransformer(method="yeo-johnson", standardize=True),
        )

    if name == "quantile-normal":
        # n_quantiles capped below fold-train-size (~970) inside a small helper
        # transformer so cross_val_score's internal fit (on ~970 rows/fold)
        # doesn't warn/clip unexpectedly.
        return TransformedTargetRegressor(
            regressor=base_pipe,
            transformer=QuantileTransformer(
                n_quantiles=200, output_distribution="normal", random_state=RANDOM_STATE
            ),
        )

    raise ValueError(name)


def main():
    t_start = time.time()
    X_train, y_train, train_ids, X_test, test_ids = load_data()
    print(f"Loaded: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"Age range: [{y_train.min()}, {y_train.max()}]  (all > 0 -> Box-Cox is valid)")

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    variants = ["identity", "log", "sqrt", "boxcox", "yeo-johnson", "quantile-normal"]
    results = {}

    for name in variants:
        t0 = time.time()
        model = make_variant(name)
        scores = cross_val_score(model, X_train, y_train, cv=kf, scoring="r2", n_jobs=1)
        elapsed = time.time() - t0
        results[name] = (scores.mean(), scores.std())
        print(
            f"  [{name:16s}] R^2 = {scores.mean():.4f} +/- {scores.std():.4f}  "
            f"(folds: {np.round(scores, 4)})  [{elapsed:.1f}s]"
        )

    print("\n=== Summary (vs current best untransformed XGBoost = {:.4f}) ===".format(CURRENT_BEST))
    for name in variants:
        mean, std = results[name]
        delta = mean - CURRENT_BEST
        flag = " <-- BEATS CURRENT BEST" if mean > CURRENT_BEST else ""
        print(f"  {name:16s}: R^2 = {mean:.4f} +/- {std:.4f}  (delta {delta:+.4f}){flag}")

    best_name = max(results, key=lambda k: results[k][0])
    print(f"\nBest variant: {best_name} (R^2 = {results[best_name][0]:.4f})")
    print(f"\nTotal runtime: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
