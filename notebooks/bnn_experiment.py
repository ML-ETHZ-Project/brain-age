"""Lane F: lightweight Bayesian neural network via MC-Dropout.

A full variational/Pyro-style BNN is overkill for n=1212 tabular rows and
832 features. MC-Dropout is a cheap, well-established approximation: train
a small feed-forward net with dropout as usual (MSE loss, point
predictions), but at *inference* time leave dropout ACTIVE and run many
stochastic forward passes per sample. The mean across passes is the point
prediction; the std across passes is a predictive-uncertainty estimate
(this approximates sampling from the posterior over network weights, per
Gal & Ghahramani 2016).

Leak-free evaluation mirrors notebooks/outlier_comparison.py: a manual
KFold(5, shuffle=True, random_state=42) loop where the preprocessing
pipeline (median-impute -> RobustScaler -> VarianceThreshold -> SelectKBest
f_regression k=100) is fit on each fold's training rows only, then applied
to that fold's validation rows. The NN is trained fresh per fold on the
fold's preprocessed training data.

This lane is NOT primarily about beating the GradientBoosting baseline's
R^2 = 0.5065 (a small MLP on 1212 rows is expected to underperform
boosting on tabular data) -- it's about checking whether the MC-Dropout
uncertainty estimate is *calibrated*: do validation rows with a larger
predicted std tend to have larger absolute prediction error? We report
both the CV R^2 of the MC-Dropout mean prediction and the pooled
error-vs-std correlation across all 5 folds' validation rows.
"""
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

RANDOM_STATE = 42
N_SPLITS = 5
K_BEST = 100
DROPOUT_P = 0.3
MC_PASSES = 50
MAX_EPOCHS = 200
PATIENCE = 15  # early-stop patience on a held-out slice of the fold's training rows
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data", "raw")

torch.manual_seed(RANDOM_STATE)


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


class MCDropoutNet(nn.Module):
    """Linear(100->128)-ReLU-Dropout(0.3)-Linear(128->64)-ReLU-Dropout(0.3)-Linear(64->1)."""

    def __init__(self, in_dim, p=DROPOUT_P):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_fold_model(X_tr, y_tr, fold_seed):
    """Train an MCDropoutNet on this fold's (already preprocessed) training
    rows, with a small internal train/early-stop split carved out of the
    training rows only (never touches validation rows)."""
    torch.manual_seed(fold_seed)

    # Internal split for early stopping, fit only on this fold's training rows.
    n = X_tr.shape[0]
    rng = np.random.RandomState(fold_seed)
    perm = rng.permutation(n)
    n_holdout = max(1, int(0.15 * n))
    holdout_idx, fit_idx = perm[:n_holdout], perm[n_holdout:]

    # Standardize targets for stable training (undone at prediction time).
    y_mean, y_std = y_tr[fit_idx].mean(), y_tr[fit_idx].std()

    X_fit_t = torch.tensor(X_tr[fit_idx], dtype=torch.float32)
    y_fit_t = torch.tensor((y_tr[fit_idx] - y_mean) / y_std, dtype=torch.float32)
    X_hold_t = torch.tensor(X_tr[holdout_idx], dtype=torch.float32)
    y_hold_t = torch.tensor((y_tr[holdout_idx] - y_mean) / y_std, dtype=torch.float32)

    model = MCDropoutNet(X_tr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    best_hold_loss = np.inf
    best_state = None
    epochs_no_improve = 0

    n_fit = X_fit_t.shape[0]
    for epoch in range(MAX_EPOCHS):
        model.train()
        epoch_perm = torch.randperm(n_fit)
        for start in range(0, n_fit, BATCH_SIZE):
            idx = epoch_perm[start:start + BATCH_SIZE]
            xb, yb = X_fit_t[idx], y_fit_t[idx]
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            hold_pred = model(X_hold_t)
            hold_loss = loss_fn(hold_pred, y_hold_t).item()

        if hold_loss < best_hold_loss - 1e-5:
            best_hold_loss = hold_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, y_mean, y_std


def mc_dropout_predict(model, X, y_mean, y_std, n_passes=MC_PASSES, seed=0):
    """Run n_passes stochastic forward passes with dropout ACTIVE (train mode)
    and return (mean_pred, std_pred) in original target units."""
    torch.manual_seed(seed)
    model.train()  # keep dropout active
    X_t = torch.tensor(X, dtype=torch.float32)
    preds = np.zeros((n_passes, X.shape[0]), dtype=np.float64)
    with torch.no_grad():
        for i in range(n_passes):
            out = model(X_t).numpy()
            preds[i] = out * y_std + y_mean
    return preds.mean(axis=0), preds.std(axis=0)


def main():
    t0 = time.time()
    X, y = load_data()
    print(f"Loaded: X {X.shape}, y {y.shape}")
    print(f"MC-Dropout net: 100->128-ReLU-Drop({DROPOUT_P})->64-ReLU-Drop({DROPOUT_P})->1, "
          f"{MC_PASSES} MC passes, max {MAX_EPOCHS} epochs (patience {PATIENCE})")

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    fold_scores = []
    all_val_errors = []
    all_val_stds = []

    for fold_i, (train_idx, val_idx) in enumerate(kf.split(X)):
        fold_seed = RANDOM_STATE + fold_i
        X_tr_raw, X_val_raw = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        prep = build_preprocessor()
        X_tr = prep.fit_transform(X_tr_raw, y_tr)
        X_val = prep.transform(X_val_raw)

        model, y_mean, y_std = train_fold_model(X_tr, y_tr, fold_seed)

        val_mean, val_std = mc_dropout_predict(model, X_val, y_mean, y_std, seed=fold_seed + 1000)

        r2 = r2_score(y_val, val_mean)
        fold_scores.append(r2)

        errors = np.abs(y_val - val_mean)
        all_val_errors.append(errors)
        all_val_stds.append(val_std)

        print(f"  Fold {fold_i}: R^2 = {r2:.4f}  (n_train={len(train_idx)}, n_val={len(val_idx)}, "
              f"mean predictive std={val_std.mean():.3f})")

    fold_scores = np.array(fold_scores)
    all_val_errors = np.concatenate(all_val_errors)
    all_val_stds = np.concatenate(all_val_stds)

    calib_corr = np.corrcoef(all_val_errors, all_val_stds)[0, 1]

    print("\n--- Summary ---")
    print(f"MC-Dropout BNN: CV R^2 = {fold_scores.mean():.4f} +/- {fold_scores.std():.4f}  "
          f"(folds: {np.round(fold_scores, 4)})")
    print(f"Baseline (GradientBoosting, SelectKBest-100): CV R^2 = 0.5065")
    print(f"Beats baseline: {fold_scores.mean() > 0.5065}")
    print(f"\nCalibration check (pooled across all {len(all_val_errors)} validation rows):")
    print(f"  corr(|prediction error|, predicted std) = {calib_corr:.4f}")
    if calib_corr > 0:
        print("  Positive correlation: higher predicted uncertainty tends to coincide with larger errors "
              "(well-calibrated direction).")
    else:
        print("  Non-positive correlation: MC-Dropout uncertainty does NOT track error magnitude well here.")

    print(f"\nTotal runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
