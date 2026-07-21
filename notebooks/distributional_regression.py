"""Lane D: distributional regression with a soft cross-entropy/KL target.

Idea: instead of regressing age directly (squared-error loss), bin age
into ~10-12 bins spanning the training range, build a soft target per
sample (a Gaussian kernel centered at the true age, evaluated at the bin
centers and normalized to sum to 1), and train a small MLP to predict a
probability distribution over bins by minimizing cross-entropy against
that soft target. Since cross-entropy = entropy(target) + KL(target ||
pred), and entropy(target) is a per-sample constant (it does not depend
on the model), minimizing cross-entropy is equivalent to minimizing
KL(target || pred) up to that constant -- so this is a genuine
"KL-divergence" distributional regression, just implemented as the
numerically-stable cross-entropy form `-sum(target * log_softmax(logits))`.

A point prediction is decoded as the probability-weighted mean of bin
centers: sum(bin_centers * predicted_probs). This is a soft, ordinal-aware
alternative to a hard classification-into-bins scheme (which would throw
away the fact that adjacent bins are "close" in age).

Leak-free evaluation, mirroring notebooks/outlier_comparison.py: a manual
KFold(5, shuffle=True, random_state=42) loop where imputer / scaler /
variance-threshold / SelectKBest(f_regression, 100) are ALL fit only on
each fold's training rows, and the bin edges / bin centers / Gaussian
kernel sigma are also derived only from the fold's training-y (never from
validation-y or test data). A small held-out slice (15%) of each fold's
training rows is used for a simple early-stopping check on the KL/CE loss,
so we don't need to hand-tune a fixed epoch count.

Compared against the baseline: R^2 = 0.5065 (5-fold CV, GradientBoosting
on SelectKBest-100, median impute). This is an exploratory reframing --
cross-entropy over soft-binned targets does not directly target R^2 the
way squared error does, and bin discretization introduces an information
bottleneck (two ages mapping to very similar soft distributions become
hard to tell apart at decode time) -- so we report honestly whether it
helps or hurts.
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
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import RobustScaler

RANDOM_STATE = 42
N_SPLITS = 5
K_BEST = 100
N_BINS = 11  # ~10-12 bins spanning the training age range, per task spec
MAX_EPOCHS = 150
PATIENCE = 15
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
    """Same steps as src/baseline.py's build_preprocessor, used stage-by-stage
    (not as a single Pipeline) since this experiment needs custom torch
    training logic that doesn't fit inside a Pipeline step."""
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    var_thresh = VarianceThreshold(threshold=1e-8)
    selector = SelectKBest(f_regression, k=K_BEST)
    return imputer, scaler, var_thresh, selector


def make_bin_centers_and_sigma(y_fold_train, n_bins=N_BINS):
    """Bin edges via np.linspace over the fold-train y range only (never
    touches validation or test y) -- avoids any leak of val-fold age info
    into the binning scheme. Sigma is set to half the bin width, a common
    default that gives adjacent bins meaningful overlap without smearing
    the whole range."""
    lo, hi = y_fold_train.min(), y_fold_train.max()
    edges = np.linspace(lo, hi, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    bin_width = edges[1] - edges[0]
    sigma = bin_width / 2.0
    return centers, sigma


def soft_targets(y, centers, sigma):
    """Gaussian kernel centered at true age, evaluated at bin centers,
    normalized to sum to 1 per sample."""
    y = np.asarray(y).reshape(-1, 1)
    centers = centers.reshape(1, -1)
    logits = -0.5 * ((y - centers) / sigma) ** 2
    # subtract max per row for numerical stability before exponentiating
    logits -= logits.max(axis=1, keepdims=True)
    w = np.exp(logits)
    w /= w.sum(axis=1, keepdims=True)
    return w.astype(np.float32)


class MLP(nn.Module):
    def __init__(self, n_in, n_bins):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_bins),
        )

    def forward(self, x):
        return self.net(x)  # raw logits; softmax applied via log_softmax + KLDivLoss


def train_one_fold(X_tr, y_tr, X_val, n_bins, seed):
    """Trains the MLP on this fold's training rows only (with an internal
    85/15 train/early-stop split of those same rows -- no validation-fold
    data touched), returns decoded predictions for X_val."""
    centers, sigma = make_bin_centers_and_sigma(y_tr, n_bins)

    # internal split for early stopping, carved out of fold-train only
    X_fit, X_es, y_fit, y_es = train_test_split(
        X_tr, y_tr, test_size=0.15, random_state=seed
    )

    target_fit = soft_targets(y_fit, centers, sigma)
    target_es = soft_targets(y_es, centers, sigma)

    torch.manual_seed(seed)
    device = torch.device("cpu")
    model = MLP(X_tr.shape[1], n_bins).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    kl_loss = nn.KLDivLoss(reduction="batchmean")

    Xt_fit = torch.tensor(X_fit, dtype=torch.float32)
    Tt_fit = torch.tensor(target_fit, dtype=torch.float32)
    Xt_es = torch.tensor(X_es, dtype=torch.float32)
    Tt_es = torch.tensor(target_es, dtype=torch.float32)

    n = Xt_fit.shape[0]
    best_es_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            xb, tb = Xt_fit[idx], Tt_fit[idx]
            opt.zero_grad()
            logits = model(xb)
            log_probs = torch.log_softmax(logits, dim=1)
            loss = kl_loss(log_probs, tb)  # KL(target || pred); CE = KL + const entropy(target)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            es_logits = model(Xt_es)
            es_log_probs = torch.log_softmax(es_logits, dim=1)
            es_loss = kl_loss(es_log_probs, Tt_es).item()

        if es_loss < best_es_loss - 1e-5:
            best_es_loss = es_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        val_logits = model(torch.tensor(X_val, dtype=torch.float32))
        val_probs = torch.softmax(val_logits, dim=1).numpy()

    preds = val_probs @ centers  # probability-weighted mean of bin centers
    return preds, epoch + 1


def run_cv(X, y, n_bins):
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_scores = []
    fold_epochs = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        imputer, scaler, var_thresh, selector = build_preprocessor()
        Xt = imputer.fit_transform(X_train_fold)
        Xt = scaler.fit_transform(Xt)
        Xt = var_thresh.fit_transform(Xt)
        Xt = selector.fit_transform(Xt, y_train_fold)

        Xv = imputer.transform(X_val_fold)
        Xv = scaler.transform(Xv)
        Xv = var_thresh.transform(Xv)
        Xv = selector.transform(Xv)

        preds, n_epochs = train_one_fold(Xt, y_train_fold, Xv, n_bins, seed=RANDOM_STATE + fold_idx)
        score = r2_score(y_val_fold, preds)
        fold_scores.append(score)
        fold_epochs.append(n_epochs)
        print(f"  Fold {fold_idx}: R^2 = {score:.4f} (trained {n_epochs} epochs)")

    return np.array(fold_scores), fold_epochs


def main():
    t0 = time.time()
    print("Loading data...")
    X, y = load_data()
    print(f"  X shape: {X.shape}, y shape: {y.shape}, age range: [{y.min():.1f}, {y.max():.1f}]")

    print(f"\nRunning leak-free 5-fold CV, distributional regression (n_bins={N_BINS})...")
    scores, epochs = run_cv(X, y, N_BINS)

    print(f"\n  Fold R^2 scores: {np.round(scores, 4).tolist()}")
    print(f"  Mean CV R^2: {scores.mean():.4f} (std {scores.std():.4f})")
    print(f"  Mean epochs trained per fold: {np.mean(epochs):.1f}")

    baseline = 0.5065
    print(f"\nBaseline (GradientBoosting, SelectKBest-100, no removal): R^2 = {baseline}")
    if scores.mean() > baseline:
        print(f"  --> Distributional regression BEATS baseline by {scores.mean() - baseline:.4f}")
    else:
        print(f"  --> Distributional regression UNDERPERFORMS baseline by {baseline - scores.mean():.4f}")

    print(f"\nTotal runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
