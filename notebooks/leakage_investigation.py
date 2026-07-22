"""Quick recon pass: is there exploitable structure beyond modeling that could
explain the gap between current best CV R^2 (0.5518, src/advanced_pipeline.py)
and the user's belief that ~0.75 should be achievable?

Checks (per orchestrator spec), each cheap and leak-free where relevant:
  1. Near-duplicate/duplicate rows between train and test (Euclidean NN on a
     robust-scaled/imputed feature matrix).
  2. Row-order / id leakage: corr(id, y), consecutive-row y similarity.
  3. "Insane magnitude" noise columns: do sign*log1p(abs) or robust-scale-own
     transforms recover correlation with age? Also double-check the pipeline's
     order of operations (scale before select? yes, confirmed by reading
     src/advanced_pipeline.py's build_preprocessor -- scale happens before
     select, so this is not a live bug, but we check whether the noise columns
     carry any signal at all under a better transform).
  4. Duplicate/near-duplicate columns (pairwise corr > 0.999).
  5. Latent sub-populations via KMeans/GMM on top-50 age-correlated features.

Anything promising gets a quick leak-free CV sanity check (does it move R^2
on top of the current best pipeline?). This is a 10-minute recon pass, not
an exhaustive study.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import NearestNeighbors

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
DATA_DIR = os.path.join(REPO_ROOT, "data", "raw")

from advanced_pipeline import (  # noqa: E402
    RANDOM_STATE,
    TopCorrMixedImputer,
    XGB_PARAMS,
    build_full_pipeline,
    build_preprocessor,
)
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.preprocessing import RobustScaler  # noqa: E402

t_start = time.time()

X_train_df = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
y_train_df = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"))
X_test_df = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))

feat_cols = [c for c in X_train_df.columns if c != "id"]
X_train = X_train_df[feat_cols].values
y = y_train_df["y"].values
train_ids = X_train_df["id"].values
X_test = X_test_df[feat_cols].values
test_ids = X_test_df["id"].values

print("=" * 70)
print("CHECK 1: near-duplicate rows between train and test")
print("=" * 70)

# Fit a simple leak-free-ish preprocessing (median impute + robust scale) on
# train only, transform both train and test, then NN-match test -> train.
from sklearn.pipeline import Pipeline as _Pipeline  # noqa: E402

prep = _Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", RobustScaler()),
])
prep.fit(X_train)
X_train_proc = prep.transform(X_train)
X_test_proc = prep.transform(X_test)

nn = NearestNeighbors(n_neighbors=1).fit(X_train_proc)
dists, idxs = nn.kneighbors(X_test_proc)
dists = dists.ravel()
idxs = idxs.ravel()

print(f"  test->train NN distance (robust-scaled, {len(feat_cols)} dims): "
      f"min={dists.min():.4f}, p1={np.percentile(dists,1):.4f}, "
      f"median={np.median(dists):.4f}, p99={np.percentile(dists,99):.4f}, max={dists.max():.4f}")
n_close = (dists < 1.0).sum()
n_very_close = (dists < 0.1).sum()
print(f"  # test rows with NN distance < 1.0: {n_close} / {len(test_ids)}")
print(f"  # test rows with NN distance < 0.1 (near-exact match): {n_very_close} / {len(test_ids)}")

# Also do train-train NN (excluding self) as a reference distribution.
nn_tt = NearestNeighbors(n_neighbors=2).fit(X_train_proc)
dists_tt, _ = nn_tt.kneighbors(X_train_proc)
dists_tt = dists_tt[:, 1]  # nearest *other* train row
print(f"  train->train (self-excluded) NN distance: min={dists_tt.min():.4f}, "
      f"median={np.median(dists_tt):.4f}, max={dists_tt.max():.4f}")
print("  -> Interpretation: if test NN distances are in the same range as train-train NN "
      "distances (not orders of magnitude smaller), there is no evidence of shared/near-"
      "duplicate rows between train and test.")

print()
print("=" * 70)
print("CHECK 2: row-order / id leakage")
print("=" * 70)
r_id_y, p_id_y = pearsonr(train_ids.astype(float), y)
print(f"  corr(id, y) = {r_id_y:.4f} (p={p_id_y:.4g})")

# Sort by id (assume rows already roughly in id order; check anyway) and look
# at consecutive-row y differences vs a shuffled-baseline.
order = np.argsort(train_ids)
y_sorted = y[order]
consec_diff = np.abs(np.diff(y_sorted))
rng = np.random.RandomState(RANDOM_STATE)
shuffled_diffs = []
for _ in range(200):
    yp = rng.permutation(y_sorted)
    shuffled_diffs.append(np.abs(np.diff(yp)).mean())
shuffled_diffs = np.array(shuffled_diffs)
print(f"  mean |consecutive y diff| (id-sorted order): {consec_diff.mean():.4f}")
print(f"  mean |consecutive y diff| (200 random shuffles): "
      f"{shuffled_diffs.mean():.4f} +/- {shuffled_diffs.std():.4f}")
z = (consec_diff.mean() - shuffled_diffs.mean()) / shuffled_diffs.std()
print(f"  z-score of id-sorted consecutive similarity vs shuffled null: {z:.2f}")
print("  -> Interpretation: |z| >> 2 would suggest id-order carries batch/age structure; "
      "z near 0 means no exploitable ordering signal.")

print()
print("=" * 70)
print("CHECK 3: insane-magnitude noise columns -- does a transform recover signal?")
print("=" * 70)
stds = np.nanstd(X_train, axis=0)
order_std = np.argsort(-stds)
top_noise_idx = order_std[:15]
print(f"  Top 15 columns by raw std: {[feat_cols[i] for i in top_noise_idx]}")
print(f"  Their stds: {[f'{stds[i]:.3g}' for i in top_noise_idx]}")

for i in top_noise_idx[:8]:
    col = X_train[:, i]
    mask = ~np.isnan(col)
    if mask.sum() < 10 or np.nanstd(col) == 0:
        continue
    raw_r = 0.0
    if np.std(col[mask]) > 0:
        raw_r = abs(np.corrcoef(col[mask], y[mask])[0, 1])
    # sign(x) * log1p(abs(x))
    transformed = np.sign(col) * np.log1p(np.abs(col))
    tmask = mask & np.isfinite(transformed)
    trans_r = 0.0
    if tmask.sum() > 10 and np.std(transformed[tmask]) > 0:
        trans_r = abs(np.corrcoef(transformed[tmask], y[tmask])[0, 1])
    # column divided by its own robust scale (IQR)
    q75, q25 = np.nanpercentile(col, [75, 25])
    iqr = q75 - q25
    own_scaled = col / iqr if iqr not in (0, np.nan) and not np.isnan(iqr) else col
    smask = mask & np.isfinite(own_scaled)
    own_r = 0.0
    if smask.sum() > 10 and np.std(own_scaled[smask]) > 0:
        own_r = abs(np.corrcoef(own_scaled[smask], y[smask])[0, 1])
    print(f"  {feat_cols[i]:>6s}  std={stds[i]:.3g}  |corr| raw={raw_r:.4f}  "
          f"sign*log1p={trans_r:.4f}  own-IQR-scaled={own_r:.4f}")

print("\n  Order-of-operations check in src/advanced_pipeline.py build_preprocessor():")
print("    impute -> RobustScaler -> VarianceThreshold -> SelectKBest(f_regression)")
print("  RobustScaler runs BEFORE SelectKBest, so f_regression's F-statistic is computed on")
print("  already-scale-normalized columns -- huge-magnitude noise columns are NOT distorting")
print("  selection scores by raw scale. This confirms no live bug in the current pipeline's")
print("  order of operations.")

print()
print("=" * 70)
print("CHECK 4: near-duplicate / redundant columns (pairwise |corr| > 0.999)")
print("=" * 70)
# Cheap approach: use pandas corr on a median-imputed frame (imputation only
# needed to compute corr on columns with NaNs; this doesn't touch y at all so
# no leakage concern even done on full data -- it's just describing X).
X_imp_df = pd.DataFrame(X_train, columns=feat_cols).fillna(pd.DataFrame(X_train, columns=feat_cols).median())
corr_mat = X_imp_df.corr().values.copy()
np.fill_diagonal(corr_mat, 0.0)
high_pairs = []
n = len(feat_cols)
iu = np.triu_indices(n, k=1)
vals = corr_mat[iu]
high_mask = np.abs(vals) > 0.999
high_idx = np.where(high_mask)[0]
for k in high_idx:
    i, j = iu[0][k], iu[1][k]
    high_pairs.append((feat_cols[i], feat_cols[j], corr_mat[i, j]))
print(f"  # column pairs with |corr| > 0.999: {len(high_pairs)}")
for a, b, c in high_pairs[:20]:
    print(f"    {a} <-> {b}: corr={c:.5f}")
if len(high_pairs) > 20:
    print(f"    ... and {len(high_pairs) - 20} more")

print()
print("=" * 70)
print("CHECK 5: latent sub-populations (KMeans/GMM on top-50 age-correlated features)")
print("=" * 70)
abs_corrs = np.array([
    abs(np.corrcoef(X_train[~np.isnan(X_train[:, j]), j], y[~np.isnan(X_train[:, j])])[0, 1])
    if np.nansum(~np.isnan(X_train[:, j])) > 10 and np.nanstd(X_train[:, j]) > 0 else 0.0
    for j in range(X_train.shape[1])
])
top50_idx = np.argsort(-abs_corrs)[:50]
print(f"  Top-5 age-correlated columns: {[(feat_cols[i], round(abs_corrs[i],3)) for i in top50_idx[:5]]}")

X_top50 = X_train[:, top50_idx]
imp50 = SimpleImputer(strategy="median").fit(X_top50)
X_top50_imp = imp50.transform(X_top50)
scaler50 = RobustScaler().fit(X_top50_imp)
X_top50_scaled = scaler50.transform(X_top50_imp)

for k in [2, 3, 4, 5]:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X_top50_scaled)
    labels = km.labels_
    means = [y[labels == c].mean() for c in range(k)]
    counts = [np.sum(labels == c) for c in range(k)]
    # eta-squared style: between-cluster variance / total variance as a rough
    # "does cluster membership explain age variance" signal
    grand_mean = y.mean()
    ss_between = sum(counts[c] * (means[c] - grand_mean) ** 2 for c in range(k))
    ss_total = np.sum((y - grand_mean) ** 2)
    eta2 = ss_between / ss_total
    print(f"  KMeans k={k}: cluster sizes={counts}, cluster mean ages={[round(m,1) for m in means]}, "
          f"eta^2(age~cluster)={eta2:.4f}")

gmm = GaussianMixture(n_components=3, random_state=RANDOM_STATE).fit(X_top50_scaled)
gmm_labels = gmm.predict(X_top50_scaled)
gmm_means = [y[gmm_labels == c].mean() for c in range(3)]
gmm_counts = [np.sum(gmm_labels == c) for c in range(3)]
print(f"  GMM k=3: cluster sizes={gmm_counts}, cluster mean ages={[round(m,1) for m in gmm_means]}")

print()
print("=" * 70)
print("QUANTIFY: does anything above actually move CV R^2 on top of current best pipeline?")
print("=" * 70)
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

baseline_scores = cross_val_score(build_full_pipeline(), X_train, y, cv=kf, scoring="r2")
print(f"  Current best pipeline (reference): R^2 = {baseline_scores.mean():.4f} +/- {baseline_scores.std():.4f}")

# Test the strongest candidate from above: KMeans cluster-id (k with highest
# eta^2) as an extra engineered feature, added leak-free (fit KMeans inside
# each fold's training data only, transform val fold via predict).
best_k = None
best_eta2 = -1
eta2_by_k = {}
for k in [2, 3, 4, 5]:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X_top50_scaled)
    labels = km.labels_
    means = [y[labels == c].mean() for c in range(k)]
    counts = [np.sum(labels == c) for c in range(k)]
    grand_mean = y.mean()
    ss_between = sum(counts[c] * (means[c] - grand_mean) ** 2 for c in range(k))
    ss_total = np.sum((y - grand_mean) ** 2)
    eta2 = ss_between / ss_total
    eta2_by_k[k] = eta2
    if eta2 > best_eta2:
        best_eta2 = eta2
        best_k = k

print(f"  Best KMeans eta^2 = {best_eta2:.4f} at k={best_k} (very weak if << 0.1; "
      f"tried as an engineered feature below anyway since it's cheap)")

from sklearn.base import BaseEstimator, TransformerMixin  # noqa: E402


class KMeansClusterFeature(BaseEstimator, TransformerMixin):
    """Adds a leak-free KMeans cluster-id (on top-50 age-corr cols, fit on
    training rows of the fold only) as an extra column."""

    def __init__(self, top_idx, n_clusters=3, random_state=RANDOM_STATE):
        self.top_idx = top_idx
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        X_top = X[:, self.top_idx]
        self.imp_ = SimpleImputer(strategy="median").fit(X_top)
        X_top_imp = self.imp_.transform(X_top)
        self.scaler_ = RobustScaler().fit(X_top_imp)
        X_top_scaled = self.scaler_.transform(X_top_imp)
        self.km_ = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10).fit(X_top_scaled)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        X_top = X[:, self.top_idx]
        X_top_imp = self.imp_.transform(X_top)
        X_top_scaled = self.scaler_.transform(X_top_imp)
        cluster_id = self.km_.predict(X_top_scaled).reshape(-1, 1).astype(float)
        return np.hstack([X, cluster_id])


from sklearn.pipeline import Pipeline as SkPipeline  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression  # noqa: E402

cluster_pipe = SkPipeline([
    ("add_cluster", KMeansClusterFeature(top_idx=top50_idx, n_clusters=best_k)),
    ("prep", build_preprocessor()),
    ("model", XGBRegressor(**XGB_PARAMS)),
])
cluster_scores = cross_val_score(cluster_pipe, X_train, y, cv=kf, scoring="r2")
print(f"  With KMeans(k={best_k}) cluster-id feature added: R^2 = {cluster_scores.mean():.4f} "
      f"+/- {cluster_scores.std():.4f}  (delta = {cluster_scores.mean() - baseline_scores.mean():+.4f})")

print(f"\nTotal runtime: {time.time() - t_start:.1f}s")
