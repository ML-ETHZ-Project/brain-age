import os

import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")

X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"))
X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
sample = pd.read_csv(os.path.join(DATA_DIR, "sample.csv"))

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("sample shape:", sample.shape)
print()
print("X_train columns sample:", list(X_train.columns[:5]), "...", list(X_train.columns[-3:]))
print("y_train columns:", list(y_train.columns))
print("sample columns:", list(sample.columns))
print()

# id alignment
print("X_train id matches y_train id:", (X_train['id'].values == y_train['id'].values).all())
print()

feat_cols = [c for c in X_train.columns if c != 'id']
print("n features:", len(feat_cols))

# Missing values
nan_counts = X_train[feat_cols].isna().sum()
print("\nColumns with missing values:", (nan_counts > 0).sum(), "/", len(feat_cols))
print("Total NaNs in X_train:", nan_counts.sum(), "  fraction:", nan_counts.sum() / (X_train.shape[0]*len(feat_cols)))
print("Max NaNs in a single column:", nan_counts.max())
print("Rows with at least one NaN:", X_train[feat_cols].isna().any(axis=1).sum(), "/", X_train.shape[0])

nan_counts_test = X_test[feat_cols].isna().sum()
print("\nTest set total NaNs:", nan_counts_test.sum())

# y distribution
print("\ny stats:")
print(y_train['y'].describe())

# feature scale variety
print("\nFeature value ranges (first 10 features):")
print(X_train[feat_cols[:10]].describe().T[['mean','std','min','max']])

# Look for outlier-like features (huge scale differences)
stds = X_train[feat_cols].std()
print("\nStd range across features: min", stds.min(), "max", stds.max())
print("Top 10 highest-std features:", stds.sort_values(ascending=False).head(10))

# correlation of each feature with age (ignoring NaNs)
corrs = X_train[feat_cols].corrwith(y_train['y'])
print("\nTop 15 features by abs corr with age:")
print(corrs.abs().sort_values(ascending=False).head(15))
print("\nNumber of features with |corr| < 0.02 (candidate irrelevant):", (corrs.abs() < 0.02).sum())
