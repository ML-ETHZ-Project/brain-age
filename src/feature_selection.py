"""Redundancy-aware feature selector: SelectKBest ranking + a correlation cutoff.

notebooks/feature_visualization.ipynb found that several of the top age-correlated
features are themselves highly inter-correlated (e.g. x133/x334/x465 all pairwise
|r| > 0.9), so a plain SelectKBest(k=100) can spend its budget on near-duplicate
columns. CorrelationPrunedKBest ranks features the same way SelectKBest does, then
greedily walks that ranking and skips any feature too correlated with one already
kept, so the final set carries more distinct signal per feature.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_regression


class CorrelationPrunedKBest(BaseEstimator, TransformerMixin):
    """Rank features by score_func, then greedily drop redundant ones.

    Walks the score-ranked feature list best-first, keeping a feature only if its
    correlation with every already-kept feature is at most `corr_threshold`, until
    `n_features` are kept. If the threshold is too strict to reach `n_features`
    (candidates exhausted), backfills with the next-highest-scoring unpicked
    features regardless of correlation, so the output always has `n_features` columns.
    """

    def __init__(self, score_func=f_regression, n_features=25, corr_threshold=0.9):
        self.score_func = score_func
        self.n_features = n_features
        self.corr_threshold = corr_threshold

    def fit(self, X, y):
        X = np.asarray(X)
        scores = self.score_func(X, y)[0]
        ranked = np.argsort(scores)[::-1]

        corr = np.nan_to_num(np.corrcoef(X, rowvar=False), nan=0.0)

        selected = []
        for idx in ranked:
            if not selected or np.all(np.abs(corr[idx, selected]) <= self.corr_threshold):
                selected.append(idx)
            if len(selected) == self.n_features:
                break

        if len(selected) < self.n_features:
            for idx in ranked:
                if idx not in selected:
                    selected.append(idx)
                if len(selected) == self.n_features:
                    break

        self.selected_indices_ = np.array(selected)
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.selected_indices_]
