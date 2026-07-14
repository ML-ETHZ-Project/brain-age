"""Feature-engineering strategies vs the SelectKBest-100 baseline (R^2=0.5065).

The columns are anonymized (x0..x831), so domain-named engineering (volume /
intracranial-volume, named L/R asymmetry) isn't possible. Instead we exploit
the *structure* a quick probe revealed:
  - redundancy: 200 feature-pairs with |r|>0.9, 752 with |r|>0.8  -> merging
    correlated features (FeatureAgglomeration) should denoise + de-duplicate
    (also serves subtask 2's "redundant features").
  - flat PCA spectrum: 100 components explain only ~53% variance -> the
    injected noise spreads variance everywhere, so aggressive PCA reduction
    likely discards signal (tested anyway, expected weak).
  - trees already model interactions, but making the top interactions
    explicit (PolynomialFeatures on the top-k) can still surface signal.

Every strategy is the SAME cleaning stack (median-impute -> RobustScaler ->
+/-15-IQR clip -> drop constants) plus a strategy-specific block, then a
default GradientBoostingRegressor -- so the ONLY thing that varies is the
engineered feature set, directly comparable to the 0.5065 baseline. All steps
live in the pipeline, so cross_val_score refits them per fold (leak-free;
SelectKBest/agglomeration/PCA never see held-out rows).
"""
import warnings

import numpy as np
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import FeatureUnion, FunctionTransformer, Pipeline
from sklearn.preprocessing import PolynomialFeatures, RobustScaler

from stacking_comparison import CLIP_IQR, N_SPLITS, RANDOM_STATE, load_data

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def cleaning():
    """Target-unaware cleaning shared by every strategy."""
    return [
        ("impute", SimpleImputer(strategy="median")),
        ("scale", RobustScaler()),
        ("clip", FunctionTransformer(
            np.clip, kw_args={"a_min": -CLIP_IQR, "a_max": CLIP_IQR})),
        ("var", VarianceThreshold(threshold=1e-8)),
    ]


def kbest(k):
    return SelectKBest(f_regression, k=k)


# Each strategy = list of post-cleaning steps inserted before the model.
def strategies():
    return {
        "control: KBest(100)": [
            ("select", kbest(100))],
        "KBest(300)": [
            ("select", kbest(300))],
        "FeatureAgglo(100)": [
            ("agglo", FeatureAgglomeration(n_clusters=100))],
        "FeatureAgglo(250)+KBest(100)": [
            ("agglo", FeatureAgglomeration(n_clusters=250)), ("select", kbest(100))],
        "KBest(25)+Poly2int+KBest(100)": [
            ("s1", kbest(25)),
            ("poly", PolynomialFeatures(2, interaction_only=True, include_bias=False)),
            ("s2", kbest(100))],
        "KBest(30)+Poly2full+KBest(100)": [
            ("s1", kbest(30)),
            ("poly", PolynomialFeatures(2, include_bias=False)),
            ("s2", kbest(100))],
        "KBest(100) union PCA(40)": [
            ("union", FeatureUnion([
                ("kbest", kbest(100)),
                ("pca", PCA(n_components=40, random_state=RANDOM_STATE))]))],
    }


def evaluate(name, steps):
    pipe = Pipeline(cleaning() + steps + [
        ("model", GradientBoostingRegressor(random_state=RANDOM_STATE))])
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    X, y = evaluate.data
    scores = cross_val_score(pipe, X, y, cv=kf, scoring="r2", n_jobs=-1)
    print(f"{name:>34}: R^2 = {scores.mean():.4f} +/- {scores.std():.4f}  "
          f"(folds: {np.round(scores, 3)})")
    return scores.mean()


def main():
    X, y = load_data()
    evaluate.data = (X, y)
    print(f"X {X.shape}   baseline to beat: R^2 = 0.5065 (control below reproduces it)\n")

    results = {name: evaluate(name, steps) for name, steps in strategies().items()}

    control = results["control: KBest(100)"]
    best = max(results, key=results.get)
    print(f"\nBest strategy: {best} (R^2 = {results[best]:.4f}, "
          f"{results[best] - control:+.4f} vs control)")


if __name__ == "__main__":
    main()
