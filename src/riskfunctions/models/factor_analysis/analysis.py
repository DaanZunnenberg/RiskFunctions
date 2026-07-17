"""
PCA and Factor Analysis fitting (no plotting -- see
`riskfunctions.models.factor_analysis.plotting`).
"""

from __future__ import annotations

import pandas as pd
import sklearn.decomposition as skd
from factor_analyzer import FactorAnalyzer


def fit_pca(returns: pd.DataFrame, n_components: int) -> skd.PCA:
    """Fit PCA with `n_components` on `returns`."""
    pca = skd.PCA(n_components=n_components)
    pca.fit_transform(returns)
    return pca


def pca_loadings(pca: skd.PCA, returns: pd.DataFrame) -> pd.DataFrame:
    """Return the PCA loadings (components x assets) as a labeled DataFrame."""
    n_components = pca.components_.shape[0]
    columns = [f"PC{i}" for i in range(n_components)]
    return pd.DataFrame(pca.components_.T, columns=columns, index=returns.columns)


def fit_factor_analysis(returns: pd.DataFrame, n_factors: int, rotation: str = None) -> FactorAnalyzer:
    """
    Fit a FactorAnalyzer model with the given rotation:
    - None: no rotation
    - 'varimax': orthogonal rotation
    - 'oblimin': oblique rotation
    """
    fa = FactorAnalyzer(n_factors, rotation=rotation)
    fa.fit(returns)
    return fa


def factor_analysis_loadings(fa: FactorAnalyzer, returns: pd.DataFrame) -> pd.DataFrame:
    """Return the Factor Analysis loadings (factors x assets) as a labeled DataFrame."""
    n_factors = fa.loadings_.shape[1]
    columns = [f"FA{i}" for i in range(1, n_factors + 1)]
    return pd.DataFrame(fa.loadings_, columns=columns, index=returns.columns)
