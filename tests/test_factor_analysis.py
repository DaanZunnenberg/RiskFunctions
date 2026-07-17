import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
factor_analyzer = pytest.importorskip("factor_analyzer")

from riskfunctions.models.factor_analysis import (
    factor_analysis_loadings,
    fit_factor_analysis,
    fit_pca,
    pca_loadings,
)


def _factor_structured_returns(seed=1, n=300):
    rng = np.random.default_rng(seed)
    factor = rng.normal(size=n)
    return pd.DataFrame({
        "A": factor + rng.normal(scale=.3, size=n),
        "B": factor + rng.normal(scale=.3, size=n),
        "C": rng.normal(size=n),  # uncorrelated with the shared factor
    })


def test_pca_loadings_shape():
    returns = _factor_structured_returns()
    pca = fit_pca(returns, n_components=3)
    loadings = pca_loadings(pca, returns)
    assert loadings.shape == (3, 3)
    assert list(loadings.index) == ["A", "B", "C"]


def test_pca_first_component_dominated_by_correlated_assets():
    returns = _factor_structured_returns()
    pca = fit_pca(returns, n_components=3)
    loadings = pca_loadings(pca, returns)
    # A and B share the common factor and should load more heavily on PC0
    # than the independent asset C.
    assert abs(loadings.loc["A", "PC0"]) > abs(loadings.loc["C", "PC0"])
    assert abs(loadings.loc["B", "PC0"]) > abs(loadings.loc["C", "PC0"])


def test_factor_analysis_loadings_shape_and_rotation_options():
    returns = _factor_structured_returns()
    for rotation in (None, "varimax", "oblimin"):
        fa = fit_factor_analysis(returns, n_factors=2, rotation=rotation)
        loadings = factor_analysis_loadings(fa, returns)
        assert loadings.shape == (3, 2)
