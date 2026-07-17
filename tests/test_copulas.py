import numpy as np
import pandas as pd
import pytest

copulas = pytest.importorskip("copulas")

from riskfunctions.models.copulas import all_pairs, fit_archimedean_pair, fit_multivariate_copula, to_uniform
from copulas.univariate import GaussianUnivariate


def _correlated_returns(seed=0, n=400):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = 0.6 * a + rng.normal(size=n) * 0.8
    return pd.DataFrame({"X": a, "Y": b})


def test_to_uniform_is_in_unit_interval():
    rng = np.random.default_rng(2)
    sample = rng.normal(size=200)
    u = to_uniform(sample)
    assert (u > 0).all() and (u < 1).all()


def test_all_pairs_no_duplicates_or_self_pairs():
    pairs = all_pairs(["A", "B", "C"])
    assert pairs == [("A", "B"), ("A", "C"), ("B", "C")]


def test_fit_archimedean_pair_returns_one_fit_per_model():
    returns = _correlated_returns()
    fits = fit_archimedean_pair(returns, ("X", "Y"))
    assert {f.model_name for f in fits} == {"Clayton", "Frank", "Gumbel"}
    for f in fits:
        assert -1 <= f.tau <= 1


def test_fit_multivariate_copula_correlation_reflects_input_correlation():
    returns = _correlated_returns()
    copula, log_likelihood = fit_multivariate_copula(returns, ("X", "Y"), GaussianUnivariate)
    corr = copula.to_dict()["correlation"][0][1]
    assert corr > 0.5  # X, Y constructed with 0.6 correlation coefficient
    assert np.isfinite(log_likelihood)
