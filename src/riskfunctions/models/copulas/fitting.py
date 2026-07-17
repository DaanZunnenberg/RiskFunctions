"""
Copula fitting: bivariate Archimedean copulas (Clayton, Frank, Gumbel),
bivariate Gaussian / Student-t copulas (via GaussianMultivariate with
different univariate marginals), and full multivariate Gaussian /
Student-t copulas.

Requires the `copulas` package (pip install copulas). Pure fitting logic —
no plotting (see `riskfunctions.models.copulas.plotting`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from copulas.bivariate import Clayton, Frank, Gumbel
from copulas.multivariate import GaussianMultivariate
from copulas.univariate import GaussianUnivariate, StudentTUnivariate

ARCHIMEDEAN_MODELS = (Clayton, Frank, Gumbel)


def to_uniform(sample: np.ndarray) -> np.ndarray:
    """Transform a 1-D sample to uniform [0, 1] using the empirical CDF (rank transform)."""
    return (rankdata(sample) - 0.5) / len(sample)


def to_uniform_pair(returns: pd.DataFrame, pair: tuple) -> np.ndarray:
    """Rank-transform a pair of return columns into a 2-column uniform array."""
    values = returns[list(pair)].to_numpy()
    return np.column_stack((to_uniform(values[:, 0]), to_uniform(values[:, 1])))


def all_pairs(columns: Iterable[str]) -> list:
    """All (a, b) combinations of columns, a before b, without repeats."""
    columns = list(columns)
    return [(a, b) for idx, a in enumerate(columns) for b in columns[idx + 1:]]


def aic(model, data) -> float:
    """AIC for a fitted (bivariate/multivariate) copula model."""
    return 2 - 2 * np.sum(model.log_probability_density(data))


@dataclass
class BivariateFit:
    """Fit summary for a single bivariate copula model on a single asset pair."""

    pair: tuple
    model_name: str
    theta: float | None
    tau: float
    log_likelihood: float
    aic: float | None


def fit_archimedean_pair(returns: pd.DataFrame, pair: tuple) -> list:
    """Fit Clayton, Frank and Gumbel copulas on a single asset pair. Returns a list of BivariateFit."""
    uniform_data = to_uniform_pair(returns, pair)
    results = []
    for model_cls in ARCHIMEDEAN_MODELS:
        fitted = model_cls()
        fitted.fit(uniform_data)
        log_likelihood = np.sum(np.log(fitted.partial_derivative(uniform_data)))
        results.append(BivariateFit(
            pair=pair,
            model_name=model_cls.__name__,
            theta=fitted.theta,
            tau=fitted.tau,
            log_likelihood=log_likelihood,
            aic=aic(fitted, uniform_data),
        ))
    return results


def fit_gaussian_marginal_pair(returns: pd.DataFrame, pair: tuple) -> list:
    """
    Fit bivariate copulas with Gaussian and Student-t marginals (via
    GaussianMultivariate) on a single asset pair. Returns a list of BivariateFit
    (theta and per-model AIC are not applicable for this family; left as None).
    """
    uniform_data = to_uniform_pair(returns, pair)
    results = []
    for model_cls in (GaussianUnivariate, StudentTUnivariate):
        cop = GaussianMultivariate(distribution={pair[0]: model_cls, pair[1]: model_cls})
        cop.fit(uniform_data)
        tau = cop.to_dict()["correlation"][0][1]
        log_likelihood = np.sum(cop.log_probability_density(uniform_data)) / 3
        results.append(BivariateFit(
            pair=pair, model_name=model_cls.__name__, theta=None,
            tau=tau, log_likelihood=log_likelihood, aic=None,
        ))
    return results


def fit_all_bivariate(returns: pd.DataFrame, pairs: list = None) -> list:
    """Fit all bivariate copula families (Archimedean + Gaussian-marginal) across all pairs."""
    pairs = pairs if pairs is not None else all_pairs(returns.columns)
    results = []
    for pair in pairs:
        results.extend(fit_archimedean_pair(returns, pair))
        results.extend(fit_gaussian_marginal_pair(returns, pair))
    return results


def fit_multivariate_copula(returns: pd.DataFrame, assets: tuple, marginal) -> tuple:
    """
    Fit a full multivariate copula (Gaussian correlation structure) across
    `assets`, using `marginal` (GaussianUnivariate or StudentTUnivariate)
    for every asset's marginal distribution.

    Returns (fitted_copula, normalized_log_likelihood).
    """
    subset = returns[list(assets)]
    copula = GaussianMultivariate(distribution={a: marginal for a in assets})
    copula.fit(subset)
    normalized_log_likelihood = np.sum(copula.log_probability_density(subset)) / (1.5 * len(assets))
    return copula, normalized_log_likelihood
