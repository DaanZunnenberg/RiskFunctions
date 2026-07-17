"""
Plotting helpers for the copula analysis: a grid of scatter plots comparing
synthetic samples from fitted bivariate copulas against the raw log-returns
scatter, one row per asset pair.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from copulas.multivariate import GaussianMultivariate
from copulas.univariate import GaussianUnivariate, StudentTUnivariate

from riskfunctions.models.copulas.fitting import ARCHIMEDEAN_MODELS, all_pairs, to_uniform_pair

N_SYNTHETIC_SAMPLES = 2_000


def _scatter_archimedean_column(returns, pairs, axs_column, model_cls):
    for idx, (pair, ax) in enumerate(zip(pairs, axs_column)):
        uniform_data = to_uniform_pair(returns, pair)
        fitted = model_cls()
        fitted.fit(uniform_data)
        synthetic = fitted.sample(N_SYNTHETIC_SAMPLES)
        synthetic = pd.DataFrame(list(map(np.ravel, synthetic)), columns=["X", "Y"])
        ax.scatter(x=synthetic["X"], y=synthetic["Y"], s=.1, alpha=.5)
        if idx == 0:
            ax.set_title(model_cls.__name__, fontsize=11)


def _scatter_gaussian_marginal_column(returns, pairs, axs_column, marginal_cls, point_style):
    for idx, (pair, ax) in enumerate(zip(pairs, axs_column)):
        uniform_data = to_uniform_pair(returns, pair)
        model = GaussianMultivariate(distribution={pair[0]: marginal_cls, pair[1]: marginal_cls})
        model.fit(uniform_data)
        synthetic = model.sample(N_SYNTHETIC_SAMPLES)
        synthetic.columns = ["X", "Y"]
        ax.scatter(x=synthetic["X"], y=synthetic["Y"], **point_style)
        if idx == 0:
            ax.set_title(marginal_cls.__name__.replace("Univariate", ""), fontsize=11)


def _scatter_raw_returns_column(returns, pairs, axs_column):
    for idx, (pair, ax) in enumerate(zip(pairs, axs_column)):
        values = returns[list(pair)].to_numpy()
        ax.scatter(x=values[:, 0], y=values[:, 1], s=.25, alpha=.9)
        if idx == 0:
            ax.set_title("Log Returns", fontsize=11)


def plot_bivariate_synthetic_scatter(returns: pd.DataFrame, pairs: list = None):
    """
    Reproduce the grid of scatter plots comparing synthetic samples from
    Clayton / Frank / Gumbel / Gaussian / Student-t bivariate copulas
    against the raw log-returns scatter, one row per asset pair.
    """
    pairs = pairs if pairs is not None else all_pairs(returns.columns)

    n_rows, n_cols = len(pairs), len(ARCHIMEDEAN_MODELS) + 3
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(2 * 8.2543, 3.2 * 8.2543))

    for col, model_cls in enumerate(ARCHIMEDEAN_MODELS):
        _scatter_archimedean_column(returns, pairs, axs[:, col], model_cls)
        if col == 0:
            for idx, pair in enumerate(pairs):
                axs[idx, col].set_ylabel(f"{pair[0]}-{pair[1]}", fontsize=11)

    _scatter_gaussian_marginal_column(returns, pairs, axs[:, len(ARCHIMEDEAN_MODELS)], GaussianUnivariate, {"s": .1, "alpha": .5})
    _scatter_gaussian_marginal_column(returns, pairs, axs[:, len(ARCHIMEDEAN_MODELS) + 1], StudentTUnivariate, {"s": .15, "alpha": .75})
    _scatter_raw_returns_column(returns, pairs, axs[:, len(ARCHIMEDEAN_MODELS) + 2])

    plt.setp(fig.get_axes(), xticks=[], yticks=[])
    fig.tight_layout()
    plt.show()
