"""
Copula analysis: bivariate Archimedean copulas (Clayton, Frank, Gumbel),
bivariate Gaussian / Student-t copulas (via the `copulas` package's
GaussianMultivariate with different univariate marginals), and a full
multivariate Gaussian / Student-t copula across all 5 assets.

Originally from Assignment_3_clean.ipynb. Requires network access
(yfinance) and the `copulas` package (pip install copulas).
"""

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import rankdata

from copulas.multivariate import GaussianMultivariate
from copulas.univariate import GaussianUnivariate, StudentTUnivariate
from copulas.bivariate import Gumbel, Clayton, Frank


def load_returns(start_date="2010-01-01", end_date="2024-01-01") -> pd.DataFrame:
    """Download AAPL, ASML, AIRBUS, LVMH, EURO50 and return daily log returns."""
    stocks = ["AAPL", "ASML.AS", "AIR.PA", "MC.PA", "SC0D.DE"]
    stock_data = yf.download(" ".join(stocks), start=start_date, end=end_date)
    data = pd.DataFrame(
        stock_data["Adj Close"].values,
        columns=stock_data["Adj Close"].columns,
        index=stock_data.index,
    )
    data = data.rename(columns={"ASML.AS": "ASML", "AIR.PA": "AIRBUS", "MC.PA": "LVMH", "SC0D.DE": "EURO50"})
    returns = data.apply(np.log).diff(1).fillna(0)
    return returns


def to_uniform(data: np.ndarray) -> np.ndarray:
    """Transform a 1-D sample to uniform [0, 1] using the empirical CDF (rank transform)."""
    return (rankdata(data) - 0.5) / len(data)


def AIC(model, data) -> float:
    """AIC for a fitted (bivariate/multivariate) copula model."""
    return 2 - 2 * np.sum(model.log_probability_density(data))


def fit_bivariate_archimedean(returns: pd.DataFrame, pairs=None) -> None:
    """
    For each pair of assets, fit Clayton, Frank and Gumbel copulas (and,
    separately, Gaussian/Student-t marginal-based bivariate copulas via
    GaussianMultivariate) on the rank-transformed returns, printing the
    fitted theta, Kendall's tau, log-likelihood and AIC for each.
    """
    warnings.filterwarnings(action="ignore")
    if pairs is None:
        pairs = [(a, b) for idx, a in enumerate(returns.columns) for b in returns.columns[idx + 1:]]

    for pair in pairs:
        for model in (Clayton, Frank, Gumbel):
            print()
            print(model.__name__)
            print(pair)
            copula_data = returns[list(pair)].to_numpy()
            uniform_data = np.column_stack((to_uniform(copula_data[:, 0]), to_uniform(copula_data[:, 1])))
            fitted = model()
            fitted.fit(uniform_data)
            print(f"Fitted Theta: {fitted.theta}")
            print(f"Fitted Tau: {fitted.tau}")
            print(f"Log likelihood: {np.sum(np.log(fitted.partial_derivative(uniform_data)))}")
            print(f"AIC: {AIC(fitted, uniform_data)}")

    for pair in pairs:
        for model in (GaussianUnivariate, StudentTUnivariate):
            print()
            print(model.__name__)
            print(pair)
            copula_data = returns[list(pair)].to_numpy()
            uniform_data = np.column_stack((to_uniform(copula_data[:, 0]), to_uniform(copula_data[:, 1])))
            cop = GaussianMultivariate(distribution={pair[0]: model, pair[1]: model})
            cop.fit(uniform_data)
            print(f"Fitted Tau (correlation): {cop.to_dict()['correlation'][0][1]}")
            print(f"Log likelihood: {np.sum(cop.log_probability_density(uniform_data)) / 3}")


def plot_bivariate_synthetic_scatter(returns: pd.DataFrame, pairs=None):
    """
    Reproduce the grid of scatter plots comparing synthetic samples from
    Clayton / Frank / Gumbel / Gaussian / Student-t bivariate copulas
    against the raw log-returns scatter, one row per asset pair.
    """
    if pairs is None:
        pairs = [(a, b) for idx, a in enumerate(returns.columns) for b in returns.columns[idx + 1:]]

    fig, axs = plt.subplots(10, 6, figsize=(2 * 8.2543, 3.2 * 8.2543))

    for i, model in enumerate((Clayton, Frank, Gumbel)):
        for idx, (pair, ax) in enumerate(zip(pairs, axs[:, i].flatten())):
            copula_data = returns[list(pair)].to_numpy()
            uniform_data = np.column_stack((to_uniform(copula_data[:, 0]), to_uniform(copula_data[:, 1])))
            fitted = model()
            fitted.fit(uniform_data)
            synthetic_data = fitted.sample(2_000)
            synthetic_data = pd.DataFrame(list(map(np.ravel, synthetic_data)))
            synthetic_data.columns = ["X", "Y"]
            ax.scatter(x=synthetic_data["X"], y=synthetic_data["Y"], s=.1, alpha=.5)
            if i == 0:
                ax.set_ylabel(pair[0] + "-" + pair[1], fontsize=11)
            if idx == 0:
                ax.set_title(model.__name__, fontsize=11)

    for idx, (pair, ax) in enumerate(zip(pairs, axs[:, 3].flatten())):
        copula_data = returns[list(pair)].to_numpy()
        uniform_data = np.column_stack((to_uniform(copula_data[:, 0]), to_uniform(copula_data[:, 1])))
        model = GaussianMultivariate(distribution={pair[0]: GaussianUnivariate, pair[1]: GaussianUnivariate})
        model.fit(uniform_data)
        synthetic_data = model.sample(2_000)
        synthetic_data.columns = ["X", "Y"]
        ax.scatter(x=synthetic_data["X"], y=synthetic_data["Y"], s=.1, alpha=.5)
        if idx == 0:
            ax.set_title("Gaussian", fontsize=11)

    for idx, (pair, ax) in enumerate(zip(pairs, axs[:, 4].flatten())):
        copula_data = returns[list(pair)].to_numpy()
        uniform_data = np.column_stack((to_uniform(copula_data[:, 0]), to_uniform(copula_data[:, 1])))
        model = GaussianMultivariate(distribution={pair[0]: StudentTUnivariate, pair[1]: StudentTUnivariate})
        model.fit(uniform_data)
        synthetic_data = model.sample(2_000)
        synthetic_data.columns = ["X", "Y"]
        ax.scatter(x=synthetic_data["X"], y=synthetic_data["Y"], s=.15, alpha=.75)
        if idx == 0:
            ax.set_title("Student-$t$", fontsize=11)

    for idx, (pair, ax) in enumerate(zip(pairs, axs[:, 5].flatten())):
        copula_data = returns[list(pair)].to_numpy()
        ax.scatter(x=copula_data[:, 0], y=copula_data[:, 1], s=.25, alpha=.9)
        if idx == 0:
            ax.set_title("Log Returns", fontsize=11)

    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    fig.tight_layout()
    plt.show()


def fit_multivariate_copulas(returns: pd.DataFrame, assets=("AAPL", "AIRBUS", "ASML", "LVMH", "EURO50")):
    """
    Fit full multivariate Student-t and Gaussian copulas across all assets,
    printing the fitted correlation matrix, marginal distributions, and
    normalized log-likelihood. Returns (t_student_copula, gaussian_copula).
    """
    multidim_copula_df = returns[list(assets)]

    t_student_copula = GaussianMultivariate(distribution={a: StudentTUnivariate for a in assets})
    t_student_copula.fit(multidim_copula_df)
    t_params = t_student_copula.to_dict()
    print("\nCorrelations (Student-t copula): \n", t_params["correlation"])
    print("Fitted marginals: \n", t_params["univariates"])
    print("Normalized log-likelihood:", np.sum(t_student_copula.log_probability_density(multidim_copula_df)) / 7.5)

    gauss_copula = GaussianMultivariate(distribution={a: GaussianUnivariate for a in assets})
    gauss_copula.fit(multidim_copula_df)
    g_params = gauss_copula.to_dict()
    print("\nCorrelations (Gaussian copula): \n", g_params["correlation"])
    print("Fitted marginals: \n", g_params["univariates"])
    print("Normalized log-likelihood:", np.sum(gauss_copula.log_probability_density(multidim_copula_df)) / 7.5)

    return t_student_copula, gauss_copula


def main():
    returns = load_returns()
    fit_bivariate_archimedean(returns)
    plot_bivariate_synthetic_scatter(returns)
    fit_multivariate_copulas(returns)


if __name__ == "__main__":
    main()
