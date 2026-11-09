"""
Example: fit bivariate Archimedean (Clayton, Frank, Gumbel) and
Gaussian-marginal (Gaussian, Student-t) copulas across every pair of a
5-asset equity portfolio, plot the resulting synthetic scatter grid, and
fit a full multivariate Gaussian/Student-t copula across all 5 assets.

Requires network access (yfinance) and the `copulas` package.

Run from the repo root:
    python examples/copulas_example.py
"""

from copulas.univariate import GaussianUnivariate, StudentTUnivariate

from riskfunctions.models.copulas import (
    fit_all_bivariate,
    fit_multivariate_copula,
    load_returns,
    plot_bivariate_synthetic_scatter,
)

ASSETS = ("AAPL", "AIRBUS", "ASML", "LVMH", "EURO50")


def main():
    returns = load_returns()

    print("Bivariate copula fits (Archimedean + Gaussian-marginal):\n")
    for fit in fit_all_bivariate(returns):
        print(f"{fit.pair} | {fit.model_name} | theta={fit.theta} | tau={fit.tau:.4f} "
              f"| log-likelihood={fit.log_likelihood:.2f} | AIC={fit.aic}")

    plot_bivariate_synthetic_scatter(returns)

    t_copula, t_ll = fit_multivariate_copula(returns, ASSETS, StudentTUnivariate)
    print("\nMultivariate Student-t copula:")
    print("Correlations:\n", t_copula.to_dict()["correlation"])
    print("Normalized log-likelihood:", t_ll)

    gauss_copula, gauss_ll = fit_multivariate_copula(returns, ASSETS, GaussianUnivariate)
    print("\nMultivariate Gaussian copula:")
    print("Correlations:\n", gauss_copula.to_dict()["correlation"])
    print("Normalized log-likelihood:", gauss_ll)


if __name__ == "__main__":
    main()
# Covers copula-based dependence modeling usage.
