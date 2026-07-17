"""
Example: Variance-Covariance (Delta-Normal) VaR/ES on the stock+Euribor
portfolio, backtested year-by-year at alpha = 0.95 and 0.99, for both a
Normal and a Student-t (dof 4, 5, 6) distributional assumption.

Run from the repo root:
    python examples/var_covariance_example.py
"""

import scipy.stats as stats

from riskfunctions.config import DEFAULT_BACKTEST
from riskfunctions.models.var_covariance import Var_Cov_ES, Var_Cov_VaR
from riskfunctions.pipeline import backtest_by_year, load_portfolio_sample

# (label, quantile function for VaR, distribution object for ES, extra kwargs)
DISTRIBUTIONS = [
    ("Normal", stats.norm.ppf, stats.norm, {}),
    ("t-Student dof=4", stats.t.ppf, stats.t, {"df": 4}),
    ("t-Student dof=5", stats.t.ppf, stats.t, {"df": 5}),
    ("t-Student dof=6", stats.t.ppf, stats.t, {"df": 6}),
]


def main():
    sample = load_portfolio_sample()

    for alpha in DEFAULT_BACKTEST.alphas:
        for label, var_dist, es_dist, kwargs in DISTRIBUTIONS:
            estimate = Var_Cov_VaR(
                sample.ind_loss,
                window_size=sample.window_size,
                port_weights=sample.weights,
                alpha=alpha,
                dist=var_dist,
                **kwargs,
            )
            es = Var_Cov_ES(estimate.Vol, alpha=alpha, dist=es_dist, **kwargs)

            backtest_by_year(f"Variance-Covariance ({label})", estimate.VaR, es, sample, alpha)


if __name__ == "__main__":
    main()
