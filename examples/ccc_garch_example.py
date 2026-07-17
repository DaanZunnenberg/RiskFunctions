"""
Example: CCC-GARCH VaR/ES on the stock+Euribor portfolio, backtested
year-by-year at alpha = 0.95 and 0.99.

Run from the repo root:
    python examples/ccc_garch_example.py
"""

import pandas as pd
import scipy.stats as stats

from riskfunctions.config import DEFAULT_BACKTEST
from riskfunctions.models.ccc_garch import GARCH_portfolio_variance, GARCH_VaR
from riskfunctions.models.var_covariance import Var_Cov_ES
from riskfunctions.pipeline import backtest_by_year, load_portfolio_sample


def main():
    sample = load_portfolio_sample()

    garch_vol = GARCH_portfolio_variance(sample.ind_loss, weights=sample.weights)

    for alpha in DEFAULT_BACKTEST.alphas:
        var = pd.Series(
            GARCH_VaR(garch_vol, alpha, stats.norm.ppf),
            index=sample.ind_loss.index,
        )[sample.window_size:]
        es = pd.Series(
            Var_Cov_ES(garch_vol, alpha, stats.norm),
            index=sample.ind_loss.index,
        )[sample.window_size:]

        backtest_by_year("CCC-GARCH", var, es, sample, alpha)


if __name__ == "__main__":
    main()
