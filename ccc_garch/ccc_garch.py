"""
CCC-GARCH (Constant Conditional Correlation GARCH) VaR / ES.

Fits a GARCH(1,1)-style volatility filter per risk factor (shared
`common.volatility.volatility`), combines them via a constant correlation
matrix and portfolio weights to obtain a time-varying portfolio volatility,
then scales by a Normal quantile for VaR/ES. Runs on the stock+Euribor
portfolio (originally Assignment 1) with the full year-by-year + overall
backtest at alpha = 0.95 and 0.99.
"""

from typing import Callable

import numpy as np
import pandas as pd
import scipy.stats as stats

from common.data import load_stock_euribor_portfolio_from_csv, prepare_log_portfolio
from common.loss import loss_operator
from common.backtest import Backtest
from common.volatility import volatility
from var_covariance.var_covariance import Var_Cov_ES


def GARCH_portfolio_variance(data: pd.DataFrame, weights) -> np.ndarray:
    """
    Compute the CCC-GARCH portfolio volatility series: individual GARCH(1,1)
    volatilities combined via the (constant) correlation matrix and weights.
    """
    CM = data.corr().fillna(0)
    n = CM.shape[0]
    weights = np.array(weights)
    W = weights.reshape((n, 1)) * weights.reshape((1, len(weights)))
    CMU = CM * W

    vol_df = data * 0
    for asset in data.columns:
        x = data[asset]
        alpha, beta = .05, .93
        omega = np.var(x[:365]) * (1 - alpha - beta)
        vol_df[asset] = volatility(omega, alpha, beta, x)

    vol = []
    Sigm = vol_df.values
    for row in Sigm:
        vol.append(row @ CMU @ row)
    return np.array(vol) ** 0.5


def GARCH_VaR(vol, alpha: float, dist: Callable, **kwargs):
    """Calculate VaR from the GARCH portfolio volatility using a quantile function `dist`."""
    return vol * dist(alpha, **kwargs)


def _year_change_indexes(dates) -> list:
    indexes_year_changes = []
    for i in range(len(dates) - 1):
        if dates[i + 1].year != dates[i].year:
            indexes_year_changes.append(i + 1)
    indexes_year_changes.append(len(dates) - 1)
    return indexes_year_changes


def run_backtest_for_alpha(ind_loss, weights, window_size, indexes_year_changes, portfolio_loss_backtest, alpha, start_year):
    GARCH = GARCH_portfolio_variance(ind_loss, weights=weights)
    ES_GARCH = Var_Cov_ES(GARCH, alpha, stats.norm)
    VaR_GARCH = pd.DataFrame(GARCH_VaR(GARCH, alpha, stats.norm.ppf), index=ind_loss.index)[window_size:]
    ES_GARCH = pd.DataFrame(ES_GARCH, index=ind_loss.index)[window_size:]
    VaR = VaR_GARCH.iloc[:, 0]
    ES = ES_GARCH.iloc[:, 0]

    for i in range(0, len(indexes_year_changes) - 1):
        start, end = indexes_year_changes[i], indexes_year_changes[i + 1]
        print(f"\nCCC-GARCH | VaR{alpha} | period={start_year + i} | Trading Days={end - start}")
        print("Expected number of VaR violations: " + str((end - start) * (1 - alpha)))
        Backtest(VaR[start:end].to_numpy(), ES[start:end].to_numpy(), portfolio_loss_backtest[start:end], alpha)

    start, end = indexes_year_changes[0], indexes_year_changes[-1]
    print(f"\nCCC-GARCH | VaR{alpha} | period=Overall | Trading Days={end - start}")
    print("Expected number of VaR violations: " + str((end - start) * (1 - alpha)))
    Backtest(VaR[start:end].to_numpy(), ES[start:end].to_numpy(), portfolio_loss_backtest[start:end], alpha, qqplot_arrival=True)


def main():
    weights = [.2, .1, .1, .0, .1, .1, .4]

    data = load_stock_euribor_portfolio_from_csv()
    start_date, end_date = "2010-01-01", "2024-01-01"
    data = data[(data.index >= pd.to_datetime(start_date)) & (data.index < pd.to_datetime(end_date))]

    log_data = prepare_log_portfolio(data, weights)
    portfolio_loss, ind_loss = loss_operator(log_data, weights)

    dates = data.index
    indexes_year_changes = _year_change_indexes(dates)

    window_size = indexes_year_changes[0]
    portfolio_loss_backtest = portfolio_loss[window_size:]
    indexes_year_changes = np.add(np.subtract(indexes_year_changes, window_size), 1)

    for alpha in (0.95, 0.99):
        run_backtest_for_alpha(ind_loss, weights, window_size, indexes_year_changes, portfolio_loss_backtest, alpha, start_year=2011)


if __name__ == "__main__":
    main()
