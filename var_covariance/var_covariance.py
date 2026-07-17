"""
Variance-Covariance (Delta-Normal) VaR / ES.

Computes rolling-window portfolio volatility from the sample covariance
matrix of individual risk-factor losses, then scales by the relevant
quantile of a Normal or Student-t (dof 4, 5, 6) distribution to obtain VaR,
and by the corresponding closed-form expected-shortfall multiplier.

Runs on the stock+Euribor portfolio (originally Assignment 1) and includes
the full year-by-year + overall backtest at alpha = 0.95 and 0.99, matching
the original notebook. Run this file directly to reproduce those results
(requires network access for yfinance, or a cached data/stock_quotes_and_rates.csv).
"""

from typing import Any, Callable

import numpy as np
import pandas as pd
import scipy.stats as stats

from common.data import (
    load_stock_euribor_portfolio_from_csv,
    prepare_log_portfolio,
)
from common.loss import loss_operator
from common.backtest import Backtest


def Var_Cov_VaR(
    data: pd.DataFrame,
    window_size: int,
    port_weights,
    alpha: float,
    dist: Callable,
    **kwargs,
) -> pd.DataFrame:
    """
    Calculate rolling Variance-Covariance VaR.

    Returns a DataFrame with columns ['Vol', 'VaR'] indexed like `data`.
    """
    pf_vol = [np.nan] * window_size
    for k in range(window_size, len(data)):
        COV = data[k - window_size: k].cov(ddof=0)
        pf_vol.append(np.sqrt(port_weights @ COV @ port_weights))

    pf_vol = np.array(pf_vol)

    portfolio_VaR = pd.DataFrame(pf_vol, index=data.index, columns=["Vol"])
    portfolio_VaR["VaR"] = pf_vol * dist(alpha, **kwargs)
    return portfolio_VaR.dropna()


def Var_Cov_ES(Vol: Any, alpha: float, dist: Callable, **kwargs) -> float:
    """
    Calculate the expected shortfall corresponding to a Var-Covariance VaR,
    for Normal or Student-t distributions.
    """
    alpha = 1 - alpha
    match dist:
        case stats.norm:
            q: float = stats.norm.isf(alpha)
            p: float = stats.norm.pdf(q)
            return Vol * p / alpha
        case stats.t:
            df: int | float = kwargs.get("df", 4)
            q: float = stats.t.ppf(alpha, df=df)
            p: float = stats.t.pdf(q, df=df)
            return Vol * ((df + (q ** 2)) / (df - 1)) * p / alpha
        case _:
            raise NotImplementedError()


def _year_change_indexes(dates) -> list:
    indexes_year_changes = []
    for i in range(len(dates) - 1):
        if dates[i + 1].year != dates[i].year:
            indexes_year_changes.append(i + 1)
    indexes_year_changes.append(len(dates) - 1)
    return indexes_year_changes


def run_backtest_for_alpha(ind_loss, portfolio_loss, weights, window_size, indexes_year_changes,
                            portfolio_loss_backtest, alpha, start_year):
    """Run the Var-Cov VaR/ES backtest (Normal + t dof 4/5/6) for a single alpha level."""
    configs = [
        ("Normal", stats.norm.ppf, stats.norm, {}),
        ("t-Student dof=4", stats.t.ppf, stats.t, {"df": 4}),
        ("t-Student dof=5", stats.t.ppf, stats.t, {"df": 5}),
        ("t-Student dof=6", stats.t.ppf, stats.t, {"df": 6}),
    ]

    for label, var_dist, es_dist, kwargs in configs:
        HS = Var_Cov_VaR(ind_loss, window_size=window_size, port_weights=weights,
                          alpha=alpha, dist=var_dist, **kwargs)
        VaR = HS.VaR
        ES = Var_Cov_ES(HS.Vol, alpha=alpha, dist=es_dist, **kwargs)

        for i in range(0, len(indexes_year_changes) - 1):
            start, end = indexes_year_changes[i], indexes_year_changes[i + 1]
            print(f"\nVariance Covariance {label} | VaR{alpha} | period={start_year + i} | Trading Days={end - start}")
            print("Expected number of VaR violations: " + str((end - start) * (1 - alpha)))
            Backtest(VaR[start:end].to_numpy(), ES[start:end].to_numpy(),
                     portfolio_loss_backtest[start:end], alpha)

        start, end = indexes_year_changes[0], indexes_year_changes[-1]
        print(f"\nVariance Covariance {label} | VaR{alpha} | period=Overall | Trading Days={end - start}")
        print("Expected number of VaR violations: " + str((end - start) * (1 - alpha)))
        Backtest(VaR[start:end].to_numpy(), ES[start:end].to_numpy(),
                  portfolio_loss_backtest[start:end], alpha, qqplot_arrival=True)


def main():
    weights = [.2, .1, .1, .0, .1, .1, .4]  # AAPL$, AIRBUS€, ASML€, EUR_USD, LVMH€, EURO50€, EURIBOR

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
        run_backtest_for_alpha(ind_loss, portfolio_loss, weights, window_size,
                                indexes_year_changes, portfolio_loss_backtest, alpha, start_year=2011)


if __name__ == "__main__":
    main()
