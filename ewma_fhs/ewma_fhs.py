"""
EWMA / Filtered Historical Simulation (FHS) VaR / ES.

De-volatilizes each risk factor's loss series using an EWMA-style filter
(the shared `common.volatility.volatility` function with alpha=.05,
beta=.95), then re-scales the standardized residuals by the latest
predicted volatility and applies historic simulation on the resulting
portfolio return series to obtain VaR/ES. Runs on the stock+Euribor
portfolio (originally Assignment 1) with the full year-by-year + overall
backtest at alpha = 0.95 and 0.99.
"""

import numpy as np
import pandas as pd

from common.data import load_stock_euribor_portfolio_from_csv, prepare_log_portfolio
from common.loss import loss_operator
from common.backtest import Backtest
from common.volatility import volatility


def filtered_HS(portfolio_risk_ind: pd.DataFrame, weights, _alpha: float, window_size: int) -> pd.DataFrame:
    """
    Compute EWMA-filtered Historical Simulation VaR/ES.

    Returns a DataFrame with columns ['VaR', 'ES'].
    """
    Devol = pd.DataFrame([], index=portfolio_risk_ind.index)
    Vol = pd.DataFrame([], index=portfolio_risk_ind.index)
    for col in portfolio_risk_ind.columns:
        x = portfolio_risk_ind[col]
        alpha, beta = .05, .95
        omega = 0
        vol = volatility(omega, alpha, beta, x)
        Vol[f"Vol {col}"] = vol
        Devol[f"Devol {col}"] = portfolio_risk_ind[col] / vol

    Devol = Devol.fillna(0)
    df = pd.DataFrame([], index=portfolio_risk_ind.index, columns=["VaR"])

    ES = [np.nan] * window_size
    VaR = [np.nan] * window_size
    for k in range(window_size, len(portfolio_risk_ind)):
        D = Devol.iloc[k - window_size: k] * Vol.iloc[k].values

        portfolio_returns = D @ weights
        VaR.append(portfolio_returns.rolling(window=window_size).quantile(_alpha).iloc[-1])

        window = portfolio_returns[portfolio_returns >= VaR[k]]
        ES.append(np.mean(window))
    df["VaR"] = VaR
    df["ES"] = np.array(ES)
    return df[window_size:]


def _year_change_indexes(dates) -> list:
    indexes_year_changes = []
    for i in range(len(dates) - 1):
        if dates[i + 1].year != dates[i].year:
            indexes_year_changes.append(i + 1)
    indexes_year_changes.append(len(dates) - 1)
    return indexes_year_changes


def run_backtest_for_alpha(ind_loss, weights, window_size, indexes_year_changes, portfolio_loss_backtest, alpha, start_year):
    EWMA_HS = filtered_HS(ind_loss, weights, alpha, window_size)
    VaR, ES = EWMA_HS.VaR, EWMA_HS.ES

    for i in range(0, len(indexes_year_changes) - 1):
        start, end = indexes_year_changes[i], indexes_year_changes[i + 1]
        print(f"\nEWMA-FHS | VaR{alpha} | period={start_year + i} | Trading Days={end - start}")
        print("Expected number of VaR violations: " + str((end - start) * (1 - alpha)))
        Backtest(VaR[start:end].to_numpy(), ES[start:end].to_numpy(), portfolio_loss_backtest[start:end], alpha)

    start, end = indexes_year_changes[0], indexes_year_changes[-1]
    print(f"\nEWMA-FHS | VaR{alpha} | period=Overall | Trading Days={end - start}")
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
