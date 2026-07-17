"""
Historic Simulation VaR / ES.

Uses a rolling empirical quantile of realised portfolio losses as the VaR
estimate, and the mean loss beyond that quantile within the window as the
ES estimate. Runs on the stock+Euribor portfolio (originally Assignment 1)
with the full year-by-year + overall backtest at alpha = 0.95 and 0.99.
"""

import numpy as np
import pandas as pd

from common.data import load_stock_euribor_portfolio_from_csv, prepare_log_portfolio
from common.loss import loss_operator
from common.backtest import Backtest


def Historic_Simulation(portfolio_risk: pd.DataFrame, window_size: int, port_weights, alpha: float, **kwargs) -> pd.DataFrame:
    """
    Calculate VaR and ES using historic simulation over a rolling window.

    Returns a DataFrame with columns ['VaR', 'ES'].
    """
    df = pd.DataFrame([], index=portfolio_risk.index, columns=["VaR"])
    VaR = portfolio_risk.rolling(window=window_size).quantile(alpha)
    ES = [np.nan] * window_size
    for k in range(window_size, len(portfolio_risk)):
        window = portfolio_risk.iloc[k - window_size: k]
        window = window[window >= VaR.iloc[k]]
        ES.append(np.mean(window))

    df["VaR"] = VaR
    df["ES"] = ES
    return df.dropna()


def _year_change_indexes(dates) -> list:
    indexes_year_changes = []
    for i in range(len(dates) - 1):
        if dates[i + 1].year != dates[i].year:
            indexes_year_changes.append(i + 1)
    indexes_year_changes.append(len(dates) - 1)
    return indexes_year_changes


def run_backtest_for_alpha(portfolio_loss, weights, window_size, indexes_year_changes,
                            portfolio_loss_backtest, alpha, start_year):
    VaR_ES = Historic_Simulation(portfolio_loss, window_size, weights, alpha)
    VaR, ES = VaR_ES.VaR, VaR_ES.ES

    for i in range(0, len(indexes_year_changes) - 1):
        start, end = indexes_year_changes[i], indexes_year_changes[i + 1]
        print(f"\nHistoric Simulation | VaR{alpha} | period={start_year + i} | Trading Days={end - start}")
        print("Expected number of VaR violations: " + str((end - start) * (1 - alpha)))
        Backtest(VaR[start:end].to_numpy(), ES[start:end].to_numpy(), portfolio_loss_backtest[start:end], alpha)

    start, end = indexes_year_changes[0], indexes_year_changes[-1]
    print(f"\nHistoric Simulation | VaR{alpha} | period=Overall | Trading Days={end - start}")
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
        run_backtest_for_alpha(portfolio_loss, weights, window_size, indexes_year_changes,
                                portfolio_loss_backtest, alpha, start_year=2011)


if __name__ == "__main__":
    main()
