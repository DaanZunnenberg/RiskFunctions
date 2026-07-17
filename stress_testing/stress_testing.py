"""
Stress Testing: shock AAPL prices and/or Euribor interest rates on the
stock+Euribor portfolio, then re-run the Var-Covariance / Historic
Simulation / CCC-GARCH / EWMA-FHS VaR & ES estimation and backtesting on
the shocked data, matching the original Assignment 1 stress-test section.

Two scenarios are implemented (as in the original notebook):
  1. Shock AAPL down by 20% over the last week of Dec 2021 (with an extra
     -20% on the middle 2 days, compounding to -36%).
  2. Shock Euribor rates up by +1% over the same windows.
Both can be combined, or run individually via `run_scenario`.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import yfinance as yf

from common.data import DATA_DIR, prepare_log_portfolio
from common.loss import loss_operator
from common.backtest import Backtest
from common.volatility import volatility
from var_covariance.var_covariance import Var_Cov_VaR, Var_Cov_ES
from historic_simulation.historic_simulation import Historic_Simulation
from ccc_garch.ccc_garch import GARCH_portfolio_variance, GARCH_VaR
from ewma_fhs.ewma_fhs import filtered_HS


def load_raw_portfolio(start_date="2010-01-01", end_date="2024-01-01") -> pd.DataFrame:
    """Re-download the raw (unshocked) stock+Euribor portfolio for shocking."""
    stocks = ["AAPL", "ASML.AS", "AIR.PA", "MC.PA", "SC0D.DE", "EURUSD=X"]
    stock_data = yf.download(" ".join(stocks), start=start_date, end=end_date)
    data = pd.DataFrame(
        stock_data["Adj Close"].values,
        columns=stock_data["Adj Close"].columns,
        index=stock_data.index,
    )

    euribor_3m = pd.read_csv(DATA_DIR / "BundesBank Euribor 3M Daily.csv")
    euribor_3m["Date"] = pd.to_datetime(euribor_3m["Date"], format="%Y-%m-%d")
    euribor_3m = euribor_3m[
        (euribor_3m["Date"] >= pd.to_datetime(start_date))
        & (euribor_3m["Date"] <= pd.to_datetime(end_date))
    ]
    euribor_3m = euribor_3m.set_index("Date")
    data = pd.merge(data, euribor_3m, on="Date", how="inner")

    data = data.rename(
        columns={
            "ASML.AS": "ASML",
            "AIR.PA": "AIRBUS",
            "MC.PA": "LVMH",
            "SC0D.DE": "EURO50",
            "EURIBOR_3M": "EURIBOR",
            "EURUSD=X": "EUR_USD",
        }
    )
    data = data.loc[~(data[["AAPL", "AIRBUS", "ASML", "LVMH", "EURO50"]].isnull()).all(axis=1)]
    data = data.ffill().bfill()
    return data


def shock_aapl(data: pd.DataFrame) -> pd.DataFrame:
    """Shock AAPL down by 20% over the last week of Dec 2021 (extra -20% for Dec 25-27)."""
    data = data.copy()
    mask1 = (data.index >= pd.to_datetime("2021-12-23")) & (data.index <= pd.to_datetime("2022-01-01"))
    data.loc[mask1, "AAPL"] = data.loc[mask1, "AAPL"] * 0.8

    mask2 = (data.index >= pd.to_datetime("2021-12-25")) & (data.index <= pd.to_datetime("2021-12-27"))
    data.loc[mask2, "AAPL"] = data.loc[mask2, "AAPL"] * 0.8
    return data


def shock_euribor(data: pd.DataFrame) -> pd.DataFrame:
    """Shock Euribor rates up by +1% over the last week of Dec 2021 (extra +1% for Dec 25-27)."""
    data = data.copy()
    mask1 = (data.index >= pd.to_datetime("2021-12-23")) & (data.index <= pd.to_datetime("2022-01-01"))
    data.loc[mask1, "EURIBOR"] = data.loc[mask1, "EURIBOR"] + 1.0

    mask2 = (data.index >= pd.to_datetime("2021-12-25")) & (data.index <= pd.to_datetime("2021-12-27"))
    data.loc[mask2, "EURIBOR"] = data.loc[mask2, "EURIBOR"] + 1.0
    return data


def _year_change_indexes(dates) -> list:
    indexes_year_changes = []
    for i in range(len(dates) - 1):
        if dates[i + 1].year != dates[i].year:
            indexes_year_changes.append(i + 1)
    indexes_year_changes.append(len(dates) - 1)
    return indexes_year_changes


def run_scenario(data: pd.DataFrame, weights, alphas=(0.95, 0.99), years_range=(11, 13)):
    """
    Re-run all four VaR/ES models (Var-Cov Normal/t, Historic Simulation,
    CCC-GARCH, EWMA-FHS) and backtest them on the (possibly shocked)
    portfolio `data`, restricted to `years_range` (indices into the yearly
    breakpoints) to focus on the shock period, matching the original
    notebook's behaviour.
    """
    log_data = prepare_log_portfolio(data, weights)
    portfolio_loss, ind_loss = loss_operator(log_data, weights)

    dates = data.index
    indexes_year_changes = _year_change_indexes(dates)

    window_size = indexes_year_changes[0]
    portfolio_loss_backtest = portfolio_loss[window_size:]
    indexes_year_changes = np.add(np.subtract(indexes_year_changes, window_size), 1)

    for alpha in alphas:
        # Var-Covariance (Normal + t dof 4,5,6)
        configs = [
            ("Normal", stats.norm.ppf, stats.norm, {}),
            ("t-Student dof=4", stats.t.ppf, stats.t, {"df": 4}),
            ("t-Student dof=5", stats.t.ppf, stats.t, {"df": 5}),
            ("t-Student dof=6", stats.t.ppf, stats.t, {"df": 6}),
        ]
        for label, var_dist, es_dist, kwargs in configs:
            HS = Var_Cov_VaR(ind_loss, window_size=window_size, port_weights=weights, alpha=alpha, dist=var_dist, **kwargs)
            VaR, ES = HS.VaR, Var_Cov_ES(HS.Vol, alpha=alpha, dist=es_dist, **kwargs)
            for i in range(*years_range):
                start, end = indexes_year_changes[i], indexes_year_changes[i + 1]
                print(f"\n[Stress] Var-Cov {label} | VaR{alpha} | period={2011 + i} | Trading Days={end - start}")
                print("Expected number of VaR violations: " + str((end - start) * (1 - alpha)))
                Backtest(VaR[start:end].to_numpy(), ES[start:end].to_numpy(), portfolio_loss_backtest[start:end], alpha)

        # Historic Simulation
        VaR_ES_HS = Historic_Simulation(portfolio_loss, window_size, weights, alpha)
        VaR, ES = VaR_ES_HS.VaR, VaR_ES_HS.ES
        for i in range(*years_range):
            start, end = indexes_year_changes[i], indexes_year_changes[i + 1]
            print(f"\n[Stress] Historic Simulation | VaR{alpha} | period={2011 + i} | Trading Days={end - start}")
            print("Expected number of VaR violations: " + str((end - start) * (1 - alpha)))
            Backtest(VaR[start:end].to_numpy(), ES[start:end].to_numpy(), portfolio_loss_backtest[start:end], alpha)

        # CCC-GARCH
        GARCH = GARCH_portfolio_variance(ind_loss, weights=weights)
        ES_GARCH = Var_Cov_ES(GARCH, alpha, stats.norm)
        VaR_GARCH = pd.DataFrame(GARCH_VaR(GARCH, alpha, stats.norm.ppf), index=ind_loss.index)[window_size:]
        ES_GARCH = pd.DataFrame(ES_GARCH, index=ind_loss.index)[window_size:]
        VaR, ES = VaR_GARCH.iloc[:, 0], ES_GARCH.iloc[:, 0]
        for i in range(*years_range):
            start, end = indexes_year_changes[i], indexes_year_changes[i + 1]
            print(f"\n[Stress] CCC-GARCH | VaR{alpha} | period={2011 + i} | Trading Days={end - start}")
            print("Expected number of VaR violations: " + str((end - start) * (1 - alpha)))
            Backtest(VaR[start:end].to_numpy(), ES[start:end].to_numpy(), portfolio_loss_backtest[start:end], alpha)

        # EWMA-FHS
        EWMA_HS = filtered_HS(ind_loss, weights, alpha, window_size)
        VaR, ES = EWMA_HS.VaR, EWMA_HS.ES
        for i in range(*years_range):
            start, end = indexes_year_changes[i], indexes_year_changes[i + 1]
            print(f"\n[Stress] EWMA-FHS | VaR{alpha} | period={2011 + i} | Trading Days={end - start}")
            print("Expected number of VaR violations: " + str((end - start) * (1 - alpha)))
            Backtest(VaR[start:end].to_numpy(), ES[start:end].to_numpy(), portfolio_loss_backtest[start:end], alpha)


def main():
    weights = [.2, .1, .1, .0, .1, .1, .4]

    raw_data = load_raw_portfolio()

    print("=" * 80)
    print("Scenario 1: AAPL price shock (-20% / -36% over last week of Dec 2021)")
    print("=" * 80)
    data_aapl_shock = shock_aapl(raw_data)
    run_scenario(data_aapl_shock, weights)

    print("=" * 80)
    print("Scenario 2: AAPL + EURIBOR shock (+1% / +2% rate shock over same window)")
    print("=" * 80)
    data_combined_shock = shock_euribor(shock_aapl(raw_data))
    run_scenario(data_combined_shock, weights)


if __name__ == "__main__":
    main()
