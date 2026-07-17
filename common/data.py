"""
Shared data loading / cleaning utilities.

This module covers the two portfolios used across the original QFRM
assignments:

1. The "yfinance + Euribor" portfolio (originally Assignment 1):
   AAPL, ASML, AIRBUS, LVMH, EURO50 stocks + EUR/USD FX + Euribor 3M rate.
   See `load_stock_euribor_portfolio`.

2. The "close prices" portfolio (originally Assignment 2):
   S&P 500 (in EUR), DAX, AEX, Euribor rates, read from an .ods workbook.
   See `load_close_prices_portfolio`.

Also includes `credit_risk`, the credit-risk loss/valuation helper that
operates on Euribor-derived short rates and feeds into the loss operators
in `common/loss.py`.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def credit_risk(credit: pd.Series, short: bool = True):
    """
    Calculate the loss (and cumulative value) at each time t of a credit/loan
    position driven by a short-rate-like series `credit`.

    Parameters
    ----------
    credit : pd.Series
        The (daily) short rate series, e.g. derived from Euribor.
    short : bool
        Whether the position is short (True) or long (False) the rate risk.

    Returns
    -------
    loss : np.ndarray
        Per-period log loss contribution.
    val : np.ndarray
        Cumulative valuation implied by `loss` (exp of cumulative sum).
    """
    T = len(credit)
    t = np.arange(T)

    # take the difference in yield
    x = credit.diff().fillna(0)

    # y is defined as the yield, i.e. the credit
    y = credit
    p = np.exp(-(T - t) * y)
    loss = p * (y - (T - t) * x) * ((-1) ** short)
    val = np.exp(np.cumsum(loss))
    return loss, val


def load_stock_euribor_portfolio(
    start_date: str = "2010-01-01",
    end_date: str = "2024-01-01",
    save_csv: bool = False,
) -> pd.DataFrame:
    """
    Download/build the stock + Euribor portfolio used in the Var-Covariance,
    Historic Simulation, CCC-GARCH, EWMA-FHS and Stress Testing analyses.

    Downloads AAPL, ASML, AIRBUS, LVMH, EURO50 (STOXX 50) and EUR/USD from
    yfinance, merges with the local Euribor 3M daily CSV, cleans up missing
    trading days, and returns a single wide DataFrame indexed by Date with
    columns: AAPL, ASML, AIRBUS, LVMH, EURO50, EUR_USD, EURIBOR.

    Note: requires network access (yfinance). If you already have a saved
    `stock_quotes_and_rates.csv` in `data/`, prefer
    `load_stock_euribor_portfolio_from_csv`.
    """
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
    # Drop days where none of the equities traded (holidays)
    data = data.loc[~(data[["AAPL", "AIRBUS", "ASML", "LVMH", "EURO50"]].isnull()).all(axis=1)]
    # Impute missing quotes with closest valid ones
    data = data.ffill().bfill()

    if save_csv:
        data.to_csv(DATA_DIR / "stock_quotes_and_rates.csv")

    return data


def load_stock_euribor_portfolio_from_csv() -> pd.DataFrame:
    """Load the previously saved stock+Euribor portfolio CSV (no network)."""
    data = pd.read_csv(DATA_DIR / "stock_quotes_and_rates.csv", index_col="Date", parse_dates=True)
    return data


def prepare_log_portfolio(data: pd.DataFrame, weights=None) -> pd.DataFrame:
    """
    Given the raw stock+Euribor portfolio, apply the standard transforms:
    - Convert EURIBOR (in %) to a daily compounded short rate
    - Convert AAPL (USD) into EUR terms using EUR/USD
    - Take logs of all columns except EURIBOR (kept level, not logged again
      since it is already converted to the daily short rate)

    Returns the transformed ("log_data") DataFrame, ready for `loss_operator`.
    """
    data = data.copy()
    credit = ((1 + data["EURIBOR"] / 100) ** (1 / 365)) - 1
    data["EURIBOR"] = credit

    data["AAPL"] = data["AAPL"] / data["EUR_USD"]

    log_data = data
    log_data[data.columns[:-1]] = np.log(data[data.columns[:-1]])
    return log_data


def load_close_prices_portfolio(start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Load the "close prices" portfolio (S&P 500 in EUR, DAX, AEX, Euribor
    rates, USD/EUR) from data/close_prices_portfolio.ods, as used in the
    CCC-GARCH / Var-Covariance / Historic Simulation / EWMA-FHS analyses
    that were originally in Assignment 2.
    """
    data = pd.read_excel(
        DATA_DIR / "close_prices_portfolio.ods", engine="odf", index_col="Date", parse_dates=True
    )
    if start_date is not None or end_date is not None:
        if start_date is not None:
            data = data[data.index >= pd.to_datetime(start_date)]
        if end_date is not None:
            data = data[data.index < pd.to_datetime(end_date)]
    return data


def prepare_log_close_prices(data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the standard log transform to the close-prices portfolio: log
    everything except the last column (USD/EUR), which is instead
    exponentiated (kept as a level FX rate for downstream use).
    """
    data_log = pd.DataFrame(data.values, columns=data.columns, index=data.index)
    data_log.index.name = "Date"
    data_log[data_log.columns[:-1]] = np.log(data_log[data_log.columns[:-1]])
    data_log["USD/EUR"] = np.exp(data_log["USD/EUR"])
    return data_log
