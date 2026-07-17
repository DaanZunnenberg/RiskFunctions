"""
Data loading for the factor analysis: an extended 10-asset portfolio
designed to exhibit clear factor structure across region (EU/US), industry
(aerospace/tech), and idiosyncratic (gold uncorrelated, oil broadly
correlated) factors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

DEFAULT_ASSETS = ("SC0D.DE", "ASML.AS", "AIR.PA", "SPY", "AAPL", "GM", "DAL", "0700.HK", "CL=F", "GC=F")
_RENAME = {
    "ASML.AS": "ASML", "AIR.PA": "AIRBUS", "SPY": "SNP500", "SC0D.DE": "EURO50",
    "DAL": "DELTA", "0700.HK": "TENCENT", "CL=F": "OIL", "GC=F": "GOLD",
}


def load_extended_returns(start_date: str = "2010-01-01", end_date: str = "2024-01-01") -> pd.DataFrame:
    """
    Download the extended 10-asset portfolio (EURO50, ASML, AIRBUS, S&P
    500, AAPL, GM, Delta, Tencent, Oil, Gold) and return daily log returns
    (zero-filled).
    """
    stock_data = yf.download(" ".join(DEFAULT_ASSETS), start=start_date, end=end_date)
    data = pd.DataFrame(
        stock_data["Adj Close"].values,
        columns=stock_data["Adj Close"].columns,
        index=stock_data.index,
    )
    data = data.rename(columns=_RENAME)
    return data.apply(np.log).diff(1).fillna(0)
