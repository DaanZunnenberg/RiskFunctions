"""
Data loading for the copula analysis: daily log returns of a 5-asset
European equity portfolio (AAPL, ASML, AIRBUS, LVMH, EURO50).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

DEFAULT_ASSETS = ("AAPL", "ASML.AS", "AIR.PA", "MC.PA", "SC0D.DE")
_RENAME = {"ASML.AS": "ASML", "AIR.PA": "AIRBUS", "MC.PA": "LVMH", "SC0D.DE": "EURO50"}


def load_returns(start_date: str = "2010-01-01", end_date: str = "2024-01-01") -> pd.DataFrame:
    """Download AAPL, ASML, AIRBUS, LVMH, EURO50 and return daily log returns (zero-filled)."""
    stock_data = yf.download(" ".join(DEFAULT_ASSETS), start=start_date, end=end_date)
    data = pd.DataFrame(
        stock_data["Adj Close"].values,
        columns=stock_data["Adj Close"].columns,
        index=stock_data.index,
    )
    data = data.rename(columns=_RENAME)
    return data.apply(np.log).diff(1).fillna(0)
