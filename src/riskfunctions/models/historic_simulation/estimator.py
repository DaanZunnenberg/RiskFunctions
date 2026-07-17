"""
Historic Simulation VaR / ES estimator.

Pure function only. See `examples/historic_simulation_example.py` for a
runnable demonstration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def Historic_Simulation(
    portfolio_risk: pd.Series,
    window_size: int,
    port_weights,
    alpha: float,
    **kwargs,
) -> pd.DataFrame:
    """
    Calculate VaR and ES using historic simulation over a rolling window.

    VaR is the rolling empirical `alpha`-quantile of realised portfolio
    losses; ES is the mean loss among observations in the window that meet
    or exceed that quantile.

    `port_weights` is accepted for interface symmetry with the other
    models' estimators but is unused here, since `portfolio_risk` is
    already the portfolio-level (weighted) loss series.

    Returns
    -------
    pd.DataFrame
        Columns ['VaR', 'ES'], with warm-up rows dropped.
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
