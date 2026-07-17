"""
CCC-GARCH (Constant Conditional Correlation GARCH) VaR estimator.

Pure functions only. See `examples/ccc_garch_example.py` for a runnable
demonstration.
"""

from __future__ import annotations
from typing import Callable

import numpy as np
import pandas as pd

from riskfunctions.volatility import volatility

# GARCH(1,1) persistence parameters used to filter each risk factor's
# volatility. Fixed rather than fitted by MLE, matching the original
# coursework's simplification.
GARCH_ALPHA = 0.05
GARCH_BETA = 0.93


def GARCH_portfolio_variance(data: pd.DataFrame, weights) -> np.ndarray:
    """
    Compute the CCC-GARCH portfolio volatility series.

    Each risk factor's volatility is filtered independently with a
    GARCH(1,1) process (`riskfunctions.volatility.volatility`), then
    combined into a portfolio volatility via the *constant* (full-sample)
    correlation matrix and the portfolio weights.

    Parameters
    ----------
    data : pd.DataFrame
        Per-risk-factor loss series (one column per asset/risk factor).
    weights : array-like
        Portfolio weights aligned with data.columns.

    Returns
    -------
    np.ndarray
        Portfolio volatility series, one value per row of `data`.
    """
    corr = data.corr().fillna(0)
    n = corr.shape[0]
    weights = np.array(weights)
    weight_outer = weights.reshape((n, 1)) * weights.reshape((1, len(weights)))
    weighted_corr = corr * weight_outer

    vol_df = data * 0
    for asset in data.columns:
        x = data[asset]
        omega = np.var(x[:365]) * (1 - GARCH_ALPHA - GARCH_BETA)
        vol_df[asset] = volatility(omega, GARCH_ALPHA, GARCH_BETA, x)

    portfolio_vol = [row @ weighted_corr @ row for row in vol_df.values]
    return np.array(portfolio_vol) ** 0.5


def GARCH_VaR(vol, alpha: float, dist: Callable, **kwargs):
    """Scale a GARCH portfolio volatility series into VaR using quantile function `dist`."""
    return vol * dist(alpha, **kwargs)
