"""
Variance-Covariance (Delta-Normal) VaR / ES estimator.

Pure functions only, no I/O, no printing, no plotting. See
`examples/var_covariance_example.py` for a runnable demonstration that
loads data, calls these functions, and backtests the result.
"""

from __future__ import annotations
from typing import Any, Callable

import numpy as np
import pandas as pd
import scipy.stats as stats


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

    For each day, the portfolio volatility is the sqrt of the quadratic
    form w'Cov(w) over the trailing `window_size` observations of `data`
    (per-risk-factor losses), then scaled by the alpha-quantile of `dist`.

    Parameters
    ----------
    data : pd.DataFrame
        Per-risk-factor loss series (one column per asset/risk factor).
    window_size : int
        Rolling covariance window, in trading days.
    port_weights : array-like
        Portfolio weights aligned with data.columns.
    alpha : float
        Confidence level (e.g. 0.95, 0.99).
    dist : Callable
        A quantile function, e.g. `scipy.stats.norm.ppf` or `scipy.stats.t.ppf`.
    **kwargs
        Extra keyword arguments forwarded to `dist` (e.g. `df=4` for Student-t).

    Returns
    -------
    pd.DataFrame
        Columns ['Vol', 'VaR'], indexed like `data`, with the initial
        `window_size` rows (used purely for warm-up) dropped.
    """
    pf_vol = [np.nan] * window_size
    for k in range(window_size, len(data)):
        cov = data[k - window_size: k].cov(ddof=0)
        pf_vol.append(np.sqrt(port_weights @ cov @ port_weights))

    pf_vol = np.array(pf_vol)

    portfolio_var = pd.DataFrame(pf_vol, index=data.index, columns=["Vol"])
    portfolio_var["VaR"] = pf_vol * dist(alpha, **kwargs)
    return portfolio_var.dropna()


def Var_Cov_ES(Vol: Any, alpha: float, dist: Callable, **kwargs) -> float:
    """
    Closed-form expected shortfall corresponding to a Var-Covariance VaR,
    for Normal or Student-t distributions.

    Parameters
    ----------
    Vol : array-like
        Portfolio volatility series (the 'Vol' column from `Var_Cov_VaR`).
    alpha : float
        Confidence level (e.g. 0.95, 0.99) -- NOT the tail probability.
    dist : Callable
        Either `scipy.stats.norm` or `scipy.stats.t` (the distribution
        object, not a quantile function).
    **kwargs
        `df` is required (and defaults to 4) when `dist is scipy.stats.t`.
    """
    tail = 1 - alpha
    if dist is stats.norm:
        q = stats.norm.isf(tail)
        p = stats.norm.pdf(q)
        return Vol * p / tail
    elif dist is stats.t:
        df = kwargs.get("df", 4)
        q = stats.t.ppf(tail, df=df)
        p = stats.t.pdf(q, df=df)
        return Vol * ((df + (q ** 2)) / (df - 1)) * p / tail
    else:
        raise NotImplementedError(f"Var_Cov_ES only supports scipy.stats.norm or scipy.stats.t, got {dist!r}")
