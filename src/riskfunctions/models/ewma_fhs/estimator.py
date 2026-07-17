"""
EWMA / Filtered Historical Simulation (FHS) VaR / ES estimator.

Pure function only. See `examples/ewma_fhs_example.py` for a runnable
demonstration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from riskfunctions.volatility import volatility

# EWMA persistence parameters (RiskMetrics-style), used to de-volatilize
# each risk factor before re-scaling by the latest predicted volatility.
EWMA_ALPHA = 0.05
EWMA_BETA = 0.95


def filtered_HS(
    portfolio_risk_ind: pd.DataFrame,
    weights,
    alpha: float,
    window_size: int,
) -> pd.DataFrame:
    """
    Compute EWMA-filtered Historical Simulation VaR/ES.

    Each risk factor's loss series is de-volatilized by its own EWMA
    filter (`riskfunctions.volatility.volatility`), then the standardized
    residuals are re-scaled by the *current* predicted volatility and
    combined into a portfolio return series each period; historic
    simulation (rolling empirical quantile) is then applied to that series.

    Parameters
    ----------
    portfolio_risk_ind : pd.DataFrame
        Per-risk-factor loss series (one column per asset/risk factor).
    weights : array-like
        Portfolio weights aligned with portfolio_risk_ind.columns.
    alpha : float
        Confidence level (e.g. 0.95, 0.99).
    window_size : int
        Rolling historic-simulation window, in trading days.

    Returns
    -------
    pd.DataFrame
        Columns ['VaR', 'ES'], with the warm-up window dropped.
    """
    devol = pd.DataFrame([], index=portfolio_risk_ind.index)
    vol = pd.DataFrame([], index=portfolio_risk_ind.index)
    for col in portfolio_risk_ind.columns:
        x = portfolio_risk_ind[col]
        filtered_vol = volatility(0, EWMA_ALPHA, EWMA_BETA, x)
        vol[f"Vol {col}"] = filtered_vol
        devol[f"Devol {col}"] = portfolio_risk_ind[col] / filtered_vol

    devol = devol.fillna(0)

    ES = [np.nan] * window_size
    VaR = [np.nan] * window_size
    for k in range(window_size, len(portfolio_risk_ind)):
        rescaled_window = devol.iloc[k - window_size: k] * vol.iloc[k].values
        portfolio_returns = rescaled_window @ weights

        var_k = portfolio_returns.rolling(window=window_size).quantile(alpha).iloc[-1]
        VaR.append(var_k)

        tail = portfolio_returns[portfolio_returns >= var_k]
        ES.append(np.mean(tail))

    df = pd.DataFrame({"VaR": VaR, "ES": ES}, index=portfolio_risk_ind.index)
    return df[window_size:]
