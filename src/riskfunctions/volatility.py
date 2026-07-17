"""
Shared GARCH(1,1)-style volatility filter, used by both the CCC-GARCH model
and the EWMA / Filtered Historical Simulation model (with different alpha/beta
parameterisations).
"""

from __future__ import annotations

import numpy as np


def volatility(omega: float, alpha: float, beta: float, x) -> np.ndarray:
    """
    Main volatility filter: sigma_t^2 = omega + alpha * x_{t-1}^2 + beta * sigma_{t-1}^2

    Re-estimates `omega` every 365 observations from the trailing year's
    unconditional variance (except at t=0, which is initialised from the
    unconditional variance of the first 50 observations).

    `x` may be an array-like (list, np.ndarray) or a `pandas.Series`; it is
    always accessed positionally (`x[t]`), so any pandas input is coerced to
    a plain ndarray first to avoid label-based lookups on a non-integer
    index (e.g. a DatetimeIndex) raising a KeyError.
    """
    x = np.asarray(x)
    T = len(x)

    sig = np.zeros(T)
    sig[0] = np.var(x[:50], ddof=0)  # initialize volatility at unconditional variance

    for t in range(0, T - 1):
        if t % 365 == 0:
            if t != 0:
                omega = np.var(x[t - 365: t]) * (1 - alpha - beta)
        sig[t + 1] = omega + alpha * (x[t] ** 2) + beta * sig[t]

    return sig ** 0.5
