"""
Stress-test scenario definitions: pure data transforms that shock the raw
stock+Euribor portfolio before it is fed through the usual VaR/ES models.

No I/O, no printing. See `examples/stress_testing_example.py` for a
runnable demonstration that applies these shocks and re-runs every model.
"""

from __future__ import annotations

import pandas as pd

# The shock window used throughout the original coursework: the last week
# of December 2021, with an additional compounding shock over its middle
# three days (Dec 25-27).
_OUTER_WINDOW = ("2021-12-23", "2022-01-01")
_INNER_WINDOW = ("2021-12-25", "2021-12-27")


def _window_mask(index: pd.DatetimeIndex, start: str, end: str) -> pd.Series:
    return (index >= pd.to_datetime(start)) & (index <= pd.to_datetime(end))


def shock_aapl(data: pd.DataFrame, factor: float = 0.8) -> pd.DataFrame:
    """
    Shock AAPL prices by `factor` over the outer window, and again over the
    inner window (so the inner window compounds to `factor ** 2`).

    Reproduces the original -20%/-36% AAPL price shock scenario.
    """
    data = data.copy()
    outer_mask = _window_mask(data.index, *_OUTER_WINDOW)
    data.loc[outer_mask, "AAPL"] *= factor

    inner_mask = _window_mask(data.index, *_INNER_WINDOW)
    data.loc[inner_mask, "AAPL"] *= factor
    return data


def shock_euribor(data: pd.DataFrame, bump: float = 1.0) -> pd.DataFrame:
    """
    Shock Euribor rates up by `bump` (percentage points) over the outer
    window, and again over the inner window (so the inner window is
    bumped by `2 * bump`).

    Reproduces the original +1%/+2% Euribor rate shock scenario.
    """
    data = data.copy()
    outer_mask = _window_mask(data.index, *_OUTER_WINDOW)
    data.loc[outer_mask, "EURIBOR"] += bump

    inner_mask = _window_mask(data.index, *_INNER_WINDOW)
    data.loc[inner_mask, "EURIBOR"] += bump
    return data
