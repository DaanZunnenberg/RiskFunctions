"""
Central configuration for the stock+Euribor portfolio analyses.

Every model script (var_covariance, historic_simulation, ccc_garch,
ewma_fhs, stress_testing) shares the same portfolio, weights and alpha
levels. Previously these were copy-pasted as magic numbers into every
`main()`; they now live in one place so changing the portfolio or the
confidence levels under test does not require touching five files.
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PortfolioConfig:
    """Definition of the stock+Euribor portfolio shared by all VaR models."""

    start_date: str = "2010-01-01"
    end_date: str = "2024-01-01"

    # Order matches riskfunctions.data.load_stock_euribor_portfolio columns:
    # AAPL$, AIRBUS€, ASML€, EUR_USD, LVMH€, EURO50€, EURIBOR
    asset_order: tuple = ("AAPL", "AIRBUS", "ASML", "EUR_USD", "LVMH", "EURO50", "EURIBOR")
    weights: tuple = (.2, .1, .1, .0, .1, .1, .4)


@dataclass(frozen=True)
class BacktestConfig:
    """Alpha levels and reporting parameters shared by every model's backtest."""

    alphas: tuple = (0.95, 0.99)
    first_backtest_year: int = 2011


DEFAULT_PORTFOLIO = PortfolioConfig()
DEFAULT_BACKTEST = BacktestConfig()
