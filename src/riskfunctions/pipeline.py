"""
Shared backtesting pipeline plumbing.

Every model script (var_covariance, historic_simulation, ccc_garch,
ewma_fhs, stress_testing) needs to:

1. Load and log-transform the stock+Euribor portfolio.
2. Compute the portfolio/individual loss series.
3. Split the sample into calendar years, using the first year as the
   training window and holding out the rest for backtesting.
4. For each alpha level, run the model's VaR/ES estimator and backtest it
   year-by-year plus overall.

Steps 1-3 and the per-year backtest loop were previously copy-pasted
(with minor renaming) into every model file. They are consolidated here so
each model file only has to provide its own VaR/ES estimator.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from riskfunctions.backtest import Backtest
from riskfunctions.config import DEFAULT_BACKTEST, DEFAULT_PORTFOLIO, BacktestConfig, PortfolioConfig
from riskfunctions.data import load_stock_euribor_portfolio_from_csv, prepare_log_portfolio
from riskfunctions.loss import loss_operator


@dataclass
class PortfolioSample:
    """A prepared portfolio sample, ready for VaR/ES estimation and backtesting."""

    portfolio_loss: pd.Series
    ind_loss: pd.DataFrame
    window_size: int
    year_breaks: np.ndarray
    portfolio_loss_backtest: pd.Series
    weights: tuple


def year_change_indexes(dates) -> list:
    """Return the row indexes where the calendar year changes, plus the final index."""
    indexes = [i + 1 for i in range(len(dates) - 1) if dates[i + 1].year != dates[i].year]
    indexes.append(len(dates) - 1)
    return indexes


def build_portfolio_sample(data: pd.DataFrame, weights: tuple) -> PortfolioSample:
    """
    Build a `PortfolioSample` from an already-loaded raw stock+Euribor
    DataFrame (e.g. the output of `riskfunctions.data.load_stock_euribor_portfolio`
    or a stress-tested variant of it), applying the standard log transform,
    loss operator, and year-boundary split used for backtesting.

    This is the building block `load_portfolio_sample` wraps for the common
    (un-shocked, CSV-cached) case; stress-testing examples call this
    directly with their own shocked `data`.
    """
    log_data = prepare_log_portfolio(data, list(weights))
    portfolio_loss, ind_loss = loss_operator(log_data, list(weights))

    year_breaks = year_change_indexes(data.index)
    window_size = year_breaks[0]
    portfolio_loss_backtest = portfolio_loss[window_size:]
    year_breaks = np.add(np.subtract(year_breaks, window_size), 1)

    return PortfolioSample(
        portfolio_loss=portfolio_loss,
        ind_loss=ind_loss,
        window_size=window_size,
        year_breaks=year_breaks,
        portfolio_loss_backtest=portfolio_loss_backtest,
        weights=weights,
    )


def load_portfolio_sample(portfolio: PortfolioConfig = DEFAULT_PORTFOLIO) -> PortfolioSample:
    """
    Load the stock+Euribor portfolio (from the cached CSV, no network
    required) and build a `PortfolioSample` from it. The first calendar
    year is used purely as the training window for rolling estimators and
    is excluded from `portfolio_loss_backtest`.
    """
    data = load_stock_euribor_portfolio_from_csv()
    data = data[
        (data.index >= pd.to_datetime(portfolio.start_date))
        & (data.index < pd.to_datetime(portfolio.end_date))
    ]
    return build_portfolio_sample(data, portfolio.weights)


def backtest_by_year(
    label: str,
    VaR: pd.Series,
    ES: pd.Series,
    sample: PortfolioSample,
    alpha: float,
    backtest_cfg: BacktestConfig = DEFAULT_BACKTEST,
    year_range: tuple[int, int] | None = None,
) -> None:
    """
    Run `Backtest` for `VaR`/`ES` against `sample.portfolio_loss_backtest`,
    once per calendar year in `year_range` (defaults to the full sample)
    plus once for the overall period (with the inter-arrival QQ-plot).

    This replaces the "for i in range(...): ... Backtest(...)" loop that
    was duplicated verbatim across every model script.
    """
    year_breaks = sample.year_breaks
    lo, hi = year_range if year_range is not None else (0, len(year_breaks) - 1)

    for i in range(lo, hi):
        start, end = year_breaks[i], year_breaks[i + 1]
        year = backtest_cfg.first_backtest_year + i
        print(f"\n{label} | VaR{alpha} | period={year} | Trading Days={end - start}")
        print("Expected number of VaR violations: " + str((end - start) * (1 - alpha)))
        Backtest(
            VaR[start:end].to_numpy(),
            ES[start:end].to_numpy(),
            sample.portfolio_loss_backtest[start:end],
            alpha,
        )

    start, end = year_breaks[0], year_breaks[-1]
    print(f"\n{label} | VaR{alpha} | period=Overall | Trading Days={end - start}")
    print("Expected number of VaR violations: " + str((end - start) * (1 - alpha)))
    Backtest(
        VaR[start:end].to_numpy(),
        ES[start:end].to_numpy(),
        sample.portfolio_loss_backtest[start:end],
        alpha,
        qqplot_arrival=True,
    )
