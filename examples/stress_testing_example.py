"""
Example: stress-test the stock+Euribor portfolio by shocking AAPL prices
and/or Euribor rates over the last week of December 2021, then re-run
every VaR/ES model (Variance-Covariance, Historic Simulation, CCC-GARCH,
EWMA-FHS) and backtest the shock-period years.

Requires network access (yfinance) to re-download the raw portfolio, since
shocking requires the pre-log-transform price levels.

Run from the repo root:
    python examples/stress_testing_example.py
"""

import pandas as pd
import scipy.stats as stats

from riskfunctions.config import DEFAULT_BACKTEST, DEFAULT_PORTFOLIO
from riskfunctions.data import load_stock_euribor_portfolio
from riskfunctions.models.ccc_garch import GARCH_portfolio_variance, GARCH_VaR
from riskfunctions.models.ewma_fhs import filtered_HS
from riskfunctions.models.historic_simulation import Historic_Simulation
from riskfunctions.models.stress_testing import shock_aapl, shock_euribor
from riskfunctions.models.var_covariance import Var_Cov_ES, Var_Cov_VaR
from riskfunctions.pipeline import backtest_by_year, build_portfolio_sample

# The stress-test backtest focuses on the shock year and the one following
# it (year indices 11-12 relative to 2011).
SHOCK_YEAR_RANGE = (11, 13)

DISTRIBUTIONS = [
    ("Normal", stats.norm.ppf, stats.norm, {}),
    ("t-Student dof=4", stats.t.ppf, stats.t, {"df": 4}),
    ("t-Student dof=5", stats.t.ppf, stats.t, {"df": 5}),
    ("t-Student dof=6", stats.t.ppf, stats.t, {"df": 6}),
]


def run_all_models(sample, scenario_label: str):
    for alpha in DEFAULT_BACKTEST.alphas:
        for label, var_dist, es_dist, kwargs in DISTRIBUTIONS:
            estimate = Var_Cov_VaR(
                sample.ind_loss, window_size=sample.window_size,
                port_weights=sample.weights, alpha=alpha, dist=var_dist, **kwargs,
            )
            es = Var_Cov_ES(estimate.Vol, alpha=alpha, dist=es_dist, **kwargs)
            backtest_by_year(
                f"[{scenario_label}] Variance-Covariance ({label})", estimate.VaR, es,
                sample, alpha, year_range=SHOCK_YEAR_RANGE,
            )

        hs_estimate = Historic_Simulation(sample.portfolio_loss, sample.window_size, sample.weights, alpha)
        backtest_by_year(
            f"[{scenario_label}] Historic Simulation", hs_estimate.VaR, hs_estimate.ES,
            sample, alpha, year_range=SHOCK_YEAR_RANGE,
        )

        garch_vol = GARCH_portfolio_variance(sample.ind_loss, weights=sample.weights)
        garch_var = pd.Series(GARCH_VaR(garch_vol, alpha, stats.norm.ppf), index=sample.ind_loss.index)[sample.window_size:]
        garch_es = pd.Series(Var_Cov_ES(garch_vol, alpha, stats.norm), index=sample.ind_loss.index)[sample.window_size:]
        backtest_by_year(
            f"[{scenario_label}] CCC-GARCH", garch_var, garch_es,
            sample, alpha, year_range=SHOCK_YEAR_RANGE,
        )

        ewma_estimate = filtered_HS(sample.ind_loss, sample.weights, alpha, sample.window_size)
        backtest_by_year(
            f"[{scenario_label}] EWMA-FHS", ewma_estimate.VaR, ewma_estimate.ES,
            sample, alpha, year_range=SHOCK_YEAR_RANGE,
        )


def main():
    raw_data = load_stock_euribor_portfolio(
        start_date=DEFAULT_PORTFOLIO.start_date,
        end_date=DEFAULT_PORTFOLIO.end_date,
    )

    print("=" * 80)
    print("Scenario 1: AAPL price shock (-20% / -36% over last week of Dec 2021)")
    print("=" * 80)
    aapl_shocked = shock_aapl(raw_data)
    run_all_models(build_portfolio_sample(aapl_shocked, DEFAULT_PORTFOLIO.weights), "AAPL shock")

    print("=" * 80)
    print("Scenario 2: AAPL + EURIBOR shock (+1% / +2% rate shock over same window)")
    print("=" * 80)
    combined_shocked = shock_euribor(shock_aapl(raw_data))
    run_all_models(build_portfolio_sample(combined_shocked, DEFAULT_PORTFOLIO.weights), "AAPL+EURIBOR shock")


if __name__ == "__main__":
    main()
# Covers stress scenario construction.
