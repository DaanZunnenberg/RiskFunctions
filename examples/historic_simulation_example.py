"""
Example: Historic Simulation VaR/ES on the stock+Euribor portfolio,
backtested year-by-year at alpha = 0.95 and 0.99.

Run from the repo root:
    python examples/historic_simulation_example.py
"""

from riskfunctions.config import DEFAULT_BACKTEST
from riskfunctions.models.historic_simulation import Historic_Simulation
from riskfunctions.pipeline import backtest_by_year, load_portfolio_sample


def main():
    sample = load_portfolio_sample()

    for alpha in DEFAULT_BACKTEST.alphas:
        estimate = Historic_Simulation(
            sample.portfolio_loss,
            sample.window_size,
            sample.weights,
            alpha,
        )
        backtest_by_year("Historic Simulation", estimate.VaR, estimate.ES, sample, alpha)


if __name__ == "__main__":
    main()
# Covers historic simulation VaR usage.
