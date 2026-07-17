"""
Example: EWMA / Filtered Historical Simulation VaR/ES on the stock+Euribor
portfolio, backtested year-by-year at alpha = 0.95 and 0.99.

Run from the repo root:
    python examples/ewma_fhs_example.py
"""

from riskfunctions.config import DEFAULT_BACKTEST
from riskfunctions.models.ewma_fhs import filtered_HS
from riskfunctions.pipeline import backtest_by_year, load_portfolio_sample


def main():
    sample = load_portfolio_sample()

    for alpha in DEFAULT_BACKTEST.alphas:
        estimate = filtered_HS(sample.ind_loss, sample.weights, alpha, sample.window_size)
        backtest_by_year("EWMA-FHS", estimate.VaR, estimate.ES, sample, alpha)


if __name__ == "__main__":
    main()
