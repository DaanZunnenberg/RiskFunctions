# RiskFunctions

Risk models, organized by model family, as an installable Python package.

## Layout

```
src/riskfunctions/           Library code (importable, no side effects on import)
    data.py                  Portfolio loading + cleaning (yfinance, Euribor CSV, close-prices .ods)
    loss.py                  Loss operators (loss_operator, close_prices_loss_operator)
    backtest.py               Backtest, calculate_distances, christoffersen_test, binomial_test
    volatility.py             Shared GARCH(1,1)-style volatility filter (used by CCC-GARCH and EWMA-FHS)
    config.py                 Portfolio/backtest configuration (weights, alphas, date ranges)
    pipeline.py                Shared "load portfolio -> build sample -> backtest by year" plumbing
    models/
        var_covariance/       Var_Cov_VaR, Var_Cov_ES
        historic_simulation/  Historic_Simulation
        ccc_garch/             GARCH_portfolio_variance, GARCH_VaR
        ewma_fhs/              filtered_HS
        stress_testing/       AAPL / Euribor shock scenario generators
        copulas/               Bivariate Archimedean/Gaussian/Student-t + multivariate copula fitting
        factor_analysis/       PCA + Factor Analysis (varimax/oblimin rotation)

examples/                     Runnable scripts demonstrating each model end-to-end
tests/                        Unit tests for the pure/deterministic library functions
data/                         Input data (Euribor CSV, cached stock quotes, close-prices workbook)
docs/                         Reference PDFs
```

Each `models/<name>/` package holds only estimator/analysis functions:
pure functions operating on DataFrames/arrays, no I/O beyond what a caller
explicitly passes in, no `print`/`plt.show()` side effects baked into the
math. Anything that prints results, produces plots, or wires together data
loading + a model + backtesting lives in `examples/`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"          # core + test dependencies
pip install -e ".[multivariate]" # needed only for the copulas/factor_analysis examples
```

## Running the examples

All examples are run from the repo root so the `src/` layout resolves:

```bash
python examples/var_covariance_example.py
python examples/historic_simulation_example.py
python examples/ccc_garch_example.py
python examples/ewma_fhs_example.py
python examples/stress_testing_example.py      # requires network access (yfinance)
python examples/copulas_example.py             # requires network access + [multivariate] extra
python examples/factor_analysis_example.py     # requires network access + [multivariate] extra
```

The Variance-Covariance / Historic Simulation / CCC-GARCH / EWMA-FHS
examples run entirely off the cached `data/stock_quotes_and_rates.csv` and
do not require network access.

## Running the tests

```bash
pytest
```

## Data provenance

- `data/BundesBank Euribor 3M Daily.csv`, `data/stock_quotes_and_rates.csv`:
  stock+Euribor portfolio (AAPL, ASML, AIRBUS, LVMH, EURO50, EUR/USD, Euribor 3M).
- `data/close_prices_portfolio.ods`: alternate close-prices portfolio
  (S&P 500 in EUR, DAX, AEX, Euribor rates), used historically for a
  separate CCC-GARCH/backtesting exploration; not wired into the current
  examples but the loader (`riskfunctions.data.load_close_prices_portfolio`)
  and matching loss operator (`riskfunctions.loss.close_prices_loss_operator`)
  are kept for reference.
- `docs/`: reference PDFs (HS/QQ plots, VaR-Covar comparisons), one pair
  per portfolio since the two were not byte-identical.

<!-- Last reviewed: 2026-11-03 -->
