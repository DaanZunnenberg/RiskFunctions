# Test Report

Generated while validating the `src/riskfunctions` package restructure.
Environment: Python 3.11.15 (venv), macOS (Darwin 23.3.0).

```
pandas==2.3.3        numpy==2.4.6          scipy==1.17.1
statsmodels==0.14.6  matplotlib==3.11.0    yfinance==1.5.1
odfpy==1.4.1         seaborn==0.13.2       scikit-learn==1.4.2
factor_analyzer==0.5.1  copulas==0.14.1    pytest==9.1.1
```

## Unit tests

```
pytest -v
```

```
============================= test session starts ==============================
platform darwin -- Python 3.11.15, pytest-9.1.1, pluggy-1.6.0
collected 29 items

tests/test_backtest.py::test_calculate_distances_basic PASSED
tests/test_backtest.py::test_calculate_distances_single_violation_is_empty PASSED
tests/test_backtest.py::test_calculate_distances_no_violations_is_empty PASSED
tests/test_backtest.py::test_christoffersen_test_shape_and_columns PASSED
tests/test_backtest.py::test_christoffersen_test_no_violations_returns_nan PASSED
tests/test_copulas.py::test_to_uniform_is_in_unit_interval PASSED
tests/test_copulas.py::test_all_pairs_no_duplicates_or_self_pairs PASSED
tests/test_copulas.py::test_fit_archimedean_pair_returns_one_fit_per_model PASSED
tests/test_copulas.py::test_fit_multivariate_copula_correlation_reflects_input_correlation PASSED
tests/test_factor_analysis.py::test_pca_loadings_shape PASSED
tests/test_factor_analysis.py::test_pca_first_component_dominated_by_correlated_assets PASSED
tests/test_factor_analysis.py::test_factor_analysis_loadings_shape_and_rotation_options PASSED
tests/test_historic_simulation.py::test_historic_simulation_drops_warmup_and_has_var_es_columns PASSED
tests/test_historic_simulation.py::test_historic_simulation_es_at_least_var_on_average PASSED
tests/test_loss.py::test_credit_risk_returns_matching_lengths PASSED
tests/test_loss.py::test_credit_risk_short_flips_sign_vs_long PASSED
tests/test_loss.py::test_close_prices_loss_operator_output_shape_and_columns PASSED
tests/test_stress_scenarios.py::test_shock_aapl_leaves_outside_window_untouched PASSED
tests/test_stress_scenarios.py::test_shock_aapl_compounds_in_inner_window PASSED
tests/test_stress_scenarios.py::test_shock_euribor_additive_and_compounding PASSED
tests/test_stress_scenarios.py::test_shocks_do_not_mutate_input PASSED
tests/test_var_covariance.py::test_var_cov_var_drops_warmup_window PASSED
tests/test_var_covariance.py::test_var_cov_var_scales_with_quantile PASSED
tests/test_var_covariance.py::test_var_cov_es_normal_exceeds_var_quantile_multiplier PASSED
tests/test_var_covariance.py::test_var_cov_es_student_t_requires_df_default PASSED
tests/test_var_covariance.py::test_var_cov_es_unsupported_distribution_raises PASSED
tests/test_volatility.py::test_volatility_returns_correct_length PASSED
tests/test_volatility.py::test_volatility_is_nonnegative PASSED
tests/test_volatility.py::test_volatility_zero_shocks_decays_to_omega_steady_state PASSED

============================== 29 passed in 1.52s ===============================
```

## Static checks

```
python -m compileall -q src examples tests     # exit 0, no syntax errors
python -c "import riskfunctions; import riskfunctions.models.*"   # all modules import cleanly
```

## End-to-end example runs (no network required; use cached `data/stock_quotes_and_rates.csv`)

| Example | Exit code | Notes |
|---|---|---|
| `examples/var_covariance_example.py` | 0 | Overall VaR99: 35 violations vs 33.6 expected; ES z-test p=0.667 (not significant) |
| `examples/historic_simulation_example.py` | 0 | Overall VaR99: 51 violations vs 33.6 expected (binomial test flags this, as expected for a naive empirical-quantile method); ES z-test p=0.751 |
| `examples/ccc_garch_example.py` | 0 | Overall VaR99: 70 violations vs 33.6 expected (flagged); ES z-test p=0.112 |
| `examples/ewma_fhs_example.py` | 0 | Overall VaR99: 37 violations vs 33.6 expected; ES z-test p=0.909 |

`examples/stress_testing_example.py`, `examples/copulas_example.py`, and
`examples/factor_analysis_example.py` require network access (yfinance
downloads) and were validated by import + a synthetic-data smoke test
covering the same estimator code paths (see `tests/test_copulas.py`,
`tests/test_factor_analysis.py`, `tests/test_stress_scenarios.py`) rather
than a live network run in this environment.

## Bugs found and fixed during validation

Running the examples end-to-end (not just importing them) surfaced three
real bugs inherited from the original notebooks, all caused by mixing
positional (`x[t]`) and pandas label-based indexing on DatetimeIndex-ed
Series; this used to silently work on older pandas and now raises
`KeyError`:

1. **`riskfunctions/loss.py` (`loss_operator`)** - `Z["EUR_USD"][0]` /
   `Z["EUR_USD"][:-1]` used positional-style access on a label (date)
   index. Fixed by coercing to `.to_numpy()` before indexing.
2. **`riskfunctions/loss.py` (`loss_operator`)** - the returned
   `portfolio_loss` was a bare `np.ndarray`, not a `pd.Series`, so any
   caller doing `.index`/`.iloc` on it (e.g. `Historic_Simulation`) broke.
   Fixed by wrapping the result in a `pd.Series` indexed like the input.
3. **`riskfunctions/volatility.py` (`volatility`)** - `x[t]` assumed `x` is
   positionally indexable, but callers pass a DatetimeIndex-ed `pd.Series`
   column directly. Fixed by coercing `x` to `np.asarray(x)` up front.
4. **`riskfunctions/backtest.py` (`christoffersen_test`)** - computed
   `p01`/`p11` unconditionally (causing a spurious 0/0 `RuntimeWarning`)
   even though they're only used inside the `n1 > 0` branch. Fixed by
   moving the computation inside the guard with explicit zero-division
   handling.
5. **`pyproject.toml`** - `factor_analyzer==0.5.1` calls
   `sklearn.check_array(force_all_finite=...)`, a kwarg removed in
   `scikit-learn>=1.5` (renamed `ensure_all_finite`). Pinned
   `scikit-learn>=1.3,<1.5` in the `multivariate` extra until upstream
   releases a fix.

None of these change the underlying model math; they are indexing/typing
fixes required to make the extracted functions actually runnable outside
the original notebook's execution order and pandas version.
