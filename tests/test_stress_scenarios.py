import numpy as np
import pandas as pd

from riskfunctions.models.stress_testing import shock_aapl, shock_euribor


def _sample_portfolio():
    idx = pd.date_range("2021-12-01", "2022-01-10", freq="D")
    return pd.DataFrame(
        {
            "AAPL": np.full(len(idx), 100.0),
            "EURIBOR": np.full(len(idx), 0.5),
        },
        index=idx,
    )


def test_shock_aapl_leaves_outside_window_untouched():
    data = _sample_portfolio()
    shocked = shock_aapl(data)
    outside = shocked.loc[shocked.index < "2021-12-23"]
    assert (outside["AAPL"] == 100.0).all()


def test_shock_aapl_compounds_in_inner_window():
    data = _sample_portfolio()
    shocked = shock_aapl(data, factor=0.8)
    outer_only_day = pd.Timestamp("2021-12-23")
    inner_day = pd.Timestamp("2021-12-26")

    # Outer window only: single shock (100 * 0.8)
    assert np.isclose(shocked.loc[outer_only_day, "AAPL"], 100.0 * 0.8)
    # Inner window: shocked twice (100 * 0.8 * 0.8)
    assert np.isclose(shocked.loc[inner_day, "AAPL"], 100.0 * 0.8 * 0.8)


def test_shock_euribor_additive_and_compounding():
    data = _sample_portfolio()
    shocked = shock_euribor(data, bump=1.0)
    outer_only_day = pd.Timestamp("2021-12-23")
    inner_day = pd.Timestamp("2021-12-26")

    assert np.isclose(shocked.loc[outer_only_day, "EURIBOR"], 0.5 + 1.0)
    assert np.isclose(shocked.loc[inner_day, "EURIBOR"], 0.5 + 2.0)


def test_shocks_do_not_mutate_input():
    data = _sample_portfolio()
    original = data.copy()
    shock_aapl(data)
    shock_euribor(data)
    pd.testing.assert_frame_equal(data, original)
