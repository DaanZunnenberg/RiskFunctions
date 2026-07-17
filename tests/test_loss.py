import numpy as np
import pandas as pd

from riskfunctions.loss import group4_loss_operator
from riskfunctions.data import credit_risk


def test_credit_risk_returns_matching_lengths():
    idx = pd.date_range("2020-01-01", periods=50, freq="D")
    credit = pd.Series(np.linspace(0.0001, 0.0002, 50), index=idx)
    loss, val = credit_risk(credit)
    assert len(loss) == len(credit)
    assert len(val) == len(credit)


def test_credit_risk_short_flips_sign_vs_long():
    idx = pd.date_range("2020-01-01", periods=30, freq="D")
    credit = pd.Series(np.linspace(0.0001, 0.0003, 30), index=idx)
    loss_short, _ = credit_risk(credit, short=True)
    loss_long, _ = credit_risk(credit, short=False)
    np.testing.assert_allclose(loss_short, -loss_long)


def test_group4_loss_operator_output_shape_and_columns():
    idx = pd.date_range("2020-01-01", periods=20, freq="D")
    rng = np.random.default_rng(7)
    log_data = pd.DataFrame(
        {
            "S&P 500 in EUR": np.cumsum(rng.normal(scale=0.01, size=20)),
            "DAX": np.cumsum(rng.normal(scale=0.01, size=20)),
            "AEX": np.cumsum(rng.normal(scale=0.01, size=20)),
            "Euribor rates": np.cumsum(rng.normal(scale=0.0001, size=20)),
            "USD/EUR": np.exp(rng.normal(scale=0.001, size=20)),
        },
        index=idx,
    )
    portfolio_loss, ind_loss = group4_loss_operator(log_data)

    assert len(portfolio_loss) == len(log_data)
    assert list(ind_loss.columns) == ["S&P 500", "DAX", "AEX", "EURIBOR"]
    assert len(ind_loss) == len(log_data)
