"""
Loss operator functions shared across all risk models.

There are two loss operators in the original coursework, corresponding to
the two portfolios in `common/data.py`:

- `loss_operator`: for the stock+Euribor portfolio (Assignment 1), which
  models AAPL FX exposure and Euribor credit/time risk explicitly.
- `group4_loss_operator`: for the close-prices portfolio (Assignment 2),
  a simpler weighted-log-return loss with a scaled Euribor rate risk term.
"""

import numpy as np
import pandas as pd


def loss_operator(Z: pd.DataFrame, weights, Value: float = 10 ** 8):
    """
    The loss distribution is the weighted sum of negative log returns (X)
    multiplied by exp(Z(t) - Z(0)) where Z(x) is the log price at time x.

    NOTE: For us, Z(0) = 0 as all returns are standardised at t = 0.

    Parameters
    ----------
    Z : pd.DataFrame
        Log-transformed risk factors (output of
        `common.data.prepare_log_portfolio`), columns must include
        'EUR_USD' and 'EURIBOR'.
    weights : array-like
        Portfolio weights aligned with Z.columns.
    Value : float
        Notional portfolio value.

    Returns
    -------
    portfolio_loss : pd.Series (negated)
    individual_loss : pd.DataFrame (negated), one column per risk factor
    """
    loss = np.zeros(len(Z) - 1)
    loss_ind = Z[:-1].copy()
    T = len(Z) - 1
    t = np.arange(T)
    for Z_k, weight, name in zip(Z.values.T, weights, Z.columns):
        X_k = np.diff(Z_k)
        X_k = np.array([0] + X_k.tolist())
        X_k_shifted = X_k[1:]  # from t = 2 to T
        X_k = X_k[:-1]  # from t = 1 to T - 1
        Z_k = Z_k[:-1]
        if name == "AAPL":
            # Include EUR/USD exposure
            loss_sub = Value * weight * np.exp((Z_k - Z_k[0]) + (Z["EUR_USD"][:-1] - Z["EUR_USD"][0])) * X_k_shifted
        elif name == "EURIBOR":
            # The time risk
            w = Value * np.exp(Z_k[0])
            loss_sub = -w * np.exp(-(T - t) * Z_k) * Z_k

            # Add credit risk
            loss_sub += Value * weight * np.exp(Z_k[0] - (T - t) * Z_k) * (T - t) * X_k
        else:
            loss_sub = Value * weight * np.exp(Z_k - Z_k[0]) * X_k_shifted
        loss += loss_sub
        loss_ind[name] = loss_sub * (1 / weight) if name != "EUR_USD" else 0
    return -loss, -loss_ind


def group4_loss_operator(log_data: pd.DataFrame):
    """
    Simplified loss operator for the close-prices portfolio: S&P 500 (EUR),
    DAX, AEX with fixed 0.2 weights each, plus a scaled Euribor rate-risk
    term (0.4 weight).

    Parameters
    ----------
    log_data : pd.DataFrame
        Output of `common.data.prepare_log_close_prices`, with columns
        'S&P 500 in EUR', 'DAX', 'AEX', 'Euribor rates', 'USD/EUR'.

    Returns
    -------
    portfolio_loss : pd.Series (negated)
    individual_loss : pd.DataFrame (negated), columns
        ['S&P 500', 'DAX', 'AEX', 'EURIBOR']
    """
    rates = log_data["Euribor rates"]
    rates_risk = 0.4 * (0.0002 + 0.25 * rates.diff().fillna(0))
    other_risk = 0.2 * log_data[["DAX", "AEX", "S&P 500 in EUR"]].diff().fillna(0)
    ind_loss = pd.DataFrame(
        data=log_data[["S&P 500 in EUR", "DAX", "AEX", "Euribor rates"]].diff().fillna(0).values,
        index=log_data.index,
        columns=["S&P 500", "DAX", "AEX", "EURIBOR"],
    )
    ind_loss["EURIBOR"] = rates_risk
    return -1 * (other_risk.sum(axis=1) + rates_risk), -1 * ind_loss
