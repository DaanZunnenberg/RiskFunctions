"""
Shared backtesting utilities used to validate all VaR/ES models.

Includes:
- `calculate_distances`: inter-arrival distances between VaR violations.
- `Backtest`: binomial coverage test for VaR + z-test for ES accuracy,
  with an optional QQ-plot of violation inter-arrival times vs Exponential.
- `binomial_test`: standalone two-sided binomial test of violation counts
  (used ad hoc in the CCC-GARCH exploration notebook).
- `christoffersen_test`: Christoffersen (1998) likelihood-ratio framework
  (unconditional coverage / independence / conditional coverage), which
  supplements the simpler binomial `Backtest` above.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt


def calculate_distances(array) -> np.ndarray:
    """
    Given a 0/1 array of VaR violations, return the distances (in indices)
    between consecutive violations ("inter-arrival times").
    """
    ones_indices = np.where(array == 1)[0]
    if len(ones_indices) > 1:
        distances = np.diff(ones_indices)
    else:
        distances = np.array([])
    return distances


def Backtest(VaR, ES, Loss, VaR_alpha: float, qqplot_arrival: bool = False):
    """
    Backtest a VaR/ES pair against realised losses.

    - Binomial test of the number of VaR violations vs the expected count
      at the given confidence level.
    - Two-sample z-test comparing the average realised loss conditional on
      a violation (empirical ES) against the average predicted ES.
    - Optionally, a QQ-plot of violation inter-arrival times against the
      Exponential distribution (a well-specified VaR model should produce
      Poisson-arrival violations).

    Parameters
    ----------
    VaR, ES, Loss : array-like
        VaR estimates, ES estimates and realised losses, aligned in time.
    VaR_alpha : float
        Confidence level used for VaR (e.g. 0.95, 0.975, 0.99).
    qqplot_arrival : bool
        If True, plot the inter-arrival QQ-plot.
    """
    Violations_VaR = Loss > VaR
    print("\nVaR Violations: ", sum(Violations_VaR))

    n = len(Violations_VaR)
    p = 1 - VaR_alpha
    x = sum(Violations_VaR)
    alpha = 0.05

    p_value_left = stats.binom.cdf(x, n, p)
    p_value_right = 1 - stats.binom.cdf(x - 1, n, p)
    p_value = min(p_value_left, p_value_right) * 2

    if p_value <= alpha:
        print("VaR Violations amount differs significantly from Expected")

    if qqplot_arrival:
        distances = calculate_distances(Violations_VaR)
        if len(distances) > 0:
            plt.figure(figsize=(12, 12))
            sm.qqplot(distances, dist=stats.expon, line="q", ax=plt.subplot(1, 1, 1))
            plt.title("QQ-Plot: InterArrival Distance vs. Exponential Distribution")
            plt.show()
        else:
            print("Not enough data points (1s) to generate a QQ-plot.")

    average_Real_ES = np.mean(Loss[Violations_VaR])
    average_Theoretical_ES = np.mean(ES[Violations_VaR])
    print("\nAverage ES Real and Theoretical: ", average_Real_ES, average_Theoretical_ES)

    standard_error = np.std(Loss[Violations_VaR]) / np.sqrt(np.sum(Violations_VaR))
    Z_Test_ES = (np.abs(average_Real_ES - average_Theoretical_ES)) / standard_error
    ES_p = stats.norm.sf(abs(Z_Test_ES)) * 2
    print("ES t-Test significance of difference: ", ES_p, " | Reject if <0.05\n\n")


def binomial_test(violations, p: float):
    """Two-sided binomial test of the observed violation count vs expected probability p."""
    n = len(violations)
    k = int(np.sum(violations))
    return stats.binomtest(k=k, n=n, p=p)


def VaR_violations(loss: pd.DataFrame, VaR) -> tuple:
    """Return (violation dates, 0/1 violation DataFrame) for loss vs VaR."""
    try:
        vv = np.where(loss >= VaR, 1, 0)
        vv = pd.DataFrame(vv, index=loss.index, columns=["violation dates"])
    except Exception:
        vv = np.where(loss.loc[VaR.index] >= VaR, 1, 0)
        vv = pd.DataFrame(vv, index=VaR.index, columns=["violation dates"])
    return vv[vv["violation dates"] == 1].index, vv


def christoffersen_test(hits: np.ndarray, alpha: float) -> pd.DataFrame:
    """
    Likelihood ratio framework of Christoffersen (1998) for VaR backtesting.

    Returns a DataFrame with Statistic/p-value rows for Unconditional
    coverage, Independence, and Conditional coverage tests.
    """
    tr = hits[1:] - hits[:-1]  # Sequence to find transitions

    n01, n10 = (tr == 1).sum(), (tr == -1).sum()
    n11, n00 = (hits[1:][tr == 0] == 1).sum(), (hits[1:][tr == 0] == 0).sum()

    n0, n1 = n01 + n00, n10 + n11
    n = n0 + n1

    if n1 > 0:
        # Guard against 0/0 divisions when a transition count is zero;
        # these p01/p11 are only used below, inside the n1 > 0 branch.
        p01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
        p11 = n11 / (n11 + n10) if (n11 + n10) > 0 else 0.0
        p = n1 / n

        uc_h0 = n0 * np.log(1 - alpha) + n1 * np.log(alpha)
        uc_h1 = n0 * np.log(1 - p) + n1 * np.log(p)
        uc = -2 * (uc_h0 - uc_h1)

        ind_h0 = (n00 + n01) * np.log(1 - p) + (n01 + n11) * np.log(p)
        ind_h1 = n00 * np.log(1 - p01) + n01 * np.log(p01) + n10 * np.log(1 - p11)
        if p11 > 0:
            ind_h1 += n11 * np.log(p11)
        ind = -2 * (ind_h0 - ind_h1)

        cc = uc + ind

        df = pd.concat(
            [
                pd.Series([uc, ind, cc]),
                pd.Series(
                    [
                        1 - stats.chi2.cdf(uc, 1),
                        1 - stats.chi2.cdf(ind, 1),
                        1 - stats.chi2.cdf(cc, 2),
                    ]
                ),
            ],
            axis=1,
        )
    else:
        df = pd.DataFrame(np.zeros((3, 2))).replace(0, np.nan)

    df.columns = ["Statistic", "p-value"]
    df.index = ["Unconditional", "Independence", "Conditional"]

    return df
