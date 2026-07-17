import numpy as np

from riskfunctions.volatility import volatility


def test_volatility_returns_correct_length():
    rng = np.random.default_rng(1)
    x = rng.normal(size=800)
    vol = volatility(omega=1e-6, alpha=0.05, beta=0.93, x=x)
    assert len(vol) == len(x)


def test_volatility_is_nonnegative():
    rng = np.random.default_rng(2)
    x = rng.normal(size=500)
    vol = volatility(omega=1e-6, alpha=0.05, beta=0.93, x=x)
    assert np.all(vol >= 0)


def test_volatility_zero_shocks_decays_to_omega_steady_state():
    # With alpha=0 and zero shocks, sigma_t^2 should equal omega (after the
    # first observation seeds it via the unconditional variance).
    x = np.zeros(200)
    vol = volatility(omega=0.04, alpha=0.0, beta=0.5, x=x)
    # sigma[1] = omega + 0 + beta * sigma[0]; sigma[0] is seeded from np.var(x[:50]) == 0
    assert np.isclose(vol[1] ** 2, 0.04)
