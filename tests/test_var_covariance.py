import numpy as np
import pandas as pd
import pytest
import scipy.stats as stats

from riskfunctions.models.var_covariance import Var_Cov_ES, Var_Cov_VaR


def test_var_cov_var_drops_warmup_window():
    rng = np.random.default_rng(3)
    data = pd.DataFrame(rng.normal(size=(100, 2)), columns=["A", "B"])
    result = Var_Cov_VaR(data, window_size=20, port_weights=np.array([0.5, 0.5]), alpha=0.95, dist=stats.norm.ppf)
    assert len(result) == 100 - 20
    assert list(result.columns) == ["Vol", "VaR"]


def test_var_cov_var_scales_with_quantile():
    rng = np.random.default_rng(4)
    data = pd.DataFrame(rng.normal(size=(60, 2)), columns=["A", "B"])
    weights = np.array([0.5, 0.5])
    result_95 = Var_Cov_VaR(data, window_size=10, port_weights=weights, alpha=0.95, dist=stats.norm.ppf)
    result_99 = Var_Cov_VaR(data, window_size=10, port_weights=weights, alpha=0.99, dist=stats.norm.ppf)
    # Same volatility path, higher alpha -> larger VaR multiplier -> larger VaR
    assert (result_99.VaR.to_numpy() >= result_95.VaR.to_numpy()).all()


def test_var_cov_es_normal_exceeds_var_quantile_multiplier():
    vol = np.array([1.0, 2.0, 3.0])
    es = Var_Cov_ES(vol, alpha=0.95, dist=stats.norm)
    var_multiplier = stats.norm.ppf(0.95)
    # ES multiplier for a Normal tail always exceeds the VaR quantile multiplier
    assert (es > vol * var_multiplier).all()


def test_var_cov_es_student_t_requires_df_default():
    vol = np.array([1.0])
    es_default = Var_Cov_ES(vol, alpha=0.95, dist=stats.t)
    es_df4 = Var_Cov_ES(vol, alpha=0.95, dist=stats.t, df=4)
    assert np.isclose(es_default[0], es_df4[0])


def test_var_cov_es_unsupported_distribution_raises():
    with pytest.raises(NotImplementedError):
        Var_Cov_ES(np.array([1.0]), alpha=0.95, dist=stats.expon)
# Checks variance-covariance VaR calculations.
