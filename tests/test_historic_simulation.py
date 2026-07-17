import numpy as np
import pandas as pd

from riskfunctions.models.historic_simulation import Historic_Simulation


def test_historic_simulation_drops_warmup_and_has_var_es_columns():
    rng = np.random.default_rng(5)
    losses = pd.Series(rng.normal(size=100))
    result = Historic_Simulation(losses, window_size=20, port_weights=None, alpha=0.95)
    assert list(result.columns) == ["VaR", "ES"]
    assert len(result) <= 100 - 20


def test_historic_simulation_es_at_least_var_on_average():
    rng = np.random.default_rng(6)
    losses = pd.Series(rng.normal(size=300))
    result = Historic_Simulation(losses, window_size=50, port_weights=None, alpha=0.95)
    # ES conditions on losses >= VaR, so on average ES should not be below VaR
    assert (result.ES.to_numpy() >= result.VaR.to_numpy() - 1e-9).mean() > 0.9
