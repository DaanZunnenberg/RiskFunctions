import numpy as np

from riskfunctions.backtest import calculate_distances, christoffersen_test


def test_calculate_distances_basic():
    violations = np.array([0, 1, 0, 0, 1, 0, 1])
    # violations at indices 1, 4, 6 -> distances [3, 2]
    np.testing.assert_array_equal(calculate_distances(violations), [3, 2])


def test_calculate_distances_single_violation_is_empty():
    violations = np.array([0, 0, 1, 0, 0])
    assert len(calculate_distances(violations)) == 0


def test_calculate_distances_no_violations_is_empty():
    violations = np.array([0, 0, 0])
    assert len(calculate_distances(violations)) == 0


def test_christoffersen_test_shape_and_columns():
    rng = np.random.default_rng(0)
    hits = (rng.random(500) < 0.05).astype(int)
    result = christoffersen_test(hits, alpha=0.05)

    assert list(result.columns) == ["Statistic", "p-value"]
    assert list(result.index) == ["Unconditional", "Independence", "Conditional"]
    assert result.shape == (3, 2)


def test_christoffersen_test_no_violations_returns_nan():
    hits = np.zeros(100, dtype=int)
    result = christoffersen_test(hits, alpha=0.05)
    assert result.isna().all().all()
