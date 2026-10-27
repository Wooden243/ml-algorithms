"""Unit tests: invariants, boundary conditions, and reproducibility."""

import numpy as np
import pytest

from bandit import simulate_budget_path


def test_output_shape_ranges_and_accounting():
    path, m = simulate_budget_path(alpha=0.2, epsilon=0.1, n_runs=200, T=25, seed=7)
    assert path.shape == (26,)
    assert path[0] == 20
    assert np.all(path >= 0)
    assert 0 <= m["bankruptcy_rate"] <= 1
    assert 0 <= m["good_arm_rate"] <= 1
    assert 0 < m["mean_steps"] <= 25
    assert np.isclose(m["final_expected_budget"], 20 + m["mean_total_reward"])


def test_fixed_seed_is_reproducible():
    a = simulate_budget_path(0.2, 0.1, n_runs=50, T=30, seed=123)
    b = simulate_budget_path(0.2, 0.1, n_runs=50, T=30, seed=123)
    np.testing.assert_array_equal(a[0], b[0])
    assert a[1] == b[1]


def test_always_hit_one_arm_no_bankruptcy():
    path, m = simulate_budget_path(0.5, 0.4, initial_budget=2, K=1,
                                   T=7, n_runs=30, p_good=1, p_bad=0, seed=9)
    np.testing.assert_array_equal(path, np.arange(2, 10, dtype=float))
    assert m["good_arm_rate"] == 1
    assert m["bankruptcy_rate"] == 0
    assert m["mean_total_reward"] == 7


def test_always_miss_all_runs_go_bankrupt():
    path, m = simulate_budget_path(0.5, 0.4, initial_budget=3, K=5,
                                   T=8, n_runs=30, p_good=0, p_bad=0, seed=9)
    np.testing.assert_array_equal(path, [3, 2, 1, 0, 0, 0, 0, 0, 0])
    assert m["bankruptcy_rate"] == 1
    assert m["mean_steps"] == 3


@pytest.mark.parametrize("kwargs", [
    {"alpha": 0}, {"alpha": 1.01}, {"epsilon": -0.1}, {"epsilon": 1.1},
    {"K": 0}, {"T": 0}, {"n_runs": 0}, {"initial_budget": -1},
    {"p_good": 1.1}, {"p_bad": -0.1}, {"seed": -2},
])
def test_invalid_config_rejected(kwargs):
    args = {"alpha": 0.2, "epsilon": 0.1}
    args.update(kwargs)
    with pytest.raises(ValueError):
        simulate_budget_path(**args)
