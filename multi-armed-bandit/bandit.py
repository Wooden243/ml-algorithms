"""Vectorized epsilon-greedy simulations for a budget-constrained multi-armed bandit.

A single favorable arm (p_good) and K-1 ordinary arms (p_bad) are sampled
independently for each run. Each pull gives a reward of +1 (hit) or -1 (miss).
The run ends when the initial budget reaches 0 or after T pulls.

This is a *stateless multi-armed bandit*, not Q-learning over MDP states.
"""
from __future__ import annotations

import numpy as np


def simulate_budget_path(
    alpha: float,
    epsilon: float,
    initial_budget: int = 20,
    K: int = 10,
    T: int = 100,
    n_runs: int = 5_000,
    p_good: float = 0.65,
    p_bad: float = 0.44,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return expected budget at times 0..T and per-run-averaged metrics.

    Parameters
    ----------
    alpha : float
        Constant step size for action-value updates, 0 < alpha <= 1.
    epsilon : float
        Probability of selecting a uniformly random arm, 0 <= epsilon <= 1.
    initial_budget : int
        Positive initial number of budget units.
    K : int
        Number of arms; exactly one favorable arm per run.
    T : int
        Maximum number of pulls per run.
    n_runs : int
        Independent runs, vectorized across the run axis.
    p_good, p_bad : float
        Bernoulli hit probabilities; +1 if hit and -1 otherwise.
    seed : int
        Seed for independent NumPy Generator to allow exact replication.

    The good-arm rate is the *mean of within-run proportions*, not the pooled
    proportion across all pulls. This matches the original course notebook.
    """
    if not 0 < alpha <= 1:
        raise ValueError("alpha must satisfy 0 < alpha <= 1")
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must satisfy 0 <= epsilon <= 1")
    if any(not isinstance(v, (int, np.integer)) or isinstance(v, (bool, np.bool_)) or v < 1
           for v in (initial_budget, K, T, n_runs)):
        raise ValueError("initial_budget, K, T, and n_runs must be positive integers")
    if not (0 <= p_bad <= 1 and 0 <= p_good <= 1):
        raise ValueError("probabilities must be between 0 and 1")
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    rng = np.random.default_rng(seed)
    good_arms = rng.integers(0, K, size=n_runs)
    q_values = np.zeros((n_runs, K), dtype=float)
    wealth = np.full(n_runs, float(initial_budget))
    budget_paths = np.zeros((n_runs, T + 1), dtype=float)
    budget_paths[:, 0] = wealth

    total_reward = np.zeros(n_runs)
    steps_played = np.zeros(n_runs, dtype=int)
    bankrupt = np.zeros(n_runs, dtype=bool)
    good_arm_count = np.zeros(n_runs)
    alive = np.ones(n_runs, dtype=bool)

    for t in range(1, T + 1):
        active = np.where(alive)[0]
        if active.size == 0:
            budget_paths[:, t:] = 0
            break

        # Randomized tie breaking to avoid index-dependent preference.
        jitter = rng.uniform(0, 1e-12, size=(len(active), K))
        greedy_actions = np.argmax(q_values[active] + jitter, axis=1)
        random_actions = rng.integers(0, K, size=len(active))
        explore = rng.random(len(active)) < epsilon
        actions = np.where(explore, random_actions, greedy_actions)

        selected_is_good = actions == good_arms[active]
        p_hit = np.where(selected_is_good, p_good, p_bad)
        hits = rng.random(len(active)) < p_hit
        rewards = np.where(hits, 1.0, -1.0)

        old_q = q_values[active, actions]
        q_values[active, actions] = old_q + alpha * (rewards - old_q)

        wealth[active] += rewards
        total_reward[active] += rewards
        steps_played[active] += 1
        good_arm_count[active] += selected_is_good

        newly_bankrupt = wealth[active] <= 0
        if np.any(newly_bankrupt):
            bankrupt_indices = active[newly_bankrupt]
            bankrupt[bankrupt_indices] = True
            alive[bankrupt_indices] = False
            wealth[bankrupt_indices] = 0.0
        budget_paths[:, t] = wealth

    expected_budget_path = budget_paths.mean(axis=0)
    safe_steps = np.maximum(steps_played, 1)
    summary = {
        "alpha": float(alpha),
        "epsilon": float(epsilon),
        "initial_budget": initial_budget,
        "final_expected_budget": float(expected_budget_path[-1]),
        "mean_total_reward": float(total_reward.mean()),
        "bankruptcy_rate": float(bankrupt.mean()),
        "mean_steps": float(steps_played.mean()),
        "good_arm_rate": float((good_arm_count / safe_steps).mean()),
    }
    return expected_budget_path, summary
