"""Reproduce the course notebook's nine-condition epsilon-greedy experiment.

Run from any directory with:
    python multi-armed-bandit/experiment.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Headless CI and script execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bandit import simulate_budget_path

ALPHAS = {"low": 0.05, "middle": 0.20, "high": 0.80}
EPSILONS = {"low": 0.02, "middle": 0.10, "high": 0.40}
INITIAL_BUDGET = 20
N_RUNS = 5_000
T = 100
BASE_SEED = 12_345


def run_experiment(results_dir: Path, n_runs: int = N_RUNS) -> pd.DataFrame:
    results_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[tuple[str, str], np.ndarray] = {}
    records = []

    # Deliberately preserve SEED + counter to reproduce the 2026 course run.
    counter = 0
    for alpha_name, alpha in ALPHAS.items():
        for epsilon_name, epsilon in EPSILONS.items():
            counter += 1
            path, summary = simulate_budget_path(
                alpha=alpha,
                epsilon=epsilon,
                initial_budget=INITIAL_BUDGET,
                n_runs=n_runs,
                T=T,
                seed=BASE_SEED + counter,
            )
            paths[(alpha_name, epsilon_name)] = path
            records.append({**summary, "alpha_level": alpha_name,
                            "epsilon_level": epsilon_name})

    df = pd.DataFrame.from_records(records)
    df.to_csv(results_dir / "summary.csv", index=False)

    times = np.arange(T + 1)
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True, sharey=True)
    for i, (alpha_name, alpha) in enumerate(ALPHAS.items()):
        for j, (epsilon_name, epsilon) in enumerate(EPSILONS.items()):
            ax = axes[i, j]
            ax.plot(times, paths[(alpha_name, epsilon_name)])
            ax.axhline(INITIAL_BUDGET, linestyle="--", linewidth=1)
            ax.set_title(f"alpha={alpha}, epsilon={epsilon}")
            ax.set_xlabel("Step")
            ax.set_ylabel("Mean budget")
            ax.grid(True, alpha=0.3)
    fig.suptitle("Expected budget trajectories (9 conditions)")
    fig.tight_layout()
    fig.savefig(results_dir / "budget_paths_grid.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 7))
    for alpha_name, alpha in ALPHAS.items():
        for epsilon_name, epsilon in EPSILONS.items():
            ax.plot(times, paths[(alpha_name, epsilon_name)],
                    label=f"alpha={alpha}, epsilon={epsilon}")
    ax.axhline(INITIAL_BUDGET, linestyle="--", linewidth=1, label="initial budget")
    ax.set(xlabel="Step", ylabel="Expected budget", title="Expected budget across 9 conditions")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / "budget_paths_overlay.png", dpi=150)
    plt.close(fig)

    for col, basename, label, fmt in [
        ("final_expected_budget", "final_budget_heatmap.png", "Expected final budget", ".2f"),
        ("bankruptcy_rate", "bankruptcy_heatmap.png", "Bankruptcy rate", ".3f"),
        ("good_arm_rate", "good_arm_rate_heatmap.png", "Good arm selection rate", ".3f"),
    ]:
        pivot = df.pivot(index="alpha_level", columns="epsilon_level", values=col)
        pivot = pivot.loc[list(ALPHAS), list(EPSILONS)]
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(pivot.to_numpy(), aspect="auto", origin="upper")
        fig.colorbar(im, ax=ax, label=label)
        ax.set_xticks(range(len(EPSILONS)), [str(x) for x in EPSILONS.values()])
        ax.set_yticks(range(len(ALPHAS)), [str(x) for x in ALPHAS.values()])
        ax.set(xlabel="Exploration rate epsilon", ylabel="Learning rate alpha",
               title=label)
        for i in range(len(ALPHAS)):
            for j in range(len(EPSILONS)):
                ax.text(j, i, format(pivot.to_numpy()[i, j], fmt),
                        ha="center", va="center", color="white")
        fig.tight_layout()
        fig.savefig(results_dir / basename, dpi=150)
        plt.close(fig)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-runs", type=int, default=N_RUNS)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    results = run_experiment(args.output, n_runs=args.n_runs)
    print(results.to_string(index=False))
    best = results.loc[results["final_expected_budget"].idxmax()]
    print("\nBest condition (among nine tested, by final expected budget):")
    print(best[["alpha", "epsilon", "final_expected_budget", "bankruptcy_rate"]])
    print(f"\nArtifacts saved under {args.output}")


if __name__ == "__main__":
    main()
