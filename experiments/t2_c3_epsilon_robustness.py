"""Evaluate the relaxed Case 3 attacker across epsilon in [0, 1]."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import matplotlib
import numpy as np
from scipy.optimize import brentq

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))

from find_t2_dse_failure_unteachable_case import induced_b
from t2_policy_dependent_case_search import Tree, TREES, observed_tree_value
from two_period_joint_policy_experiments import (
    NUM_ACTIONS,
    NUM_S1,
    NUM_S2,
    build_action_dependent_factored_pomdp,
)


CASE_SEED = 116
CASE_ACTION_CONTROL = 0.85
CASE_OBSERVATION_INFORMATION = 0.95
CASE_TARGET: Tree = (1, (0, 1))
BASE_ACTION1 = np.array([[0.0, 1.0], [1.0, 0.0]])
BINDING_TOL = 1e-9


@dataclass(frozen=True)
class Evaluation:
    epsilon: float
    margin: float
    binding_s1: int
    binding_tree: Tree


def relaxed_attacker(epsilon: float) -> np.ndarray:
    action1 = (1.0 - epsilon) * BASE_ACTION1 + epsilon / NUM_ACTIONS
    attacker = np.empty((NUM_S1, NUM_S2, NUM_ACTIONS), dtype=float)
    attacker[:, :, 1] = action1
    attacker[:, :, 0] = 1.0 - action1
    return attacker


def evaluate(pomdp, epsilon: float) -> Evaluation:
    mixtures = induced_b(relaxed_attacker(epsilon))
    if mixtures is None:
        raise RuntimeError(f"epsilon={epsilon} produced missing observed coverage")
    comparisons = []
    for initial_s1 in range(NUM_S1):
        target_value = observed_tree_value(
            pomdp,
            CASE_TARGET,
            initial_s1,
            mixtures,
        )
        comparisons.extend(
            (
                target_value
                - observed_tree_value(pomdp, tree, initial_s1, mixtures),
                initial_s1,
                tree,
            )
            for tree in TREES
            if tree != CASE_TARGET
        )
    margin = min(comparison[0] for comparison in comparisons)
    # Symmetric constraints can differ at machine precision. Treat near-ties
    # as the same binding set and choose a stable representative for reporting.
    binding_candidates = [
        comparison
        for comparison in comparisons
        if comparison[0] <= margin + BINDING_TOL
    ]
    _, binding_s1, binding_tree = min(
        binding_candidates,
        key=lambda comparison: (comparison[1], comparison[2]),
    )
    return Evaluation(float(epsilon), float(margin), binding_s1, binding_tree)


def tree_label(tree: Tree) -> str:
    return f"(a{tree[0]},a{tree[1][0]},a{tree[1][1]})"


def binding_segments(evaluations: list[Evaluation]):
    segments = []
    start = evaluations[0].epsilon
    current = (evaluations[0].binding_s1, evaluations[0].binding_tree)
    for previous, evaluation in zip(evaluations, evaluations[1:]):
        binding = (evaluation.binding_s1, evaluation.binding_tree)
        if binding != current:
            segments.append((start, previous.epsilon, *current))
            start = evaluation.epsilon
            current = binding
    segments.append((start, evaluations[-1].epsilon, *current))
    return segments


def first_nonpositive_threshold(pomdp, evaluations: list[Evaluation]) -> float:
    if evaluations[0].margin <= 0.0:
        return 0.0
    for left, right in zip(evaluations, evaluations[1:]):
        if right.margin <= 0.0:
            if right.margin == 0.0:
                return right.epsilon
            return float(
                brentq(
                    lambda epsilon: evaluate(pomdp, epsilon).margin,
                    left.epsilon,
                    right.epsilon,
                    xtol=1e-13,
                )
            )
    return 1.0


def write_csv(path: Path, evaluations: list[Evaluation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epsilon", "margin", "binding_s1", "binding_tree"])
        for evaluation in evaluations:
            writer.writerow(
                [
                    f"{evaluation.epsilon:.9f}",
                    f"{evaluation.margin:.12f}",
                    evaluation.binding_s1,
                    tree_label(evaluation.binding_tree),
                ]
            )


def write_plot(
    path: Path,
    evaluations: list[Evaluation],
    threshold: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    epsilons = [evaluation.epsilon for evaluation in evaluations]
    margins = [evaluation.margin for evaluation in evaluations]
    fig, axis = plt.subplots(figsize=(6.4, 3.8))
    axis.plot(epsilons, margins, color="#225ea8", linewidth=2)
    axis.axhline(0.0, color="#333333", linewidth=1, linestyle="--")
    axis.axvline(
        threshold,
        color="#d7301f",
        linewidth=1.2,
        linestyle=":",
        label=rf"positivity threshold $\varepsilon={threshold:.4f}$",
    )
    axis.set_xlabel(r"relaxation $\varepsilon$")
    axis.set_ylabel(r"$\Delta_{\mathrm{PD}}(\pi^\dagger_\varepsilon)$")
    axis.set_title(r"Case 3 $\varepsilon$-robustness")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def summary_text(
    evaluations: list[Evaluation],
    threshold: float,
) -> str:
    lines = [
        "CASE3_EPSILON_ROBUSTNESS",
        f"seed {CASE_SEED}",
        f"action_control {CASE_ACTION_CONTROL:.2f}",
        f"observation_information {CASE_OBSERVATION_INFORMATION:.2f}",
        f"target_tree {tree_label(CASE_TARGET)}",
        f"epsilon_zero_margin {evaluations[0].margin:+.12f}",
        f"epsilon_one_margin {evaluations[-1].margin:+.12f}",
        f"positive_connected_threshold {threshold:.12f}",
        f"binding_changes {len(binding_segments(evaluations)) - 1}",
        "# epsilon_start epsilon_end binding_s1 binding_tree",
    ]
    lines.extend(
        f"{start:.9f} {end:.9f} {s1} {tree_label(tree)}"
        for start, end, s1, tree in binding_segments(evaluations)
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=1001)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("outputs/t2/c3-epsilon-robustness"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.points < 2:
        raise SystemExit("--points must be at least 2")
    pomdp = build_action_dependent_factored_pomdp(
        CASE_SEED,
        action_control=CASE_ACTION_CONTROL,
        p_s1_matches_s2=CASE_OBSERVATION_INFORMATION,
    )
    evaluations = [
        evaluate(pomdp, epsilon)
        for epsilon in np.linspace(0.0, 1.0, args.points)
    ]
    threshold = first_nonpositive_threshold(pomdp, evaluations)
    summary = summary_text(evaluations, threshold)
    print(summary, end="")

    prefix = args.output_prefix
    summary_path = prefix.with_suffix(".txt")
    csv_path = prefix.with_suffix(".csv")
    plot_path = prefix.with_suffix(".png")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary, encoding="utf-8")
    write_csv(csv_path, evaluations)
    write_plot(plot_path, evaluations, threshold)
    print(f"summary_output {summary_path}")
    print(f"curve_output {csv_path}")
    print(f"plot_output {plot_path}")


if __name__ == "__main__":
    main()
