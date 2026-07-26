"""Exactly verify the unrestricted T2-PD margin for Case 4.

For binary actions and hidden states, Bayes plausibility is the union of four
linear regions: at each observed state, the fixed hidden-state prior lies
between the two action-conditioned posteriors in one of two orientations.
This script solves the max-margin LP in every region and reports their maximum.
"""

from __future__ import annotations

import os
import sys
from itertools import product

import numpy as np

sys.path.append(os.path.dirname(__file__))

from find_t2_dse_inducible_observed_model_case import (
    b_index,
    observed_transition,
    prior_hidden_prob,
    rhat_coefficients,
    solve_linear_feasibility,
)
from t2_policy_dependent_case_search import Tree, TREES
from two_period_joint_policy_experiments import (
    NUM_ACTIONS,
    NUM_S1,
    build_action_dependent_factored_pomdp,
)


CASE_SEED = 0
CASE_ACTION_CONTROL = 0.55
CASE_OBSERVATION_INFORMATION = 0.55
CASE_TARGET: Tree = (1, (1, 1))
EXPECTED_MARGIN = -0.02307575
EXPECTED_MIXTURES = np.array([0.0, 1.0, 0.0, 1.0])
VERIFY_TOL = 1e-8


def observed_tree_value_coefficients(
    pomdp,
    tree: Tree,
    initial_s1: int,
) -> tuple[float, np.ndarray]:
    """Return constant and mixture coefficients for a rooted tree's value."""
    root_action, continuation = tree
    constant, coefficients = rhat_coefficients(pomdp, initial_s1, root_action)
    transition = observed_transition(pomdp, root_action)
    for next_s1, probability in enumerate(transition):
        next_constant, next_coefficients = rhat_coefficients(
            pomdp,
            next_s1,
            continuation[next_s1],
        )
        constant += pomdp.discount * probability * next_constant
        coefficients += pomdp.discount * probability * next_coefficients
    return float(constant), coefficients


def policy_tree_margin_constraints(
    pomdp,
    target: Tree,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct A mu >= c constraints for every rooted-tree comparison."""
    rows = []
    thresholds = []
    for initial_s1 in range(NUM_S1):
        target_constant, target_coefficients = observed_tree_value_coefficients(
            pomdp,
            target,
            initial_s1,
        )
        for competitor in TREES:
            if competitor == target:
                continue
            competitor_constant, competitor_coefficients = (
                observed_tree_value_coefficients(pomdp, competitor, initial_s1)
            )
            rows.append(target_coefficients - competitor_coefficients)
            thresholds.append(competitor_constant - target_constant)
    return np.asarray(rows), np.asarray(thresholds)


def format_orientation(orientation: tuple[int, ...]) -> str:
    return "(" + ",".join(str(value) for value in orientation) + ")"


def closed_bayes_orientation_constraints(
    orientation: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Require the prior to lie weakly between the two action posteriors."""
    rows = []
    thresholds = []
    for s1, ordering in enumerate(orientation):
        prior = prior_hidden_prob(s1)
        lower_action, upper_action = (0, 1) if ordering == 0 else (1, 0)

        lower_row = np.zeros(NUM_S1 * NUM_ACTIONS)
        lower_row[b_index(s1, lower_action)] = -1.0
        rows.append(lower_row)
        thresholds.append(-prior)

        upper_row = np.zeros(NUM_S1 * NUM_ACTIONS)
        upper_row[b_index(s1, upper_action)] = 1.0
        rows.append(upper_row)
        thresholds.append(prior)
    return np.asarray(rows), np.asarray(thresholds)


def main() -> None:
    pomdp = build_action_dependent_factored_pomdp(
        CASE_SEED,
        action_control=CASE_ACTION_CONTROL,
        p_s1_matches_s2=CASE_OBSERVATION_INFORMATION,
    )
    constraints, thresholds = policy_tree_margin_constraints(pomdp, CASE_TARGET)

    results = []
    print("CASE4_UNRESTRICTED_T2_PD")
    print(f"seed {CASE_SEED}")
    print(f"action_control {CASE_ACTION_CONTROL:.2f}")
    print(f"observation_information {CASE_OBSERVATION_INFORMATION:.2f}")
    print("target_tree (a1,a1,a1)")
    print("# orientation  optimal_margin  mixtures")

    for orientation in product((0, 1), repeat=NUM_S1):
        orientation_constraints, orientation_thresholds = (
            closed_bayes_orientation_constraints(orientation)
        )
        result = solve_linear_feasibility(
            constraints,
            thresholds,
            orientation_constraints,
            orientation_thresholds,
        )
        if result.witness_b is None:
            raise RuntimeError(f"orientation {orientation} LP was infeasible")
        results.append((result.margin, orientation, result.witness_b.copy()))
        mixtures = " ".join(f"{value:.8f}" for value in result.witness_b)
        print(
            f"{format_orientation(orientation):>5} "
            f"{result.margin:+.8f}  {mixtures}"
        )

    best_margin, best_orientation, best_mixtures = max(
        results,
        key=lambda item: item[0],
    )
    print(f"overall_optimal_margin {best_margin:+.8f}")
    print(f"maximizing_orientation {format_orientation(best_orientation)}")
    print(
        "maximizing_mixtures "
        + " ".join(f"{value:.8f}" for value in best_mixtures)
    )
    print(f"target_teachable {int(best_margin >= 0.0)}")

    if abs(best_margin - EXPECTED_MARGIN) > VERIFY_TOL:
        raise AssertionError(
            f"expected margin {EXPECTED_MARGIN:+.8f}, got {best_margin:+.8f}"
        )
    if not np.allclose(best_mixtures, EXPECTED_MIXTURES, atol=VERIFY_TOL):
        raise AssertionError(
            f"expected mixtures {EXPECTED_MIXTURES}, got {best_mixtures}"
        )
    print("verification_passed 1")


if __name__ == "__main__":
    main()
