"""Verify that the reduced T2-PD margin equals direct tree enumeration."""

from __future__ import annotations

import argparse
from itertools import product

from find_t2_dse_failure_unteachable_case import allowed_attacker_policies, induced_b
from t2_policy_dependent_case_search import (
    TOL,
    TREES,
    Tree,
    evaluate_observed_t2_pd,
)
from two_period_joint_policy_experiments import (
    NUM_ACTIONS,
    build_action_dependent_factored_pomdp,
)


PARAMETERS = (0.55, 0.70, 0.85, 0.95)


def reduced_margin(values: dict[Tree, float], target: Tree) -> float:
    """Partition competitors into same-root and alternative-root classes."""
    target_value = values[target]
    target_root = target[0]
    same_root = min(
        target_value - values[tree]
        for tree in TREES
        if tree != target and tree[0] == target_root
    )
    alternative_roots = min(
        target_value - max(values[tree] for tree in TREES if tree[0] == root)
        for root in range(NUM_ACTIONS)
        if root != target_root
    )
    return min(same_root, alternative_roots)


def direct_margin(values: dict[Tree, float], target: Tree) -> float:
    """Compare the target directly with every competing policy tree."""
    return min(
        values[target] - values[competitor]
        for competitor in TREES
        if competitor != target
    )


def run_check(max_seed: int, tolerance: float) -> tuple[int, float, bool]:
    checked = 0
    max_discrepancy = 0.0
    found = False

    for seed in range(max_seed):
        for control, obs_info in product(PARAMETERS, repeat=2):
            pomdp = build_action_dependent_factored_pomdp(
                seed,
                action_control=control,
                p_s1_matches_s2=obs_info,
            )
            for target_tree in TREES:
                for attacker in allowed_attacker_policies():
                    attacked_b = induced_b(attacker)
                    if attacked_b is None:
                        continue
                    checked += 1

                    evaluations = [
                        evaluate_observed_t2_pd(
                            pomdp, attacked_b, initial_s1, target_tree
                        )
                        for initial_s1 in range(2)
                    ]
                    reduced = min(
                        reduced_margin(result.values, target_tree)
                        for result in evaluations
                    )
                    direct = min(
                        direct_margin(result.values, target_tree)
                        for result in evaluations
                    )
                    discrepancy = abs(reduced - direct)
                    max_discrepancy = max(max_discrepancy, discrepancy)
                    found |= discrepancy > tolerance

    return checked, max_discrepancy, found


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the reduced policy-dependent margin with the direct "
            "strict policy-tree optimality margin."
        )
    )
    parser.add_argument("--max-seed", type=int, default=100)
    parser.add_argument("--tolerance", type=float, default=TOL)
    args = parser.parse_args()

    checked, max_discrepancy, found = run_check(args.max_seed, args.tolerance)

    print("=== Reduced T2-PD margin / direct enumeration check ===")
    print(f"seeds searched: {args.max_seed}")
    print(f"attacker-target combinations checked: {checked}")
    print(f"largest |reduced margin - direct margin|: {max_discrepancy:.3e}")
    print(
        "result: NUMERICAL DISCREPANCY ABOVE TOLERANCE FOUND"
        if found
        else "result: no discrepancy above tolerance found"
    )
    if not found:
        print(
            "reason: the reduced comparisons partition all non-target policy trees"
        )


if __name__ == "__main__":
    main()
