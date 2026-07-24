"""Check whether T2-PD and direct strict teachability ever disagree."""

from __future__ import annotations

import argparse
from itertools import product

from find_t2_dse_failure_unteachable_case import allowed_attacker_policies, induced_b
from t2_policy_dependent_case_search import (
    TOL,
    TREES,
    Tree,
    evaluate_observed_t2_pd,
    pointwise_t2_dse_witness,
)
from two_period_joint_policy_experiments import build_action_dependent_factored_pomdp


PARAMETERS = (0.55, 0.70, 0.85, 0.95)


def direct_margin(values: dict[Tree, float], target: Tree) -> float:
    return min(
        values[target] - values[competitor]
        for competitor in TREES
        if competitor != target
    )


def run_search(max_seed: int, tolerance: float) -> tuple[int, float, bool]:
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
                dse_margin = pointwise_t2_dse_witness(pomdp, target_tree)[0]
                if dse_margin > tolerance:
                    continue

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
                    pd = min(result.margin for result in evaluations)
                    direct = min(
                        direct_margin(result.values, target_tree)
                        for result in evaluations
                    )
                    max_discrepancy = max(max_discrepancy, abs(pd - direct))
                    found |= pd <= tolerance and direct > tolerance

    return checked, max_discrepancy, found


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Look for a case where T2-DSE and T2-PD fail but direct policy-tree "
            "enumeration says the target is strictly teachable."
        )
    )
    parser.add_argument("--max-seed", type=int, default=100)
    parser.add_argument("--tolerance", type=float, default=TOL)
    args = parser.parse_args()

    checked, max_discrepancy, found = run_search(args.max_seed, args.tolerance)

    print("=== T2-DSE fails / T2-PD fails / teachable search ===")
    print(f"seeds searched: {args.max_seed}")
    print(f"attacker-target combinations checked: {checked}")
    print(f"largest |PD margin - direct margin|: {max_discrepancy:.3e}")
    print(
        "result: CONTRADICTORY WITNESS FOUND"
        if found
        else "result: no witness found (expected under the current T2-PD definition)"
    )
    if not found:
        print(
            "reason: the T2-PD comparisons partition all non-target policy trees, "
            "so its margin equals the direct strict-teachability margin"
        )


if __name__ == "__main__":
    main()
