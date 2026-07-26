"""Compare restricted-grid and continuous T2-PD attacker searches.

The continuous result is the best numerical witness found by differential
evolution; it is a lower bound on the unrestricted supremum, not a certificate
of global optimality.  A sign separation is nevertheless conclusive in the
useful direction: a positive continuous witness proves that the restricted
three-row enumeration missed a teaching attack.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.optimize import differential_evolution, minimize

sys.path.append(os.path.dirname(__file__))

from find_t2_dse_failure_unteachable_case import (
    allowed_attacker_policies,
    induced_b,
)
from t2_policy_dependent_case_search import TREES, evaluate_observed_t2_pd
from two_period_joint_policy_experiments import (
    NUM_ACTIONS,
    NUM_S1,
    NUM_S2,
    build_action_dependent_factored_pomdp,
)


DEFAULT_PARAMETERS = (0.55, 0.70, 0.85, 0.95)


@dataclass(frozen=True)
class SearchResult:
    margin: float
    attacker: np.ndarray
    induced_mixtures: np.ndarray


def attacker_from_action1_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Return pi_dagger(a | s1,s2) from four probabilities of action a1."""
    action1 = np.asarray(probabilities, dtype=float).reshape(NUM_S1, NUM_S2)
    attacker = np.empty((NUM_S1, NUM_S2, NUM_ACTIONS), dtype=float)
    attacker[:, :, 1] = action1
    attacker[:, :, 0] = 1.0 - action1
    return attacker


def pd_margin(pomdp, target_tree, attacker: np.ndarray) -> tuple[float, np.ndarray] | None:
    mixtures = induced_b(attacker)
    if mixtures is None:
        return None
    margin = min(
        evaluate_observed_t2_pd(pomdp, mixtures, s1, target_tree).margin
        for s1 in range(NUM_S1)
    )
    return float(margin), mixtures


def strongest_grid_attack(pomdp, target_tree) -> SearchResult:
    best: SearchResult | None = None
    for attacker in allowed_attacker_policies():
        evaluated = pd_margin(pomdp, target_tree, attacker)
        if evaluated is None:
            continue
        margin, mixtures = evaluated
        if best is None or margin > best.margin:
            best = SearchResult(margin, attacker.copy(), mixtures.copy())
    if best is None:
        raise RuntimeError("the restricted grid contains no covered attacker")
    return best


def strongest_continuous_witness(
    pomdp,
    target_tree,
    *,
    coverage_floor: float,
    maxiter: int,
    popsize: int,
    rng_seed: int,
) -> SearchResult:
    bounds = [(coverage_floor, 1.0 - coverage_floor)] * (NUM_S1 * NUM_S2)

    def objective(action1_probabilities: np.ndarray) -> float:
        attacker = attacker_from_action1_probabilities(action1_probabilities)
        evaluated = pd_margin(pomdp, target_tree, attacker)
        if evaluated is None:
            return 1e6
        return -evaluated[0]

    global_result = differential_evolution(
        objective,
        bounds,
        seed=rng_seed,
        maxiter=maxiter,
        popsize=popsize,
        polish=False,
        updating="immediate",
        workers=1,
    )
    local_result = minimize(
        objective,
        global_result.x,
        method="Nelder-Mead",
        options={"maxiter": 2000, "xatol": 1e-10, "fatol": 1e-10},
    )
    probabilities = np.clip(
        local_result.x if local_result.fun <= global_result.fun else global_result.x,
        coverage_floor,
        1.0 - coverage_floor,
    )
    attacker = attacker_from_action1_probabilities(probabilities)
    evaluated = pd_margin(pomdp, target_tree, attacker)
    if evaluated is None:
        raise RuntimeError("continuous optimizer returned an uncovered attacker")
    margin, mixtures = evaluated
    return SearchResult(margin, attacker, mixtures.copy())


def tree_label(tree) -> str:
    return f"(a{tree[0]},a{tree[1][0]},a{tree[1][1]})"


def print_attacker(label: str, result: SearchResult) -> None:
    print(f"{label}_margin {result.margin:+.9f}")
    print(f"{label}_attacker_pi_a1")
    for s1, s2 in product(range(NUM_S1), range(NUM_S2)):
        print(f"  {s1}{s2} {result.attacker[s1, s2, 1]:.9f}")
    print(f"{label}_mixtures_P_s2_1")
    for s1, action in product(range(NUM_S1), range(NUM_ACTIONS)):
        index = s1 * NUM_ACTIONS + action
        print(f"  s1={s1} a{action} {result.induced_mixtures[index]:.9f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search for a target with negative restricted-grid T2-PD margin "
            "and a positive continuous-attacker witness."
        )
    )
    parser.add_argument("--max-seed", type=int, default=25)
    parser.add_argument("--maxiter", type=int, default=120)
    parser.add_argument("--popsize", type=int, default=12)
    parser.add_argument("--coverage-floor", type=float, default=1e-6)
    parser.add_argument("--separation-tol", type=float, default=1e-5)
    parser.add_argument("--rng-seed", type=int, default=20260725)
    parser.add_argument(
        "--stop-first",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_seed <= 0:
        raise SystemExit("--max-seed must be positive")
    if not 0.0 < args.coverage_floor < 0.5:
        raise SystemExit("--coverage-floor must lie strictly between 0 and 0.5")

    tested = 0
    optimized = 0
    best_gap = float("-inf")
    best_record = None

    for seed in range(args.max_seed):
        for control, obs_info in product(DEFAULT_PARAMETERS, repeat=2):
            pomdp = build_action_dependent_factored_pomdp(
                seed,
                action_control=control,
                p_s1_matches_s2=obs_info,
            )
            for target_tree in TREES:
                tested += 1
                grid = strongest_grid_attack(pomdp, target_tree)
                # We only need continuous optimization where the grid has not
                # already supplied a strict teaching witness.
                if grid.margin > args.separation_tol:
                    continue
                optimized += 1
                continuous = strongest_continuous_witness(
                    pomdp,
                    target_tree,
                    coverage_floor=args.coverage_floor,
                    maxiter=args.maxiter,
                    popsize=args.popsize,
                    rng_seed=args.rng_seed + tested,
                )
                gap = continuous.margin - grid.margin
                if gap > best_gap:
                    best_gap = gap
                    best_record = (seed, control, obs_info, target_tree, grid, continuous)

                if (
                    grid.margin < -args.separation_tol
                    and continuous.margin > args.separation_tol
                ):
                    print("SIGN_SEPARATION_FOUND")
                    print(f"seed {seed}")
                    print(f"action_control {control:.2f}")
                    print(f"observation_information {obs_info:.2f}")
                    print(f"target_tree {tree_label(target_tree)}")
                    print_attacker("grid", grid)
                    print_attacker("continuous", continuous)
                    print(f"margin_gap {gap:+.9f}")
                    print(f"models_targets_tested {tested}")
                    print(f"continuous_optimizations {optimized}")
                    if args.stop_first:
                        return

    print("NO_SIGN_SEPARATION_FOUND")
    print(f"models_targets_tested {tested}")
    print(f"continuous_optimizations {optimized}")
    if best_record is not None:
        seed, control, obs_info, target_tree, grid, continuous = best_record
        print("BEST_MARGIN_IMPROVEMENT")
        print(f"seed {seed}")
        print(f"action_control {control:.2f}")
        print(f"observation_information {obs_info:.2f}")
        print(f"target_tree {tree_label(target_tree)}")
        print_attacker("grid", grid)
        print_attacker("continuous", continuous)
        print(f"margin_gap {best_gap:+.9f}")


if __name__ == "__main__":
    main()
