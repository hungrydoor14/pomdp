from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from itertools import product

import numpy as np

sys.path.append(os.path.dirname(__file__))

from find_t2_dse_inducible_observed_model_case import (
    INITIAL_MATCH_PROB,
    b_index,
    observed_transition,
    observed_value,
    prior_b_vector,
)
from two_period_joint_policy_experiments import (
    NUM_ACTIONS,
    NUM_S1,
    NUM_S2,
    build_action_dependent_factored_pomdp,
    join_state,
)


TOL = 1e-10
DEFAULT_CONTROLS = (0.55, 0.70, 0.85, 0.95)
DEFAULT_OBS_INFOS = (0.55, 0.70, 0.85, 0.95)

# Every row is pi_dagger(a0,a1 | s1,s2).  No other mixtures are searched.
ALLOWED_POLICY_ROWS = (
    (1.0, 0.0),
    (0.0, 1.0),
    (0.5, 0.5),
)


@dataclass(frozen=True)
class Case4:
    seed: int
    control: float
    obs_info: float
    pomdp: object
    target: tuple[int, int]
    t2_dse_margins: tuple[float, float]
    best_margin: float
    attacker: np.ndarray
    attacked_b: np.ndarray


def allowed_attacker_policies():
    """Yield all 3^4 full-state policies allowed by the Case 4 search."""
    for rows in product(ALLOWED_POLICY_ROWS, repeat=NUM_S1 * NUM_S2):
        yield np.asarray(rows, dtype=float).reshape(NUM_S1, NUM_S2, NUM_ACTIONS)


def prior_hidden_probability(s1: int) -> float:
    return INITIAL_MATCH_PROB if s1 == 1 else 1.0 - INITIAL_MATCH_PROB


def induced_b(attacker: np.ndarray) -> np.ndarray | None:
    """Compute P(S2=1 | S1=s1,A=a); reject missing observed action coverage."""
    b = np.empty(NUM_S1 * NUM_ACTIONS)
    for s1, action in product(range(NUM_S1), range(NUM_ACTIONS)):
        q = prior_hidden_probability(s1)
        hidden_probs = np.array([1.0 - q, q])
        action_mass = float(hidden_probs @ attacker[s1, :, action])
        if action_mass <= TOL:
            return None
        b[b_index(s1, action)] = (
            hidden_probs[1] * attacker[s1, 1, action] / action_mass
        )
    return b


def t2_dse_margins(pomdp, target_action: int) -> tuple[float, float]:
    """
    Case 1's range condition, evaluated at both observed states:
    max_s2 R(s1,s2,target) - min_s2 R(s1,s2,alternative) > 0.
    """
    alternative = 1 - target_action
    margins = []
    for s1 in range(NUM_S1):
        target_rewards = [
            pomdp.rewards[join_state(s1, s2), target_action]
            for s2 in range(NUM_S2)
        ]
        alternative_rewards = [
            pomdp.rewards[join_state(s1, s2), alternative]
            for s2 in range(NUM_S2)
        ]
        margins.append(max(target_rewards) - min(alternative_rewards))
    return tuple(float(value) for value in margins)


def target_margin(pomdp, target: tuple[int, int], b: np.ndarray) -> float:
    """Smallest target advantage over every other sequence and initial S1."""
    margins = []
    for initial_s1 in range(NUM_S1):
        target_value = observed_value(pomdp, target, initial_s1, b)
        for alternative in product(range(NUM_ACTIONS), repeat=2):
            if alternative == target:
                continue
            margins.append(
                target_value - observed_value(pomdp, alternative, initial_s1, b)
            )
    return float(min(margins))


def strongest_restricted_attack(pomdp, target: tuple[int, int]):
    best = None
    for attacker in allowed_attacker_policies():
        b = induced_b(attacker)
        if b is None:
            continue
        margin = target_margin(pomdp, target, b)
        if best is None or margin > best[0]:
            best = (margin, attacker.copy(), b)
    if best is None:
        raise RuntimeError("no allowed attacker has observed state-action coverage")
    return best


def find_case(max_seed: int) -> Case4 | None:
    # A stationary target (a,a) is the direct converse of the Case 1 target:
    # it requests the same target action at every point in the two-period plan.
    for seed in range(max_seed):
        for control, obs_info in product(DEFAULT_CONTROLS, DEFAULT_OBS_INFOS):
            pomdp = build_action_dependent_factored_pomdp(
                seed,
                action_control=control,
                p_s1_matches_s2=obs_info,
            )
            for target_action in range(NUM_ACTIONS):
                target = (target_action, target_action)
                range_margins = t2_dse_margins(pomdp, target_action)
                if min(range_margins) > TOL:
                    continue
                best_margin, attacker, attacked_b = strongest_restricted_attack(
                    pomdp, target
                )
                if best_margin <= TOL:
                    return Case4(
                        seed=seed,
                        control=control,
                        obs_info=obs_info,
                        pomdp=pomdp,
                        target=target,
                        t2_dse_margins=range_margins,
                        best_margin=best_margin,
                        attacker=attacker,
                        attacked_b=attacked_b,
                    )
    return None


def print_values(pomdp, section: str, b: np.ndarray) -> None:
    for initial_s1 in range(NUM_S1):
        print(f"# {section}_values_s1_{initial_s1}")
        if initial_s1 == 0:
            print("# sequence  value")
        for sequence in product(range(NUM_ACTIONS), repeat=2):
            value = observed_value(pomdp, sequence, initial_s1, b)
            print(f"a{sequence[0]} a{sequence[1]} {value: .6f}")
        print()


def print_b(section: str, b: np.ndarray) -> None:
    print(f"# {section}")
    if section == "original_b":
        print("# s1  action  P(S2=1 | S1=s1,A=action)")
    for s1, action in product(range(NUM_S1), range(NUM_ACTIONS)):
        print(f"{s1} a{action} {b[b_index(s1, action)]:.6f}")
    print()


def print_transitions(case: Case4, section: str) -> None:
    print(f"# {section}")
    print("# action  P(next_S1=0)  P(next_S1=1)")
    for action in range(NUM_ACTIONS):
        transition = observed_transition(case.pomdp, action)
        print(f"a{action} {transition[0]:.6f} {transition[1]:.6f}")
    print()


def print_transitions_by_s1(case: Case4, section: str) -> None:
    print(f"# {section}")
    print("# s1  action  P(next_S1=0)  P(next_S1=1)")
    for s1, action in product(range(NUM_S1), range(NUM_ACTIONS)):
        # Case 1's transition is independent of current S1 and hidden mixtures.
        transition = observed_transition(case.pomdp, action)
        print(f"{s1} a{action} {transition[0]:.6f} {transition[1]:.6f}")
    print()


def print_case(case: Case4) -> None:
    original_b = prior_b_vector()

    print("# meta")
    print(f"seed {case.seed}")
    print(f"control {case.control:.2f}")
    print(f"obs_info {case.obs_info:.2f}")
    print(f"target a{case.target[0]} a{case.target[1]}")
    print(f"margin {case.best_margin:.6f}")
    print("t2_dse_passes 0")
    for s1, margin in enumerate(case.t2_dse_margins):
        print(f"t2_dse_margin_s1_{s1} {margin:.6f}")
    print("target_teachable 0")
    print(f"policies_checked {len(ALLOWED_POLICY_ROWS) ** (NUM_S1 * NUM_S2)}")
    print()

    print("# rewards")
    print("# state  R(state,a0)  R(state,a1)")
    for s1, s2 in product(range(NUM_S1), range(NUM_S2)):
        state = join_state(s1, s2)
        print(
            f"{s1}{s2} {case.pomdp.rewards[state, 0]: .6f} "
            f"{case.pomdp.rewards[state, 1]: .6f}"
        )
    print()

    print_values(case.pomdp, "original", original_b)
    print_values(case.pomdp, "attacked", case.attacked_b)
    print_b("original_b", original_b)
    print_b("attacked_b", case.attacked_b)
    print_transitions(case, "original_transitions")
    print_transitions(case, "attacked_transitions")
    print_transitions_by_s1(case, "original_transitions_by_s1")
    print_transitions_by_s1(case, "attacked_transitions_by_s1")

    print("# attacker_policy")
    print("# state  pi_dagger(a1 | S1,S2)")
    for s1, s2 in product(range(NUM_S1), range(NUM_S2)):
        print(f"{s1}{s2} {case.attacker[s1, s2, 1]:.1f}")
    print()

    print("# coverage")
    print("# quantity  covered")
    print("observed_state 1")
    print("observed_state_action 1")
    print("hidden_state 1")
    print("full_state_action 0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find the simple two-period converse of T2-DSE Case 1."
    )
    parser.add_argument("--max-seed", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_seed <= 0:
        raise SystemExit("--max-seed must be positive")
    case = find_case(args.max_seed)
    if case is None:
        raise SystemExit(f"no Case 4 found below seed {args.max_seed}")
    print_case(case)


if __name__ == "__main__":
    main()
