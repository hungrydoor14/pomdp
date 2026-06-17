from __future__ import annotations

import os
import sys
from itertools import product

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from finite_horizon_solver import (
    TabularPOMDP,
    action_after_observation_history,
    best_alpha,
    greedy_action,
    solve_finite_horizon,
)


NUM_S1 = 2
NUM_S2 = 2
NUM_STATES = NUM_S1 * NUM_S2


def join_state(s1: int, s2: int) -> int:
    return s1 * NUM_S2 + s2


def split_state(state: int) -> tuple[int, int]:
    return divmod(state, NUM_S2)


def build_factored_demo() -> TabularPOMDP:
    # State is (S1, S2). The agent observes S1 perfectly, while S2 is hidden.
    # Observing S1(t + 1) is informative about S2(t + 1), so the posterior b2 moves.
    transitions = np.zeros((2, NUM_STATES, NUM_STATES))
    for action, state in product(range(2), range(NUM_STATES)):
        _, s2 = split_state(state)
        for next_s2 in range(NUM_S2):
            p_s2 = 0.9 if next_s2 == s2 else 0.1
            for next_s1 in range(NUM_S1):
                p_s1 = 0.8 if next_s1 == next_s2 else 0.2
                transitions[action, state, join_state(next_s1, next_s2)] = p_s1 * p_s2

    observations = np.zeros((2, NUM_STATES, NUM_S1))
    for action, state in product(range(2), range(NUM_STATES)):
        s1, _ = split_state(state)
        observations[action, state, s1] = 1.0

    rewards = np.zeros((NUM_STATES, 2))
    for state in range(NUM_STATES):
        _, s2 = split_state(state)
        rewards[state, 0] = 1.0 if s2 == 0 else -1.0
        rewards[state, 1] = 1.0 if s2 == 1 else -1.0

    return TabularPOMDP(
        transitions=transitions,
        observations=observations,
        rewards=rewards,
        discount=0.95,
        state_names=tuple(f"s1={s1},s2={s2}" for s1, s2 in product(range(NUM_S1), range(NUM_S2))),
        action_names=("a0", "a1"),
        observation_names=("s1=0", "s1=1"),
    )


def initial_belief_given_s1(observed_s1: int) -> np.ndarray:
    belief = np.zeros(NUM_STATES)
    # Replace this prior with your model's P(S2 | S1(t=1)).
    p_s2_matches_s1 = 0.7
    for s2 in range(NUM_S2):
        belief[join_state(observed_s1, s2)] = (
            p_s2_matches_s1 if s2 == observed_s1 else 1.0 - p_s2_matches_s1
        )
    return belief


def b2_from_joint_belief(belief: np.ndarray, observed_s1: int) -> np.ndarray:
    b2 = np.array([belief[join_state(observed_s1, s2)] for s2 in range(NUM_S2)])
    return b2 / b2.sum()


def main() -> None:
    pomdp = build_factored_demo()
    horizon = 2
    stages = solve_finite_horizon(pomdp, horizon=horizon)

    print("Converting pi(s1, b2) to pi*(s1(t=1), s1(t=2))")
    print(f"horizon: {horizon}")
    print()
    print("Rows below show reachable second-step beliefs:")
    print("s1_t1  s1_t2  b2=P(S2 | history)      pi(s1_t2,b2)  pi*(s1_t1,s1_t2)")

    induced_policy: dict[tuple[int, int], str] = {}
    for s1_t1 in range(NUM_S1):
        initial_belief = initial_belief_given_s1(s1_t1)
        root = best_alpha(stages[horizon], initial_belief)

        for s1_t2 in range(NUM_S1):
            posterior = pomdp.belief_update(initial_belief, root.action, s1_t2)
            b2 = b2_from_joint_belief(posterior, s1_t2)
            belief_policy_action = greedy_action(stages[horizon - 1], posterior)
            history_policy_action = action_after_observation_history(root, [s1_t2])

            belief_policy_name = pomdp.action_names[belief_policy_action]
            history_policy_name = pomdp.action_names[history_policy_action]
            induced_policy[(s1_t1, s1_t2)] = history_policy_name

            print(
                f"  {s1_t1:<5}  {s1_t2:<5}  "
                f"{np.round(b2, 3).tolist()!s:<22} "
                f"{belief_policy_name:<13} {history_policy_name}"
            )

    print()
    print("pi* table:")
    for s1_t1 in range(NUM_S1):
        entries = [
            f"pi*({s1_t1}, {s1_t2})={induced_policy[(s1_t1, s1_t2)]}"
            for s1_t2 in range(NUM_S1)
        ]
        print("  " + "  ".join(entries))


if __name__ == "__main__":
    main()
