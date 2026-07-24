from __future__ import annotations

from itertools import product

import numpy as np

from find_t2_dse_transition_mislearning_case import (
    best_sequences,
    induced_observed_transition,
    prior_b_vector,
    sequence_values,
)


NUM_S1 = 2
NUM_S2 = 2
NUM_ACTIONS = 2
NUM_SAMPLES = 1_000
REWARD_SEED = 5
TARGET = (1, 1)
DISCOUNT = 0.95

# Full-support Period 1 distribution: P(S1)=1/2 and P(S2=S1 | S1)=0.7.
PERIOD1_PROBS = np.array([[0.35, 0.15], [0.15, 0.35]])

# pi_dagger(a1 | S1,S2).  Every row is deterministic; both observed actions
# remain covered at each S1 because the behavior depends on hidden S2.
ATTACKER_A1 = np.array([[0.0, 1.0], [1.0, 0.0]])
ATTACKED_B = np.array([0.0, 1.0, 1.0, 0.0])


def make_action_suppression_transitions() -> np.ndarray:
    """Return T[a,s1,s2,next_s1,next_s2] for the Case 2.2 construction."""
    transitions = np.zeros((2, 2, 2, 2, 2))
    for s1, s2 in product(range(2), repeat=2):
        # a0 preserves the complete state.  a1 maps to the diagonal state
        # indexed by S2.  Under pi_dagger, both actions therefore reach only
        # 00 or 11, even though all four states have Period 1 support.
        transitions[0, s1, s2, s1, s2] = 1.0
        transitions[1, s1, s2, s2, s2] = 1.0
    return transitions


def deterministic_counts():
    period1 = np.rint(NUM_SAMPLES * PERIOD1_PROBS).astype(int)
    actions = np.zeros((2, 2, 2), dtype=int)
    period2 = np.zeros((2, 2), dtype=int)
    transitions = make_action_suppression_transitions()

    for s1, s2 in product(range(2), repeat=2):
        action = int(ATTACKER_A1[s1, s2])
        count = int(period1[s1, s2])
        actions[s1, s2, action] = count
        next_state = np.argwhere(transitions[action, s1, s2] == 1.0)[0]
        period2[tuple(next_state)] += count
    return period1, period2, actions


def rewards() -> np.ndarray:
    return np.random.default_rng(REWARD_SEED).normal(size=(2, 2, 2))


def target_margin(reward: np.ndarray, transitions: np.ndarray) -> float:
    margins = []
    for s1 in range(2):
        values = dict(sequence_values(reward, transitions, ATTACKED_B, s1))
        margins.append(values[TARGET] - max(v for q, v in values.items() if q != TARGET))
    return float(min(margins))


def print_values(reward, transitions, prefix, b):
    for s1 in range(2):
        print(f"# {prefix}_values_s1_{s1}")
        if s1 == 0:
            print("# sequence  value")
        for sequence, value in sequence_values(reward, transitions, b, s1):
            print(f"a{sequence[0]} a{sequence[1]} {value: .6f}")
        print()


def print_case_file() -> None:
    transition = make_action_suppression_transitions()
    reward = rewards()
    original_b = prior_b_vector()
    period1, period2, action_counts = deterministic_counts()

    assert np.all(period1 > 0)
    assert np.all(action_counts.sum(axis=1) > 0)
    assert period2[0, 1] == period2[1, 0] == 0
    assert all(TARGET in best_sequences(sequence_values(reward, transition, ATTACKED_B, s)) for s in range(2))

    print("# meta")
    print(f"seed {REWARD_SEED}")
    print(f"discount {DISCOUNT:.2f}")
    print("control 1.00")
    print("obs_info 0.70")
    print("target a1 a1")
    print(f"margin {target_margin(reward, transition):.6f}")
    print()

    print("# rewards")
    print("# state  R(state,a0)  R(state,a1)")
    for s1, s2 in product(range(2), repeat=2):
        print(f"{s1}{s2} {reward[s1,s2,0]: .6f} {reward[s1,s2,1]: .6f}")
    print()

    print_values(reward, transition, "original", original_b)
    print_values(reward, transition, "attacked", ATTACKED_B)

    for section, b in (("original_b", original_b), ("attacked_b", ATTACKED_B)):
        print(f"# {section}")
        if section == "original_b":
            print("# s1  action  P(S2=1 | S1=s1,A=action)")
        for s1, action in product(range(2), repeat=2):
            print(f"{s1} a{action} {b[2*s1+action]:.6f}")
        print()

    for section, b in (("original_transitions", original_b), ("attacked_transitions", ATTACKED_B)):
        print(f"# {section}")
        print("# action  P(next_S1=0)  P(next_S1=1)")
        for action in range(2):
            row = induced_observed_transition(transition, b, 0, action)
            print(f"a{action} {row[0]:.6f} {row[1]:.6f}")
        print()

    for section, b in (("original_transitions_by_s1", original_b), ("attacked_transitions_by_s1", ATTACKED_B)):
        print(f"# {section}")
        print("# s1  action  P(next_S1=0)  P(next_S1=1)")
        for s1, action in product(range(2), repeat=2):
            row = induced_observed_transition(transition, b, s1, action)
            print(f"{s1} a{action} {row[0]:.6f} {row[1]:.6f}")
        print()

    print("# attacker_policy")
    print("# state  pi_dagger(a1 | S1,S2)")
    for s1, s2 in product(range(2), repeat=2):
        print(f"{s1}{s2} {ATTACKER_A1[s1,s2]:.1f}")
    print()

    for section, counts in (("period1_state_counts", period1), ("period2_state_counts", period2), ("hidden_state_counts", period2)):
        print(f"# {section}")
        print("# state  count")
        for s1, s2 in product(range(2), repeat=2):
            print(f"{s1}{s2} {counts[s1,s2]}")
        print()

    print("# observed_state_counts")
    print("# s1  count")
    for s1 in range(2):
        print(f"{s1} {period1[s1].sum()}")
    print()

    print("# observed_state_action_counts")
    print("# s1  action  count")
    for s1, action in product(range(2), repeat=2):
        print(f"{s1} a{action} {action_counts[s1,:,action].sum()}")
    print()

    print("# full_state_action_counts")
    print("# state  action  count")
    for s1, s2, action in product(range(2), repeat=3):
        print(f"{s1}{s2} a{action} {action_counts[s1,s2,action]}")
    print()

    print("# coverage")
    print("# quantity  covered")
    print("observed_state 1")
    print("observed_state_action 1")
    print("period1_hidden_state 1")
    print("period2_hidden_state 0")
    print("full_state_action 0")


if __name__ == "__main__":
    print_case_file()
