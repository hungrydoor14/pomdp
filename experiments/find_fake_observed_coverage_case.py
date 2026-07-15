from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from find_transition_mislearning_case import (
    Candidate,
    induced_observed_transition,
    prior_b_vector,
    search,
    sequence_values,
)


NUM_S1 = 2
NUM_S2 = 2
NUM_ACTIONS = 2


def join_state(s1: int, s2: int) -> int:
    return s1 * NUM_S2 + s2


def split_state(state: int) -> tuple[int, int]:
    return divmod(state, NUM_S2)


@dataclass(frozen=True)
class CoverageCase:
    seed: int
    state_counts: np.ndarray
    action_counts: np.ndarray


def sample_dataset(
    *,
    seed: int,
    num_samples: int,
    state_distribution: np.ndarray,
    action_distribution: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    states = rng.choice(
        NUM_S1 * NUM_S2,
        size=num_samples,
        p=state_distribution,
    )

    state_counts = np.zeros((NUM_S1, NUM_S2), dtype=int)
    action_counts = np.zeros((NUM_S1, NUM_S2, NUM_ACTIONS), dtype=int)

    for state in states:
        s1, s2 = split_state(int(state))
        action = rng.choice(
            NUM_ACTIONS,
            p=action_distribution[s1, s2],
        )
        state_counts[s1, s2] += 1
        action_counts[s1, s2, action] += 1

    return state_counts, action_counts


def has_observed_s1_coverage(state_counts: np.ndarray) -> bool:
    return bool(np.all(state_counts.sum(axis=1) > 0))


def has_hidden_state_coverage(state_counts: np.ndarray) -> bool:
    return bool(np.all(state_counts > 0))


def has_observed_action_coverage(action_counts: np.ndarray) -> bool:
    observed_action_counts = action_counts.sum(axis=1)
    return bool(np.all(observed_action_counts > 0))


def has_full_state_action_coverage(action_counts: np.ndarray) -> bool:
    return bool(np.all(action_counts > 0))


def find_case(
    *,
    num_samples: int = 1_000,
) -> CoverageCase:
    """
    Find a dataset where S1 looks fully covered, but S2 is not.

    The cleanest witness is not random: each observed state S1 appears many
    times, but only one hidden state S2 appears under each S1.
    """

    state_distribution = np.zeros(NUM_S1 * NUM_S2)
    state_distribution[join_state(0, 0)] = 0.50
    state_distribution[join_state(1, 1)] = 0.50

    action_distribution = np.full(
        (NUM_S1, NUM_S2, NUM_ACTIONS),
        0.50,
    )

    for seed in range(10_000):
        state_counts, action_counts = sample_dataset(
            seed=seed,
            num_samples=num_samples,
            state_distribution=state_distribution,
            action_distribution=action_distribution,
        )

        if (
            has_observed_s1_coverage(state_counts)
            and has_observed_action_coverage(action_counts)
            and not has_hidden_state_coverage(state_counts)
            and not has_full_state_action_coverage(action_counts)
        ):
            return CoverageCase(
                seed=seed,
                state_counts=state_counts,
                action_counts=action_counts,
            )

    raise RuntimeError("No fake observed-coverage case found.")


def print_observed_model_sections(candidate: Candidate) -> None:
    original_b = prior_b_vector()
    attacked_b = candidate.attacked_b

    print("# meta")
    print(f"seed {candidate.seed}")
    print(f"control {candidate.action_control:.2f}")
    print(f"obs_info {candidate.hidden_effect:.2f}")
    print("target a1 a1")
    print(f"margin {candidate.max_transition_shift:.3f}")
    print()

    print("# rewards")
    print("# state  R(state,a0)  R(state,a1)")
    for s1, s2 in product(range(NUM_S1), range(NUM_S2)):
        print(
            f"{s1}{s2} "
            f"{candidate.rewards[s1, s2, 0]: .3f} "
            f"{candidate.rewards[s1, s2, 1]: .3f}"
        )
    print()

    for prefix, b in (
        ("original", original_b),
        ("attacked", attacked_b),
    ):
        for initial_s1 in range(NUM_S1):
            print(f"# {prefix}_values_s1_{initial_s1}")
            if initial_s1 == 0:
                print("# sequence  value")
            for sequence, value in sequence_values(
                candidate.rewards,
                candidate.transitions,
                b,
                initial_s1,
            ):
                first_action, second_action = sequence
                print(
                    f"a{first_action} a{second_action} "
                    f"{value: .3f}"
                )
            print()

    for section, b in (
        ("original_b", original_b),
        ("attacked_b", attacked_b),
    ):
        print(f"# {section}")
        if section == "original_b":
            print("# s1  action  P(S2=1 | S1=s1,A=action)")
        for s1, action in product(range(NUM_S1), range(NUM_ACTIONS)):
            print(f"{s1} a{action} {b[s1 * NUM_ACTIONS + action]:.3f}")
        print()

    for section, b in (
        ("original_transitions", original_b),
        ("attacked_transitions", attacked_b),
    ):
        print(f"# {section}")
        print("# action  P(next_S1=0)  P(next_S1=1)")
        for action in range(NUM_ACTIONS):
            transition = induced_observed_transition(
                candidate.transitions,
                b,
                0,
                action,
            )
            print(f"a{action} {transition[0]:.3f} {transition[1]:.3f}")
        print()


def print_case_file(
    candidate: Candidate,
    case: CoverageCase,
) -> None:
    print_observed_model_sections(candidate)
    print()

    print("# hidden_state_counts")
    print("# state  count")
    for s1, s2 in product(range(NUM_S1), range(NUM_S2)):
        print(f"{s1}{s2} {case.state_counts[s1, s2]}")
    print()

    print("# observed_state_counts")
    print("# s1  count")
    for s1 in range(NUM_S1):
        print(f"{s1} {case.state_counts[s1].sum()}")
    print()

    print("# observed_state_action_counts")
    print("# s1  action  count")
    observed_action_counts = case.action_counts.sum(axis=1)
    for s1, action in product(range(NUM_S1), range(NUM_ACTIONS)):
        print(f"{s1} a{action} {observed_action_counts[s1, action]}")
    print()

    print("# full_state_action_counts")
    print("# state  action  count")
    for s1, s2, action in product(
        range(NUM_S1),
        range(NUM_S2),
        range(NUM_ACTIONS),
    ):
        print(f"{s1}{s2} a{action} {case.action_counts[s1, s2, action]}")
    print()

    print("# coverage")
    print("# quantity  covered")
    print(f"observed_state {int(has_observed_s1_coverage(case.state_counts))}")
    print(f"observed_state_action {int(has_observed_action_coverage(case.action_counts))}")
    print(f"hidden_state {int(has_hidden_state_coverage(case.state_counts))}")
    print(f"full_state_action {int(has_full_state_action_coverage(case.action_counts))}")


def main() -> None:
    print_case_file(
        candidate=search(),
        case=find_case(),
    )


if __name__ == "__main__":
    main()
