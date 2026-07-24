from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np


NUM_S1 = 2
NUM_S2 = 2
NUM_ACTIONS = 2
INITIAL_MATCH_PROB = 0.70
DISCOUNT = 0.95
TOL = 1e-9


def b_index(s1: int, action: int) -> int:
    return s1 * NUM_ACTIONS + action


def prior_hidden_prob(s1: int) -> float:
    return INITIAL_MATCH_PROB if s1 == 1 else 1.0 - INITIAL_MATCH_PROB


def prior_b_vector() -> np.ndarray:
    return np.array(
        [
            prior_hidden_prob(s1)
            for s1 in range(NUM_S1)
            for _action in range(NUM_ACTIONS)
        ],
        dtype=float,
    )


def format_sequence(sequence: tuple[int, int]) -> str:
    return f"(a{sequence[0]},a{sequence[1]})"


def format_b(b: np.ndarray) -> str:
    return "  ".join(
        f"b({s1},a{action})={b[b_index(s1, action)]:.3f}"
        for s1 in range(NUM_S1)
        for action in range(NUM_ACTIONS)
    )


def make_hidden_dependent_transition(
    *,
    hidden_effect: float,
    action_effect: float,
    action_control: float,
) -> np.ndarray:
    """
    Return T[a, s1, s2, next_s1, next_s2].

    The key departure from the old T2-DSE case is that the next observed state
    depends on the current hidden state. This lets the attack change the
    learner's induced observed transition by changing P(S2 | S1, A).
    """

    transitions = np.zeros((NUM_ACTIONS, NUM_S1, NUM_S2, NUM_S1, NUM_S2))

    for action, s1, s2 in product(
        range(NUM_ACTIONS),
        range(NUM_S1),
        range(NUM_S2),
    ):
        p_next_s1_one = (1.0 - hidden_effect) if s2 == 0 else hidden_effect
        p_next_s1_one += action_effect if action == 1 else -action_effect
        p_next_s1_one = float(np.clip(p_next_s1_one, 0.02, 0.98))

        for next_s1 in range(NUM_S1):
            p_s1 = p_next_s1_one if next_s1 == 1 else 1.0 - p_next_s1_one

            for next_s2 in range(NUM_S2):
                p_s2 = action_control if next_s2 == action else 1.0 - action_control
                transitions[action, s1, s2, next_s1, next_s2] = p_s1 * p_s2

    return transitions


def induced_observed_transition(
    transitions: np.ndarray,
    b: np.ndarray,
    s1: int,
    action: int,
) -> np.ndarray:
    hidden_prob_one = b[b_index(s1, action)]
    hidden_probs = np.array([1.0 - hidden_prob_one, hidden_prob_one])

    observed = np.zeros(NUM_S1)
    for s2 in range(NUM_S2):
        for next_s1 in range(NUM_S1):
            observed[next_s1] += (
                hidden_probs[s2]
                * transitions[action, s1, s2, next_s1, :].sum()
            )

    return observed


def induced_reward(
    rewards: np.ndarray,
    b: np.ndarray,
    s1: int,
    action: int,
) -> float:
    hidden_prob_one = b[b_index(s1, action)]
    return float(
        (1.0 - hidden_prob_one) * rewards[s1, 0, action]
        + hidden_prob_one * rewards[s1, 1, action]
    )


def observed_sequence_value(
    rewards: np.ndarray,
    transitions: np.ndarray,
    b: np.ndarray,
    sequence: tuple[int, int],
    initial_s1: int,
) -> float:
    first_action, second_action = sequence
    value = induced_reward(rewards, b, initial_s1, first_action)
    observed_transition = induced_observed_transition(
        transitions,
        b,
        initial_s1,
        first_action,
    )

    for next_s1, probability in enumerate(observed_transition):
        value += (
            DISCOUNT
            * probability
            * induced_reward(rewards, b, next_s1, second_action)
        )

    return float(value)


def sequence_values(
    rewards: np.ndarray,
    transitions: np.ndarray,
    b: np.ndarray,
    initial_s1: int,
) -> list[tuple[tuple[int, int], float]]:
    return [
        (
            sequence,
            observed_sequence_value(
                rewards,
                transitions,
                b,
                sequence,
                initial_s1,
            ),
        )
        for sequence in product(range(NUM_ACTIONS), repeat=2)
    ]


def best_sequences(values: list[tuple[tuple[int, int], float]]) -> set[tuple[int, int]]:
    best_value = max(value for _sequence, value in values)
    return {
        sequence
        for sequence, value in values
        if abs(value - best_value) <= 1e-8
    }


def sample_bayes_plausible_b(rng: np.random.Generator) -> np.ndarray:
    """
    Sample posteriors b(s1,a)=P(S2=1 | S1=s1,A=a) that can be induced
    by a full-state behavior policy under the prior q=P(S2=1 | S1=s1).
    """

    b = np.zeros(NUM_S1 * NUM_ACTIONS)

    for s1 in range(NUM_S1):
        q = prior_hidden_prob(s1)
        lower = rng.uniform(0.001, q - 0.001)
        upper = rng.uniform(q + 0.001, 0.999)

        if rng.random() < 0.5:
            b[b_index(s1, 0)] = lower
            b[b_index(s1, 1)] = upper
        else:
            b[b_index(s1, 0)] = upper
            b[b_index(s1, 1)] = lower

    return b


def construct_attacker_from_b(b: np.ndarray) -> np.ndarray | None:
    action_probs = np.zeros((NUM_S1, NUM_S2, NUM_ACTIONS))

    for s1 in range(NUM_S1):
        q = prior_hidden_prob(s1)
        b0 = b[b_index(s1, 0)]
        b1 = b[b_index(s1, 1)]

        if abs(b0 - b1) <= TOL:
            p_action1 = np.array([0.5, 0.5])
        else:
            if q < min(b0, b1) - 1e-8 or q > max(b0, b1) + 1e-8:
                return None

            marginal_action1 = (q - b0) / (b1 - b0)
            marginal_action1 = float(np.clip(marginal_action1, 0.0, 1.0))

            p_action1_given_s2_1 = (
                marginal_action1 * b1 / q
                if q > TOL
                else 0.0
            )
            p_action1_given_s2_0 = (
                marginal_action1 * (1.0 - b1) / (1.0 - q)
                if 1.0 - q > TOL
                else 0.0
            )
            p_action1 = np.clip(
                [p_action1_given_s2_0, p_action1_given_s2_1],
                0.0,
                1.0,
            )

        for s2 in range(NUM_S2):
            action_probs[s1, s2, 1] = p_action1[s2]
            action_probs[s1, s2, 0] = 1.0 - p_action1[s2]

    return action_probs


@dataclass(frozen=True)
class Candidate:
    score: float
    seed: int
    hidden_effect: float
    action_effect: float
    action_control: float
    rewards: np.ndarray
    transitions: np.ndarray
    attacked_b: np.ndarray
    max_transition_shift: float


def transition_shift(
    transitions: np.ndarray,
    original_b: np.ndarray,
    attacked_b: np.ndarray,
) -> float:
    return max(
        abs(
            induced_observed_transition(transitions, attacked_b, s1, action)[1]
            - induced_observed_transition(transitions, original_b, s1, action)[1]
        )
        for s1 in range(NUM_S1)
        for action in range(NUM_ACTIONS)
    )


def tstar_changed(
    rewards: np.ndarray,
    transitions: np.ndarray,
    original_b: np.ndarray,
    attacked_b: np.ndarray,
) -> bool:
    for initial_s1 in range(NUM_S1):
        original_winners = best_sequences(
            sequence_values(rewards, transitions, original_b, initial_s1)
        )
        attacked_winners = best_sequences(
            sequence_values(rewards, transitions, attacked_b, initial_s1)
        )
        if original_winners != attacked_winners:
            return True

    return False


def search() -> Candidate:
    rng = np.random.default_rng(123)
    original_b = prior_b_vector()

    best: Candidate | None = None

    for seed in range(5000):
        reward_rng = np.random.default_rng(seed)
        rewards = reward_rng.normal(0.0, 1.0, size=(NUM_S1, NUM_S2, NUM_ACTIONS))

        for hidden_effect in (0.75, 0.85, 0.95):
            for action_effect in (0.00, 0.10, 0.20):
                for action_control in (0.65, 0.75, 0.85):
                    transitions = make_hidden_dependent_transition(
                        hidden_effect=hidden_effect,
                        action_effect=action_effect,
                        action_control=action_control,
                    )

                    for _ in range(300):
                        attacked_b = sample_bayes_plausible_b(rng)
                        if construct_attacker_from_b(attacked_b) is None:
                            continue

                        shift = transition_shift(
                            transitions,
                            original_b,
                            attacked_b,
                        )
                        if shift < 0.35:
                            continue

                        if not tstar_changed(
                            rewards,
                            transitions,
                            original_b,
                            attacked_b,
                        ):
                            continue

                        # Prefer large transition shifts and clear value changes.
                        score = shift
                        candidate = Candidate(
                            score=score,
                            seed=seed,
                            hidden_effect=hidden_effect,
                            action_effect=action_effect,
                            action_control=action_control,
                            rewards=rewards,
                            transitions=transitions,
                            attacked_b=attacked_b,
                            max_transition_shift=shift,
                        )

                        if best is None or candidate.score > best.score:
                            best = candidate

        if best is not None and best.score >= 0.55:
            return best

    if best is None:
        raise RuntimeError("No transition-mislearning case found.")

    return best


def print_case(candidate: Candidate) -> None:
    original_b = prior_b_vector()
    rewards = candidate.rewards
    transitions = candidate.transitions
    attacked_b = candidate.attacked_b
    attacker = construct_attacker_from_b(attacked_b)

    print("=== Transition-mislearning case ===")
    print(
        f"seed={candidate.seed} "
        f"discount={DISCOUNT:.2f} "
        f"hidden_effect={candidate.hidden_effect:.2f} "
        f"action_effect={candidate.action_effect:.2f} "
        f"action_control={candidate.action_control:.2f}"
    )
    print(f"max observed-transition shift={candidate.max_transition_shift:.3f}")
    print()

    print("Rewards:")
    for s1, s2 in product(range(NUM_S1), range(NUM_S2)):
        print(
            f"  S1={s1} S2={s2} "
            f"R(a0)={rewards[s1, s2, 0]:+.3f} "
            f"R(a1)={rewards[s1, s2, 1]:+.3f}"
        )
    print()

    print("Full transition P(S1'=1 | S1,S2,a):")
    for s1, s2 in product(range(NUM_S1), range(NUM_S2)):
        row = []
        for action in range(NUM_ACTIONS):
            row.append(transitions[action, s1, s2, 1, :].sum())
        print(f"  S1={s1} S2={s2}  a0={row[0]:.3f}  a1={row[1]:.3f}")
    print()

    print("Mixtures:")
    print(f"  original: {format_b(original_b)}")
    print(f"  attacked: {format_b(attacked_b)}")
    print()

    print("Attacker policy pi_dagger(a1 | S1,S2):")
    assert attacker is not None
    for s1, s2 in product(range(NUM_S1), range(NUM_S2)):
        print(f"  S1={s1} S2={s2}: {attacker[s1, s2, 1]:.6f}")
    print()

    print("Learned observed transition P(S1'=1 | S1,a):")
    for s1, action in product(range(NUM_S1), range(NUM_ACTIONS)):
        original = induced_observed_transition(transitions, original_b, s1, action)[1]
        attacked = induced_observed_transition(transitions, attacked_b, s1, action)[1]
        print(
            f"  S1={s1} a{action}: "
            f"original={original:.3f} attacked={attacked:.3f} "
            f"shift={attacked - original:+.3f}"
        )
    print()

    for label, b in (("Original", original_b), ("Attacked", attacked_b)):
        print(f"{label} open-loop values:")
        for initial_s1 in range(NUM_S1):
            values = sequence_values(rewards, transitions, b, initial_s1)
            winners = best_sequences(values)
            formatted = "  ".join(
                f"{format_sequence(sequence)}={value:+.3f}"
                + ("*" if sequence in winners else "")
                for sequence, value in sorted(values)
            )
            print(f"  s1={initial_s1}: {formatted}")
        print()


def main() -> None:
    print_case(search())


if __name__ == "__main__":
    main()
