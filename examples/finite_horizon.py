from __future__ import annotations

import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from pomdp_starter import TabularPOMDP, best_alpha, pretty_plan, solve_finite_horizon


def build_tiger() -> TabularPOMDP:
    # The classic Tiger benchmark has two hidden states:
    # tiger_left and tiger_right.
    #
    # Actions:
    #   listen      : noisy information, state does not change
    #   open_left   : commit to opening the left door, then reset
    #   open_right  : commit to opening the right door, then reset
    transitions = np.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.5, 0.5], [0.5, 0.5]],
            [[0.5, 0.5], [0.5, 0.5]],
        ]
    )

    observations = np.array(
        [
            # Listening is informative but noisy: 85% correct. PG 119
            [[0.85, 0.15], [0.15, 0.85]],
            # Opening a door gives no useful observation in this simple model.
            [[0.5, 0.5], [0.5, 0.5]],
            [[0.5, 0.5], [0.5, 0.5]],
        ]
    )

    rewards = np.array(
        [
            # If the tiger is left, opening right is good and opening left is disastrous.
            [-1.0, -100.0, 10.0],
            # If the tiger is right, the payoffs flip.
            [-1.0, 10.0, -100.0],
        ]
    )

    return TabularPOMDP(
        transitions=transitions,
        observations=observations,
        rewards=rewards,
        discount=0.95,
        state_names=("tiger_left", "tiger_right"),
        action_names=("listen", "open_left", "open_right"),
        observation_names=("hear_left", "hear_right"),
    )


def main() -> None:
    # Start from complete uncertainty, which is the usual Tiger setup.
    pomdp = build_tiger()
    horizon = 3
    initial_belief = np.array([0.5, 0.5])

    # Solve the finite-horizon problem exactly, then inspect the final alpha-set.
    stages = solve_finite_horizon(pomdp, horizon=horizon)
    final_stage = stages[horizon]
    best = best_alpha(final_stage, initial_belief)

    print(f"Horizon: {horizon}")
    print(f"Initial belief: {initial_belief.tolist()}")
    print(f"Number of alpha-vectors at final stage: {len(final_stage)}")
    print(f"Best initial action: {pomdp.action_names[best.action]}")
    print(f"Initial value: {best.value_at(initial_belief):.3f}")
    print()
    print("Best contingent plan from the initial belief:")
    print(pretty_plan(best, pomdp))
    print()
    print("Parsimonious alpha-vectors:")

    for alpha in final_stage:
        action_name = pomdp.action_names[alpha.action]
        print(f"  action={action_name:<10} values={np.round(alpha.values, 3).tolist()}")
    print()

    # Belief traces for the paper's listening intuition:
    #    open the opposite door after hearing one side twice more than the other

    for observations in (
        ("hear_left",),
        ("hear_left", "hear_left"),
        ("hear_left", "hear_right"),
        ("hear_right", "hear_right"),
    ):
        describe_observation_sequence(pomdp, initial_belief, observations)


def describe_observation_sequence(
    pomdp: TabularPOMDP, initial_belief: np.ndarray, observations: tuple[str, ...]
) -> None:
    # This helper just walks through a fixed observation history to show how
    # the Bayesian belief update changes the agent's confidence.
    action_index = pomdp.action_names.index("listen")
    belief = initial_belief.copy()
    left_count = 0
    right_count = 0

    print(f"sequence={list(observations)}")
    print(f"  start belief={np.round(belief, 3).tolist()}")

    for observation_name in observations:
        observation_index = pomdp.observation_names.index(observation_name)
        belief = pomdp.belief_update(belief, action_index, observation_index)

        # The paper's infinite-horizon Tiger policy can be read informally as
        # "keep track of which side has been heard more often."
        if observation_name == "hear_left":
            left_count += 1
        else:
            right_count += 1

        # A gap of two is the paper's easy-to-read summary of when to commit.
        difference = left_count - right_count
        leaning = "open_right" if difference >= 2 else "open_left" if difference <= -2 else "listen"

        print(
            "  after "
            f"{observation_name:<10} belief={np.round(belief, 3).tolist()} "
            f"net_left_minus_right={difference:+d} suggested_action={leaning}"
        )

    print()


if __name__ == "__main__":
    main()
