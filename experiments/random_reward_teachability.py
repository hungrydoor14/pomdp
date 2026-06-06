from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np


NUM_S1 = 2
NUM_S2 = 2
NUM_STATES = NUM_S1 * NUM_S2
NUM_ACTIONS = 2


# The true game has four states:
# (S1=0, S2=0), (S1=0, S2=1), (S1=1, S2=0), (S1=1, S2=1).
# The victim sees S1, while S2 is hidden.
@dataclass(frozen=True)
class FactoredGame:
    # transitions[a, state, next_state] = P(s' | s, a)
    transitions: np.ndarray

    # rewards[state, a] = R(s1, s2, a)
    rewards: np.ndarray
    discount: float


@dataclass(frozen=True)
class AttackerPolicy:
    # action_probs[s1, s2, a] = pi_dagger(a | s1, s2)
    # Unlike the victim, the attacker may condition its behavior on hidden S2.
    action_probs: np.ndarray

    def sample_action(self, state: int, rng: np.random.Generator):
        # This is how pi_dagger generates one action for the training set.
        s1, s2 = split_state(state)
        return int(rng.choice(NUM_ACTIONS, p=self.action_probs[s1, s2]))

    def label(self):
        entries = []
        for s1, s2 in product(range(NUM_S1), range(NUM_S2)):
            p_a1 = self.action_probs[s1, s2, 1]
            entries.append(f"({s1},{s2}):P(a1)={p_a1:.2f}")
        return "  ".join(entries)


@dataclass(frozen=True)
class LearnedMDP:
    # The victim treats observed S1 as the complete state.
    # Therefore this estimated model has only two states, not the four true states.
    transitions: np.ndarray
    rewards: np.ndarray
    discount: float


def join_state(s1: int, s2: int):
    return s1 * NUM_S2 + s2


def split_state(state: int):
    return divmod(state, NUM_S2)


def build_random_game(seed: int):
    # Fixing the seed gives us one concrete R and transition model to study.
    rng = np.random.default_rng(seed)
    transitions = random_stochastic_tensor(
        rng, (NUM_ACTIONS, NUM_STATES, NUM_STATES), alpha=1.4
    )

    # The notes suggest starting with concrete, non-identical random rewards.
    rewards = np.round(rng.normal(0.0, 1.0, size=(NUM_STATES, NUM_ACTIONS)), 3)
    while len(np.unique(rewards)) != rewards.size:
        rewards = np.round(rng.normal(0.0, 1.0, size=(NUM_STATES, NUM_ACTIONS)), 3)

    return FactoredGame(transitions=transitions, rewards=rewards, discount=0.9)


def random_stochastic_tensor(
    rng: np.random.Generator, shape: tuple[int, int, int], alpha: float
):
    rows = np.empty(shape)
    for index in np.ndindex(shape[0], shape[1]):
        rows[index] = rng.dirichlet(np.full(shape[2], alpha))
    return rows


def attacker_policy_grid(grid: tuple[float, ...]):
    # A candidate pi_dagger chooses a probability of a1 at each full state (S1, S2).
    # With five grid values and four full states, this creates 5^4 = 625 policies.
    policies = []
    for p_a1_by_state in product(grid, repeat=NUM_STATES):
        action_probs = np.empty((NUM_S1, NUM_S2, NUM_ACTIONS))
        for state, p_a1 in enumerate(p_a1_by_state):
            s1, s2 = split_state(state)
            action_probs[s1, s2] = (1.0 - p_a1, p_a1)
        policies.append(AttackerPolicy(action_probs=action_probs))
    return policies


def pure_target_policies():
    # pi_star[s1] is the action selected from the victim's observation S1.
    # With two S1 values and two actions, the four targets are:
    # (a0,a0), (a0,a1), (a1,a0), and (a1,a1).
    return list(product(range(NUM_ACTIONS), repeat=NUM_S1))


def simulate_observed_dataset(
    game: FactoredGame,
    attacker: AttackerPolicy,
    *,
    episodes: int,
    steps: int,
    initial_distribution: np.ndarray,
    rng: np.random.Generator,
):
    # The attacker acts using (S1, S2), but hides S2 before giving data to the victim.
    # Thus pi_dagger controls which hidden S2 values are associated with each action.
    data = []
    for _ in range(episodes):
        state = int(rng.choice(NUM_STATES, p=initial_distribution))
        for _ in range(steps):
            s1, _ = split_state(state)
            action = attacker.sample_action(state, rng)
            reward = float(game.rewards[state, action])
            next_state = int(rng.choice(NUM_STATES, p=game.transitions[action, state]))
            next_s1, _ = split_state(next_state)

            # This is the complete training example visible to the victim.
            data.append((s1, action, reward, next_s1))
            state = next_state
    return data


def fit_observed_mle(
    data: list[tuple[int, int, float, int]],
    *,
    discount: float,
    transition_prior: float = 0.5,
    reward_prior_count: float = 1.0,
    reward_prior_mean: float = 0.0,
):
    # The victim pools samples that have the same observed S1 and action. Because S2
    # is missing, rewards from different hidden states are averaged together.
    transition_counts = np.full(
        (NUM_ACTIONS, NUM_S1, NUM_S1), transition_prior, dtype=float
    )
    reward_sums = np.full(
        (NUM_S1, NUM_ACTIONS), reward_prior_count * reward_prior_mean, dtype=float
    )
    reward_counts = np.full((NUM_S1, NUM_ACTIONS), reward_prior_count, dtype=float)

    for s1, action, reward, next_s1 in data:
        transition_counts[action, s1, next_s1] += 1.0
        reward_sums[s1, action] += reward
        reward_counts[s1, action] += 1.0

    # These are the victim's MLE estimates after S2 has been hidden.
    transitions = transition_counts / transition_counts.sum(axis=-1, keepdims=True)
    rewards = reward_sums / reward_counts
    return LearnedMDP(transitions=transitions, rewards=rewards, discount=discount)


def solve_mdp(mdp: LearnedMDP, *, tolerance: float = 1e-12):
    # Solve the victim's estimated two-state model with value iteration.
    values = np.zeros(NUM_S1)
    while True:
        q_values = mdp.rewards.T + mdp.discount * np.einsum(
            "asn,n->as", mdp.transitions, values
        )
        next_values = q_values.max(axis=0)
        if np.max(np.abs(next_values - values)) <= tolerance:
            values = next_values
            break
        values = next_values

    q_values = mdp.rewards.T + mdp.discount * np.einsum(
        "asn,n->as", mdp.transitions, values
    )
    policy = tuple(int(action) for action in np.argmax(q_values, axis=0))
    return policy, values, q_values


def target_margin(target: tuple[int, int], q_values: np.ndarray):
    # A positive margin means the target action strictly beats the other action
    # at both values of S1. Larger margins mean the learned choice is more robust.
    margins = []
    for s1, chosen_action in enumerate(target):
        other_action = 1 - chosen_action
        margins.append(q_values[chosen_action, s1] - q_values[other_action, s1])
    return float(min(margins))


def true_policy_value(
    game: FactoredGame,
    policy: tuple[int, int],
    initial_distribution: np.ndarray,
):
    # This evaluates pi_star in the original four-state game. It is for comparison:
    # the victim chooses pi_star using the learned model, not this true value.
    policy_transition = np.empty((NUM_STATES, NUM_STATES))
    policy_reward = np.empty(NUM_STATES)
    for state in range(NUM_STATES):
        s1, _ = split_state(state)
        action = policy[s1]
        policy_transition[state] = game.transitions[action, state]
        policy_reward[state] = game.rewards[state, action]

    values = np.linalg.solve(
        np.eye(NUM_STATES) - game.discount * policy_transition,
        policy_reward,
    )
    return float(initial_distribution @ values)


def print_reward_table(game: FactoredGame):
    print("Fixed original reward R(s1, s2, a)")
    print("s1  s2       a0       a1")
    for state in range(NUM_STATES):
        s1, s2 = split_state(state)
        print(f" {s1}   {s2}   {game.rewards[state, 0]:>7.3f}  {game.rewards[state, 1]:>7.3f}")


def main():
    # Experiment settings. Change these values directly and rerun the file.
    seed = 7
    episodes = 100
    steps = 20
    simulation_seed = 10_007
    attacker_probability_grid = (0.0, 0.25, 0.5, 0.75, 1.0)

    game = build_random_game(seed)
    initial_distribution = np.full(NUM_STATES, 1.0 / NUM_STATES)
    attackers = attacker_policy_grid(attacker_probability_grid)
    targets = pure_target_policies()

    # For each teachable pi_star, store the strongest pi_dagger found for it.
    witnesses: dict[tuple[int, int], tuple[AttackerPolicy, float, LearnedMDP]] = {}

    # Generate a separate training set under every candidate pi_dagger.
    for attacker_index, attacker in enumerate(attackers):
        rng = np.random.default_rng(simulation_seed + attacker_index)
        data = simulate_observed_dataset(
            game,
            attacker,
            episodes=episodes,
            steps=steps,
            initial_distribution=initial_distribution,
            rng=rng,
        )
        learned_mdp = fit_observed_mle(data, discount=game.discount)
        learned_policy, _, q_values = solve_mdp(learned_mdp)
        margin = target_margin(learned_policy, q_values)

        # The attacker is a witness for whichever policy the victim learns.
        # Keep the witness with the largest positive optimality margin.
        existing = witnesses.get(learned_policy)
        if existing is None or margin > existing[1]:
            witnesses[learned_policy] = (attacker, margin, learned_mdp)

    print_reward_table(game)
    print()
    print("Information structure")
    print("  victim pi_star observes: S1")
    print("  attacker pi_dagger observes: (S1, S2)")
    print("  victim training tuple: (S1, action, reward, next_S1)")
    print(f"  pi_dagger policies searched: {len(attackers)}")
    print(f"  samples per pi_dagger: {episodes * steps}")
    print()
    print("Target-policy teachability")
    print("pi_star          true value    teachable    best learned margin")

    for target in targets:
        true_value = true_policy_value(game, target, initial_distribution)
        witness = witnesses.get(target)
        teachable = "yes" if witness is not None else "not found"
        margin = f"{witness[1]:.4f}" if witness is not None else "-"
        print(f"{target!s:<16} {true_value:>10.4f}    {teachable:<10}   {margin}")

    for target in targets:
        witness = witnesses.get(target)
        if witness is None:
            continue
        attacker, margin, learned_mdp = witness
        print()
        print(f"Witness for pi_star={target}")
        print(f"  pi_dagger: {attacker.label()}")
        print(f"  optimality margin in learned model: {margin:.4f}")
        print("  learned R_hat(s1, a):")
        print(np.round(learned_mdp.rewards, 3))
        print("  learned P_hat(a, s1, next_s1):")
        print(np.round(learned_mdp.transitions, 3))

    missing = [target for target in targets if target not in witnesses]
    if missing:
        print()
        print(f"Not found by this finite grid: {missing}")
        print("'Not found' is empirical, not a proof that the target is impossible.")


if __name__ == "__main__":
    main()
