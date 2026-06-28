from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

print("start")

NUM_S1 = 2
NUM_S2 = 2
NUM_STATES = NUM_S1 * NUM_S2
NUM_ACTIONS = 2


@dataclass(frozen=True)
class FactoredGame:
    # rewards[state, action] = R(s1, s2, a)
    rewards: np.ndarray


@dataclass(frozen=True)
class AttackerPolicy:
    # action_probs[s1, s2, a] = pi_dagger(a | s1, s2)
    # The attacker observes both S1 and hidden S2.
    action_probs: np.ndarray

    def sample_action(self, state: int, rng: np.random.Generator):
        s1, s2 = split_state(state)
        return int(rng.choice(NUM_ACTIONS, p=self.action_probs[s1, s2]))

    def label(self):
        entries = []
        for s1, s2 in product(range(NUM_S1), range(NUM_S2)):
            p_a1 = self.action_probs[s1, s2, 1]
            entries.append(f"({s1},{s2}):P(a1)={p_a1:.6f}")
        return "  ".join(entries)


@dataclass(frozen=True)
class LearnedRewardModel:
    # The victim only observes S1, so it estimates R_hat(s1, a).
    rewards: np.ndarray
    counts: np.ndarray


@dataclass(frozen=True)
class ExactFeasibilityResult:
    feasible: bool
    maximum_margin: float
    attacker: AttackerPolicy | None


def join_state(s1: int, s2: int):
    return s1 * NUM_S2 + s2


def split_state(state: int):
    return divmod(state, NUM_S2)


def build_random_game(seed: int):
    # The seed produces one concrete, reproducible reward function R.
    rng = np.random.default_rng(seed)
    rewards = np.round(rng.normal(0.0, 1.0, size=(NUM_STATES, NUM_ACTIONS)), 3)

    # Resample only in the unlikely event that rounding creates duplicate entries.
    while len(np.unique(rewards)) != rewards.size:
        rewards = np.round(rng.normal(0.0, 1.0, size=(NUM_STATES, NUM_ACTIONS)), 3)

    return FactoredGame(rewards=rewards)


def attacker_policy_grid(grid: tuple[float, ...]):
    # Each candidate chooses P(a1 | s1, s2) at all four complete states.
    # Five probability choices at four states gives 5^4 = 625 policies.
    policies = []
    for p_a1_by_state in product(grid, repeat=NUM_STATES):
        action_probs = np.empty((NUM_S1, NUM_S2, NUM_ACTIONS))
        for state, p_a1 in enumerate(p_a1_by_state):
            s1, s2 = split_state(state)
            action_probs[s1, s2] = (1.0 - p_a1, p_a1)
        policies.append(AttackerPolicy(action_probs=action_probs))
    return policies


def pure_target_policies():
    # pi_star[s1] is the action selected from observation S1.
    return list(product(range(NUM_ACTIONS), repeat=NUM_S1))


def generate_single_period_dataset(
    game: FactoredGame,
    attacker: AttackerPolicy,
    *,
    num_samples: int,
    state_distribution: np.ndarray,
    rng: np.random.Generator,
):
    # Every sample is an independent one-period interaction.
    # The attacker sees (S1,S2), but the victim receives only (S1, action, reward).
    data = []
    for _ in range(num_samples):
        state = int(rng.choice(NUM_STATES, p=state_distribution))
        s1, _ = split_state(state)
        action = attacker.sample_action(state, rng)
        reward = float(game.rewards[state, action])
        data.append((s1, action, reward))
    return data


def fit_reward_mle(data: list[tuple[int, int, float]]):
    # MLE for deterministic rewards is the sample mean for each observed (S1,a).
    reward_sums = np.zeros((NUM_S1, NUM_ACTIONS))
    reward_counts = np.zeros((NUM_S1, NUM_ACTIONS), dtype=int)

    for s1, action, reward in data:
        reward_sums[s1, action] += reward
        reward_counts[s1, action] += 1

    # A candidate without examples for every (S1,a) cannot support a fair comparison.
    if np.any(reward_counts == 0):
        return None

    rewards = reward_sums / reward_counts
    return LearnedRewardModel(rewards=rewards, counts=reward_counts)


def solve_single_period(model: LearnedRewardModel):
    # With one period, the victim simply selects the action with larger R_hat.
    return tuple(int(action) for action in np.argmax(model.rewards, axis=1))


def reward_margin(target: tuple[int, int], learned_rewards: np.ndarray):
    # The minimum reward advantage of the target action across observed states.
    margins = []
    for s1, target_action in enumerate(target):
        other_action = 1 - target_action
        margins.append(
            learned_rewards[s1, target_action] - learned_rewards[s1, other_action]
        )
    return float(min(margins))


def hidden_distribution(state_distribution: np.ndarray, s1: int):
    state_probs = np.array(
        [state_distribution[join_state(s1, s2)] for s2 in range(NUM_S2)]
    )
    return state_probs / state_probs.sum()


def learned_reward_gap(
    rewards: np.ndarray,
    hidden_probs: np.ndarray,
    target_action: int,
    target_action_probs: np.ndarray,
):
    # Infinite-data MLE reward difference induced by pi_dagger at one S1.
    other_action = 1 - target_action
    target_mass = hidden_probs @ target_action_probs
    other_mass = hidden_probs @ (1.0 - target_action_probs)

    if target_mass <= 0.0 or other_mass <= 0.0:
        return None

    target_reward = (
        hidden_probs * target_action_probs * rewards[:, target_action]
    ).sum() / target_mass
    other_reward = (
        hidden_probs * (1.0 - target_action_probs) * rewards[:, other_action]
    ).sum() / other_mass
    return float(target_reward - other_reward)


def exact_state_feasibility(
    rewards: np.ndarray,
    hidden_probs: np.ndarray,
    target_action: int,
    tolerance: float = 1e-12,
):
    # Let u=P(S2=0 | target action), v=P(S2=0 | other action), and
    # q=P(S2=0 | S1). Bayes plausibility requires q to lie between u and v.
    # The reward gap is linear in (u,v), so its supremum occurs at a corner.
    q = float(hidden_probs[0])
    posterior_corners = [
        (u, v) for u in (q, 1.0) for v in (0.0, q)
    ] + [
        (u, v) for u in (0.0, q) for v in (q, 1.0)
    ]

    other_action = 1 - target_action
    target_rewards = rewards[:, target_action]
    other_rewards = rewards[:, other_action]

    def posterior_gap(u, v):
        target_mean = u * target_rewards[0] + (1.0 - u) * target_rewards[1]
        other_mean = v * other_rewards[0] + (1.0 - v) * other_rewards[1]
        return float(target_mean - other_mean)

    best_u, best_v = max(posterior_corners, key=lambda pair: posterior_gap(*pair))
    maximum_margin = posterior_gap(best_u, best_v)

    if maximum_margin <= tolerance:
        return False, maximum_margin, None

    if abs(best_u - best_v) <= tolerance:
        return True, maximum_margin, np.full(NUM_S2, 0.5)

    # A boundary optimum may give one action zero probability. Move slightly into
    # the full-coverage region while retaining a positive teaching margin.
    interior_offset = min(q, 1.0 - q) / 2.0
    if best_u >= q >= best_v:
        interior_u, interior_v = q + interior_offset, q - interior_offset
    else:
        interior_u, interior_v = q - interior_offset, q + interior_offset

    for epsilon in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1):
        u = (1.0 - epsilon) * best_u + epsilon * interior_u
        v = (1.0 - epsilon) * best_v + epsilon * interior_v
        target_frequency = (q - v) / (u - v)
        target_action_probs = np.array(
            [
                target_frequency * u / hidden_probs[0],
                target_frequency * (1.0 - u) / hidden_probs[1],
            ]
        )
        target_action_probs = np.clip(target_action_probs, 0.0, 1.0)
        margin = learned_reward_gap(
            rewards, hidden_probs, target_action, target_action_probs
        )
        if margin is not None and margin > tolerance:
            return True, maximum_margin, target_action_probs

    raise RuntimeError("positive exact margin found, but witness construction failed")


def exact_single_period_feasibility(
    game: FactoredGame,
    target: tuple[int, int],
    state_distribution: np.ndarray,
):
    # The two observed values of S1 can be checked independently in one period.
    action_probs = np.empty((NUM_S1, NUM_S2, NUM_ACTIONS))
    state_margins = []

    for s1, target_action in enumerate(target):
        hidden_probs = hidden_distribution(state_distribution, s1)
        rewards = np.array(
            [game.rewards[join_state(s1, s2)] for s2 in range(NUM_S2)]
        )
        feasible, maximum_margin, target_probs = exact_state_feasibility(
            rewards, hidden_probs, target_action
        )
        state_margins.append(maximum_margin)

        if not feasible:
            return ExactFeasibilityResult(
                feasible=False,
                maximum_margin=min(state_margins),
                attacker=None,
            )

        for s2 in range(NUM_S2):
            probability_of_target = target_probs[s2]
            action_probs[s1, s2, target_action] = probability_of_target
            action_probs[s1, s2, 1 - target_action] = 1.0 - probability_of_target

    return ExactFeasibilityResult(
        feasible=True,
        maximum_margin=min(state_margins),
        attacker=AttackerPolicy(action_probs=action_probs),
    )


def print_reward_table(game: FactoredGame):
    print("Fixed original reward R(s1, s2, a)")
    print("s1  s2       a0       a1")
    for state in range(NUM_STATES):
        s1, s2 = split_state(state)
        print(
            f" {s1}   {s2}   "
            f"{game.rewards[state, 0]:>7.3f}  {game.rewards[state, 1]:>7.3f}"
        )


def range_condition_details(game: FactoredGame, target: tuple[int, int]):
    details = []
    for s1, target_action in enumerate(target):
        target_rewards = np.array(
            [
                game.rewards[join_state(s1, s2), target_action]
                for s2 in range(NUM_S2)
            ]
        )
        for alternative_action in range(NUM_ACTIONS):
            if alternative_action == target_action:
                continue
            alternative_rewards = np.array(
                [
                    game.rewards[join_state(s1, s2), alternative_action]
                    for s2 in range(NUM_S2)
                ]
            )
            details.append(
                {
                    "s1": s1,
                    "target_action": target_action,
                    "alternative_action": alternative_action,
                    "min_alternative": float(np.min(alternative_rewards)),
                    "max_alternative": float(np.max(alternative_rewards)),
                    "min_target": float(np.min(target_rewards)),
                    "max_target": float(np.max(target_rewards)),
                    # New theorem: min R(alternative) < max R(target).
                    "range_margin": float(
                        np.max(target_rewards) - np.min(alternative_rewards)
                    ),
                    # Old universal condition: max R(alternative) < min R(target).
                    "universal_margin": float(
                        np.min(target_rewards) - np.max(alternative_rewards)
                    ),
                }
            )
    return details


def print_range_condition_examples(game: FactoredGame, targets: list[tuple[int, int]]):
    print()
    print("Range-condition checks")
    print("pi_star          range margin    universal margin    status")

    rows = []
    for target in targets:
        details = range_condition_details(game, target)
        range_margin = min(item["range_margin"] for item in details)
        universal_margin = min(item["universal_margin"] for item in details)
        range_ok = range_margin > 0.0
        universal_ok = universal_margin > 0.0
        if range_ok and universal_ok:
            status = "range and universal"
        elif range_ok:
            status = "range only"
        else:
            status = "range fails"
        rows.append((target, range_margin, universal_margin, status, details))
        print(
            f"{target!s:<16} {range_margin:>12.4f} "
            f"{universal_margin:>17.4f}    {status}"
        )

    range_only = [row for row in rows if row[1] > 0.0 and row[2] <= 0.0]
    examples = range_only if range_only else [row for row in rows if row[1] > 0.0]
    if not examples:
        return

    target, _, _, status, details = examples[0]
    print()
    print(f"Example for pi_star={target} ({status})")
    for item in details:
        print(
            "  "
            f"s1={item['s1']} target=a{item['target_action']} "
            f"alternative=a{item['alternative_action']}: "
            f"min alternative={item['min_alternative']:.3f}, "
            f"max target={item['max_target']:.3f}, "
            f"range margin={item['range_margin']:.3f}; "
            f"max alternative={item['max_alternative']:.3f}, "
            f"min target={item['min_target']:.3f}, "
            f"universal margin={item['universal_margin']:.3f}"
        )


def main():
    seed = 0
    num_samples = 10_000
    simulation_seed = 10_007
    attacker_probability_grid = (0.0, 0.25, 0.5, 0.75, 1.0)

    game = build_random_game(seed)
    state_distribution = np.full(NUM_STATES, 1.0 / NUM_STATES)
    attackers = attacker_policy_grid(attacker_probability_grid)
    targets = pure_target_policies()

    # The heuristic stores the strongest finite-sample witness for each pi_star.
    witnesses: dict[
        tuple[int, int],
        tuple[AttackerPolicy, float, LearnedRewardModel],
    ] = {}
    skipped_for_missing_coverage = 0

    for attacker_index, attacker in enumerate(attackers):
        rng = np.random.default_rng(simulation_seed + attacker_index)
        data = generate_single_period_dataset(
            game,
            attacker,
            num_samples=num_samples,
            state_distribution=state_distribution,
            rng=rng,
        )
        learned_model = fit_reward_mle(data)
        if learned_model is None:
            skipped_for_missing_coverage += 1
            continue

        learned_policy = solve_single_period(learned_model)
        margin = reward_margin(learned_policy, learned_model.rewards)
        existing = witnesses.get(learned_policy)
        if existing is None or margin > existing[1]:
            witnesses[learned_policy] = (attacker, margin, learned_model)

    exact_results = {
        target: exact_single_period_feasibility(game, target, state_distribution)
        for target in targets
    }

    print_reward_table(game)
    print()
    print("Single-period information structure")
    print(f"  random reward seed: {seed}")
    print("  victim pi_star observes: S1")
    print("  attacker pi_dagger observes: (S1, S2)")
    print("  victim training tuple: (S1, action, reward)")
    print(f"  pi_dagger policies searched: {len(attackers)}")
    print(f"  independent samples per pi_dagger: {num_samples}")
    print(f"  skipped for missing action coverage: {skipped_for_missing_coverage}")
    print()
    print("Target-policy teachability")
    print("pi_star          sampled search    exact result    exact max margin")

    for target in targets:
        sampled_result = "witness found" if target in witnesses else "not found"
        exact = exact_results[target]
        exact_label = "feasible" if exact.feasible else "IMPOSSIBLE"
        print(
            f"{target!s:<16} {sampled_result:<17} "
            f"{exact_label:<15} {exact.maximum_margin:>8.4f}"
        )

    print_range_condition_examples(game, targets)

    for target in targets:
        witness = witnesses.get(target)
        if witness is None:
            continue
        attacker, margin, learned_model = witness
        print()
        print(f"Finite-sample witness for pi_star={target}")
        print(f"  pi_dagger: {attacker.label()}")
        print(f"  learned reward margin: {margin:.4f}")
        print("  learned R_hat(s1, a):")
        print(np.round(learned_model.rewards, 3))
        print("  sample counts for (s1, a):")
        print(learned_model.counts)

    for target in targets:
        exact = exact_results[target]
        if not exact.feasible:
            continue
        print()
        print(f"Exact witness for pi_star={target}")
        print(f"  pi_dagger: {exact.attacker.label()}")
        print(f"  supremum teaching margin: {exact.maximum_margin:.4f}")

    impossible = [target for target in targets if not exact_results[target].feasible]
    if impossible:
        print()
        print(f"Exactly impossible targets: {impossible}")


if __name__ == "__main__":
    main()
