from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from collections import defaultdict, Counter

import numpy as np

import os
import sys

sys.path.append(
    os.path.join(os.path.dirname(__file__), "..", "src")
)

from finite_horizon_solver import (
    AlphaVector,
    TabularPOMDP,
    action_after_observation_history,
    best_alpha,
    greedy_action,
    solve_finite_horizon,
)

NUM_S1 = 2
NUM_S2 = 2
NUM_STATES = NUM_S1 * NUM_S2
NUM_ACTIONS = 2


def join_state(s1: int, s2: int) -> int:
    return s1 * NUM_S2 + s2


def split_state(state: int) -> tuple[int, int]:
    return divmod(state, NUM_S2)


def initial_belief_given_s1(observed_s1: int, initial_match_prob: float) -> np.ndarray:
    belief = np.zeros(NUM_STATES)
    for s2 in range(NUM_S2):
        belief[join_state(observed_s1, s2)] = (
            initial_match_prob if s2 == observed_s1 else 1.0 - initial_match_prob
        )
    return belief


def hidden_probs_from_belief(belief: np.ndarray, observed_s1: int) -> np.ndarray:
    probs = np.array([belief[join_state(observed_s1, s2)] for s2 in range(NUM_S2)])
    return probs / probs.sum()


def build_action_dependent_factored_pomdp(
    seed: int,
    *,
    action_control: float,
    p_s1_matches_s2: float,
    discount: float = 0.95,
    reward_scale: float = 1.0,
) -> TabularPOMDP:
    """
    Action-dependent horizon-2 lab.

    State is (S1,S2). S1 is observed, S2 is hidden.

    action_control controls how much action a pushes next hidden state S2' toward a:
      action_control=0.50 means action does not control S2' much.
      action_control=0.90 means action a usually makes S2'=a.

    p_s1_matches_s2 controls how informative next public state S1' is about S2'.
    """
    rng = np.random.default_rng(seed)
    rewards = np.round(rng.normal(0.0, reward_scale, size=(NUM_STATES, NUM_ACTIONS)), 3)
    while len(np.unique(rewards)) != rewards.size:
        rewards = np.round(rng.normal(0.0, reward_scale, size=(NUM_STATES, NUM_ACTIONS)), 3)

    transitions = np.zeros((NUM_ACTIONS, NUM_STATES, NUM_STATES))
    for action, state in product(range(NUM_ACTIONS), range(NUM_STATES)):
        for next_s2 in range(NUM_S2):
            p_s2 = action_control if next_s2 == action else 1.0 - action_control
            for next_s1 in range(NUM_S1):
                p_s1 = p_s1_matches_s2 if next_s1 == next_s2 else 1.0 - p_s1_matches_s2
                transitions[action, state, join_state(next_s1, next_s2)] = p_s1 * p_s2

    observations = np.zeros((NUM_ACTIONS, NUM_STATES, NUM_S1))
    for action, state in product(range(NUM_ACTIONS), range(NUM_STATES)):
        s1, _ = split_state(state)
        observations[action, state, s1] = 1.0

    return TabularPOMDP(
        transitions=transitions,
        observations=observations,
        rewards=rewards,
        discount=discount,
        state_names=tuple(f"s1={s1},s2={s2}" for s1, s2 in product(range(NUM_S1), range(NUM_S2))),
        action_names=("a0", "a1"),
        observation_names=("s1=0", "s1=1"),
    )


def expected_immediate_reward(pomdp: TabularPOMDP, belief: np.ndarray, action: int) -> float:
    return float(belief @ pomdp.rewards[:, action])


def expected_future_value(pomdp: TabularPOMDP, belief: np.ndarray, action: int, stage1: list[AlphaVector]) -> float:
    total = 0.0
    for obs in range(NUM_S1):
        predicted = belief @ pomdp.transitions[action]
        obs_prob = float(predicted @ pomdp.observations[action, :, obs])
        if obs_prob <= 1e-12:
            continue
        posterior = pomdp.belief_update(belief, action, obs)
        total += obs_prob * best_alpha(stage1, posterior).value_at(posterior)
    return float(pomdp.discount * total)


def q_decomposition(pomdp: TabularPOMDP, belief: np.ndarray, action: int, stage1: list[AlphaVector]) -> tuple[float, float, float]:
    imm = expected_immediate_reward(pomdp, belief, action)
    fut = expected_future_value(pomdp, belief, action, stage1)
    return imm + fut, imm, fut


def terminal_feasibility_for_history(
    pomdp: TabularPOMDP,
    posterior: np.ndarray,
    current_s1: int,
    target_action: int,
) -> tuple[bool, float, bool, float, bool, float]:
    # Weak/existential feasibility from the paper: some attacker-induced hidden-state
    # mixture can make the target action beat each alternative.
    other_action = 1 - target_action
    target_values = np.array([
        pomdp.rewards[join_state(current_s1, s2), target_action]
        for s2 in range(NUM_S2)
    ])
    other_values = np.array([
        pomdp.rewards[join_state(current_s1, s2), other_action]
        for s2 in range(NUM_S2)
    ])
    feasible_margin = float(np.max(target_values) - np.min(other_values))

    # Strong dominance analogue: every hidden-state reward for the target beats
    # every hidden-state reward for the alternative.
    dominance_margin = float(np.min(target_values) - np.max(other_values))

    # Actual optimality under the POMDP posterior reached at this history.
    hidden_posterior = hidden_probs_from_belief(posterior, current_s1)
    posterior_margin = float(hidden_posterior @ target_values - hidden_posterior @ other_values)

    return (
        feasible_margin > 1e-12,
        feasible_margin,
        dominance_margin > 1e-12,
        dominance_margin,
        posterior_margin > 1e-12,
        posterior_margin,
    )


@dataclass(frozen=True)
class InstanceSummary:
    seed: int
    action_control: float
    p_s1_matches_s2: float
    initial_match_prob: float
    pi1: tuple[int, int]
    pi2: tuple[int, int, int, int]  # order: (0,0),(0,1),(1,0),(1,1)
    history_dependent: bool
    terminal_all_feasible: bool
    min_terminal_margin: float
    terminal_all_dominant: bool
    min_terminal_dominance_margin: float
    terminal_all_posterior_optimal: bool
    min_terminal_posterior_margin: float
    both_components_help: tuple[bool, bool]
    q_gaps: tuple[float, float]
    imm_gaps: tuple[float, float]
    fut_gaps: tuple[float, float]


def analyze(seed: int, action_control: float, p_s1_matches_s2: float, initial_match_prob: float) -> InstanceSummary:
    pomdp = build_action_dependent_factored_pomdp(
        seed,
        action_control=action_control,
        p_s1_matches_s2=p_s1_matches_s2,
    )
    stages = solve_finite_horizon(pomdp, horizon=2)
    stage1 = stages[1]
    stage2 = stages[2]

    pi1: list[int] = []
    pi2: list[int] = []
    terminal_feasible: list[bool] = []
    terminal_margins: list[float] = []
    terminal_dominant: list[bool] = []
    terminal_dominance_margins: list[float] = []
    terminal_posterior_optimal: list[bool] = []
    terminal_posterior_margins: list[float] = []
    both_help: list[bool] = []
    q_gaps: list[float] = []
    imm_gaps: list[float] = []
    fut_gaps: list[float] = []

    for s1_t1 in range(NUM_S1):
        b0 = initial_belief_given_s1(s1_t1, initial_match_prob)
        root = best_alpha(stage2, b0)
        target_a1 = root.action
        other_a1 = 1 - target_a1
        pi1.append(target_a1)

        target_q, target_imm, target_fut = q_decomposition(pomdp, b0, target_a1, stage1)
        other_q, other_imm, other_fut = q_decomposition(pomdp, b0, other_a1, stage1)
        q_gap = target_q - other_q
        imm_gap = target_imm - other_imm
        fut_gap = target_fut - other_fut
        q_gaps.append(q_gap)
        imm_gaps.append(imm_gap)
        fut_gaps.append(fut_gap)
        both_help.append(imm_gap > 0 and fut_gap > 0)

        for s1_t2 in range(NUM_S1):
            posterior = pomdp.belief_update(b0, target_a1, s1_t2)
            #print(s1_t1,s1_t2,hidden_probs_from_belief(posterior, s1_t2))
            target_a2 = action_after_observation_history(root, [s1_t2])
            pi2.append(target_a2)
            (
                feasible,
                feasible_margin,
                dominant,
                dominance_margin,
                posterior_optimal,
                posterior_margin,
            ) = terminal_feasibility_for_history(pomdp, posterior, s1_t2, target_a2)
            terminal_feasible.append(feasible)
            terminal_margins.append(feasible_margin)
            terminal_dominant.append(dominant)
            terminal_dominance_margins.append(dominance_margin)
            terminal_posterior_optimal.append(posterior_optimal)
            terminal_posterior_margins.append(posterior_margin)

    history_dependent = (pi2[0] != pi2[2]) or (pi2[1] != pi2[3])
    return InstanceSummary(
        seed=seed,
        action_control=action_control,
        p_s1_matches_s2=p_s1_matches_s2,
        initial_match_prob=initial_match_prob,
        pi1=tuple(pi1),
        pi2=tuple(pi2),
        history_dependent=history_dependent,
        terminal_all_feasible=all(terminal_feasible),
        min_terminal_margin=min(terminal_margins),
        terminal_all_dominant=all(terminal_dominant),
        min_terminal_dominance_margin=min(terminal_dominance_margins),
        terminal_all_posterior_optimal=all(terminal_posterior_optimal),
        min_terminal_posterior_margin=min(terminal_posterior_margins),
        both_components_help=tuple(both_help),
        q_gaps=tuple(q_gaps),
        imm_gaps=tuple(imm_gaps),
        fut_gaps=tuple(fut_gaps),
    )


def policy_label(pi2: tuple[int, int, int, int]) -> str:
    return "{" + ", ".join(
        f"({h[0]},{h[1]}):a{a}" for h, a in zip(product(range(2), range(2)), pi2)
    ) + "}"


def print_instance(row: InstanceSummary) -> None:
    print(f"seed={row.seed}  action_control={row.action_control:.2f}  obs_info={row.p_s1_matches_s2:.2f}")
    print(f"  pi1*: s1=0 -> a{row.pi1[0]}, s1=1 -> a{row.pi1[1]}")
    print(f"  pi2*: {policy_label(row.pi2)}")
    print(f"  history dependent pi2*: {row.history_dependent}")
    print(f"  terminal all feasible: {row.terminal_all_feasible}, min feasible margin={row.min_terminal_margin:.4f}")
    print(f"  terminal all dominant: {row.terminal_all_dominant}, min dominance margin={row.min_terminal_dominance_margin:.4f}")
    print(f"  terminal all posterior-optimal: {row.terminal_all_posterior_optimal}, min posterior margin={row.min_terminal_posterior_margin:.4f}")
    for s1 in range(NUM_S1):
        print(
            f"  s1={s1}: Q gap={row.q_gaps[s1]:+.4f}, "
            f"immediate gap={row.imm_gaps[s1]:+.4f}, "
            f"future gap={row.fut_gaps[s1]:+.4f}, "
            f"both help={row.both_components_help[s1]}"
        )


def main() -> None:
    seeds = range(80)
    action_controls = [0.55, 0.70, 0.85, 0.95]
    obs_infos = [0.55, 0.70, 0.85, 0.95]
    initial_match_prob = 0.70

    rows: list[InstanceSummary] = []
    for control in action_controls:
        for obs_info in obs_infos:
            for seed in seeds:
                rows.append(analyze(seed, control, obs_info, initial_match_prob))

    print("=== Joint horizon-2 experiment summary ===")
    print(f"instances: {len(rows)}")
    print(f"history-dependent pi2*: {sum(r.history_dependent for r in rows)}")
    print(f"terminal all feasible: {sum(r.terminal_all_feasible for r in rows)}")
    print(f"terminal all dominant: {sum(r.terminal_all_dominant for r in rows)}")
    print(f"terminal all posterior-optimal: {sum(r.terminal_all_posterior_optimal for r in rows)}")
    print(f"history-dependent and terminal feasible: {sum(r.history_dependent and r.terminal_all_feasible for r in rows)}")
    print(f"history-dependent and terminal dominant: {sum(r.history_dependent and r.terminal_all_dominant for r in rows)}")
    print(f"history-dependent and terminal posterior-optimal: {sum(r.history_dependent and r.terminal_all_posterior_optimal for r in rows)}")
    print(f"both components help for both s1 values: {sum(all(r.both_components_help) for r in rows)}")
    print()

    print("=== By action_control / obs_info ===")
    grouped: dict[tuple[float, float], list[InstanceSummary]] = defaultdict(list)
    for r in rows:
        grouped[(r.action_control, r.p_s1_matches_s2)].append(r)
    print("control  obsinfo  hist_dep  term_feas  term_dom  post_opt  both_help_both_s1")
    for key in sorted(grouped):
        g = grouped[key]
        print(
            f"{key[0]:>7.2f}  {key[1]:>7.2f}  "
            f"{sum(r.history_dependent for r in g):>8}/{len(g):<3}  "
            f"{sum(r.terminal_all_feasible for r in g):>9}/{len(g):<3}  "
            f"{sum(r.terminal_all_dominant for r in g):>8}/{len(g):<3}  "
            f"{sum(r.terminal_all_posterior_optimal for r in g):>8}/{len(g):<3}  "
            f"{sum(all(r.both_components_help) for r in g):>17}/{len(g):<3}"
        )
    print()

    print("=== Most common joint policies ===")
    joint_counter = Counter((r.pi1, r.pi2) for r in rows)
    for (pi1, pi2), count in joint_counter.most_common(10):
        print(f"{count:>4}  pi1={pi1}, pi2={policy_label(pi2)}")
    print()

    print("=== Clean candidate pattern: first action helps immediate AND future, terminal posterior-optimal ===")
    candidates = [
        r for r in rows
        if all(r.both_components_help) and r.terminal_all_posterior_optimal
    ]
    candidates.sort(key=lambda r: (min(r.imm_gaps) + min(r.fut_gaps) + r.min_terminal_margin), reverse=True)
    if not candidates:
        print("No clean cases found. Try more seeds or adjust reward/transition parameters.")
    else:
        for r in candidates[:8]:
            print_instance(r)

    print("\n=== Interesting but messy: history-dependent pi2*, terminal posterior-optimal ===")
    interesting = [r for r in rows if r.history_dependent and r.terminal_all_posterior_optimal]
    interesting.sort(key=lambda r: r.min_terminal_posterior_margin, reverse=True)
    for r in interesting[:8]:
        print_instance(r)


if __name__ == "__main__":
    main()
