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


def join_state(s1, s2):
    return s1 * NUM_S2 + s2


def split_state(state):
    return divmod(state, NUM_S2)


def initial_belief_given_s1(observed_s1, initial_match_prob):
    belief = np.zeros(NUM_STATES)
    for s2 in range(NUM_S2):
        belief[join_state(observed_s1, s2)] = (
            initial_match_prob if s2 == observed_s1 else 1.0 - initial_match_prob
        )
    return belief


def hidden_probs_from_belief(belief, observed_s1):
    probs = np.array([belief[join_state(observed_s1, s2)] for s2 in range(NUM_S2)])
    return probs / probs.sum()


def build_action_dependent_factored_pomdp(
    seed,
    *,
    action_control,
    p_s1_matches_s2,
    discount=0.95,
    reward_scale=1.0,
):
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


def expected_immediate_reward(pomdp, belief, action):
    return float(belief @ pomdp.rewards[:, action])


def expected_future_value(pomdp, belief, action, stage1):
    total = 0.0
    for obs in range(NUM_S1):
        predicted = belief @ pomdp.transitions[action]
        obs_prob = float(predicted @ pomdp.observations[action, :, obs])
        if obs_prob <= 1e-12:
            continue
        posterior = pomdp.belief_update(belief, action, obs)
        total += obs_prob * best_alpha(stage1, posterior).value_at(posterior)
    return float(pomdp.discount * total)


def q_decomposition(pomdp, belief, action, stage1):
    imm = expected_immediate_reward(pomdp, belief, action)
    fut = expected_future_value(pomdp, belief, action, stage1)
    return imm + fut, imm, fut


def root_q2_state_values(
    pomdp,
    belief,
    action,
    stage1,
):
    # Belief-conditioned Q2 analogue:
    # Q2(s,a) = R(s,a) + gamma * sum_o P(o | s,a) V1(tau(b,a,o)).
    continuation_by_observation = np.zeros(NUM_S1)
    for obs in range(NUM_S1):
        predicted = belief @ pomdp.transitions[action]
        obs_prob = float(predicted @ pomdp.observations[action, :, obs])
        if obs_prob <= 1e-12:
            continue
        posterior = pomdp.belief_update(belief, action, obs)
        continuation_by_observation[obs] = best_alpha(stage1, posterior).value_at(posterior)

    values = pomdp.rewards[:, action].copy()
    for state in range(NUM_STATES):
        future = 0.0
        for next_state in range(NUM_STATES):
            transition_prob = pomdp.transitions[action, state, next_state]
            for obs in range(NUM_S1):
                obs_prob = pomdp.observations[action, next_state, obs]
                future += transition_prob * obs_prob * continuation_by_observation[obs]
        values[state] += pomdp.discount * future
    return values


def root_q2_diagnostics(
    pomdp,
    belief,
    observed_s1,
    target_action,
    stage1,
):
    other_action = 1 - target_action
    target_values = root_q2_state_values(pomdp, belief, target_action, stage1)
    other_values = root_q2_state_values(pomdp, belief, other_action, stage1)
    target_hidden_values = np.array([
        target_values[join_state(observed_s1, s2)]
        for s2 in range(NUM_S2)
    ])
    other_hidden_values = np.array([
        other_values[join_state(observed_s1, s2)]
        for s2 in range(NUM_S2)
    ])

    feasible_margin = float(np.max(target_hidden_values) - np.min(other_hidden_values))
    dominance_margin = float(np.min(target_hidden_values) - np.max(other_hidden_values))
    posterior_margin = float(belief @ target_values - belief @ other_values)

    return (
        feasible_margin > 1e-12,
        feasible_margin,
        dominance_margin > 1e-12,
        dominance_margin,
        posterior_margin > 1e-12,
        posterior_margin,
    )


def policy_tree_alpha_values(
    pomdp,
    root_action,
    continuation_actions,
):
    values = pomdp.rewards[:, root_action].copy()
    for state in range(NUM_STATES):
        future = 0.0
        for next_state in range(NUM_STATES):
            transition_prob = pomdp.transitions[root_action, state, next_state]
            for obs, continuation_action in enumerate(continuation_actions):
                obs_prob = pomdp.observations[root_action, next_state, obs]
                future += (
                    transition_prob
                    * obs_prob
                    * pomdp.rewards[next_state, continuation_action]
                )
        values[state] += pomdp.discount * future
    return values


def target_tree_from_alpha(root):
    return root.action, tuple(subplan.action for subplan in root.observation_subplans)


def joint_tree_diagnostics(
    pomdp,
    belief,
    observed_s1,
    target_tree,
):
    target_values = policy_tree_alpha_values(
        pomdp,
        target_tree[0],
        target_tree[1],
    )
    target_hidden_values = np.array([
        target_values[join_state(observed_s1, s2)]
        for s2 in range(NUM_S2)
    ])

    feasible_margins: list[float] = []
    dominance_margins: list[float] = []
    posterior_margins: list[float] = []
    for root_action in range(NUM_ACTIONS):
        for continuation_actions in product(range(NUM_ACTIONS), repeat=NUM_S1):
            candidate_tree = (root_action, tuple(continuation_actions))
            if candidate_tree == target_tree:
                continue
            candidate_values = policy_tree_alpha_values(
                pomdp,
                root_action,
                tuple(continuation_actions),
            )
            candidate_hidden_values = np.array([
                candidate_values[join_state(observed_s1, s2)]
                for s2 in range(NUM_S2)
            ])
            feasible_margins.append(float(np.max(target_hidden_values) - np.min(candidate_hidden_values)))
            dominance_margins.append(float(np.min(target_hidden_values - candidate_hidden_values)))
            posterior_margins.append(float(belief @ target_values - belief @ candidate_values))

    return (
        min(feasible_margins) > 1e-12,
        min(feasible_margins),
        min(dominance_margins) > 1e-12,
        min(dominance_margins),
        min(posterior_margins) > 1e-12,
        min(posterior_margins),
    )


def terminal_feasibility_for_history(
    pomdp,
    posterior,
    current_s1,
    target_action,
):
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
    root_all_feasible: bool
    min_root_margin: float
    root_all_dominant: bool
    min_root_dominance_margin: float
    root_all_posterior_optimal: bool
    min_root_posterior_margin: float
    joint_tree_all_feasible: bool
    min_joint_tree_margin: float
    joint_tree_all_dominant: bool
    min_joint_tree_dominance_margin: float
    joint_tree_all_posterior_optimal: bool
    min_joint_tree_posterior_margin: float
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


def analyze(seed, action_control, p_s1_matches_s2, initial_match_prob):
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
    root_feasible: list[bool] = []
    root_margins: list[float] = []
    root_dominant: list[bool] = []
    root_dominance_margins: list[float] = []
    root_posterior_optimal: list[bool] = []
    root_posterior_margins: list[float] = []
    joint_tree_feasible: list[bool] = []
    joint_tree_margins: list[float] = []
    joint_tree_dominant: list[bool] = []
    joint_tree_dominance_margins: list[float] = []
    joint_tree_posterior_optimal: list[bool] = []
    joint_tree_posterior_margins: list[float] = []
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

        (
            tree_is_feasible,
            tree_feasible_margin,
            tree_is_dominant,
            tree_dominance_margin,
            tree_is_posterior_optimal,
            tree_posterior_margin,
        ) = joint_tree_diagnostics(pomdp, b0, s1_t1, target_tree_from_alpha(root))
        joint_tree_feasible.append(tree_is_feasible)
        joint_tree_margins.append(tree_feasible_margin)
        joint_tree_dominant.append(tree_is_dominant)
        joint_tree_dominance_margins.append(tree_dominance_margin)
        joint_tree_posterior_optimal.append(tree_is_posterior_optimal)
        joint_tree_posterior_margins.append(tree_posterior_margin)

        (
            root_is_feasible,
            root_feasible_margin,
            root_is_dominant,
            root_dominance_margin,
            root_is_posterior_optimal,
            root_posterior_margin,
        ) = root_q2_diagnostics(pomdp, b0, s1_t1, target_a1, stage1)
        root_feasible.append(root_is_feasible)
        root_margins.append(root_feasible_margin)
        root_dominant.append(root_is_dominant)
        root_dominance_margins.append(root_dominance_margin)
        root_posterior_optimal.append(root_is_posterior_optimal)
        root_posterior_margins.append(root_posterior_margin)

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
        root_all_feasible=all(root_feasible),
        min_root_margin=min(root_margins),
        root_all_dominant=all(root_dominant),
        min_root_dominance_margin=min(root_dominance_margins),
        root_all_posterior_optimal=all(root_posterior_optimal),
        min_root_posterior_margin=min(root_posterior_margins),
        joint_tree_all_feasible=all(joint_tree_feasible),
        min_joint_tree_margin=min(joint_tree_margins),
        joint_tree_all_dominant=all(joint_tree_dominant),
        min_joint_tree_dominance_margin=min(joint_tree_dominance_margins),
        joint_tree_all_posterior_optimal=all(joint_tree_posterior_optimal),
        min_joint_tree_posterior_margin=min(joint_tree_posterior_margins),
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


def policy_label(pi2):
    return "{" + ", ".join(
        f"({h[0]},{h[1]}):a{a}" for h, a in zip(product(range(2), range(2)), pi2)
    ) + "}"


def print_instance(row):
    print(f"seed={row.seed}  action_control={row.action_control:.2f}  obs_info={row.p_s1_matches_s2:.2f}")
    print(f"  pi1*: s1=0 -> a{row.pi1[0]}, s1=1 -> a{row.pi1[1]}")
    print(f"  pi2*: {policy_label(row.pi2)}")
    print(f"  history dependent pi2*: {row.history_dependent}")
    print(f"  root Q2 all feasible: {row.root_all_feasible}, min feasible margin={row.min_root_margin:.4f}")
    print(f"  root Q2 all dominant: {row.root_all_dominant}, min dominance margin={row.min_root_dominance_margin:.4f}")
    print(f"  root Q2 all posterior-optimal: {row.root_all_posterior_optimal}, min posterior margin={row.min_root_posterior_margin:.4f}")
    print(f"  joint tree all feasible: {row.joint_tree_all_feasible}, min feasible margin={row.min_joint_tree_margin:.4f}")
    print(f"  joint tree all dominant: {row.joint_tree_all_dominant}, min dominance margin={row.min_joint_tree_dominance_margin:.4f}")
    print(f"  joint tree all posterior-optimal: {row.joint_tree_all_posterior_optimal}, min posterior margin={row.min_joint_tree_posterior_margin:.4f}")
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


def main():
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
    print(f"root Q2 all feasible: {sum(r.root_all_feasible for r in rows)}")
    print(f"root Q2 all dominant: {sum(r.root_all_dominant for r in rows)}")
    print(f"root Q2 all posterior-optimal: {sum(r.root_all_posterior_optimal for r in rows)}")
    print(f"joint tree all feasible: {sum(r.joint_tree_all_feasible for r in rows)}")
    print(f"joint tree all dominant: {sum(r.joint_tree_all_dominant for r in rows)}")
    print(f"joint tree all posterior-optimal: {sum(r.joint_tree_all_posterior_optimal for r in rows)}")
    print(f"terminal all feasible: {sum(r.terminal_all_feasible for r in rows)}")
    print(f"terminal all dominant: {sum(r.terminal_all_dominant for r in rows)}")
    print(f"terminal all posterior-optimal: {sum(r.terminal_all_posterior_optimal for r in rows)}")
    print(f"history-dependent and root Q2 dominant: {sum(r.history_dependent and r.root_all_dominant for r in rows)}")
    print(f"history-dependent and joint tree dominant: {sum(r.history_dependent and r.joint_tree_all_dominant for r in rows)}")
    print(f"history-dependent and joint tree posterior-optimal: {sum(r.history_dependent and r.joint_tree_all_posterior_optimal for r in rows)}")
    print(f"history-dependent and terminal feasible: {sum(r.history_dependent and r.terminal_all_feasible for r in rows)}")
    print(f"history-dependent and terminal dominant: {sum(r.history_dependent and r.terminal_all_dominant for r in rows)}")
    print(f"history-dependent and terminal posterior-optimal: {sum(r.history_dependent and r.terminal_all_posterior_optimal for r in rows)}")
    print(f"root Q2 dominant and terminal dominant: {sum(r.root_all_dominant and r.terminal_all_dominant for r in rows)}")
    print(f"joint tree dominant and terminal dominant: {sum(r.joint_tree_all_dominant and r.terminal_all_dominant for r in rows)}")
    print(f"root Q2 dominant and both components help: {sum(r.root_all_dominant and all(r.both_components_help) for r in rows)}")
    print(f"joint tree dominant and both components help: {sum(r.joint_tree_all_dominant and all(r.both_components_help) for r in rows)}")
    print(f"both components help for both s1 values: {sum(all(r.both_components_help) for r in rows)}")
    print()

    print("=== By action_control / obs_info ===")
    grouped: dict[tuple[float, float], list[InstanceSummary]] = defaultdict(list)
    for r in rows:
        grouped[(r.action_control, r.p_s1_matches_s2)].append(r)
    print("control  obsinfo  hist_dep  tree_dom  root_dom  term_dom  both_help_both_s1")
    for key in sorted(grouped):
        g = grouped[key]
        print(
            f"{key[0]:>7.2f}  {key[1]:>7.2f}  "
            f"{sum(r.history_dependent for r in g):>8}/{len(g):<3}  "
            f"{sum(r.joint_tree_all_dominant for r in g):>8}/{len(g):<3}  "
            f"{sum(r.root_all_dominant for r in g):>8}/{len(g):<3}  "
            f"{sum(r.terminal_all_dominant for r in g):>8}/{len(g):<3}  "
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

    print("\n=== Root Q2 dominant candidates ===")
    root_dominant = [r for r in rows if r.root_all_dominant]
    root_dominant.sort(key=lambda r: r.min_root_dominance_margin, reverse=True)
    if not root_dominant:
        print("No root Q2 dominant cases found.")
    else:
        for r in root_dominant[:8]:
            print_instance(r)

    print("\n=== Joint tree dominant candidates ===")
    joint_tree_dominant = [r for r in rows if r.joint_tree_all_dominant]
    joint_tree_dominant.sort(key=lambda r: r.min_joint_tree_dominance_margin, reverse=True)
    if not joint_tree_dominant:
        print("No joint tree dominant cases found.")
    else:
        for r in joint_tree_dominant[:8]:
            print_instance(r)

    print("\n=== Interesting but messy: history-dependent pi2*, terminal posterior-optimal ===")
    interesting = [r for r in rows if r.history_dependent and r.terminal_all_posterior_optimal]
    interesting.sort(key=lambda r: r.min_terminal_posterior_margin, reverse=True)
    for r in interesting[:8]:
        print_instance(r)


if __name__ == "__main__":
    main()
