from __future__ import annotations

from horizon2_joint_policy_experiments import (
    NUM_ACTIONS,
    analyze,
    build_action_dependent_factored_pomdp,
    initial_belief_given_s1,
    policy_label,
    policy_tree_alpha_values,
)


TOL = 1e-10


def best_memoryless_plan_value(pomdp, belief):
    best_value = float("-inf")
    best_plan = None
    for first_action in range(NUM_ACTIONS):
        for second_action in range(NUM_ACTIONS):
            alpha = policy_tree_alpha_values(
                pomdp,
                first_action,
                (second_action, second_action),
            )
            value = float(belief @ alpha)
            if value > best_value:
                best_value = value
                best_plan = (first_action, second_action)
    return best_plan, best_value


def target_tree_value(pomdp, belief, row, initial_s1: int) -> tuple[tuple[int, tuple[int, int]], float]:
    continuation = (
        row.pi2[2 * initial_s1],
        row.pi2[2 * initial_s1 + 1],
    )
    tree = (row.pi1[initial_s1], continuation)
    alpha = policy_tree_alpha_values(pomdp, tree[0], tree[1])
    return tree, float(belief @ alpha)


def main() -> None:
    seeds = range(80)
    action_controls = [0.55, 0.70, 0.85, 0.95]
    obs_infos = [0.55, 0.70, 0.85, 0.95]
    initial_match_prob = 0.70

    rows = []
    for action_control in action_controls:
        for obs_info in obs_infos:
            for seed in seeds:
                row = analyze(seed, action_control, obs_info, initial_match_prob)
                if row.history_dependent and row.joint_tree_all_dominant:
                    rows.append(row)

    print("=== Memoryless two-action plan vs history-based joint tree ===")
    print(f"history-dependent joint-tree dominant cases: {len(rows)}")
    print()

    matches_both_initial_states = 0
    matches_history_dependent_initial_states = 0
    history_dependent_initial_states = 0

    case_results = []
    for row in rows:
        pomdp = build_action_dependent_factored_pomdp(
            row.seed,
            action_control=row.action_control,
            p_s1_matches_s2=row.p_s1_matches_s2,
        )

        per_initial_state = []
        for initial_s1 in range(2):
            belief = initial_belief_given_s1(initial_s1, initial_match_prob)
            tree, tree_value = target_tree_value(pomdp, belief, row, initial_s1)
            memoryless_plan, memoryless_value = best_memoryless_plan_value(pomdp, belief)
            gap = tree_value - memoryless_value
            uses_history = tree[1][0] != tree[1][1]

            per_initial_state.append(
                {
                    "initial_s1": initial_s1,
                    "tree": tree,
                    "tree_value": tree_value,
                    "memoryless_plan": memoryless_plan,
                    "memoryless_value": memoryless_value,
                    "gap": gap,
                    "uses_history": uses_history,
                    "matches": abs(gap) <= TOL,
                }
            )

        if all(item["matches"] for item in per_initial_state):
            matches_both_initial_states += 1

        for item in per_initial_state:
            if item["uses_history"]:
                history_dependent_initial_states += 1
                if item["matches"]:
                    matches_history_dependent_initial_states += 1

        worst_gap = min(item["gap"] for item in per_initial_state)
        best_gap = max(item["gap"] for item in per_initial_state)
        case_results.append((worst_gap, best_gap, row, per_initial_state))

    print(f"cases where memoryless matches both initial states: {matches_both_initial_states}/{len(rows)}")
    print(
        "history-dependent initial-state trees matched by memoryless: "
        f"{matches_history_dependent_initial_states}/{history_dependent_initial_states}"
    )
    print()

    case_results.sort(key=lambda item: item[0], reverse=True)
    for index, (_, _, row, per_initial_state) in enumerate(case_results, start=1):
        print(f"--- Case {index} ---")
        print(
            f"seed={row.seed}  "
            f"action_control={row.action_control:.2f}  "
            f"obs_info={row.p_s1_matches_s2:.2f}"
        )
        print(f"pi1*: s1=0 -> a{row.pi1[0]}, s1=1 -> a{row.pi1[1]}")
        print(f"pi2*: {policy_label(row.pi2)}")
        print(f"joint tree dominance margin: {row.min_joint_tree_dominance_margin:+.4f}")
        for item in per_initial_state:
            tree = item["tree"]
            memoryless = item["memoryless_plan"]
            print(
                f"  initial s1={item['initial_s1']}: "
                f"target tree=(a{tree[0]}, a2(0)=a{tree[1][0]}, a2(1)=a{tree[1][1]}), "
                f"best memoryless=(a{memoryless[0]}, then a{memoryless[1]}), "
                f"gap={item['gap']:+.6f}, "
                f"uses_history={item['uses_history']}, "
                f"matches={item['matches']}"
            )
        print()


if __name__ == "__main__":
    main()
