from __future__ import annotations

from collections import Counter

from horizon2_joint_policy_experiments import (
    analyze,
    build_action_dependent_factored_pomdp,
    policy_label,
)


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

    rows.sort(
        key=lambda row: (
            row.min_joint_tree_dominance_margin,
            row.min_root_posterior_margin,
            row.min_terminal_posterior_margin,
        ),
        reverse=True,
    )

    print("=== History-dependent joint-tree dominant cases ===")
    print(f"cases: {len(rows)}")
    print()

    print("=== By action_control / obs_info ===")
    grouped = Counter((row.action_control, row.p_s1_matches_s2) for row in rows)
    for key in sorted(grouped):
        print(f"action_control={key[0]:.2f}  obs_info={key[1]:.2f}  count={grouped[key]}")
    print()

    for index, row in enumerate(rows, start=1):
        pomdp = build_action_dependent_factored_pomdp(
            row.seed,
            action_control=row.action_control,
            p_s1_matches_s2=row.p_s1_matches_s2,
        )

        print(f"--- Case {index} ---")
        print(
            f"seed={row.seed}  "
            f"action_control={row.action_control:.2f}  "
            f"obs_info={row.p_s1_matches_s2:.2f}"
        )
        print(f"pi1*: s1=0 -> a{row.pi1[0]}, s1=1 -> a{row.pi1[1]}")
        print(f"pi2*: {policy_label(row.pi2)}")
        print(f"joint tree dominance margin: {row.min_joint_tree_dominance_margin:+.4f}")
        print(f"joint tree posterior margin: {row.min_joint_tree_posterior_margin:+.4f}")
        print(f"root Q2 dominance margin: {row.min_root_dominance_margin:+.4f}")
        print(f"terminal dominance margin: {row.min_terminal_dominance_margin:+.4f}")
        print(f"terminal posterior margin: {row.min_terminal_posterior_margin:+.4f}")
        print(
            "both components help: "
            f"s1=0 -> {row.both_components_help[0]}, "
            f"s1=1 -> {row.both_components_help[1]}"
        )
        print("rewards:")
        for s1 in range(2):
            for s2 in range(2):
                state = 2 * s1 + s2
                r0, r1 = pomdp.rewards[state]
                print(f"  R(s1={s1}, s2={s2}, a0)={r0:+.3f}  R(s1={s1}, s2={s2}, a1)={r1:+.3f}")
        print()


if __name__ == "__main__":
    main()
