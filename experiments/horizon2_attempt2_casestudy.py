from __future__ import annotations

from dataclasses import dataclass

from horizon2_joint_policy_experiments import (
    analyze,
    build_action_dependent_factored_pomdp,
    initial_belief_given_s1,
    join_state,
    policy_tree_alpha_values,
)


INITIAL_MATCH_PROB = 0.70
SEEDS = range(80)
ACTION_CONTROLS = [0.55, 0.70, 0.85, 0.95]
OBS_INFOS = [0.55, 0.70, 0.85, 0.95]


@dataclass(frozen=True)
class FixedSequenceCase:
    seed: int
    action_control: float
    obs_info: float
    first_action: int
    second_action: int
    joint_margin: float
    root_margin: float
    terminal_margin: float


def is_fixed_sequence(row):
    return row.pi1[0] == row.pi1[1] and len(set(row.pi2)) == 1


def fixed_sequence_cases():
    cases: list[FixedSequenceCase] = []
    for seed in SEEDS:
        for action_control in ACTION_CONTROLS:
            for obs_info in OBS_INFOS:
                row = analyze(seed, action_control, obs_info, INITIAL_MATCH_PROB)
                if not is_fixed_sequence(row):
                    continue
                if row.min_joint_tree_dominance_margin <= 0:
                    continue
                if row.min_root_dominance_margin >= 0:
                    continue
                if row.min_terminal_dominance_margin >= 0:
                    continue

                cases.append(
                    FixedSequenceCase(
                        seed=seed,
                        action_control=action_control,
                        obs_info=obs_info,
                        first_action=row.pi1[0],
                        second_action=row.pi2[0],
                        joint_margin=row.min_joint_tree_dominance_margin,
                        root_margin=row.min_root_dominance_margin,
                        terminal_margin=row.min_terminal_dominance_margin,
                    )
                )
    return sorted(cases, key=lambda case: case.joint_margin, reverse=True)


def sequence_value(pomdp, initial_s1, first_action, second_action):
    belief = initial_belief_given_s1(initial_s1, INITIAL_MATCH_PROB)
    values = policy_tree_alpha_values(pomdp, first_action, (second_action, second_action))
    return float(belief @ values)


def hidden_min_gap(
    pomdp,
    initial_s1,
    target_first_action,
    target_second_action,
    candidate_first_action,
    candidate_second_action,
):
    target_values = policy_tree_alpha_values(
        pomdp,
        target_first_action,
        (target_second_action, target_second_action),
    )
    candidate_values = policy_tree_alpha_values(
        pomdp,
        candidate_first_action,
        (candidate_second_action, candidate_second_action),
    )
    return min(
        target_values[join_state(initial_s1, s2)]
        - candidate_values[join_state(initial_s1, s2)]
        for s2 in range(2)
    )


def print_ranked_cases(cases, limit=12):
    print("=== Attempt 2 fixed-sequence case-study candidates ===")
    print("Selection rule: fixed sequence, joint margin > 0, root margin < 0, terminal margin < 0")
    print(f"cases: {len(cases)}")
    print()
    print("rank  seed  control  obsinfo  target  joint_margin  root_margin  terminal_margin")
    for index, case in enumerate(cases[:limit], start=1):
        print(
            f"{index:>4}  "
            f"{case.seed:>4}  "
            f"{case.action_control:>7.2f}  "
            f"{case.obs_info:>7.2f}  "
            f"(a{case.first_action},a{case.second_action})  "
            f"{case.joint_margin:>+12.4f}  "
            f"{case.root_margin:>+11.4f}  "
            f"{case.terminal_margin:>+15.4f}"
        )


def print_detailed_case(case):
    pomdp = build_action_dependent_factored_pomdp(
        case.seed,
        action_control=case.action_control,
        p_s1_matches_s2=case.obs_info,
    )

    print()
    print("=== Detailed fixed-sequence case study ===")
    print(
        f"seed={case.seed}  "
        f"action_control={case.action_control:.2f}  "
        f"obs_info={case.obs_info:.2f}"
    )
    print(f"target sequence: (a{case.first_action}, a{case.second_action})")
    print(f"joint two-period margin: {case.joint_margin:+.4f}")
    print(f"root-only margin: {case.root_margin:+.4f}")
    print(f"terminal margin: {case.terminal_margin:+.4f}")
    print()

    print("rewards:")
    for s1 in range(2):
        for s2 in range(2):
            state = join_state(s1, s2)
            print(
                f"  R(s1={s1}, s2={s2}, a0)={pomdp.rewards[state, 0]:+.3f}  "
                f"R(s1={s1}, s2={s2}, a1)={pomdp.rewards[state, 1]:+.3f}"
            )
    print()

    print("fixed two-action sequence comparisons:")
    for initial_s1 in range(2):
        target_value = sequence_value(
            pomdp,
            initial_s1,
            case.first_action,
            case.second_action,
        )
        print(f"initial s1={initial_s1}: target value={target_value:+.6f}")

        best_switch = (float("-inf"), None)
        for first_action in range(2):
            for second_action in range(2):
                value = sequence_value(pomdp, initial_s1, first_action, second_action)
                belief_gap = target_value - value
                min_gap = hidden_min_gap(
                    pomdp,
                    initial_s1,
                    case.first_action,
                    case.second_action,
                    first_action,
                    second_action,
                )
                print(
                    f"  candidate=(a{first_action},a{second_action})  "
                    f"value={value:+.6f}  "
                    f"belief_gap={belief_gap:+.6f}  "
                    f"hidden_min_gap={min_gap:+.6f}"
                )
                if first_action != case.first_action and value > best_switch[0]:
                    best_switch = (value, (first_action, second_action))

        best_value, best_actions = best_switch
        print(
            f"  best first-action switch=(a{best_actions[0]},a{best_actions[1]})  "
            f"value={best_value:+.6f}  "
            f"gap={target_value - best_value:+.6f}"
        )
        print()


def main():
    cases = fixed_sequence_cases()
    print_ranked_cases(cases)
    if cases:
        print_detailed_case(cases[0])


if __name__ == "__main__":
    main()
