from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import product

import numpy as np

from horizon2_joint_policy_experiments import (
    NUM_ACTIONS,
    NUM_S1,
    NUM_S2,
    analyze,
    build_action_dependent_factored_pomdp,
    initial_belief_given_s1,
    join_state,
    policy_label,
    policy_tree_alpha_values,
)


INITIAL_MATCH_PROB = 0.70
TOL = 1e-12


@dataclass(frozen=True)
class LocalMargins:
    global_margin: float
    one_deviation_margin: float
    root_deviation_margin: float
    continuation_deviation_margin: float
    different_root_any_continuation_margin: float
    same_root_any_continuation_margin: float
    same_continuation_root_margin: float
    posterior_margin: float
    worst_tree: tuple[int, tuple[int, int]]
    worst_one_deviation: tuple[int, tuple[int, int]]


def hamming_distance(left, right):
    return int(left[0] != right[0]) + sum(
        int(left[1][obs] != right[1][obs]) for obs in range(NUM_S1)
    )


def hidden_slice_values(values, observed_s1):
    return np.array([values[join_state(observed_s1, s2)] for s2 in range(NUM_S2)])


def tree_margins(pomdp, belief, observed_s1, target_tree):
    target_values = policy_tree_alpha_values(pomdp, target_tree[0], target_tree[1])
    target_hidden = hidden_slice_values(target_values, observed_s1)

    global_margins: list[tuple[float, tuple[int, tuple[int, int]]]] = []
    one_dev_margins: list[tuple[float, tuple[int, tuple[int, int]]]] = []
    root_dev_margins: list[float] = []
    cont_dev_margins: list[float] = []
    different_root_margins: list[float] = []
    same_root_margins: list[float] = []
    same_cont_root_margins: list[float] = []
    posterior_margins: list[float] = []

    for root_action in range(NUM_ACTIONS):
        for continuation in product(range(NUM_ACTIONS), repeat=NUM_S1):
            candidate_tree = (root_action, tuple(continuation))
            if candidate_tree == target_tree:
                continue
            candidate_values = policy_tree_alpha_values(pomdp, root_action, tuple(continuation))
            candidate_hidden = hidden_slice_values(candidate_values, observed_s1)
            margin = float(np.min(target_hidden - candidate_hidden))
            global_margins.append((margin, candidate_tree))
            posterior_margins.append(float(belief @ target_values - belief @ candidate_values))

            distance = hamming_distance(target_tree, candidate_tree)
            if distance == 1:
                one_dev_margins.append((margin, candidate_tree))
                if root_action != target_tree[0]:
                    root_dev_margins.append(margin)
                else:
                    cont_dev_margins.append(margin)
            if root_action != target_tree[0]:
                different_root_margins.append(margin)
            if root_action == target_tree[0]:
                same_root_margins.append(margin)
            if tuple(continuation) == target_tree[1]:
                same_cont_root_margins.append(margin)

    worst_margin, worst_tree = min(global_margins, key=lambda item: item[0])
    worst_one_margin, worst_one_tree = min(one_dev_margins, key=lambda item: item[0])
    return LocalMargins(
        global_margin=worst_margin,
        one_deviation_margin=worst_one_margin,
        root_deviation_margin=min(root_dev_margins),
        continuation_deviation_margin=min(cont_dev_margins),
        different_root_any_continuation_margin=min(different_root_margins),
        same_root_any_continuation_margin=min(same_root_margins),
        same_continuation_root_margin=min(same_cont_root_margins),
        posterior_margin=min(posterior_margins),
        worst_tree=worst_tree,
        worst_one_deviation=worst_one_tree,
    )


def target_tree_for_initial_state(row, initial_s1):
    return (
        row.pi1[initial_s1],
        (
            row.pi2[2 * initial_s1],
            row.pi2[2 * initial_s1 + 1],
        ),
    )


def sign(value):
    return value > TOL


def summarize_boolean_rule(name, rows, predicate):
    positives = [row for row in rows if predicate(row)]
    true_positives = sum(row["joint_tree_dominant"] and predicate(row) for row in rows)
    false_positives = sum((not row["joint_tree_dominant"]) and predicate(row) for row in rows)
    false_negatives = sum(row["joint_tree_dominant"] and not predicate(row) for row in rows)
    print(
        f"{name:<38} "
        f"count={len(positives):>4}  "
        f"TP={true_positives:>4}  FP={false_positives:>4}  FN={false_negatives:>4}"
    )


def main():
    seeds = range(80)
    action_controls = [0.55, 0.70, 0.85, 0.95]
    obs_infos = [0.55, 0.70, 0.85, 0.95]

    instance_rows = []
    local_rows = []
    for action_control in action_controls:
        for obs_info in obs_infos:
            for seed in seeds:
                row = analyze(seed, action_control, obs_info, INITIAL_MATCH_PROB)
                pomdp = build_action_dependent_factored_pomdp(
                    seed,
                    action_control=action_control,
                    p_s1_matches_s2=obs_info,
                )
                local_by_initial = []
                for initial_s1 in range(NUM_S1):
                    belief = initial_belief_given_s1(initial_s1, INITIAL_MATCH_PROB)
                    target_tree = target_tree_for_initial_state(row, initial_s1)
                    margins = tree_margins(pomdp, belief, initial_s1, target_tree)
                    local = {
                        "seed": seed,
                        "action_control": action_control,
                        "obs_info": obs_info,
                        "initial_s1": initial_s1,
                        "pi1": row.pi1,
                        "pi2": row.pi2,
                        "target_tree": target_tree,
                        "history_dependent": row.history_dependent,
                        "global_dominant": sign(margins.global_margin),
                        "one_deviation_dominant": sign(margins.one_deviation_margin),
                        "root_deviation_dominant": sign(margins.root_deviation_margin),
                        "continuation_deviation_dominant": sign(margins.continuation_deviation_margin),
                        "different_root_any_continuation_dominant": sign(
                            margins.different_root_any_continuation_margin
                        ),
                        "same_root_any_continuation_dominant": sign(margins.same_root_any_continuation_margin),
                        "same_continuation_root_dominant": sign(margins.same_continuation_root_margin),
                        "posterior_optimal": sign(margins.posterior_margin),
                        "global_margin": margins.global_margin,
                        "one_deviation_margin": margins.one_deviation_margin,
                        "root_deviation_margin": margins.root_deviation_margin,
                        "continuation_deviation_margin": margins.continuation_deviation_margin,
                        "different_root_any_continuation_margin": margins.different_root_any_continuation_margin,
                        "same_root_any_continuation_margin": margins.same_root_any_continuation_margin,
                        "same_continuation_root_margin": margins.same_continuation_root_margin,
                        "posterior_margin": margins.posterior_margin,
                        "worst_tree": margins.worst_tree,
                        "worst_one_deviation": margins.worst_one_deviation,
                    }
                    local_rows.append(local)
                    local_by_initial.append(local)

                instance_rows.append(
                    {
                        "summary": row,
                        "joint_tree_dominant": all(item["global_dominant"] for item in local_by_initial),
                        "one_deviation_dominant": all(item["one_deviation_dominant"] for item in local_by_initial),
                        "root_deviation_dominant": all(item["root_deviation_dominant"] for item in local_by_initial),
                        "continuation_deviation_dominant": all(
                            item["continuation_deviation_dominant"] for item in local_by_initial
                        ),
                        "same_root_any_continuation_dominant": all(
                            item["same_root_any_continuation_dominant"] for item in local_by_initial
                        ),
                        "different_root_any_continuation_dominant": all(
                            item["different_root_any_continuation_dominant"] for item in local_by_initial
                        ),
                        "same_continuation_root_dominant": all(
                            item["same_continuation_root_dominant"] for item in local_by_initial
                        ),
                        "posterior_optimal": all(item["posterior_optimal"] for item in local_by_initial),
                        "locals": local_by_initial,
                    }
                )

    print("=== Attempt 2: neighbor-deviation patterns ===")
    print(f"instances: {len(instance_rows)}")
    print(f"initial-state local problems: {len(local_rows)}")
    print()

    print("=== Candidate rules vs joint whole-policy dominance ===")
    summarize_boolean_rule("one-deviation dominance", instance_rows, lambda r: r["one_deviation_dominant"])
    summarize_boolean_rule("root-deviation dominance", instance_rows, lambda r: r["root_deviation_dominant"])
    summarize_boolean_rule("continuation-deviation dominance", instance_rows, lambda r: r["continuation_deviation_dominant"])
    summarize_boolean_rule("root AND continuation deviations", instance_rows, lambda r: r["root_deviation_dominant"] and r["continuation_deviation_dominant"])
    summarize_boolean_rule("same-root all continuations", instance_rows, lambda r: r["same_root_any_continuation_dominant"])
    summarize_boolean_rule("different-root all continuations", instance_rows, lambda r: r["different_root_any_continuation_dominant"])
    summarize_boolean_rule("same-continuation root flip", instance_rows, lambda r: r["same_continuation_root_dominant"])
    summarize_boolean_rule("posterior optimal against all trees", instance_rows, lambda r: r["posterior_optimal"])
    print()

    print("=== Local equivalence checks ===")
    local_global = sum(item["global_dominant"] for item in local_rows)
    local_one = sum(item["one_deviation_dominant"] for item in local_rows)
    local_mismatch = [
        item for item in local_rows
        if item["one_deviation_dominant"] != item["global_dominant"]
    ]
    print(f"global dominant local problems: {local_global}/{len(local_rows)}")
    print(f"one-deviation dominant local problems: {local_one}/{len(local_rows)}")
    print(f"one-deviation/global mismatches: {len(local_mismatch)}")
    if local_mismatch:
        print("first mismatches:")
        for item in local_mismatch[:8]:
            print(
                f"  seed={item['seed']} control={item['action_control']:.2f} "
                f"obs={item['obs_info']:.2f} initial_s1={item['initial_s1']} "
                f"target={item['target_tree']} global_margin={item['global_margin']:+.4f} "
                f"one_dev_margin={item['one_deviation_margin']:+.4f} "
                f"worst={item['worst_tree']} worst_one={item['worst_one_deviation']}"
            )
    print()

    print("=== Binding deviations among locally dominant history-dependent cases ===")
    history_dominant_locals = [
        item for item in local_rows
        if item["history_dependent"] and item["global_dominant"]
    ]
    binding_counter = Counter()
    for item in history_dominant_locals:
        margins = {
            "root flip only": item["same_continuation_root_margin"],
            "one continuation flip": item["continuation_deviation_margin"],
            "different root, any continuation": item["different_root_any_continuation_margin"],
            "same root, any continuation": item["same_root_any_continuation_margin"],
            "any tree": item["global_margin"],
        }
        binding_counter[min(margins, key=margins.get)] += 1
    print(f"history-dependent locally dominant problems: {len(history_dominant_locals)}")
    for label, count in binding_counter.most_common():
        print(f"{label:<30} {count}")
    print()

    print("=== History-dependent joint-dominant instance shapes ===")
    hist_joint = [
        item for item in instance_rows
        if item["summary"].history_dependent and item["joint_tree_dominant"]
    ]
    print(f"instances: {len(hist_joint)}")
    print("policy shapes:")
    for (pi1, pi2), count in Counter((item["summary"].pi1, item["summary"].pi2) for item in hist_joint).most_common():
        print(f"  {count:>2}  pi1={pi1}, pi2={policy_label(pi2)}")
    print()

    print("=== Strongest history-dependent joint-dominant examples ===")
    hist_joint.sort(key=lambda item: item["summary"].min_joint_tree_dominance_margin, reverse=True)
    for index, item in enumerate(hist_joint[:10], start=1):
        row = item["summary"]
        print(f"--- Case {index} ---")
        print(
            f"seed={row.seed} action_control={row.action_control:.2f} "
            f"obs_info={row.p_s1_matches_s2:.2f}"
        )
        print(f"pi1={row.pi1}, pi2={policy_label(row.pi2)}")
        print(f"joint margin={row.min_joint_tree_dominance_margin:+.4f}")
        for local in item["locals"]:
            print(
                f"  initial_s1={local['initial_s1']} target={local['target_tree']} "
                f"global={local['global_margin']:+.4f} one-dev={local['one_deviation_margin']:+.4f} "
                f"root={local['root_deviation_margin']:+.4f} cont={local['continuation_deviation_margin']:+.4f} "
                f"diff-root={local['different_root_any_continuation_margin']:+.4f} "
                f"worst={local['worst_tree']}"
            )
        print()


if __name__ == "__main__":
    main()
