from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np

sys.path.append(os.path.dirname(__file__))

from two_period_joint_policy_experiments import (
    NUM_ACTIONS,
    NUM_S1,
    build_action_dependent_factored_pomdp,
    initial_belief_given_s1,
    join_state,
    policy_tree_alpha_values,
)
from find_t2_dse_failure_unteachable_case import (
    ALLOWED_POLICY_ROWS,
    allowed_attacker_policies,
    induced_b,
)
from find_t2_dse_inducible_observed_model_case import (
    b_index,
    observed_transition,
    observed_value,
    prior_b_vector,
)


Tree = tuple[int, tuple[int, int]]
TREES: tuple[Tree, ...] = tuple(
    (root_action, tuple(continuation))
    for root_action in range(NUM_ACTIONS)
    for continuation in product(range(NUM_ACTIONS), repeat=NUM_S1)
)
TOL = 1e-10


@dataclass(frozen=True)
class AttackerCase3:
    seed: int
    control: float
    obs_info: float
    pomdp: object
    target_tree: Tree
    attacker: np.ndarray
    attacked_b: np.ndarray
    original_evaluations: tuple["ObservedT2PDEvaluation", ...]
    attacked_evaluations: tuple["ObservedT2PDEvaluation", ...]
    dse_margin: float
    dse_witness_s1: int
    dse_witness_s2: int
    dse_witness_tree: Tree


@dataclass(frozen=True)
class ObservedT2PDEvaluation:
    values: dict[Tree, float]
    target_continuation_margin: float
    competing_root_margin: float
    margin: float
    passes: bool
    optimal_continuations: dict[int, tuple[Tree, ...]]


@dataclass(frozen=True)
class T2PDCaseEvaluation:
    target_tree: Tree
    target_value: float
    target_continuation_margin: float
    pd_root_margin: float
    pd_margin: float
    t2_pd_passes: bool
    t2_dse_margin: float
    t2_dse_passes: bool
    dse_witness_tree: Tree
    dse_witness_s2: int
    values: dict[Tree, float]
    hidden_values: dict[Tree, np.ndarray]
    optimal_continuations: dict[int, tuple[Tree, ...]]


def tree_label(tree: Tree) -> str:
    root_action, continuation = tree
    return (
        f"(a{root_action}, "
        f"pi2(s1'=0)=a{continuation[0]}, "
        f"pi2(s1'=1)=a{continuation[1]})"
    )


def hidden_slice(alpha: np.ndarray, observed_s1: int) -> np.ndarray:
    return np.array(
        [alpha[join_state(observed_s1, s2)] for s2 in range(2)],
        dtype=float,
    )


def evaluate_t2_pd(
    pomdp,
    belief: np.ndarray,
    observed_s1: int,
    target_tree: Tree,
) -> T2PDCaseEvaluation:
    """Evaluate whether T2-PD passes while the pointwise T2-DSE test fails."""
    alphas = {
        tree: policy_tree_alpha_values(pomdp, tree[0], tree[1])
        for tree in TREES
    }
    values = {tree: float(belief @ alpha) for tree, alpha in alphas.items()}
    hidden_values = {
        tree: hidden_slice(alpha, observed_s1) for tree, alpha in alphas.items()
    }

    target_value = values[target_tree]
    target_root = target_tree[0]

    same_root_competitors = [
        tree for tree in TREES if tree[0] == target_root and tree != target_tree
    ]
    target_continuation_margin = min(
        target_value - values[tree] for tree in same_root_competitors
    )

    optimal_continuations: dict[int, tuple[Tree, ...]] = {}
    competing_root_margins: list[float] = []
    for root_action in range(NUM_ACTIONS):
        conditional_trees = [tree for tree in TREES if tree[0] == root_action]
        best_value = max(values[tree] for tree in conditional_trees)
        optimal_continuations[root_action] = tuple(
            tree
            for tree in conditional_trees
            if abs(values[tree] - best_value) <= TOL
        )
        if root_action != target_root:
            competing_root_margins.append(target_value - best_value)

    pd_root_margin = min(competing_root_margins)
    pd_margin = min(target_continuation_margin, pd_root_margin)
    t2_pd_passes = pd_margin > TOL

    pointwise_comparisons: list[tuple[float, Tree, int]] = []
    target_hidden = hidden_values[target_tree]
    for tree in TREES:
        if tree == target_tree:
            continue
        for hidden_s2 in range(2):
            pointwise_comparisons.append(
                (
                    float(target_hidden[hidden_s2] - hidden_values[tree][hidden_s2]),
                    tree,
                    hidden_s2,
                )
            )

    t2_dse_margin, dse_witness_tree, dse_witness_s2 = min(
        pointwise_comparisons,
        key=lambda item: item[0],
    )

    return T2PDCaseEvaluation(
        target_tree=target_tree,
        target_value=target_value,
        target_continuation_margin=target_continuation_margin,
        pd_root_margin=pd_root_margin,
        pd_margin=pd_margin,
        t2_pd_passes=t2_pd_passes,
        t2_dse_margin=t2_dse_margin,
        t2_dse_passes=t2_dse_margin > TOL,
        dse_witness_tree=dse_witness_tree,
        dse_witness_s2=dse_witness_s2,
        values=values,
        hidden_values=hidden_values,
        optimal_continuations=optimal_continuations,
    )


def print_description(
    *,
    seed: int,
    action_control: float,
    observation_information: float,
    initial_match_probability: float,
    observed_s1: int,
    pomdp,
    description: T2PDCaseEvaluation,
) -> None:
    print("=== T2-PD passes / T2-DSE fails case ===")
    print(f"seed: {seed}")
    print(f"action control: {action_control:.2f}")
    print(f"observation information: {observation_information:.2f}")
    print(f"initial match probability: {initial_match_probability:.2f}")
    print(f"initial observed state: s1={observed_s1}")
    print(f"target tree: {tree_label(description.target_tree)}")
    print()

    print("rewards R(s1,s2,a):")
    print("s1  s2       a0       a1")
    for s1 in range(2):
        for s2 in range(2):
            state = join_state(s1, s2)
            print(
                f" {s1}   {s2}  "
                f"{pomdp.rewards[state, 0]:+8.3f} "
                f"{pomdp.rewards[state, 1]:+8.3f}"
            )
    print()

    print("belief-level values in the induced model:")
    print("root  continuation(s1'=0,s1'=1)      value")
    for tree in TREES:
        marker = "  target" if tree == description.target_tree else ""
        print(
            f" a{tree[0]}        (a{tree[1][0]},a{tree[1][1]})"
            f"             {description.values[tree]:+10.6f}{marker}"
        )
    print()

    print("optimal continuation(s) conditional on each first action:")
    for root_action, trees in description.optimal_continuations.items():
        labels = ", ".join(tree_label(tree) for tree in trees)
        print(f"  a{root_action}: {labels}")
    print()

    print("T2-PD:")
    print(
        "  unique target-continuation margin: "
        f"{description.target_continuation_margin:+.6f}"
    )
    print(f"  competing-root margin: {description.pd_root_margin:+.6f}")
    print(f"  overall margin: {description.pd_margin:+.6f}")
    print(f"  result: {'PASSES' if description.t2_pd_passes else 'FAILS'}")
    print()

    witness = description.dse_witness_tree
    target_hidden = description.hidden_values[description.target_tree]
    witness_hidden = description.hidden_values[witness]
    print("T2-DSE pointwise test:")
    print(f"  worst margin: {description.t2_dse_margin:+.6f}")
    print(f"  witness hidden state: (s1,s2)=({observed_s1},{description.dse_witness_s2})")
    print(f"  witness tree: {tree_label(witness)}")
    print(
        "  target/witness values at that hidden state: "
        f"{target_hidden[description.dse_witness_s2]:+.6f} / "
        f"{witness_hidden[description.dse_witness_s2]:+.6f}"
    )
    print(f"  result: {'PASSES' if description.t2_dse_passes else 'FAILS'}")


def observed_reward(pomdp, b: np.ndarray, s1: int, action: int) -> float:
    probability_s2_1 = b[b_index(s1, action)]
    reward_s2_0 = pomdp.rewards[join_state(s1, 0), action]
    reward_s2_1 = pomdp.rewards[join_state(s1, 1), action]
    return float(
        (1.0 - probability_s2_1) * reward_s2_0
        + probability_s2_1 * reward_s2_1
    )


def observed_tree_value(
    pomdp,
    tree: Tree,
    initial_s1: int,
    b: np.ndarray,
) -> float:
    root_action, continuation = tree
    value = observed_reward(pomdp, b, initial_s1, root_action)
    transition = observed_transition(pomdp, root_action)
    for next_s1 in range(NUM_S1):
        value += (
            pomdp.discount
            * transition[next_s1]
            * observed_reward(pomdp, b, next_s1, continuation[next_s1])
        )
    return float(value)


def evaluate_observed_t2_pd(
    pomdp,
    b: np.ndarray,
    initial_s1: int,
    target_tree: Tree,
) -> ObservedT2PDEvaluation:
    values = {
        tree: observed_tree_value(pomdp, tree, initial_s1, b)
        for tree in TREES
    }
    target_value = values[target_tree]
    target_root = target_tree[0]
    same_root_competitors = [
        tree for tree in TREES if tree[0] == target_root and tree != target_tree
    ]
    target_continuation_margin = min(
        target_value - values[tree] for tree in same_root_competitors
    )

    optimal_continuations: dict[int, tuple[Tree, ...]] = {}
    competing_root_margins: list[float] = []
    for root_action in range(NUM_ACTIONS):
        conditional_trees = [tree for tree in TREES if tree[0] == root_action]
        best_value = max(values[tree] for tree in conditional_trees)
        optimal_continuations[root_action] = tuple(
            tree
            for tree in conditional_trees
            if abs(values[tree] - best_value) <= TOL
        )
        if root_action != target_root:
            competing_root_margins.append(target_value - best_value)

    competing_root_margin = min(competing_root_margins)
    margin = min(target_continuation_margin, competing_root_margin)
    return ObservedT2PDEvaluation(
        values=values,
        target_continuation_margin=target_continuation_margin,
        competing_root_margin=competing_root_margin,
        margin=margin,
        passes=margin > TOL,
        optimal_continuations=optimal_continuations,
    )


def pointwise_t2_dse_witness(pomdp, target_tree: Tree):
    alphas = {
        tree: policy_tree_alpha_values(pomdp, tree[0], tree[1])
        for tree in TREES
    }
    target_alpha = alphas[target_tree]
    comparisons = []
    for tree in TREES:
        if tree == target_tree:
            continue
        for s1 in range(NUM_S1):
            for s2 in range(2):
                state = join_state(s1, s2)
                comparisons.append(
                    (
                        float(target_alpha[state] - alphas[tree][state]),
                        s1,
                        s2,
                        tree,
                    )
                )
    return min(comparisons, key=lambda item: item[0])


def find_attacker_case(
    *,
    max_seed: int,
    target_tree: Tree,
    min_pd_margin: float,
    min_dse_failure: float,
) -> AttackerCase3 | None:
    controls = (0.55, 0.70, 0.85, 0.95)
    observation_information_values = (0.55, 0.70, 0.85, 0.95)
    original_b = prior_b_vector()
    best_case = None
    best_score = float("-inf")

    for seed in range(max_seed):
        for control, obs_info in product(controls, observation_information_values):
            pomdp = build_action_dependent_factored_pomdp(
                seed,
                action_control=control,
                p_s1_matches_s2=obs_info,
            )
            dse_margin, witness_s1, witness_s2, witness_tree = (
                pointwise_t2_dse_witness(pomdp, target_tree)
            )
            if dse_margin > -min_dse_failure:
                continue

            original_evaluations = tuple(
                evaluate_observed_t2_pd(pomdp, original_b, s1, target_tree)
                for s1 in range(NUM_S1)
            )
            # The attack should do real work: reject cases already certified before it.
            if all(evaluation.passes for evaluation in original_evaluations):
                continue

            for attacker in allowed_attacker_policies():
                attacked_b = induced_b(attacker)
                if attacked_b is None:
                    continue
                attacked_evaluations = tuple(
                    evaluate_observed_t2_pd(pomdp, attacked_b, s1, target_tree)
                    for s1 in range(NUM_S1)
                )
                attacked_margin = min(
                    evaluation.margin for evaluation in attacked_evaluations
                )
                if attacked_margin < min_pd_margin:
                    continue

                score = attacked_margin - 0.1 * dse_margin
                if score <= best_score:
                    continue
                best_score = score
                best_case = AttackerCase3(
                    seed=seed,
                    control=control,
                    obs_info=obs_info,
                    pomdp=pomdp,
                    target_tree=target_tree,
                    attacker=attacker.copy(),
                    attacked_b=attacked_b.copy(),
                    original_evaluations=original_evaluations,
                    attacked_evaluations=attacked_evaluations,
                    dse_margin=dse_margin,
                    dse_witness_s1=witness_s1,
                    dse_witness_s2=witness_s2,
                    dse_witness_tree=witness_tree,
                )
    return best_case


def json_tree_key(tree: Tree) -> str:
    return f"a{tree[0]}|a{tree[1][0]},a{tree[1][1]}"


def sequence_values(pomdp, b: np.ndarray, initial_s1: int) -> dict[str, float]:
    return {
        f"a{first_action},a{second_action}": observed_value(
            pomdp,
            (first_action, second_action),
            initial_s1,
            b,
        )
        for first_action, second_action in product(range(NUM_ACTIONS), repeat=2)
    }


def transition_json(pomdp) -> dict[str, dict[str, float]]:
    return {
        f"a{action}": {
            str(next_s1): float(probability)
            for next_s1, probability in enumerate(observed_transition(pomdp, action))
        }
        for action in range(NUM_ACTIONS)
    }


def transition_by_s1_json(pomdp):
    transition = transition_json(pomdp)
    return {
        str(s1): {
            action: dict(probabilities)
            for action, probabilities in transition.items()
        }
        for s1 in range(NUM_S1)
    }


def b_json(b: np.ndarray):
    return {
        str(s1): {
            f"a{action}": float(b[b_index(s1, action)])
            for action in range(NUM_ACTIONS)
        }
        for s1 in range(NUM_S1)
    }


def build_case_json(case: AttackerCase3) -> dict:
    original_b = prior_b_vector()
    transition = transition_json(case.pomdp)
    transition_by_s1 = transition_by_s1_json(case.pomdp)
    attacked_margin = min(evaluation.margin for evaluation in case.attacked_evaluations)

    return {
        "meta": {
            "seed": case.seed,
            "control": case.control,
            "obs_info": case.obs_info,
            "target": ["a1", "a1"],
            "target_tree": {
                "root": f"a{case.target_tree[0]}",
                "after_s1_0": f"a{case.target_tree[1][0]}",
                "after_s1_1": f"a{case.target_tree[1][1]}",
            },
            "margin": attacked_margin,
            "t2_pd_passes": True,
            "t2_dse_passes": False,
            "policies_checked": len(ALLOWED_POLICY_ROWS) ** 4,
        },
        "t2_pd": {
            str(s1): {
                "target_continuation_margin": evaluation.target_continuation_margin,
                "competing_root_margin": evaluation.competing_root_margin,
                "margin": evaluation.margin,
                "passes": evaluation.passes,
                "optimal_continuations": {
                    f"a{root}": [json_tree_key(tree) for tree in trees]
                    for root, trees in evaluation.optimal_continuations.items()
                },
            }
            for s1, evaluation in enumerate(case.attacked_evaluations)
        },
        "t2_dse": {
            "margin": case.dse_margin,
            "passes": False,
            "witness_state": f"{case.dse_witness_s1}{case.dse_witness_s2}",
            "witness_tree": json_tree_key(case.dse_witness_tree),
        },
        "rewards": {
            f"{s1}{s2}": {
                f"a{action}": float(case.pomdp.rewards[join_state(s1, s2), action])
                for action in range(NUM_ACTIONS)
            }
            for s1, s2 in product(range(NUM_S1), range(2))
        },
        "values": {
            "original": {
                str(s1): sequence_values(case.pomdp, original_b, s1)
                for s1 in range(NUM_S1)
            },
            "attacked": {
                str(s1): sequence_values(case.pomdp, case.attacked_b, s1)
                for s1 in range(NUM_S1)
            },
        },
        "policy_tree_values": {
            "original": {
                str(s1): {
                    json_tree_key(tree): value
                    for tree, value in evaluation.values.items()
                }
                for s1, evaluation in enumerate(case.original_evaluations)
            },
            "attacked": {
                str(s1): {
                    json_tree_key(tree): value
                    for tree, value in evaluation.values.items()
                }
                for s1, evaluation in enumerate(case.attacked_evaluations)
            },
        },
        "b": {
            "original": b_json(original_b),
            "attacked": b_json(case.attacked_b),
        },
        "transitions": {
            "original": transition,
            "attacked": transition,
        },
        "transitions_by_s1": {
            "original": transition_by_s1,
            "attacked": transition_by_s1,
        },
        "attacker_policy": {
            f"{s1}{s2}": float(case.attacker[s1, s2, 1])
            for s1, s2 in product(range(NUM_S1), range(2))
        },
        "coverage": {
            "observed_state": True,
            "observed_state_action": True,
            "hidden_state": True,
            "full_state_action": bool(
                np.all(case.attacker > TOL)
            ),
        },
    }


def write_case_json(case: AttackerCase3, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(build_case_json(case), output_file, indent=2)
        output_file.write("\n")


def print_tree_value_section(
    label: str,
    evaluations: tuple[ObservedT2PDEvaluation, ...],
    target_tree: Tree,
) -> None:
    print(f"{label} policy-tree values in the induced observed model:")
    for s1, evaluation in enumerate(evaluations):
        print(f"  initial s1={s1}")
        print("  root  continuation(s1'=0,s1'=1)      value")
        for tree in TREES:
            marker = "  target" if tree == target_tree else ""
            print(
                f"   a{tree[0]}        (a{tree[1][0]},a{tree[1][1]})"
                f"             {evaluation.values[tree]:+10.6f}{marker}"
            )
        print()


def print_optimal_continuations(
    label: str,
    evaluations: tuple[ObservedT2PDEvaluation, ...],
) -> None:
    print(f"optimal continuation(s), {label} model:")
    for s1, evaluation in enumerate(evaluations):
        print(f"  initial s1={s1}")
        for root_action, trees in evaluation.optimal_continuations.items():
            tree_list = ", ".join(tree_label(tree) for tree in trees)
            print(f"    after root a{root_action}: {tree_list}")
    print()


def print_full_attacker_case(
    case: AttackerCase3,
    output_json: Path,
    *,
    report_title: str,
    certificate_label: str,
) -> None:
    original_b = prior_b_vector()
    print(f"=== {report_title} ===")
    print(f"seed: {case.seed}")
    print(f"action control: {case.control:.2f}")
    print(f"observation information: {case.obs_info:.2f}")
    print("initial match probability: 0.70")
    print(f"target tree: {tree_label(case.target_tree)}")
    print("initial observed states checked: s1=0, s1=1")
    print("allowed attacker rows: (1,0), (0,1), (0.5,0.5)")
    print(f"policies checked per instance: {len(ALLOWED_POLICY_ROWS) ** 4}")
    print()

    print("rewards R(s1,s2,a):")
    print("s1  s2       a0       a1")
    for s1, s2 in product(range(NUM_S1), range(2)):
        state = join_state(s1, s2)
        print(
            f" {s1}   {s2}  "
            f"{case.pomdp.rewards[state, 0]:+8.3f} "
            f"{case.pomdp.rewards[state, 1]:+8.3f}"
        )
    print()

    print_tree_value_section(
        "original",
        case.original_evaluations,
        case.target_tree,
    )
    print_tree_value_section(
        "attacked",
        case.attacked_evaluations,
        case.target_tree,
    )
    print_optimal_continuations("original", case.original_evaluations)
    print_optimal_continuations("attacked", case.attacked_evaluations)

    print(f"{certificate_label} after attack:")
    for s1, evaluation in enumerate(case.attacked_evaluations):
        print(f"  initial s1={s1}")
        print(
            "    unique target-continuation margin: "
            f"{evaluation.target_continuation_margin:+.6f}"
        )
        print(
            "    competing-root margin: "
            f"{evaluation.competing_root_margin:+.6f}"
        )
        print(f"    overall margin: {evaluation.margin:+.6f}")
        print(f"    result: {'PASSES' if evaluation.passes else 'FAILS'}")
    print()

    target_alpha = policy_tree_alpha_values(
        case.pomdp,
        case.target_tree[0],
        case.target_tree[1],
    )
    witness_alpha = policy_tree_alpha_values(
        case.pomdp,
        case.dse_witness_tree[0],
        case.dse_witness_tree[1],
    )
    witness_state = join_state(case.dse_witness_s1, case.dse_witness_s2)
    print("T2-DSE pointwise test:")
    print(f"  worst margin: {case.dse_margin:+.6f}")
    print(
        "  witness hidden state: "
        f"(s1,s2)=({case.dse_witness_s1},{case.dse_witness_s2})"
    )
    print(f"  witness tree: {tree_label(case.dse_witness_tree)}")
    print(
        "  target/witness values at that hidden state: "
        f"{target_alpha[witness_state]:+.6f} / "
        f"{witness_alpha[witness_state]:+.6f}"
    )
    print("  result: FAILS")
    print()

    print("hidden-state mixtures P(S2=1 | S1=s1,A=a):")
    print("s1  action   original   attacked")
    for s1, action in product(range(NUM_S1), range(NUM_ACTIONS)):
        print(
            f" {s1}     a{action}     "
            f"{original_b[b_index(s1, action)]:.6f}   "
            f"{case.attacked_b[b_index(s1, action)]:.6f}"
        )
    print()

    print("observed transitions (unchanged by this attack):")
    print("action  P(next_S1=0)  P(next_S1=1)")
    for action in range(NUM_ACTIONS):
        transition = observed_transition(case.pomdp, action)
        print(f"  a{action}       {transition[0]:.6f}          {transition[1]:.6f}")
    print()

    print("attacker policy pi_dagger(a1 | s1,s2):")
    for s1, s2 in product(range(NUM_S1), range(2)):
        print(f"  {s1}{s2}: {case.attacker[s1, s2, 1]:.1f}")
    print()
    print(f"saved JSON: {output_json}")


def parse_args(
    *,
    default_target: Tree,
    default_output_json: Path,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find a fixed-target example where T2-PD passes in the induced "
            "model while the pointwise T2-DSE condition fails."
        )
    )
    parser.add_argument("--max-seed", type=int, default=500)
    parser.add_argument("--initial-match", type=float, default=0.70)
    parser.add_argument(
        "--target-root",
        type=int,
        choices=(0, 1),
        default=default_target[0],
    )
    parser.add_argument(
        "--target-after-0",
        type=int,
        choices=(0, 1),
        default=default_target[1][0],
    )
    parser.add_argument(
        "--target-after-1",
        type=int,
        choices=(0, 1),
        default=default_target[1][1],
    )
    parser.add_argument("--min-pd-margin", type=float, default=0.05)
    parser.add_argument("--min-dse-failure", type=float, default=0.05)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=default_output_json,
    )
    return parser.parse_args()


def main(
    *,
    default_target: Tree = (1, (1, 1)),
    default_output_json: Path = Path("graphing/case_study-c3.json"),
    report_title: str = "T2-PD certifies / T2-DSE fails after restricted attack",
    certificate_label: str = "T2-PD",
) -> None:
    args = parse_args(
        default_target=default_target,
        default_output_json=default_output_json,
    )
    target_tree: Tree = (
        args.target_root,
        (args.target_after_0, args.target_after_1),
    )
    if abs(args.initial_match - 0.70) > TOL:
        raise SystemExit(
            "the restricted attacker enumeration currently uses --initial-match 0.70"
        )

    case = find_attacker_case(
        max_seed=args.max_seed,
        target_tree=target_tree,
        min_pd_margin=args.min_pd_margin,
        min_dse_failure=args.min_dse_failure,
    )
    if case is None:
        raise SystemExit(
            "No restricted-attacker Case 3 found. Try increasing --max-seed "
            "or lowering the margin thresholds."
        )

    write_case_json(case, args.output_json)
    print_full_attacker_case(
        case,
        args.output_json,
        report_title=report_title,
        certificate_label=certificate_label,
    )


if __name__ == "__main__":
    main()
