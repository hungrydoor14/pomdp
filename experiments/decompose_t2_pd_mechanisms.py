"""Reward/transition ablations for a restricted two-period target family.

The same rooted tree is evaluated at both initial observed states.  The
reward-only and transition-only models are counterfactual hybrids; they need
not be jointly inducible by a single attacker policy.
"""

import argparse
from itertools import product
from pathlib import Path

import numpy as np

from find_t2_dse_transition_mislearning_case import (
    DISCOUNT,
    NUM_ACTIONS,
    NUM_S1,
    induced_observed_transition,
    induced_reward,
    construct_attacker_from_b,
    make_hidden_dependent_transition,
    prior_b_vector,
    sample_bayes_plausible_b,
    search as find_case_21,
)
from find_t2_dse_inducible_observed_model_case import induced_b_from_attacker


TREES = tuple(
    (root, tuple(continuation))
    for root in range(NUM_ACTIONS)
    for continuation in product(range(NUM_ACTIONS), repeat=NUM_S1)
)
class InducedModel:
    def __init__(self, rewards, transitions):
        self.rewards = rewards
        self.transitions = transitions


class MarginResult:
    def __init__(self, margin, binding_s1, binding_tree):
        self.margin = margin
        self.binding_s1 = binding_s1
        self.binding_tree = binding_tree


class Decomposition:
    def __init__(
        self, original, reward_only, transition_only, full_attack, margin_tol
    ):
        self.original = original
        self.reward_only = reward_only
        self.transition_only = transition_only
        self.full_attack = full_attack
        self.margin_tol = margin_tol

    @property
    def classification(self):
        reward_works = self.reward_only.margin > self.margin_tol
        transition_works = self.transition_only.margin > self.margin_tol
        full_works = self.full_attack.margin > self.margin_tol
        if not full_works:
            return "unsuccessful"
        if not reward_works and not transition_works:
            return "joint-only"
        if reward_works and transition_works:
            return "either-component-alone"
        if reward_works:
            return "reward-driven"
        return "transition-driven"


def tree_label(tree):
    return f"(a{tree[0]},a{tree[1][0]},a{tree[1][1]})"


def build_induced_model(rewards, transitions, b):
    induced_rewards = np.zeros((NUM_S1, NUM_ACTIONS))
    induced_transitions = np.zeros((NUM_S1, NUM_ACTIONS, NUM_S1))
    for s1, action in product(range(NUM_S1), range(NUM_ACTIONS)):
        induced_rewards[s1, action] = induced_reward(rewards, b, s1, action)
        induced_transitions[s1, action] = induced_observed_transition(
            transitions, b, s1, action
        )
    return InducedModel(induced_rewards, induced_transitions)


def tree_value(model, tree, initial_s1):
    root, continuation = tree
    value = model.rewards[initial_s1, root]
    value += DISCOUNT * sum(
        model.transitions[initial_s1, root, next_s1]
        * model.rewards[next_s1, continuation[next_s1]]
        for next_s1 in range(NUM_S1)
    )
    return float(value)


def teaching_margin(model, target):
    comparisons = [
        (
            tree_value(model, target, s1) - tree_value(model, competitor, s1),
            s1,
            competitor,
        )
        for s1 in range(NUM_S1)
        for competitor in TREES
        if competitor != target
    ]
    margin, binding_s1, binding_tree = min(comparisons, key=lambda item: item[0])
    return MarginResult(float(margin), binding_s1, binding_tree)


def decompose(original, attacked, target, margin_tol):
    # The explicit constructors below make clear which component each hybrid uses.
    reward_only = InducedModel(attacked.rewards, original.transitions)
    transition_only = InducedModel(original.rewards, attacked.transitions)
    return Decomposition(
        teaching_margin(original, target),
        teaching_margin(reward_only, target),
        teaching_margin(transition_only, target),
        teaching_margin(attacked, target),
        margin_tol,
    )


def known_case_21():
    case = find_case_21()
    expected = (0, 0.95, 0.0, 0.75)
    actual = (
        case.seed,
        case.hidden_effect,
        case.action_effect,
        case.action_control,
    )
    if actual != expected:
        raise RuntimeError(
            f"Expected published Case 2.1 parameters {expected}, got {actual}"
        )
    return case.rewards, case.transitions, case.attacked_b, (1, (1, 1)), case


def find_joint_only(
    max_seed,
    samples_per_model,
    search_rng_seed,
    margin_tol,
    paper_margin_threshold,
    stop_after_paper_witness,
):
    rng = np.random.default_rng(search_rng_seed)
    original_b = prior_b_vector()
    best = None
    counts = {"models": 0, "mixtures": 0, "targets": 0}

    for seed in range(max_seed):
        rewards = np.random.default_rng(seed).normal(
            0.0, 1.0, size=(NUM_S1, 2, NUM_ACTIONS)
        )
        for hidden_effect, action_effect in product(
            (0.75, 0.85, 0.95), (0.0, 0.1, 0.2)
        ):
            counts["models"] += 1
            transitions = make_hidden_dependent_transition(
                hidden_effect=hidden_effect,
                action_effect=action_effect,
                action_control=0.75,
            )
            original = build_induced_model(rewards, transitions, original_b)
            for _ in range(samples_per_model):
                counts["mixtures"] += 1
                attacked_b = sample_bayes_plausible_b(rng)
                attacker = construct_attacker_from_b(attacked_b)
                if attacker is None:
                    raise RuntimeError("Sampled mixture was not behaviorally inducible")
                reconstructed_b = induced_b_from_attacker(attacker)
                if not np.allclose(reconstructed_b, attacked_b, atol=1e-10):
                    raise RuntimeError("Attacker does not reconstruct sampled mixtures")
                attacked = build_induced_model(
                    rewards, transitions, reconstructed_b
                )
                for target in TREES:
                    counts["targets"] += 1
                    result = decompose(original, attacked, target, margin_tol)
                    if result.classification != "joint-only":
                        continue
                    if result.original.margin >= -margin_tol:
                        continue
                    witness = {
                        "seed": seed,
                        "hidden_effect": hidden_effect,
                        "action_effect": action_effect,
                        "action_control": 0.75,
                        "target": target,
                        "attacked_b": reconstructed_b.copy(),
                        "attacker": attacker.copy(),
                        "rewards": rewards.copy(),
                        "transitions": transitions.copy(),
                        "decomposition": result,
                    }
                    if (
                        best is None
                        or result.full_attack.margin
                        > best["decomposition"].full_attack.margin
                    ):
                        best = witness
            if (
                stop_after_paper_witness
                and best is not None
                and best["decomposition"].full_attack.margin
                >= paper_margin_threshold
            ):
                return best, counts
    return best, counts


def format_result(label, result):
    lines = [label]
    for name, margin_result in (
        ("original", result.original),
        ("reward_only", result.reward_only),
        ("transition_only", result.transition_only),
        ("full_attack", result.full_attack),
    ):
        lines.extend(
            [
                f"{name}_margin {margin_result.margin:+.9f}",
                f"{name}_binding_s1 {margin_result.binding_s1}",
                f"{name}_binding_tree {tree_label(margin_result.binding_tree)}",
            ]
        )
    lines.append(f"classification {result.classification}")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-seed", type=int, default=250)
    parser.add_argument("--samples-per-model", type=int, default=250)
    parser.add_argument("--search-rng-seed", type=int, default=20260726)
    parser.add_argument("--margin-tol", type=float, default=1e-6)
    parser.add_argument(
        "--paper-margin-threshold",
        "--paper-margin",
        dest="paper_margin_threshold",
        type=float,
        default=0.05,
    )
    parser.add_argument("--stop-after-paper-witness", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/t2/t2-pd-mechanism-decomposition.txt"),
    )
    args = parser.parse_args()

    original_b = prior_b_vector()
    rewards, transitions, attacked_b, target, case_21_source = known_case_21()
    original = build_induced_model(rewards, transitions, original_b)
    attacked = build_induced_model(rewards, transitions, attacked_b)
    case_21 = decompose(original, attacked, target, args.margin_tol)

    lines = ["T2_PD_MECHANISM_DECOMPOSITION"]
    lines.extend(format_result("CASE_2_1", case_21))
    lines.append(f"target_tree {tree_label(target)}")
    lines.extend(
        [
            "case_2_1_uses_exact_generator_values 1",
            f"seed {case_21_source.seed}",
            f"hidden_effect {case_21_source.hidden_effect:.2f}",
            f"action_effect {case_21_source.action_effect:.2f}",
            f"action_control {case_21_source.action_control:.2f}",
        ]
    )

    lines.extend(
        [
            "JOINT_ONLY_ATTACKER_SEARCH_METADATA",
            f"search_rng_seed {args.search_rng_seed}",
            f"max_seed {args.max_seed}",
            f"samples_per_model {args.samples_per_model}",
            f"margin_tolerance {args.margin_tol:.9f}",
            "paper_witness_margin_threshold "
            f"{args.paper_margin_threshold:.9f}",
            "stop_after_paper_witness "
            f"{int(args.stop_after_paper_witness)}",
            "joint_only_search_is_randomized 1",
            "hybrid_models_are_counterfactual 1",
            "same_target_tree_used_for_both_initial_states 1",
        ]
    )

    witness, counts = find_joint_only(
        args.max_seed,
        args.samples_per_model,
        args.search_rng_seed,
        args.margin_tol,
        args.paper_margin_threshold,
        args.stop_after_paper_witness,
    )
    lines.extend(
        [
            f"models_tested {counts['models']}",
            f"mixtures_sampled {counts['mixtures']}",
            f"target_evaluations {counts['targets']}",
        ]
    )
    if witness is None:
        lines.extend(
            ["JOINT_ONLY_ATTACKER_SEARCH", "joint_only_found 0"]
        )
    else:
        expected_transition_shape = (
            NUM_ACTIONS,
            NUM_S1,
            2,
            NUM_S1,
            2,
        )
        if witness["transitions"].shape != expected_transition_shape:
            raise RuntimeError(
                "Unexpected transition shape: "
                f"expected {expected_transition_shape}, "
                f"got {witness['transitions'].shape}"
            )
        lines.extend(
            format_result(
                "JOINT_ONLY_ATTACKER_SEARCH", witness["decomposition"]
            )
        )
        lines.extend(
            [
                "joint_only_found 1",
                f"seed {witness['seed']}",
                f"hidden_effect {witness['hidden_effect']:.2f}",
                f"action_effect {witness['action_effect']:.2f}",
                f"action_control {witness['action_control']:.2f}",
                "transition_generator make_hidden_dependent_transition",
                "transition_shape "
                + " ".join(str(size) for size in witness["transitions"].shape),
                f"target_tree {tree_label(witness['target'])}",
                "original_mixtures "
                + " ".join(f"{value:.9f}" for value in original_b),
                "attacked_mixtures "
                + " ".join(f"{value:.9f}" for value in witness["attacked_b"]),
                "attacker_reconstruction_verified 1",
                "# attacker: s1 s2 pi(a1|s1,s2)",
                "# rewards: s1 s2 R(a0) R(a1)",
            ]
        )
        attacker_lines = []
        for s1, s2 in product(range(NUM_S1), range(2)):
            attacker_lines.append(
                f"{s1} {s2} {witness['attacker'][s1, s2, 1]:.9f}"
            )
        rewards_header = lines.pop()
        lines.extend(attacker_lines)
        lines.append(rewards_header)
        for s1, s2 in product(range(NUM_S1), range(2)):
            lines.append(
                f"{s1} {s2} {witness['rewards'][s1, s2, 0]:+.9f} "
                f"{witness['rewards'][s1, s2, 1]:+.9f}"
            )
        lines.append("# transitions: action s1 s2 next_s1 next_s2 probability")
        for action, s1, s2, next_s1, next_s2 in product(
            range(NUM_ACTIONS), range(NUM_S1), range(2), range(NUM_S1), range(2)
        ):
            lines.append(
                f"{action} {s1} {s2} {next_s1} {next_s2} "
                f"{witness['transitions'][action, s1, s2, next_s1, next_s2]:.9f}"
            )

    text = "\n".join(lines) + "\n"
    print(text, end="")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
