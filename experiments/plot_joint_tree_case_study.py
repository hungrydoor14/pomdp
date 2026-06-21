from __future__ import annotations

import os
from itertools import product
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

from horizon2_joint_policy_experiments import (
    NUM_ACTIONS,
    analyze,
    build_action_dependent_factored_pomdp,
    initial_belief_given_s1,
    join_state,
    policy_tree_alpha_values,
    root_q2_diagnostics,
    terminal_feasibility_for_history,
)
from finite_horizon_solver import solve_finite_horizon


SEED = 30
ACTION_CONTROL = 0.95
OBS_INFO = 0.55
INITIAL_MATCH_PROB = 0.70
INITIAL_S1 = 0


def plan_label(root_action: int, continuation: tuple[int, int]) -> str:
    return f"a{root_action}; if 0->a{continuation[0]}, if 1->a{continuation[1]}"


def short_plan_label(root_action: int, continuation: tuple[int, int]) -> str:
    return f"({root_action},{continuation[0]},{continuation[1]})"


def main() -> None:
    output_dir = Path("outputs/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    row = analyze(SEED, ACTION_CONTROL, OBS_INFO, INITIAL_MATCH_PROB)
    pomdp = build_action_dependent_factored_pomdp(
        SEED,
        action_control=ACTION_CONTROL,
        p_s1_matches_s2=OBS_INFO,
    )
    belief = initial_belief_given_s1(INITIAL_S1, INITIAL_MATCH_PROB)

    target_tree = (
        row.pi1[INITIAL_S1],
        (
            row.pi2[2 * INITIAL_S1],
            row.pi2[2 * INITIAL_S1 + 1],
        ),
    )

    plans = []
    for root_action in range(NUM_ACTIONS):
        for continuation in product(range(NUM_ACTIONS), repeat=2):
            continuation = tuple(continuation)
            alpha = policy_tree_alpha_values(pomdp, root_action, continuation)
            value = float(belief @ alpha)
            hidden_values = [
                alpha[join_state(INITIAL_S1, s2)]
                for s2 in range(2)
            ]
            plans.append(
                {
                    "root": root_action,
                    "continuation": continuation,
                    "alpha": alpha,
                    "value": value,
                    "hidden_values": hidden_values,
                    "is_target": (root_action, continuation) == target_tree,
                    "is_memoryless": continuation[0] == continuation[1],
                }
            )

    target = next(plan for plan in plans if plan["is_target"])
    target_value = target["value"]
    for plan in plans:
        plan["gap_from_target"] = target_value - plan["value"]
        plan["joint_margin"] = min(
            target["alpha"][join_state(INITIAL_S1, s2)]
            - plan["alpha"][join_state(INITIAL_S1, s2)]
            for s2 in range(2)
        )

    alternatives = [plan for plan in plans if not plan["is_target"]]
    best_memoryless = max(
        (plan for plan in plans if plan["is_memoryless"]),
        key=lambda plan: plan["value"],
    )
    best_alternative = max(alternatives, key=lambda plan: plan["value"])
    joint_tree_margin = min(plan["joint_margin"] for plan in alternatives)
    memoryless_gap = target_value - best_memoryless["value"]

    root_diag = root_q2_diagnostics(
        pomdp,
        belief,
        INITIAL_S1,
        target_tree[0],
        solve_finite_horizon(pomdp, horizon=2)[1],
    )

    terminal_margins = []
    for obs, target_action in enumerate(target_tree[1]):
        posterior = pomdp.belief_update(belief, target_tree[0], obs)
        terminal_diag = terminal_feasibility_for_history(pomdp, posterior, obs, target_action)
        terminal_margins.append(terminal_diag[3])
    terminal_margin = min(terminal_margins)

    plans_for_plot = sorted(plans, key=lambda plan: plan["value"])
    labels = [
        short_plan_label(plan["root"], plan["continuation"])
        for plan in plans_for_plot
    ]
    values = [plan["value"] for plan in plans_for_plot]
    colors = []
    edgecolors = []
    hatches = []
    for plan in plans_for_plot:
        if plan["is_target"]:
            colors.append("#1b6ca8")
            edgecolors.append("#0b3554")
            hatches.append("")
        elif plan["is_memoryless"]:
            colors.append("#d95f02")
            edgecolors.append("#8c3b00")
            hatches.append("//")
        else:
            colors.append("#9aa8ba")
            edgecolors.append("#5e6b7c")
            hatches.append("")

    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    bars = ax.barh(labels, values, color=colors, edgecolor=edgecolors, linewidth=1.1)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    ax.axvline(target_value, color="#1b6ca8", linestyle=":", linewidth=1.5)
    ax.set_xlabel("value at initial belief")
    ax.set_ylabel("plan (a1, action after obs 0, action after obs 1)")
    ax.set_title("History-dependent plan beats all two-period alternatives")
    ax.grid(True, axis="x", alpha=0.25)
    ax.text(
        0.98,
        0.02,
        "blue = target history plan\norange hatched = non-history plans\ngray = other history plans",
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
    )

    fig.tight_layout()
    values_output_path = output_dir / "joint_tree_case_study_seed30_values.png"
    fig.savefig(values_output_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    margin_labels = [
        "joint whole-plan",
        "root-only Q2",
        "terminal",
        "history advantage",
    ]
    margin_values = [
        joint_tree_margin,
        root_diag[3],
        terminal_margin,
        memoryless_gap,
    ]
    margin_colors = [
        "#1b6ca8" if value > 0 else "#c44e52"
        for value in margin_values
    ]
    ax.barh(margin_labels, margin_values, color=margin_colors, edgecolor="#333333", linewidth=0.8)
    ax.axvline(0.0, color="#333333", linewidth=1.0)
    ax.set_title("What certifies the example?")
    ax.set_xlabel("margin")
    ax.grid(True, axis="x", alpha=0.25)
    ax.invert_yaxis()
    left_limit = min(margin_values) - 0.25
    right_limit = max(margin_values) + 0.25
    ax.set_xlim(left_limit, right_limit)

    for y_pos, value in enumerate(margin_values):
        horizontal_align = "right" if value >= 0 else "left"
        x_offset = -0.045 if value >= 0 else 0.045
        ax.text(
            value + x_offset,
            y_pos,
            f"{value:+.3f}",
            va="center",
            ha=horizontal_align,
            fontsize=10,
            fontweight="bold",
        )

    summary = (
        f"target: {short_plan_label(target_tree[0], target_tree[1])}\n"
        f"best non-history: {short_plan_label(best_memoryless['root'], best_memoryless['continuation'])}\n"
        f"belief P(S2=1 | S1=0) = {belief[join_state(INITIAL_S1, 1)]:.2f}"
    )
    ax.text(
        0.02,
        -0.22,
        summary,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9.5,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.95},
    )

    fig.tight_layout()
    margins_output_path = output_dir / "joint_tree_case_study_seed30_margins.png"
    fig.savefig(margins_output_path, dpi=180)
    plt.close(fig)

    print(values_output_path)
    print(margins_output_path)
    print("Target:", plan_label(target_tree[0], target_tree[1]), f"value={target_value:.6f}")
    print(
        "Best memoryless:",
        plan_label(best_memoryless["root"], best_memoryless["continuation"]),
        f"value={best_memoryless['value']:.6f}",
    )
    print("Best alternative:", plan_label(best_alternative["root"], best_alternative["continuation"]), f"value={best_alternative['value']:.6f}")
    print(f"history advantage over best memoryless: {memoryless_gap:.6f}")
    print(f"joint whole-plan margin: {joint_tree_margin:+.6f}")
    print(f"root-only Q2 margin: {root_diag[3]:+.6f}")
    print(f"terminal margin: {terminal_margin:+.6f}")


if __name__ == "__main__":
    main()
