from __future__ import annotations

import os
import sys
from itertools import product
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np

sys.path.append(os.path.dirname(__file__))

from two_period_joint_policy_experiments import (
    NUM_ACTIONS,
    NUM_S1,
    NUM_S2,
    build_action_dependent_factored_pomdp,
)
from find_t2_dse_inducible_observed_model_case import (
    b_index,
    construct_attacker_from_b,
    format_sequence,
    full_state_value,
    induced_b_from_attacker,
    observed_transition,
    observed_value,
    positive_coverage_b,
    prior_b_vector,
)


SEED = 30
ACTION_CONTROL = 0.70
OBS_INFO = 0.70
TARGET = (1, 1)
WITNESS_B = np.array([1.0, 0.0, 0.7, 0.7])

BLUE = "#1b6ca8"
BLUE_DARK = "#0b3554"
ORANGE = "#d95f02"
ORANGE_DARK = "#8c3b00"
GRAY = "#9aa8ba"
GRAY_DARK = "#5e6b7c"
RED = "#c44e52"
GREEN = "#2f8f5b"


def sequence_values(pomdp, initial_s1, b):
    return [
        {
            "sequence": sequence,
            "value": observed_value(pomdp, sequence, initial_s1, b),
            "is_target": sequence == TARGET,
        }
        for sequence in product(range(NUM_ACTIONS), repeat=2)
    ]


def best_sequence(values):
    return max(values, key=lambda item: item["value"])


def style_axes(ax):
    ax.grid(True, alpha=0.25)
    ax.set_axisbelow(True)


def plot_attacked_values(pomdp, actual_b, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharex=True)

    for initial_s1, ax in enumerate(axes):
        values = sorted(sequence_values(pomdp, initial_s1, actual_b), key=lambda item: item["value"])
        labels = [format_sequence(item["sequence"]) for item in values]
        heights = [item["value"] for item in values]
        colors = [BLUE if item["is_target"] else GRAY for item in values]
        edgecolors = [BLUE_DARK if item["is_target"] else GRAY_DARK for item in values]

        bars = ax.barh(labels, heights, color=colors, edgecolor=edgecolors, linewidth=1.1)
        best = best_sequence(values)
        ax.axvline(best["value"], color=BLUE, linestyle=":", linewidth=1.6)
        for bar, item in zip(bars, values):
            if item["is_target"]:
                bar.set_linewidth(1.5)

        ax.set_title(f"Attacked observed state $s_1={initial_s1}$")
        ax.set_xlabel("open-loop value")
        style_axes(ax)

    axes[0].set_ylabel("sequence $(a^{(1)},a^{(2)})$")
    fig.suptitle("Attacker-induced model makes the target sequence optimal", y=0.98)
    axes[0].text(
        0.98,
        0.04,
        "blue = target sequence\ngray = alternatives\ndotted line = target value",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))

    path = output_dir / "t2-dse_case_study_attacked_values.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_before_after_values(pomdp, prior_b, actual_b, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)

    x = np.arange(4)
    width = 0.34
    sequences = list(product(range(NUM_ACTIONS), repeat=2))
    labels = [format_sequence(sequence) for sequence in sequences]

    for initial_s1, ax in enumerate(axes):
        baseline = [observed_value(pomdp, sequence, initial_s1, prior_b) for sequence in sequences]
        attacked = [observed_value(pomdp, sequence, initial_s1, actual_b) for sequence in sequences]

        baseline_bars = ax.bar(
            x - width / 2,
            baseline,
            width,
            label="observed-only baseline",
            color=ORANGE,
            edgecolor=ORANGE_DARK,
            linewidth=1.0,
        )
        attacked_bars = ax.bar(
            x + width / 2,
            attacked,
            width,
            label="under $\\pi^\\dagger$",
            color=BLUE,
            edgecolor=BLUE_DARK,
            linewidth=1.0,
        )
        for bar in baseline_bars:
            bar.set_hatch("//")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_title(f"Observed state $s_1={initial_s1}$")
        ax.axhline(0.0, color="#333333", linewidth=0.8)
        style_axes(ax)

    axes[0].set_ylabel("estimated open-loop value")
    axes[1].legend(loc="upper left", fontsize=9, frameon=True)
    fig.suptitle("Attacker-induced model changes estimated open-loop values", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    path = output_dir / "t2-dse_case_study_before_after_values.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_hidden_mixtures(prior_b, actual_b, output_dir):
    prior = np.array([[prior_b[b_index(s1, action)] for action in range(NUM_ACTIONS)] for s1 in range(NUM_S1)])
    attacked = np.array([[actual_b[b_index(s1, action)] for action in range(NUM_ACTIONS)] for s1 in range(NUM_S1)])
    diff = attacked - prior

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.6), sharex=True, sharey=True)
    panels = [
        ("baseline $P(S_2=1|S_1)$", prior, "Blues", 0.0, 1.0),
        ("under $\\pi^\\dagger$", attacked, "Blues", 0.0, 1.0),
        ("attacker shift", diff, "RdBu_r", -1.0, 1.0),
    ]

    for ax, (title, data, cmap, vmin, vmax) in zip(axes, panels):
        image = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(title)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["a0", "a1"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["s1=0", "s1=1"])
        for row in range(NUM_S1):
            for col in range(NUM_ACTIONS):
                ax.text(
                    col,
                    row,
                    f"{data[row, col]:+.3f}" if title == "attacker shift" else f"{data[row, col]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    color="white" if abs(data[row, col]) > 0.55 else "#222222",
                )
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    axes[0].set_ylabel("observed state")
    fig.suptitle("The attack works by changing hidden-state mixtures", y=1.02)
    fig.tight_layout()

    path = output_dir / "t2-dse_case_study_hidden_mixtures.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_transition_graph(pomdp, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=True)
    action_colors = {0: (ORANGE, ORANGE_DARK), 1: (BLUE, BLUE_DARK)}

    for action, ax in enumerate(axes):
        color, edge = action_colors[action]
        ax.set_xlim(-0.45, 3.45)
        ax.set_ylim(-0.35, 1.35)
        ax.axis("off")
        ax.set_title(f"action $a_{action}$", color=edge, fontsize=13, fontweight="bold")

        positions = {
            "now": (0.0, 0.5),
            ("next", 0): (3.0, 1.0),
            ("next", 1): (3.0, 0.0),
        }

        x, y = positions["now"]
        circle = Circle((x, y), 0.18, facecolor="white", edgecolor=edge, linewidth=1.5)
        ax.add_patch(circle)
        ax.text(x, y, "$S_1$", ha="center", va="center", fontsize=10.5)

        for next_s1 in range(NUM_S1):
            x, y = positions[("next", next_s1)]
            circle = Circle((x, y), 0.18, facecolor="white", edgecolor=edge, linewidth=1.5)
            ax.add_patch(circle)
            ax.text(x, y, f"$S_1'={next_s1}$", ha="center", va="center", fontsize=10.5)

        probs = observed_transition(pomdp, action)
        for next_s1, prob in enumerate(probs):
            start = positions["now"]
            end = positions[("next", next_s1)]
            rad = 0.16 if next_s1 == 0 else -0.16
            arrow = FancyArrowPatch(
                (start[0] + 0.22, start[1]),
                (end[0] - 0.22, end[1]),
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=2.2,
                color=color,
                alpha=0.85,
            )
            ax.add_patch(arrow)

            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            label_offset = 0.18 if next_s1 == 0 else -0.18
            ax.text(
                mid_x,
                mid_y + label_offset,
                f"{prob:.3f}",
                color=edge,
                ha="center",
                va="center",
                fontsize=11,
                bbox={"facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.9, "pad": 1.5},
            )

        ax.text(0.0, 1.23, "current", ha="center", fontsize=10, fontweight="bold")
        ax.text(3.0, 1.23, "next", ha="center", fontsize=10, fontweight="bold")

    fig.suptitle("Static observed transition graph", fontsize=14, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    path = output_dir / "t2-dse_case_study_transition_graph.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_full_information_first_actions(pomdp, output_dir):
    matrix = np.zeros((NUM_S1, NUM_S2))
    labels = [["" for _ in range(NUM_S2)] for _ in range(NUM_S1)]

    for s1 in range(NUM_S1):
        for s2 in range(NUM_S2):
            state = s1 * NUM_S2 + s2
            values = [
                (sequence, full_state_value(pomdp, sequence, state))
                for sequence in product(range(NUM_ACTIONS), repeat=2)
            ]
            best_value = max(value for _, value in values)
            winners = [sequence for sequence, value in values if abs(value - best_value) <= 1e-9]
            first_actions = sorted({sequence[0] for sequence in winners})
            matrix[s1, s2] = first_actions[0]
            labels[s1][s2] = ",".join(f"a{action}" for action in first_actions)

    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    image = ax.imshow(matrix, cmap=plt.matplotlib.colors.ListedColormap([ORANGE, BLUE]), vmin=0, vmax=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["s2=0", "s2=1"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["s1=0", "s1=1"])
    ax.set_xlabel("hidden state")
    ax.set_ylabel("observed state")
    ax.set_title("Full-information first action")

    for s1 in range(NUM_S1):
        for s2 in range(NUM_S2):
            ax.text(s2, s1, labels[s1][s2], ha="center", va="center", fontsize=12, fontweight="bold", color="white")

    cbar = fig.colorbar(image, ax=ax, ticks=[0.25, 0.75], fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels(["a0", "a1"])
    fig.tight_layout()

    path = output_dir / "t2-dse_case_study_full_information_policy.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main():
    output_dir = Path("outputs/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    pomdp = build_action_dependent_factored_pomdp(
        SEED,
        action_control=ACTION_CONTROL,
        p_s1_matches_s2=OBS_INFO,
    )
    prior_b = prior_b_vector()
    actual_b = positive_coverage_b(WITNESS_B)
    attacker = construct_attacker_from_b(actual_b)
    induced_b = induced_b_from_attacker(attacker)

    paths = [
        plot_attacked_values(pomdp, induced_b, output_dir),
        plot_before_after_values(pomdp, prior_b, induced_b, output_dir),
        plot_hidden_mixtures(prior_b, induced_b, output_dir),
        plot_transition_graph(pomdp, output_dir),
        plot_full_information_first_actions(pomdp, output_dir),
    ]

    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
