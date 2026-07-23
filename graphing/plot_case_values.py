import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib.pyplot as plt

from common import SEQUENCES, derive_model_results, read_case_file
from style import (
    AXIS_LABEL_SIZE,
    BLUE,
    GRAY,
    LEGEND_SIZE,
    LIGHT_GRAY,
    PANEL_EDGE,
    PANEL_FILL,
    PANEL_TITLE_SIZE,
    RED,
    TICK_SIZE,
    TITLE_SIZE,
    VALUE_LABEL_SIZE,
)


INPUT_FILE = Path("graphing/case_study-c1.json")
OUTPUT_DIR = Path("outputs/l1")
THEOREM = "t2"
CURRENT_PART = "pd"
PNG_OUTPUT = OUTPUT_DIR / f"{THEOREM}-{CURRENT_PART}_{INPUT_FILE.stem}_values.png"
PDF_OUTPUT = OUTPUT_DIR / f"{THEOREM}-{CURRENT_PART}_{INPUT_FILE.stem}_values.pdf"


def case_label_from_path(path):
    parts = path.stem.split("-")

    for index, part in enumerate(parts):
        if not part.startswith("c") or not part[1:].isdigit():
            continue

        case_number = int(part[1:])

        if index + 1 < len(parts) and parts[index + 1].isdigit():
            return f"Case {case_number}.{int(parts[index + 1])}"

        return f"Case {case_number}"

    return "Case"


def sequence_label(sequence):
    first = sequence[0][-1]
    second = sequence[1][-1]
    return rf"$(a_{first},a_{second})$"


def value_for(data, prefix, s1, sequence):
    return data[f"{prefix}_values_s1_{s1}"][sequence]


def draw_sequence_panel(ax, data, original_results, attacked_results, sequence):
    values = {
        ("original", 0): value_for(data, "original", 0, sequence),
        ("attacked", 0): value_for(data, "attacked", 0, sequence),
        ("original", 1): value_for(data, "original", 1, sequence),
        ("attacked", 1): value_for(data, "attacked", 1, sequence),
    }

    x_positions = [-0.13, 0.13, 0.87, 1.13]
    bar_values = [
        values[("original", 0)],
        values[("attacked", 0)],
        values[("original", 1)],
        values[("attacked", 1)],
    ]
    is_optimal = [
        original_results[0]["sequence"] == sequence,
        attacked_results[0]["sequence"] == sequence,
        original_results[1]["sequence"] == sequence,
        attacked_results[1]["sequence"] == sequence,
    ]
    fills = [
        BLUE if is_optimal[0] else "#DDE3EE",
        RED if is_optimal[1] else "#F0D8D8",
        BLUE if is_optimal[2] else "#DDE3EE",
        RED if is_optimal[3] else "#F0D8D8",
    ]
    edges = [
        "#0F3B82" if is_optimal[0] else PANEL_EDGE,
        "#8F1D1D" if is_optimal[1] else PANEL_EDGE,
        "#0F3B82" if is_optimal[2] else PANEL_EDGE,
        "#8F1D1D" if is_optimal[3] else PANEL_EDGE,
    ]
    widths = [1.5 if optimal else 0.8 for optimal in is_optimal]

    bars = ax.bar(
        x_positions,
        bar_values,
        color=fills,
        edgecolor=edges,
        linewidth=widths,
        width=0.22,
    )

    for bar, value in zip(bars, bar_values):
        y = value + 0.06 if value >= 0 else value - 0.10
        va = "bottom" if value >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{value:.3f}",
            ha="center",
            va=va,
            fontsize=VALUE_LABEL_SIZE,
            color=GRAY,
        )

    ax.axhline(0, color=LIGHT_GRAY, linewidth=1.0)
    ax.set_xlim(-0.45, 1.45)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([r"$S_1=0$", r"$S_1=1$"], fontsize=TICK_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_SIZE, colors=GRAY)
    ax.grid(axis="y", color="#EEF1F5", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_facecolor(PANEL_FILL)
    ax.set_box_aspect(1)
    ax.set_title(sequence_label(sequence), fontsize=PANEL_TITLE_SIZE, fontweight="bold", color=GRAY)

    for spine in ax.spines.values():
        spine.set_color(PANEL_EDGE)
        spine.set_linewidth(0.9)


def main():
    data = read_case_file(INPUT_FILE)
    original_results = derive_model_results(data, "original")
    attacked_results = derive_model_results(data, "attacked")

    all_values = [
        value
        for prefix in ("original", "attacked")
        for s1 in (0, 1)
        for value in data[f"{prefix}_values_s1_{s1}"].values()
    ]
    y_min = min(all_values) - 0.65
    y_max = max(all_values) + 0.65

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 8.4), sharey=True)

    sequence_positions = {
        ("a0", "a1"): (0, 0),
        ("a1", "a1"): (0, 1),
        ("a0", "a0"): (1, 0),
        ("a1", "a0"): (1, 1),
    }

    for sequence, (row, col) in sequence_positions.items():
        ax = axes[row, col]
        draw_sequence_panel(ax, data, original_results, attacked_results, sequence)
        ax.set_ylim(y_min, y_max)

    axes[0, 0].set_ylabel("Period 2: " + r"$a_1$", fontsize=AXIS_LABEL_SIZE, color=GRAY)
    axes[1, 0].set_ylabel("Period 2: " + r"$a_0$", fontsize=AXIS_LABEL_SIZE, color=GRAY)
    axes[1, 0].set_xlabel("Period 1: " + r"$a_0$", fontsize=AXIS_LABEL_SIZE, color=GRAY)
    axes[1, 1].set_xlabel("Period 1: " + r"$a_1$", fontsize=AXIS_LABEL_SIZE, color=GRAY)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=BLUE, edgecolor="#0F3B82"),
        plt.Rectangle((0, 0), 1, 1, facecolor=RED, edgecolor="#8F1D1D"),
    ]
    fig.legend(
        handles,
        ["Original value", "Attacker-induced value"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=2,
        frameon=False,
        fontsize=LEGEND_SIZE,
    )

    fig.suptitle(
        rf"{case_label_from_path(INPUT_FILE)} value comparison",
        fontsize=TITLE_SIZE,
        fontweight="bold",
        color=BLUE,
        y=0.975,
    )

    fig.subplots_adjust(
        left=0.12,
        right=0.985,
        bottom=0.075,
        top=0.805,
        wspace=0.18,
        hspace=0.38,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_OUTPUT, dpi=300, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    #fig.savefig(PDF_OUTPUT, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    plt.close(fig)

    print(f"Input: {INPUT_FILE}")
    print(f"Saved PNG: {PNG_OUTPUT}")
    print(f"Saved PDF: {PDF_OUTPUT}")


if __name__ == "__main__":
    main()
