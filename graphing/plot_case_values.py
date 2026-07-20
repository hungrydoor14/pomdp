import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib.pyplot as plt

from common import SEQUENCES, derive_model_results, read_case_file
from style import BLUE, GRAY, LIGHT_GRAY, PANEL_EDGE, PANEL_FILL, RED


INPUT_FILE = Path("graphing/case_study-c1.json")
OUTPUT_DIR = Path("outputs/t3")
PNG_OUTPUT = OUTPUT_DIR / f"{INPUT_FILE.stem}_values.png"
PDF_OUTPUT = OUTPUT_DIR / f"{INPUT_FILE.stem}_values.pdf"


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


def sequence_values(data, prefix, s1):
    return [data[f"{prefix}_values_s1_{s1}"][sequence] for sequence in SEQUENCES]


def draw_value_panel(ax, data, results, prefix, s1):
    values = sequence_values(data, prefix, s1)
    best_sequence = results[s1]["sequence"]

    colors = [
        BLUE if sequence == best_sequence else "#DDE3EE"
        for sequence in SEQUENCES
    ]

    if prefix == "attacked":
        colors = [
            RED if sequence == best_sequence else "#E7EAF0"
            for sequence in SEQUENCES
        ]

    winner_edge = "#0F3B82" if prefix == "original" else "#8F1D1D"

    bars = ax.bar(
        range(len(SEQUENCES)),
        values,
        color=colors,
        edgecolor=[
            winner_edge if sequence == best_sequence else PANEL_EDGE
            for sequence in SEQUENCES
        ],
        linewidth=[
            1.4 if sequence == best_sequence else 0.8
            for sequence in SEQUENCES
        ],
        width=0.68,
    )

    for bar, value in zip(bars, values):
        y = value + 0.06 if value >= 0 else value - 0.10
        va = "bottom" if value >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{value:.3f}",
            ha="center",
            va=va,
            fontsize=8.5,
            color=GRAY,
        )

    ax.axhline(0, color=LIGHT_GRAY, linewidth=1.0)
    ax.set_xticks(range(len(SEQUENCES)))
    ax.set_xticklabels([sequence_label(sequence) for sequence in SEQUENCES], fontsize=9)
    ax.tick_params(axis="y", labelsize=8.5, colors=GRAY)
    ax.grid(axis="y", color="#EEF1F5", linewidth=0.8)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_color(PANEL_EDGE)
        spine.set_linewidth(0.9)

    ax.set_facecolor(PANEL_FILL)


def main():
    data = read_case_file(INPUT_FILE)
    original_results = derive_model_results(data, "original")
    attacked_results = derive_model_results(data, "attacked")

    all_values = [
        value
        for prefix in ("original", "attacked")
        for s1 in (0, 1)
        for value in sequence_values(data, prefix, s1)
    ]
    y_min = min(all_values) - 0.35
    y_max = max(all_values) + 0.35

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.8), sharey=True)

    for row, s1 in enumerate((0, 1)):
        draw_value_panel(axes[row, 0], data, original_results, "original", s1)
        draw_value_panel(axes[row, 1], data, attacked_results, "attacked", s1)

        axes[row, 0].set_ylabel(rf"$S_1={s1}$", fontsize=12, fontweight="bold", color=GRAY)

        for col in (0, 1):
            axes[row, col].set_ylim(y_min, y_max)

    axes[0, 0].set_title("Original observed model", fontsize=13, fontweight="bold", color=BLUE)
    axes[0, 1].set_title("Attacker-induced observed model", fontsize=13, fontweight="bold", color=RED)

    meta = data["meta"]
    fig.suptitle(
        (
            rf"{case_label_from_path(INPUT_FILE)} value comparison "
            rf"(seed {meta['seed']}, target "
            rf"{sequence_label(meta['target'])})"
        ),
        fontsize=15,
        fontweight="bold",
        color=BLUE,
        y=0.98,
    )

    fig.text(
        0.5,
        0.035,
        "Highlighted bars are the learned optimal open-loop sequence for each observed initial state.",
        ha="center",
        fontsize=10,
        color=GRAY,
    )

    fig.tight_layout(rect=[0.045, 0.07, 0.995, 0.93])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_OUTPUT, dpi=300, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    fig.savefig(PDF_OUTPUT, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    plt.close(fig)

    print(f"Input: {INPUT_FILE}")
    print(f"Saved PNG: {PNG_OUTPUT}")
    print(f"Saved PDF: {PDF_OUTPUT}")


if __name__ == "__main__":
    main()
