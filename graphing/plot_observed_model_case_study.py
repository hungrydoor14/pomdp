from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import (
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
    Wedge,
)


# ============================================================
# Paths
# ============================================================

DEFAULT_INPUT_FILE = Path("graphing/case_study-c2.txt")
OUTPUT_DIR = Path("outputs/t3")


# ============================================================
# Visual configuration
# ============================================================

BLUE = "#174EA6"
BLUE_LIGHT = "#4F7DFF"
RED = "#C62828"
GRAY = "#606975"
LIGHT_GRAY = "#D5D9E0"
PANEL_EDGE = "#AEB5C0"
PANEL_FILL = "#FCFCFD"

POSITIVE_BASE = "#2457FF"
NEGATIVE_BASE = "#F28C28"
TRANSITION_CMAP = "viridis"
TRANSITION_COLOR_MIN = 0.35
TRANSITION_COLOR_MAX = 0.65

NODE_RADIUS = 0.43
ACTION_DIAMOND_HALF_WIDTH = 0.25
ACTION_DIAMOND_HALF_HEIGHT = 0.18
TRANSITION_ALPHA = 0.54
ARROW_LINEWIDTH = 2.6

STATE_ORDER = [
    "00",
    "01",
    "10",
    "11",
]

STATE_Y = {
    "00": 1.85,
    "01": 0.65,
    "10": -0.55,
    "11": -1.75,
}


# ============================================================
# Parse the text file
# ============================================================

def read_case_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}"
        )

    data: dict[str, Any] = {}
    current_section: str | None = None

    known_sections = {
        "meta",
        "rewards",
        "original_values_s1_0",
        "original_values_s1_1",
        "attacked_values_s1_0",
        "attacked_values_s1_1",
        "original_b",
        "attacked_b",
        "original_transitions",
        "attacked_transitions",
    }

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, raw_line in enumerate(
            file,
            start=1,
        ):
            line = raw_line.strip()

            if not line:
                continue

            if line.startswith("#"):
                heading = line[1:].strip()

                if heading in known_sections:
                    current_section = heading
                    data[current_section] = {}

                continue

            if current_section is None:
                raise ValueError(
                    f"Line {line_number}: "
                    "data appears before a section."
                )

            parts = line.split()

            if current_section == "meta":
                key = parts[0]

                if key == "target":
                    data[current_section][key] = (
                        parts[1],
                        parts[2],
                    )

                elif key == "seed":
                    data[current_section][key] = int(
                        parts[1]
                    )

                else:
                    data[current_section][key] = float(
                        parts[1]
                    )

            elif current_section == "rewards":
                if len(parts) != 3:
                    raise ValueError(
                        f"Line {line_number}: "
                        "reward rows require "
                        "STATE A0_VALUE A1_VALUE."
                    )

                state, a0_value, a1_value = parts

                data[current_section][state] = {
                    "a0": float(a0_value),
                    "a1": float(a1_value),
                }

            elif "values_s1_" in current_section:
                if len(parts) != 3:
                    raise ValueError(
                        f"Line {line_number}: "
                        "sequence rows require "
                        "FIRST_ACTION SECOND_ACTION VALUE."
                    )

                first_action, second_action, value = parts

                data[current_section][
                    (first_action, second_action)
                ] = float(value)

            elif current_section in {
                "original_b",
                "attacked_b",
            }:
                if len(parts) != 3:
                    raise ValueError(
                        f"Line {line_number}: "
                        "belief rows require "
                        "S1 ACTION PROBABILITY."
                    )

                s1, action, probability = parts

                data[current_section][
                    (int(s1), action)
                ] = float(probability)

            elif current_section in {
                "original_transitions",
                "attacked_transitions",
            }:
                if len(parts) != 3:
                    raise ValueError(
                        f"Line {line_number}: "
                        "transition rows require "
                        "ACTION P_NEXT_0 P_NEXT_1."
                    )

                action, p0, p1 = parts

                data[current_section][action] = {
                    0: float(p0),
                    1: float(p1),
                }

    validate_input(data)

    return data


def validate_input(
    data: dict[str, Any],
) -> None:
    required_sections = {
        "meta",
        "rewards",
        "original_values_s1_0",
        "original_values_s1_1",
        "attacked_values_s1_0",
        "attacked_values_s1_1",
        "original_b",
        "attacked_b",
        "original_transitions",
        "attacked_transitions",
    }

    missing = required_sections.difference(data)

    if missing:
        raise ValueError(
            "Missing sections: "
            + ", ".join(sorted(missing))
        )

    expected_states = set(STATE_ORDER)

    if set(data["rewards"]) != expected_states:
        raise ValueError(
            "Rewards must be supplied for states "
            "00, 01, 10, and 11."
        )

    expected_sequences = {
        ("a0", "a0"),
        ("a0", "a1"),
        ("a1", "a0"),
        ("a1", "a1"),
    }

    for section in (
        "original_values_s1_0",
        "original_values_s1_1",
        "attacked_values_s1_0",
        "attacked_values_s1_1",
    ):
        if set(data[section]) != expected_sequences:
            raise ValueError(
                f"{section} must contain "
                "all four open-loop sequences."
            )


# ============================================================
# Derive optimal sequences
# ============================================================

def best_sequence(
    values: dict[tuple[str, str], float],
) -> tuple[tuple[str, str], float]:
    sequence = max(
        values,
        key=values.get,
    )

    return sequence, values[sequence]


def derive_model_results(
    data: dict[str, Any],
    prefix: str,
) -> dict[int, dict[str, Any]]:
    results: dict[int, dict[str, Any]] = {}

    for initial_s1 in (0, 1):
        section = (
            f"{prefix}_values_s1_{initial_s1}"
        )

        sequence, value = best_sequence(
            data[section]
        )

        results[initial_s1] = {
            "sequence": sequence,
            "value": value,
            "all_values": data[section],
        }

    return results


# ============================================================
# Reward color encoding
# ============================================================

def mix_with_white(
    base_color: str,
    intensity: float,
) -> tuple[float, float, float]:
    base_rgb = colors.to_rgb(base_color)
    white_rgb = colors.to_rgb("#FFFFFF")

    intensity = max(
        0.0,
        min(1.0, intensity),
    )

    return tuple(
        white_component * (1.0 - intensity)
        + base_component * intensity
        for white_component, base_component
        in zip(white_rgb, base_rgb)
    )


def reward_color(
    value: float,
    maximum_absolute_reward: float,
) -> tuple[float, float, float]:
    if maximum_absolute_reward <= 0:
        normalized = 0.0

    else:
        normalized = min(
            abs(value) / maximum_absolute_reward,
            1.0,
        )

    intensity = 0.16 + 0.84 * normalized

    base = (
        POSITIVE_BASE
        if value >= 0
        else NEGATIVE_BASE
    )

    return mix_with_white(
        base,
        intensity,
    )


def transition_color(
    probability: float,
    *,
    clip_to_observed_range: bool = True,
) -> tuple[float, float, float, float]:
    cmap = plt.get_cmap(TRANSITION_CMAP)

    if clip_to_observed_range:
        scaled_probability = (
            probability - TRANSITION_COLOR_MIN
        ) / (
            TRANSITION_COLOR_MAX - TRANSITION_COLOR_MIN
        )
    else:
        scaled_probability = probability

    scaled_probability = max(
        0.0,
        min(1.0, scaled_probability),
    )

    return cmap(scaled_probability)


# ============================================================
# Drawing helpers
# ============================================================

def draw_panel_background(
    ax: plt.Axes,
    left: float,
    bottom: float,
    width: float,
    height: float,
) -> None:
    panel = FancyBboxPatch(
        (left, bottom),
        width,
        height,
        boxstyle=(
            "round,pad=0.08,"
            "rounding_size=0.12"
        ),
        linewidth=1.5,
        edgecolor=PANEL_EDGE,
        facecolor=PANEL_FILL,
        zorder=0,
    )

    ax.add_patch(panel)


def draw_state_node(
    ax: plt.Axes,
    x: float,
    y: float,
    state: str,
    rewards: dict[str, float],
    selected_action: str,
    maximum_absolute_reward: float,
) -> None:
    """
    Left half  = reward under a0.
    Right half = reward under a1.
    """

    left_color = reward_color(
        rewards["a0"],
        maximum_absolute_reward,
    )

    right_color = reward_color(
        rewards["a1"],
        maximum_absolute_reward,
    )

    left_half = Wedge(
        (x, y),
        NODE_RADIUS,
        theta1=90,
        theta2=270,
        facecolor=left_color,
        edgecolor="none",
        zorder=5,
    )

    right_half = Wedge(
        (x, y),
        NODE_RADIUS,
        theta1=-90,
        theta2=90,
        facecolor=right_color,
        edgecolor="none",
        zorder=5,
    )

    ax.add_patch(left_half)
    ax.add_patch(right_half)

    outline = Circle(
        (x, y),
        NODE_RADIUS,
        facecolor="none",
        edgecolor=BLUE,
        linewidth=2.2,
        zorder=7,
    )

    ax.add_patch(outline)

    ax.plot(
        [x, x],
        [
            y - NODE_RADIUS,
            y + NODE_RADIUS,
        ],
        color=BLUE,
        linewidth=1.2,
        zorder=7,
    )

    ax.text(
        x,
        y + 0.08,
        state,
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="black",
        zorder=9,
    )

    ax.text(
        x - 0.20,
        y - 0.22,
        r"$a_0$",
        ha="center",
        va="center",
        fontsize=8.5,
        color="black",
        zorder=9,
    )

    ax.text(
        x + 0.20,
        y - 0.22,
        r"$a_1$",
        ha="center",
        va="center",
        fontsize=8.5,
        color="black",
        zorder=9,
    )

    selected_x = (
        x - 0.20
        if selected_action == "a0"
        else x + 0.20
    )

    ax.text(
        selected_x,
        y + 0.29,
        r"$\star$",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color=RED,
        zorder=10,
    )

    if selected_action == "a0":
        theta1, theta2 = 90, 270

    else:
        theta1, theta2 = -90, 90

    selected_arc = Wedge(
        (x, y),
        NODE_RADIUS + 0.022,
        theta1=theta1,
        theta2=theta2,
        width=0.035,
        facecolor=RED,
        edgecolor=RED,
        zorder=8,
    )

    ax.add_patch(selected_arc)


def draw_vertical_state_column(
    ax: plt.Axes,
    x: float,
    rewards: dict[str, dict[str, float]],
    selected_actions: dict[int, str],
    maximum_absolute_reward: float,
) -> None:
    ax.plot(
        [x, x],
        [
            STATE_Y[STATE_ORDER[-1]],
            STATE_Y[STATE_ORDER[0]],
        ],
        color=LIGHT_GRAY,
        linewidth=2.0,
        zorder=1,
    )

    for state in STATE_ORDER:
        observed_s1 = int(state[0])

        draw_state_node(
            ax=ax,
            x=x,
            y=STATE_Y[state],
            state=state,
            rewards=rewards[state],
            selected_action=(
                selected_actions[observed_s1]
            ),
            maximum_absolute_reward=(
                maximum_absolute_reward
            ),
        )


def draw_action_diamond(
    ax: plt.Axes,
    x: float,
    y: float,
    action: str,
) -> None:
    action_index = action[-1]

    diamond = Polygon(
        [
            (x, y + ACTION_DIAMOND_HALF_HEIGHT),
            (x + ACTION_DIAMOND_HALF_WIDTH, y),
            (x, y - ACTION_DIAMOND_HALF_HEIGHT),
            (x - ACTION_DIAMOND_HALF_WIDTH, y),
        ],
        closed=True,
        facecolor="#FFF5F5",
        edgecolor=RED,
        linewidth=1.7,
        zorder=7,
    )

    ax.add_patch(diamond)

    ax.text(
        x,
        y,
        rf"$a_{action_index}$",
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color=RED,
        zorder=8,
    )


def draw_selected_action_arrow(
    ax: plt.Axes,
    source_x: float,
    source_y: float,
    diamond_x: float,
    diamond_y: float,
) -> None:
    arrow = FancyArrowPatch(
        (
            source_x + NODE_RADIUS,
            source_y,
        ),
        (
            diamond_x - ACTION_DIAMOND_HALF_WIDTH + 0.05,
            diamond_y,
        ),
        arrowstyle="-|>",
        shrinkA=0,
        shrinkB=0,
        mutation_scale=12,
        linewidth=ARROW_LINEWIDTH,
        color=RED,
        zorder=9,
    )

    ax.add_patch(arrow)


def draw_probability_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    probability: float,
) -> None:
    """
    Draw a straight observed-transition arrow.
    """
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=ARROW_LINEWIDTH,
        color=transition_color(probability),
        alpha=1.0,
        zorder=3,
    )

    ax.add_patch(arrow)


def target_state_for_transition(
    source_state: str,
    next_s1: int,
) -> str:
    """
    The printed transition data determines the next observed state S1',
    but not a full next hidden state.

    For this visualization, retain the source hidden-state index s2
    while changing the observed component to next_s1.

    Example:
        source 01 and next_s1=1 -> target 11
        source 10 and next_s1=0 -> target 00
    """
    source_s2 = source_state[1]

    return f"{next_s1}{source_s2}"


def draw_four_state_transition_structure(
    ax: plt.Axes,
    period1_x: float,
    period2_x: float,
    results: dict[int, dict[str, Any]],
    transitions: dict[str, dict[int, float]],
) -> None:
    """
    Give each complete Period-1 state its own selected-action arrow,
    action diamond, and two straight transition arrows.

    Transition probabilities are encoded by arrow color and summarized
    in the legend, keeping the crossing arrows readable.
    """

    diamond_x = (
        period1_x
        + 0.42 * (period2_x - period1_x)
    )

    for source_state in STATE_ORDER:
        observed_s1 = int(source_state[0])

        selected_sequence = results[observed_s1]["sequence"]
        first_action = selected_sequence[0]

        source_y = STATE_Y[source_state]
        diamond_y = source_y

        for next_s1 in (0, 1):
            probability = transitions[first_action][next_s1]

            target_state = target_state_for_transition(
                source_state,
                next_s1,
            )

            target_y = STATE_Y[target_state]

            start = (
                diamond_x + 0.26,
                diamond_y,
            )

            end = (
                period2_x - NODE_RADIUS - 0.05,
                target_y,
            )

            draw_probability_arrow(
                ax=ax,
                start=start,
                end=end,
                probability=probability,
            )

    # --------------------------------------------------------
    # Draw selected actions afterward so they stay visible.
    # --------------------------------------------------------

    for source_state in STATE_ORDER:
        observed_s1 = int(source_state[0])

        selected_sequence = results[observed_s1]["sequence"]
        first_action = selected_sequence[0]

        source_y = STATE_Y[source_state]

        draw_action_diamond(
            ax=ax,
            x=diamond_x,
            y=source_y,
            action=first_action,
        )

        draw_selected_action_arrow(
            ax=ax,
            source_x=period1_x,
            source_y=source_y,
            diamond_x=diamond_x,
            diamond_y=source_y,
        )

# ============================================================
# Sequence summary
# ============================================================

def format_sequence(
    sequence: tuple[str, str],
) -> str:
    first = sequence[0][-1]
    second = sequence[1][-1]

    return rf"$(a_{first},a_{second})$"


def draw_sequence_value_box(
    ax: plt.Axes,
    center_x: float,
    bottom_y: float,
    results: dict[int, dict[str, Any]],
) -> None:
    sequence_0 = results[0]["sequence"]
    sequence_1 = results[1]["sequence"]

    value_0 = results[0]["value"]
    value_1 = results[1]["value"]

    line_1 = (
        rf"$s_1=0$: "
        rf"{format_sequence(sequence_0)}, "
        rf"value $={value_0:.3f}$"
    )

    line_2 = (
        rf"$s_1=1$: "
        rf"{format_sequence(sequence_1)}, "
        rf"value $={value_1:.3f}$"
    )

    ax.text(
        center_x,
        bottom_y,
        line_1 + "\n" + line_2,
        ha="center",
        va="center",
        fontsize=9.5,
        color="black",
        bbox={
            "boxstyle": "round,pad=0.48",
            "facecolor": "#F4F7FF",
            "edgecolor": BLUE,
            "linewidth": 1.15,
        },
        zorder=12,
    )


# ============================================================
# Model panel
# ============================================================

def draw_model_panel(
    ax: plt.Axes,
    panel_left: float,
    title: str,
    rewards: dict[str, dict[str, float]],
    results: dict[int, dict[str, Any]],
    transitions: dict[str, dict[int, float]],
    maximum_absolute_reward: float,
) -> None:
    panel_width = 7.10
    panel_bottom = -2.65
    panel_height = 5.75

    draw_panel_background(
        ax=ax,
        left=panel_left,
        bottom=panel_bottom,
        width=panel_width,
        height=panel_height,
    )

    panel_center = (
        panel_left + panel_width / 2
    )

    period1_x = panel_left + 1.05
    period2_x = panel_left + 6.05

    ax.text(
        panel_center,
        3.50,
        title,
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
        color=BLUE,
    )

    ax.text(
        period1_x,
        2.72,
        "Period 1",
        ha="center",
        va="center",
        fontsize=12.5,
        fontweight="bold",
        color=GRAY,
    )

    ax.text(
        period2_x,
        2.72,
        "Period 2",
        ha="center",
        va="center",
        fontsize=12.5,
        fontweight="bold",
        color=GRAY,
    )

    period1_actions = {
        observed_s1:
            results[observed_s1]["sequence"][0]
        for observed_s1 in (0, 1)
    }

    period2_actions = {
        observed_s1:
            results[observed_s1]["sequence"][1]
        for observed_s1 in (0, 1)
    }

    # Draw transition arrows before nodes.
    draw_four_state_transition_structure(
        ax=ax,
        period1_x=period1_x,
        period2_x=period2_x,
        results=results,
        transitions=transitions,
    )

    draw_vertical_state_column(
        ax=ax,
        x=period1_x,
        rewards=rewards,
        selected_actions=period1_actions,
        maximum_absolute_reward=(
            maximum_absolute_reward
        ),
    )

    draw_vertical_state_column(
        ax=ax,
        x=period2_x,
        rewards=rewards,
        selected_actions=period2_actions,
        maximum_absolute_reward=(
            maximum_absolute_reward
        ),
    )

    draw_sequence_value_box(
        ax=ax,
        center_x=panel_center,
        bottom_y=-2.31,
        results=results,
    )


# ============================================================
# Legends
# ============================================================

def draw_reward_legend_entry(
    ax: plt.Axes,
    center_x: float,
    y: float,
) -> None:
    values = [
        -1.0,
        -0.4,
        0.4,
        1.0,
    ]

    width = 0.30
    height = 0.17
    swatch_x = center_x - 0.60

    for index, value in enumerate(values):
        rectangle = FancyBboxPatch(
            (
                swatch_x + index * width,
                y,
            ),
            width,
            height,
            boxstyle="square,pad=0",
            linewidth=0,
            facecolor=reward_color(
                value,
                1.0,
            ),
            zorder=5,
        )

        ax.add_patch(rectangle)

    ax.text(
        swatch_x - 0.06,
        y + height / 2,
        "negative",
        ha="right",
        va="center",
        fontsize=7.5,
        color=GRAY,
    )

    ax.text(
        swatch_x + 4 * width + 0.06,
        y + height / 2,
        "positive",
        ha="left",
        va="center",
        fontsize=7.5,
        color=GRAY,
    )

    ax.text(
        center_x,
        y + 0.31,
        "Reward sign and magnitude",
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color=GRAY,
    )


def draw_transition_legend_entry(
    ax: plt.Axes,
    center_x: float,
    y: float,
) -> None:
    ax.text(
        center_x,
        y + 0.31,
        "Observed transition probability",
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color=GRAY,
    )

    arrow_y = y + 0.085
    arrow_specs = [
        (0.05, "0.05"),
        (0.25, "0.25"),
        (0.50, "0.50"),
        (0.75, "0.75"),
        (0.95, "0.95"),
    ]
    spacing = 0.58
    first_x = center_x - 2 * spacing

    for index, (probability, label) in enumerate(arrow_specs):
        x = first_x + index * spacing
        arrow = FancyArrowPatch(
            (x - 0.18, arrow_y),
            (x + 0.18, arrow_y),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=ARROW_LINEWIDTH,
            color=transition_color(
                probability,
                clip_to_observed_range=False,
            ),
            alpha=1.0,
            zorder=5,
        )
        ax.add_patch(arrow)
        ax.text(
            x,
            y - 0.13,
            label,
            ha="center",
            va="center",
            fontsize=7.2,
            color=GRAY,
        )

    ax.text(
        center_x,
        y - 0.35,
        r"$P(S_1'\mid S_1,a)$",
        ha="center",
        va="center",
        fontsize=8.0,
        color=GRAY,
    )


def draw_visual_legend(
    ax: plt.Axes,
    center_x: float,
    y: float,
) -> None:
    draw_reward_legend_entry(
        ax=ax,
        center_x=center_x - 1.85,
        y=y,
    )

    draw_transition_legend_entry(
        ax=ax,
        center_x=center_x + 1.85,
        y=y,
    )


# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render the two-panel observed-model case-study figure."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help=(
            "Case data file. Examples: "
            "graphing/case_study-c1.txt or "
            "graphing/case_study-c2.txt."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for the rendered PNG/PDF.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help=(
            "Output basename without extension. "
            "Defaults to the input file stem."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_file = args.input
    output_name = args.output_name or input_file.stem
    png_output = args.output_dir / f"{output_name}.png"
    pdf_output = args.output_dir / f"{output_name}.pdf"

    data = read_case_file(input_file)

    original_results = derive_model_results(
        data,
        "original",
    )

    attacked_results = derive_model_results(
        data,
        "attacked",
    )

    maximum_absolute_reward = max(
        abs(value)
        for state_rewards
        in data["rewards"].values()
        for value
        in state_rewards.values()
    )

    fig, ax = plt.subplots(
        figsize=(18.5, 8.0),
    )

    draw_model_panel(
        ax=ax,
        panel_left=0.0,
        title="Original observed model",
        rewards=data["rewards"],
        results=original_results,
        transitions=data["original_transitions"],
        maximum_absolute_reward=(
            maximum_absolute_reward
        ),
    )

    draw_model_panel(
        ax=ax,
        panel_left=7.65,
        title="Attacker-induced observed model",
        rewards=data["rewards"],
        results=attacked_results,
        transitions=data["attacked_transitions"],
        maximum_absolute_reward=(
            maximum_absolute_reward
        ),
    )

    draw_visual_legend(
        ax=ax,
        center_x=7.35,
        y=-3.26,
    )

    ax.text(
        7.35,
        -3.78,
        (
            r"Each complete Period-1 state has its own "
            r"selected-action arrow. "
            r"Each circle is split by action: "
            r"left $=a_0$, right $=a_1$."
        ),
        ha="center",
        va="center",
        fontsize=9.2,
        color="black",
    )

    meta = data["meta"]

    ax.text(
        7.35,
        4.10,
        (
            rf"Seed {meta['seed']}, "
            rf"control ${meta['control']:.2f}$, "
            rf"observation information "
            rf"${meta['obs_info']:.2f}$, "
            rf"LP margin ${meta['margin']:+.4f}$"
        ),
        ha="center",
        va="center",
        fontsize=10,
        color=GRAY,
    )

    ax.set_xlim(
        -0.35,
        15.05,
    )

    ax.set_ylim(
        -4.05,
        4.40,
    )

    ax.set_aspect("equal")
    ax.axis("off")

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        png_output,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.12,
        facecolor="white",
    )

    fig.savefig(
        pdf_output,
        bbox_inches="tight",
        pad_inches=0.12,
        facecolor="white",
    )

    plt.close(fig)

    print(
        "Original best sequence at s1=0:",
        original_results[0]["sequence"],
    )

    print(
        "Original best sequence at s1=1:",
        original_results[1]["sequence"],
    )

    print(
        "Attacked best sequence at s1=0:",
        attacked_results[0]["sequence"],
    )

    print(
        "Attacked best sequence at s1=1:",
        attacked_results[1]["sequence"],
    )

    print(f"Input: {input_file}")
    print(f"Saved PNG: {png_output}")
    print(f"Saved PDF: {pdf_output}")


if __name__ == "__main__":
    main()
