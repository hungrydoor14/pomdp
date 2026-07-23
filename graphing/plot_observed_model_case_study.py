import os
from pathlib import Path

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

from common import (
    STATE_ORDER,
    derive_model_results,
    read_case_file,
)

from style import (
    ACTION_DIAMOND_HALF_HEIGHT,
    ACTION_DIAMOND_HALF_WIDTH,
    ARROW_LINEWIDTH,
    AXIS_LABEL_SIZE,
    BLUE,
    GRAY,
    LEGEND_SIZE,
    LIGHT_GRAY,
    NEGATIVE_BASE,
    NODE_RADIUS,
    NOTE_SIZE,
    PANEL_EDGE,
    PANEL_FILL,
    PANEL_TITLE_SIZE,
    POSITIVE_BASE,
    RED,
    TICK_SIZE,
    TITLE_SIZE,
    TRANSITION_CMAP,
    TRANSITION_COLOR_MAX,
    TRANSITION_COLOR_MIN,
    VALUE_LABEL_SIZE,
)


# ============================================================
# Paths
# ============================================================

INPUT_FILE = Path("graphing/case_study-l1.json")
OUTPUT_DIR = Path("outputs/l1")
CURRENT_PART = "pd"
THEOREM = "l1"
PNG_OUTPUT = OUTPUT_DIR / f"{THEOREM}-{CURRENT_PART}_{INPUT_FILE.stem}.png"
PDF_OUTPUT = OUTPUT_DIR / f"{THEOREM}-{CURRENT_PART}_{INPUT_FILE.stem}.pdf"


# ============================================================
# Visual configuration
# ============================================================

STATE_Y = {
    "00": 1.85,
    "01": 0.65,
    "10": -0.55,
    "11": -1.75,
}


# ============================================================
# Reward color encoding
# ============================================================

def mix_with_white(base_color, intensity):
    base_rgb = colors.to_rgb(base_color)
    white_rgb = colors.to_rgb("#FFFFFF")

    intensity = max(0.0, min(1.0, intensity))

    return tuple(
        white_component * (1.0 - intensity)
        + base_component * intensity
        for white_component, base_component
        in zip(white_rgb, base_rgb)
    )


def reward_color(value, maximum_absolute_reward):
    if maximum_absolute_reward <= 0:
        normalized = 0.0

    else:
        normalized = min(abs(value) / maximum_absolute_reward, 1.0)

    intensity = 0.16 + 0.84 * normalized

    base = (
        POSITIVE_BASE
        if value >= 0
        else NEGATIVE_BASE
    )

    return mix_with_white(base, intensity)


def transition_color(probability, *, clip_to_observed_range=True):
    cmap = plt.get_cmap(TRANSITION_CMAP)

    if clip_to_observed_range:
        scaled_probability = (
            probability - TRANSITION_COLOR_MIN
        ) / (
            TRANSITION_COLOR_MAX - TRANSITION_COLOR_MIN
        )
    else:
        scaled_probability = probability

    scaled_probability = max(0.0, min(1.0, scaled_probability))

    return cmap(scaled_probability)


# ============================================================
# Drawing helpers
# ============================================================

def draw_panel_background(ax, left, bottom, width, height):
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


def draw_state_node(ax, x, y, state, rewards, selected_action, maximum_absolute_reward):
    """
    Left half  = reward under a0.
    Right half = reward under a1.
    """

    left_color = reward_color(rewards["a0"], maximum_absolute_reward)

    right_color = reward_color(rewards["a1"], maximum_absolute_reward)

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
        fontsize=PANEL_TITLE_SIZE,
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
        fontsize=VALUE_LABEL_SIZE,
        color="black",
        zorder=9,
    )

    ax.text(
        x + 0.20,
        y - 0.22,
        r"$a_1$",
        ha="center",
        va="center",
        fontsize=VALUE_LABEL_SIZE,
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
        fontsize=PANEL_TITLE_SIZE,
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
    ax,
    x,
    rewards,
    selected_actions,
    maximum_absolute_reward,
    visible_states,
):
    visible_state_order = [state for state in STATE_ORDER if state in visible_states]

    if not visible_state_order:
        return

    ax.plot(
        [x, x],
        [
            STATE_Y[visible_state_order[-1]],
            STATE_Y[visible_state_order[0]],
        ],
        color=LIGHT_GRAY,
        linewidth=2.0,
        zorder=1,
    )

    for state in visible_state_order:
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


def draw_action_diamond(ax, x, y, action):
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
        fontsize=VALUE_LABEL_SIZE,
        fontweight="bold",
        color=RED,
        zorder=8,
    )


def draw_selected_action_arrow(ax, source_x, source_y, diamond_x, diamond_y):
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


def draw_probability_arrow(ax, start, end, probability):
    """
    Draw a straight observed-transition arrow.
    """
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=ARROW_LINEWIDTH,
        color=transition_color(
            probability,
            clip_to_observed_range=False,
        ),
        alpha=1.0,
        zorder=3,
    )

    ax.add_patch(arrow)


def target_state_for_transition(source_state, next_s1, period2_visible_states):
    """
    The printed transition data determines the next observed state S1',
    but not a full next hidden state.

    For this visualization, retain the source hidden-state index s2
    while changing the observed component to next_s1.

    Example:
        source 01 and next_s1=1 -> target 11
        source 10 and next_s1=0 -> target 00
    """
    candidates = [
        state
        for state in period2_visible_states
        if int(state[0]) == next_s1
    ]

    if len(candidates) == 1:
        return candidates[0]

    source_s2 = source_state[1]

    return f"{next_s1}{source_s2}"


def transition_for_source(transitions, observed_s1, action):
    """Read a state-dependent transition, falling back to the legacy format."""
    state_action_key = (observed_s1, action)

    if state_action_key in transitions:
        return transitions[state_action_key]

    return transitions[action]


def draw_four_state_transition_structure(
    ax,
    period1_x,
    period2_x,
    results,
    transitions,
    period1_visible_states,
    period2_visible_states,
):
    """
    Give each complete Period-1 state its own selected-action arrow,
    action diamond, and two straight transition arrows.

    Transition probabilities are encoded by arrow color and summarized
    in the legend, keeping the crossing arrows readable.
    """

    diamond_x = (period1_x + 0.42 * (period2_x - period1_x))

    for source_state in STATE_ORDER:
        if source_state not in period1_visible_states:
            continue

        observed_s1 = int(source_state[0])

        selected_sequence = results[observed_s1]["sequence"]
        first_action = selected_sequence[0]

        source_y = STATE_Y[source_state]
        diamond_y = source_y

        source_transition = transition_for_source(
            transitions,
            observed_s1,
            first_action,
        )

        for next_s1 in (0, 1):
            probability = source_transition[next_s1]

            if probability <= 1e-12:
                continue

            target_state = target_state_for_transition(
                source_state,
                next_s1,
                period2_visible_states,
            )

            if target_state not in period2_visible_states:
                continue

            target_y = STATE_Y[target_state]

            start = (diamond_x + 0.26, diamond_y)

            end = (period2_x - NODE_RADIUS - 0.05, target_y)

            draw_probability_arrow(ax=ax, start=start, end=end, probability=probability)

    # --------------------------------------------------------
    # Draw selected actions afterward so they stay visible.
    # --------------------------------------------------------

    for source_state in STATE_ORDER:
        if source_state not in period1_visible_states:
            continue

        observed_s1 = int(source_state[0])

        selected_sequence = results[observed_s1]["sequence"]
        first_action = selected_sequence[0]

        source_y = STATE_Y[source_state]

        draw_action_diamond(ax=ax, x=diamond_x, y=source_y, action=first_action)

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

def format_sequence(sequence):
    first = sequence[0][-1]
    second = sequence[1][-1]

    return rf"$(a_{first},a_{second})$"


def draw_sequence_value_box(ax, center_x, bottom_y, results):
    sequence_0 = results[0]["sequence"]
    sequence_1 = results[1]["sequence"]

    value_0 = results[0]["value"]
    value_1 = results[1]["value"]

    line_1 = (rf"$s_1=0$: " rf"{format_sequence(sequence_0)}, " rf"value $={value_0:.3f}$")

    line_2 = (rf"$s_1=1$: " rf"{format_sequence(sequence_1)}, " rf"value $={value_1:.3f}$")

    ax.text(
        center_x,
        bottom_y,
        line_1 + "\n" + line_2,
        ha="center",
        va="center",
        fontsize=VALUE_LABEL_SIZE,
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
    ax,
    panel_left,
    title,
    rewards,
    results,
    transitions,
    maximum_absolute_reward,
    period1_visible_states,
    period2_visible_states,
):
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

    panel_center = (panel_left + panel_width / 2)

    period1_x = panel_left + 1.05
    period2_x = panel_left + 6.05

    ax.text(
        panel_center,
        3.50,
        title,
        ha="center",
        va="center",
        fontsize=TITLE_SIZE,
        fontweight="bold",
        color=BLUE,
    )

    ax.text(
        period1_x,
        2.72,
        "Period 1",
        ha="center",
        va="center",
        fontsize=AXIS_LABEL_SIZE,
        fontweight="bold",
        color=GRAY,
    )

    ax.text(
        period2_x,
        2.72,
        "Period 2",
        ha="center",
        va="center",
        fontsize=AXIS_LABEL_SIZE,
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
        period1_visible_states=period1_visible_states,
        period2_visible_states=period2_visible_states,
    )

    draw_vertical_state_column(
        ax=ax,
        x=period1_x,
        rewards=rewards,
        selected_actions=period1_actions,
        maximum_absolute_reward=(
            maximum_absolute_reward
        ),
        visible_states=period1_visible_states,
    )

    draw_vertical_state_column(
        ax=ax,
        x=period2_x,
        rewards=rewards,
        selected_actions=period2_actions,
        maximum_absolute_reward=(
            maximum_absolute_reward
        ),
        visible_states=period2_visible_states,
    )

    draw_sequence_value_box(ax=ax, center_x=panel_center, bottom_y=-2.31, results=results)


# ============================================================
# Legends
# ============================================================

def draw_reward_legend_entry(ax, center_x, y):
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
        fontsize=NOTE_SIZE,
        color=GRAY,
    )

    ax.text(
        swatch_x + 4 * width + 0.06,
        y + height / 2,
        "positive",
        ha="left",
        va="center",
        fontsize=NOTE_SIZE,
        color=GRAY,
    )

    ax.text(
        center_x,
        y + 0.31,
        "Reward sign and magnitude",
        ha="center",
        va="center",
        fontsize=LEGEND_SIZE,
        fontweight="bold",
        color=GRAY,
    )


def draw_transition_legend_entry(ax, center_x, y):
    ax.text(
        center_x,
        y + 0.31,
        "Observed transition probability",
        ha="center",
        va="center",
        fontsize=LEGEND_SIZE,
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
        ax.text(x, y - 0.13, label, ha="center", va="center", fontsize=NOTE_SIZE, color=GRAY)

    ax.text(
        center_x,
        y - 0.35,
        r"$P(S_1'\mid S_1,a)$",
        ha="center",
        va="center",
        fontsize=NOTE_SIZE,
        color=GRAY,
    )


def draw_visual_legend(ax, center_x, y):
    draw_reward_legend_entry(ax=ax, center_x=center_x - 2.55, y=y)

    draw_transition_legend_entry(ax=ax, center_x=center_x + 2.55, y=y)


# ============================================================
# Main
# ============================================================

def main():
    data = read_case_file(INPUT_FILE)

    all_states = set(STATE_ORDER)
    attacked_period2_visible_states = {
        state
        for state in STATE_ORDER
        if data.get(
            "period2_state_counts",
            data.get("hidden_state_counts", {}),
        ).get(state, 1) > 0
    }

    original_results = derive_model_results(data, "original")

    attacked_results = derive_model_results(data, "attacked")

    maximum_absolute_reward = max(
        abs(value)
        for state_rewards
        in data["rewards"].values()
        for value
        in state_rewards.values()
    )

    fig, ax = plt.subplots(figsize=(18.5, 8.0))

    draw_model_panel(
        ax=ax,
        panel_left=0.0,
        title="Original observed model",
        rewards=data["rewards"],
        results=original_results,
        transitions=data.get(
            "original_transitions_by_s1",
            data["original_transitions"],
        ),
        maximum_absolute_reward=(
            maximum_absolute_reward
        ),
        period1_visible_states=all_states,
        period2_visible_states=all_states,
    )

    draw_model_panel(
        ax=ax,
        panel_left=7.65,
        title="Attacker-induced observed model",
        rewards=data["rewards"],
        results=attacked_results,
        transitions=data.get(
            "attacked_transitions_by_s1",
            data["attacked_transitions"],
        ),
        maximum_absolute_reward=(
            maximum_absolute_reward
        ),
        period1_visible_states=all_states,
        period2_visible_states=attacked_period2_visible_states,
    )

    draw_visual_legend(ax=ax, center_x=7.35, y=-3.45)

    ax.text(
        7.35,
        -4.15,
        (
            r"Each complete Period-1 state has its own "
            r"selected-action arrow. "
            r"Each circle is split by action: "
            r"left $=a_0$, right $=a_1$."
        ),
        ha="center",
        va="center",
        fontsize=NOTE_SIZE,
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
        fontsize=NOTE_SIZE,
        color=GRAY,
    )

    ax.set_xlim(-0.35, 15.05)

    ax.set_ylim(-4.55, 4.40)

    ax.set_aspect("equal")
    ax.axis("off")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig.savefig(PNG_OUTPUT, dpi=300, bbox_inches="tight", pad_inches=0.12, facecolor="white")

    #fig.savefig(PDF_OUTPUT, bbox_inches="tight", pad_inches=0.12, facecolor="white")

    plt.close(fig)

    print("Original best sequence at s1=0:", original_results[0]["sequence"])

    print("Original best sequence at s1=1:", original_results[1]["sequence"])

    print("Attacked best sequence at s1=0:", attacked_results[0]["sequence"])

    print("Attacked best sequence at s1=1:", attacked_results[1]["sequence"])

    print(f"Input: {INPUT_FILE}")
    print(f"Saved PNG: {PNG_OUTPUT}")
    #print(f"Saved PDF: {PDF_OUTPUT}")


if __name__ == "__main__":
    main()
