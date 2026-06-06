from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / ".matplotlib-cache"),
)
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.append(os.path.dirname(__file__))

from tiger_finite_horizon_demo import build_tiger, solve_finite_horizon


def main() -> None:
    horizon = 3
    output_path = (
        Path(__file__).resolve().parents[1] / "outputs" / "tiger_horizon_3_alpha_lines.png"
    )

    pomdp = build_tiger()
    stages = solve_finite_horizon(pomdp, horizon=horizon)
    vectors = stages[horizon]

    plot_alpha_lines(pomdp, vectors, output_path)
    print(f"Saved exact alpha-line plot to: {output_path}")


def plot_alpha_lines(pomdp, vectors, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)

    palette = {
        "listen": "#2f6f73",
        "open_left": "#b34f3f",
        "open_right": "#6f5aa7",
    }
    envelope_segments = exact_upper_envelope_segments(vectors)

    envelope_values = [
        alpha.value_at(np.array([x, 1.0 - x]))
        for start, end, alpha in envelope_segments
        for x in (start, end)
    ]
    y_min = min(envelope_values)
    y_max = max(envelope_values)
    padding = max((y_max - y_min) * 0.18, 1.0)

    for alpha in vectors:
        action_name = pomdp.action_names[alpha.action]
        y_at_zero, y_at_one = alpha_line_endpoints(alpha)
        axis.plot(
            [0.0, 1.0],
            [y_at_zero, y_at_one],
            color=palette.get(action_name, "0.45"),
            linewidth=1.2,
            alpha=0.28,
        )

    legend_labels = set()
    for start, end, alpha in envelope_segments:
        action_name = pomdp.action_names[alpha.action]
        y_start = alpha.value_at(np.array([start, 1.0 - start]))
        y_end = alpha.value_at(np.array([end, 1.0 - end]))
        label = action_name if action_name not in legend_labels else None
        legend_labels.add(action_name)
        axis.plot(
            [start, end],
            [y_start, y_end],
            color=palette.get(action_name, "black"),
            linewidth=4.0,
            solid_capstyle="round",
            label=label,
        )

    axis.legend(title="Envelope action", loc="best")
    axis.set_title("Tiger finite-horizon alpha-vectors")
    axis.set_xlabel(f"Belief P({pomdp.state_names[0]})")
    axis.set_ylabel("Expected value")
    axis.set_xlim(0.0, 1.08)
    axis.set_ylim(y_min - padding, y_max + padding)
    axis.grid(True, alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)

    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def alpha_line_endpoints(alpha) -> tuple[float, float]:
    return float(alpha.values[1]), float(alpha.values[0])


def exact_upper_envelope_segments(vectors, tol: float = 1e-10):
    breakpoints = {0.0, 1.0}
    for i, left in enumerate(vectors):
        left_y0, left_y1 = alpha_line_endpoints(left)
        left_slope = left_y1 - left_y0

        for right in vectors[i + 1 :]:
            right_y0, right_y1 = alpha_line_endpoints(right)
            right_slope = right_y1 - right_y0
            denominator = left_slope - right_slope
            if abs(denominator) <= tol:
                continue

            crossing = (right_y0 - left_y0) / denominator
            if -tol <= crossing <= 1.0 + tol:
                breakpoints.add(float(np.clip(crossing, 0.0, 1.0)))

    xs = sorted(breakpoints)
    segments = []

    for start, end in zip(xs, xs[1:]):
        if end - start <= tol:
            continue

        midpoint = (start + end) / 2.0
        belief = np.array([midpoint, 1.0 - midpoint])
        winner = max(vectors, key=lambda alpha: alpha.value_at(belief))

        if segments and segments[-1][2] is winner and abs(segments[-1][1] - start) <= tol:
            segments[-1] = (segments[-1][0], end, winner)
        else:
            segments.append((start, end, winner))

    return segments


if __name__ == "__main__":
    main()
