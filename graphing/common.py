from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


ACTIONS = [
    "a0",
    "a1",
]

STATE_ORDER = [
    "00",
    "01",
    "10",
    "11",
]

SEQUENCES = [
    ("a0", "a0"),
    ("a0", "a1"),
    ("a1", "a0"),
    ("a1", "a1"),
]


def read_case_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}"
        )

    if path.suffix == ".json":
        return read_json_case_file(path)

    return read_text_case_file(path)


def read_text_case_file(path: Path) -> dict[str, Any]:
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
        "original_transitions_by_s1",
        "attacked_transitions_by_s1",
        "attacker_policy",
        "hidden_state_counts",
        "observed_state_counts",
        "observed_state_action_counts",
        "full_state_action_counts",
        "coverage",
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
                parse_meta_row(
                    data[current_section],
                    parts,
                )

            elif current_section == "rewards":
                parse_reward_row(
                    data[current_section],
                    parts,
                    line_number,
                )

            elif "values_s1_" in current_section:
                parse_value_row(
                    data[current_section],
                    parts,
                    line_number,
                )

            elif current_section in {
                "original_b",
                "attacked_b",
            }:
                parse_belief_row(
                    data[current_section],
                    parts,
                    line_number,
                )

            elif current_section in {
                "original_transitions",
                "attacked_transitions",
            }:
                parse_action_transition_row(
                    data[current_section],
                    parts,
                    line_number,
                )

            elif current_section in {
                "original_transitions_by_s1",
                "attacked_transitions_by_s1",
            }:
                parse_state_action_transition_row(
                    data[current_section],
                    parts,
                    line_number,
                )

            elif current_section == "attacker_policy":
                parse_attacker_policy_row(
                    data[current_section],
                    parts,
                    line_number,
                )

            elif current_section == "hidden_state_counts":
                parse_hidden_state_count_row(
                    data[current_section],
                    parts,
                    line_number,
                )

            elif current_section == "observed_state_counts":
                parse_observed_state_count_row(
                    data[current_section],
                    parts,
                    line_number,
                )

            elif current_section == "observed_state_action_counts":
                parse_observed_state_action_count_row(
                    data[current_section],
                    parts,
                    line_number,
                )

            elif current_section == "full_state_action_counts":
                parse_full_state_action_count_row(
                    data[current_section],
                    parts,
                    line_number,
                )

            elif current_section == "coverage":
                parse_coverage_row(
                    data[current_section],
                    parts,
                    line_number,
                )

    validate_input(data)

    return data


def read_json_case_file(path: Path) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        raw_data = json.load(file)

    data = normalize_json_case(raw_data)
    validate_input(data)

    return data


def write_json_case_file(
    data: dict[str, Any],
    path: Path,
) -> None:
    validate_input(data)
    json_data = case_to_json(data)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            json_data,
            file,
            indent=2,
            sort_keys=False,
            allow_nan=True,
        )
        file.write("\n")


def sequence_key(
    sequence: tuple[str, str],
) -> str:
    return ",".join(sequence)


def parse_sequence_key(
    key: str,
) -> tuple[str, str]:
    first_action, second_action = key.split(",")
    return first_action, second_action


def case_to_json(
    data: dict[str, Any],
) -> dict[str, Any]:
    json_data: dict[str, Any] = {
        "meta": data["meta"],
        "rewards": data["rewards"],
        "values": {
            prefix: {
                str(s1): {
                    sequence_key(sequence): value
                    for sequence, value
                    in data[f"{prefix}_values_s1_{s1}"].items()
                }
                for s1 in (0, 1)
            }
            for prefix in ("original", "attacked")
        },
        "b": {
            prefix: {
                str(s1): {
                    action: data[f"{prefix}_b"][(s1, action)]
                    for action in ACTIONS
                }
                for s1 in (0, 1)
            }
            for prefix in ("original", "attacked")
        },
        "transitions": {
            prefix: {
                action: {
                    str(next_s1): probability
                    for next_s1, probability
                    in data[f"{prefix}_transitions"][action].items()
                }
                for action in ACTIONS
            }
            for prefix in ("original", "attacked")
        },
    }

    if (
        "original_transitions_by_s1" in data
        and "attacked_transitions_by_s1" in data
    ):
        json_data["transitions_by_s1"] = {
            prefix: {
                str(s1): {
                    action: {
                        str(next_s1): probability
                        for next_s1, probability
                        in data[f"{prefix}_transitions_by_s1"][
                            (s1, action)
                        ].items()
                    }
                    for action in ACTIONS
                    if (s1, action)
                    in data[f"{prefix}_transitions_by_s1"]
                }
                for s1 in (0, 1)
            }
            for prefix in ("original", "attacked")
        }

    optional_direct_sections = (
        "attacker_policy",
        "hidden_state_counts",
        "observed_state_counts",
        "coverage",
    )

    for section in optional_direct_sections:
        if section in data:
            json_data[section] = {
                str(key): json_safe_value(value)
                for key, value in data[section].items()
            }

    if "observed_state_action_counts" in data:
        json_data["observed_state_action_counts"] = {
            str(s1): {
                action: data["observed_state_action_counts"][
                    (s1, action)
                ]
                for action in ACTIONS
            }
            for s1 in (0, 1)
        }

    if "full_state_action_counts" in data:
        json_data["full_state_action_counts"] = {
            state: {
                action: data["full_state_action_counts"][
                    (state, action)
                ]
                for action in ACTIONS
            }
            for state in STATE_ORDER
        }

    return json_data


def json_safe_value(
    value: Any,
) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None

    return value


def normalize_json_case(
    raw_data: dict[str, Any],
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "meta": normalize_meta(raw_data["meta"]),
        "rewards": raw_data["rewards"],
    }

    for prefix in ("original", "attacked"):
        for s1 in (0, 1):
            section = f"{prefix}_values_s1_{s1}"
            data[section] = {
                parse_sequence_key(sequence): float(value)
                for sequence, value
                in raw_data["values"][prefix][str(s1)].items()
            }

        data[f"{prefix}_b"] = {
            (s1, action): float(
                raw_data["b"][prefix][str(s1)][action]
            )
            for s1 in (0, 1)
            for action in ACTIONS
        }

        data[f"{prefix}_transitions"] = {
            action: {
                int(next_s1): float(probability)
                for next_s1, probability
                in raw_data["transitions"][prefix][action].items()
            }
            for action in ACTIONS
        }

    if "transitions_by_s1" in raw_data:
        for prefix in ("original", "attacked"):
            data[f"{prefix}_transitions_by_s1"] = {
                (s1, action): {
                    int(next_s1): float(probability)
                    for next_s1, probability
                    in raw_data["transitions_by_s1"][prefix][
                        str(s1)
                    ][action].items()
                }
                for s1 in (0, 1)
                for action in ACTIONS
                if action
                in raw_data["transitions_by_s1"][prefix][str(s1)]
            }

    if "attacker_policy" in raw_data:
        data["attacker_policy"] = {
            state: (
                math.nan
                if probability is None
                else float(probability)
            )
            for state, probability
            in raw_data["attacker_policy"].items()
        }

    if "hidden_state_counts" in raw_data:
        data["hidden_state_counts"] = {
            state: int(count)
            for state, count
            in raw_data["hidden_state_counts"].items()
        }

    if "observed_state_counts" in raw_data:
        data["observed_state_counts"] = {
            int(s1): int(count)
            for s1, count
            in raw_data["observed_state_counts"].items()
        }

    if "observed_state_action_counts" in raw_data:
        data["observed_state_action_counts"] = {
            (int(s1), action): int(count)
            for s1, action_counts
            in raw_data["observed_state_action_counts"].items()
            for action, count
            in action_counts.items()
        }

    if "full_state_action_counts" in raw_data:
        data["full_state_action_counts"] = {
            (state, action): int(count)
            for state, action_counts
            in raw_data["full_state_action_counts"].items()
            for action, count
            in action_counts.items()
        }

    if "coverage" in raw_data:
        data["coverage"] = {
            quantity: bool(covered)
            for quantity, covered
            in raw_data["coverage"].items()
        }

    return data


def normalize_meta(
    meta: dict[str, Any],
) -> dict[str, Any]:
    normalized = dict(meta)

    if "target" in normalized:
        normalized["target"] = tuple(normalized["target"])

    if "seed" in normalized:
        normalized["seed"] = int(normalized["seed"])

    for key in (
        "control",
        "obs_info",
        "hidden_effect",
        "action_control",
        "action_effect",
        "margin",
    ):
        if key in normalized:
            normalized[key] = float(normalized[key])

    return normalized


def parse_meta_row(
    section: dict[str, Any],
    parts: list[str],
) -> None:
    key = parts[0]

    if key == "target":
        section[key] = (
            parts[1],
            parts[2],
        )

    elif key == "seed":
        section[key] = int(parts[1])

    elif key == "case":
        section[key] = parts[1]

    else:
        section[key] = float(parts[1])


def parse_reward_row(
    section: dict[str, dict[str, float]],
    parts: list[str],
    line_number: int,
) -> None:
    if len(parts) != 3:
        raise ValueError(
            f"Line {line_number}: "
            "reward rows require "
            "STATE A0_VALUE A1_VALUE."
        )

    state, a0_value, a1_value = parts

    section[state] = {
        "a0": float(a0_value),
        "a1": float(a1_value),
    }


def parse_value_row(
    section: dict[tuple[str, str], float],
    parts: list[str],
    line_number: int,
) -> None:
    if len(parts) != 3:
        raise ValueError(
            f"Line {line_number}: "
            "sequence rows require "
            "FIRST_ACTION SECOND_ACTION VALUE."
        )

    first_action, second_action, value = parts

    section[(first_action, second_action)] = float(value)


def parse_belief_row(
    section: dict[tuple[int, str], float],
    parts: list[str],
    line_number: int,
) -> None:
    if len(parts) != 3:
        raise ValueError(
            f"Line {line_number}: "
            "belief rows require "
            "S1 ACTION PROBABILITY."
        )

    s1, action, probability = parts

    section[(int(s1), action)] = float(probability)


def parse_action_transition_row(
    section: dict[str, dict[int, float]],
    parts: list[str],
    line_number: int,
) -> None:
    if len(parts) != 3:
        raise ValueError(
            f"Line {line_number}: "
            "transition rows require "
            "ACTION P_NEXT_0 P_NEXT_1."
        )

    action, p0, p1 = parts

    section[action] = {
        0: float(p0),
        1: float(p1),
    }


def parse_state_action_transition_row(
    section: dict[tuple[int, str], dict[int, float]],
    parts: list[str],
    line_number: int,
) -> None:
    if len(parts) != 4:
        raise ValueError(
            f"Line {line_number}: "
            "state-action transition rows require "
            "S1 ACTION P_NEXT_0 P_NEXT_1."
        )

    s1, action, p0, p1 = parts

    section[(int(s1), action)] = {
        0: float(p0),
        1: float(p1),
    }


def parse_attacker_policy_row(
    section: dict[str, float],
    parts: list[str],
    line_number: int,
) -> None:
    if len(parts) != 2:
        raise ValueError(
            f"Line {line_number}: "
            "attacker policy rows require "
            "STATE PI_DAGGER_A1."
        )

    state, probability = parts

    section[state] = (
        math.nan
        if probability.lower() == "nan"
        else float(probability)
    )


def parse_hidden_state_count_row(
    section: dict[str, int],
    parts: list[str],
    line_number: int,
) -> None:
    if len(parts) != 2:
        raise ValueError(
            f"Line {line_number}: "
            "hidden-state count rows require "
            "STATE COUNT."
        )

    state, count = parts
    section[state] = int(count)


def parse_observed_state_count_row(
    section: dict[int, int],
    parts: list[str],
    line_number: int,
) -> None:
    if len(parts) != 2:
        raise ValueError(
            f"Line {line_number}: "
            "observed-state count rows require "
            "S1 COUNT."
        )

    s1, count = parts
    section[int(s1)] = int(count)


def parse_observed_state_action_count_row(
    section: dict[tuple[int, str], int],
    parts: list[str],
    line_number: int,
) -> None:
    if len(parts) != 3:
        raise ValueError(
            f"Line {line_number}: "
            "observed-state-action count rows require "
            "S1 ACTION COUNT."
        )

    s1, action, count = parts
    section[(int(s1), action)] = int(count)


def parse_full_state_action_count_row(
    section: dict[tuple[str, str], int],
    parts: list[str],
    line_number: int,
) -> None:
    if len(parts) != 3:
        raise ValueError(
            f"Line {line_number}: "
            "full-state-action count rows require "
            "STATE ACTION COUNT."
        )

    state, action, count = parts
    section[(state, action)] = int(count)


def parse_coverage_row(
    section: dict[str, bool],
    parts: list[str],
    line_number: int,
) -> None:
    if len(parts) != 2:
        raise ValueError(
            f"Line {line_number}: "
            "coverage rows require "
            "QUANTITY COVERED."
        )

    quantity, covered = parts
    section[quantity] = covered in {"1", "true", "True"}


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

    expected_sequences = set(SEQUENCES)

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
