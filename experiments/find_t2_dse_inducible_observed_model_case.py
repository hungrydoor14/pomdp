from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import os
import sys

import numpy as np
from scipy.optimize import linprog

sys.path.append(os.path.dirname(__file__))

from two_period_joint_policy_experiments import (
    NUM_ACTIONS,
    NUM_S1,
    NUM_S2,
    build_action_dependent_factored_pomdp,
    join_state,
)


INITIAL_MATCH_PROB = 0.70
SEEDS = range(80)
ACTION_CONTROLS = [0.55, 0.70, 0.85, 0.95]
OBS_INFOS = [0.55, 0.70, 0.85, 0.95]
TOL = 1e-10
COVERAGE_EPS = 1e-4


@dataclass(frozen=True)
class FeasibilityResult:
    feasible: bool
    margin: float
    witness_b: np.ndarray | None


def b_index(s1, action):
    return s1 * NUM_ACTIONS + action


def prior_hidden_prob(s1):
    # q_s1 = P(S2=1 | S1=s1) under the initial information structure.
    return INITIAL_MATCH_PROB if s1 == 1 else 1.0 - INITIAL_MATCH_PROB


def observed_transition(pomdp, action):
    # For this lab, the next observed state distribution depends on the action.
    # It does not depend on current hidden state because S2' is action-driven.
    probs = np.zeros(NUM_S1)
    state = join_state(0, 0)
    for next_state in range(pomdp.num_states):
        next_s1, _ = divmod(next_state, NUM_S2)
        probs[next_s1] += pomdp.transitions[action, state, next_state]
    return probs


def rhat_coefficients(pomdp, s1, action):
    # b_{s1,a} = P(S2=1 | S1=s1, A=a)
    r0 = pomdp.rewards[join_state(s1, 0), action]
    r1 = pomdp.rewards[join_state(s1, 1), action]
    coeffs = np.zeros(NUM_S1 * NUM_ACTIONS)
    coeffs[b_index(s1, action)] = r1 - r0
    return float(r0), coeffs


def observed_value_coefficients(pomdp, first_action, second_action, initial_s1):
    constant, coeffs = rhat_coefficients(pomdp, initial_s1, first_action)
    transition = observed_transition(pomdp, first_action)
    for next_s1, prob in enumerate(transition):
        next_constant, next_coeffs = rhat_coefficients(pomdp, next_s1, second_action)
        constant += pomdp.discount * prob * next_constant
        coeffs += pomdp.discount * prob * next_coeffs
    return constant, coeffs


def dominance_constraints(pomdp, target):
    # A @ b >= c means the target sequence beats every alternative sequence
    # in the victim's observed two-period MDP, for both initial observed states.
    rows = []
    rhs = []
    for initial_s1 in range(NUM_S1):
        target_constant, target_coeffs = observed_value_coefficients(
            pomdp,
            target[0],
            target[1],
            initial_s1,
        )
        for candidate in product(range(NUM_ACTIONS), repeat=2):
            if candidate == target:
                continue
            candidate_constant, candidate_coeffs = observed_value_coefficients(
                pomdp,
                candidate[0],
                candidate[1],
                initial_s1,
            )
            rows.append(target_coeffs - candidate_coeffs)
            rhs.append(candidate_constant - target_constant)
    return np.array(rows), np.array(rhs)


def solve_linear_feasibility(A, c, extra_A=None, extra_c=None):
    # Maximize t subject to A b - c >= t, extra_A b >= extra_c, and b in [0,1]^4.
    dim = NUM_S1 * NUM_ACTIONS
    objective = np.zeros(dim + 1)
    objective[-1] = -1.0

    lhs = []
    rhs = []
    for row, threshold in zip(A, c):
        lp_row = np.zeros(dim + 1)
        lp_row[:dim] = -row
        lp_row[-1] = 1.0
        lhs.append(lp_row)
        rhs.append(-threshold)

    if extra_A is not None:
        for row, threshold in zip(extra_A, extra_c):
            lp_row = np.zeros(dim + 1)
            lp_row[:dim] = -row
            lhs.append(lp_row)
            rhs.append(-threshold)

    result = linprog(
        c=objective,
        A_ub=np.array(lhs),
        b_ub=np.array(rhs),
        bounds=[(0.0, 1.0)] * dim + [(None, None)],
        method="highs",
    )
    if not result.success:
        return FeasibilityResult(False, float("-inf"), None)

    return FeasibilityResult(result.x[-1] >= -TOL, float(result.x[-1]), result.x[:dim])


def bayes_orientation_constraints(orientations):
    # For each observed s1, require one of:
    #   b(s1,a0) < q_s1 < b(s1,a1), or
    #   b(s1,a1) < q_s1 < b(s1,a0).
    # The small epsilon avoids zero-coverage witnesses that cannot support
    # finite-sample reward estimation for both actions.
    rows = []
    rhs = []
    for s1, orientation in enumerate(orientations):
        q = prior_hidden_prob(s1)
        lower_action, upper_action = (0, 1) if orientation == 0 else (1, 0)

        lower_row = np.zeros(NUM_S1 * NUM_ACTIONS)
        lower_row[b_index(s1, lower_action)] = -1.0
        rows.append(lower_row)
        rhs.append(-(q - COVERAGE_EPS))

        upper_row = np.zeros(NUM_S1 * NUM_ACTIONS)
        upper_row[b_index(s1, upper_action)] = 1.0
        rows.append(upper_row)
        rhs.append(q + COVERAGE_EPS)
    return np.array(rows), np.array(rhs)


def inducible_feasibility(A, c):
    # Bayes plausibility is a union of four linear regions, one orientation for
    # each observed s1. If any region is feasible, the b-vector is inducible.
    best = FeasibilityResult(False, float("-inf"), None)
    for orientations in product((0, 1), repeat=NUM_S1):
        extra_A, extra_c = bayes_orientation_constraints(orientations)
        result = solve_linear_feasibility(A, c, extra_A, extra_c)
        if result.feasible:
            return result
        if result.margin > best.margin:
            best = result
    return best


def construct_attacker_from_b(b):
    # Return action_probs[s1,s2,a]. This realizes the requested posteriors
    # b_{s1,a}=P(S2=1|S1=s1,A=a) whenever Bayes plausibility holds.
    action_probs = np.zeros((NUM_S1, NUM_S2, NUM_ACTIONS))
    for s1 in range(NUM_S1):
        q = prior_hidden_prob(s1)
        b0 = b[b_index(s1, 0)]
        b1 = b[b_index(s1, 1)]

        if abs(b0 - b1) <= TOL:
            p_action1 = np.array([0.5, 0.5])
        else:
            low = min(b0, b1)
            high = max(b0, b1)
            if q < low - 1e-8 or q > high + 1e-8:
                return None

            # lambda is marginal P(A=1 | S1=s1).
            lambda_action1 = (q - b0) / (b1 - b0)
            lambda_action1 = float(np.clip(lambda_action1, 0.0, 1.0))
            p_a1_given_s2_1 = lambda_action1 * b1 / q if q > TOL else 0.0
            p_a1_given_s2_0 = (
                lambda_action1 * (1.0 - b1) / (1.0 - q)
                if 1.0 - q > TOL
                else 0.0
            )
            p_action1 = np.array([p_a1_given_s2_0, p_a1_given_s2_1])
            p_action1 = np.clip(p_action1, 0.0, 1.0)

        for s2 in range(NUM_S2):
            action_probs[s1, s2, 1] = p_action1[s2]
            action_probs[s1, s2, 0] = 1.0 - p_action1[s2]
    return action_probs


def induced_b_from_attacker(action_probs):
    induced = np.zeros(NUM_S1 * NUM_ACTIONS)
    for s1 in range(NUM_S1):
        q = prior_hidden_prob(s1)
        hidden_probs = np.array([1.0 - q, q])
        for action in range(NUM_ACTIONS):
            action_mass = sum(
                hidden_probs[s2] * action_probs[s1, s2, action]
                for s2 in range(NUM_S2)
            )
            if action_mass <= TOL:
                induced[b_index(s1, action)] = np.nan
            else:
                induced[b_index(s1, action)] = (
                    hidden_probs[1] * action_probs[s1, 1, action] / action_mass
                )
    return induced


def prior_b_vector():
    b = np.zeros(NUM_S1 * NUM_ACTIONS)
    for s1 in range(NUM_S1):
        for action in range(NUM_ACTIONS):
            b[b_index(s1, action)] = prior_hidden_prob(s1)
    return b


def observed_value(pomdp, sequence, initial_s1, b):
    constant, coeffs = observed_value_coefficients(
        pomdp,
        sequence[0],
        sequence[1],
        initial_s1,
    )
    return float(constant + coeffs @ b)


def full_state_value(pomdp, sequence, state):
    first_action, second_action = sequence
    future = 0.0
    for next_state in range(pomdp.num_states):
        future += (
            pomdp.transitions[first_action, state, next_state]
            * pomdp.rewards[next_state, second_action]
        )
    return float(pomdp.rewards[state, first_action] + pomdp.discount * future)


def best_sequences(sequence_values):
    best_value = max(value for _, value in sequence_values)
    winners = [
        sequence
        for sequence, value in sequence_values
        if abs(value - best_value) <= 1e-9
    ]
    return best_value, winners


def format_sequence(sequence):
    return f"(a{sequence[0]},a{sequence[1]})"


def format_sequence_set(sequences):
    return "{" + ", ".join(format_sequence(sequence) for sequence in sequences) + "}"


def format_action_set(actions):
    return "{" + ", ".join(f"a{action}" for action in actions) + "}"


def positive_coverage_b(b, eps=1e-3):
    return np.clip(b, eps, 1.0 - eps)


def format_b(b):
    if b is None:
        return "None"
    return "  ".join(
        f"b({s1},a{action})={b[b_index(s1, action)]:.3f}"
        for s1 in range(NUM_S1)
        for action in range(NUM_ACTIONS)
    )


def print_sequence_values(label, sequence_values):
    best_value, winners = best_sequences(sequence_values)
    print(label)
    for sequence, value in sorted(sequence_values):
        marker = " *" if sequence in winners else "  "
        print(f"  {marker} {format_sequence(sequence)} value={value:+.6f}")
    print(f"  learned T*: {format_sequence_set(winners)}  value={best_value:+.6f}")


def observed_tstar_by_initial_state(pomdp, b):
    winners_by_s1 = []
    for initial_s1 in range(NUM_S1):
        values = [
            (sequence, observed_value(pomdp, sequence, initial_s1, b))
            for sequence in product(range(NUM_ACTIONS), repeat=2)
        ]
        winners_by_s1.append(set(best_sequences(values)[1]))
    return winners_by_s1


def print_t2_dse_case_study(pomdp, row):
    target = row["target"]
    result = row["inducible"]
    print("=== T2-DSE case study: what the learner would learn ===")
    print(
        f"Selected instance: seed={row['seed']} "
        f"discount={pomdp.discount:.2f} "
        f"control={row['action_control']:.2f} obs={row['obs_info']:.2f} "
        f"target={format_sequence(target)}"
    )
    print("Selection rule: strongest inducible case where attack changes an observed-only T*.")
    print(f"inducible feasible: {result.feasible}")
    print(f"inducible margin: {result.margin:+.4f}")
    print()

    print("Original full-state rewards:")
    for s1 in range(NUM_S1):
        for s2 in range(NUM_S2):
            state = join_state(s1, s2)
            print(
                f"  R(s1={s1},s2={s2},a0)={pomdp.rewards[state, 0]:+.3f}  "
                f"R(s1={s1},s2={s2},a1)={pomdp.rewards[state, 1]:+.3f}"
            )
    print()

    print("1) If the learner had everything, including S2:")
    full_first_actions = {}
    for state in range(pomdp.num_states):
        s1, s2 = divmod(state, NUM_S2)
        values = [
            (sequence, full_state_value(pomdp, sequence, state))
            for sequence in product(range(NUM_ACTIONS), repeat=2)
        ]
        best_value, winners = best_sequences(values)
        full_first_actions[(s1, s2)] = sorted({sequence[0] for sequence in winners})
        print(
            f"  state (s1={s1},s2={s2}) -> T*={format_sequence_set(winners)} "
            f"first_actions={format_action_set(full_first_actions[(s1, s2)])} "
            f"value={best_value:+.6f}"
        )
    print()

    prior_b = prior_b_vector()
    print("2) With only S1 and no attack-induced action mixture:")
    print(f"  prior hidden mixtures: {format_b(prior_b)}")
    for initial_s1 in range(NUM_S1):
        values = [
            (sequence, observed_value(pomdp, sequence, initial_s1, prior_b))
            for sequence in product(range(NUM_ACTIONS), repeat=2)
        ]
        print_sequence_values(f"  observed initial s1={initial_s1}", values)
    print()

    print("3) Given pi dagger is the actual data-generating policy:")
    print(f"  LP witness mixtures: {format_b(result.witness_b)}")
    actual_b = positive_coverage_b(result.witness_b)
    print(f"  positive-coverage mixtures used by pi_dagger: {format_b(actual_b)}")
    attacker = construct_attacker_from_b(actual_b)
    if attacker is None:
        print("  no attacker could be constructed from this witness")
        return
    for s1 in range(NUM_S1):
        for s2 in range(NUM_S2):
            print(
                f"  pi_dagger(a1 | s1={s1},s2={s2})="
                f"{attacker[s1, s2, 1]:.6f}"
            )
    induced = induced_b_from_attacker(attacker)
    print(f"  induced mixtures: {format_b(induced)}")
    print()

    for initial_s1 in range(NUM_S1):
        values = [
            (sequence, observed_value(pomdp, sequence, initial_s1, actual_b))
            for sequence in product(range(NUM_ACTIONS), repeat=2)
        ]
        print_sequence_values(f"  attacked observed initial s1={initial_s1}", values)
    print()

    print("4) Static observed transition graph/table:")
    for current_s1 in range(NUM_S1):
        for action in range(NUM_ACTIONS):
            transition = observed_transition(pomdp, action)
            print(
                f"  s1={current_s1} --a{action}--> "
                f"s1'=0:{transition[0]:.3f}, s1'=1:{transition[1]:.3f}"
            )
    print()
    print(
        "Case-study conclusion: under the attacker-induced observed model, "
        f"the target sequence {format_sequence(target)} is in T* for both "
        f"observed initial states, so its first action a{target[0]} is rationalized."
    )
    print()


def main():
    rows = []
    for seed in SEEDS:
        for action_control in ACTION_CONTROLS:
            for obs_info in OBS_INFOS:
                pomdp = build_action_dependent_factored_pomdp(
                    seed,
                    action_control=action_control,
                    p_s1_matches_s2=obs_info,
                )
                for target in product(range(NUM_ACTIONS), repeat=2):
                    A, c = dominance_constraints(pomdp, target)
                    relaxed = solve_linear_feasibility(A, c)
                    inducible = inducible_feasibility(A, c)
                    rows.append(
                        {
                            "seed": seed,
                            "action_control": action_control,
                            "obs_info": obs_info,
                            "target": target,
                            "relaxed": relaxed,
                            "inducible": inducible,
                        }
                    )

    print("=== Attempt 3: inducible observed-model characterization ===")
    print(f"instance-target pairs: {len(rows)}")
    print()
    print("=== Feasibility counts ===")
    print(f"relaxed b in [0,1]^4: {sum(r['relaxed'].feasible for r in rows)}")
    print(f"Bayes-plausible inducible b: {sum(r['inducible'].feasible for r in rows)}")
    print()

    pinned = [
        r for r in rows
        if r["seed"] == 54
        and r["action_control"] == 0.55
        and r["obs_info"] == 0.95
        and r["target"] == (1, 1)
    ][0]
    case_study_candidates = []
    prior_b = prior_b_vector()
    for row in rows:
        if not row["inducible"].feasible:
            continue
        pomdp = build_action_dependent_factored_pomdp(
            row["seed"],
            action_control=row["action_control"],
            p_s1_matches_s2=row["obs_info"],
        )
        prior_tstar = observed_tstar_by_initial_state(pomdp, prior_b)
        attack_tstar = observed_tstar_by_initial_state(pomdp, row["inducible"].witness_b)
        target = row["target"]
        target_changed_baseline = any(target not in winners for winners in prior_tstar)
        target_optimal_under_attack = all(target in winners for winners in attack_tstar)
        if target_changed_baseline and target_optimal_under_attack:
            case_study_candidates.append(row)

    case_study = max(
        case_study_candidates,
        key=lambda row: row["inducible"].margin,
        default=pinned,
    )
    case_study_pomdp = build_action_dependent_factored_pomdp(
        case_study["seed"],
        action_control=case_study["action_control"],
        p_s1_matches_s2=case_study["obs_info"],
    )
    print_t2_dse_case_study(case_study_pomdp, case_study)
    print()

    examples = [r for r in rows if r["inducible"].feasible]
    examples.sort(key=lambda r: r["inducible"].margin, reverse=True)
    print("=== Strongest inducible examples ===")
    for index, row in enumerate(examples[:12], start=1):
        print(
            f"{index:>2}. seed={row['seed']:>2} "
            f"control={row['action_control']:.2f} obs={row['obs_info']:.2f} "
            f"target=(a{row['target'][0]},a{row['target'][1]}) "
            f"margin={row['inducible'].margin:+.4f}"
        )
        print(f"    witness {format_b(row['inducible'].witness_b)}")


if __name__ == "__main__":
    main()
