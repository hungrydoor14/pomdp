from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import numpy as np
from scipy.optimize import linprog


Array = np.ndarray


@dataclass(frozen=True)
class AlphaVector:
    # The action to take at the root of this policy tree.
    action: int

    # The alpha-vector values, one entry per latent world state.
    # For a belief b, the value of this contingent plan is b . values.
    values: Array

    # One subtree per possible observation. If this vector is used at horizon t,
    # these are the policy trees to follow at horizon t - 1 after each observation.
    observation_subplans: tuple["AlphaVector", ...] = ()

    def value_at(self, belief: Array) -> float:
        # Evaluate this alpha-vector at a particular belief state.
        return float(np.dot(belief, self.values))


@dataclass(frozen=True)
class TabularPOMDP:
    # transitions[a, s, s'] = P(s' | s, a)
    transitions: Array

    # observations[a, s', o] = P(o | s', a)
    observations: Array

    # rewards[s, a] = expected immediate reward for taking action a in state s
    rewards: Array
    discount: float
    state_names: tuple[str, ...] | None = None
    action_names: tuple[str, ...] | None = None
    observation_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        # Convert everything to dense float arrays once so the rest of the code can
        # assume simple numeric tensors.
        t = np.asarray(self.transitions, dtype=float)
        z = np.asarray(self.observations, dtype=float)
        r = np.asarray(self.rewards, dtype=float)

        # The implementation here follows the standard tabular finite POMDP setup.
        if t.ndim != 3:
            raise ValueError("transitions must have shape [num_actions, num_states, num_states]")
        if z.ndim != 3:
            raise ValueError(
                "observations must have shape [num_actions, num_states, num_observations]"
            )
        if r.ndim != 2:
            raise ValueError("rewards must have shape [num_states, num_actions]")
        if not (0.0 < self.discount <= 1.0):
            raise ValueError("discount must be in (0, 1]")

        num_actions, num_states, next_states = t.shape
        obs_actions, obs_states, _ = z.shape

        if num_states != next_states:
            raise ValueError("transition state dimensions must match")
        if obs_actions != num_actions or obs_states != num_states:
            raise ValueError("observation tensor must agree with transition dimensions")
        if r.shape != (num_states, num_actions):
            raise ValueError("rewards must have shape [num_states, num_actions]")

        # Every action-state row must define a valid probability distribution.
        if not np.allclose(t.sum(axis=2), 1.0):
            raise ValueError("each transition row must sum to 1")
        if not np.allclose(z.sum(axis=2), 1.0):
            raise ValueError("each observation row must sum to 1")

        object.__setattr__(self, "transitions", t)
        object.__setattr__(self, "observations", z)
        object.__setattr__(self, "rewards", r)

    @property
    def num_actions(self) -> int:
        return self.transitions.shape[0]

    @property
    def num_states(self) -> int:
        return self.transitions.shape[1]

    @property
    def num_observations(self) -> int:
        return self.observations.shape[2]

    def belief_update(self, belief: Array, action: int, observation: int) -> Array:
        # Step 1: predict the next-state distribution after taking action a.
        belief = np.asarray(belief, dtype=float)
        predicted = belief @ self.transitions[action]

        # Step 2: reweight by how compatible each next state is with the observation.
        weighted = predicted * self.observations[action, :, observation]
        normalizer = weighted.sum()

        if normalizer <= 0:
            raise ValueError("observation has zero probability under the given belief and action")

        # Step 3: normalize to get the posterior belief b'.
        return weighted / normalizer


def solve_finite_horizon(pomdp: TabularPOMDP, horizon: int) -> list[list[AlphaVector]]:
    # Stage t stores the parsimonious alpha-vectors for a t-step problem.
    if horizon < 0:
        raise ValueError("horizon must be non-negative")

    # Horizon 0 has zero future reward and no real action.
    terminal = AlphaVector(action=-1, values=np.zeros(pomdp.num_states))
    stages: list[list[AlphaVector]] = [[terminal]]

    # Build value functions backward, one dynamic-programming backup at a time.
    for _ in range(horizon):
        stages.append(_backup_stage(pomdp, stages[-1]))

    return stages


def best_alpha(vectors: Iterable[AlphaVector], belief: Array) -> AlphaVector:
    # Among all candidate policy trees, choose the one with highest value at b.
    return max(vectors, key=lambda alpha: alpha.value_at(belief))


def greedy_action(vectors: Iterable[AlphaVector], belief: Array) -> int:
    # Convenience wrapper when only the root action matters.
    return best_alpha(vectors, belief).action


def pretty_plan(alpha: AlphaVector, pomdp: TabularPOMDP, indent: int = 0) -> str:
    # Render the policy tree carried by an alpha-vector in a readable nested format.
    pad = " " * indent
    if alpha.action < 0:
        return f"{pad}terminal"

    action_name = _action_name(pomdp, alpha.action)
    lines = [f"{pad}{action_name}"]
    for observation, subtree in enumerate(alpha.observation_subplans):
        observation_name = _observation_name(pomdp, observation)
        lines.append(f"{pad}  if {observation_name}:")
        lines.append(pretty_plan(subtree, pomdp, indent + 4))
    return "\n".join(lines)


def _backup_stage(pomdp: TabularPOMDP, previous_stage: list[AlphaVector]) -> list[AlphaVector]:
    # This is the exact finite-horizon backup:
    # for each root action and each choice of one subtree per observation, construct
    # the resulting alpha-vector, then prune to the parsimonious set.
    candidates: list[AlphaVector] = []

    for action in range(pomdp.num_actions):
        # A t-step policy tree chooses a root action and, for each observation, one
        # (t - 1)-step subtree from the previous stage.
        for observation_plan in product(previous_stage, repeat=pomdp.num_observations):
            # Start with immediate reward R(s, a).
            alpha = pomdp.rewards[:, action].copy()
            continuation = np.zeros(pomdp.num_states)

            for state in range(pomdp.num_states):
                total = 0.0
                for next_state in range(pomdp.num_states):
                    transition_prob = pomdp.transitions[action, state, next_state]
                    obs_return = 0.0
                    for observation, next_alpha in enumerate(observation_plan):
                        # If the next hidden state is s', the chosen subtree for each
                        # observation contributes according to P(o | s', a).
                        obs_prob = pomdp.observations[action, next_state, observation]
                        obs_return += obs_prob * next_alpha.values[next_state]
                    # Sum over next states using P(s' | s, a).
                    total += transition_prob * obs_return
                continuation[state] = total

            # Final backup equation:
            # alpha_a,plan(s) = R(s, a) + gamma * sum_s' T(s, a, s') sum_o O(s', a, o) alpha_o(s')
            alpha += pomdp.discount * continuation
            candidates.append(
                AlphaVector(
                    action=action,
                    values=alpha,
                    observation_subplans=tuple(observation_plan),
                )
            )

    return _prune_to_parsimonious_set(candidates)


def _prune_to_parsimonious_set(vectors: list[AlphaVector], tol: float = 1e-10) -> list[AlphaVector]:
    # The paper keeps only useful vectors: ones that are optimal for at least one belief.
    # We first collapse exact duplicates, then search for a witness belief for each vector.
    unique = _prune_duplicate_vectors(vectors, tol=tol)
    useful: list[AlphaVector] = []

    for candidate in unique:
        witness = _find_witness_belief(candidate, unique, tol=tol)
        if witness is not None:
            useful.append(candidate)

    return useful


def _prune_duplicate_vectors(vectors: list[AlphaVector], tol: float = 1e-10) -> list[AlphaVector]:
    # Duplicate value vectors define the same linear function over belief space, so
    # only one copy needs to survive.
    unique: list[AlphaVector] = []

    for candidate in vectors:
        is_duplicate = any(
            np.allclose(candidate.values, existing.values, atol=tol, rtol=0.0)
            for existing in unique
        )
        if not is_duplicate:
            unique.append(candidate)

    return unique


def _find_witness_belief(
    candidate: AlphaVector, vectors: list[AlphaVector], tol: float = 1e-10
) -> Array | None:
    # A vector is useful iff there exists some belief b where it strictly beats all others.
    # This is the textbook upper-envelope idea phrased as a linear program.
    others = [other for other in vectors if other is not candidate]
    if not others:
        # If candidate is the only vector, every belief chooses it.
        return np.full(candidate.values.shape[0], 1.0 / candidate.values.shape[0])

    num_states = candidate.values.shape[0]

    # Variables are [b_0, ..., b_{|S|-1}, delta].
    # We maximize delta subject to:
    #   b . candidate >= b . other + delta     for all other vectors
    #   sum_i b_i = 1
    #   b_i >= 0
    # If the best delta is positive, candidate is strictly optimal somewhere.
    c = np.zeros(num_states + 1)
    c[-1] = -1.0

    a_ub = []
    b_ub = []
    for other in others:
        row = np.zeros(num_states + 1)
        row[:num_states] = other.values - candidate.values
        row[-1] = 1.0
        a_ub.append(row)
        b_ub.append(0.0)

    a_eq = [np.append(np.ones(num_states), 0.0)]
    b_eq = [1.0]
    bounds = [(0.0, 1.0) for _ in range(num_states)] + [(None, None)]

    result = linprog(
        c=c,
        A_ub=np.asarray(a_ub, dtype=float),
        b_ub=np.asarray(b_ub, dtype=float),
        A_eq=np.asarray(a_eq, dtype=float),
        b_eq=np.asarray(b_eq, dtype=float),
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        raise RuntimeError(f"belief-space pruning LP failed: {result.message}")

    if result.x[-1] > tol:
        # Return the witness belief so callers can inspect or debug if they want.
        return result.x[:num_states]

    return None


def _action_name(pomdp: TabularPOMDP, action: int) -> str:
    # Friendly names help the printed policy tree read like a plan instead of an index dump.
    if pomdp.action_names is None:
        return f"action_{action}"
    return pomdp.action_names[action]


def _observation_name(pomdp: TabularPOMDP, observation: int) -> str:
    if pomdp.observation_names is None:
        return f"obs_{observation}"
    return pomdp.observation_names[observation]
