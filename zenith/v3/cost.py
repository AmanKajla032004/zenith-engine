"""Deterministic planning cost functions for Zenith v3."""

from typing import List

from .constraints import detect_overshoot, is_stable

State = List[float]

_WEIGHTS: List[float] = [1.0, 1.0, 1.0, 1.3]
_OVERSHOOT_PENALTY = 0.5
_STABILITY_REWARD = -1.0


def state_cost(state: State) -> float:
    """Weighted sum of absolute state deviations."""
    return sum(w * abs(v) for w, v in zip(_WEIGHTS, state))


def overshoot_penalty(overshoot_flags: List[bool]) -> float:
    """Fixed penalty for each dimension that exhibits a sign flip."""
    return sum(_OVERSHOOT_PENALTY for f in overshoot_flags if f)


def step_cost(prev_state: State, next_state: State) -> float:
    """Total cost of a single planning step including state cost and overshoot penalty."""
    cost = state_cost(next_state)
    flags = detect_overshoot(prev_state, next_state)
    cost += overshoot_penalty(flags)
    return cost


def goal_reward(state: State) -> float:
    """Return a negative cost bonus when the state is within the stability bound."""
    if is_stable(state):
        return _STABILITY_REWARD
    return 0.0