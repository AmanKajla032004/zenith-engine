"""Deterministic constraint functions for Zenith v3 planning."""

from typing import List

State = List[float]

_MAX_STEP_MAGNITUDE = 1.0
_STABILITY_BOUND = 1.5


def is_step_valid(prev_state: State, next_state: State) -> bool:
    """Return True if no single dimension changes by more than the max step magnitude."""
    return all(
        abs(n - p) <= _MAX_STEP_MAGNITUDE
        for p, n in zip(prev_state, next_state)
    )


def detect_overshoot(prev_state: State, next_state: State) -> List[bool]:
    """Return a per-dimension list indicating sign flips across zero (oscillation risk)."""
    return [
        (p * n < 0.0) and (abs(p) > 0.0)
        for p, n in zip(prev_state, next_state)
    ]


def is_stable(state: State) -> bool:
    """Return True if all state dimensions are within the stability bound [-1.5, 1.5]."""
    return all(abs(v) <= _STABILITY_BOUND for v in state)


def compute_balance_score(state: State) -> float:
    """Compute balance score as 10 minus the sum of absolute deviations."""
    return 10.0 - sum(abs(v) for v in state)