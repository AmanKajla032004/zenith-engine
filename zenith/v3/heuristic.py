"""Admissible heuristic for Zenith v3 planning search."""

from typing import List

State = List[float]

_WEIGHTS: List[float] = [1.0, 1.0, 1.0, 1.3]
_STABILITY_BOUND = 1.5


def heuristic_cost(state: State) -> float:
    """Weighted distance from the stability region.

    Each dimension contributes zero if already within [-1.5, 1.5],
    otherwise contributes the weighted excess beyond the bound.
    Admissible: never overestimates the true cost to reach stability.
    """
    cost = 0.0
    for w, v in zip(_WEIGHTS, state):
        excess = abs(v) - _STABILITY_BOUND
        if excess > 0.0:
            cost += w * excess
    return cost