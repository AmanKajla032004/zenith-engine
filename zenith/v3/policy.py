"""Deterministic policy selection for Zenith v3 planning."""

from typing import List

State = List[float]

_RECOVERY_INDEX = 1


def select_policy(state: State) -> str:
    """Select the search algorithm based on current state severity.

    Priority order:
    1. astar when any dimension exceeds 2.0 in magnitude
    2. dfs when recovery is strictly the largest instability
    3. bfs otherwise
    """
    abs_values = [abs(v) for v in state]
    max_abs = max(abs_values)

    if max_abs > 2.0:
        return "astar"

    recovery_abs = abs_values[_RECOVERY_INDEX]
    if all(recovery_abs > abs_values[i] for i in range(len(abs_values)) if i != _RECOVERY_INDEX):
        return "dfs"

    return "bfs"


def select_depth(state: State) -> int:
    """Determine planning search depth based on state severity."""
    max_abs = max(abs(v) for v in state)

    if max_abs > 2.5:
        return 6
    if max_abs > 2.0:
        return 5
    if max_abs > 1.5:
        return 4
    return 3