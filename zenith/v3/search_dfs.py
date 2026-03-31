"""Depth-limited DFS search for Zenith v3 recovery-focused planning."""

from typing import List, Optional, Set, Tuple

from .actions import get_actions
from .constraints import is_stable, is_step_valid
from .transition import transition

State = List[float]


def dfs_search(initial_state: State, max_depth: int) -> List[str]:
    """Find an action sequence that reaches stability using depth-limited DFS.

    Returns the shortest stable path found, or an empty list if none exists.
    """
    if is_stable(initial_state):
        return []

    actions = get_actions()
    visited: Set[Tuple[float, ...]] = set()
    visited.add(tuple(round(v, 6) for v in initial_state))

    best: List[Optional[List[str]]] = [None]

    def _recurse(state: State, path: List[str], depth: int) -> None:
        if depth >= max_depth:
            return

        for name, delta in actions.items():
            next_state = transition(state, delta)

            if not is_step_valid(state, next_state):
                continue

            state_key = tuple(round(v, 6) for v in next_state)
            if state_key in visited:
                continue
            visited.add(state_key)

            new_path = path + [name]

            if is_stable(next_state):
                if best[0] is None or len(new_path) < len(best[0]):
                    best[0] = new_path
                visited.discard(state_key)
                continue

            if best[0] is None or len(new_path) < len(best[0]):
                _recurse(next_state, new_path, depth + 1)

            visited.discard(state_key)

    _recurse(initial_state, [], 0)

    return best[0] if best[0] is not None else []