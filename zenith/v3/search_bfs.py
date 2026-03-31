"""BFS search for Zenith v3 short-horizon stabilization planning."""

from collections import deque
from typing import Deque, List, Tuple

from .actions import get_actions
from .constraints import is_stable, is_step_valid
from .transition import transition

State = List[float]


def bfs_search(initial_state: State, max_depth: int) -> List[str]:
    """Find the shortest action sequence that reaches stability using BFS.

    Returns the first stable path found, or the deepest path explored
    if no stable state is reached within the depth limit.
    """
    if is_stable(initial_state):
        return []

    actions = get_actions()
    visited: set = set()
    visited.add(tuple(round(v, 6) for v in initial_state))

    queue: Deque[Tuple[State, List[str]]] = deque()
    queue.append((initial_state, []))

    best_path: List[str] = []

    while queue:
        state, path = queue.popleft()

        if len(path) >= max_depth:
            if not best_path:
                best_path = path
            continue

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
                return new_path

            queue.append((next_state, new_path))

    return best_path