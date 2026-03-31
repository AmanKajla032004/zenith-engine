"""A* search for Zenith v3 stabilization planning."""

import heapq
from typing import List, Tuple

from .actions import get_actions
from .constraints import is_stable, is_step_valid
from .cost import goal_reward, step_cost
from .heuristic import heuristic_cost
from .transition import transition

State = List[float]


def astar_search(initial_state: State, max_depth: int) -> List[str]:
    """Find an action sequence that moves the state toward stability using A* search.

    Returns the best action sequence found within the depth limit.
    """
    actions = get_actions()

    # (f_cost, tie_breaker, g_cost, state, path)
    counter = 0
    h = heuristic_cost(initial_state)
    start: Tuple[float, int, float, State, List[str]] = (h, counter, 0.0, initial_state, [])
    open_list: List[Tuple[float, int, float, State, List[str]]] = [start]

    best_path: List[str] = []
    best_cost = float("inf")

    visited: set = set()

    while open_list:
        f, _, g, state, path = heapq.heappop(open_list)

        state_key = tuple(round(v, 6) for v in state)
        if state_key in visited:
            continue
        visited.add(state_key)

        reward = goal_reward(state)
        total = g + reward

        if is_stable(state) and total < best_cost:
            best_cost = total
            best_path = path

        if is_stable(state):
            continue

        if len(path) >= max_depth:
            if g < best_cost and not best_path:
                best_cost = g
                best_path = path
            continue

        for name, delta in actions.items():
            next_state = transition(state, delta)

            if not is_step_valid(state, next_state):
                continue

            sc = step_cost(state, next_state)
            new_g = g + sc
            h = heuristic_cost(next_state)
            new_f = new_g + h

            counter += 1
            heapq.heappush(open_list, (new_f, counter, new_g, next_state, path + [name]))

    return best_path