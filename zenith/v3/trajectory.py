"""Deterministic trajectory simulation for Zenith v3 planning."""

from typing import List

from .actions import get_actions
from .transition import transition

State = List[float]


def simulate_trajectory(initial_state: State, actions_list: List[str]) -> List[State]:
    """Simulate applying a sequence of actions and return the full state trajectory.

    The returned list begins with the initial state, followed by one state
    per action applied, for a total length of len(actions_list) + 1.
    """
    action_deltas = get_actions()
    trajectory: List[State] = [initial_state]
    current = initial_state

    for name in actions_list:
        delta = action_deltas.get(name)
        if delta is None:
            continue
        current = transition(current, delta)
        trajectory.append(current)

    return trajectory