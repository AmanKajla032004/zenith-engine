from typing import List

State = List[float]

_CLAMP_MIN = -3.0
_CLAMP_MAX = 3.0


def apply_action(state: State, action_delta: State) -> State:
    """Return a new state by adding action deltas to the current state."""
    return [s + d for s, d in zip(state, action_delta)]


def clamp_state(state: State) -> State:
    """Clamp each state dimension to [-3.0, 3.0]."""
    return [max(_CLAMP_MIN, min(_CLAMP_MAX, v)) for v in state]


def transition(state: State, action_delta: State) -> State:
    """Apply an action delta to the state and clamp the result."""
    return clamp_state(apply_action(state, action_delta))