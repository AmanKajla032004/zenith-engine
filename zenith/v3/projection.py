"""Trajectory projection analysis for Zenith v3 planning."""

from typing import Dict, List, Optional, Union

from .constraints import compute_balance_score, is_stable

State = List[float]

Projection = Dict[str, Union[State, float, Optional[int]]]


def compute_projection(trajectory: List[State]) -> Projection:
    """Analyze a simulated trajectory and return projection metrics."""
    initial_state = trajectory[0]
    final_state = trajectory[-1]

    initial_balance = compute_balance_score(initial_state)
    final_balance = compute_balance_score(final_state)

    steps_to_stable: Optional[int] = None
    for i, state in enumerate(trajectory):
        if is_stable(state):
            steps_to_stable = i
            break

    return {
        "final_state": final_state,
        "final_balance_score": round(final_balance, 4),
        "steps_to_stable": steps_to_stable,
        "improvement": round(final_balance - initial_balance, 4),
    }