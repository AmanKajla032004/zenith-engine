"""Top-level planner orchestrating Zenith v3 behavioral planning."""

from typing import Any, Dict, List

from .policy import select_depth, select_policy
from .projection import compute_projection
from .search_astar import astar_search
from .search_bfs import bfs_search
from .search_dfs import dfs_search
from .trajectory import simulate_trajectory

State = List[float]

_SEARCH = {
    "astar": astar_search,
    "bfs": bfs_search,
    "dfs": dfs_search,
}


def plan_behavior(state: State) -> Dict[str, Any]:
    """Run the full Zenith v3 planning pipeline for a given state vector."""
    policy = select_policy(state)
    depth = select_depth(state)

    search_fn = _SEARCH[policy]
    actions = search_fn(state, depth)
    actions = [action.lstrip("0123456789").strip() for action in actions]

    # fallback if planner returns no actions but system unstable
    if not actions:
        imbalance = sum(abs(x) for x in state)
        if imbalance > 0.5:
            actions = ["reduce_workload", "sleep_extension", "mindfulness"]

    raw_trajectory = simulate_trajectory(state, actions)

    def balance(s: State) -> float:
        return 10 - (abs(s[0]) + abs(s[1]) + abs(s[2]) + abs(s[3]))

    # find best balance step and truncate
    balances = [balance(s) for s in raw_trajectory]
    best_step = balances.index(max(balances))
    actions = actions[:best_step]
    raw_trajectory = raw_trajectory[: best_step + 1]

    trajectory = [
        {"step": i, "balance": balance(s)}
        for i, s in enumerate(raw_trajectory)
    ]
    projection = compute_projection(raw_trajectory)

    return {
        "policy": policy,
        "depth": depth,
        "actions": actions,
        "trajectory": trajectory,
        "projection": projection,
    }