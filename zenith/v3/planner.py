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

    trajectory = simulate_trajectory(state, actions)
    projection = compute_projection(trajectory)

    return {
        "policy": policy,
        "depth": depth,
        "actions": actions,
        "trajectory": trajectory,
        "projection": projection,
    }