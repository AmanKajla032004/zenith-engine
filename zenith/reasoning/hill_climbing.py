"""
zenith.reasoning.hill_climbing

Identifies the dominant behavioral driver affecting system balance
using a single-step hill climbing approach. Each state component is
tested with a small adjustment to determine which one yields the
greatest improvement to overall balance.
"""


def detect_dominant_driver(state: dict) -> dict:
    """
    Detect the state component whose improvement most benefits balance.

    Applies a +1 adjustment to each component independently and
    selects the one producing the largest balance improvement.

    Args:
        state: State vector dict produced by build_state_vector().

    Returns:
        Dict with dominant_driver (str) and improvement (float).
    """
    balance = sum(state.values())

    best_driver = None
    best_improvement = 0.0

    for component in state:
        adjusted = dict(state)
        adjusted[component] += 1
        new_balance = sum(adjusted.values())
        improvement = new_balance - balance

        if best_driver is None or improvement > best_improvement:
            best_driver = component
            best_improvement = improvement

    return {
        "dominant_driver": best_driver,
        "improvement": best_improvement,
    }