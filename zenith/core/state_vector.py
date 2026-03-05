"""
zenith.core.state_vector

Converts normalized behavioral deviations into a compact behavioral
state vector. Each component captures a distinct axis of the user's
current behavioral state relative to their baseline.
"""


def build_state_vector(normalized: dict) -> dict:
    """
    Build a behavioral state vector from normalized deviations.

    Args:
        normalized: Deviation dict produced by normalize_entry().

    Returns:
        Dict with float components for performance, recovery,
        energy, and emotional state.
    """
    return {
        "performance": normalized["completion_dev"],
        "recovery": (normalized["recovery_dev"] + normalized["sleep_dev"]) / 2,
        "energy": normalized["deep_work_dev"],
        "emotional": normalized["mood_dev"] - normalized["stress_dev"],
    }