"""Deterministic behavioral actions and state-vector effect definitions for Zenith v3."""

from typing import Dict, List

Action = List[float]

# State vector order: [performance, recovery, energy, emotional]
ACTIONS: Dict[str, Action] = {
    # Performance
    "reduce_workload":      [-0.3,  0.1,  0.1,  0.2],
    "increase_focus_block": [ 0.4, -0.1, -0.1,  0.0],
    # Recovery
    "rest_block":           [-0.1,  0.4,  0.2,  0.1],
    "sleep_extension":      [ 0.0,  0.5,  0.3,  0.1],
    # Energy
    "light_activity":       [ 0.1,  0.1,  0.4,  0.2],
    "reduce_activity":      [-0.1,  0.3,  0.2,  0.0],
    # Emotional
    "social_support":       [ 0.0,  0.1,  0.1,  0.5],
    "mindfulness":          [ 0.1,  0.2,  0.1,  0.4],
    "reduce_stressors":     [ 0.1,  0.1,  0.1,  0.5],
    # Neutral
    "maintain":             [ 0.0,  0.0,  0.0,  0.0],
}

DRIVER_ACTION_MAP: Dict[str, List[str]] = {
    "performance": ["reduce_workload", "increase_focus_block"],
    "recovery":    ["rest_block", "sleep_extension"],
    "energy":      ["light_activity", "reduce_activity"],
    "emotional":   ["social_support", "mindfulness", "reduce_stressors"],
}


def get_actions() -> Dict[str, Action]:
    """Return the complete action-to-delta mapping."""
    return ACTIONS


def get_driver_actions(driver: str) -> List[str]:
    """Return the list of action names relevant to a dominant driver.
    Falls back to all action names when the driver is not found."""
    return DRIVER_ACTION_MAP.get(driver, list(ACTIONS.keys()))