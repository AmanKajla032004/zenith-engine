"""
zenith.reasoning.reasoning_engine

Orchestrates all v2 reasoning algorithms and combines their outputs
into a single unified reasoning result. Acts as the entry point for
the reasoning layer.
"""

from zenith.reasoning.hill_climbing import detect_dominant_driver
from zenith.reasoning.overcommitment import detect_overcommitment
from zenith.reasoning.bayesian_model import estimate_probabilities


def run_reasoning(state: dict, entry: dict) -> dict:
    """
    Run all reasoning algorithms and return combined results.

    Args:
        state: State vector dict produced by build_state_vector().
        entry: A single entry dict matching the Zenith entry schema.

    Returns:
        Dict combining driver detection, overcommitment detection,
        and probability estimation outputs.
    """
    driver = detect_dominant_driver(state)
    pressure = detect_overcommitment(entry)
    probs = estimate_probabilities(entry)

    return {
        "dominant_driver": driver["dominant_driver"],
        "driver_improvement": driver["improvement"],
        "behavioral_pressure_index": pressure["behavioral_pressure_index"],
        "capacity_mismatch": pressure["capacity_mismatch"],
        "burnout_probability": probs["burnout_probability"],
        "fatigue_probability": probs["fatigue_probability"],
        "recovery_failure_probability": probs["recovery_failure_probability"],
    }