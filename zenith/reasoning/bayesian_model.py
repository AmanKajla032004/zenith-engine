"""
zenith.reasoning.bayesian_model

Estimates burnout, fatigue, and recovery failure probabilities
using a simple weighted probabilistic model. Raw behavioral signals
are normalized to a 0–1 range and combined with fixed priors to
produce calibrated risk estimates.
"""


def _clamp(value: float) -> float:
    """Clamp a value to the 0–1 range."""
    return max(0.0, min(1.0, value))


def estimate_probabilities(entry: dict) -> dict:
    """
    Estimate behavioral risk probabilities from a single entry.

    Args:
        entry: A single entry dict matching the Zenith entry schema.

    Returns:
        Dict with burnout_probability, fatigue_probability, and
        recovery_failure_probability, each clamped to 0–1.
    """
    sleep_factor = max(0.0, (7 - entry["sleep_hours"]) / 7)
    stress_factor = entry["stress"] / 10
    recovery_factor = 1 - (entry["recovery"] / 10)

    burnout = (
        sleep_factor * 0.4
        + stress_factor * 0.4
        + recovery_factor * 0.2
    )

    fatigue = (
        sleep_factor * 0.6
        + stress_factor * 0.2
        + recovery_factor * 0.2
    )

    recovery_failure = (
        recovery_factor * 0.7
        + stress_factor * 0.3
    )

    return {
        "burnout_probability": _clamp(burnout),
        "fatigue_probability": _clamp(fatigue),
        "recovery_failure_probability": _clamp(recovery_failure),
    }