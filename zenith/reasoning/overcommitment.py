"""
zenith.reasoning.overcommitment

Detects workload vs capacity mismatch by computing a behavioral
pressure index. Pressure rises when task load is high relative to
deep work capacity and when stress outweighs recovery.
"""


def detect_overcommitment(entry: dict) -> dict:
    """
    Compute behavioral pressure and detect capacity mismatch.

    Args:
        entry: A single entry dict matching the Zenith entry schema.

    Returns:
        Dict with behavioral_pressure_index (float) and
        capacity_mismatch (bool).
    """
    if entry["deep_work_hours"] == 0:
        pressure = 0.0
    else:
        pressure = (
            (entry["tasks_total"] / entry["deep_work_hours"])
            * (entry["stress"] / max(entry["recovery"], 1))
        )

    return {
        "behavioral_pressure_index": pressure,
        "capacity_mismatch": pressure > 1,
    }