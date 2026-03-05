"""
zenith.core.normalization

Converts raw behavioral signals into deviations from a personal baseline.
Each deviation represents how far an entry's value sits above or below
the user's established norm for that dimension.
"""


def normalize_entry(entry: dict, baseline: dict) -> dict:
    """
    Compute per-dimension deviations of a single entry from baseline.

    Args:
        entry: A single entry dict matching the Zenith entry schema.
        baseline: Baseline dict produced by compute_baseline().

    Returns:
        Dict with float deviations for sleep, deep_work, stress,
        recovery, mood, and completion_rate.
    """
    if entry["tasks_total"] > 0:
        completion_rate = entry["tasks_completed"] / entry["tasks_total"]
    else:
        completion_rate = 0.0

    return {
        "sleep_dev": entry["sleep_hours"] - baseline["sleep"],
        "deep_work_dev": entry["deep_work_hours"] - baseline["deep_work"],
        "stress_dev": entry["stress"] - baseline["stress"],
        "recovery_dev": entry["recovery"] - baseline["recovery"],
        "mood_dev": entry["mood"] - baseline["mood"],
        "completion_dev": completion_rate - baseline["completion_rate"],
    }