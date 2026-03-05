"""
zenith.core.baseline

Computes personal behavioral baselines from historical entries.
Uses the most recent entries within the baseline window to derive
mean values across key behavioral dimensions.
"""

BASELINE_WINDOW = 60


def compute_baseline(entries: list) -> dict:
    """
    Compute a personal behavioral baseline from historical entries.

    Takes the most recent `BASELINE_WINDOW` entries (or all available
    if fewer exist) and returns mean values for each behavioral dimension.

    Args:
        entries: List of entry dicts matching the Zenith entry schema.

    Returns:
        Dict with baseline floats for sleep, deep_work, stress,
        recovery, mood, and completion_rate.
    """
    recent = entries[-BASELINE_WINDOW:]

    if not recent:
        return {
            "sleep": 0.0,
            "deep_work": 0.0,
            "stress": 0.0,
            "recovery": 0.0,
            "mood": 0.0,
            "completion_rate": 0.0,
        }

    n = len(recent)

    sleep = sum(e["sleep_hours"] for e in recent) / n
    deep_work = sum(e["deep_work_hours"] for e in recent) / n
    stress = sum(e["stress"] for e in recent) / n
    recovery = sum(e["recovery"] for e in recent) / n
    mood = sum(e["mood"] for e in recent) / n

    valid = [e for e in recent if e["tasks_total"] != 0]
    if valid:
        completion_rate = sum(
            e["tasks_completed"] / e["tasks_total"] for e in valid
        ) / len(valid)
    else:
        completion_rate = 0.0

    return {
        "sleep": sleep,
        "deep_work": deep_work,
        "stress": stress,
        "recovery": recovery,
        "mood": mood,
        "completion_rate": completion_rate,
    }