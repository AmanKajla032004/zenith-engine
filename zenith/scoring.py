"""
Domain scoring: transforms raw daily inputs into [0, 10] domain scores.

Each function is a pure column transform — takes a DataFrame, returns it
with new columns appended. No rolling stats or cross-day logic here.
"""

import numpy as np
import pandas as pd

from zenith.config import ZenithConfig


DOMAIN_COLUMNS = (
    "recovery_score",
    "emotional_score",
    "performance_score",
    "energy_focus_score",
)


def compute_domain_scores(df: pd.DataFrame, cfg: ZenithConfig) -> pd.DataFrame:
    """Compute all four domain scores and the global balance in one pass."""
    s = cfg.scoring
    dw = cfg.domain_weights
    gw = cfg.global_weights
    cap = s.scale_max

    # -- Intermediate scores --------------------------------------------------

    # Task completion: proportion × 10, clamped [0, 10]
    # Guard against tasks_total == 0
    safe_total = df["tasks_total"].replace(0, np.nan)
    df["completion_score"] = np.clip(
        (df["tasks_completed"] / safe_total) * cap, 0, cap
    ).fillna(0.0)

    # Deep work: linear up to cap hours, then saturates
    df["deep_work_score"] = (
        np.minimum(df["deep_work_hours"].clip(lower=0), s.deep_work_cap_hours)
        / s.deep_work_cap_hours
    ) * cap

    # Sleep: penalty for deviation from optimal
    df["sleep_hour_score"] = np.clip(
        cap - np.abs(df["sleep_hours"] - s.sleep_optimal_hours) * s.sleep_penalty_per_hour,
        0, cap,
    )

    # Inverted stress (used in multiple domains, computed once)
    stress_inv = cap - df["stress"].clip(0, cap)

    # -- Domain scores --------------------------------------------------------

    df["recovery_score"] = (
        df["sleep_hour_score"] * dw.recovery_sleep
        + df["recovery"] * dw.recovery_raw
    )

    df["emotional_score"] = (
        df["mood"] * dw.emotional_mood
        + stress_inv * dw.emotional_stress_inv
    )

    df["performance_score"] = (
        df["completion_score"] * dw.performance_completion
        + df["deep_work_score"] * dw.performance_deep_work
    )

    df["energy_focus_score"] = (
        df["deep_work_score"] * dw.energy_deep_work
        + df["mood"] * dw.energy_mood
        + stress_inv * dw.energy_stress_inv
    )

    # -- Global balance -------------------------------------------------------

    df["global_balance"] = (
        df["recovery_score"] * gw.recovery
        + df["emotional_score"] * gw.emotional
        + df["performance_score"] * gw.performance
        + df["energy_focus_score"] * gw.energy_focus
    )

    return df