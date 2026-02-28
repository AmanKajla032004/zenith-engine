"""
Pattern detectors: burnout, compensation, early warning.

Each detector is a pure function that inspects scored/signaled data
and returns structured flags. No side effects.
"""

from typing import Dict, List

import pandas as pd

from zenith.config import ZenithConfig


# ---------------------------------------------------------------------------
# Burnout detection
# ---------------------------------------------------------------------------

def compute_burnout_flags(df: pd.DataFrame, cfg: ZenithConfig) -> pd.DataFrame:
    """
    Flag days where recovery is consecutively low while performance stays high.

    A burnout flag fires on day T if:
        - recovery_score < threshold for the last `consecutive_days` days (inclusive)
        - performance_score > threshold on day T

    Uses rolling min over the consecutive window to avoid shift-chain fragility.
    """
    bt = cfg.burnout
    window = bt.consecutive_days

    low_recovery_streak = (
        df["recovery_score"]
        .rolling(window, min_periods=window)
        .max()                    # max in window; if max < threshold, ALL are low
        .lt(bt.low_recovery)
    )

    high_perf = df["performance_score"] > bt.high_performance

    df["burnout_flag"] = low_recovery_streak & high_perf
    # Fill NaN from rolling with False
    df["burnout_flag"] = df["burnout_flag"].fillna(False).astype(bool)

    return df


# ---------------------------------------------------------------------------
# Compensation detection (declarative rule engine)
# ---------------------------------------------------------------------------

def detect_compensation(
    domain_trends: Dict[str, Dict],
    cfg: ZenithConfig,
) -> List[str]:
    """
    Scan all configured compensation rules against current domain trends.

    A rule fires when:
        - The rising_domain slope > rising threshold
        - The falling_domain slope < falling threshold

    Rules are defined in config.compensation_rules, making new patterns
    trivially addable without code changes.
    """
    ct = cfg.compensation
    flags: List[str] = []

    for rule in cfg.compensation_rules:
        rising = domain_trends.get(rule.rising_domain, {})
        falling = domain_trends.get(rule.falling_domain, {})

        rising_slope = rising.get("slope", 0.0)
        falling_slope = falling.get("slope", 0.0)

        if rising_slope > ct.rising_slope and falling_slope < ct.falling_slope:
            flags.append(rule.message)

    return flags


# ---------------------------------------------------------------------------
# Early warning
# ---------------------------------------------------------------------------

def detect_early_warning(
    trend_label: str,
    deviation: float,
    volatility: float,
    cfg: ZenithConfig,
) -> bool:
    """
    Trigger early warning when three conditions coincide:
        1. Global trend is declining (any severity)
        2. Short-vs-long deviation is below floor
        3. Volatility exceeds ceiling
    """
    ew = cfg.early_warning
    declining = "Declining" in trend_label
    return declining and deviation < ew.deviation_floor and volatility > ew.volatility_ceiling