"""
Signal extraction: rolling statistics, volatility, trend, and deviation.

These are temporal features computed over the scored DataFrame.
All functions are pure transforms on the DataFrame.
"""

from typing import Dict

import numpy as np
import pandas as pd

from zenith.config import ZenithConfig, TrendThresholds
from zenith.scoring import DOMAIN_COLUMNS


# ---------------------------------------------------------------------------
# Rolling means
# ---------------------------------------------------------------------------

def compute_rolling_means(df: pd.DataFrame, cfg: ZenithConfig) -> pd.DataFrame:
    """Add short-window and long-window rolling means for each domain + global."""
    w = cfg.windows
    targets = list(DOMAIN_COLUMNS) + ["global_balance"]

    for col in targets:
        df[f"{col}_7d_mean"] = df[col].rolling(w.short, min_periods=w.min_periods).mean()
        df[f"{col}_30d_mean"] = df[col].rolling(w.long, min_periods=w.min_periods).mean()

    return df


# ---------------------------------------------------------------------------
# Volatility index
# ---------------------------------------------------------------------------

def compute_volatility(df: pd.DataFrame, cfg: ZenithConfig) -> pd.DataFrame:
    """Weighted rolling standard deviation across mood, stress, deep_work."""
    w = cfg.windows
    vw = cfg.volatility

    components = {
        "mood": vw.mood,
        "stress": vw.stress,
        "deep_work_hours": vw.deep_work,
    }

    vol = pd.Series(0.0, index=df.index)
    for col, weight in components.items():
        std = df[col].rolling(w.short, min_periods=w.min_periods).std().fillna(0.0)
        vol += std * weight

    df["volatility_index"] = vol
    return df


# ---------------------------------------------------------------------------
# Trend / momentum via OLS slope
# ---------------------------------------------------------------------------

def _ols_slope(y: np.ndarray) -> float:
    """
    Ordinary least-squares slope for evenly-spaced data.

    Uses the closed-form solution:  slope = Σ(x_c * y_c) / Σ(x_c²)
    where x_c and y_c are mean-centered.  Avoids polyfit overhead and is
    numerically stable for short windows.
    """
    n = len(y)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    x_c = x - x.mean()
    y_c = y - y.mean()
    denom = np.dot(x_c, x_c)
    if denom == 0.0:
        return 0.0
    return float(np.dot(x_c, y_c) / denom)


def classify_slope(slope: float, t: TrendThresholds) -> str:
    """Map a numeric slope to a human-readable trend label."""
    if slope > t.strong_positive:
        return "Improving (Strong)"
    if slope > t.weak_positive:
        return "Improving (Weak)"
    if slope < t.strong_negative:
        return "Declining (Sharp)"
    if slope < t.weak_negative:
        return "Declining (Weak)"
    return "Stable"


def compute_trend(series: pd.Series, cfg: ZenithConfig) -> Dict[str, object]:
    """Compute slope and label over the trailing short window."""
    t = cfg.trend
    window = series.tail(cfg.windows.short)

    if len(window) < t.min_data_points:
        return {"label": "Insufficient Data", "slope": 0.0}

    slope = _ols_slope(window.values)
    label = classify_slope(slope, t)
    return {"label": label, "slope": round(slope, 4)}


def compute_all_trends(df: pd.DataFrame, cfg: ZenithConfig) -> Dict[str, Dict]:
    """Return trend dicts for all four domains and global balance."""
    domain_map = {
        "recovery": "recovery_score",
        "emotional": "emotional_score",
        "performance": "performance_score",
        "energy_focus": "energy_focus_score",
    }

    global_trend = compute_trend(df["global_balance"], cfg)
    domain_trends = {
        name: compute_trend(df[col], cfg) for name, col in domain_map.items()
    }
    return global_trend, domain_trends


# ---------------------------------------------------------------------------
# Deviation (short vs long mean)
# ---------------------------------------------------------------------------

def compute_deviation(df: pd.DataFrame) -> float:
    """Signed difference: recent short-window mean minus long-window mean."""
    short = df["global_balance_7d_mean"].iloc[-1]
    long_ = df["global_balance_30d_mean"].iloc[-1]
    return round(float(short - long_), 3)