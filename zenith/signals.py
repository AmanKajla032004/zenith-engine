"""
Signal extraction: rolling statistics, volatility, trend, confidence, and divergence.

These are temporal features computed over the scored DataFrame.
All functions are pure transforms — no I/O, no side effects.

v1.1 additions:
    - Multi-horizon trend (short + long window slopes)
    - OLS R² (goodness-of-fit) for trend reliability
    - Composite trend confidence score [0, 1]
    - Cross-domain divergence index
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from zenith.config import ZenithConfig, TrendThresholds
from zenith.scoring import DOMAIN_COLUMNS


# ---------------------------------------------------------------------------
# Domain name → column mapping (used by multiple functions)
# ---------------------------------------------------------------------------

DOMAIN_MAP = {
    "recovery": "recovery_score",
    "emotional": "emotional_score",
    "performance": "performance_score",
    "energy_focus": "energy_focus_score",
}


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
# OLS primitives (slope + R²)
# ---------------------------------------------------------------------------

def _ols_slope(y: np.ndarray) -> float:
    """
    Ordinary least-squares slope for evenly-spaced data.

    Uses the closed-form solution:  slope = Σ(x_c · y_c) / Σ(x_c²)
    where x_c and y_c are mean-centered.
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


def _ols_r_squared(y: np.ndarray) -> float:
    """
    Coefficient of determination (R²) for a linear fit on evenly-spaced data.

    R² = 1 - SS_res / SS_tot

    Returns 0.0 for degenerate cases (n < 2, constant series).
    Clamped to [0, 1] for numerical safety.
    """
    n = len(y)
    if n < 2:
        return 0.0

    x = np.arange(n, dtype=np.float64)
    x_c = x - x.mean()
    y_c = y - y.mean()

    ss_tot = np.dot(y_c, y_c)
    if ss_tot == 0.0:
        # Constant series — a flat line fits perfectly, but the trend
        # is trivially uninformative.  Return 0.0 (no explanatory power).
        return 0.0

    denom = np.dot(x_c, x_c)
    if denom == 0.0:
        return 0.0

    slope = np.dot(x_c, y_c) / denom
    intercept = y.mean() - slope * x.mean()
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)

    r2 = 1.0 - ss_res / ss_tot
    return float(np.clip(r2, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Trend classification
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Single-series trend (short or long horizon)
# ---------------------------------------------------------------------------

def _compute_trend_for_window(
    series: pd.Series,
    window_size: int,
    cfg: ZenithConfig,
) -> Dict[str, object]:
    """
    Compute slope, label, and R² over the trailing `window_size` days.

    Returns:
        {"label": str, "slope": float, "r_squared": float}
    """
    t = cfg.trend
    tail = series.tail(window_size)

    if len(tail) < t.min_data_points:
        return {"label": "Insufficient Data", "slope": 0.0, "r_squared": 0.0}

    values = tail.values.astype(np.float64)
    slope = _ols_slope(values)
    r2 = _ols_r_squared(values)
    label = classify_slope(slope, t)

    return {"label": label, "slope": round(slope, 4), "r_squared": round(r2, 4)}


def compute_trend(series: pd.Series, cfg: ZenithConfig) -> Dict[str, object]:
    """Backward-compatible: compute trend over the short window (7d)."""
    return _compute_trend_for_window(series, cfg.windows.short, cfg)


# ---------------------------------------------------------------------------
# Multi-horizon trends (Feature 1)
# ---------------------------------------------------------------------------

def compute_multi_horizon_trends(
    df: pd.DataFrame,
    cfg: ZenithConfig,
) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Compute short-window (7d) and long-window (30d) trends for global balance
    and all four domains.

    Returns:
        (global_short, global_long, domain_shorts, domain_longs)

    global_short / global_long:
        {"label": str, "slope": float, "r_squared": float}

    domain_shorts / domain_longs:
        {"recovery": {...}, "emotional": {...}, ...}
    """
    gb = df["global_balance"]

    global_short = _compute_trend_for_window(gb, cfg.windows.short, cfg)
    global_long = _compute_trend_for_window(gb, cfg.windows.long, cfg)

    domain_shorts = {
        name: _compute_trend_for_window(df[col], cfg.windows.short, cfg)
        for name, col in DOMAIN_MAP.items()
    }
    domain_longs = {
        name: _compute_trend_for_window(df[col], cfg.windows.long, cfg)
        for name, col in DOMAIN_MAP.items()
    }

    return global_short, global_long, domain_shorts, domain_longs


# Backward-compatible alias
def compute_all_trends(
    df: pd.DataFrame,
    cfg: ZenithConfig,
) -> Tuple[Dict, Dict]:
    """Return (global_trend, domain_trends) using the short window. Legacy API."""
    global_short, _, domain_shorts, _ = compute_multi_horizon_trends(df, cfg)
    # Strip r_squared from legacy return to keep exact backward compat in structure
    # Actually — the extra key is additive, not breaking.  Keep it.
    return global_short, domain_shorts


# ---------------------------------------------------------------------------
# Trend confidence (Feature 2)
# ---------------------------------------------------------------------------

def _sign(x: float) -> int:
    """Return +1, -1, or 0."""
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def compute_trend_confidence(
    short_trend: Dict,
    long_trend: Dict,
    volatility: float,
    cfg: ZenithConfig,
) -> float:
    """
    Composite trend confidence score ∈ [0, 1].

    Components:
        1. R² of the short-window OLS fit (how linear is the recent trend)
        2. Sign agreement between 7d and 30d slopes (are horizons aligned?)
        3. Volatility penalty (noisy systems → less trustworthy trend)

    Formula:
        confidence = w_r2 * R²  +  w_agreement * agreement  -  w_vol * vol_penalty
        clamped to [0, 1]
    """
    tc = cfg.trend_confidence

    r2 = short_trend.get("r_squared", 0.0)

    short_sign = _sign(short_trend["slope"])
    long_sign = _sign(long_trend["slope"])
    # Agreement: 1.0 if same sign (including both zero), 0.0 otherwise
    agreement = 1.0 if short_sign == long_sign else 0.0

    vol_penalty = min(volatility / tc.volatility_normalizer, 1.0)

    raw = (
        tc.weight_r2 * r2
        + tc.weight_agreement * agreement
        - tc.weight_volatility * vol_penalty
    )

    return round(float(np.clip(raw, 0.0, 1.0)), 4)


# ---------------------------------------------------------------------------
# Cross-domain divergence index (Feature 3)
# ---------------------------------------------------------------------------

def compute_divergence_index(domain_trends: Dict[str, Dict]) -> float:
    """
    Standard deviation of domain slopes — measures inter-domain coherence.

    Low divergence  → domains moving in sync (coherent system)
    High divergence → domains pulling in opposite directions (fragmented)

    Uses population std (ddof=0) since we always have exactly 4 domains
    and want the actual dispersion, not a sample estimate.
    """
    slopes = np.array(
        [t["slope"] for t in domain_trends.values()],
        dtype=np.float64,
    )

    if len(slopes) < 2:
        return 0.0

    return round(float(np.std(slopes, ddof=0)), 4)


# ---------------------------------------------------------------------------
# Deviation (short vs long mean)
# ---------------------------------------------------------------------------

def compute_deviation(df: pd.DataFrame) -> float:
    """Signed difference: recent short-window mean minus long-window mean."""
    short = df["global_balance_7d_mean"].iloc[-1]
    long_ = df["global_balance_30d_mean"].iloc[-1]
    return round(float(short - long_), 3)