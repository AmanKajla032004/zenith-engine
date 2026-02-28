"""
Macro regime classification and persistence tracking.

Maps (slope, volatility) → a named regime label.
Designed as a decision tree for interpretability; future versions
can replace with a trained classifier.

v1.1: Added regime persistence — computes how many consecutive days
the current regime label has held by classifying each historical day.
"""

from typing import Dict

import numpy as np
import pandas as pd

from zenith.config import ZenithConfig


# ---------------------------------------------------------------------------
# Single-day regime classification
# ---------------------------------------------------------------------------

def classify_regime(
    slope: float,
    volatility: float,
    cfg: ZenithConfig,
) -> str:
    """
    Classify the current macro state into one of five regimes.

    Decision order matters — first match wins (most specific first).

    Regimes:
        Stable Expansion   — positive momentum, low volatility
        Volatile Growth     — positive momentum, high volatility
        Systemic Decline    — negative momentum, high volatility
        Stable Plateau      — flat momentum, low volatility
        Transitional State  — everything else (ambiguous)
    """
    r = cfg.regime

    if slope > r.growth_slope and volatility < r.low_volatility:
        return "Stable Expansion"

    if slope > r.growth_slope and volatility >= r.low_volatility:
        return "Volatile Growth"

    if slope < r.decline_slope and volatility > r.high_volatility:
        return "Systemic Decline"

    if abs(slope) <= r.growth_slope and volatility < r.low_volatility:
        return "Stable Plateau"

    return "Transitional State"


# ---------------------------------------------------------------------------
# Regime persistence (Feature 4)
# ---------------------------------------------------------------------------

def compute_regime_persistence(
    df: pd.DataFrame,
    cfg: ZenithConfig,
) -> int:
    """
    Count how many consecutive days (ending at the latest day) the current
    regime label has been in effect.

    Approach:
        For each day that has enough history for a 7-day slope, classify its
        regime using the rolling slope and volatility available at that day.
        Then count the trailing streak of the final regime label.

    The rolling slope is computed as the OLS slope over the trailing short
    window of global_balance.  Volatility is read from the pre-computed
    volatility_index column.

    Returns:
        Number of consecutive days the current regime has persisted (≥ 1).
    """
    w = cfg.windows.short
    n = len(df)

    if n == 0:
        return 0

    gb = df["global_balance"].values.astype(np.float64)
    vol = df["volatility_index"].values.astype(np.float64)

    # Pre-compute rolling slopes for every day that has enough data
    regimes = []
    for i in range(n):
        if i < w - 1:
            # Not enough history for a full window — use whatever is available
            window = gb[: i + 1]
        else:
            window = gb[i - w + 1 : i + 1]

        if len(window) < 2:
            slope = 0.0
        else:
            x = np.arange(len(window), dtype=np.float64)
            x_c = x - x.mean()
            y_c = window - window.mean()
            denom = np.dot(x_c, x_c)
            slope = float(np.dot(x_c, y_c) / denom) if denom != 0 else 0.0

        regimes.append(classify_regime(slope, float(vol[i]), cfg))

    # Count trailing streak of the final regime
    current = regimes[-1]
    streak = 0
    for label in reversed(regimes):
        if label == current:
            streak += 1
        else:
            break

    return streak