"""
Macro regime classification.

Maps (slope, volatility, deviation) → a named regime label.
Designed as a decision tree for interpretability; future versions
can replace with a trained classifier.
"""

from zenith.config import ZenithConfig


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