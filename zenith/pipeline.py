"""
Pipeline orchestration: load → score → signal → detect → classify → report.

This is the only module with I/O (file loading, report formatting).
All analytical logic is delegated to scoring, signals, detectors, regime.

v1.2 updates:
    - Core analysis extracted to _analyze_df (pure function)
    - Added analyze_data() for UI/backend integration
    - CLI compatibility preserved via analyze(filepath)
"""

import json
from pathlib import Path
from typing import Dict, Union

import pandas as pd

from zenith.config import ZenithConfig
from zenith.scoring import compute_domain_scores
from zenith.signals import (
    compute_rolling_means,
    compute_volatility,
    compute_multi_horizon_trends,
    compute_trend_confidence,
    compute_divergence_index,
    compute_deviation,
)
from zenith.detectors import (
    compute_burnout_flags,
    detect_compensation,
    detect_early_warning,
)
from zenith.regime import classify_regime, compute_regime_persistence


# ---------------------------------------------------------------------------
# Data loading (CLI mode only)
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = {
    "date", "tasks_completed", "tasks_total",
    "deep_work_hours", "sleep_hours", "mood", "stress", "recovery",
}


def load_data(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load and validate daily tracking data from a JSON file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    if not data:
        raise ValueError("Data file is empty")

    df = pd.DataFrame(data)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ---------------------------------------------------------------------------
# Core analysis (PURE FUNCTION — NO FILE I/O)
# ---------------------------------------------------------------------------

def _analyze_df(df: pd.DataFrame, cfg: ZenithConfig) -> Dict:
    """
    Core analysis operating purely on a DataFrame.

    Stateless.
    No file reads.
    Safe for backend / API usage.
    """

    # Stage 1: Score
    df = compute_domain_scores(df, cfg)

    # Stage 2: Signals
    df = compute_rolling_means(df, cfg)
    df = compute_volatility(df, cfg)
    df = compute_burnout_flags(df, cfg)

    # Stage 3: Multi-horizon trends
    global_short, global_long, domain_shorts, domain_longs = (
        compute_multi_horizon_trends(df, cfg)
    )

    latest = df.iloc[-1]
    deviation = compute_deviation(df)
    volatility = round(float(latest["volatility_index"]), 3)

    # Stage 4: Confidence + divergence
    trend_confidence = compute_trend_confidence(
        short_trend=global_short,
        long_trend=global_long,
        volatility=volatility,
        cfg=cfg,
    )

    divergence_index = compute_divergence_index(domain_shorts)

    # Stage 5: Pattern detection
    compensation_flags = detect_compensation(domain_shorts, cfg)

    early_warning = detect_early_warning(
        trend_label=global_short["label"],
        deviation=deviation,
        volatility=volatility,
        cfg=cfg,
    )

    # Stage 6: Regime classification
    regime = classify_regime(
        slope=global_short["slope"],
        volatility=volatility,
        cfg=cfg,
    )

    regime_persistence_days = compute_regime_persistence(df, cfg)

    return {
        "global_balance": round(float(latest["global_balance"]), 3),
        "trend": {
            "short": global_short,
            "long": global_long,
        },
        "domain_trends": {
            "short": domain_shorts,
            "long": domain_longs,
        },
        "trend_confidence": trend_confidence,
        "divergence_index": divergence_index,
        "volatility": volatility,
        "burnout_risk": bool(latest["burnout_flag"]),
        "deviation": deviation,
        "compensation_flags": compensation_flags,
        "early_warning": early_warning,
        "regime": regime,
        "regime_persistence_days": regime_persistence_days,
    }


# ---------------------------------------------------------------------------
# Public Entry Points
# ---------------------------------------------------------------------------

def analyze(
    filepath: Union[str, Path],
    cfg: ZenithConfig | None = None,
) -> Dict:
    """
    CLI-compatible entry point.
    Reads JSON file and runs analysis.
    """
    if cfg is None:
        cfg = ZenithConfig()

    df = load_data(filepath)
    return _analyze_df(df, cfg)


def analyze_data(
    data: list[dict],
    cfg: ZenithConfig | None = None,
) -> Dict:
    """
    Backend / UI integration entry point.

    Accepts list-of-dict JSON data directly.
    No file system usage.
    """
    if cfg is None:
        cfg = ZenithConfig()

    if not data:
        raise ValueError("Input data cannot be empty")

    df = pd.DataFrame(data)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    return _analyze_df(df, cfg)


# ---------------------------------------------------------------------------
# Report generation (unchanged)
# ---------------------------------------------------------------------------

def generate_report(result: Dict) -> str:
    """Format the analysis result as a human-readable text report."""
    t_short = result["trend"]["short"]
    t_long = result["trend"]["long"]

    lines = [
        "ZENITH STATUS REPORT",
        "=" * 58,
        "",
        f"  Global Balance      : {result['global_balance']}",
        f"  Trend (7d)          : {t_short['label']} (slope: {t_short['slope']}, R²: {t_short['r_squared']})",
        f"  Trend (30d)         : {t_long['label']} (slope: {t_long['slope']}, R²: {t_long['r_squared']})",
        f"  Trend Confidence    : {result['trend_confidence']}",
        f"  Regime              : {result['regime']} ({result['regime_persistence_days']}d streak)",
        f"  Volatility          : {result['volatility']}",
        f"  Divergence Index    : {result['divergence_index']}",
        f"  Deviation (7d-30d)  : {result['deviation']}",
        f"  Burnout Risk        : {'YES' if result['burnout_risk'] else 'No'}",
        "",
        "  Domain Trends (7d / 30d):",
    ]

    d_short = result["domain_trends"]["short"]
    d_long = result["domain_trends"]["long"]

    for name in d_short:
        label = name.replace("_", " ").title()
        s = d_short[name]
        l = d_long[name]
        lines.append(
            f"    {label:15s} : {s['label']:22s} (slope: {s['slope']:+.4f})"
            f"  |  {l['label']:22s} (slope: {l['slope']:+.4f})"
        )

    if result["compensation_flags"]:
        lines.append("")
        lines.append("  Compensation Detected:")
        for flag in result["compensation_flags"]:
            lines.append(f"    - {flag}")

    if result["early_warning"]:
        lines.append("")
        lines.append("  ⚠  EARLY WARNING: System entering decline trajectory")

    lines.append("")
    lines.append("=" * 58)
    return "\n".join(lines)