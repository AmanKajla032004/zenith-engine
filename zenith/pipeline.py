"""
Pipeline orchestration: load → score → signal → detect → classify → report.

This is the only module with I/O (file loading, report formatting).
All analytical logic is delegated to scoring, signals, detectors, regime.
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
    compute_all_trends,
    compute_deviation,
)
from zenith.detectors import (
    compute_burnout_flags,
    detect_compensation,
    detect_early_warning,
)
from zenith.regime import classify_regime


# ---------------------------------------------------------------------------
# Data loading
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
# Main analysis
# ---------------------------------------------------------------------------

def analyze(
    filepath: Union[str, Path],
    cfg: ZenithConfig | None = None,
) -> Dict:
    """
    Run the full Zenith analysis pipeline.

    Returns a structured result dict suitable for downstream consumption
    (reporting, serialization, or future ML feature extraction).
    """
    if cfg is None:
        cfg = ZenithConfig()

    # Stage 1: Load + validate
    df = load_data(filepath)

    # Stage 2: Score
    df = compute_domain_scores(df, cfg)

    # Stage 3: Signals
    df = compute_rolling_means(df, cfg)
    df = compute_volatility(df, cfg)
    df = compute_burnout_flags(df, cfg)

    # Stage 4: Extract latest-day snapshot
    latest = df.iloc[-1]
    global_trend, domain_trends = compute_all_trends(df, cfg)
    deviation = compute_deviation(df)
    volatility = round(float(latest["volatility_index"]), 3)

    # Stage 5: Detect patterns
    compensation_flags = detect_compensation(domain_trends, cfg)
    early_warning = detect_early_warning(
        trend_label=global_trend["label"],
        deviation=deviation,
        volatility=volatility,
        cfg=cfg,
    )

    # Stage 6: Classify regime
    regime = classify_regime(
        slope=global_trend["slope"],
        volatility=volatility,
        cfg=cfg,
    )

    return {
        "global_balance": round(float(latest["global_balance"]), 3),
        "trend": global_trend,
        "domain_trends": domain_trends,
        "volatility": volatility,
        "burnout_risk": bool(latest["burnout_flag"]),
        "deviation": deviation,
        "compensation_flags": compensation_flags,
        "early_warning": early_warning,
        "regime": regime,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(result: Dict) -> str:
    """Format the analysis result as a human-readable text report."""
    lines = [
        "ZENITH STATUS REPORT",
        "=" * 50,
        "",
        f"  Global Balance : {result['global_balance']}",
        f"  Trend          : {result['trend']['label']} (slope: {result['trend']['slope']})",
        f"  Regime         : {result['regime']}",
        f"  Volatility     : {result['volatility']}",
        f"  Deviation      : {result['deviation']}",
        f"  Burnout Risk   : {'YES' if result['burnout_risk'] else 'No'}",
        "",
        "  Domain Trends:",
    ]

    for name, trend in result["domain_trends"].items():
        label = name.replace("_", " ").title()
        lines.append(f"    {label:15s} : {trend['label']} (slope: {trend['slope']})")

    if result["compensation_flags"]:
        lines.append("")
        lines.append("  Compensation Detected:")
        for flag in result["compensation_flags"]:
            lines.append(f"    - {flag}")

    if result["early_warning"]:
        lines.append("")
        lines.append("  ⚠  EARLY WARNING: System entering decline trajectory")

    lines.append("")
    lines.append("=" * 50)
    return "\n".join(lines)