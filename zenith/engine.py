"""
zenith.engine

Top-level pipeline for Zenith behavioral analysis. Chains the core
processing stages — baseline computation, normalization, state vector
construction — into the v2 reasoning layer and returns a unified result.
"""

from zenith.core.baseline import compute_baseline
from zenith.core.normalization import normalize_entry
from zenith.core.state_vector import build_state_vector
from zenith.reasoning.reasoning_engine import run_reasoning


def run_zenith(entries: list) -> dict:
    """
    Run the full Zenith behavioral analysis pipeline.

    Args:
        entries: List of entry dicts matching the Zenith entry schema.

    Returns:
        Dict containing baseline, state vector, balance score,
        regime classification, and all reasoning outputs.
        Returns empty dict if entries is empty.
    """
    if not entries:
        return {}

    baseline = compute_baseline(entries)
    entry = entries[-1]
    normalized = normalize_entry(entry, baseline)
    state = build_state_vector(normalized)
    reasoning = run_reasoning(state, entry)

    balance_score = 10 - (
        abs(state["performance"])
        + abs(state["recovery"])
        + abs(state["energy"])
        + abs(state["emotional"])
    )
    balance_score = max(0.0, min(10.0, balance_score))

    if balance_score >= 8:
        regime = "stable"
    elif reasoning["capacity_mismatch"]:
        regime = "overloaded"
    elif balance_score >= 5:
        regime = "moderate_strain"
    else:
        regime = "declining"

    return {
        "baseline": baseline,
        "state": state,
        "balance_score": balance_score,
        "regime": regime,
        **reasoning,
    }