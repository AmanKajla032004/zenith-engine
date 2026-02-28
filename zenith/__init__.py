"""
ZENITH v1.0 — Rule-Based Personal Stability Engine

A deterministic, interpretable behavioral modeling engine that analyzes
daily self-tracking data and produces structured intelligence about
system stability, trends, and risk patterns.

Architecture:
    config      — All thresholds, weights, and window sizes (single source of truth)
    scoring     — Domain score computations (Recovery, Emotional, Performance, Energy)
    signals     — Rolling statistics, volatility, trend/momentum, deviation
    detectors   — Pattern detectors (burnout, compensation, early warning)
    regime      — Macro regime classification
    pipeline    — Orchestration: load → score → signal → detect → classify → report
"""

"""
ZENITH v1.2 — Integration-Ready Deterministic Stability Engine

Core engine is fully stateless and safe for backend/API usage.

Public API:
    analyze(filepath)      → CLI mode
    analyze_data(data)     → UI / backend mode
    generate_report(result) → formatted report
"""

from zenith.pipeline import analyze, analyze_data, generate_report

__version__ = "1.2.0"

__all__ = ["analyze", "analyze_data", "generate_report"]