"""Zenith v1.1 — Standalone test suite (no pytest dependency)."""
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from zenith.config import ZenithConfig, GlobalWeights, TrendConfidenceParams
from zenith.scoring import compute_domain_scores, DOMAIN_COLUMNS
from zenith.signals import (
    _ols_slope, _ols_r_squared, compute_trend, compute_volatility,
    compute_multi_horizon_trends, compute_trend_confidence,
    compute_divergence_index, compute_rolling_means, compute_deviation,
)
from zenith.detectors import compute_burnout_flags, detect_compensation, detect_early_warning
from zenith.regime import classify_regime, compute_regime_persistence
from zenith.pipeline import analyze, load_data, generate_report

CFG = ZenithConfig()
TEST_DATA = Path(__file__).parent / "test_data.json"

passed = 0
failed = 0


def test(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  ✓ {name}")
        passed += 1
    except Exception as e:
        print(f"  ✗ {name}: {e}")
        traceback.print_exc()
        failed += 1


def approx(a, b, tol=0.01):
    assert abs(a - b) < tol, f"{a} != {b} (tol={tol})"


def scored_df():
    df = load_data(TEST_DATA)
    return compute_domain_scores(df, CFG)


def full_df():
    df = scored_df()
    df = compute_rolling_means(df, CFG)
    df = compute_volatility(df, CFG)
    df = compute_burnout_flags(df, CFG)
    return df


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════
print("\n[Config]")

def t_weights_sum():
    gw = CFG.global_weights
    approx(gw.recovery + gw.emotional + gw.performance + gw.energy_focus, 1.0, 1e-9)
test("global weights sum to 1", t_weights_sum)

def t_bad_weights():
    try:
        GlobalWeights(recovery=0.5, emotional=0.5, performance=0.5, energy_focus=0.5)
        raise RuntimeError("Should have raised ValueError")
    except ValueError:
        pass
test("invalid global weights raises ValueError", t_bad_weights)

def t_trend_confidence_config():
    tc = CFG.trend_confidence
    total = tc.weight_r2 + tc.weight_agreement + tc.weight_volatility
    approx(total, 1.0, 1e-9)
test("trend confidence weights sum to 1", t_trend_confidence_config)


# ═══════════════════════════════════════════════════════════════════════
# SCORING
# ═══════════════════════════════════════════════════════════════════════
print("\n[Scoring]")

def t_domain_cols():
    df = scored_df()
    for col in DOMAIN_COLUMNS:
        assert col in df.columns, f"Missing: {col}"
test("all domain columns created", t_domain_cols)

def t_score_range():
    df = scored_df()
    for col in DOMAIN_COLUMNS:
        assert df[col].min() >= 0, f"{col} negative"
        assert df[col].max() <= 10.5, f"{col} > 10.5"
test("scores in valid range", t_score_range)

def t_global_exists():
    assert "global_balance" in scored_df().columns
test("global_balance column exists", t_global_exists)

def t_zero_tasks():
    rows = [{"date": "2026-01-01", "tasks_completed": 0, "tasks_total": 0,
             "deep_work_hours": 4, "sleep_hours": 7, "mood": 7, "stress": 3, "recovery": 6}]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = compute_domain_scores(df, CFG)
    assert df["completion_score"].iloc[0] == 0.0
    assert not np.isnan(df["global_balance"].iloc[0])
test("zero tasks handled gracefully", t_zero_tasks)


# ═══════════════════════════════════════════════════════════════════════
# OLS PRIMITIVES
# ═══════════════════════════════════════════════════════════════════════
print("\n[OLS Primitives]")

def t_flat_slope():
    approx(_ols_slope(np.array([5.0, 5.0, 5.0, 5.0])), 0.0, 1e-9)
test("flat series → slope=0", t_flat_slope)

def t_pos_slope():
    approx(_ols_slope(np.array([1.0, 2.0, 3.0, 4.0])), 1.0, 1e-9)
test("linear series → slope=1", t_pos_slope)

def t_single_slope():
    approx(_ols_slope(np.array([7.0])), 0.0, 1e-9)
test("single point → slope=0", t_single_slope)

def t_r2_perfect():
    approx(_ols_r_squared(np.array([1.0, 2.0, 3.0, 4.0])), 1.0, 1e-9)
test("R² = 1.0 for perfect linear", t_r2_perfect)

def t_r2_constant():
    approx(_ols_r_squared(np.array([5.0, 5.0, 5.0, 5.0])), 0.0, 1e-9)
test("R² = 0.0 for constant series", t_r2_constant)

def t_r2_noisy():
    r2 = _ols_r_squared(np.array([1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 7.0]))
    assert 0.0 < r2 < 1.0, f"Noisy R² should be in (0,1), got {r2}"
test("R² in (0,1) for noisy data", t_r2_noisy)

def t_r2_single():
    approx(_ols_r_squared(np.array([7.0])), 0.0, 1e-9)
test("R² = 0.0 for single point", t_r2_single)


# ═══════════════════════════════════════════════════════════════════════
# TREND CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════
print("\n[Trend Classification]")

def t_trend_improving():
    r = compute_trend(pd.Series([1, 2, 3, 4, 5, 6, 7]), CFG)
    assert r["label"] == "Improving (Strong)", f"Got: {r['label']}"
    assert r["slope"] > 0
    assert "r_squared" in r
test("improving strong trend", t_trend_improving)

def t_trend_stable():
    r = compute_trend(pd.Series([5.0, 5.01, 5.0, 4.99, 5.0, 5.01, 5.0]), CFG)
    assert r["label"] == "Stable", f"Got: {r['label']}"
test("stable trend", t_trend_stable)


# ═══════════════════════════════════════════════════════════════════════
# FEATURE 1: MULTI-HORIZON TRENDS
# ═══════════════════════════════════════════════════════════════════════
print("\n[Feature 1: Multi-Horizon Trends]")

def t_multi_returns_four():
    df = full_df()
    result = compute_multi_horizon_trends(df, CFG)
    assert len(result) == 4, f"Expected 4-tuple, got {len(result)}"
test("multi-horizon returns 4-tuple", t_multi_returns_four)

def t_multi_short_has_keys():
    df = full_df()
    g_short, g_long, d_short, d_long = compute_multi_horizon_trends(df, CFG)
    for trend in [g_short, g_long]:
        assert "slope" in trend and "label" in trend and "r_squared" in trend
test("global trends have slope/label/r_squared", t_multi_short_has_keys)

def t_multi_domains_complete():
    df = full_df()
    _, _, d_short, d_long = compute_multi_horizon_trends(df, CFG)
    expected = {"recovery", "emotional", "performance", "energy_focus"}
    assert set(d_short.keys()) == expected
    assert set(d_long.keys()) == expected
test("all 4 domains present in short and long", t_multi_domains_complete)

def t_multi_short_long_differ():
    df = full_df()
    g_short, g_long, _, _ = compute_multi_horizon_trends(df, CFG)
    # With 15 data points and different window sizes, slopes should differ
    assert g_short["slope"] != g_long["slope"], "Short and long slopes should differ for test data"
test("7d and 30d slopes differ for test data", t_multi_short_long_differ)


# ═══════════════════════════════════════════════════════════════════════
# FEATURE 2: TREND CONFIDENCE
# ═══════════════════════════════════════════════════════════════════════
print("\n[Feature 2: Trend Confidence]")

def t_conf_range():
    df = full_df()
    g_short, g_long, _, _ = compute_multi_horizon_trends(df, CFG)
    vol = df["volatility_index"].iloc[-1]
    c = compute_trend_confidence(g_short, g_long, vol, CFG)
    assert 0.0 <= c <= 1.0, f"Confidence out of range: {c}"
test("confidence in [0, 1]", t_conf_range)

def t_conf_perfect_linear():
    # Perfect linear data, zero volatility, same sign → near max confidence
    short = {"slope": 1.0, "r_squared": 1.0, "label": "Improving (Strong)"}
    long_ = {"slope": 0.5, "r_squared": 0.9, "label": "Improving (Weak)"}
    c = compute_trend_confidence(short, long_, 0.0, CFG)
    assert c > 0.7, f"Expected high confidence, got {c}"
test("high confidence for perfect linear + zero vol", t_conf_perfect_linear)

def t_conf_disagreement():
    # Slopes disagree in sign → agreement = 0
    short = {"slope": 0.5, "r_squared": 0.8, "label": "Improving"}
    long_ = {"slope": -0.3, "r_squared": 0.5, "label": "Declining"}
    c = compute_trend_confidence(short, long_, 0.0, CFG)
    # Without agreement bonus: 0.5 * 0.8 = 0.4
    assert c < 0.5, f"Expected lower confidence on disagreement, got {c}"
test("lower confidence when horizons disagree", t_conf_disagreement)

def t_conf_high_volatility_penalty():
    short = {"slope": 1.0, "r_squared": 1.0, "label": "Improving"}
    long_ = {"slope": 0.5, "r_squared": 0.9, "label": "Improving"}
    c_low_vol = compute_trend_confidence(short, long_, 0.0, CFG)
    c_high_vol = compute_trend_confidence(short, long_, 3.0, CFG)
    assert c_high_vol < c_low_vol, f"High vol ({c_high_vol}) should < low vol ({c_low_vol})"
test("higher volatility → lower confidence", t_conf_high_volatility_penalty)

def t_conf_floor_at_zero():
    # Everything bad: low R², disagreement, max volatility → clamp to 0
    short = {"slope": 0.01, "r_squared": 0.0, "label": "Stable"}
    long_ = {"slope": -0.5, "r_squared": 0.0, "label": "Declining"}
    c = compute_trend_confidence(short, long_, 10.0, CFG)
    assert c == 0.0, f"Expected 0.0, got {c}"
test("confidence floors at 0.0", t_conf_floor_at_zero)


# ═══════════════════════════════════════════════════════════════════════
# FEATURE 3: CROSS-DOMAIN DIVERGENCE
# ═══════════════════════════════════════════════════════════════════════
print("\n[Feature 3: Divergence Index]")

def t_div_zero_when_aligned():
    trends = {
        "recovery": {"slope": 0.5}, "emotional": {"slope": 0.5},
        "performance": {"slope": 0.5}, "energy_focus": {"slope": 0.5},
    }
    approx(compute_divergence_index(trends), 0.0, 1e-9)
test("divergence = 0 when all slopes equal", t_div_zero_when_aligned)

def t_div_positive_when_spread():
    trends = {
        "recovery": {"slope": 1.0}, "emotional": {"slope": -1.0},
        "performance": {"slope": 0.5}, "energy_focus": {"slope": -0.5},
    }
    d = compute_divergence_index(trends)
    assert d > 0, f"Expected positive divergence, got {d}"
test("divergence > 0 when slopes differ", t_div_positive_when_spread)

def t_div_known_value():
    # slopes: [1, -1, 0, 0] → mean=0, std = sqrt((1+1+0+0)/4) = sqrt(0.5) ≈ 0.7071
    trends = {
        "a": {"slope": 1.0}, "b": {"slope": -1.0},
        "c": {"slope": 0.0}, "d": {"slope": 0.0},
    }
    approx(compute_divergence_index(trends), 0.7071, 0.001)
test("divergence matches known std", t_div_known_value)


# ═══════════════════════════════════════════════════════════════════════
# VOLATILITY
# ═══════════════════════════════════════════════════════════════════════
print("\n[Volatility]")

def t_constant_vol():
    rows = [{"date": f"2026-01-{i+1:02d}", "mood": 7, "stress": 3, "deep_work_hours": 4,
             "tasks_completed": 5, "tasks_total": 7, "sleep_hours": 7, "recovery": 6}
            for i in range(7)]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = compute_domain_scores(df, CFG)
    df = compute_volatility(df, CFG)
    approx(df["volatility_index"].iloc[-1], 0.0, 1e-9)
test("constant data → zero volatility", t_constant_vol)


# ═══════════════════════════════════════════════════════════════════════
# DETECTORS
# ═══════════════════════════════════════════════════════════════════════
print("\n[Detectors]")

def t_burnout_fires():
    df = pd.DataFrame([
        {"recovery_score": 3.0, "performance_score": 8.0},
        {"recovery_score": 4.0, "performance_score": 8.0},
    ])
    df = compute_burnout_flags(df, CFG)
    assert df["burnout_flag"].iloc[-1] == True
test("burnout fires on consecutive low recovery + high perf", t_burnout_fires)

def t_no_burnout():
    df = pd.DataFrame([
        {"recovery_score": 7.0, "performance_score": 8.0},
        {"recovery_score": 7.0, "performance_score": 8.0},
    ])
    df = compute_burnout_flags(df, CFG)
    assert not df["burnout_flag"].iloc[-1]
test("no burnout when recovery fine", t_no_burnout)

def t_comp_detected():
    trends = {"performance": {"slope": 0.2}, "recovery": {"slope": -0.2},
              "emotional": {"slope": 0.0}, "energy_focus": {"slope": 0.0}}
    flags = detect_compensation(trends, CFG)
    assert any("recovery" in f.lower() for f in flags), f"Got: {flags}"
test("compensation: perf↑ recovery↓ detected", t_comp_detected)

def t_no_comp():
    trends = {"performance": {"slope": 0.2}, "recovery": {"slope": 0.2},
              "emotional": {"slope": 0.1}, "energy_focus": {"slope": 0.1}}
    assert len(detect_compensation(trends, CFG)) == 0
test("no compensation when aligned", t_no_comp)

def t_ew_triggers():
    assert detect_early_warning("Declining (Sharp)", -1.0, 2.0, CFG) is True
test("early warning triggers", t_ew_triggers)

def t_ew_silent():
    assert detect_early_warning("Stable", 0.0, 0.5, CFG) is False
test("early warning silent when stable", t_ew_silent)


# ═══════════════════════════════════════════════════════════════════════
# REGIME
# ═══════════════════════════════════════════════════════════════════════
print("\n[Regime Classification]")

def t_expansion():
    assert classify_regime(0.2, 0.5, CFG) == "Stable Expansion"
test("stable expansion", t_expansion)

def t_decline():
    assert classify_regime(-0.2, 2.0, CFG) == "Systemic Decline"
test("systemic decline", t_decline)

def t_plateau():
    assert classify_regime(0.0, 0.5, CFG) == "Stable Plateau"
test("stable plateau", t_plateau)

def t_volatile():
    assert classify_regime(0.2, 1.5, CFG) == "Volatile Growth"
test("volatile growth", t_volatile)

def t_transition():
    assert classify_regime(-0.03, 1.3, CFG) == "Transitional State"
test("transitional state", t_transition)


# ═══════════════════════════════════════════════════════════════════════
# FEATURE 4: REGIME PERSISTENCE
# ═══════════════════════════════════════════════════════════════════════
print("\n[Feature 4: Regime Persistence]")

def t_persist_positive():
    df = full_df()
    days = compute_regime_persistence(df, CFG)
    assert days >= 1, f"Persistence must be >= 1, got {days}"
test("persistence >= 1 for nonempty data", t_persist_positive)

def t_persist_constant_regime():
    # Build constant data where regime never changes → persistence = len(data)
    rows = [{"date": f"2026-01-{i+1:02d}", "mood": 7, "stress": 3, "deep_work_hours": 4,
             "tasks_completed": 5, "tasks_total": 7, "sleep_hours": 7.5, "recovery": 7}
            for i in range(10)]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = compute_domain_scores(df, CFG)
    df = compute_rolling_means(df, CFG)
    df = compute_volatility(df, CFG)
    df = compute_burnout_flags(df, CFG)
    days = compute_regime_persistence(df, CFG)
    assert days == 10, f"Constant data should give persistence=10, got {days}"
test("constant data → persistence = N", t_persist_constant_regime)

def t_persist_regime_change():
    # Build data with a clear regime transition: first 7 days stable, then 3 days volatile
    stable = [{"date": f"2026-01-{i+1:02d}", "mood": 7, "stress": 3, "deep_work_hours": 4,
               "tasks_completed": 5, "tasks_total": 7, "sleep_hours": 7.5, "recovery": 7}
              for i in range(7)]
    volatile = [{"date": f"2026-01-{i+8:02d}", "mood": 3, "stress": 9, "deep_work_hours": 1,
                 "tasks_completed": 1, "tasks_total": 8, "sleep_hours": 4, "recovery": 2}
                for i in range(3)]
    df = pd.DataFrame(stable + volatile)
    df["date"] = pd.to_datetime(df["date"])
    df = compute_domain_scores(df, CFG)
    df = compute_rolling_means(df, CFG)
    df = compute_volatility(df, CFG)
    df = compute_burnout_flags(df, CFG)
    days = compute_regime_persistence(df, CFG)
    assert days < len(df), f"After regime change, persistence should be < {len(df)}, got {days}"
test("persistence < N after regime change", t_persist_regime_change)


# ═══════════════════════════════════════════════════════════════════════
# INTEGRATION
# ═══════════════════════════════════════════════════════════════════════
print("\n[Integration]")

def t_full_keys():
    result = analyze(TEST_DATA)
    expected = {
        "global_balance", "trend", "domain_trends", "volatility",
        "burnout_risk", "deviation", "compensation_flags", "early_warning",
        "regime", "trend_confidence", "divergence_index", "regime_persistence_days",
    }
    actual = set(result.keys())
    missing = expected - actual
    extra = actual - expected
    assert not missing, f"Missing keys: {missing}"
    assert not extra, f"Unexpected keys: {extra}"
test("full analysis returns all v1.1 keys", t_full_keys)

def t_trend_structure():
    result = analyze(TEST_DATA)
    assert "short" in result["trend"] and "long" in result["trend"]
    assert "short" in result["domain_trends"] and "long" in result["domain_trends"]
test("trend has short/long structure", t_trend_structure)

def t_report():
    result = analyze(TEST_DATA)
    report = generate_report(result)
    assert isinstance(report, str)
    assert "ZENITH" in report
    assert "7d" in report and "30d" in report
    assert "Confidence" in report
    assert "Divergence" in report
    assert "streak" in report
    assert len(report) > 200
test("report contains all v1.1 sections", t_report)

def t_parity_balance():
    approx(analyze(TEST_DATA)["global_balance"], 7.944, 0.01)
test("parity: global_balance ≈ 7.944", t_parity_balance)

def t_parity_vol():
    approx(analyze(TEST_DATA)["volatility"], 1.562, 0.01)
test("parity: volatility ≈ 1.562", t_parity_vol)

def t_parity_short_slope():
    result = analyze(TEST_DATA)
    approx(result["trend"]["short"]["slope"], 0.2647, 0.01)
test("parity: 7d slope ≈ 0.2647", t_parity_short_slope)

def t_missing_file():
    try:
        analyze("nonexistent.json")
        raise RuntimeError("Should have raised FileNotFoundError")
    except FileNotFoundError:
        pass
test("missing file raises FileNotFoundError", t_missing_file)

def t_custom_cfg():
    result = analyze(TEST_DATA, cfg=ZenithConfig())
    assert result["global_balance"] > 0
    assert "trend_confidence" in result
test("custom config injection works", t_custom_cfg)


# ═══════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 58}")
print(f"  {passed} passed, {failed} failed")
print(f"{'=' * 58}")
sys.exit(1 if failed else 0)