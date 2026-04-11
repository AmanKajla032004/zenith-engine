import React from "react";
import { useInsights } from "../hooks/useInsights";
import StateRadarChart from "../components/StateVector/StateRadarChart";
import BalanceScoreGauge from "../components/BalanceRegime/BalanceScoreGauge";
import RegimeBadge from "../components/BalanceRegime/RegimeBadge";
import RiskPanel from "../components/RiskIndicators/RiskPanel";
import PressureIndicator from "../components/WorkloadPressure/PressureIndicator";
import BehavioralTrendChart from "../components/Trends/BehavioralTrendChart";
import EntryForm from "../components/EntryForm/EntryForm";
import MetricCard from "../components/common/MetricCard";
import ThemeToggle from "../components/common/ThemeToggle";
import PlannerSummaryCard from "../components/zenith-v3/PlannerSummaryCard";
import TrajectoryGraph from "../components/zenith-v3/TrajectoryGraph";
import ActionTimeline from "../components/zenith-v3/ActionTimeline";
import ProjectionMetrics from "../components/zenith-v3/ProjectionMetrics";

const REGIME_COLOR = {
  stable:          "var(--success)",
  moderate_strain: "var(--warning)",
  overloaded:      "var(--danger)",
  declining:       "#f97316",
};

const REGIME_BORDER = {
  stable:          "#22c55e",
  moderate_strain: "#f59e0b",
  overloaded:      "#ef4444",
  declining:       "#f97316",
};

const REGIME_SHADOW = {
  stable:          "0 4px 16px rgba(34, 197, 94, 0.12)",
  moderate_strain: "0 4px 16px rgba(245, 158, 11, 0.12)",
  overloaded:      "0 4px 16px rgba(239, 68, 68, 0.12)",
  declining:       "0 4px 16px rgba(249, 115, 22, 0.12)",
};

const REGIME_LABEL = {
  stable: "stable",
  moderate_strain: "moderate strain",
  overloaded: "overloaded",
  declining: "declining",
};

function getPressureLabel(pressure) {
  if (pressure > 4) return "critical";
  if (pressure >= 2) return "elevated";
  return "normal";
}

export default function Dashboard() {
  const { insights, loading, error, refreshInsights, insightsHistory } = useInsights();

  if (loading) {
    return <div className="dashboard-status">Loading insights...</div>;
  }

  if (error) {
    return <div className="dashboard-status dashboard-error">Error: {error}</div>;
  }

  if (!insights) {
    return <div className="dashboard-status">No insight data available.</div>;
  }

  const interp = insights.interpretation;
  const regime = insights.regime;
  const regimeColor = REGIME_COLOR[regime] || "var(--text-secondary)";
  const pressureLabel = getPressureLabel(insights.behavioral_pressure_index);

  return (
    <div className="dashboard-container" style={{ gap: "16px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h1 className="dashboard-title">Axiom — Behavioral Intelligence</h1>
        <ThemeToggle />
      </div>

      {/* 0. System Status Header */}
      <div className="card" style={{
        padding: "14px 20px",
        borderLeft: `3px solid ${REGIME_BORDER[regime] || "var(--border-card)"}`,
      }}>
        <p style={{
          fontSize: "0.88rem",
          color: "var(--text-primary)",
          margin: 0,
          lineHeight: 1.5,
        }}>
          System is in{" "}
          <span style={{ fontWeight: 600, color: regimeColor }}>
            {REGIME_LABEL[regime] || regime}
          </span>
          {" "}driven by{" "}
          <span style={{ fontWeight: 600, color: "var(--accent-blue)" }}>
            {insights.dominant_driver}
          </span>
          {" "}with{" "}
          <span style={{ fontWeight: 600 }}>
            {pressureLabel}
          </span>
          {" "}pressure.
        </p>
      </div>

      <section className="card">
        <h2 className="section-title">Submit Daily Entry</h2>
        <EntryForm onEntrySubmitted={refreshInsights} />
      </section>

      {/* 1. Metric Cards */}
      <div className="dashboard-grid metrics-row">
        <div className="card metric-card-equal" style={{
          borderColor: REGIME_BORDER[regime] || "var(--border-card)",
          borderWidth: "1.5px",
          boxShadow: REGIME_SHADOW[regime] || "var(--shadow-soft)",
        }}>
          <p style={{
            fontSize: "0.7rem",
            fontWeight: 600,
            textTransform: "uppercase",
            letterSpacing: "0.06em",
            color: "var(--text-secondary)",
            margin: 0,
          }}>Balance Score</p>
          <p style={{
            fontSize: "2.8rem",
            fontWeight: 700,
            letterSpacing: "-0.02em",
            color: "var(--text-primary)",
            margin: "6px 0 2px",
            lineHeight: 1.1,
          }}>{insights.balance_score?.toFixed(2)}</p>
          <p style={{
            fontSize: "0.8rem",
            color: "var(--text-secondary)",
            margin: 0,
          }}>{insights.regime}</p>
        </div>
        <MetricCard label="Pressure Index" value={insights.behavioral_pressure_index?.toFixed(2)} />
        <MetricCard label="Burnout Risk" value={`${(insights.burnout_probability * 100).toFixed(0)}%`} />
        <MetricCard label="Fatigue Risk" value={`${(insights.fatigue_probability * 100).toFixed(0)}%`} />
      </div>

      {/* 2. System Insight Panel */}
      <section className="card" style={{
        padding: "30px 32px",
        background: "var(--bg-insight)",
        borderLeft: `3px solid ${REGIME_BORDER[regime] || "var(--accent-blue)"}`,
      }}>
        <h2 className="section-title">System Insight</h2>
        <p style={{
          fontSize: "1.1rem",
          color: regimeColor,
          lineHeight: 1.6,
          margin: "0 0 16px",
          fontWeight: 500,
          opacity: 0.85,
        }}>
          {interp.summary}
        </p>
        <div style={{
          padding: "16px",
          background: "var(--bg-primary)",
          borderRadius: "10px",
          border: "1px solid var(--border-card)",
        }}>
          <p style={{
            fontSize: "0.78rem",
            fontWeight: 600,
            textTransform: "uppercase",
            letterSpacing: "0.08em",
            color: "var(--accent-blue)",
            margin: "0 0 8px",
          }}>
            {interp.title}
          </p>
          <p style={{
            fontSize: "0.9rem",
            color: "var(--text-secondary)",
            lineHeight: 1.7,
            margin: "0 0 12px",
          }}>
            {interp.explanation}
          </p>
          <p style={{
            fontSize: "0.85rem",
            color: "var(--text-primary)",
            lineHeight: 1.6,
            margin: 0,
            padding: "10px 14px",
            background: "var(--accent-soft)",
            borderRadius: "8px",
            border: "1px solid var(--accent-border)",
          }}>
            {interp.impact}
          </p>
        </div>
      </section>

      {/* 3. Risk + Pressure + State + Balance */}
      <div className="dashboard-grid">
        <section className="card">
          <h2 className="section-title">Risk Indicators</h2>
          <RiskPanel
            burnout={insights.burnout_probability}
            fatigue={insights.fatigue_probability}
            recoveryFailure={insights.recovery_failure_probability}
          />
        </section>

        <section className="card">
          <h2 className="section-title">Workload Pressure</h2>
          <PressureIndicator
            pressure={insights.behavioral_pressure_index}
            capacityMismatch={insights.capacity_mismatch}
          />
        </section>

        <section className="card">
          <h2 className="section-title">State Vector</h2>
          <StateRadarChart state={insights.state} />
        </section>

        <section className="card">
          <h2 className="section-title">Balance &amp; Regime</h2>
          <BalanceScoreGauge score={insights.balance_score} />
          <RegimeBadge regime={insights.regime} />
        </section>
      </div>

      {/* 4. Behavioral Trends */}
      <section className="card">
        <h2 className="section-title">Behavioral Trends</h2>
        <BehavioralTrendChart data={insightsHistory} />
      </section>

      {/* 5. Zenith v3 — Stabilization Planner */}
      {insights.zenith_v3 && (
        <>
          <section className="card" style={{ borderLeft: "3px solid var(--accent-blue)" }}>
            <h2 className="section-title">Zenith v3 — Stabilization Plan</h2>
            <PlannerSummaryCard planner={insights.zenith_v3} />
          </section>

          <section className="card">
            <ActionTimeline planner={insights.zenith_v3} />
          </section>

          <section className="card">
            <ProjectionMetrics
              planner={insights.zenith_v3}
              insights={insights}
            />
          </section>

          <section className="card">
            <TrajectoryGraph planner={insights.zenith_v3} />
          </section>
        </>
      )}
    </div>
  );
}