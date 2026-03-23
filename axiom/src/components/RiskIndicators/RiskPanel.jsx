import React from "react";

function getColor(value) {
  if (value >= 0.7) return "var(--danger)";
  if (value >= 0.4) return "var(--warning)";
  return "var(--success)";
}

function RiskBar({ label, value = 0 }) {
  const pct = Math.round(value * 100);
  const color = getColor(value);

  return (
    <div style={{ marginBottom: "14px" }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          marginBottom: "4px",
          fontSize: "0.85rem",
        }}
      >
        <span style={{ fontWeight: 500, color: "var(--text-primary)" }}>{label}</span>
        <span style={{ color }}>{pct}%</span>
      </div>
      <div
        style={{
          height: "10px",
          borderRadius: "5px",
          background: "var(--bg-bar-track)",
          overflow: "hidden",
        }}
      >
        <div
          style={{
            width: `${pct}%`,
            height: "100%",
            borderRadius: "5px",
            background: color,
            transition: "width 0.3s ease",
          }}
        />
      </div>
    </div>
  );
}

export default function RiskPanel({ burnout = 0, fatigue = 0, recoveryFailure = 0 }) {
  return (
    <div>
      <RiskBar label="Burnout Risk" value={burnout} />
      <RiskBar label="Fatigue Risk" value={fatigue} />
      <RiskBar label="Recovery Failure" value={recoveryFailure} />
    </div>
  );
}