import React from "react";

function getStatus(pressure) {
  if (pressure > 4) return { label: "Critical", color: "var(--danger)" };
  if (pressure >= 2) return { label: "Elevated", color: "var(--warning)" };
  return { label: "Normal", color: "var(--success)" };
}

const MAX_PRESSURE = 6;

export default function PressureIndicator({ pressure = 0, capacityMismatch = false }) {
  const { label, color } = getStatus(pressure);
  const pct = Math.min(Math.round((pressure / MAX_PRESSURE) * 100), 100);

  return (
    <div>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "baseline",
          marginBottom: "6px",
        }}
      >
        <span style={{ fontSize: "1.25rem", fontWeight: 600, color: "var(--text-primary)" }}>{pressure.toFixed(1)}</span>
        <span style={{ fontSize: "0.85rem", fontWeight: 500, color }}>{label}</span>
      </div>

      <div
        style={{
          height: "10px",
          borderRadius: "5px",
          background: "var(--bg-bar-track)",
          overflow: "hidden",
          marginBottom: "12px",
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

      {capacityMismatch && (
        <div
          style={{
            padding: "8px 12px",
            borderRadius: "6px",
            background: "var(--bg-warning)",
            color: "var(--danger-text)",
            fontSize: "0.85rem",
            fontWeight: 500,
          }}
        >
          Capacity mismatch -- workload exceeds behavioral capacity.
        </div>
      )}
    </div>
  );
}