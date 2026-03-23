import React from "react";

const styles = {
  label: {
    fontSize: "0.7rem",
    fontWeight: 600,
    textTransform: "uppercase",
    letterSpacing: "0.06em",
    color: "var(--text-secondary)",
    margin: 0,
  },
  value: {
    fontSize: "2.2rem",
    fontWeight: 700,
    lineHeight: 1.2,
    color: "var(--text-primary)",
    margin: "6px 0 2px",
  },
  subtitle: {
    fontSize: "0.8rem",
    color: "var(--text-secondary)",
    margin: 0,
  },
};

export default function MetricCard({ label, value, subtitle }) {
  return (
    <div className="card">
      <p className="metric-label" style={styles.label}>{label}</p>
      <p className="metric-value" style={styles.value}>{value}</p>
      {subtitle && (
        <p className="metric-subtitle" style={styles.subtitle}>{subtitle}</p>
      )}
    </div>
  );
}