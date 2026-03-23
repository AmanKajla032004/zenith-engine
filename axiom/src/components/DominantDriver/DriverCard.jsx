import React from "react";

const DRIVER_MAP = {
  performance: {
    label: "Performance",
    description: "Performance demand is dominating the system. When output pressure exceeds recovery and emotional capacity, the system enters strain and burnout risk increases.",
  },
  recovery: {
    label: "Recovery",
    description: "Recovery capacity is below system demand. Sustained low recovery reduces stability and gradually increases fatigue risk.",
  },
  energy: {
    label: "Energy",
    description: "Energy availability is insufficient relative to workload demands. This often appears when cognitive effort exceeds rest cycles.",
  },
  emotional: {
    label: "Emotional",
    description: "Emotional strain is influencing system stability. Emotional pressure often amplifies fatigue and reduces cognitive recovery.",
  },
};

const DEFAULT_DRIVER = {
  label: "Unknown",
  description: "Driver data unavailable.",
};

const styles = {
  card: {
    background: "var(--bg-driver)",
    border: "1px solid var(--border-driver)",
    borderRadius: "12px",
    padding: "18px",
  },
  title: {
    fontSize: "1.1rem",
    fontWeight: 600,
    color: "var(--text-primary)",
    margin: "0 0 8px",
  },
  description: {
    fontSize: "0.9rem",
    color: "var(--text-secondary)",
    lineHeight: 1.6,
    margin: 0,
  },
};

export default function DriverCard({ driver }) {
  const { label, description } = DRIVER_MAP[driver] || DEFAULT_DRIVER;

  return (
    <div className="driver-card" style={styles.card}>
      <h3 className="driver-title" style={styles.title}>{label}</h3>
      <p className="driver-description" style={styles.description}>{description}</p>
    </div>
  );
}