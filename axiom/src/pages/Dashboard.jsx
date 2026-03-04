import { useState, useEffect } from "react";

const API = "http://127.0.0.1:8000/insights";

const trendColor = (label) => {
  if (!label) return "#a1a1aa";
  const l = label.toLowerCase();
  if (l.includes("improving") || l.includes("rising")) return "#4ade80";
  if (l.includes("declining") || l.includes("falling")) return "#f87171";
  return "#a1a1aa";
};

const humanizeFlag = (flag) =>
  flag
    .replace(/\bcompensating\s+([\w/]+)\s+decline\b/gi, "compensating for declining $1")
    .replace(/\bmasking\s+([\w/]+)\s+decline\b/gi, "masking declining $1")
    .replace(/\boffsetting\s+([\w/]+)\s+drop\b/gi, "offsetting drop in $1");

const generateInterpretation = (data) => {
  const parts = [];

  const b = data.global_balance;
  if (b >= 7) parts.push("Strong overall balance");
  else if (b >= 5) parts.push("Moderate balance");
  else parts.push("Low balance");

  const shortLabel = data.trend?.short?.label?.toLowerCase() || "";
  if (shortLabel.includes("declining")) parts.push("with recent downward movement");
  else if (shortLabel.includes("improving")) parts.push("with recent upward movement");

  if (data.compensation_flags?.length > 0)
    parts.push("Performance appears to be compensating for declines in other domains");

  let result = parts[0];
  if (parts[1]?.startsWith("with")) result += " " + parts[1];
  const rest = parts.slice(parts[1]?.startsWith("with") ? 2 : 1);
  if (rest.length) result += ". " + rest.join(". ");

  if (data.burnout_risk) result += ". Burnout risk indicators detected";

  return result + ".";
};

export default function Dashboard() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(API)
      .then((r) => {
        if (!r.ok) throw new Error(r.status);
        return r.json();
      })
      .then(setData)
      .catch(() => setData(null))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <p style={{ color: "#52525b", fontSize: "0.85rem" }}>Loading…</p>;
  }

  if (!data) {
    return (
      <div style={styles.card}>
        <h1 style={styles.value}>--</h1>
        <p style={styles.regime}>Awaiting Data</p>
      </div>
    );
  }

  const ds = data.domain_trends?.short || {};
  const hasFlags = data.compensation_flags?.length > 0;
  const domains = [
    { key: "recovery", name: "Recovery" },
    { key: "emotional", name: "Emotional" },
    { key: "performance", name: "Performance" },
    { key: "energy_focus", name: "Energy / Focus" },
  ];

  return (
    <div style={styles.wrap}>
      {/* ── State Card ── */}
      <div style={styles.card}>
        <p style={styles.label} title="Overall behavioral balance (0–10 scale). Higher values indicate stronger overall system stability.">Global Balance</p>
        <h1 style={styles.value}>{data.global_balance ?? "--"}</h1>
        <p style={styles.regime} title="Current system regime derived from recent behavioral patterns.">{data.regime ?? "Unknown"}</p>
        {data.regime_persistence_days != null && (
          <p style={styles.persistence} title="Number of consecutive days the system has remained in this regime.">
            {data.regime_persistence_days}d in regime
          </p>
        )}
        {data.burnout_risk && <p style={styles.burnout}>Burnout risk elevated</p>}
      </div>

      {/* ── System Interpretation ── */}
      <div style={styles.section}>
        <p style={styles.sectionTitle}>System Interpretation</p>
        <p style={styles.interpretation}>{generateInterpretation(data)}</p>
      </div>

      {/* ── Trend Overview ── */}
      <div style={styles.section}>
        <p style={styles.sectionTitle}>Trends</p>
        <div style={styles.grid}>
          <div style={styles.cell} title="Recent directional movement in overall balance.">
            <p style={styles.cellLabel}>Short</p>
            <p style={{ ...styles.cellValue, color: trendColor(data.trend?.short?.label) }}>
              {data.trend?.short?.label ?? "—"}
            </p>
          </div>
          <div style={styles.cell} title="Longer-term directional movement across the observation window.">
            <p style={styles.cellLabel}>Long</p>
            <p style={{ ...styles.cellValue, color: trendColor(data.trend?.long?.label) }}>
              {data.trend?.long?.label ?? "—"}
            </p>
          </div>
          <div style={styles.cell} title="Trend strength indicates statistical confidence in the detected direction. Higher values mean stronger directional consistency (0–1 scale).">
            <p style={styles.cellLabel}>Confidence</p>
            <p style={styles.cellValue}>
              {data.trend_confidence != null
                ? data.trend_confidence.toFixed(2)
                : "—"}
            </p>
          </div>
          <div style={styles.cell} title="Degree of fluctuation in overall balance. Higher values indicate greater instability.">
            <p style={styles.cellLabel}>Volatility</p>
            <p style={styles.cellValue}>
              {data.volatility != null ? data.volatility.toFixed(2) : "—"}
            </p>
          </div>
        </div>
      </div>

      {/* ── Domain Signals ── */}
      <div style={styles.section}>
        <p style={styles.sectionTitle}>Domain Signals</p>
        <div style={styles.grid}>
          {domains.map((d) => {
            const label = ds[d.key]?.label ?? "—";
            return (
              <div style={styles.cell} key={d.key} title="Recent directional movement within this domain.">
                <p style={styles.cellLabel}>{d.name}</p>
                <p style={{ ...styles.cellValue, color: trendColor(label) }}>
                  {label}
                </p>
              </div>
            );
          })}
        </div>
      </div>

      {/* ── System Signals ── */}
      {(hasFlags || data.early_warning) && (
        <div style={styles.section}>
          <p style={styles.sectionTitle} title="Cross-domain behavioral interactions detected by the system.">System Signals</p>
          <div style={styles.signals}>
            {hasFlags &&
              data.compensation_flags.map((flag, i) => (
                <p style={styles.signal} key={i}>· {humanizeFlag(flag)}</p>
              ))}
            {data.early_warning && (
              <p style={styles.signal}>· Early warning signal detected</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

const styles = {
  wrap: {
    width: "100%",
    maxWidth: "480px",
    padding: "0 1.5rem",
    display: "flex",
    flexDirection: "column",
    gap: "1.5rem",
  },
  card: {
    textAlign: "center",
    padding: "2.5rem 3rem",
    background: "#18181b",
    borderRadius: "12px",
    border: "1px solid #27272a",
    boxShadow: "0 4px 24px rgba(0,0,0,0.4)",
  },
  label: {
    fontSize: "0.7rem",
    textTransform: "uppercase",
    letterSpacing: "0.12em",
    color: "#71717a",
    margin: "0 0 0.5rem",
  },
  value: {
    fontSize: "3.5rem",
    fontWeight: 700,
    margin: "0 0 0.25rem",
    color: "#fafafa",
  },
  regime: {
    fontSize: "1rem",
    color: "#a1a1aa",
    margin: 0,
  },
  persistence: {
    fontSize: "0.7rem",
    color: "#52525b",
    margin: "0.4rem 0 0",
  },
  burnout: {
    fontSize: "0.7rem",
    color: "#f87171",
    margin: "0.6rem 0 0",
    letterSpacing: "0.04em",
  },
  section: {
    background: "#18181b",
    borderRadius: "12px",
    border: "1px solid #27272a",
    padding: "1.25rem 1.5rem",
  },
  sectionTitle: {
    fontSize: "0.7rem",
    textTransform: "uppercase",
    letterSpacing: "0.1em",
    color: "#52525b",
    margin: "0 0 1rem",
  },
  grid: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: "1rem",
  },
  cell: {
    display: "flex",
    flexDirection: "column",
    gap: "0.2rem",
  },
  cellLabel: {
    fontSize: "0.65rem",
    textTransform: "uppercase",
    letterSpacing: "0.08em",
    color: "#71717a",
    margin: 0,
  },
  cellValue: {
    fontSize: "0.9rem",
    fontWeight: 500,
    color: "#a1a1aa",
    margin: 0,
  },
  signals: {
    display: "flex",
    flexDirection: "column",
    gap: "0.5rem",
  },
  signal: {
    fontSize: "0.85rem",
    color: "#d4d4d8",
    margin: 0,
    lineHeight: 1.4,
    background: "#141416",
    borderLeft: "2px solid #3f3f46",
    paddingLeft: "10px",
  },
  interpretation: {
    fontSize: "0.85rem",
    color: "#d4d4d8",
    margin: 0,
    lineHeight: 1.5,
    background: "#141416",
    borderLeft: "2px solid #3f3f46",
    padding: "10px",
  },
};