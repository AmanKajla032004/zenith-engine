import React, { useMemo, useState, useEffect } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

const PRIMARY_LINE = {
  key: "scaled_balance",
  name: "Balance",
  strokeWidth: 3,
  opacity: 1,
};

const SECONDARY_LINES = [
  { key: "performance", color: "#5b9bd5", name: "Performance" },
  { key: "recovery",    color: "#2da44e", name: "Recovery" },
  { key: "energy",      color: "#d4a017", name: "Energy" },
  { key: "emotional",   color: "#d94848", name: "Emotional" },
];

function normalizeData(data) {
  return data.map((entry) => ({
    ...entry,
    scaled_balance: entry.balance_score != null
      ? (entry.balance_score / 5) - 1
      : 0,
  }));
}

const DATA_KEYS = [PRIMARY_LINE.key, ...SECONDARY_LINES.map((l) => l.key)];
const PADDING = 0.2;

function computeDomain(data) {
  let min = Infinity;
  let max = -Infinity;
  for (const entry of data) {
    for (const key of DATA_KEYS) {
      const v = entry[key];
      if (v != null) {
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }
  }
  if (!isFinite(min)) return [-1, 1];
  return [min - PADDING, max + PADDING];
}

function readThemeColors() {
  const s = getComputedStyle(document.documentElement);
  return {
    grid: s.getPropertyValue("--chart-grid").trim() || "rgba(0,0,0,0.06)",
    axis: s.getPropertyValue("--chart-axis").trim() || "rgba(0,0,0,0.10)",
    text: s.getPropertyValue("--text-secondary").trim() || "#6b7280",
    primary: s.getPropertyValue("--text-primary").trim() || "#1f2937",
    card: s.getPropertyValue("--bg-card").trim() || "#ffffff",
    border: s.getPropertyValue("--border-card").trim() || "#e5e9f2",
    accent: s.getPropertyValue("--accent-blue").trim() || "#4f6ef7",
    shadow: s.getPropertyValue("--shadow-soft").trim() || "0 4px 16px rgba(0,0,0,0.05)",
  };
}

function useThemeColors() {
  const [colors, setColors] = useState(readThemeColors);

  useEffect(() => {
    const observer = new MutationObserver(() => {
      setColors(readThemeColors());
    });
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });
    return () => observer.disconnect();
  }, []);

  return colors;
}

export default function BehavioralTrendChart({ data = [] }) {
  const normalized = useMemo(() => normalizeData(data), [data]);
  const domain = useMemo(() => computeDomain(normalized), [normalized]);
  const c = useThemeColors();

  if (normalized.length < 2) {
    return (
      <p style={{
        color: "var(--text-secondary)",
        fontSize: "0.88rem",
        fontStyle: "italic",
        opacity: 0.7,
        padding: "20px 0",
      }}>
        Add more daily entries to observe behavioral trajectory.
      </p>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={normalized}>
        <CartesianGrid stroke={c.grid} strokeDasharray="3 3" />
        <XAxis
          dataKey="date"
          tick={{ fontSize: 11, fill: c.text }}
          axisLine={{ stroke: c.axis }}
          tickLine={false}
        />
        <YAxis
          domain={domain}
          tickFormatter={(v) => v.toFixed(2)}
          tick={{ fontSize: 11, fill: c.text }}
          axisLine={{ stroke: c.axis }}
          tickLine={false}
        />
        <Tooltip
          formatter={(value) => value.toFixed(2)}
          contentStyle={{
            background: c.card,
            border: `1px solid ${c.border}`,
            borderRadius: "8px",
            color: c.primary,
            fontSize: "0.8rem",
            boxShadow: c.shadow,
          }}
        />
        <Legend
          wrapperStyle={{ fontSize: "0.75rem", color: c.text }}
        />
        <Line
          type="monotone"
          dataKey={PRIMARY_LINE.key}
          stroke={c.accent}
          name={PRIMARY_LINE.name}
          strokeWidth={PRIMARY_LINE.strokeWidth}
          strokeOpacity={PRIMARY_LINE.opacity}
          dot={{ r: 3, fill: c.accent }}
          activeDot={{ r: 5 }}
          isAnimationActive={true}
          animationDuration={400}
        />
        {SECONDARY_LINES.map((line) => (
          <Line
            key={line.key}
            type="monotone"
            dataKey={line.key}
            stroke={line.color}
            name={line.name}
            strokeWidth={1.2}
            strokeOpacity={0.45}
            dot={false}
            isAnimationActive={true}
            animationDuration={400}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}