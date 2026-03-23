import React, { useState, useEffect } from "react";
import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

const AXES = ["Performance", "Recovery", "Energy", "Emotional"];
const KEYS = ["performance", "recovery", "energy", "emotional"];
const PADDING = 0.2;

function toChartData(state) {
  return AXES.map((axis, i) => ({
    axis,
    value: state?.[KEYS[i]] ?? 0,
  }));
}

function getRadiusLimit(state) {
  const baseline = 1;
  const maxMagnitude = KEYS.reduce(
    (max, key) => Math.max(max, Math.abs(state?.[key] ?? 0)),
    0
  );
  return Math.max(baseline, maxMagnitude) + PADDING;
}

function readThemeColors() {
  const s = getComputedStyle(document.documentElement);
  return {
    grid: s.getPropertyValue("--chart-grid").trim() || "rgba(0,0,0,0.06)",
    text: s.getPropertyValue("--text-secondary").trim() || "#6b7280",
    primary: s.getPropertyValue("--text-primary").trim() || "#1f2937",
    card: s.getPropertyValue("--bg-card").trim() || "#ffffff",
    border: s.getPropertyValue("--border-card").trim() || "#e5e9f2",
    accent: s.getPropertyValue("--chart-accent").trim() || "#4f6ef7",
    fill: s.getPropertyValue("--chart-fill").trim() || "rgba(79,110,247,0.25)",
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

export default function StateRadarChart({ state = {} }) {
  const data = toChartData(state);
  const radiusLimit = getRadiusLimit(state);
  const c = useThemeColors();

  return (
    <ResponsiveContainer width="100%" height={300}>
      <RadarChart data={data}>
        <PolarGrid stroke={c.grid} />
        <PolarAngleAxis
          dataKey="axis"
          tick={{ fontSize: 12, fill: c.text }}
        />
        <PolarRadiusAxis
          domain={[-radiusLimit, radiusLimit]}
          tickCount={5}
          tick={{ fontSize: 10, fill: c.text }}
          axisLine={false}
        />
        <Radar
          name="State Vector"
          dataKey="value"
          stroke={c.accent}
          fill={c.fill}
          strokeWidth={2}
          dot
          isAnimationActive={true}
          animationDuration={400}
          animationEasing="ease-out"
        />
        <Tooltip
          contentStyle={{
            background: c.card,
            border: `1px solid ${c.border}`,
            borderRadius: "8px",
            color: c.primary,
            fontSize: "0.85rem",
            boxShadow: c.shadow,
          }}
        />
      </RadarChart>
    </ResponsiveContainer>
  );
} 