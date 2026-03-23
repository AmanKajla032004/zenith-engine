import React from "react";
import {
  RadialBarChart,
  RadialBar,
  ResponsiveContainer,
} from "recharts";

function getColor(score) {
  if (score >= 7) return "var(--success)";
  if (score >= 4) return "var(--warning)";
  return "var(--danger)";
}

export default function BalanceScoreGauge({ score = 0 }) {
  const color = getColor(score);
  const data = [{ name: "Balance", value: score, fill: color }];

  return (
    <ResponsiveContainer width="100%" height={250}>
      <RadialBarChart
        innerRadius="70%"
        outerRadius="100%"
        startAngle={210}
        endAngle={-30}
        data={data}
      >
        <RadialBar
          dataKey="value"
          background
          cornerRadius={8}
          domain={[0, 10]}
        />
        <text
          x="50%"
          y="50%"
          textAnchor="middle"
          dominantBaseline="central"
          style={{ fontSize: "2rem", fontWeight: 700, fill: "var(--text-primary)" }}
        >
          {score.toFixed(2)}
        </text>
        <text
          x="50%"
          y="60%"
          textAnchor="middle"
          dominantBaseline="central"
          style={{ fontSize: "0.75rem", fill: "var(--text-secondary)" }}
        >
          / 10
        </text>
      </RadialBarChart>
    </ResponsiveContainer>
  );
}