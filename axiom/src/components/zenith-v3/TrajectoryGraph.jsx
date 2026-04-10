import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from "recharts";

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg bg-neutral-900/90 border border-white/[0.08] backdrop-blur-sm px-3 py-2 shadow-xl">
      <p className="text-[11px] text-neutral-500 mb-0.5">Step {label}</p>
      <p className="text-sm font-mono text-neutral-200 tabular-nums">
        {payload[0].value.toFixed(3)}
      </p>
    </div>
  );
}

export default function TrajectoryGraph({ planner }) {
  if (!planner?.trajectory?.length) return null;

  const { trajectory } = planner;

  const minBalance = Math.min(...trajectory.map((d) => d.balance));
  const maxBalance = Math.max(...trajectory.map((d) => d.balance));
  const padding = (maxBalance - minBalance) * 0.15 || 0.1;
  const yMin = Math.floor((minBalance - padding) * 100) / 100;
  const yMax = Math.ceil((maxBalance + padding) * 100) / 100;

  const finalBalance = trajectory[trajectory.length - 1]?.balance ?? 0;
  const strokeColor =
    finalBalance >= 0.7
      ? "#34d399"
      : finalBalance >= 0.4
        ? "#fbbf24"
        : "#f87171";

  return (
    <div className="rounded-2xl border border-white/[0.06] bg-white/[0.03] backdrop-blur-sm p-6 space-y-4">
      <h3 className="text-sm font-medium tracking-wide uppercase text-neutral-400">
        Trajectory Projection
      </h3>

      <ResponsiveContainer width="100%" height={220}>
        <LineChart
          data={trajectory}
          margin={{ top: 4, right: 8, bottom: 0, left: -12 }}
        >
          <CartesianGrid
            stroke="rgba(255,255,255,0.04)"
            strokeDasharray="3 3"
            vertical={false}
          />
          <XAxis
            dataKey="step"
            tick={{ fontSize: 11, fill: "#737373" }}
            axisLine={{ stroke: "rgba(255,255,255,0.06)" }}
            tickLine={false}
          />
          <YAxis
            domain={[yMin, yMax]}
            tick={{ fontSize: 11, fill: "#737373" }}
            axisLine={false}
            tickLine={false}
            tickFormatter={(v) => v.toFixed(1)}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ stroke: "rgba(255,255,255,0.08)" }} />
          <Line
            type="monotone"
            dataKey="balance"
            stroke={strokeColor}
            strokeWidth={2}
            dot={false}
            activeDot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}