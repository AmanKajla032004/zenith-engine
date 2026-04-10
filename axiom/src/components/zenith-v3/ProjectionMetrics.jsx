export default function ProjectionMetrics({ planner, insights }) {
  if (!planner?.projection) return null;

  const currentBalance = insights?.balance_score ?? 0;
  const projectedBalance = planner.projection.projected_balance ?? 0;
  const improvement = projectedBalance - currentBalance;
  const steps = planner.actions?.length ?? 0;

  const improvementColor = improvement >= 0 ? "text-emerald-400" : "text-red-400";
  const improvementPrefix = improvement >= 0 ? "+" : "";

  const projectedColor =
    projectedBalance >= 0.7
      ? "text-emerald-400"
      : projectedBalance >= 0.4
        ? "text-amber-400"
        : "text-red-400";

  const metrics = [
    { label: "Current Balance", value: currentBalance.toFixed(2), color: "text-neutral-200" },
    { label: "Projected Balance", value: projectedBalance.toFixed(2), color: projectedColor },
    { label: "Improvement", value: `${improvementPrefix}${improvement.toFixed(2)}`, color: improvementColor },
    { label: "Steps to Stability", value: steps, color: "text-neutral-200" },
  ];

  return (
    <div className="rounded-2xl border border-white/[0.06] bg-white/[0.03] backdrop-blur-sm p-6 space-y-4">
      <h3 className="text-sm font-medium tracking-wide uppercase text-neutral-400">
        Projection Metrics
      </h3>

      <div className="grid grid-cols-2 gap-3">
        {metrics.map((m) => (
          <div
            key={m.label}
            className="rounded-xl bg-white/[0.03] border border-white/[0.04] px-4 py-3"
          >
            <p className="text-[11px] uppercase tracking-wider text-neutral-500 mb-1">
              {m.label}
            </p>
            <p className={`text-2xl font-light tabular-nums ${m.color}`}>
              {m.value}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}