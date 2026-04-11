const POLICY_LABEL = {
  astar: "A* Search",
  bfs: "Breadth-First Search",
  dfs: "Depth-First Search",
};

export default function PlannerSummaryCard({ planner }) {
  if (!planner) return null;

  const { policy, actions, projection } = planner;
  const balance = projection?.projected_balance ?? 0;

  const balanceColor =
    balance >= 0.7
      ? "text-emerald-400"
      : balance >= 0.4
        ? "text-amber-400"
        : "text-red-400";

  return (
    <div className="rounded-2xl border border-white/[0.06] bg-white/[0.03] backdrop-blur-sm p-6 space-y-5">
      {/* Header */}
      <div>
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-medium tracking-wide uppercase text-neutral-400">
            Stabilization Planner
          </h3>
          <span className="text-[11px] font-mono px-2.5 py-1 rounded-full bg-white/[0.05] text-neutral-500 border border-white/[0.06]">
            Policy: {POLICY_LABEL[policy] || policy}
          </span>
        </div>
      </div>

      {/* Metrics row */}
      <div className="grid grid-cols-2 gap-4">
        <div className="rounded-xl bg-white/[0.03] border border-white/[0.04] px-4 py-3">
          <p className="text-[11px] uppercase tracking-wider text-neutral-500 mb-1">
            Steps to Stabilization
          </p>
          <p className="text-2xl font-light text-neutral-200 tabular-nums">
            {actions.length}
          </p>
        </div>
        <div className="rounded-xl bg-white/[0.03] border border-white/[0.04] px-4 py-3">
          <p className="text-[11px] uppercase tracking-wider text-neutral-500 mb-1">
            Projected Balance Score
          </p>
          <p className={`text-2xl font-light tabular-nums ${balanceColor}`}>
            {balance.toFixed(2)}
          </p>
        </div>
      </div>
    </div>
  );
}