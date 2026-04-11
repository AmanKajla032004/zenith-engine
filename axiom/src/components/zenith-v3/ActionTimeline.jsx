export default function ActionTimeline({ planner }) {
  if (!planner?.actions) return null;

  const { actions } = planner;

  return (
    <div className="rounded-2xl border border-white/[0.06] bg-white/[0.03] backdrop-blur-sm p-6 space-y-4">
      <div>
        <h3 className="text-sm font-medium tracking-wide uppercase text-neutral-400">
          Action Sequence
        </h3>
        <p className="text-xs text-neutral-500 mt-2 leading-relaxed">
          Suggested order of behavioral adjustments
        </p>
      </div>

      {actions.length === 0 && (
        <p className="text-sm text-neutral-500 italic">
          No stabilization actions required. System is already balanced.
        </p>
      )}

      {actions.length > 0 && (
        <div>
        <ol className="space-y-1 list-decimal list-inside">
          {actions.map((action, i) => (
            <li
              key={i}
              className="relative py-2.5 group text-sm text-neutral-300 tracking-wide group-hover:text-neutral-100 transition-colors"
            >
              {action
                .replace(/[0-9]/g, "")
                .replace(/_/g, " ")
                .replace(/\b\w/g, l => l.toUpperCase())
                .trim()}
            </li>
          ))}
        </ol>
      </div>
      )}
    </div>
  );
}