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
        <div className="relative pl-4">
        {/* Vertical connector line */}
        {actions.length > 1 && (
          <div
            className="absolute left-[1.125rem] top-3 w-px bg-gradient-to-b from-white/[0.1] via-white/[0.06] to-transparent"
            style={{ height: `calc(100% - 1.5rem)` }}
          />
        )}

        <ol className="space-y-1 list-none">
          {actions.map((action, i) => (
            <li
              key={i}
              className="relative flex items-center gap-4 py-2.5 group"
            >
              {/* Circular number indicator */}
              <span className="relative z-10 shrink-0 w-6 h-6 rounded-full bg-neutral-900 border border-white/[0.1] flex items-center justify-center text-[11px] font-mono text-neutral-500 group-hover:border-white/[0.2] group-hover:text-neutral-300 transition-colors">
                {i + 1}
              </span>

              {/* Action text */}
              <span className="text-sm text-neutral-300 tracking-wide group-hover:text-neutral-100 transition-colors">
                {action
                  .replace(/[0-9]/g, "")
                  .replace(/_/g, " ")
                  .replace(/\b\w/g, l => l.toUpperCase())
                  .trim()}
                  
              </span>
            </li>
          ))}
        </ol>
      </div>
      )}
    </div>
  );
}