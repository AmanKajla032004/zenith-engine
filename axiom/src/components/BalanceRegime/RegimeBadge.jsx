import React from "react";

const REGIME_STYLES = {
  stable:          { bg: "var(--regime-stable-bg)",     color: "var(--regime-stable-text)",     label: "Stable" },
  moderate_strain: { bg: "var(--regime-strain-bg)",     color: "var(--regime-strain-text)",     label: "Moderate Strain" },
  overloaded:      { bg: "var(--regime-overloaded-bg)", color: "var(--regime-overloaded-text)", label: "Overloaded" },
  declining:       { bg: "var(--regime-declining-bg)",  color: "var(--regime-declining-text)",  label: "Declining" },
};

const DEFAULT_STYLE = { bg: "var(--regime-unknown-bg)", color: "var(--regime-unknown-text)", label: "Unknown" };

export default function RegimeBadge({ regime }) {
  const { bg, color, label } = REGIME_STYLES[regime] || DEFAULT_STYLE;

  return (
    <span
      style={{
        display: "inline-block",
        padding: "4px 12px",
        borderRadius: "9999px",
        fontSize: "0.85rem",
        fontWeight: 600,
        background: bg,
        color,
      }}
    >
      {label}
    </span>
  );
}