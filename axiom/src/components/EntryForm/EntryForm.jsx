import React, { useState } from "react";
import { submitEntry } from "../../api/zenithService";

const FIELDS = [
  { key: "tasks_completed", label: "Tasks Completed", step: 1 },
  { key: "tasks_total",     label: "Tasks Total",     step: 1 },
  { key: "deep_work_hours", label: "Deep Work Hours",  step: 0.5 },
  { key: "sleep_hours",     label: "Sleep Hours",      step: 0.5 },
  { key: "mood",            label: "Mood (1-10)",      step: 1 },
  { key: "stress",          label: "Stress (1-10)",    step: 1 },
  { key: "recovery",        label: "Recovery (1-10)",  step: 1 },
];

const INITIAL_VALUES = FIELDS.reduce((acc, f) => {
  acc[f.key] = "";
  return acc;
}, {});

export default function EntryForm({ onEntrySubmitted }) {
  const [values, setValues] = useState(INITIAL_VALUES);
  const [status, setStatus] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  function handleChange(key, raw) {
    setValues((prev) => ({ ...prev, [key]: raw }));
  }

  async function handleSubmit(e) {
    e.preventDefault();
    setStatus(null);
    setSubmitting(true);

    const entry = {
      date: new Date().toISOString().split("T")[0],
    };

    for (const field of FIELDS) {
      const num = Number(values[field.key]);
      if (isNaN(num) || values[field.key] === "") {
        setStatus({ type: "error", message: `Invalid value for ${field.label}.` });
        setSubmitting(false);
        return;
      }
      entry[field.key] = num;
    }

    try {
      await submitEntry(entry);
      setStatus({ type: "success", message: "Entry submitted." });
      setValues(INITIAL_VALUES);
      if (onEntrySubmitted) {
        onEntrySubmitted();
      }
    } catch (err) {
      setStatus({ type: "error", message: err.message });
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} style={{ maxWidth: "400px" }}>
      {FIELDS.map((field) => (
        <div key={field.key} style={{ marginBottom: "12px" }}>
          <label
            htmlFor={field.key}
            style={{
              display: "block",
              marginBottom: "4px",
              fontSize: "0.85rem",
              fontWeight: 500,
              color: "var(--text-primary)",
            }}
          >
            {field.label}
          </label>
          <input
            id={field.key}
            type="number"
            step={field.step}
            value={values[field.key]}
            onChange={(e) => handleChange(field.key, e.target.value)}
            style={{
              width: "100%",
              boxSizing: "border-box",
            }}
          />
        </div>
      ))}

      <button
        type="submit"
        disabled={submitting}
        style={{
          padding: "8px 20px",
          fontSize: "0.9rem",
          fontWeight: 600,
          borderRadius: "6px",
          border: "none",
          background: submitting ? "var(--accent-disabled)" : "var(--accent-blue)",
          color: "var(--text-inverse)",
          cursor: submitting ? "default" : "pointer",
        }}
      >
        {submitting ? "Submitting..." : "Submit Entry"}
      </button>

      {status && (
        <p
          style={{
            marginTop: "10px",
            fontSize: "0.85rem",
            color: status.type === "error" ? "var(--danger-muted)" : "var(--success-text)",
          }}
        >
          {status.message}
        </p>
      )}
    </form>
  );
}