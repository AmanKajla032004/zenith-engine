import { useState } from "react";

const API = "http://127.0.0.1:8000/submit-entry";

const defaults = {
  date: "",
  tasks_completed: "",
  tasks_total: "",
  deep_work_hours: "",
  sleep_hours: "",
  mood: "",
  stress: "",
  recovery: "",
};

export default function Entry() {
  const [form, setForm] = useState({ ...defaults });
  const [status, setStatus] = useState(null);

  const set = (key) => (e) =>
    setForm((f) => ({ ...f, [key]: e.target.value }));

  const taskError =
    form.tasks_completed !== "" &&
    form.tasks_total !== "" &&
    parseInt(form.tasks_completed, 10) > parseInt(form.tasks_total, 10);

  const submit = async (e) => {
    e.preventDefault();
    if (taskError) return;
    setStatus(null);

    const body = {
      date: form.date,
      tasks_completed: parseInt(form.tasks_completed, 10),
      tasks_total: parseInt(form.tasks_total, 10),
      deep_work_hours: parseFloat(form.deep_work_hours),
      sleep_hours: parseFloat(form.sleep_hours),
      mood: parseInt(form.mood, 10),
      stress: parseInt(form.stress, 10),
      recovery: parseInt(form.recovery, 10),
    };

    try {
      const res = await fetch(API, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error(res.status);
      setForm({ ...defaults });
      setStatus("success");
    } catch {
      setStatus("error");
    }
  };

  const field = (label, key, type = "number", extra = {}) => (
    <div style={styles.field} key={key}>
      <label style={styles.label}>{label}</label>
      <input
        style={{
          ...styles.input,
          ...(taskError && (key === "tasks_completed" || key === "tasks_total")
            ? styles.inputError
            : {}),
        }}
        type={type}
        value={form[key]}
        onChange={set(key)}
        required
        {...extra}
      />
    </div>
  );

  return (
    <div style={styles.wrap}>
      <h1 style={styles.title}>New Daily Entry</h1>

      {status === "success" && (
        <p style={styles.success}>Entry saved.</p>
      )}
      {status === "error" && (
        <p style={styles.error}>Submission failed.</p>
      )}

      <form onSubmit={submit} style={styles.form}>
        {field("Date", "date", "date")}

        <div style={styles.row}>
          {field("Tasks Done", "tasks_completed", "number", { min: 0 })}
          {field("Tasks Total", "tasks_total", "number", { min: 0 })}
        </div>
        {taskError && (
          <p style={styles.fieldError}>Completed cannot exceed total.</p>
        )}

        <div style={styles.row}>
          {field("Deep Work (hrs)", "deep_work_hours", "number", { min: 0, step: 0.5 })}
          {field("Sleep (hrs)", "sleep_hours", "number", { min: 0, step: 0.5 })}
        </div>

        <div style={styles.row}>
          {field("Mood (1-10)", "mood", "number", { min: 1, max: 10 })}
          {field("Stress (1-10)", "stress", "number", { min: 1, max: 10 })}
        </div>

        {field("Recovery (1-10)", "recovery", "number", { min: 1, max: 10 })}

        <button type="submit" style={styles.btn}>Submit</button>
      </form>
    </div>
  );
}

const styles = {
  wrap: {
    width: "100%",
    maxWidth: "400px",
    padding: "0 1.5rem",
  },
  title: {
    fontSize: "1.5rem",
    fontWeight: 600,
    color: "var(--text-primary)",
    margin: "0 0 1.5rem",
    textAlign: "center",
  },
  form: {
    display: "flex",
    flexDirection: "column",
    gap: "1rem",
  },
  row: {
    display: "flex",
    gap: "1rem",
  },
  field: {
    display: "flex",
    flexDirection: "column",
    gap: "0.3rem",
    flex: 1,
  },
  label: {
    fontSize: "0.7rem",
    textTransform: "uppercase",
    letterSpacing: "0.1em",
    color: "var(--text-secondary)",
  },
  input: {
    background: "var(--bg-card)",
    border: "1px solid var(--border-card)",
    borderRadius: "8px",
    padding: "0.6rem 0.75rem",
    fontSize: "0.9rem",
    color: "var(--text-primary)",
    outline: "none",
    fontFamily: "inherit",
    width: "100%",
    boxSizing: "border-box",
    transition: "border-color 0.2s",
  },
  inputError: {
    borderColor: "var(--danger)",
  },
  btn: {
    marginTop: "0.5rem",
    padding: "0.7rem",
    background: "var(--accent-blue)",
    color: "var(--text-inverse)",
    border: "none",
    borderRadius: "8px",
    fontSize: "0.85rem",
    fontWeight: 600,
    cursor: "pointer",
    fontFamily: "inherit",
    letterSpacing: "0.02em",
  },
  success: {
    textAlign: "center",
    fontSize: "0.85rem",
    color: "var(--success-text)",
    margin: "0 0 1rem",
  },
  error: {
    textAlign: "center",
    fontSize: "0.85rem",
    color: "var(--danger-muted)",
    margin: "0 0 1rem",
  },
  fieldError: {
    fontSize: "0.75rem",
    color: "var(--danger)",
    margin: "-0.5rem 0 0",
  },
};