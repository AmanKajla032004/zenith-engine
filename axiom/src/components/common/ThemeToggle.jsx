import React, { useState, useEffect } from "react";

function getInitialTheme() {
  const stored = localStorage.getItem("axiom-theme");
  return stored === "dark" ? "dark" : "light";
}

export default function ThemeToggle() {
  const [theme, setTheme] = useState(getInitialTheme);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("axiom-theme", theme);
  }, [theme]);

  function toggle() {
    setTheme((prev) => (prev === "light" ? "dark" : "light"));
  }

  return (
    <button
      onClick={toggle}
      aria-label="Toggle theme"
      style={{
        padding: "6px 14px",
        fontSize: "0.8rem",
        fontWeight: 500,
        borderRadius: "8px",
        border: "1px solid var(--border-card)",
        background: "var(--bg-card)",
        color: "var(--text-secondary)",
        cursor: "pointer",
        transition: "border-color 0.2s ease",
      }}
    >
      {theme === "light" ? "Dark" : "Light"}
    </button>
  );
}