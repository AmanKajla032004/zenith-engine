import { BrowserRouter, Routes, Route, NavLink } from "react-router-dom";
import Dashboard from "./pages/Dashboard";
import Entry from "./pages/Entry";

export default function App() {
  return (
    <BrowserRouter>
      <div style={styles.shell}>
        <nav style={styles.nav}>
          <NavLink to="/" style={({ isActive }) => link(isActive)}>
            Dashboard
          </NavLink>
          <NavLink to="/entry" style={({ isActive }) => link(isActive)}>
            New Entry
          </NavLink>
        </nav>
        <main style={styles.main}>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/entry" element={<Entry />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

const link = (active) => ({
  color: active ? "var(--text-primary)" : "var(--text-secondary)",
  textDecoration: "none",
  fontSize: "0.8rem",
  letterSpacing: "0.06em",
  fontWeight: active ? 600 : 400,
  transition: "color 0.2s",
});

const styles = {
  shell: {
    minHeight: "100vh",
    background: "var(--bg-primary)",
    fontFamily: "var(--font-family)",
    color: "var(--text-primary)",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
  },
  nav: {
    display: "flex",
    gap: "2rem",
    padding: "1.25rem 0",
    borderBottom: "1px solid var(--border-card)",
    width: "100%",
    justifyContent: "center",
    background: "var(--bg-card)",
  },
  main: {
    flex: 1,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    width: "100%",
  },
};