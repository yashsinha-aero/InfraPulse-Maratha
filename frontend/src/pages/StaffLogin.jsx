import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { login, signup } from "../api/client";
import { useAuth } from "../context/AuthContext";

const CATEGORIES = ["Structural", "Functional", "Performance"];

export default function StaffLogin() {
  const [mode, setMode] = useState("login");
  const [form, setForm] = useState({
    name: "",
    email: "",
    password: "",
    category: CATEGORIES[0],
  });
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const { loginWithToken } = useAuth();
  const navigate = useNavigate();

  async function submit(e) {
    e.preventDefault();
    setError("");
    setSubmitting(true);
    try {
      const res =
        mode === "login" ? await login(form) : await signup("staff", form);
      loginWithToken(res);
      navigate("/staff/dashboard");
    } catch (err) {
      setError(err.message);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="auth-shell">
      <div className="auth-page">
        <div className="brand">
          <div className="brand-mark">IP</div>
          <div className="brand-text">
            <h1>InfraPulse Staff</h1>
            <p>Manage your category's priority queue</p>
          </div>
        </div>

        <div className="tabs">
          <button
            type="button"
            className={mode === "login" ? "active" : ""}
            onClick={() => setMode("login")}
          >
            Log In
          </button>
          <button
            type="button"
            className={mode === "signup" ? "active" : ""}
            onClick={() => setMode("signup")}
          >
            Sign Up
          </button>
        </div>

        <form onSubmit={submit}>
          {mode === "signup" && (
            <>
              <div className="field-group">
                <label className="field-label">Full name</label>
                <input
                  placeholder="Jane Doe"
                  value={form.name}
                  onChange={(e) => setForm({ ...form, name: e.target.value })}
                  required
                />
              </div>
              <div className="field-group">
                <label className="field-label">Category</label>
                <select
                  value={form.category}
                  onChange={(e) => setForm({ ...form, category: e.target.value })}
                >
                  {CATEGORIES.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
              </div>
            </>
          )}
          <div className="field-group">
            <label className="field-label">Email</label>
            <input
              type="email"
              placeholder="you@example.com"
              value={form.email}
              onChange={(e) => setForm({ ...form, email: e.target.value })}
              required
            />
          </div>
          <div className="field-group">
            <label className="field-label">Password</label>
            <input
              type="password"
              placeholder="••••••••"
              value={form.password}
              onChange={(e) => setForm({ ...form, password: e.target.value })}
              required
            />
          </div>
          {error && <p className="error">{error}</p>}
          <button type="submit" disabled={submitting}>
            {submitting && <span className="spinner" />}
            {mode === "login" ? "Log In" : "Create Staff Account"}
          </button>
        </form>

        <p className="auth-switch-link">
          <Link to="/">← Back to user login</Link>
        </p>
      </div>
    </div>
  );
}
