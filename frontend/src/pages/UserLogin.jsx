import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { login, signup } from "../api/client";
import { useAuth } from "../context/AuthContext";

export default function UserLogin() {
  const [mode, setMode] = useState("login");
  const [form, setForm] = useState({ name: "", email: "", password: "" });
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
        mode === "login" ? await login(form) : await signup("user", form);
      loginWithToken(res);
      navigate("/user/dashboard");
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
            <h1>InfraPulse</h1>
            <p>Report it. Track it. Get it fixed.</p>
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
            <div className="field-group">
              <label className="field-label">Full name</label>
              <input
                placeholder="Jane Doe"
                value={form.name}
                onChange={(e) => setForm({ ...form, name: e.target.value })}
                required
              />
            </div>
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
            {mode === "login" ? "Log In" : "Create Account"}
          </button>
        </form>

        <p className="auth-switch-link">
          <Link to="/staff/login">Are you staff? Sign in here →</Link>
        </p>
      </div>
    </div>
  );
}
