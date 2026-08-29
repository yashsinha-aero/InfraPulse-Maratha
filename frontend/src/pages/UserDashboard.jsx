import { useEffect, useRef, useState } from "react";
import { imageUrl, myComplaints, submitComplaint, WS_URL } from "../api/client";
import { useAuth } from "../context/AuthContext";

export default function UserDashboard() {
  const { logout } = useAuth();
  const [complaints, setComplaints] = useState([]);
  const [form, setForm] = useState({ name: "", address: "", description: "" });
  const [photo, setPhoto] = useState(null);
  const [photoPreview, setPhotoPreview] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const formRef = useRef(null);
  const wsRef = useRef(null);

  async function refresh() {
    try {
      setComplaints(await myComplaints());
    } catch (err) {
      setError(err.message);
    }
  }

  useEffect(() => {
    refresh();
    const ws = new WebSocket(`${WS_URL}/ws/queue`);
    wsRef.current = ws;
    ws.onmessage = () => refresh();
    return () => ws.close();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function handlePhoto(file) {
    setPhoto(file);
    setPhotoPreview(file ? URL.createObjectURL(file) : null);
  }

  async function submit(e) {
    e.preventDefault();
    if (!photo) return setError("Please attach a photo");
    setSubmitting(true);
    setError("");
    try {
      const fd = new FormData();
      fd.append("name", form.name);
      fd.append("address", form.address);
      fd.append("description", form.description);
      fd.append("photo", photo);
      await submitComplaint(fd);
      setForm({ name: "", address: "", description: "" });
      handlePhoto(null);
      formRef.current?.reset();
      refresh();
    } catch (err) {
      setError(err.message);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="dashboard">
      <div className="topbar">
        <div className="brand">
          <div className="brand-mark">IP</div>
          <div className="brand-text">
            <h1>InfraPulse</h1>
            <p>User dashboard</p>
          </div>
        </div>
        <div className="topbar-right">
          <span className="chip">👤 User</span>
          <button className="btn-ghost" onClick={logout}>Log out</button>
        </div>
      </div>

      <div className="card">
        <h3>Register a new complaint</h3>
        <form onSubmit={submit} ref={formRef}>
          <div className="field-group">
            <label className="field-label">Your name</label>
            <input
              value={form.name}
              onChange={(e) => setForm({ ...form, name: e.target.value })}
              required
            />
          </div>
          <div className="field-group">
            <label className="field-label">Address / location</label>
            <input
              placeholder="e.g. Hostel 4, Corridor near Room 212"
              value={form.address}
              onChange={(e) => setForm({ ...form, address: e.target.value })}
              required
            />
          </div>
          <div className="field-group">
            <label className="field-label">Description</label>
            <textarea
              rows={3}
              placeholder="Briefly describe what you see"
              value={form.description}
              onChange={(e) => setForm({ ...form, description: e.target.value })}
            />
          </div>
          <div className="field-group">
            <label className="field-label">Photo of the defect</label>
            <div className="file-drop">
              <span className="file-drop-label">
                {photo ? "Click to change photo" : "📷 Click to choose a photo"}
              </span>
              <input
                type="file"
                accept="image/*"
                onChange={(e) => handlePhoto(e.target.files[0])}
              />
            </div>
            {photoPreview && (
              <div className="file-preview">
                <img src={photoPreview} alt="preview" />
                <span>{photo.name}</span>
              </div>
            )}
          </div>
          {error && <p className="error">{error}</p>}
          <button className="btn-primary" type="submit" disabled={submitting}>
            {submitting && <span className="spinner" />}
            {submitting ? "Analyzing photo..." : "Submit Complaint"}
          </button>
        </form>
      </div>

      <div className="section-title">
        Your Complaints <span className="count">({complaints.length})</span>
      </div>

      {complaints.length === 0 ? (
        <div className="empty-state">
          <span className="emoji">📭</span>
          No complaints yet — submit one above to get started.
        </div>
      ) : (
        <div className="complaint-list">
          {complaints.map((c) => (
            <div className="complaint-card" key={c.id}>
              <img src={imageUrl(c.image_path)} alt={c.defect_label} />
              <div className="complaint-body">
                <div className="complaint-top-row">
                  <span className="defect-name">{c.defect_label.replace(/_/g, " ")}</span>
                  <span className="category-pill">{c.category}</span>
                  <span className={`status-pill status-${c.status.replace(/\s/g, "")}`}>
                    {c.status}
                  </span>
                </div>
                <div className="meta-row">
                  <span>Confidence: <b>{(c.confidence * 100).toFixed(0)}%</b></span>
                  {c.status !== "Resolved" && c.queue_position && (
                    <span>Queue position: <b>#{c.queue_position}</b></span>
                  )}
                </div>
                <div className="severity-bar-track" style={{ marginTop: 4 }}>
                  <div
                    className="severity-bar-fill"
                    style={{ width: `${Math.min(c.severity_score, 100)}%` }}
                  />
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
