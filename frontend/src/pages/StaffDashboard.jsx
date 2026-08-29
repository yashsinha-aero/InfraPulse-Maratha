import { useEffect, useRef, useState } from "react";
import { imageUrl, staffQueue, updateStatus, WS_URL } from "../api/client";
import { useAuth } from "../context/AuthContext";

const NEXT_STATUS = {
  Submitted: "Assigned",
  Assigned: "In Progress",
  "In Progress": "Resolved",
};

export default function StaffDashboard() {
  const { auth, logout } = useAuth();
  const [queue, setQueue] = useState([]);
  const [error, setError] = useState("");
  const [busyId, setBusyId] = useState(null);
  const wsRef = useRef(null);

  async function refresh() {
    try {
      setQueue(await staffQueue());
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

  async function advance(c) {
    const next = NEXT_STATUS[c.status];
    if (!next) return;
    setBusyId(c.id);
    try {
      await updateStatus(c.id, next);
      refresh();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusyId(null);
    }
  }

  return (
    <div className="dashboard">
      <div className="topbar">
        <div className="brand">
          <div className="brand-mark">IP</div>
          <div className="brand-text">
            <h1>InfraPulse Staff</h1>
            <p>{auth?.category} priority queue</p>
          </div>
        </div>
        <div className="topbar-right">
          <span className="chip">🛠️ {auth?.category}</span>
          <button className="btn-ghost" onClick={logout}>Log out</button>
        </div>
      </div>

      {error && <p className="error">{error}</p>}

      <div className="section-title">
        Live Queue <span className="count">({queue.length})</span>
      </div>

      {queue.length === 0 ? (
        <div className="empty-state">
          <span className="emoji">✅</span>
          Queue is empty — nothing pending in your category right now.
        </div>
      ) : (
        <div className="queue-list">
          {queue.map((c) => (
            <div className="queue-row" key={c.id}>
              <span className="queue-pos">#{c.queue_position}</span>
              <img src={imageUrl(c.image_path)} alt={c.defect_label} />
              <div className="queue-info">
                <div className="complaint-top-row">
                  <span className="defect-name">{c.defect_label.replace(/_/g, " ")}</span>
                  <span className={`status-pill status-${c.status.replace(/\s/g, "")}`}>
                    {c.status}
                  </span>
                </div>
                <div className="meta-row">
                  <span>{c.name} — {c.address}</span>
                </div>
                {c.description && <p className="desc-text">{c.description}</p>}
                <div className="meta-row">
                  <span>Severity: <b>{c.severity_score.toFixed(0)}/100</b></span>
                  <span>Priority: <b>{c.type_priority}</b></span>
                </div>
                <div className="severity-bar-track">
                  <div
                    className="severity-bar-fill"
                    style={{ width: `${Math.min(c.severity_score, 100)}%` }}
                  />
                </div>
              </div>
              {NEXT_STATUS[c.status] && (
                <button
                  className="btn-primary"
                  onClick={() => advance(c)}
                  disabled={busyId === c.id}
                >
                  {busyId === c.id && <span className="spinner" />}
                  Mark {NEXT_STATUS[c.status]}
                </button>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
