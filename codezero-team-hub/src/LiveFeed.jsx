import { useState, useEffect, useRef } from "react";
import { G, Dot, Bdg } from "./ui.jsx";

export default function LiveFeed() {
  const [events, setEvents] = useState([]);
  const [filter, setFilter] = useState("all");
  const [paused, setPaused] = useState(false);
  const [connected, setConnected] = useState(false);
  const endRef = useRef(null);

  useEffect(() => {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${proto}//${window.location.host}/api/telemetry`);
    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onerror = () => setConnected(false);
    ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data);
        setEvents(prev => [...prev.slice(-499), { ...msg, _id: Date.now() + Math.random() }]);
      } catch {}
    };
    return () => ws.close();
  }, []);

  useEffect(() => {
    if (!paused) endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [events, paused]);

  const typeColors = { "decision.signal": "#06b6d4", "risk.reject": "#ff3b5c", "watcher.alert": "#f0b429", "features.snapshot": "#a855f7", "telemetry.lag": "#64748b" };
  const actionColors = { LONG: "#4ade80", BUY: "#4ade80", SHORT: "#ff3b5c", SELL: "#ff3b5c", HOLD: "#64748b" };
  const filtered = filter === "all" ? events : events.filter(e => e.type === filter);

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12, flexWrap: "wrap", gap: 8 }}>
        <div style={{ fontSize: 12, color: "#64748b", fontFamily: "monospace" }}>LIVE TELEMETRY FEED (WS :8765)</div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <Dot color={connected ? "#4ade80" : "#ff3b5c"} />
          <span style={{ fontSize: 11, fontFamily: "monospace", color: connected ? "#4ade80" : "#ff3b5c" }}>{connected ? "CONNECTED" : "DISCONNECTED"}</span>
        </div>
      </div>
      <div style={{ display: "flex", gap: 6, marginBottom: 12, flexWrap: "wrap" }}>
        {["all", "decision.signal", "risk.reject", "watcher.alert", "features.snapshot"].map(f => (
          <button key={f} onClick={() => setFilter(f)} style={{ padding: "4px 10px", fontSize: 10, fontFamily: "monospace", background: filter === f ? `${typeColors[f] || "#06b6d4"}15` : "transparent", border: filter === f ? `1px solid ${typeColors[f] || "#06b6d4"}40` : "1px solid #1e293b", color: filter === f ? (typeColors[f] || "#06b6d4") : "#64748b", borderRadius: 4, cursor: "pointer" }}>{f === "all" ? "ALL" : f.split(".")[1].toUpperCase()}</button>
        ))}
        <button onClick={() => setPaused(!paused)} style={{ marginLeft: "auto", padding: "4px 10px", fontSize: 10, fontFamily: "monospace", background: paused ? "#f0b42915" : "#4ade8015", border: `1px solid ${paused ? "#f0b429" : "#4ade80"}40`, color: paused ? "#f0b429" : "#4ade80", borderRadius: 4, cursor: "pointer" }}>{paused ? "\u25b6 RESUME" : "\u23f8 PAUSE"}</button>
        {events.length > 0 && <button onClick={() => setEvents([])} style={{ padding: "4px 10px", fontSize: 10, fontFamily: "monospace", background: "transparent", border: "1px solid #1e293b", color: "#64748b", borderRadius: 4, cursor: "pointer" }}>CLEAR</button>}
      </div>
      <div style={{ background: "#12121f", border: "1px solid #1e293b", borderRadius: 8, padding: 12, maxHeight: 600, overflowY: "auto" }}>
        {filtered.length === 0 && <div style={{ textAlign: "center", padding: 40, color: "#334155", fontFamily: "monospace", fontSize: 12 }}>{connected ? "Waiting for events..." : "Connect to engine to see live events"}</div>}
        {filtered.map((ev, i) => {
          const tc = typeColors[ev.type] || "#64748b";
          return (
            <div key={ev._id || i} style={{ padding: 10, marginBottom: 6, background: `${tc}08`, borderRadius: 6, borderLeft: `3px solid ${tc}` }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4, flexWrap: "wrap", gap: 4 }}>
                <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
                  <Bdg color={tc}>{ev.type?.split(".")[1] || ev.type}</Bdg>
                  {ev.symbol && <span style={{ fontSize: 12, fontWeight: 600, fontFamily: "monospace", color: "#e2e8f0" }}>{ev.symbol}</span>}
                  {ev.action && <Bdg color={actionColors[ev.action] || "#64748b"}>{ev.action}</Bdg>}
                  {ev.severity && <Bdg color={ev.severity === "Critical" ? "#ff3b5c" : ev.severity === "Warning" ? "#f0b429" : "#64748b"}>{ev.severity}</Bdg>}
                </div>
                <span style={{ fontSize: 9, color: "#475569", fontFamily: "monospace" }}>{ev.ts ? new Date(ev.ts * 1000).toLocaleTimeString() : ""}</span>
              </div>
              {ev.confidence != null && <span style={{ fontSize: 11, fontFamily: "monospace", color: "#94a3b8" }}>Confidence: <G color={ev.confidence >= 0.7 ? "#4ade80" : ev.confidence >= 0.4 ? "#f0b429" : "#ff3b5c"}>{(ev.confidence * 100).toFixed(0)}%</G></span>}
              {ev.reason && <div style={{ fontSize: 11, color: "#94a3b8", fontFamily: "monospace", marginTop: 2 }}>{ev.reason}</div>}
              {ev.text && <div style={{ fontSize: 11, color: "#94a3b8", fontFamily: "monospace", marginTop: 2 }}>{ev.text}</div>}
              <div style={{ display: "flex", gap: 8, marginTop: 2 }}>
                {ev.signal && <span style={{ fontSize: 10, color: "#475569", fontFamily: "monospace" }}>[{ev.signal}]</span>}
                {ev.lane && <span style={{ fontSize: 10, color: "#475569", fontFamily: "monospace" }}>Lane:{ev.lane}</span>}
                {ev.bucket && <span style={{ fontSize: 10, color: "#475569", fontFamily: "monospace" }}>Bucket:{ev.bucket}</span>}
              </div>
            </div>
          );
        })}
        <div ref={endRef} />
      </div>
      <div style={{ fontSize: 10, color: "#334155", fontFamily: "monospace", marginTop: 8, textAlign: "right" }}>{events.length} events total \u2022 {filtered.length} shown</div>
    </div>
  );
}
