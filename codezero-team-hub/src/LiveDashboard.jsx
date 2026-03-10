import { useState, useEffect } from "react";
import { G, Dot, Bdg, PB } from "./ui.jsx";

export default function LiveDashboard() {
  const [health, setHealth] = useState(null);
  const [err, setErr] = useState(null);
  const [brains, setBrains] = useState({});

  useEffect(() => {
    const f = async () => {
      try { const r = await fetch("/api/health"); setHealth(await r.json()); setErr(null); }
      catch (e) { setErr(e.message); }
    };
    f(); const iv = setInterval(f, 5000);
    return () => clearInterval(iv);
  }, []);

  useEffect(() => {
    const ping = async () => {
      const b = {};
      try { const t = performance.now(); const r = await fetch("/api/health"); b.engine = { ok: r.ok, ms: Math.round(performance.now() - t) }; } catch { b.engine = { ok: false, ms: 0 }; }
      try { const t = performance.now(); const r = await fetch("/api/ping/entry"); b.qwen = { ok: r.ok, ms: Math.round(performance.now() - t) }; } catch { b.qwen = { ok: false, ms: 0 }; }
      try { const t = performance.now(); const r = await fetch("/api/ping/exit"); b.exit = { ok: r.ok, ms: Math.round(performance.now() - t) }; } catch { b.exit = { ok: false, ms: 0 }; }
      try { const t = performance.now(); const r = await fetch("/api/atlas/health"); b.atlas = { ok: r.ok, ms: Math.round(performance.now() - t) }; } catch { b.atlas = { ok: false, ms: 0 }; }
      setBrains(b);
    };
    ping(); const iv = setInterval(ping, 30000);
    return () => clearInterval(iv);
  }, []);

  const h = health;
  const card = { background: "#12121f", border: "1px solid #1e293b", borderRadius: 8, padding: 14 };
  const lbl = { fontSize: 9, color: "#475569", fontFamily: "monospace", textTransform: "uppercase", letterSpacing: 1 };
  const vl = (c = "#e2e8f0") => ({ fontSize: 18, fontWeight: 700, fontFamily: "monospace", color: c });

  return (
    <div>
      <div style={{ fontSize: 12, color: "#64748b", fontFamily: "monospace", marginBottom: 12 }}>AI TEAM STATUS</div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12, marginBottom: 24 }}>
        {[
          { name: "Engine", sub: ":9091 health", data: brains.engine, color: "#06b6d4" },
          { name: "AI_1", sub: "GPU :8081 entry", data: brains.qwen, color: "#4ade80" },
          { name: "AI_2", sub: "GPU :8082 exit", data: brains.exit, color: "#a855f7" },
          { name: "AI_3", sub: ":8083 pre-scan + sentiment", data: brains.atlas, color: "#f0b429" },
        ].map((b, i) => (
          <div key={i} style={{ ...card, borderTop: `3px solid ${b.data?.ok ? b.color : b.data ? "#ff3b5c" : "#64748b"}` }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
              <Dot color={b.data?.ok ? "#4ade80" : b.data ? "#ff3b5c" : "#64748b"} />
              <span style={{ fontSize: 13, fontWeight: 600, fontFamily: "monospace" }}><G color={b.color}>{b.name}</G></span>
            </div>
            <div style={lbl}>{b.sub}</div>
            <div style={{ fontSize: 12, fontFamily: "monospace", color: b.data?.ok ? "#4ade80" : b.data ? "#ff3b5c" : "#64748b", marginTop: 4 }}>
              {b.data ? (b.data.ok ? `OK \u2022 ${b.data.ms}ms` : "OFFLINE") : "CHECKING..."}
            </div>
          </div>
        ))}
      </div>

      {h && (() => {
        const m = h.meters_60s || {};
        const aiCalls = m.ai_calls || 0;
        const exitChecks = m.nemo_exit_checks || 0;
        const npuRejects = m.npu_rejects || 0;
        const npuPasses = m.npu_passes || 0;
        const npuTotal = npuRejects + npuPasses;
        const total = aiCalls + exitChecks + npuTotal || 1;
        const gpuPct = Math.round((aiCalls / total) * 100);
        const cpuPct = Math.round((exitChecks / total) * 100);
        const npuPct = Math.round((npuTotal / total) * 100);
        const loadColor = (pct) => pct > 80 ? "#ff3b5c" : pct > 50 ? "#f0b429" : "#4ade80";
        const wkCard = { ...card, padding: 12 };
        return (
          <>
            <div style={{ fontSize: 12, color: "#64748b", fontFamily: "monospace", marginBottom: 12 }}>WORKLOAD BALANCE</div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12, marginBottom: 24 }}>
              {[
                { name: "GPU AI_1", role: "Context supervisor + 8 tools", calls: aiCalls, pct: gpuPct, color: "#4ade80", target: "~35-40%" },
                { name: "GPU AI_2", role: "Exit advisor (advisory only, math has authority)", calls: exitChecks, pct: cpuPct, color: "#a855f7", target: "~5-10%" },
                { name: "NPU AI_3", role: "Pre-scan gate + sentiment", calls: npuTotal, pct: npuPct, color: "#f0b429", target: "~30%" },
              ].map((w, i) => (
                <div key={i} style={{ ...wkCard, borderTop: `3px solid ${w.color}` }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                    <span style={{ fontSize: 12, fontWeight: 600, fontFamily: "monospace", color: w.color }}>{w.name}</span>
                    <span style={{ fontSize: 18, fontWeight: 700, fontFamily: "monospace", color: loadColor(w.pct) }}>{w.pct}%</span>
                  </div>
                  <PB value={w.pct} color={w.color} h={8} />
                  <div style={{ fontSize: 10, color: "#475569", fontFamily: "monospace", marginTop: 6 }}>{w.role}</div>
                  <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4 }}>
                    <span style={{ fontSize: 10, fontFamily: "monospace", color: "#64748b" }}>{w.calls} calls/60s</span>
                    <span style={{ fontSize: 10, fontFamily: "monospace", color: "#475569" }}>target {w.target}</span>
                  </div>
                </div>
              ))}
            </div>
          </>
        );
      })()}

      {err && <div style={{ background: "#ff3b5c15", border: "1px solid #ff3b5c40", borderRadius: 8, padding: 12, marginBottom: 16, color: "#ff3b5c", fontFamily: "monospace", fontSize: 12 }}>Engine offline: {err}</div>}

      {!h ? (
        <div style={{ textAlign: "center", padding: 40, color: "#64748b", fontFamily: "monospace" }}>Connecting to engine...</div>
      ) : (
        <>
          <div style={{ fontSize: 12, color: "#64748b", fontFamily: "monospace", marginBottom: 12 }}>ENGINE STATUS</div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: 12, marginBottom: 24 }}>
            {[
              { l: "Status", v: (h.status || "\u2014").toUpperCase(), c: h.status === "ok" ? "#4ade80" : "#ff3b5c" },
              { l: "Tick", v: h.tick?.toLocaleString() || "\u2014", c: "#06b6d4" },
              { l: "Heartbeat Age", v: h.heartbeat_age_sec != null ? `${h.heartbeat_age_sec}s` : "\u2014", c: (h.heartbeat_age_sec || 0) < 30 ? "#4ade80" : "#ff3b5c" },
              { l: "Kill Switch", v: h.manual_kill_switch?.active ? "ACTIVE" : "OFF", c: h.manual_kill_switch?.active ? "#ff3b5c" : "#4ade80" },
            ].map((s, i) => (
              <div key={i} style={card}><div style={lbl}>{s.l}</div><div style={vl(s.c)}>{s.v}</div></div>
            ))}
          </div>

          <div style={{ fontSize: 12, color: "#64748b", fontFamily: "monospace", marginBottom: 12 }}>BALANCES</div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 12, marginBottom: 24 }}>
            {[
              { l: "Available USD", v: `$${(h.balances?.available_usd || 0).toFixed(2)}`, c: "#4ade80" },
              { l: "Portfolio Value", v: `$${(h.balances?.portfolio_value || 0).toFixed(2)}`, c: "#06b6d4" },
              { l: "Holdings", v: h.balances?.holdings_count || 0, c: "#a855f7" },
            ].map((s, i) => (
              <div key={i} style={card}><div style={lbl}>{s.l}</div><div style={vl(s.c)}>{s.v}</div></div>
            ))}
          </div>
          {h.balances?.holdings && Object.keys(h.balances.holdings).length > 0 && (
            <div style={{ ...card, marginBottom: 24 }}>
              <div style={lbl}>HOLDINGS</div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginTop: 8 }}>
                {Object.entries(h.balances.holdings).map(([sym, qty]) => (
                  <Bdg key={sym} color="#06b6d4">{sym}: {typeof qty === "number" ? qty.toFixed(4) : qty}</Bdg>
                ))}
              </div>
            </div>
          )}

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 12, marginBottom: 24 }}>
            <div style={card}><div style={lbl}>Open Positions</div><div style={vl("#06b6d4")}>{h.positions?.open ?? "\u2014"}</div></div>
            <div style={card}><div style={lbl}>Pending</div><div style={vl("#f0b429")}>{h.positions?.pending ?? "\u2014"}</div></div>
          </div>

          <div style={{ fontSize: 12, color: "#64748b", fontFamily: "monospace", marginBottom: 12 }}>CIRCUIT BREAKER</div>
          <div style={{ ...card, marginBottom: 24, borderLeft: `3px solid ${h.circuit?.stop_trading ? "#ff3b5c" : h.circuit?.force_defensive ? "#f0b429" : "#4ade80"}` }}>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 12 }}>
              {[
                { l: "Daily PnL", v: `$${(h.circuit?.daily_pnl_usd || 0).toFixed(2)}`, c: (h.circuit?.daily_pnl_usd || 0) >= 0 ? "#4ade80" : "#ff3b5c" },
                { l: "Unrealized PnL", v: `$${(h.circuit?.unrealized_pnl_usd || 0).toFixed(2)}`, c: (h.circuit?.unrealized_pnl_usd || 0) >= 0 ? "#4ade80" : "#ff3b5c" },
                { l: "Defensive Mode", v: h.circuit?.force_defensive ? "ON" : "OFF", c: h.circuit?.force_defensive ? "#f0b429" : "#4ade80" },
                { l: "Stop Trading", v: h.circuit?.stop_trading ? "YES" : "NO", c: h.circuit?.stop_trading ? "#ff3b5c" : "#4ade80" },
              ].map((s, i) => (
                <div key={i}><div style={lbl}>{s.l}</div><div style={vl(s.c)}>{s.v}</div></div>
              ))}
            </div>
          </div>

          <div style={{ fontSize: 12, color: "#64748b", fontFamily: "monospace", marginBottom: 12 }}>60-SECOND METERS</div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))", gap: 12, marginBottom: 16 }}>
            {[
              { l: "AI Calls", v: h.meters_60s?.ai_calls ?? "\u2014", c: "#06b6d4" },
              { l: "BUY", v: h.meters_60s?.ai_action_buy ?? "\u2014", c: "#4ade80" },
              { l: "SELL", v: h.meters_60s?.ai_action_sell ?? "\u2014", c: "#ff3b5c" },
              { l: "HOLD", v: h.meters_60s?.ai_action_hold ?? "\u2014", c: "#64748b" },
              { l: "Entry Rejects", v: h.meters_60s?.entry_reject_total ?? "\u2014", c: "#f0b429" },
              { l: "Exit Checks", v: h.meters_60s?.nemo_exit_checks ?? "\u2014", c: "#a855f7" },
              { l: "Exit Sells", v: h.meters_60s?.nemo_exit_sells ?? "\u2014", c: "#ff3b5c" },
              { l: "Exit Holds", v: h.meters_60s?.nemo_exit_holds ?? "\u2014", c: "#4ade80" },
            ].map((s, i) => (
              <div key={i} style={card}><div style={lbl}>{s.l}</div><div style={vl(s.c)}>{s.v}</div></div>
            ))}
          </div>
          {h.meters_60s?.top_reject_reasons?.length > 0 && (
            <div style={{ ...card, marginBottom: 24 }}>
              <div style={lbl}>TOP REJECT REASONS</div>
              <div style={{ marginTop: 8 }}>
                {h.meters_60s.top_reject_reasons.map((r, i) => (
                  <div key={i} style={{ display: "flex", justifyContent: "space-between", padding: "4px 0", borderBottom: "1px solid #1e293b20" }}>
                    <span style={{ fontSize: 12, color: "#f0b429", fontFamily: "monospace" }}>{r.reason}</span>
                    <span style={{ fontSize: 12, color: "#e2e8f0", fontFamily: "monospace" }}>{r.count}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {h.hardware_guard && (
            <>
              <div style={{ fontSize: 12, color: "#64748b", fontFamily: "monospace", marginBottom: 12 }}>HARDWARE GUARD</div>
              <div style={{ ...card, marginBottom: 24, borderLeft: `3px solid ${h.hardware_guard.level === "green" || h.hardware_guard.level === "normal" ? "#4ade80" : h.hardware_guard.level === "yellow" || h.hardware_guard.level === "warm" ? "#f0b429" : "#ff3b5c"}` }}>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: 12 }}>
                  {[
                    { l: "Level", v: (h.hardware_guard.level || "\u2014").toUpperCase(), c: h.hardware_guard.level === "green" || h.hardware_guard.level === "normal" ? "#4ade80" : "#f0b429" },
                    { l: "Block Entries", v: h.hardware_guard.block_entries ? "YES" : "NO", c: h.hardware_guard.block_entries ? "#ff3b5c" : "#4ade80" },
                    { l: "AI Latency P95", v: h.hardware_guard.ai_latency_p95_ms != null ? `${h.hardware_guard.ai_latency_p95_ms.toFixed(0)}ms` : "\u2014", c: "#06b6d4" },
                    ...(h.hardware_guard.sample ? [
                      { l: "GPU Mem", v: h.hardware_guard.sample.gpu_mem_used_mb != null ? `${Math.round(h.hardware_guard.sample.gpu_mem_used_mb)}/${Math.round(h.hardware_guard.sample.gpu_mem_total_mb || 0)}MB` : "N/A", c: "#a855f7" },
                      { l: "GPU Temp", v: h.hardware_guard.sample.gpu_temp_c != null ? `${Math.round(h.hardware_guard.sample.gpu_temp_c)}\u00b0C` : "N/A", c: (h.hardware_guard.sample.gpu_temp_c || 0) > 80 ? "#ff3b5c" : "#4ade80" },
                      { l: "RAM Free", v: h.hardware_guard.sample.ram_free_mb != null ? `${Math.round(h.hardware_guard.sample.ram_free_mb)}MB` : "N/A", c: "#06b6d4" },
                    ] : []),
                  ].map((s, i) => (
                    <div key={i}><div style={lbl}>{s.l}</div><div style={vl(s.c)}>{s.v}</div></div>
                  ))}
                </div>
              </div>
            </>
          )}

          {h.watchdog && (
            <>
              <div style={{ fontSize: 12, color: "#64748b", fontFamily: "monospace", marginBottom: 12 }}>WATCHDOG</div>
              <div style={{ ...card, borderLeft: `3px solid ${h.watchdog.ok ? "#4ade80" : "#ff3b5c"}` }}>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: 12 }}>
                  {[
                    { l: "Status", v: h.watchdog.ok ? "OK" : "FAIL", c: h.watchdog.ok ? "#4ade80" : "#ff3b5c" },
                    { l: "Fail Count", v: h.watchdog.fail_count ?? 0, c: (h.watchdog.fail_count || 0) > 0 ? "#f0b429" : "#4ade80" },
                    { l: "Block Entries", v: h.watchdog.block_entries ? "YES" : "NO", c: h.watchdog.block_entries ? "#ff3b5c" : "#4ade80" },
                  ].map((s, i) => (
                    <div key={i}><div style={lbl}>{s.l}</div><div style={vl(s.c)}>{s.v}</div></div>
                  ))}
                </div>
                {h.watchdog.failing_paths?.length > 0 && (
                  <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginTop: 8 }}>
                    {h.watchdog.failing_paths.map((p, i) => <Bdg key={i} color="#ff3b5c">{p}</Bdg>)}
                  </div>
                )}
              </div>
            </>
          )}
        </>
      )}
    </div>
  );
}
