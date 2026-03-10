import { useState, useEffect, useCallback } from "react";
import "./storage.js"; // Initialize window.storage polyfill
import { G, Dot, Bdg, PB } from "./ui.jsx";
import LiveDashboard from "./LiveDashboard.jsx";
import QwenChat from "./QwenChat.jsx";
import LiveFeed from "./LiveFeed.jsx";
import Schedule from "./Schedule.jsx";

const SK = "codezero-hub-v4";

const defData = {
  tasks: [
    { id: 1, title: "Fix zombie positions — auto-track Kraken holdings (src/main.rs)", status: "done", priority: "high", category: "development", assignee: "Simeon" },
    { id: 2, title: "Fix HARD_MAX_SPREAD_PCT gate (2%→15%) — was blocking 65/72 coins", status: "done", priority: "high", category: "development", assignee: "Simeon" },
    { id: 3, title: "COIN_POOL expansion + dead coins removed from SYMBOLS", status: "done", priority: "high", category: "ops", assignee: "Simeon" },
    { id: 4, title: "MTF 7d/30d trend system — monitor performance live", status: "in_progress", priority: "medium", category: "trading", assignee: "Simeon" },
    { id: 5, title: "GBDT Slot 4 — p_win/p_loss ML model for dynamic stop tightening", status: "todo", priority: "medium", category: "research", assignee: "Simeon" },
    { id: 6, title: "Enable hardware guard GPU sampling (returning null currently)", status: "todo", priority: "medium", category: "ops", assignee: "Simeon" },
    { id: 7, title: "Raise BASE_USD → $20 once account consistently > $150", status: "todo", priority: "low", category: "trading", assignee: "Simeon" },
  ],
  nextId: 8, teamMembers: ["Simeon", "Unassigned"], notes: [],
  journal: [], journalNextId: 1
};

const pri = { high: "#ff3b5c", medium: "#f0b429", low: "#4ade80" };
const cats = { development: "#06b6d4", trading: "#a855f7", research: "#3b82f6", ops: "#f97316" };
const sts = ["todo", "in_progress", "done"];
const stL = { todo: "TO DO", in_progress: "IN PROGRESS", done: "DONE" };
const stC = { todo: "#64748b", in_progress: "#06b6d4", done: "#4ade80" };

const archModels = [
  { name: "AI_1", hw: "GPU — RTX 5060 Ti 16GB • :8081", role: "CONTEXT SUPERVISOR (~35-40%)", color: "#4ade80", desc: "Regime context, risk flags, and confidence weight adjustments. Quant scores make primary entry decisions — AI_1 adjusts the weight, not the direction. 8 tools for deep investigation.", tools: ["get_coin_features", "get_trade_history", "get_market_context", "get_ai_memory", "get_correlated_coins", "get_engine_status", "get_top_movers", "get_open_positions"] },
  { name: "AI_2", hw: "GPU — RTX 5060 Ti 16GB • :8082", role: "EXIT ADVISOR (advisory only)", color: "#a855f7", desc: "Exit decisions are deterministic (ATR-based stop stack). AI_2 logs advisory opinions for review but does NOT execute exits. Math has exit authority.", tools: ["get_coin_features", "get_trade_history", "get_market_context", "get_ai_memory", "get_correlated_coins", "get_engine_status", "get_top_movers"] },
  { name: "AI_3", hw: "Intel NPU — Ultra 7 265F • :8083", role: "PRE-SCAN GATE + SENTIMENT (~30%)", color: "#06b6d4", desc: "Dual role: (1) AI_3 Scanner pre-screens candidates in 200ms — rejects junk before GPU sees it. (2) Sentiment model scores news headline sentiment. Zero GPU/CPU load.", tools: ["pre_scan_gate", "sentiment_batch", "headline_classify"] },
  { name: "Slot 4", hw: "Available — hot-swap capable", role: "GBDT SLOT (future)", color: "#64748b", desc: "Reserved for GBDT ML model — p_win/p_loss predictions for dynamic stop tightening. Not yet active.", tools: [] },
];

const phases = [
  { p: 1, n: "Market Data Ingestion", d: "Kraken WebSocket feeds → tick data + order book depth" },
  { p: 2, n: "Feature Extraction", d: "Multi-scale fingerprint (5 horizons × 8 metrics) + greed index" },
  { p: 3, n: "Market Intelligence", d: "Market watcher: cross-coin signals, whale detection, sector rotation" },
  { p: 4, n: "NPU Pre-Scan Gate (AI_3 ~30%)", d: "AI_3 Scanner on NPU screens candidates in 200ms — rejects junk before GPU. Saves ~40% GPU cycles." },
  { p: 5, n: "AI_3 Sentiment (NPU :8083)", d: "Sentiment model scores news headlines → per-coin sentiment injection every 5 min" },
  { p: 6, n: "Strategy & Lane Classification", d: "L1 trend / L2 mean-revert / L3 moderate bucket + lane assignment" },
  { p: 7, n: "Quant Entry + AI Context", d: "Quant score is primary. AI_1 adjusts confidence weight (not direction). Schedule-aware margin gate (0.15-0.20 by time block)." },
  { p: 8, n: "Risk Gates & Order Execution", d: "Spread gate, circuit breaker, hardware guard, 3-loss cooldown → Kraken REST API" },
  { p: 9, n: "Deterministic Exit Stack", d: "ATR hard stop (1.5-2.0×) + ATR trailing (lane-dependent) + time contract. AI_2 logs advisory only." },
  { p: 10, n: "Portfolio & Risk Management", d: "CVaR optimizer, daily PnL limits, hardware guard, watchdog" },
];

const playbooks = [
  { name: "Momentum Breakout", desc: "High-confidence trend-following with quant scoring and AI context weighting.", flow: "Feature vector → quant threshold → AI_1 adjusts confidence weight → ATR trailing exit", rules: ["Entry: Quant score ≥ threshold + AI_1 confidence weight ≥ 0.70", "Watch: AI_3 flags momentum divergence from 20-period baseline", "Exit: ATR hard stop (2.0×) + ATR trailing stop (2.0× L1) + time contract enforcement", "Session: HIGH VOL block (05:30–09:30 AKST) — full size, margin ≥0.15", "Position: 2–5% portfolio via Kelly criterion"], risk: "Max drawdown 5%/day. 3 consecutive losses → 1hr entry block." },
  { name: "Mean Reversion Scalp", desc: "Short-term reversion during low-volatility windows with deterministic exits.", flow: "Feature vector → L2 quant threshold → AI_1 adjusts weight → ATR tight trailing exit", rules: ["Entry: Price deviation ≥ 2σ + quant BUY + AI_1 weight ≥ 0.70", "Watch: AI_3 confirms low-volatility regime", "Exit: ATR hard stop (1.5×) + ATR trailing stop (1.3× L2) + time contract", "Session: MIDDAY block (09:30–12:00 AKST) — conservative, tighter ATR", "Position: 1–3% portfolio"], risk: "3 consecutive losses → 1hr cooldown. Schedule reduces L2 aggression in AFTERNOON." },
  { name: "NPU Sentiment Sweep", desc: "Sentiment-driven entry via NPU with quant confirmation.", flow: "NPU sentiment scoring → quant validates divergence → AI_1 adjusts context weight", rules: ["Entry: Sentiment divergence > 0.3 + quant BUY + AI_1 weight adjustment", "Watch: AI_3 real-time sentiment on news feeds", "Exit: ATR hard stop + trailing stop + 4hr max_hold_sec time contract", "Session: Pre-NYSE open for news catalysts", "Position: 1–2% portfolio (experimental)"], risk: "Paper trade first 2 weeks. Max 5% total." }
];

const sopSteps = [
  { title: "1. System Access & Rust Toolchain", items: ["Get CODE ZERO repo access (Rust codebase)", "Install Rust toolchain (rustup + nightly)", "Install CUDA toolkit + OpenVINO for GPU/NPU", "Configure Kraken API keys (read-only first)", "Run cargo build --release and verify compilation"] },
  { title: "2. Architecture Familiarization", items: ["Review 9-phase pipeline architecture", "Understand model roles: AI_1 entry, AI_3 watch, AI_2 exit", "Study Rust module structure and data flow", "Review Kraken WebSocket integration code", "Run cargo test to verify all pipeline phases"] },
  { title: "3. Team Protocols", items: ["Join team comms channel", "Review all trading playbooks & model flow diagrams", "Shadow live trading sessions (min 3)", "Complete risk management quiz", "Understand emergency shutdown procedure (kill switch)"] },
  { title: "4. Go Live Checklist", items: ["System health check passing (all 3 models responding)", "Paper traded minimum 50 trades", "Confirmed 0.038ms or better pipeline latency", "Risk parameters reviewed with lead", "Emergency shutdown procedure memorized"] },
];


function TC({ task, onMove, onDel }) {
  return (
    <div style={{ background: "#12121f", border: "1px solid #1e293b", borderRadius: 8, padding: 12, marginBottom: 8, borderLeft: `3px solid ${pri[task.priority]}` }}>
      <div style={{ fontSize: 13, color: "#e2e8f0", marginBottom: 8, fontWeight: 500 }}>{task.title}</div>
      <div style={{ display: "flex", gap: 6, flexWrap: "wrap", alignItems: "center", marginBottom: 8 }}>
        <Bdg color={pri[task.priority]}>{task.priority}</Bdg>
        <Bdg color={cats[task.category]}>{task.category}</Bdg>
        <span style={{ fontSize: 10, color: "#64748b", marginLeft: "auto" }}>{task.assignee}</span>
      </div>
      <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
        {sts.filter(s => s !== task.status).map(s => (
          <button key={s} onClick={() => onMove(task.id, s)} style={{ fontSize: 9, padding: "2px 6px", background: `${stC[s]}15`, color: stC[s], border: `1px solid ${stC[s]}30`, borderRadius: 4, cursor: "pointer", fontFamily: "monospace" }}>→ {stL[s]}</button>
        ))}
        <button onClick={() => onDel(task.id)} style={{ fontSize: 9, padding: "2px 6px", background: "#ff3b5c10", color: "#ff3b5c", border: "1px solid #ff3b5c30", borderRadius: 4, cursor: "pointer", marginLeft: "auto", fontFamily: "monospace" }}>✕</button>
      </div>
    </div>
  );
}

function ATF({ onAdd, tm }) {
  const [o, setO] = useState(false);
  const [t, setT] = useState(""); const [p, setP] = useState("medium"); const [c, setC] = useState("development"); const [a, setA] = useState("Unassigned");
  const sub = () => { if (!t.trim()) return; onAdd({ title: t.trim(), priority: p, category: c, assignee: a }); setT(""); setP("medium"); setC("development"); setA("Unassigned"); setO(false); };
  if (!o) return <button onClick={() => setO(true)} style={{ width: "100%", padding: 10, background: "#06b6d410", border: "1px dashed #06b6d440", borderRadius: 8, color: "#06b6d4", cursor: "pointer", fontSize: 13, fontFamily: "monospace" }}>+ Add Task</button>;
  const ss = { flex: 1, padding: 6, background: "#0a0a14", border: "1px solid #1e293b", borderRadius: 4, color: "#e2e8f0", fontSize: 11 };
  return (
    <div style={{ background: "#12121f", border: "1px solid #06b6d430", borderRadius: 8, padding: 12 }}>
      <input value={t} onChange={e => setT(e.target.value)} placeholder="Task title..." onKeyDown={e => e.key === "Enter" && sub()} style={{ width: "100%", padding: 8, background: "#0a0a14", border: "1px solid #1e293b", borderRadius: 4, color: "#e2e8f0", fontSize: 13, marginBottom: 8, boxSizing: "border-box" }} autoFocus />
      <div style={{ display: "flex", gap: 6, marginBottom: 8, flexWrap: "wrap" }}>
        <select value={p} onChange={e => setP(e.target.value)} style={ss}><option value="high">High</option><option value="medium">Medium</option><option value="low">Low</option></select>
        <select value={c} onChange={e => setC(e.target.value)} style={ss}><option value="development">Dev</option><option value="trading">Trading</option><option value="research">Research</option><option value="ops">Ops</option></select>
        <select value={a} onChange={e => setA(e.target.value)} style={ss}>{tm.map(m => <option key={m} value={m}>{m}</option>)}</select>
      </div>
      <div style={{ display: "flex", gap: 6 }}>
        <button onClick={sub} style={{ flex: 1, padding: 6, background: "#06b6d4", border: "none", borderRadius: 4, color: "#0a0a14", cursor: "pointer", fontWeight: 600, fontSize: 12 }}>Add</button>
        <button onClick={() => setO(false)} style={{ padding: 6, background: "transparent", border: "1px solid #1e293b", borderRadius: 4, color: "#64748b", cursor: "pointer", fontSize: 12 }}>Cancel</button>
      </div>
    </div>
  );
}

function RiskCalc() {
  const [bal, setBal] = useState(62);
  const [riskPct, setRiskPct] = useState(2);
  const [entry, setEntry] = useState(100);
  const [stop, setStop] = useState(98.5);
  const [winRate, setWinRate] = useState(55);
  const [avgWin, setAvgWin] = useState(3);
  const [avgLoss, setAvgLoss] = useState(1.5);

  const riskAmt = bal * (riskPct / 100);
  const stopDist = Math.abs(entry - stop);
  const posSize = stopDist > 0 ? riskAmt / stopDist : 0;
  const posValue = posSize * entry;
  const posPct = bal > 0 ? (posValue / bal) * 100 : 0;
  const wr = winRate / 100;
  const kelly = wr > 0 && avgLoss > 0 ? ((wr * avgWin) - ((1 - wr) * avgLoss)) / avgWin : 0;
  const halfKelly = kelly / 2;
  const expectancy = (wr * avgWin) - ((1 - wr) * avgLoss);

  const is = { width: "100%", padding: 8, background: "#0a0a14", border: "1px solid #1e293b", borderRadius: 4, color: "#e2e8f0", fontSize: 13, boxSizing: "border-box" };

  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 16 }}>
        <div style={{ background: "#12121f", border: "1px solid #1e293b", borderRadius: 8, padding: 16 }}>
          <div style={{ fontSize: 12, color: "#06b6d4", fontFamily: "monospace", marginBottom: 12 }}>◆ POSITION SIZER</div>
          <div style={{ marginBottom: 8 }}><label style={{ fontSize: 11, color: "#64748b", fontFamily: "monospace" }}>ACCOUNT BALANCE ($)</label><input type="number" value={bal} onChange={e => setBal(+e.target.value)} style={is} /></div>
          <div style={{ marginBottom: 8 }}><label style={{ fontSize: 11, color: "#64748b", fontFamily: "monospace" }}>RISK PER TRADE (%)</label><input type="number" value={riskPct} onChange={e => setRiskPct(+e.target.value)} step="0.5" style={is} /></div>
          <div style={{ marginBottom: 8 }}><label style={{ fontSize: 11, color: "#64748b", fontFamily: "monospace" }}>ENTRY PRICE ($)</label><input type="number" value={entry} onChange={e => setEntry(+e.target.value)} style={is} /></div>
          <div style={{ marginBottom: 12 }}><label style={{ fontSize: 11, color: "#64748b", fontFamily: "monospace" }}>STOP LOSS ($)</label><input type="number" value={stop} onChange={e => setStop(+e.target.value)} style={is} /></div>
          <div style={{ background: "#0a0a14", borderRadius: 6, padding: 12, border: "1px solid #06b6d420" }}>
            <div style={{ fontSize: 10, color: "#64748b", fontFamily: "monospace", marginBottom: 8 }}>RESULTS</div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
              {[{ l: "Risk Amount", v: `$${riskAmt.toFixed(2)}`, c: "#f0b429" }, { l: "Stop Distance", v: `$${stopDist.toFixed(2)}`, c: "#ff3b5c" }, { l: "Position Size", v: `${posSize.toFixed(6)} units`, c: "#06b6d4" }, { l: "Position Value", v: `$${posValue.toFixed(2)}`, c: "#4ade80" }].map((r, i) => (
                <div key={i}><div style={{ fontSize: 9, color: "#475569", fontFamily: "monospace" }}>{r.l}</div><div style={{ fontSize: 14, fontWeight: 700, fontFamily: "monospace" }}><G color={r.c}>{r.v}</G></div></div>
              ))}
            </div>
            <div style={{ marginTop: 8 }}><div style={{ fontSize: 9, color: "#475569", fontFamily: "monospace", marginBottom: 4 }}>PORTFOLIO EXPOSURE: {posPct.toFixed(1)}%</div><PB value={Math.min(posPct, 100)} color={posPct > 50 ? "#ff3b5c" : posPct > 25 ? "#f0b429" : "#4ade80"} h={8} />{posPct > 25 && <div style={{ fontSize: 10, color: "#ff3b5c", fontFamily: "monospace", marginTop: 4 }}>⚠ High exposure — consider reducing position</div>}</div>
          </div>
        </div>
        <div style={{ background: "#12121f", border: "1px solid #1e293b", borderRadius: 8, padding: 16 }}>
          <div style={{ fontSize: 12, color: "#a855f7", fontFamily: "monospace", marginBottom: 12 }}>◆ KELLY CRITERION & EXPECTANCY</div>
          <div style={{ marginBottom: 8 }}><label style={{ fontSize: 11, color: "#64748b", fontFamily: "monospace" }}>WIN RATE (%)</label><input type="number" value={winRate} onChange={e => setWinRate(+e.target.value)} style={is} /></div>
          <div style={{ marginBottom: 8 }}><label style={{ fontSize: 11, color: "#64748b", fontFamily: "monospace" }}>AVG WIN (%)</label><input type="number" value={avgWin} onChange={e => setAvgWin(+e.target.value)} step="0.5" style={is} /></div>
          <div style={{ marginBottom: 12 }}><label style={{ fontSize: 11, color: "#64748b", fontFamily: "monospace" }}>AVG LOSS (%)</label><input type="number" value={avgLoss} onChange={e => setAvgLoss(+e.target.value)} step="0.5" style={is} /></div>
          <div style={{ background: "#0a0a14", borderRadius: 6, padding: 12, border: "1px solid #a855f720" }}>
            <div style={{ fontSize: 10, color: "#64748b", fontFamily: "monospace", marginBottom: 8 }}>RESULTS</div>
            {[{ l: "Full Kelly", v: `${(kelly * 100).toFixed(1)}%`, c: kelly > 0 ? "#4ade80" : "#ff3b5c", d: "Theoretical optimal — too aggressive for live" }, { l: "Half Kelly (Recommended)", v: `${(halfKelly * 100).toFixed(1)}%`, c: halfKelly > 0 ? "#06b6d4" : "#ff3b5c", d: "Safer — reduces variance, captures most of the edge" }, { l: "Expectancy per Trade", v: `${expectancy > 0 ? "+" : ""}${expectancy.toFixed(2)}%`, c: expectancy > 0 ? "#4ade80" : "#ff3b5c", d: expectancy > 0 ? "Positive edge — profitable long-term" : "Negative edge — do NOT trade this live" }].map((r, i) => (
              <div key={i} style={{ marginBottom: 10 }}><div style={{ fontSize: 9, color: "#475569", fontFamily: "monospace" }}>{r.l}</div><div style={{ fontSize: 18, fontWeight: 700, fontFamily: "monospace" }}><G color={r.c}>{r.v}</G></div><div style={{ fontSize: 10, color: "#64748b" }}>{r.d}</div></div>
            ))}
          </div>
          <div style={{ marginTop: 12, padding: 10, background: "#f0b42908", border: "1px solid #f0b42920", borderRadius: 6 }}>
            <div style={{ fontSize: 10, color: "#f0b429", fontFamily: "monospace" }}>⚠ DRAWDOWN RULES</div>
            <div style={{ fontSize: 11, color: "#fcd34d", marginTop: 4, fontFamily: "monospace", lineHeight: 1.7 }}>
              <div>• Daily max loss: 5% → stop trading for the day</div>
              <div>• Weekly max loss: 10% → reduce size 50% next week</div>
              <div>• Monthly max loss: 15% → full pipeline review</div>
              <div>• 3 consecutive losses → mandatory 1hr cooldown</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function Emergency() {
  return (
    <div>
      <div style={{ background: "#ff3b5c10", border: "2px solid #ff3b5c40", borderRadius: 8, padding: 20, marginBottom: 16, textAlign: "center" }}>
        <div style={{ fontSize: 24, fontWeight: 700, fontFamily: "monospace" }}><G color="#ff3b5c">⚠ EMERGENCY PROCEDURES ⚠</G></div>
        <div style={{ fontSize: 12, color: "#fca5a5", marginTop: 8 }}>Every team member must memorize these. Practice quarterly.</div>
      </div>
      {[
        { title: "🔴 LEVEL 1: KILL SWITCH — FULL SHUTDOWN", color: "#ff3b5c", steps: ["SSH into server: ssh codezero@[SERVER_IP]", "Run kill command: ./kill_switch.sh (cancels ALL open orders)", "Verify: Kraken dashboard — 0 open orders, 0 positions", "Notify team immediately", "Do NOT restart until post-mortem complete"], when: "System malfunction, unexpected massive loss, API erratic, or any situation you don't understand." },
        { title: "🟠 LEVEL 2: PAUSE TRADING — SOFT STOP", color: "#f97316", steps: ["Run: cargo run --bin pause_pipeline", "Halts new orders, monitors existing positions", "CPU AI_2 continues exit monitoring", "Review logs: tail -f /var/log/codezero/pipeline.log", "Resume: cargo run --bin resume_pipeline"], when: "Daily loss limit hit (5%), flash crash, 3+ consecutive losses, or maintenance." },
        { title: "🟡 LEVEL 3: REDUCE EXPOSURE — DEFENSIVE MODE", color: "#f0b429", steps: ["Run: cargo run --bin defensive_mode", "Cuts all position sizes by 50%", "Raises confidence threshold to 0.90", "Tightens all stops by 1%", "Auto-reverts after 24hrs unless extended"], when: "Weekly loss approaching 10%, high-vol macro event, or model confidence dropping." },
      ].map((level, i) => (
        <div key={i} style={{ background: "#12121f", border: `1px solid ${level.color}30`, borderRadius: 8, padding: 16, marginBottom: 12, borderLeft: `4px solid ${level.color}` }}>
          <div style={{ fontSize: 14, fontWeight: 700, fontFamily: "monospace", marginBottom: 8 }}><G color={level.color}>{level.title}</G></div>
          <div style={{ fontSize: 11, color: "#94a3b8", marginBottom: 12, fontStyle: "italic" }}>{level.when}</div>
          {level.steps.map((s, j) => (
            <div key={j} style={{ display: "flex", gap: 10, padding: "8px 0", borderBottom: j < level.steps.length - 1 ? "1px solid #1e293b20" : "none" }}>
              <div style={{ width: 24, height: 24, borderRadius: 4, background: `${level.color}15`, border: `1px solid ${level.color}30`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11, fontWeight: 700, color: level.color, flexShrink: 0, fontFamily: "monospace" }}>{j + 1}</div>
              <div style={{ fontSize: 13, color: "#cbd5e1", fontFamily: "monospace" }}>{s}</div>
            </div>
          ))}
        </div>
      ))}
      <div style={{ background: "#12121f", border: "1px solid #1e293b", borderRadius: 8, padding: 16, marginTop: 16 }}>
        <div style={{ fontSize: 12, color: "#06b6d4", fontFamily: "monospace", marginBottom: 12 }}>◆ POST-INCIDENT CHECKLIST</div>
        {["Document what happened (time, trigger, conditions)", "Review pipeline logs for anomalies", "Check model outputs for unexpected behavior", "Calculate actual vs expected loss", "Update playbook if new risk pattern found", "Team debrief within 24 hours", "File incident in Trade Journal with INCIDENT tag"].map((item, i) => (
          <div key={i} style={{ fontSize: 12, color: "#94a3b8", padding: "6px 0", borderBottom: "1px solid #1e293b10", fontFamily: "monospace" }}><span style={{ color: "#06b6d4" }}>☐</span> {item}</div>
        ))}
      </div>
    </div>
  );
}

function Journal({ entries, onAdd, onDel }) {
  const [open, setOpen] = useState(false);
  const [f, setF] = useState({ coin: "", direction: "LONG", thesis: "", entry: "", target: "", stop: "", size: "", timeframe: "Swing", confidence: "", outcome: "", notes: "", tag: "TRADE" });
  const sub = () => { if (!f.coin.trim() || !f.thesis.trim()) return; onAdd({ ...f, id: Date.now(), date: new Date().toLocaleDateString() + " " + new Date().toLocaleTimeString() }); setF({ coin: "", direction: "LONG", thesis: "", entry: "", target: "", stop: "", size: "", timeframe: "Swing", confidence: "", outcome: "", notes: "", tag: "TRADE" }); setOpen(false); };
  const is = { width: "100%", padding: 8, background: "#0a0a14", border: "1px solid #1e293b", borderRadius: 4, color: "#e2e8f0", fontSize: 12, boxSizing: "border-box" };
  const ss = { ...is, flex: 1 };
  const tagColors = { TRADE: "#06b6d4", INCIDENT: "#ff3b5c", RESEARCH: "#a855f7", LESSON: "#f0b429" };

  return (
    <div>
      {!open ? (
        <button onClick={() => setOpen(true)} style={{ width: "100%", padding: 14, background: "#06b6d410", border: "1px dashed #06b6d440", borderRadius: 8, color: "#06b6d4", cursor: "pointer", fontSize: 14, fontFamily: "monospace", marginBottom: 16 }}>+ NEW JOURNAL ENTRY</button>
      ) : (
        <div style={{ background: "#12121f", border: "1px solid #06b6d430", borderRadius: 8, padding: 16, marginBottom: 16 }}>
          <div style={{ fontSize: 12, color: "#06b6d4", fontFamily: "monospace", marginBottom: 12 }}>◆ NEW ENTRY</div>
          <div style={{ display: "flex", gap: 8, marginBottom: 8, flexWrap: "wrap" }}>
            <input value={f.coin} onChange={e => setF({...f, coin: e.target.value})} placeholder="COIN (e.g. BTC/USD)" style={ss} />
            <select value={f.direction} onChange={e => setF({...f, direction: e.target.value})} style={ss}><option>LONG</option><option>SHORT</option><option>N/A</option></select>
            <select value={f.tag} onChange={e => setF({...f, tag: e.target.value})} style={ss}><option>TRADE</option><option>INCIDENT</option><option>RESEARCH</option><option>LESSON</option></select>
          </div>
          <textarea value={f.thesis} onChange={e => setF({...f, thesis: e.target.value})} placeholder="Thesis — Why are you taking this trade?" rows={3} style={{ ...is, marginBottom: 8, resize: "vertical" }} />
          <div style={{ display: "flex", gap: 8, marginBottom: 8, flexWrap: "wrap" }}>
            <input value={f.entry} onChange={e => setF({...f, entry: e.target.value})} placeholder="Entry $" style={ss} />
            <input value={f.target} onChange={e => setF({...f, target: e.target.value})} placeholder="Target $" style={ss} />
            <input value={f.stop} onChange={e => setF({...f, stop: e.target.value})} placeholder="Stop $" style={ss} />
          </div>
          <div style={{ display: "flex", gap: 8, marginBottom: 8, flexWrap: "wrap" }}>
            <input value={f.size} onChange={e => setF({...f, size: e.target.value})} placeholder="Position size %" style={ss} />
            <select value={f.timeframe} onChange={e => setF({...f, timeframe: e.target.value})} style={ss}><option>Scalp</option><option>Swing</option><option>Position</option></select>
            <input value={f.confidence} onChange={e => setF({...f, confidence: e.target.value})} placeholder="Confidence (0-1)" style={ss} />
          </div>
          <select value={f.outcome} onChange={e => setF({...f, outcome: e.target.value})} style={{ ...is, marginBottom: 8 }}><option value="">Outcome (fill after close)</option><option value="WIN">WIN</option><option value="LOSS">LOSS</option><option value="BREAKEVEN">BREAKEVEN</option><option value="OPEN">STILL OPEN</option></select>
          <textarea value={f.notes} onChange={e => setF({...f, notes: e.target.value})} placeholder="Notes — what did you learn?" rows={2} style={{ ...is, marginBottom: 8, resize: "vertical" }} />
          <div style={{ display: "flex", gap: 6 }}>
            <button onClick={sub} style={{ flex: 1, padding: 8, background: "#06b6d4", border: "none", borderRadius: 4, color: "#0a0a14", cursor: "pointer", fontWeight: 600 }}>Save Entry</button>
            <button onClick={() => setOpen(false)} style={{ padding: 8, background: "transparent", border: "1px solid #1e293b", borderRadius: 4, color: "#64748b", cursor: "pointer" }}>Cancel</button>
          </div>
        </div>
      )}
      {entries.length === 0 && <div style={{ textAlign: "center", padding: 40, color: "#334155", fontFamily: "monospace", fontSize: 12 }}>No journal entries yet. Log your first trade thesis above.</div>}
      {entries.map(e => (
        <div key={e.id} style={{ background: "#12121f", border: `1px solid ${tagColors[e.tag] || "#1e293b"}25`, borderRadius: 8, padding: 14, marginBottom: 10, borderLeft: `3px solid ${tagColors[e.tag] || "#64748b"}` }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8, flexWrap: "wrap", gap: 6 }}>
            <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
              <span style={{ fontSize: 14, fontWeight: 700, fontFamily: "monospace" }}><G color={tagColors[e.tag]}>{e.coin || "—"}</G></span>
              <Bdg color={e.direction === "LONG" ? "#4ade80" : e.direction === "SHORT" ? "#ff3b5c" : "#64748b"}>{e.direction}</Bdg>
              <Bdg color={tagColors[e.tag]}>{e.tag}</Bdg>
              {e.outcome && <Bdg color={e.outcome === "WIN" ? "#4ade80" : e.outcome === "LOSS" ? "#ff3b5c" : "#f0b429"}>{e.outcome}</Bdg>}
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span style={{ fontSize: 10, color: "#475569", fontFamily: "monospace" }}>{e.date}</span>
              <button onClick={() => onDel(e.id)} style={{ background: "none", border: "none", color: "#475569", cursor: "pointer", fontSize: 12 }}>✕</button>
            </div>
          </div>
          <div style={{ fontSize: 12, color: "#cbd5e1", lineHeight: 1.6, marginBottom: 8 }}>{e.thesis}</div>
          {(e.entry || e.target || e.stop) && (
            <div style={{ display: "flex", gap: 12, fontSize: 11, fontFamily: "monospace", color: "#64748b", marginBottom: 6 }}>
              {e.entry && <span>Entry: <span style={{ color: "#e2e8f0" }}>${e.entry}</span></span>}
              {e.target && <span>Target: <span style={{ color: "#4ade80" }}>${e.target}</span></span>}
              {e.stop && <span>Stop: <span style={{ color: "#ff3b5c" }}>${e.stop}</span></span>}
              {e.size && <span>Size: <span style={{ color: "#06b6d4" }}>{e.size}%</span></span>}
            </div>
          )}
          {e.notes && <div style={{ fontSize: 11, color: "#94a3b8", marginTop: 6, padding: 8, background: "#0a0a14", borderRadius: 4, fontStyle: "italic" }}>{e.notes}</div>}
        </div>
      ))}
      {entries.length > 0 && (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))", gap: 10, marginTop: 16 }}>
          {[{ l: "Total", v: entries.length, c: "#06b6d4" }, { l: "Wins", v: entries.filter(e => e.outcome === "WIN").length, c: "#4ade80" }, { l: "Losses", v: entries.filter(e => e.outcome === "LOSS").length, c: "#ff3b5c" }, { l: "Win Rate", v: entries.filter(e => e.outcome).length > 0 ? `${((entries.filter(e => e.outcome === "WIN").length / entries.filter(e => e.outcome === "WIN" || e.outcome === "LOSS").length) * 100 || 0).toFixed(0)}%` : "—", c: "#f0b429" }].map((s, i) => (
            <div key={i} style={{ background: "#12121f", border: "1px solid #1e293b", borderRadius: 8, padding: 12, textAlign: "center" }}>
              <div style={{ fontSize: 20, fontWeight: 700, fontFamily: "monospace" }}><G color={s.c}>{s.v}</G></div>
              <div style={{ fontSize: 10, color: "#64748b", fontFamily: "monospace" }}>{s.l}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function Notes({ notes, onAdd, onDel }) {
  const [t, setT] = useState("");
  const add = () => { if (!t.trim()) return; onAdd({ id: Date.now(), text: t.trim(), date: new Date().toLocaleDateString(), author: "Team Lead" }); setT(""); };
  return (
    <div>
      <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
        <input value={t} onChange={e => setT(e.target.value)} placeholder="Add a team note..." onKeyDown={e => e.key === "Enter" && add()} style={{ flex: 1, padding: 10, background: "#12121f", border: "1px solid #1e293b", borderRadius: 6, color: "#e2e8f0", fontSize: 13 }} />
        <button onClick={add} style={{ padding: "10px 16px", background: "#06b6d4", border: "none", borderRadius: 6, color: "#0a0a14", cursor: "pointer", fontWeight: 600, fontSize: 12, fontFamily: "monospace" }}>POST</button>
      </div>
      {notes.length === 0 && <div style={{ textAlign: "center", padding: 32, color: "#334155", fontFamily: "monospace", fontSize: 12 }}>No notes yet.</div>}
      {notes.map(n => (
        <div key={n.id} style={{ background: "#12121f", border: "1px solid #1e293b", borderRadius: 8, padding: 12, marginBottom: 8 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
            <span style={{ fontSize: 11, color: "#06b6d4", fontFamily: "monospace" }}>{n.author} • {n.date}</span>
            <button onClick={() => onDel(n.id)} style={{ background: "none", border: "none", color: "#475569", cursor: "pointer", fontSize: 12 }}>✕</button>
          </div>
          <div style={{ fontSize: 13, color: "#cbd5e1", lineHeight: 1.5 }}>{n.text}</div>
        </div>
      ))}
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useState("live");
  const [data, setData] = useState(defData);
  const [loaded, setLoaded] = useState(false);
  const [filter, setFilter] = useState("all");
  const [expSop, setExpSop] = useState(0);
  const [expPlay, setExpPlay] = useState(null);

  useEffect(() => { (async () => { try { const r = await window.storage.get(SK); if (r?.value) setData(JSON.parse(r.value)); } catch {} setLoaded(true); })(); }, []);
  const save = useCallback(async d => { setData(d); try { await window.storage.set(SK, JSON.stringify(d)); } catch {} }, []);

  const moveTask = (id, s) => save({ ...data, tasks: data.tasks.map(t => t.id === id ? { ...t, status: s } : t) });
  const delTask = id => save({ ...data, tasks: data.tasks.filter(t => t.id !== id) });
  const addTask = t => save({ ...data, tasks: [...data.tasks, { ...t, id: data.nextId, status: "todo" }], nextId: data.nextId + 1 });
  const addNote = n => save({ ...data, notes: [n, ...data.notes] });
  const delNote = id => save({ ...data, notes: data.notes.filter(n => n.id !== id) });
  const addJournal = e => save({ ...data, journal: [e, ...data.journal] });
  const delJournal = id => save({ ...data, journal: data.journal.filter(j => j.id !== id) });
  const ft = filter === "all" ? data.tasks : data.tasks.filter(t => t.category === filter);

  const tabs = [
    { id: "live", label: "◉ LIVE" },
    { id: "qwen", label: "⬡ QWEN" },
    { id: "feed", label: "⟁ FEED" },
    { id: "schedule", label: "⏱ SCHEDULE" },
    { id: "architecture", label: "⬡ ARCHITECTURE" },
    { id: "tasks", label: "◈ TASKS" },
    { id: "playbooks", label: "▷ PLAYBOOKS" },
    { id: "guides", label: "⚡ GUIDES" },
    { id: "risk", label: "⛨ RISK" },
    { id: "emergency", label: "🔴 EMERGENCY" },
    { id: "journal", label: "📓 JOURNAL" },
    { id: "onboarding", label: "⊞ ONBOARDING" },
    { id: "notes", label: "✎ NOTES" },
  ];

  if (!loaded) return <div style={{ minHeight: "100vh", background: "#0a0a14", display: "flex", alignItems: "center", justifyContent: "center" }}><G>LOADING CODE ZERO...</G></div>;

  return (
    <div style={{ minHeight: "100vh", background: "#0a0a14", color: "#e2e8f0", fontFamily: "'Segoe UI', system-ui, sans-serif" }}>
      <style>{`@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}::selection{background:#06b6d440}*{box-sizing:border-box}`}</style>

      <div style={{ borderBottom: "1px solid #1e293b", padding: "16px 20px", background: "linear-gradient(180deg, #0f0f1a 0%, #0a0a14 100%)" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 12 }}>
          <div>
            <div style={{ fontSize: 20, fontWeight: 700, fontFamily: "monospace", letterSpacing: 3 }}><G>CODE ZERO</G> <span style={{ fontSize: 12, color: "#64748b", fontWeight: 400 }}>TEAM HUB</span></div>
            <div style={{ fontSize: 10, color: "#475569", fontFamily: "monospace", marginTop: 2 }}>RUST-POWERED LLM CRYPTO TRADING • 9-PHASE PIPELINE • v9.NOPU7</div>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}><Dot /> <span style={{ fontSize: 11, color: "#4ade80", fontFamily: "monospace" }}>PIPELINE ACTIVE</span></div>
        </div>
        <div style={{ display: "flex", gap: 4, marginTop: 16, flexWrap: "wrap" }}>
          {tabs.map(t => (
            <button key={t.id} onClick={() => setTab(t.id)} style={{ padding: "8px 14px", background: tab === t.id ? "#06b6d415" : "transparent", border: tab === t.id ? "1px solid #06b6d440" : "1px solid transparent", borderRadius: 6, color: tab === t.id ? "#06b6d4" : "#64748b", cursor: "pointer", fontSize: 11, fontFamily: "monospace", letterSpacing: 1, whiteSpace: "nowrap" }}>{t.label}</button>
          ))}
        </div>
      </div>

      <div style={{ padding: 20, maxWidth: 1100, margin: "0 auto" }}>
        {tab === "live" && <LiveDashboard />}
        {tab === "qwen" && <QwenChat />}
        {tab === "feed" && <LiveFeed />}
        {tab === "schedule" && <Schedule />}

        {tab === "architecture" && (
          <div>
            <div style={{ fontSize: 12, color: "#64748b", fontFamily: "monospace", marginBottom: 16 }}>AI MODEL ARCHITECTURE</div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: 12, marginBottom: 32 }}>
              {archModels.map((m, i) => (
                <div key={i} style={{ background: "#12121f", border: `1px solid ${m.color}25`, borderRadius: 8, padding: 16, borderTop: `3px solid ${m.color}` }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}><span style={{ fontSize: 15, fontWeight: 700, fontFamily: "monospace" }}><G color={m.color}>{m.name}</G></span>{m.color !== "#64748b" && <Dot color={m.color} />}</div>
                  <div style={{ fontSize: 9, color: m.color, fontFamily: "monospace", letterSpacing: 1, marginBottom: 4 }}>◆ {m.role}</div>
                  <div style={{ fontSize: 10, color: "#475569", fontFamily: "monospace", marginBottom: 10 }}>{m.hw}</div>
                  <div style={{ fontSize: 12, color: "#94a3b8", lineHeight: 1.5, marginBottom: m.tools?.length ? 10 : 0 }}>{m.desc}</div>
                  {m.tools?.length > 0 && (
                    <div>
                      <div style={{ fontSize: 9, color: m.color, fontFamily: "monospace", marginBottom: 6 }}>{m.tools.length} TOOLS ENABLED</div>
                      <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
                        {m.tools.map((t, j) => <span key={j} style={{ fontSize: 8, padding: "2px 5px", borderRadius: 3, background: `${m.color}10`, color: `${m.color}cc`, border: `1px solid ${m.color}20`, fontFamily: "monospace" }}>{t}</span>)}
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
            <div style={{ fontSize: 12, color: "#64748b", fontFamily: "monospace", marginBottom: 16 }}>DATA FLOW — 9-PHASE RUST PIPELINE</div>
            <div style={{ background: "#12121f", border: "1px solid #1e293b", borderRadius: 8, padding: 16 }}>
              <div style={{ fontSize: 10, color: "#475569", fontFamily: "monospace", marginBottom: 12 }}>cargo build --release • Target latency: 0.038ms • Kraken Exchange</div>
              {phases.map((ph, i) => (
                <div key={i} style={{ display: "flex", alignItems: "center", gap: 12, padding: "10px 0", borderBottom: i < 8 ? "1px solid #1e293b20" : "none" }}>
                  <div style={{ width: 32, height: 32, borderRadius: 6, background: "#06b6d410", border: "1px solid #06b6d430", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 13, fontWeight: 700, fontFamily: "monospace", color: "#06b6d4", flexShrink: 0 }}>{ph.p}</div>
                  <div><div style={{ fontSize: 13, fontWeight: 600, color: "#e2e8f0" }}>{ph.n}</div><div style={{ fontSize: 11, color: "#64748b" }}>{ph.d}</div></div>
                  {i < 8 && <div style={{ marginLeft: "auto", color: "#1e293b", fontSize: 16 }}>↓</div>}
                </div>
              ))}
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 12, marginTop: 24 }}>
              {[{ l: "Language", v: "RUST", c: "#f97316" }, { l: "Latency", v: "0.038ms", c: "#4ade80" }, { l: "Phases", v: "9", c: "#06b6d4" }, { l: "AI Models", v: "3+1 AVAIL", c: "#a855f7" }, { l: "Exchange", v: "KRAKEN", c: "#3b82f6" }, { l: "Location", v: "NUIQSUT, AK", c: "#64748b" }].map(m => (
                <div key={m.l} style={{ background: "#12121f", border: "1px solid #1e293b", borderRadius: 8, padding: 14, textAlign: "center" }}><div style={{ fontSize: 18, fontWeight: 700, fontFamily: "monospace" }}><G color={m.c}>{m.v}</G></div><div style={{ fontSize: 10, color: "#64748b", fontFamily: "monospace", marginTop: 4 }}>{m.l}</div></div>
              ))}
            </div>
          </div>
        )}

        {tab === "tasks" && (
          <div>
            <div style={{ display: "flex", gap: 6, marginBottom: 16, flexWrap: "wrap" }}>
              {["all", "development", "trading", "research", "ops"].map(f => (
                <button key={f} onClick={() => setFilter(f)} style={{ padding: "4px 12px", fontSize: 10, fontFamily: "monospace", textTransform: "uppercase", background: filter === f ? (f === "all" ? "#06b6d415" : `${cats[f]}15`) : "transparent", border: filter === f ? `1px solid ${f === "all" ? "#06b6d4" : cats[f]}40` : "1px solid #1e293b", color: filter === f ? (f === "all" ? "#06b6d4" : cats[f]) : "#64748b", borderRadius: 4, cursor: "pointer" }}>{f}</button>
              ))}
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 16 }}>
              {sts.map(s => (
                <div key={s}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}><div style={{ width: 8, height: 8, borderRadius: 2, background: stC[s] }} /><span style={{ fontSize: 11, fontFamily: "monospace", color: stC[s], letterSpacing: 1 }}>{stL[s]}</span><span style={{ fontSize: 10, color: "#475569", fontFamily: "monospace" }}>({ft.filter(t => t.status === s).length})</span></div>
                  {ft.filter(t => t.status === s).map(t => <TC key={t.id} task={t} onMove={moveTask} onDel={delTask} />)}
                  {s === "todo" && <ATF onAdd={addTask} tm={data.teamMembers} />}
                </div>
              ))}
            </div>
          </div>
        )}

        {tab === "playbooks" && (
          <div>
            <div style={{ fontSize: 12, color: "#64748b", fontFamily: "monospace", marginBottom: 16 }}>TRADING STRATEGIES — MODEL FLOW INTEGRATED</div>
            {playbooks.map((p, i) => (
              <div key={i} onClick={() => setExpPlay(expPlay === i ? null : i)} style={{ background: "#12121f", border: expPlay === i ? "1px solid #a855f740" : "1px solid #1e293b", borderRadius: 8, padding: 16, marginBottom: 12, cursor: "pointer" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}><div><div style={{ fontSize: 15, fontWeight: 600 }}><G color="#a855f7">{p.name}</G></div><div style={{ fontSize: 12, color: "#94a3b8", marginTop: 4 }}>{p.desc}</div></div><span style={{ color: "#475569", fontSize: 18 }}>{expPlay === i ? "▾" : "▸"}</span></div>
                {expPlay === i && (
                  <div style={{ marginTop: 16, borderTop: "1px solid #1e293b", paddingTop: 12 }}>
                    <div style={{ padding: 10, background: "#06b6d408", border: "1px solid #06b6d420", borderRadius: 6, marginBottom: 12 }}><div style={{ fontSize: 10, color: "#06b6d4", fontFamily: "monospace", marginBottom: 4 }}>⟁ MODEL FLOW</div><div style={{ fontSize: 12, color: "#94a3b8", fontFamily: "monospace" }}>{p.flow}</div></div>
                    <div style={{ fontSize: 11, color: "#64748b", fontFamily: "monospace", marginBottom: 8 }}>RULES</div>
                    {p.rules.map((r, j) => <div key={j} style={{ fontSize: 12, color: "#cbd5e1", padding: "6px 0", borderBottom: "1px solid #1e293b10", fontFamily: "monospace" }}><span style={{ color: "#06b6d4" }}>▹</span> {r}</div>)}
                    <div style={{ marginTop: 12, padding: 10, background: "#ff3b5c08", border: "1px solid #ff3b5c20", borderRadius: 6 }}><div style={{ fontSize: 10, color: "#ff3b5c", fontFamily: "monospace", marginBottom: 4 }}>⚠ RISK MANAGEMENT</div><div style={{ fontSize: 12, color: "#fca5a5", fontFamily: "monospace" }}>{p.risk}</div></div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {tab === "guides" && (
          <div>
            <div style={{ fontSize: 12, color: "#64748b", fontFamily: "monospace", marginBottom: 16 }}>STRATEGY GUIDES — SWAPS, ANALYSIS & THESIS BUILDING</div>
            <div style={{ background: "#12121f", border: "1px solid #4ade8030", borderRadius: 8, padding: 20, marginBottom: 16, borderTop: "3px solid #4ade80" }}>
              <div style={{ fontSize: 16, fontWeight: 700, marginBottom: 4 }}><G color="#4ade80">Gainer Coin Swaps & Routing</G></div>
              <div style={{ fontSize: 11, color: "#64748b", fontFamily: "monospace", marginBottom: 16 }}>HOW TO CAPTURE MOMENTUM ON GAINING COINS</div>
              <div style={{ fontSize: 13, color: "#94a3b8", lineHeight: 1.7, marginBottom: 16 }}>When a coin is gaining fast, the key is getting in and out with minimal slippage and maximum speed.</div>
              {[{ t: "CEX Direct (Kraken)", d: "Best for high-liquidity pairs. CODE ZERO connects here." }, { t: "DEX Aggregators (1inch, Jupiter)", d: "Scan multiple DEXs for best price path." }, { t: "Cross-Chain Routers (Symbiosis, LI.FI)", d: "Route through bridges + DEXs in one transaction." }, { t: "Direct DEX (Uniswap, PancakeSwap)", d: "Single-chain swaps." }].map((r, i) => (
                <div key={i} style={{ background: "#0a0a14", borderRadius: 6, padding: 12, marginBottom: 8, borderLeft: "2px solid #4ade8040" }}><div style={{ fontSize: 13, fontWeight: 600, color: "#e2e8f0" }}>{r.t}</div><div style={{ fontSize: 12, color: "#94a3b8", marginTop: 4 }}>{r.d}</div></div>
              ))}
            </div>
            <div style={{ background: "#12121f", border: "1px solid #06b6d430", borderRadius: 8, padding: 20, marginBottom: 16, borderTop: "3px solid #06b6d4" }}>
              <div style={{ fontSize: 16, fontWeight: 700, marginBottom: 4 }}><G>Market Analysis Methods</G></div>
              {[{ name: "Technical Analysis", color: "#06b6d4", items: ["Chart patterns, indicators (RSI, MACD, BB)", "Moving average crossovers", "Volume confirmation", "GPU-accelerated in Phase 3"] }, { name: "Fundamental Analysis", color: "#a855f7", items: ["On-chain metrics, tokenomics", "Developer activity, revenue/fees", "Team & backing evaluation"] }, { name: "Sentiment Analysis", color: "#4ade80", items: ["Fear & Greed, funding rates", "Social media volume, open interest", "NPU runs lightweight sentiment models"] }, { name: "Macro Analysis", color: "#f97316", items: ["Interest rates, DXY correlation", "Equity correlation, regulatory catalysts", "M2 money supply cycles"] }].map((s, i) => (
                <div key={i} style={{ marginBottom: 12 }}><div style={{ fontSize: 14, fontWeight: 600, marginBottom: 6 }}><G color={s.color}>{s.name}</G></div>{s.items.map((item, j) => <div key={j} style={{ fontSize: 12, color: "#cbd5e1", padding: "4px 0 4px 12px", borderLeft: `2px solid ${s.color}30` }}><span style={{ color: s.color }}>▹</span> {item}</div>)}</div>
              ))}
            </div>
            <div style={{ background: "#12121f", border: "1px solid #a855f730", borderRadius: 8, padding: 20, borderTop: "3px solid #a855f7" }}>
              <div style={{ fontSize: 16, fontWeight: 700, marginBottom: 12 }}><G color="#a855f7">Building a Trading Thesis</G></div>
              {[{ n: "1. NARRATIVE", d: "Why does this asset move?" }, { n: "2. CATALYST", d: "What triggers it?" }, { n: "3. EVIDENCE", d: "TA + FA + Sentiment alignment" }, { n: "4. RISK CASE", d: "What invalidates?" }, { n: "5. POSITION PLAN", d: "Entry, exit, size (1-2% risk)" }, { n: "6. TIME HORIZON", d: "Scalp / Swing / Position" }].map((s, i) => (
                <div key={i} style={{ background: "#0a0a14", borderRadius: 6, padding: 10, marginBottom: 6, borderLeft: "2px solid #a855f740" }}><span style={{ fontSize: 12, fontWeight: 700, color: "#a855f7", fontFamily: "monospace" }}>{s.n}</span> <span style={{ fontSize: 12, color: "#94a3b8" }}>— {s.d}</span></div>
              ))}
            </div>
          </div>
        )}

        {tab === "risk" && <><div style={{ fontSize: 12, color: "#64748b", fontFamily: "monospace", marginBottom: 16 }}>RISK MANAGEMENT DASHBOARD</div><RiskCalc /></>}
        {tab === "emergency" && <Emergency />}
        {tab === "journal" && <><div style={{ fontSize: 12, color: "#64748b", fontFamily: "monospace", marginBottom: 16 }}>LIVE TRADE JOURNAL</div><Journal entries={data.journal} onAdd={addJournal} onDel={delJournal} /></>}

        {tab === "onboarding" && (
          <div>
            <div style={{ fontSize: 12, color: "#64748b", fontFamily: "monospace", marginBottom: 16 }}>ONBOARDING & SOPs</div>
            <div style={{ background: "#12121f", border: "1px solid #1e293b", borderRadius: 8, padding: 16, marginBottom: 24 }}>
              <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 8 }}><G>Welcome to CODE ZERO</G></div>
              <div style={{ fontSize: 13, color: "#94a3b8", lineHeight: 1.6 }}>Rust-powered 9-phase crypto trading pipeline. GPU entry, NPU market watch, CPU exit. Kraken exchange. Sub-millisecond latency. Nuiqsut, Alaska.</div>
            </div>
            {sopSteps.map((s, i) => (
              <div key={i} onClick={() => setExpSop(expSop === i ? -1 : i)} style={{ background: "#12121f", border: expSop === i ? "1px solid #06b6d440" : "1px solid #1e293b", borderRadius: 8, padding: 14, marginBottom: 8, cursor: "pointer" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}><span style={{ fontSize: 14, fontWeight: 600, fontFamily: "monospace" }}><G>{s.title}</G></span><span style={{ color: "#475569" }}>{expSop === i ? "▾" : "▸"}</span></div>
                {expSop === i && <div style={{ marginTop: 12 }}>{s.items.map((item, j) => (
                  <div key={j} style={{ display: "flex", alignItems: "center", gap: 10, padding: "8px 0", borderBottom: "1px solid #1e293b10" }}><div style={{ width: 20, height: 20, borderRadius: 4, border: "1px solid #06b6d440", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 10, color: "#06b6d4", flexShrink: 0 }}>{j + 1}</div><span style={{ fontSize: 13, color: "#cbd5e1" }}>{item}</span></div>
                ))}</div>}
              </div>
            ))}
          </div>
        )}

        {tab === "notes" && <><div style={{ fontSize: 12, color: "#64748b", fontFamily: "monospace", marginBottom: 16 }}>TEAM NOTES</div><Notes notes={data.notes} onAdd={addNote} onDel={delNote} /></>}
      </div>

      <div style={{ borderTop: "1px solid #1e293b", padding: "12px 20px", textAlign: "center", marginTop: 40 }}><span style={{ fontSize: 10, color: "#334155", fontFamily: "monospace" }}>CODE ZERO TEAM HUB • RUST-POWERED • 9NOPU7 • NUIQSUT OPS</span></div>
    </div>
  );
}
