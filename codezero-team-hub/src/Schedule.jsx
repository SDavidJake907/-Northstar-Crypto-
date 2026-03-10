import { useState } from "react";
import { G, Bdg } from "./ui.jsx";

export default function Schedule() {
  const [expBlock, setExpBlock] = useState(null);
  const toggle = i => setExpBlock(expBlock === i ? null : i);

  const daily = [
    { time: "05:30", window: "05:30 \u2013 09:30 AKST", name: "HIGH VOL \u2014 FULL SIZE", color: "#ff3b5c", icon: "\u25c6",
      params: { margin: "\u22650.15", tools: "2 rounds", atr: "1.0\u00d7", mode: "Full size", concurrent: 3 },
      tasks: [
      { do: "Open LIVE tab \u2014 verify all 4 status dots are GREEN (engine + 3 brains)", where: "LIVE tab \u2192 AI Team Status" },
      { do: "Check WORKLOAD BALANCE \u2014 GPU ~35-40% (context), NPU ~30%, Math ~25%", where: "LIVE tab \u2192 Workload Balance" },
      { do: "Verify NPU pre-scan active \u2014 check FEED for NPU REJECT events", where: "FEED tab \u2192 filter events" },
      { do: "Confirm deterministic exits running \u2014 ATR trailing stops active", where: "LIVE tab \u2192 60s Meters" },
      { do: "Check engine status: tick advancing, heartbeat < 30s", where: "LIVE tab \u2192 Engine Status" },
      { do: "Review overnight circuit breaker \u2014 daily PnL reset, defensive OFF", where: "LIVE tab \u2192 Circuit Breaker" },
      { do: "Confirm hardware guard level GREEN, GPU temp < 70\u00b0C", where: "LIVE tab \u2192 Hardware Guard" },
      { do: "FEED tab on screen \u2014 watch for BUY signals with confidence > 70%", where: "FEED tab \u2192 filter SIGNAL" },
      { do: "Monitor open positions \u2014 full size, 3 concurrent max", where: "LIVE tab \u2192 Open Positions" },
      { do: "Quick AI_1 ping: 'Market overview' \u2014 verify context supervisor responds", where: "QWEN tab \u2192 quick prompt" },
    ]},
    { time: "09:30", window: "09:30 \u2013 12:00 AKST", name: "MIDDAY \u2014 CONSERVATIVE", color: "#f0b429", icon: "\u2b21",
      params: { margin: "\u22650.17", tools: "2 rounds", atr: "0.8\u00d7 (tighter)", mode: "Reduced concurrency", concurrent: 2 },
      tasks: [
      { do: "Schedule auto-tightens: margin gate 0.17, ATR \u00d70.8, max 2 concurrent", where: "Automated via SCHEDULE_ENABLED" },
      { do: "Watch circuit breaker daily PnL \u2014 if approaching -3% alert team", where: "LIVE tab \u2192 Circuit Breaker" },
      { do: "Screenshot LIVE tab \u2014 balances, PnL, positions for journal", where: "LIVE tab" },
      { do: "Review top reject reasons \u2014 margin gate should be tighter now", where: "LIVE tab \u2192 Top Reject Reasons" },
      { do: "Check GPU mem + temp trend \u2014 any thermal throttling?", where: "LIVE tab \u2192 Hardware Guard" },
      { do: "Ask AI_1: 'Portfolio analysis' \u2014 get context supervisor take on positions", where: "QWEN tab \u2192 quick prompt" },
    ]},
    { time: "14:00", window: "14:00 \u2013 20:00 AKST", name: "AFTERNOON \u2014 MINIMAL", color: "#4ade80", icon: "\u25b7",
      params: { margin: "\u22650.18", tools: "1 round", atr: "0.8\u00d7", mode: "Fewer entries, reduced L2", concurrent: 2 },
      tasks: [
      { do: "Schedule reduces: margin 0.18, 1 tool round, L2 aggression halved", where: "Automated via SCHEDULE_ENABLED" },
      { do: "Lighter monitoring \u2014 check LIVE tab every 15-30 min", where: "LIVE tab" },
      { do: "Watch for watcher.alert Critical severity", where: "FEED tab \u2192 filter ALERT" },
      { do: "Monitor AI latency P95 \u2014 should stay under 5000ms", where: "LIVE tab \u2192 Hardware Guard" },
      { do: "Record daily PnL + total trades in JOURNAL tab", where: "JOURNAL tab" },
      { do: "Check holdings \u2014 are overnight positions acceptable size?", where: "LIVE tab \u2192 Holdings" },
    ]},
    { time: "20:00", window: "20:00 \u2013 05:30 AKST", name: "OVERNIGHT \u2014 CONSERVATIVE", color: "#64748b", icon: "\u25c7",
      params: { margin: "\u22650.20", tools: "1 round", atr: "0.8\u00d7", mode: "Conservative, fewer positions", concurrent: 1 },
      tasks: [
      { do: "Schedule most conservative: margin 0.20, 1 round, 1 concurrent only", where: "Automated via SCHEDULE_ENABLED" },
      { do: "Engine runs autonomously \u2014 deterministic exits protect positions", where: "Automated" },
      { do: "Circuit breaker + 3-loss cooldown protect against runaway losses", where: "Automated" },
      { do: "Optional: check LIVE tab once before bed, once on wake", where: "LIVE tab" },
    ]},
  ];

  const weekly = [
    { day: "Monday", color: "#06b6d4", tasks: ["Review weekend market moves in AI_1 chat", "Check hardware guard sample history \u2014 any spikes?", "Review previous week's journal entries for patterns", "Confirm all 3 brains healthy after weekend"] },
    { day: "Wednesday", color: "#a855f7", tasks: ["Mid-week PnL review \u2014 are we on track?", "Review workload balance across 3 brains \u2014 is GPU still overloaded?", "Ask AI_1: analyze top reject reasons this week", "Check if any coins consistently rejected by NPU pre-scan \u2192 investigate", "Review position sizing \u2014 is CVaR optimizer behaving?", "Verify exit brain making smart decisions \u2014 compare AI vs math-only exits"] },
    { day: "Friday", color: "#4ade80", tasks: ["Weekly performance journal entry (tag: LESSON)", "Screenshot full LIVE dashboard for records", "Review all INCIDENT-tagged journal entries", "Decide weekend risk posture \u2014 reduce exposure?", "Check for Rust engine updates (cargo update)"] },
  ];

  const monthly = [
    { name: "Performance Deep Dive", color: "#06b6d4", tasks: ["Calculate total PnL, win rate, avg hold time from journal", "Compare actual vs expected from playbook parameters", "Identify best/worst performing coins and timeframes", "Review AI_1 entry confidence accuracy (high conf = win?)"] },
    { name: "Model Health Audit", color: "#4ade80", tasks: ["Check AI_1 response quality \u2014 is it hallucinating?", "Review AI_2 exit timing \u2014 too early? too late?", "Verify AI_3 sentiment scores correlate with price moves", "Test all 8 entry tools + 7 exit tools via QWEN chat", "Check model memory (nemo_memory) for accuracy"] },
    { name: "Infrastructure Review", color: "#a855f7", tasks: ["GPU VRAM utilization trend \u2014 approaching 16GB limit?", "CPU thermal history \u2014 any throttling events?", "Disk space check \u2014 heartbeat + log files growing?", "Review Kraken API rate limit hits (REST throttle)", "Test emergency kill switch procedure (paper mode)"] },
    { name: "Strategy Calibration", color: "#f0b429", tasks: ["Review margin gate (0.15) \u2014 too tight? too loose?", "Analyze lane distribution \u2014 L1/L2/L3 balance healthy?", "Check greed index accuracy vs actual market regimes", "Update playbook parameters if data supports changes", "Backtest any proposed parameter changes before live"] },
  ];

  const cardStyle = { background: "#12121f", border: "1px solid #1e293b", borderRadius: 8 };

  return (
    <div>
      <div style={{ background: "#06b6d408", border: "1px solid #06b6d420", borderRadius: 8, padding: 16, marginBottom: 24 }}>
        <div style={{ fontSize: 14, fontWeight: 700, fontFamily: "monospace", marginBottom: 4 }}><G>CODE ZERO OPS SCHEDULE</G></div>
        <div style={{ fontSize: 12, color: "#94a3b8", lineHeight: 1.6 }}>
          Daily operating rhythm for the trading engine. All times Alaska Time (AKST). The engine runs 24/7 autonomously \u2014 this schedule is for <span style={{ color: "#06b6d4" }}>human monitoring, review, and intervention</span>. Every task maps to a specific tab in this hub.
        </div>
      </div>

      <div style={{ fontSize: 12, color: "#64748b", fontFamily: "monospace", marginBottom: 12 }}>DAILY SCHEDULE</div>
      {daily.map((block, i) => (
        <div key={i} onClick={() => toggle(i)} style={{ ...cardStyle, marginBottom: 8, cursor: "pointer", borderLeft: `4px solid ${block.color}`, overflow: "hidden" }}>
          <div style={{ padding: "12px 16px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <span style={{ fontSize: 14, color: block.color }}>{block.icon}</span>
              <div>
                <div style={{ fontSize: 13, fontWeight: 600 }}><G color={block.color}>{block.name}</G></div>
                <div style={{ fontSize: 10, color: "#475569", fontFamily: "monospace", marginTop: 2 }}>{block.window} AKST</div>
              </div>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <Bdg color={block.color}>{block.tasks.length} tasks</Bdg>
              <span style={{ color: "#475569", fontSize: 14 }}>{expBlock === i ? "\u25be" : "\u25b8"}</span>
            </div>
          </div>
          {expBlock === i && (
            <div style={{ padding: "0 16px 14px", borderTop: "1px solid #1e293b" }}>
              {block.params && (
                <div style={{ display: "flex", gap: 8, flexWrap: "wrap", padding: "10px 0", borderBottom: "1px solid #1e293b30" }}>
                  {Object.entries(block.params).map(([k, v]) => (
                    <span key={k} style={{ fontSize: 10, fontFamily: "monospace", padding: "2px 8px", background: `${block.color}12`, border: `1px solid ${block.color}25`, borderRadius: 4, color: block.color }}>
                      {k}: {v}
                    </span>
                  ))}
                </div>
              )}
              {block.tasks.map((t, j) => (
                <div key={j} style={{ display: "flex", gap: 10, padding: "10px 0", borderBottom: j < block.tasks.length - 1 ? "1px solid #1e293b15" : "none" }}>
                  <div style={{ width: 22, height: 22, borderRadius: 4, background: `${block.color}15`, border: `1px solid ${block.color}30`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 10, fontWeight: 700, color: block.color, flexShrink: 0, fontFamily: "monospace" }}>{j + 1}</div>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: 12, color: "#cbd5e1", lineHeight: 1.5 }}>{t.do}</div>
                    <div style={{ fontSize: 10, color: "#475569", fontFamily: "monospace", marginTop: 2 }}>\u2192 {t.where}</div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      ))}

      <div style={{ fontSize: 12, color: "#64748b", fontFamily: "monospace", marginBottom: 12, marginTop: 32 }}>WEEKLY CHECKPOINTS</div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 12, marginBottom: 32 }}>
        {weekly.map((w, i) => (
          <div key={i} style={{ ...cardStyle, padding: 16, borderTop: `3px solid ${w.color}` }}>
            <div style={{ fontSize: 14, fontWeight: 700, fontFamily: "monospace", marginBottom: 10 }}><G color={w.color}>{w.day}</G></div>
            {w.tasks.map((t, j) => (
              <div key={j} style={{ fontSize: 12, color: "#cbd5e1", padding: "6px 0", borderBottom: j < w.tasks.length - 1 ? "1px solid #1e293b15" : "none", fontFamily: "monospace" }}>
                <span style={{ color: w.color }}>{'▹'}</span> {t}
              </div>
            ))}
          </div>
        ))}
      </div>

      <div style={{ fontSize: 12, color: "#64748b", fontFamily: "monospace", marginBottom: 12 }}>MONTHLY REVIEWS</div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: 12, marginBottom: 32 }}>
        {monthly.map((m, i) => (
          <div key={i} style={{ ...cardStyle, padding: 16, borderTop: `3px solid ${m.color}` }}>
            <div style={{ fontSize: 13, fontWeight: 700, fontFamily: "monospace", marginBottom: 10 }}><G color={m.color}>{m.name}</G></div>
            {m.tasks.map((t, j) => (
              <div key={j} style={{ fontSize: 12, color: "#cbd5e1", padding: "5px 0", borderBottom: j < m.tasks.length - 1 ? "1px solid #1e293b15" : "none" }}>
                <span style={{ color: m.color }}>{'▹'}</span> {t}
              </div>
            ))}
          </div>
        ))}
      </div>

      <div style={{ ...cardStyle, padding: 16, borderLeft: "4px solid #ff3b5c" }}>
        <div style={{ fontSize: 12, color: "#ff3b5c", fontFamily: "monospace", marginBottom: 8 }}>\u26a0 CRITICAL RULES</div>
        {[
          "If ANY brain dot is RED on boot \u2014 do NOT trade until resolved",
          "If daily PnL hits -5% \u2014 circuit breaker should auto-stop. If it doesn't, hit KILL SWITCH",
          "If GPU temp > 85\u00b0C \u2014 hardware guard should throttle. If it doesn't, pause manually",
          "If watchdog shows FAIL + block_entries \u2014 engine is self-protecting. Investigate before clearing",
          "Never skip the EOD journal entry \u2014 it's how we learn and improve",
          "If AI_1 gives nonsensical answers \u2014 check GPU mem (VRAM leak?), restart model if needed",
          "3 consecutive losing days \u2014 mandatory strategy review before next session",
        ].map((r, i) => (
          <div key={i} style={{ fontSize: 12, color: "#fca5a5", padding: "6px 0", borderBottom: i < 6 ? "1px solid #ff3b5c10" : "none", fontFamily: "monospace", lineHeight: 1.5 }}>
            <span style={{ color: "#ff3b5c" }}>\u25cf</span> {r}
          </div>
        ))}
      </div>
    </div>
  );
}
