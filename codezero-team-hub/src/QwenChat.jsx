import { useState, useEffect, useRef } from "react";
import { G } from "./ui.jsx";

const BRAINS = {
  entry: {
    name: "AI_1", label: "CONTEXT", hw: "GPU :8081 (~35-40%)", color: "#4ade80", endpoint: "/api/chat",
    sys: "You are AI_1, the context supervisor brain for CODE ZERO \u2014 a Rust-powered crypto trading engine on Kraken. Quant scores make primary entry decisions. You provide regime context, risk flags, and confidence weight adjustments. You have 8 tools: get_coin_features, get_trade_history, get_market_context, get_ai_memory, get_correlated_coins, get_engine_status, get_top_movers, get_open_positions. Analyze markets, provide context, flag risks. Be direct, use data.",
    quicks: ["Market overview", "What would you buy right now?", "Portfolio analysis", "Top movers right now?"],
  },
  exit: {
    name: "AI_2", label: "ADVISOR", hw: "GPU :8082 (~5-10%)", color: "#a855f7", endpoint: "/api/exit",
    sys: "You are AI_2, the exit advisor brain for CODE ZERO \u2014 a Rust-powered crypto trading engine on Kraken. Exit decisions are deterministic (ATR-based stops). You provide advisory analysis only \u2014 your opinion is logged but does not execute trades. You have 7 tools: get_coin_features, get_trade_history, get_market_context, get_ai_memory, get_correlated_coins, get_engine_status, get_top_movers. Think about risk, analyze positions, but know the math stack has exit authority.",
    quicks: ["Review open positions", "Any exits overdue?", "Which position is weakest?", "Risk check all holdings"],
  },
  atlas: {
    name: "AI_3", label: "GATE", hw: "NPU :8083 (~30%)", color: "#f0b429", endpoint: "/api/atlas/chat",
    sys: "You are AI_3, the NPU pre-scan gate and sentiment brain for CODE ZERO. You run AI_3 Scanner on Intel NPU for fast candidate pre-screening (200ms) and sentiment model for news headline sentiment scoring. You filter out junk candidates before they reach the GPU brain, saving compute. You also score news sentiment from -1.0 (bearish) to +1.0 (bullish). Be concise and data-driven.",
    quicks: ["Sentiment for BTC", "Batch analyze top movers", "What are you rejecting?", "News sentiment summary"],
  },
};

export default function QwenChat() {
  const [brain, setBrain] = useState("entry");
  const [chats, setChats] = useState({ entry: [], exit: [], atlas: [] });
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState(null);
  const endRef = useRef(null);

  const b = BRAINS[brain];
  const msgs = chats[brain];

  const send = async (text) => {
    if (!text.trim() || loading) return;
    const userMsg = { role: "user", content: text, brain };
    const next = [...msgs, userMsg];
    setChats(prev => ({ ...prev, [brain]: next }));
    setInput("");
    setLoading(true);
    try {
      const t0 = performance.now();
      const r = await fetch(b.endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: "qwen", messages: [{ role: "system", content: b.sys }, ...next.map(m => ({ role: m.role, content: m.content }))], max_tokens: 2048 }),
      });
      const d = await r.json();
      const ms = Math.round(performance.now() - t0);
      const reply = d.choices?.[0]?.message?.content || "No response";
      setChats(prev => ({ ...prev, [brain]: [...prev[brain], { role: "assistant", content: reply, brain }] }));
      setStats({ prompt: d.usage?.prompt_tokens, completion: d.usage?.completion_tokens, total: d.usage?.total_tokens, ms, brain: b.name });
    } catch (e) {
      setChats(prev => ({ ...prev, [brain]: [...prev[brain], { role: "assistant", content: `Error: ${e.message}`, brain }] }));
    }
    setLoading(false);
  };

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [chats, brain]);

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16, flexWrap: "wrap", gap: 8 }}>
        <div style={{ fontSize: 12, color: "#64748b", fontFamily: "monospace" }}>AI TEAM CHAT</div>
        <div style={{ display: "flex", gap: 4 }}>
          {Object.entries(BRAINS).map(([key, br]) => (
            <button key={key} onClick={() => setBrain(key)} style={{ padding: "6px 14px", background: brain === key ? `${br.color}18` : "transparent", border: brain === key ? `1px solid ${br.color}50` : "1px solid #1e293b", borderRadius: 6, color: brain === key ? br.color : "#64748b", cursor: "pointer", fontSize: 11, fontFamily: "monospace", fontWeight: brain === key ? 700 : 400 }}>
              <span style={{ display: "inline-block", width: 7, height: 7, borderRadius: "50%", background: br.color, marginRight: 6, boxShadow: brain === key ? `0 0 6px ${br.color}` : "none" }} />
              {br.name} <span style={{ opacity: 0.6 }}>{br.label}</span>
            </button>
          ))}
        </div>
      </div>

      <div style={{ background: `${b.color}06`, border: `1px solid ${b.color}20`, borderRadius: 8, padding: 12, marginBottom: 12, display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 8 }}>
        <div>
          <span style={{ fontSize: 13, fontWeight: 700, fontFamily: "monospace", color: b.color }}>{b.name}</span>
          <span style={{ fontSize: 11, color: "#475569", fontFamily: "monospace", marginLeft: 8 }}>{b.hw}</span>
        </div>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          {b.quicks.map((q, i) => (
            <button key={i} onClick={() => send(q)} disabled={loading} style={{ padding: "4px 10px", background: `${b.color}10`, border: `1px solid ${b.color}30`, borderRadius: 4, color: b.color, cursor: loading ? "not-allowed" : "pointer", fontSize: 10, fontFamily: "monospace" }}>{q}</button>
          ))}
        </div>
      </div>

      <div style={{ background: "#12121f", border: "1px solid #1e293b", borderRadius: 8, padding: 16, minHeight: 320, maxHeight: 520, overflowY: "auto", marginBottom: 12 }}>
        {msgs.length === 0 && <div style={{ textAlign: "center", padding: 40, color: "#334155", fontFamily: "monospace", fontSize: 12 }}>Send a message to {b.name}...</div>}
        {msgs.map((m, i) => {
          const mb = BRAINS[m.brain] || b;
          const isUser = m.role === "user";
          return (
            <div key={i} style={{ marginBottom: 12, padding: 10, background: isUser ? "#06b6d408" : `${mb.color}08`, borderRadius: 6, borderLeft: `3px solid ${isUser ? "#06b6d4" : mb.color}` }}>
              <div style={{ fontSize: 9, fontFamily: "monospace", marginBottom: 4, textTransform: "uppercase", color: isUser ? "#06b6d4" : mb.color }}>
                {isUser ? "You" : mb.name} {!isUser && <span style={{ opacity: 0.5 }}>{mb.label}</span>}
              </div>
              <div style={{ fontSize: 13, color: "#e2e8f0", lineHeight: 1.6, whiteSpace: "pre-wrap" }}>{m.content}</div>
            </div>
          );
        })}
        {loading && <div style={{ padding: 10, color: b.color, fontFamily: "monospace", fontSize: 12, animation: "pulse 1.5s infinite" }}>{b.name} is thinking...</div>}
        <div ref={endRef} />
      </div>

      {stats && (
        <div style={{ display: "flex", gap: 12, marginBottom: 12, fontSize: 10, fontFamily: "monospace", color: "#475569", flexWrap: "wrap" }}>
          <span style={{ color: BRAINS[Object.keys(BRAINS).find(k => BRAINS[k].name === stats.brain)]?.color || "#475569" }}>{stats.brain}</span>
          <span>Prompt: {stats.prompt || "\u2014"}</span>
          <span>Completion: {stats.completion || "\u2014"}</span>
          <span>Total: {stats.total || "\u2014"} tokens</span>
          <span>Latency: {stats.ms}ms</span>
        </div>
      )}

      <div style={{ display: "flex", gap: 8 }}>
        <input value={input} onChange={e => setInput(e.target.value)} onKeyDown={e => e.key === "Enter" && send(input)} placeholder={`Ask ${b.name} anything...`} disabled={loading} style={{ flex: 1, padding: 10, background: "#12121f", border: "1px solid #1e293b", borderRadius: 6, color: "#e2e8f0", fontSize: 13, fontFamily: "monospace" }} />
        <button onClick={() => send(input)} disabled={loading} style={{ padding: "10px 20px", background: loading ? "#1e293b" : b.color, border: "none", borderRadius: 6, color: "#0a0a14", cursor: loading ? "not-allowed" : "pointer", fontWeight: 600, fontSize: 12, fontFamily: "monospace" }}>SEND</button>
      </div>
      <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
        {msgs.length > 0 && (
          <button onClick={() => { setChats(prev => ({ ...prev, [brain]: [] })); setStats(null); }} style={{ padding: "4px 12px", background: "transparent", border: "1px solid #1e293b", borderRadius: 4, color: "#64748b", cursor: "pointer", fontSize: 10, fontFamily: "monospace" }}>Clear {b.label} chat</button>
        )}
        {(chats.entry.length > 0 || chats.exit.length > 0 || chats.atlas.length > 0) && (
          <button onClick={() => { setChats({ entry: [], exit: [], atlas: [] }); setStats(null); }} style={{ padding: "4px 12px", background: "transparent", border: "1px solid #1e293b", borderRadius: 4, color: "#64748b", cursor: "pointer", fontSize: 10, fontFamily: "monospace" }}>Clear all</button>
        )}
      </div>
    </div>
  );
}
