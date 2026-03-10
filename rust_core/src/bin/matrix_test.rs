/// Matrix Signal Test — compact data stream like WebSocket feed
/// Pure numbers and letters flowing through. No JSON bloat.

fn main() {
    // Simulated live coin data — exactly what the engine computes
    let coins = vec![
        ("BTC", "TREND",    0.62, 0.41, 0.25, 55, 0.23,  67723.1),
        ("ETH", "SIDEWAYS", 0.51, 0.71, 0.18, 47, -0.15, 1970.6),
        ("SOL", "MEANREV",  0.38, 0.55, 0.30, 32, 0.45,  84.5),
        ("ADA", "TREND",    0.58, 0.39, 0.22, 36, 0.73,  0.283),
        ("ATOM","NOISE",    0.49, 0.88, 0.10, 50, -0.02, 2.35),
    ];

    println!("═══ MATRIX SIGNAL STREAM ═══");
    println!();

    // Stream format: SYM|REGIME|H|E|K|RSI|IMB|PRICE
    // This is ALL Qwen needs. 8 fields. ~40 chars per coin.
    for (sym, regime, h, e, k, rsi, imb, price) in &coins {
        // Vertical waterfall — each signal drops down
        for c in sym.chars() { println!("  {}", c); }
        println!("  │");

        // Regime as single letter: T=Trend M=MeanRev N=Noise S=Sideways
        let r = match *regime {
            "TREND" => 'T',
            "MEANREV" => 'M',
            "NOISE" => 'N',
            _ => 'S',
        };
        println!("  {}", r);
        println!("  │");

        // Core signals — pure numbers falling
        for c in format!("{:.2}", h).chars() { println!("  {}", c); }
        println!("  │");
        for c in format!("{:.2}", e).chars() { println!("  {}", c); }
        println!("  │");
        for c in format!("{:.2}", k).chars() { println!("  {}", c); }
        println!("  │");
        for c in format!("{}", rsi).chars() { println!("  {}", c); }
        println!("  │");
        for c in format!("{:+.2}", imb).chars() { println!("  {}", c); }
        println!();

        // Compact one-liner (what Qwen actually receives)
        println!("  → {}|{}|H{}|E{}|K{}|R{}|I{:+.2}|${:.0}",
            sym, r, h, e, k, rsi, imb, price);
        println!();
    }

    println!("═══ FULL PROMPT TO QWEN ═══");
    println!();

    // This is the ENTIRE prompt — replaces 2000+ chars with ~200
    let mut compact = String::new();
    compact.push_str("REGIME:SIDEWAYS CASH:$38 EQ:$75\n");
    for (sym, regime, h, e, k, rsi, imb, price) in &coins {
        let r = match *regime {
            "TREND" => 'T', "MEANREV" => 'M', "NOISE" => 'N', _ => 'S',
        };
        compact.push_str(&format!(
            "{}|{}|H{:.2}|E{:.2}|K{:.2}|R{}|I{:+.2}|${:.0}\n",
            sym, r, h, e, k, rsi, imb, price
        ));
    }
    compact.push_str("ACT? SYM|BUY/SELL/HOLD|CONF|USD\n");

    println!("{}", compact);
    println!("═══ {} chars vs ~2000 chars current ═══", compact.len());
    println!("═══ ~5x fewer tokens = ~2-3x faster inference ═══");
}
