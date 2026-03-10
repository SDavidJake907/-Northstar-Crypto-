//! Regime State Machine — deterministic per-coin regime with hysteresis + N-of-M confirmation.
//! Replaces `config::infer_regime()` which had no dwell time, no confirmation buffer,
//! and no protection against oscillation in choppy markets.

use std::collections::VecDeque;

// ── Regime enum ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Regime {
    Unknown,
    Bullish,
    Sideways,
    Bearish,
    Volatile,
}

impl Regime {
    /// Convert to string for backward-compat with existing profile/gate code.
    pub fn as_str(&self) -> &'static str {
        match self {
            Regime::Unknown  => "sideways",  // safe fallback for downstream profile select
            Regime::Bullish  => "bullish",
            Regime::Sideways => "sideways",
            Regime::Bearish  => "bearish",
            Regime::Volatile => "volatile",
        }
    }

    /// True when entries should be blocked entirely.
    pub fn blocks_entries(&self) -> bool {
        matches!(self, Regime::Unknown | Regime::Bearish)
    }
}

impl std::fmt::Display for Regime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ── Inputs ────────────────────────────────────────────────────────────────────

pub struct RegimeInputs {
    pub rsi_s:      f64,   // RSI(7)
    pub rsi_m:      f64,   // RSI(14)
    pub rsi_l:      f64,   // RSI(28)
    pub ema_fast:   f64,   // EMA(9)
    pub ema_slow:   f64,   // EMA(55)
    pub atr_norm:   f64,   // ATR(14) / price
    pub vol_zscore: f64,   // volume z-score over 20 candles
    pub mtf_7d:     f64,   // 7-day trend %, -99.0 = missing
    pub candle_count: u32, // total candles seen (warm-up gate)
}

// ── Config constants ──────────────────────────────────────────────────────────

const WARMUP_CANDLES:        u32 = 20;
const CONFIRM_WINDOW:        u8  = 5;
const CONFIRM_REQUIRED:      u8  = 3;

const DWELL_MIN_BULLISH:     u32 = 8;
const DWELL_MIN_SIDEWAYS:    u32 = 5;
const DWELL_MIN_BEARISH:     u32 = 8;
const DWELL_MIN_VOLATILE:    u32 = 3;
const VOLATILE_EXIT_CANDLES: u32 = 30;  // auto-exit VOLATILE after N candles if ATR drops

const ATR_VOLATILE_ENTER:    f64 = 0.025;
const ATR_VOLATILE_EXIT:     f64 = 0.015;
const VOL_Z_VOLATILE:        f64 = 3.0;

const EMA_GAP_BULLISH_BASE:  f64 = 0.005;
const EMA_GAP_BULLISH_WEAK:  f64 = 0.002;  // when mtf_7d > +4%
const EMA_GAP_BULLISH_TIGHT: f64 = 0.010;  // when mtf_7d < -5%
const EMA_GAP_BEARISH:       f64 = -0.005;

// ── State Machine ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct RegimeStateMachine {
    pub state:         Regime,
    candidate:         Regime,
    candidate_count:   u8,
    dwell_count:       u32,
    volatile_dwell:    u32,
    history:           VecDeque<Regime>,
}

impl Default for RegimeStateMachine {
    fn default() -> Self {
        Self {
            state:           Regime::Unknown,
            candidate:       Regime::Unknown,
            candidate_count: 0,
            dwell_count:     0,
            volatile_dwell:  0,
            history:         VecDeque::with_capacity(CONFIRM_WINDOW as usize + 1),
        }
    }
}

impl RegimeStateMachine {
    pub fn new() -> Self { Self::default() }

    /// Update the state machine with new inputs. Returns current confirmed state.
    pub fn update(&mut self, i: &RegimeInputs) -> Regime {

        // ── Warm-up gate ─────────────────────────────────────────────
        // Bypass if EMA signals are already computed — features available = data is there.
        // Only enforce strict warmup when EMAs are missing (cold start with no history).
        let has_ema_data = i.ema_fast > 0.0 && i.ema_slow > 0.0;
        if i.candle_count < WARMUP_CANDLES && !has_ema_data {
            self.state = Regime::Unknown;
            return self.state;
        }

        // ── Raw classification ────────────────────────────────────────
        let raw = classify_raw(i);

        // ── VOLATILE override — instant, no dwell, no confirmation ───
        if raw == Regime::Volatile {
            if self.state != Regime::Volatile {
                tracing::info!("[REGIME-SM] → VOLATILE (atr_norm={:.4} vol_z={:.1})",
                    i.atr_norm, i.vol_zscore);
            }
            self.state           = Regime::Volatile;
            self.volatile_dwell  = 0;
            self.candidate       = Regime::Volatile;
            self.candidate_count = 0;
            self.dwell_count     = 0;
            return self.state;
        }

        // ── VOLATILE auto-exit ────────────────────────────────────────
        if self.state == Regime::Volatile {
            self.volatile_dwell += 1;
            let atr_settled = i.atr_norm < ATR_VOLATILE_EXIT;
            let timed_out   = self.volatile_dwell >= VOLATILE_EXIT_CANDLES;
            if !atr_settled && !timed_out {
                return self.state;
            }
            tracing::info!("[REGIME-SM] VOLATILE exit (atr={:.4} dwell={})",
                i.atr_norm, self.volatile_dwell);
            self.volatile_dwell  = 0;
            // fall through to normal classification
        }

        // ── Accumulate confirmation buffer ────────────────────────────
        self.history.push_back(raw);
        if self.history.len() > CONFIRM_WINDOW as usize {
            self.history.pop_front();
        }

        if raw == self.candidate {
            self.candidate_count = (self.candidate_count + 1).min(CONFIRM_WINDOW);
        } else {
            self.candidate       = raw;
            self.candidate_count = 1;
        }

        self.dwell_count += 1;

        // ── Transition check ──────────────────────────────────────────
        let dwell_min = match self.state {
            Regime::Unknown  => 0,
            Regime::Bullish  => DWELL_MIN_BULLISH,
            Regime::Sideways => DWELL_MIN_SIDEWAYS,
            Regime::Bearish  => DWELL_MIN_BEARISH,
            Regime::Volatile => 0,
        };

        let dwell_ok   = self.dwell_count >= dwell_min;
        let confirm_ok = self.candidate_count >= CONFIRM_REQUIRED;

        if confirm_ok && dwell_ok && self.candidate != self.state {
            tracing::info!(
                "[REGIME-SM] {:?} → {:?} (dwell={} confirm={}/{})",
                self.state, self.candidate,
                self.dwell_count, self.candidate_count, CONFIRM_WINDOW
            );
            self.state           = self.candidate;
            self.dwell_count     = 0;
            self.candidate_count = 0;
        }

        self.state
    }
}

// ── Raw per-candle classifier ─────────────────────────────────────────────────

fn classify_raw(i: &RegimeInputs) -> Regime {

    // VOLATILE first — instant override
    if i.atr_norm > ATR_VOLATILE_ENTER || i.vol_zscore > VOL_Z_VOLATILE {
        return Regime::Volatile;
    }

    let ema_gap = if i.ema_slow > 0.0 {
        (i.ema_fast - i.ema_slow) / i.ema_slow
    } else {
        0.0
    };

    let rsi_avg    = (i.rsi_s + i.rsi_m) / 2.0;
    let rsi_anchor = i.rsi_l;

    // MTF-adjusted bullish threshold
    let bullish_gap_thresh = if i.mtf_7d > 4.0 {
        EMA_GAP_BULLISH_WEAK   // easier to be bullish with strong weekly trend
    } else if i.mtf_7d < -5.0 {
        EMA_GAP_BULLISH_TIGHT  // harder to be bullish against weekly downtrend
    } else {
        EMA_GAP_BULLISH_BASE
    };

    // BULLISH
    if ema_gap > bullish_gap_thresh
        && rsi_avg >= 52.0
        && rsi_anchor >= 48.0
        && i.mtf_7d > -5.0
    {
        return Regime::Bullish;
    }

    // BEARISH
    if ema_gap < EMA_GAP_BEARISH
        && rsi_avg <= 48.0
        && rsi_anchor <= 52.0
        && i.mtf_7d < -5.0
    {
        return Regime::Bearish;
    }

    // Default: SIDEWAYS
    Regime::Sideways
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Build RegimeInputs from a features_map JSON value.
pub fn inputs_from_feats(feats: &serde_json::Value, mtf_7d: f64, candle_count: u32) -> RegimeInputs {
    let f = |k: &str, d: f64| feats.get(k).and_then(|v| v.as_f64()).unwrap_or(d);
    let price = f("price", 1.0).max(0.0001);

    RegimeInputs {
        rsi_s:        f("rsi",    50.0),   // RSI(14) used as proxy for short
        rsi_m:        f("rsi",    50.0),   // same field — extend if RSI(7) added
        rsi_l:        f("rsi",    50.0),   // same field — extend if RSI(28) added
        ema_fast:     f("ema9",   0.0),
        ema_slow:     f("ema55",  0.0),
        atr_norm:     (f("atr", 0.0) / price).max(0.0),
        vol_zscore:   f("vol_ratio", 1.0) - 1.0,  // proxy: vol_ratio-1 ≈ z-score direction
        mtf_7d,
        candle_count,
    }
}
