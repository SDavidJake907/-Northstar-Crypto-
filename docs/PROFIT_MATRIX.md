# CODE ZERO — Profit Matrix v1.0

The Profit Matrix defines the exact behavior for each lane.
Rust enforces all values. AI reads lane assignment and respects the contract.

---

## Lane Overview

| Lane | Type | Coins | Entry Signal | Hold Philosophy |
|---|---|---|---|---|
| **L1** | Momentum Breakout | BTC, ETH, large caps | Price > EMA21, EMA9 > EMA21, vol ≥ 1.1x | Let it run — trend has legs |
| **L2** | Compression / Mean Reversion | Mid-caps sideways | EMAs braided, BB tight, RSI 40-60 | Quick in/out — snap back to mean |
| **L3** | Trend Following | Most alts | Price > EMA50, EMA21 > EMA50, RSI 40-70 | Patient — confirmed trend |
| **L4** | Meme Pump / Scalp | DOGE, PEPE, WIF, new coins | Vol spike OR book pressure OR any activity | HARVEST BURST — fast exit |

---

## L1 — Momentum Breakout

```
Entry trigger:   price > EMA21 AND EMA9 > EMA21 AND mom > 0 AND vol ≥ 1.1x AND trend ≥ 1
TP:              cfg_tp = 2.0%
SL:              ATR × 2.0
ATR trail mult:  2.0x (loosest — let momentum run)
Min hold:        30 min (MTF=1) / 3h (MTF=2) / 12h (MTF=3)
Position size:   Kelly-scaled, $15 max
AI threshold:    quant ≥ 0.60 AND confidence ≥ 0.65
Time contract:   min=1800s max=43200s reeval=600s
Best in:         Bullish / Trending regime
```

---

## L2 — Compression / Mean Reversion

```
Entry trigger:   EMAs braided + BB width < 0.06 + RSI 40-60 + neutral flow
TP:              cfg_tp = 2.0%
SL:              ATR × 1.3 (tightest for non-meme)
ATR trail mult:  1.3x
Min hold:        30 min
Position size:   Kelly-scaled, $15 max
AI threshold:    quant ≥ 0.55
Time contract:   min=120s max=3600s reeval=120s
Best in:         Sideways / low volatility regime
Note:            Enter as LIMIT order (not market) — price must come to us
```

---

## L3 — Trend Following

```
Entry trigger:   price > EMA50 AND EMA21 > EMA50 AND RSI 40-70 AND trend ≥ 0
TP:              cfg_tp = 2.0%
SL:              ATR × 1.5
ATR trail mult:  1.5x
Min hold:        30 min (MTF=1) / 3h (MTF=2) / 12h (MTF=3)
Position size:   Kelly-scaled, $15 max
AI threshold:    quant ≥ 0.60
Time contract:   min=900s max=21600s reeval=300s
Best in:         Bullish / Trending / Sideways regime
```

---

## L4 — Meme Pump / Scalp ★

```
Entry trigger:   vol ≥ 2x OR (book_imb ≥ 0.15 AND buy_ratio > 0.55 AND mom > 0)
                 OR any activity (book_imb > 0 OR mom > 0 OR vol > 0.5)
                 NO EMA history required
TP:              ATR × 2.5 clamped between 2.0% and MEME_TAKE_PROFIT_PCT=5.0%
SL:              ATR × 2.0 clamped between 2.0% and MEME_STOP_LOSS_PCT=2.0%
ATR trail mult:  1.0x (TIGHTEST — memes reverse hard)
Min hold:        30 min max (short hold, fast extract)
Position size:   $10 HARD CAP (no Kelly scaling above $10)
AI threshold:    Any positive signal sufficient — separate meme prompt
AI prompt:       data/nemo_meme_entry_prompt.txt (separate from standard)
Time contract:   min=60-300s max=900-1800s reeval=60-120s
Best in:         Any regime — memes are regime-agnostic
Exit priority:   Profit ladder fires FIRST (lock gains fast)
                 Trail at 1.0x ATR second
                 Hard SL third

PROFIT LADDER for L4:
  Peak ≥ 2%   → lock 60% of peak (minimum $0.12 on $10)
  Peak ≥ 4%   → lock 70% of peak
  Peak ≥ 7%   → lock 80% of peak
  Peak ≥ 10%  → lock 88% of peak (max extraction)

WHY L4 IS ISOLATED:
  ✓ $10 cap = can't blow up the book
  ✓ No EMA requirement = catches new coins instantly
  ✓ Separate AI prompt = no cross-contamination with L1/L3 logic
  ✓ Tightest trail = harvests burst, doesn't ride the dump
  ✓ Fast time contract = in and out before reversal
  ✓ Behavioral detection = scales to 300 coins without hardcoding
```

---

## Profit Ladder (All Lanes)

| Peak Reached | Trail Locks At | Min Protected |
|---|---|---|
| ≥ 2% | 60% of peak | +1.2% |
| ≥ 4% | 70% of peak | +2.8% |
| ≥ 7% | 80% of peak | +5.6% |
| ≥ 10% | 88% of peak | +8.8% |

---

## Multi-Timeframe Hold Tiers (L1/L3 only — L4 always uses SHORT)

| Alignment | Min Hold | What It Means |
|---|---|---|
| MTF:ALL (3) | 12 hours | 1d + 7d + 30d all green → strong conviction |
| MTF:1d+7d (2) | 3 hours | Medium conviction → patient exit |
| MTF:1d_only (1) | 30 min | Short-term only → quick scalp |
| None (0) | 30 min | Default |

---

## .env Configuration

```ini
# L4 Meme Settings
MEME_TAKE_PROFIT_PCT=5.0
MEME_STOP_LOSS_PCT=2.0
ATR_TRAIL_L4=1.0

# All lanes ATR trail
ATR_TRAIL_L1=2.0
ATR_TRAIL_L2=1.3
ATR_TRAIL_L3=1.5
ATR_TRAIL_L4=1.0

# Profit ladder
HOLD_L1_PCT=2.0   HOLD_L1_STOP=0.60
HOLD_L2_PCT=4.0   HOLD_L2_STOP=0.70
HOLD_L3_PCT=7.0   HOLD_L3_STOP=0.80
HOLD_L4_PCT=10.0  HOLD_L4_STOP=0.88
```
