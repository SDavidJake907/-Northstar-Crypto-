//! Portfolio PROFIT Optimizer — maximize gains, not minimize risk.
//!
//! Ported from python_orch/utils/portfolio_optimizer.py
//!
//! Philosophy:
//!   - HIGH confidence → BIGGER position
//!   - BULLISH regime  → BIGGER positions
//!   - Winners drift up → LET THEM RUN (pyramid)
//!   - Losers → CUT FAST

use std::collections::HashMap;
fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = a.len();
    if n == 0 || b.len() != n {
        return None;
    }
    for row in &a {
        if row.len() != n {
            return None;
        }
    }

    // Gaussian elimination with partial pivoting.
    for col in 0..n {
        let mut pivot = col;
        let mut pivot_val = a[col][col].abs();
        for row in (col + 1)..n {
            let v = a[row][col].abs();
            if v > pivot_val {
                pivot_val = v;
                pivot = row;
            }
        }
        if pivot_val <= 1e-12 {
            return None;
        }
        if pivot != col {
            a.swap(pivot, col);
            b.swap(pivot, col);
        }

        let diag = a[col][col];
        for j in col..n {
            a[col][j] /= diag;
        }
        b[col] /= diag;

        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = a[row][col];
            if factor == 0.0 {
                continue;
            }
            for j in col..n {
                a[row][j] -= factor * a[col][j];
            }
            b[row] -= factor * b[col];
        }
    }

    Some(b)
}

// ── Data structures ──────────────────────────────────────────────

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct Position {
    pub symbol: String,
    pub entry_price: f64,
    pub current_price: f64,
    pub qty: f64,
    pub usd_value: f64,
    pub pnl_pct: f64,
    pub pnl_usd: f64,
    pub peak_pnl: f64,
    pub confidence_at_entry: f64,
    pub tier: String,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct SizeRecommendation {
    pub symbol: String,
    pub base_usd: f64,
    pub multiplier: f64,
    pub final_usd: f64,
    pub reason: String,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct ClosedTrade {
    pub symbol: String,
    pub entry: f64,
    pub exit: f64,
    pub qty: f64,
    pub pnl_pct: f64,
    pub pnl_usd: f64,
    pub confidence: f64,
}

// ── Constants ────────────────────────────────────────────────────

// ── Profit Optimizer ─────────────────────────────────────────────

pub struct ProfitOptimizer {
    #[allow(dead_code)]
    pub max_portfolio_usd: f64,
    pub positions: HashMap<String, Position>,
    pub closed_trades: Vec<ClosedTrade>,
    /// Mean-variance optimal weights per coin — updated each tick from GPU covariance.
    /// Maps symbol → normalized weight (sum to 1.0). None until first covariance run.
    pub portfolio_weights: Option<HashMap<String, f64>>,
}

impl ProfitOptimizer {
    pub fn new(_base_usd: f64, _max_position_usd: f64, max_portfolio_usd: f64) -> Self {
        Self {
            max_portfolio_usd,
            positions: HashMap::new(),
            closed_trades: Vec::new(),
            portfolio_weights: None,
        }
    }

    /// Update position with new price.
    pub fn update_position(&mut self, symbol: &str, current_price: f64) {
        if let Some(pos) = self.positions.get_mut(symbol) {
            pos.current_price = current_price;
            if pos.entry_price > 0.0 {
                pos.pnl_pct = (current_price - pos.entry_price) / pos.entry_price;
                pos.pnl_usd = pos.qty * (current_price - pos.entry_price);
                pos.usd_value = pos.qty * current_price;
                if pos.pnl_pct > pos.peak_pnl {
                    pos.peak_pnl = pos.pnl_pct;
                }
            }
        }
    }

    /// Open a new position.
    pub fn open_position(
        &mut self,
        symbol: &str,
        entry_price: f64,
        qty: f64,
        usd_value: f64,
        confidence: f64,
        tier: &str,
    ) {
        self.positions.insert(
            symbol.to_string(),
            Position {
                symbol: symbol.to_string(),
                entry_price,
                current_price: entry_price,
                qty,
                usd_value,
                pnl_pct: 0.0,
                pnl_usd: 0.0,
                peak_pnl: 0.0,
                confidence_at_entry: confidence,
                tier: tier.to_string(),
            },
        );
    }

    /// Close position and record trade. Returns the closed trade info.
    pub fn close_position(&mut self, symbol: &str, exit_price: f64) -> Option<ClosedTrade> {
        let pos = self.positions.remove(symbol)?;
        let pnl_pct = if pos.entry_price > 0.0 {
            (exit_price - pos.entry_price) / pos.entry_price
        } else {
            0.0
        };
        let pnl_usd = pos.qty * (exit_price - pos.entry_price);

        let trade = ClosedTrade {
            symbol: symbol.to_string(),
            entry: pos.entry_price,
            exit: exit_price,
            qty: pos.qty,
            pnl_pct,
            pnl_usd,
            confidence: pos.confidence_at_entry,
        };
        self.closed_trades.push(trade.clone());
        Some(trade)
    }

    /// Mean-variance solver (ridge-regularized) for portfolio weights.
    /// Solves: (cov + lambda * I) * w = mu
    pub fn compute_weights(
        &self,
        symbols: &[String],
        cov: &[Vec<f64>],
        mean_returns: &[f64],
        lambda: f64,
        min_w: f64,
        max_w: f64,
    ) -> Option<HashMap<String, f64>> {
        let n = symbols.len();
        if n == 0 || cov.len() != n || mean_returns.len() != n {
            return None;
        }
        if cov.iter().any(|row| row.len() != n) {
            return None;
        }

        let mut a = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                a[i][j] = cov[i][j];
            }
            a[i][i] += lambda.max(1e-12);
        }
        let b = mean_returns.to_vec();
        let mut w = solve_linear_system(a, b)?;

        // Clamp weights and normalize
        for i in 0..n {
            let v = w[i];
            let clamped = v.max(min_w).min(max_w);
            w[i] = if clamped.is_finite() { clamped } else { 0.0 };
        }
        let sum: f64 = w.iter().sum();
        if sum <= 1e-12 {
            return None;
        }
        for v in &mut w {
            *v /= sum;
        }

        let mut out = HashMap::new();
        for (i, sym) in symbols.iter().enumerate() {
            out.insert(sym.clone(), w[i]);
        }
        Some(out)
    }
}
