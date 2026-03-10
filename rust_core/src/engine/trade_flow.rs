use crate::features;
use anyhow::anyhow;
use std::collections::VecDeque;
use std::str::FromStr;

#[derive(Clone, Debug)]
pub struct Trade {
    pub ts: f64,
    pub side: TradeSide,
    pub volume: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TradeSide {
    Buy,
    Sell,
}

impl FromStr for TradeSide {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.eq_ignore_ascii_case("buy") {
            Ok(TradeSide::Buy)
        } else if s.eq_ignore_ascii_case("sell") {
            Ok(TradeSide::Sell)
        } else {
            Err(anyhow!("Invalid TradeSide string: {}", s))
        }
    }
}

impl TradeSide {}

#[derive(Default, Debug)]
pub struct TradeFlowState {
    window_sec: f64,
    maxlen: usize,
    trades: VecDeque<Trade>,
    buy_sum: f64,
    sell_sum: f64,
}

impl TradeFlowState {
    pub fn new(window_sec: f64, maxlen: usize) -> Self {
        Self {
            window_sec,
            maxlen,
            trades: VecDeque::with_capacity(maxlen),
            buy_sum: 0.0,
            sell_sum: 0.0,
        }
    }

    pub fn update(&mut self, trade: Trade) {
        // Drop stale trades outside rolling time window.
        if self.window_sec > 0.0 {
            let cutoff = trade.ts - self.window_sec;
            while let Some(front) = self.trades.front() {
                if front.ts >= cutoff {
                    break;
                }
                if let Some(old) = self.trades.pop_front() {
                    match old.side {
                        TradeSide::Buy => self.buy_sum -= old.volume,
                        TradeSide::Sell => self.sell_sum -= old.volume,
                    }
                } else {
                    break;
                }
            }
        }

        // If deque at max, account for pop-left.
        if self.trades.len() >= self.maxlen {
            if let Some(old) = self.trades.pop_front() {
                match old.side {
                    TradeSide::Buy => self.buy_sum -= old.volume,
                    TradeSide::Sell => self.sell_sum -= old.volume,
                }
            }
        }

        match trade.side {
            TradeSide::Buy => self.buy_sum += trade.volume,
            TradeSide::Sell => self.sell_sum += trade.volume,
        }
        self.trades.push_back(trade);

        if self.buy_sum < 0.0 {
            self.buy_sum = 0.0;
        }
        if self.sell_sum < 0.0 {
            self.sell_sum = 0.0;
        }
    }

    pub fn get_flow(&self) -> features::TradeFlow {
        let total = self.buy_sum + self.sell_sum;
        if total <= 0.0 {
            return features::TradeFlow {
                buy_ratio: 0.5,
                sell_ratio: 0.5,
            };
        }
        features::TradeFlow {
            buy_ratio: self.buy_sum / total,
            sell_ratio: self.sell_sum / total,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Trade, TradeFlowState, TradeSide};

    #[test]
    fn flow_defaults_to_even_when_empty() {
        let flow = TradeFlowState::new(10.0, 10).get_flow();
        assert_eq!(flow.buy_ratio, 0.5);
        assert_eq!(flow.sell_ratio, 0.5);
    }

    #[test]
    fn window_prunes_stale_trades() {
        let mut s = TradeFlowState::new(10.0, 10);
        s.update(Trade {
            ts: 1.0,
            side: TradeSide::Buy,
            volume: 1.0,
        });
        s.update(Trade {
            ts: 5.0,
            side: TradeSide::Sell,
            volume: 1.0,
        });
        s.update(Trade {
            ts: 20.0,
            side: TradeSide::Buy,
            volume: 2.0,
        });
        let flow = s.get_flow();
        assert!((flow.buy_ratio - 1.0).abs() < 1e-9);
        assert!((flow.sell_ratio - 0.0).abs() < 1e-9);
    }

    #[test]
    fn maxlen_evicts_oldest_trade() {
        let mut s = TradeFlowState::new(0.0, 2);
        s.update(Trade {
            ts: 1.0,
            side: TradeSide::Buy,
            volume: 3.0,
        });
        s.update(Trade {
            ts: 2.0,
            side: TradeSide::Sell,
            volume: 1.0,
        });
        s.update(Trade {
            ts: 3.0,
            side: TradeSide::Buy,
            volume: 1.0,
        });
        let flow = s.get_flow();
        assert!((flow.buy_ratio - 0.5).abs() < 1e-9);
        assert!((flow.sell_ratio - 0.5).abs() < 1e-9);
    }
}
