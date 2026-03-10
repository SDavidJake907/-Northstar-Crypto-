use ordered_float::OrderedFloat;
use std::collections::{BTreeMap, HashMap};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Debug)]
pub struct OrderBookLevel {
    pub price: f64,
    pub volume: f64,
}

#[derive(Clone, Debug)]
#[allow(dead_code)] // timestamp set in get_top, read in tests
pub struct OrderBook {
    pub symbol: String,
    pub bids: Vec<OrderBookLevel>,
    pub asks: Vec<OrderBookLevel>,
    pub timestamp: f64,
}

#[derive(Default, Debug)]
struct BookState {
    bids: BTreeMap<OrderedFloat<f64>, f64>,
    asks: BTreeMap<OrderedFloat<f64>, f64>,
    timestamp: f64,
}

#[derive(Default)]
pub struct BookStore {
    books: HashMap<String, BookState>,
}

impl BookStore {
    pub fn new() -> Self {
        Self {
            books: HashMap::new(),
        }
    }

    pub fn apply_snapshot(&mut self, symbol: &str, bids: &[(f64, f64)], asks: &[(f64, f64)]) {
        let mut bs = BookState::default();
        for (p, q) in bids {
            if *p > 0.0 && *q > 0.0 {
                bs.bids.insert(OrderedFloat(*p), *q);
            }
        }
        for (p, q) in asks {
            if *p > 0.0 && *q > 0.0 {
                bs.asks.insert(OrderedFloat(*p), *q);
            }
        }
        bs.timestamp = now_ts();
        self.books.insert(symbol.to_string(), bs);
    }

    pub fn apply_update(&mut self, symbol: &str, bids: &[(f64, f64)], asks: &[(f64, f64)]) -> bool {
        let Some(bs) = self.books.get_mut(symbol) else {
            return false;
        };

        let mut changed = false;
        for (p, q) in bids {
            if *p <= 0.0 {
                continue;
            }
            let key = OrderedFloat(*p);
            if *q <= 0.0 {
                if bs.bids.remove(&key).is_some() {
                    changed = true;
                }
            } else if bs.bids.get(&key).is_none_or(|existing| *existing != *q) {
                bs.bids.insert(key, *q);
                changed = true;
            }
        }
        for (p, q) in asks {
            if *p <= 0.0 {
                continue;
            }
            let key = OrderedFloat(*p);
            if *q <= 0.0 {
                if bs.asks.remove(&key).is_some() {
                    changed = true;
                }
            } else if bs.asks.get(&key).is_none_or(|existing| *existing != *q) {
                bs.asks.insert(key, *q);
                changed = true;
            }
        }

        if changed {
            bs.timestamp = now_ts();
        }
        true
    }

    pub fn get_top(&self, symbol: &str, depth: usize) -> Option<OrderBook> {
        let bs = self.books.get(symbol)?;

        let mut bids = Vec::with_capacity(depth);
        let mut asks = Vec::with_capacity(depth);

        for (p, q) in bs.bids.iter().rev().take(depth) {
            bids.push(OrderBookLevel {
                price: p.0,
                volume: *q,
            });
        }
        for (p, q) in bs.asks.iter().take(depth) {
            asks.push(OrderBookLevel {
                price: p.0,
                volume: *q,
            });
        }

        Some(OrderBook {
            symbol: symbol.to_string(),
            bids,
            asks,
            timestamp: bs.timestamp,
        })
    }
}

fn now_ts() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::BookStore;

    #[test]
    fn apply_update_detects_no_change() {
        let mut store = BookStore::new();
        store.apply_snapshot("BTC", &[(100.0, 1.0)], &[(101.0, 2.0)]);
        let before = store.get_top("BTC", 1).expect("book").timestamp;
        let updated = store.apply_update("BTC", &[(100.0, 1.0)], &[(101.0, 2.0)]);
        let after = store.get_top("BTC", 1).expect("book").timestamp;
        assert!(updated);
        assert_eq!(before, after);
        let top = store.get_top("BTC", 1).expect("book");
        assert_eq!(top.bids[0].price, 100.0);
        assert_eq!(top.bids[0].volume, 1.0);
        assert_eq!(top.asks[0].price, 101.0);
        assert_eq!(top.asks[0].volume, 2.0);
    }

    #[test]
    fn apply_update_changes_volume() {
        let mut store = BookStore::new();
        store.apply_snapshot("ETH", &[(200.0, 1.0)], &[(201.0, 1.0)]);
        let changed = store.apply_update("ETH", &[(200.0, 3.0)], &[]);
        assert!(changed);
        let top = store.get_top("ETH", 1).expect("book");
        assert_eq!(top.bids[0].volume, 3.0);
    }
}
