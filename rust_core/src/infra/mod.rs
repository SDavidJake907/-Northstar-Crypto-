//! Infrastructure layer — WebSocket, order book, Kraken API, error types, event bus.

pub mod book;
pub mod error;
pub mod ws;
pub mod kraken_api;
pub mod event_bus;
