//! Centralized configuration — the single source of truth for all settings.
//!
//! Rule: only this module reads `env::var()`. Everything else receives config by reference.

pub mod trading;
pub mod llm;
pub mod addons;
pub mod paths;
pub mod schedule;

// Re-export TradingConfig and friends at config:: level for backward compatibility.
// Existing code using `config::TradingConfig` continues to work unchanged.
pub use trading::*;
pub use llm::LlmConfig;
pub use addons::AddonsConfig;
pub use paths::PathsConfig;

/// Top-level application config combining all sub-configs.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct AppConfig {
    pub trading: TradingConfig,
    pub llm: LlmConfig,
    pub addons: AddonsConfig,
    pub paths: PathsConfig,
}

#[allow(dead_code)]
impl AppConfig {
    /// Load all configuration from environment variables.
    pub fn from_env() -> Self {
        Self {
            trading: TradingConfig::from_env(),
            llm: LlmConfig::from_env(),
            addons: AddonsConfig::from_env(),
            paths: PathsConfig::from_env(),
        }
    }

    /// Hot-reload all sub-configs from .env without restart.
    /// Checks .env mtime first to avoid unnecessary I/O.
    pub fn hot_reload(&mut self) {
        // TradingConfig handles its own mtime check + dotenvy reload
        self.trading.hot_reload();
        // Sub-configs read from the already-refreshed process env
        self.llm.hot_reload();
        self.addons.hot_reload();
        // Paths are not hot-reloadable (changing paths mid-flight is dangerous)
    }
}
