//! Path configuration — all file system paths used by the engine.

/// File path configuration.
#[derive(Clone, Debug)]
pub struct PathsConfig {
    pub snapshot_json: String,
    pub snapshot_shm: String,
    pub positions_file: String,
    pub heartbeat_file: String,
    pub journal_dir: String,
}

impl Default for PathsConfig {
    fn default() -> Self {
        Self {
            snapshot_json: "data/features_snapshot.json".into(),
            snapshot_shm: "data/features_snapshot.shm".into(),
            positions_file: "data/positions.json".into(),
            heartbeat_file: "data/heartbeat_trader.json".into(),
            journal_dir: "npu_journal".into(),
        }
    }
}

impl PathsConfig {
    pub fn from_env() -> Self {
        let mut c = Self::default();
        let e = |key: &str| std::env::var(key).ok();
        let legacy = |new_key: &str, old_key: &str| -> Option<String> {
            e(new_key).or_else(|| {
                let v = e(old_key);
                if v.is_some() {
                    tracing::warn!("[CONFIG] Using legacy key {old_key} → migrate to {new_key}");
                }
                v
            })
        };

        if let Some(v) = legacy("PATH_SNAPSHOT_JSON", "SNAPSHOT_PATH") {
            c.snapshot_json = v;
        }
        if let Some(v) = legacy("PATH_SNAPSHOT_SHM", "SNAPSHOT_SHM_PATH") {
            c.snapshot_shm = v;
        }
        if let Some(v) = legacy("PATH_POSITIONS", "POSITIONS_FILE") {
            c.positions_file = v;
        }
        if let Some(v) = legacy("PATH_HEARTBEAT", "HEARTBEAT_FILE") {
            c.heartbeat_file = v;
        }
        if let Some(v) = legacy("PATH_JOURNAL_DIR", "JOURNAL_DIR") {
            c.journal_dir = v;
        }

        c
    }
}
