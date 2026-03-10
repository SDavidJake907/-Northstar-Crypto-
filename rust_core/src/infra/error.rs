use thiserror::Error;

pub type Result<T> = std::result::Result<T, HybridKrakenError>;

#[derive(Debug, Error)]
pub enum HybridKrakenError {
    #[error("http client error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("base64 decode error: {0}")]
    Base64(#[from] base64::DecodeError),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("missing env var: {0}")]
    MissingEnv(&'static str),
    #[error("hmac init error: {0}")]
    HmacInit(String),
    #[error("kraken api error: {0}")]
    KrakenApi(String),
    #[error("kraken http status: {0}")]
    KrakenHttp(String),
    #[error("{0}")]
    Other(String),
}
