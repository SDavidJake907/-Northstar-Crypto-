
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Shared GPU math client, wired into MarketBrain and feature computation.
pub type SharedGpuMath = Arc<RwLock<GpuMathClient>>;

pub fn new_shared() -> SharedGpuMath {
    Arc::new(RwLock::new(GpuMathClient::new()))
}

// ── Public result types (unchanged API) ──────────────────────────

#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct BatchFeatureResult {
    pub hurst: Vec<f64>,
    pub entropy: Vec<f64>,
    pub autocorr: Vec<f64>,
}

#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct FingerprintResult {
    pub metrics: Vec<f64>,
}

#[derive(Debug, Clone, Default)]
pub struct CovarianceResult {
    pub cov: Vec<Vec<f64>>,
    pub mean_returns: Vec<f64>,
}

// ── CUDA engine (only when gpu_cuda feature is enabled) ──────────

#[cfg(feature = "gpu_cuda")]
mod cuda_engine {
    use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
    use cudarc::cublas::sys::cublasOperation_t;
    use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::Ptx;
    use std::collections::HashMap;
    use std::sync::Arc;
    use tracing::warn;

    const MAX_COINS: usize = 128;
    const MAX_SAMPLES: usize = 512;
    #[allow(dead_code)]
    const MAX_RETURNS: usize = MAX_SAMPLES - 1;
    const N_BINS: usize = 10;
    const HURST_SIZES: [usize; 5] = [8, 16, 32, 64, 128];
    const MAX_HURST_SIZES: usize = 5;

    /// PTX source compiled by build.rs at build time.
    const PTX_SOURCE: &str = include_str!(env!("QUANT_KERNELS_PTX"));

    pub struct GpuEngine {
        stream: Arc<CudaStream>,
        module: Arc<CudaModule>,
        blas: CudaBlas,
    }

    #[allow(dead_code)]
    impl GpuEngine {
        pub fn init() -> Result<Self, String> {
            let ctx = CudaContext::new(0).map_err(|e| format!("CUDA context: {e}"))?;
            let stream = ctx.default_stream();

            let ptx = Ptx::from_src(PTX_SOURCE);
            let module = ctx.load_module(ptx).map_err(|e| format!("PTX load: {e}"))?;

            let blas = CudaBlas::new(stream.clone()).map_err(|e| format!("cuBLAS: {e}"))?;

            Ok(Self { stream, module, blas })
        }

        fn func(&self, name: &str) -> CudaFunction {
            self.module.load_function(name)
                .unwrap_or_else(|e| panic!("[GPU-MATH] kernel '{name}' not found: {e}"))
        }

        /// Flatten prices into row-major GPU buffer.
        /// Returns (device_slice, n_coins, n_samples).
        fn upload_prices(
            &self,
            prices_map: &HashMap<String, Vec<f64>>,
            symbols: &[String],
        ) -> Result<(CudaSlice<f64>, usize, usize), String> {
            if symbols.len() > MAX_COINS {
                warn!("[GPU-MATH] Too many coins for GPU path: {} > {}", symbols.len(), MAX_COINS);
                return Err("too many coins for GPU path".into());
            }

            let max_len = symbols.iter()
                .filter_map(|s| prices_map.get(s).map(|v| v.len()))
                .max()
                .unwrap_or(0);
            if max_len > MAX_SAMPLES {
                warn!("[GPU-MATH] Too many samples for GPU path: {} > {}", max_len, MAX_SAMPLES);
                return Err("too many samples for GPU path".into());
            }

            let n_coins = symbols.len().min(MAX_COINS);
            let n_samples = symbols.iter()
                .map(|s| prices_map[s].len())
                .min()
                .unwrap_or(0)
                .min(MAX_SAMPLES);

            if n_coins == 0 || n_samples < 3 {
                return Err("insufficient data".into());
            }

            let mut flat = vec![0.0f64; n_coins * n_samples];
            for (i, sym) in symbols.iter().enumerate().take(n_coins) {
                let src = &prices_map[sym];
                let offset = if src.len() > n_samples { src.len() - n_samples } else { 0 };
                for j in 0..n_samples {
                    flat[i * n_samples + j] = src[offset + j];
                }
            }

            let d_prices = self.stream.clone_htod(&flat)
                .map_err(|e| format!("htod: {e}"))?;

            Ok((d_prices, n_coins, n_samples))
        }

        /// Batch features: Hurst, entropy, autocorrelation for all coins.
        pub fn batch_features(
            &self,
            closes_map: &HashMap<String, Vec<f64>>,
        ) -> Option<HashMap<String, (f64, f64, f64)>> {
            let symbols: Vec<String> = closes_map.keys().cloned().collect();
            let (d_prices, n_coins, n_samples) = self.upload_prices(closes_map, &symbols).ok()?;
            let n_returns = n_samples - 1;

            let block_dim = 256u32.min(n_returns as u32);

            // 1. Compute simple returns
            let mut d_returns = self.stream.alloc_zeros::<f64>(n_coins * n_returns).ok()?;
            let f_returns = self.func("compute_returns");
            unsafe {
                self.stream.launch_builder(&f_returns)
                    .arg(&d_prices)
                    .arg(&mut d_returns)
                    .arg(&(n_coins as i32))
                    .arg(&(n_samples as i32))
                    .launch(LaunchConfig {
                        grid_dim: (n_coins as u32, 1, 1),
                        block_dim: (block_dim, 1, 1),
                        shared_mem_bytes: 0,
                    }).ok()?;
            }

            // 2. Compute log returns (for Hurst)
            let mut d_log_returns = self.stream.alloc_zeros::<f64>(n_coins * n_returns).ok()?;
            let f_log_returns = self.func("compute_log_returns");
            unsafe {
                self.stream.launch_builder(&f_log_returns)
                    .arg(&d_prices)
                    .arg(&mut d_log_returns)
                    .arg(&(n_coins as i32))
                    .arg(&(n_samples as i32))
                    .launch(LaunchConfig {
                        grid_dim: (n_coins as u32, 1, 1),
                        block_dim: (block_dim, 1, 1),
                        shared_mem_bytes: 0,
                    }).ok()?;
            }

            // 3. Hurst R/S blocks
            let mut d_rs_out = self.stream.alloc_zeros::<f64>(n_coins * MAX_HURST_SIZES).ok()?;
            let f_hurst = self.func("hurst_rs_blocks");

            for (size_idx, &block_size) in HURST_SIZES.iter().enumerate() {
                if block_size > n_returns { break; }
                let n_blocks = n_returns / block_size;
                if n_blocks == 0 { continue; }

                let grid = (n_coins * n_blocks) as u32;
                let shared_bytes = (block_size * 2 * std::mem::size_of::<f64>()) as u32;

                unsafe {
                    self.stream.launch_builder(&f_hurst)
                        .arg(&d_log_returns)
                        .arg(&mut d_rs_out)
                        .arg(&(n_returns as i32))
                        .arg(&(block_size as i32))
                        .arg(&(n_blocks as i32))
                        .arg(&(size_idx as i32))
                        .arg(&(MAX_HURST_SIZES as i32))
                        .arg(&(n_coins as i32))
                        .launch(LaunchConfig {
                            grid_dim: (grid, 1, 1),
                            block_dim: (block_size as u32, 1, 1),
                            shared_mem_bytes: shared_bytes,
                        }).ok()?;
                }
            }

            // Download RS values and compute Hurst via linear regression on CPU
            let rs_host: Vec<f64> = self.stream.clone_dtoh(&d_rs_out).ok()?;

            let mut hursts = vec![0.5f64; n_coins];
            for coin in 0..n_coins {
                let mut log_ns = Vec::new();
                let mut log_rs = Vec::new();
                for (size_idx, &block_size) in HURST_SIZES.iter().enumerate() {
                    if block_size > n_returns { break; }
                    let rs = rs_host[coin * MAX_HURST_SIZES + size_idx];
                    if rs > 0.0 {
                        log_ns.push((block_size as f64).ln());
                        log_rs.push(rs.ln());
                    }
                }
                if log_ns.len() >= 2 {
                    let n = log_ns.len() as f64;
                    let sx: f64 = log_ns.iter().sum();
                    let sy: f64 = log_rs.iter().sum();
                    let sxy: f64 = log_ns.iter().zip(&log_rs).map(|(x, y)| x * y).sum();
                    let sxx: f64 = log_ns.iter().map(|x| x * x).sum();
                    let slope = (n * sxy - sx * sy) / (n * sxx - sx * sx + 1e-15);
                    hursts[coin] = slope.clamp(0.0, 1.0);
                }
            }

            // 4. Entropy — minmax then histogram
            let mut d_minmax = self.stream.alloc_zeros::<f64>(n_coins * 2).ok()?;
            let f_minmax = self.func("entropy_minmax");
            unsafe {
                self.stream.launch_builder(&f_minmax)
                    .arg(&d_returns)
                    .arg(&mut d_minmax)
                    .arg(&(n_coins as i32))
                    .arg(&(n_returns as i32))
                    .launch(LaunchConfig {
                        grid_dim: (n_coins as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    }).ok()?;
            }

            let mut d_bins = self.stream.alloc_zeros::<i32>(n_coins * N_BINS).ok()?;
            let f_hist = self.func("entropy_histogram");
            unsafe {
                self.stream.launch_builder(&f_hist)
                    .arg(&d_returns)
                    .arg(&d_minmax)
                    .arg(&mut d_bins)
                    .arg(&(n_coins as i32))
                    .arg(&(n_returns as i32))
                    .launch(LaunchConfig {
                        grid_dim: (n_coins as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    }).ok()?;
            }

            let bins_host: Vec<i32> = self.stream.clone_dtoh(&d_bins).ok()?;

            let mut entropies = vec![0.5f64; n_coins];
            let max_entropy = (N_BINS as f64).log2();
            for coin in 0..n_coins {
                let total: i32 = (0..N_BINS).map(|b| bins_host[coin * N_BINS + b]).sum();
                if total == 0 { continue; }
                let mut entropy = 0.0f64;
                for b in 0..N_BINS {
                    let count = bins_host[coin * N_BINS + b];
                    if count > 0 {
                        let p = count as f64 / total as f64;
                        entropy -= p * p.log2();
                    }
                }
                entropies[coin] = if max_entropy > 0.0 {
                    (entropy / max_entropy).clamp(0.0, 1.0)
                } else {
                    0.5
                };
            }

            // 5. Autocorrelation
            let mut d_ac_stats = self.stream.alloc_zeros::<f64>(n_coins * 3).ok()?;
            let f_ac = self.func("autocorr_stats");
            unsafe {
                self.stream.launch_builder(&f_ac)
                    .arg(&d_returns)
                    .arg(&mut d_ac_stats)
                    .arg(&(n_coins as i32))
                    .arg(&(n_returns as i32))
                    .launch(LaunchConfig {
                        grid_dim: (n_coins as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    }).ok()?;
            }

            let ac_host: Vec<f64> = self.stream.clone_dtoh(&d_ac_stats).ok()?;

            let mut autocorrs = vec![0.0f64; n_coins];
            for coin in 0..n_coins {
                let var = ac_host[coin * 3 + 1];
                let cov = ac_host[coin * 3 + 2];
                if var > 1e-15 {
                    autocorrs[coin] = (cov / var).clamp(-1.0, 1.0);
                }
            }

            // Build result map
            let mut map = HashMap::new();
            for (i, sym) in symbols.iter().enumerate().take(n_coins) {
                map.insert(sym.clone(), (hursts[i], entropies[i], autocorrs[i]));
            }
            Some(map)
        }

        /// Batch correlation matrix via cuBLAS.
        #[allow(dead_code)]
        pub fn batch_correlation(
            &self,
            prices_map: &HashMap<String, Vec<f64>>,
        ) -> Option<(Vec<String>, Vec<Vec<f64>>)> {
            let symbols: Vec<String> = prices_map.keys().cloned().collect();
            let (d_prices, n_coins, n_samples) = self.upload_prices(prices_map, &symbols).ok()?;
            let n_returns = n_samples - 1;

            if n_coins < 2 || n_returns < 3 {
                let eye: Vec<Vec<f64>> = (0..n_coins)
                    .map(|i| (0..n_coins).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
                    .collect();
                return Some((symbols, eye));
            }

            // Compute returns
            let mut d_returns = self.stream.alloc_zeros::<f64>(n_coins * n_returns).ok()?;
            let f_ret = self.func("compute_returns");
            unsafe {
                self.stream.launch_builder(&f_ret)
                    .arg(&d_prices)
                    .arg(&mut d_returns)
                    .arg(&(n_coins as i32))
                    .arg(&(n_samples as i32))
                    .launch(LaunchConfig {
                        grid_dim: (n_coins as u32, 1, 1),
                        block_dim: (256u32.min(n_returns as u32), 1, 1),
                        shared_mem_bytes: 0,
                    }).ok()?;
            }

            // Center returns
            let mut d_centered = self.stream.alloc_zeros::<f64>(n_coins * n_returns).ok()?;
            let mut d_means = self.stream.alloc_zeros::<f64>(n_coins).ok()?;
            let mut d_stds = self.stream.alloc_zeros::<f64>(n_coins).ok()?;
            let f_center = self.func("center_rows");
            unsafe {
                self.stream.launch_builder(&f_center)
                    .arg(&d_returns)
                    .arg(&mut d_centered)
                    .arg(&mut d_means)
                    .arg(&mut d_stds)
                    .arg(&(n_coins as i32))
                    .arg(&(n_returns as i32))
                    .launch(LaunchConfig {
                        grid_dim: (n_coins as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    }).ok()?;
            }

            // cuBLAS: C = centered * centered^T
            // Row-major (n_coins, n_returns) = col-major (n_returns, n_coins)
            // C[i,j] = sum_k centered[i,k] * centered[j,k]
            // In col-major: C = A^T * A where A is (n_returns, n_coins) col-major
            let mut d_cov = self.stream.alloc_zeros::<f64>(n_coins * n_coins).ok()?;
            let cfg = GemmConfig {
                transa: cublasOperation_t::CUBLAS_OP_T,
                transb: cublasOperation_t::CUBLAS_OP_N,
                m: n_coins as i32,
                n: n_coins as i32,
                k: n_returns as i32,
                alpha: 1.0f64,
                lda: n_returns as i32,
                ldb: n_returns as i32,
                beta: 0.0f64,
                ldc: n_coins as i32,
            };
            unsafe { self.blas.gemm(cfg, &d_centered, &d_centered, &mut d_cov).ok()?; }

            // Normalize to correlation
            let mut d_corr = self.stream.alloc_zeros::<f64>(n_coins * n_coins).ok()?;
            let grid_norm = ((n_coins * n_coins + 255) / 256) as u32;
            let f_norm = self.func("corr_normalize");
            unsafe {
                self.stream.launch_builder(&f_norm)
                    .arg(&d_cov)
                    .arg(&d_stds)
                    .arg(&mut d_corr)
                    .arg(&(n_coins as i32))
                    .launch(LaunchConfig {
                        grid_dim: (grid_norm, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    }).ok()?;
            }

            let corr_flat: Vec<f64> = self.stream.clone_dtoh(&d_corr).ok()?;
            let corr: Vec<Vec<f64>> = (0..n_coins)
                .map(|i| (0..n_coins).map(|j| corr_flat[i * n_coins + j]).collect())
                .collect();

            Some((symbols, corr))
        }

        /// Batch covariance matrix + mean returns.
        pub fn batch_covariance(
            &self,
            prices_map: &HashMap<String, Vec<f64>>,
        ) -> Option<(Vec<String>, super::CovarianceResult)> {
            let symbols: Vec<String> = prices_map.keys().cloned().collect();
            let (d_prices, n_coins, n_samples) = self.upload_prices(prices_map, &symbols).ok()?;
            let n_returns = n_samples - 1;

            if n_coins < 2 || n_returns < 3 {
                return Some((symbols, super::CovarianceResult {
                    cov: (0..n_coins)
                        .map(|i| (0..n_coins).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
                        .collect(),
                    mean_returns: vec![0.0; n_coins],
                }));
            }

            let mut d_returns = self.stream.alloc_zeros::<f64>(n_coins * n_returns).ok()?;
            let f_ret = self.func("compute_returns");
            unsafe {
                self.stream.launch_builder(&f_ret)
                    .arg(&d_prices)
                    .arg(&mut d_returns)
                    .arg(&(n_coins as i32))
                    .arg(&(n_samples as i32))
                    .launch(LaunchConfig {
                        grid_dim: (n_coins as u32, 1, 1),
                        block_dim: (256u32.min(n_returns as u32), 1, 1),
                        shared_mem_bytes: 0,
                    }).ok()?;
            }

            let mut d_centered = self.stream.alloc_zeros::<f64>(n_coins * n_returns).ok()?;
            let mut d_means = self.stream.alloc_zeros::<f64>(n_coins).ok()?;
            let mut d_stds = self.stream.alloc_zeros::<f64>(n_coins).ok()?;
            let f_center = self.func("center_rows");
            unsafe {
                self.stream.launch_builder(&f_center)
                    .arg(&d_returns)
                    .arg(&mut d_centered)
                    .arg(&mut d_means)
                    .arg(&mut d_stds)
                    .arg(&(n_coins as i32))
                    .arg(&(n_returns as i32))
                    .launch(LaunchConfig {
                        grid_dim: (n_coins as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    }).ok()?;
            }

            // Covariance = centered * centered^T / (n-1)
            let scale = 1.0 / (n_returns as f64 - 1.0);
            let mut d_cov = self.stream.alloc_zeros::<f64>(n_coins * n_coins).ok()?;
            let cfg = GemmConfig {
                transa: cublasOperation_t::CUBLAS_OP_T,
                transb: cublasOperation_t::CUBLAS_OP_N,
                m: n_coins as i32,
                n: n_coins as i32,
                k: n_returns as i32,
                alpha: scale,
                lda: n_returns as i32,
                ldb: n_returns as i32,
                beta: 0.0f64,
                ldc: n_coins as i32,
            };
            unsafe { self.blas.gemm(cfg, &d_centered, &d_centered, &mut d_cov).ok()?; }

            let cov_flat: Vec<f64> = self.stream.clone_dtoh(&d_cov).ok()?;
            let means: Vec<f64> = self.stream.clone_dtoh(&d_means).ok()?;

            let cov: Vec<Vec<f64>> = (0..n_coins)
                .map(|i| (0..n_coins).map(|j| cov_flat[i * n_coins + j]).collect())
                .collect();

            Some((symbols, super::CovarianceResult { cov, mean_returns: means }))
        }

        /// Fingerprint: 8 market-wide metrics.
        #[allow(dead_code)]
        pub fn batch_fingerprint(
            &self,
            prices_map: &HashMap<String, Vec<f64>>,
            btc_sym: &str,
            eth_sym: &str,
        ) -> Option<super::FingerprintResult> {
            let symbols: Vec<String> = prices_map.keys().cloned().collect();
            let btc_idx = symbols.iter().position(|s| s == btc_sym).unwrap_or(0);
            let eth_idx = symbols.iter().position(|s| s == eth_sym).unwrap_or(1);

            let (d_prices, n_coins, n_samples) = self.upload_prices(prices_map, &symbols).ok()?;
            let n_returns = n_samples - 1;

            if n_returns < 2 {
                return Some(super::FingerprintResult { metrics: vec![0.0; 8] });
            }

            let mut d_returns = self.stream.alloc_zeros::<f64>(n_coins * n_returns).ok()?;
            let f_ret = self.func("compute_returns");
            unsafe {
                self.stream.launch_builder(&f_ret)
                    .arg(&d_prices)
                    .arg(&mut d_returns)
                    .arg(&(n_coins as i32))
                    .arg(&(n_samples as i32))
                    .launch(LaunchConfig {
                        grid_dim: (n_coins as u32, 1, 1),
                        block_dim: (256u32.min(n_returns as u32), 1, 1),
                        shared_mem_bytes: 0,
                    }).ok()?;
            }

            let mut d_fp = self.stream.alloc_zeros::<f64>(8).ok()?;
            let f_fp = self.func("fingerprint_latest");
            unsafe {
                self.stream.launch_builder(&f_fp)
                    .arg(&d_returns)
                    .arg(&mut d_fp)
                    .arg(&(n_coins as i32))
                    .arg(&(n_returns as i32))
                    .arg(&(btc_idx as i32))
                    .arg(&(eth_idx as i32))
                    .launch(LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    }).ok()?;
            }

            let metrics: Vec<f64> = self.stream.clone_dtoh(&d_fp).ok()?;
            Some(super::FingerprintResult { metrics })
        }

        /// Quick liveness test — tries a tiny allocation to detect dead CUDA context (e.g. after TDR).
        pub fn liveness_check(&self) -> Result<(), String> {
            self.stream.alloc_zeros::<f64>(1).map(|_| ()).map_err(|e| e.to_string())
        }
    }
}

// ── Public client (same API as before) ───────────────────────────

pub struct GpuMathClient {
    #[cfg(feature = "gpu_cuda")]
    engine: Option<cuda_engine::GpuEngine>,
    healthy: bool,
}

#[allow(dead_code)]
impl GpuMathClient {
    pub fn new() -> Self {
        #[cfg(feature = "gpu_cuda")]
        {
            match cuda_engine::GpuEngine::init() {
                Ok(engine) => {
                    tracing::info!("[GPU-MATH] CUDA engine initialized (native, in-process)");
                    return Self { engine: Some(engine), healthy: true };
                }
                Err(e) => {
                    tracing::warn!("[GPU-MATH] CUDA init failed: {e} — CPU fallback active");
                    return Self { engine: None, healthy: false };
                }
            }
        }
        #[cfg(not(feature = "gpu_cuda"))]
        {
            tracing::info!("[GPU-MATH] Compiled without gpu_cuda feature — CPU only");
            Self { healthy: false }
        }
    }

    pub async fn check_health(&mut self) -> bool {
        if !self.healthy {
            return false;
        }
        #[cfg(feature = "gpu_cuda")]
        if let Some(ref engine) = self.engine {
            if let Err(e) = engine.liveness_check() {
                tracing::warn!("[GPU-MATH] CUDA context lost (TDR/OOM): {e} — disabling GPU, CPU fallback active");
                self.healthy = false;
                return false;
            }
        }
        true
    }

    pub fn is_healthy(&self) -> bool {
        self.healthy
    }

    pub async fn batch_features(
        &self,
        closes_map: &HashMap<String, Vec<f64>>,
    ) -> Option<HashMap<String, (f64, f64, f64)>> {
        #[cfg(feature = "gpu_cuda")]
        if let Some(ref engine) = self.engine {
            let out = engine.batch_features(closes_map);
            if out.is_none() {
                let liveness = engine.liveness_check().err().unwrap_or_default();
                if liveness.is_empty() {
                    tracing::warn!("[GPU-MATH] batch_features failed (context alive) — CPU fallback");
                } else {
                    tracing::warn!("[GPU-MATH] batch_features failed: context dead ({liveness}) — CPU fallback");
                }
            }
            return out;
        }
        None
    }

    #[allow(dead_code)]
    pub async fn batch_correlation(
        &self,
        prices_map: &HashMap<String, Vec<f64>>,
    ) -> Option<(Vec<String>, Vec<Vec<f64>>)> {
        #[cfg(feature = "gpu_cuda")]
        if let Some(ref engine) = self.engine {
            let out = engine.batch_correlation(prices_map);
            if out.is_none() {
                tracing::warn!("[GPU-MATH] batch_correlation failed — CPU fallback");
            }
            return out;
        }
        None
    }

    pub async fn batch_covariance(
        &self,
        prices_map: &HashMap<String, Vec<f64>>,
    ) -> Option<(Vec<String>, CovarianceResult)> {
        #[cfg(feature = "gpu_cuda")]
        if let Some(ref engine) = self.engine {
            let out = engine.batch_covariance(prices_map);
            if out.is_none() {
                tracing::warn!("[GPU-MATH] batch_covariance failed — CPU fallback");
            }
            return out;
        }
        None
    }

    #[allow(dead_code)]
    pub async fn batch_fingerprint(
        &self,
        prices_map: &HashMap<String, Vec<f64>>,
        btc_sym: &str,
        eth_sym: &str,
    ) -> Option<FingerprintResult> {
        #[cfg(feature = "gpu_cuda")]
        if let Some(ref engine) = self.engine {
            let out = engine.batch_fingerprint(prices_map, btc_sym, eth_sym);
            if out.is_none() {
                tracing::warn!("[GPU-MATH] batch_fingerprint failed — CPU fallback");
            }
            return out;
        }
        None
    }
}
