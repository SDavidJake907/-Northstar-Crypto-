# Hybrid-Kraken Operator Runbook

## 1) Pre-Flight (once per pull/update)

Run from project root:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\verify_blob_manifest.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\scan_supply_chain.ps1
```

If you need a new baseline manifest:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\generate_blob_manifest.ps1
```

## 2) Build

Full build:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_all.ps1
```

Common variants:

```powershell
# Rust only
powershell -ExecutionPolicy Bypass -File .\scripts\build_all.ps1 -SkipGpu -SkipNpu -SkipBook

# Skip GPU rebuild (if CUDA toolchain unavailable)
powershell -ExecutionPolicy Bypass -File .\scripts\build_all.ps1 -SkipGpu
```

## 3) Local Quality Gates

```powershell
cd .\rust_core
cargo fmt --all -- --check
cargo clippy --workspace --all-targets
cargo test --workspace --all-targets
cargo test --test ffi_smoke
cd ..
```

## 4) Start Live Bot

From project root:

```powershell
.\rust_core\target\release\hybrid_kraken_core.exe
```

Background launch:

```powershell
Start-Process -FilePath ".\rust_core\target\release\hybrid_kraken_core.exe" -WorkingDirectory "."
```

## 5) Stop Bot

```powershell
Get-CimInstance Win32_Process |
  Where-Object { $_.Name -eq "hybrid_kraken_core.exe" } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

## 6) Health Checks

Process running:

```powershell
Get-CimInstance Win32_Process |
  Where-Object { $_.Name -eq "hybrid_kraken_core.exe" } |
  Select-Object ProcessId, Name, CommandLine
```

Heartbeat fresh:

```powershell
Get-Item .\data\heartbeat_trader.json | Select-Object FullName, LastWriteTime, Length
Get-Content .\data\heartbeat_trader.json
```

## 7) Runtime Troubleshooting

No heartbeat:
1. Confirm process exists (Section 6).
2. If process missing, restart (Section 4).
3. If process exists but stale heartbeat, stop + restart and check stderr/stdout logs.

Build failures:
1. `gpu_infer` failure with CUDA toolset missing: run build with `-SkipGpu`.
2. `npu_infer` failure with OpenVINO missing: run with `-SkipNpu`.
3. Continue with Rust + book engine if those pass.

## 8) Current Trading Profile (as configured)

- `BASE_USD=8`
- `AGGR_MAX_POSITIONS=20`
- `DEF_MAX_POSITIONS=20`
- `QUANTUM_MAX_POSITIONS=20`
- `MEME_ENABLED=0`
- `MIN_HOLD_MINUTES=180`

## 9) CI Pipeline Coverage

CI workflow: `.github/workflows/rust-quality.yml`

- `cargo fmt --check`
- `cargo clippy`
- `cargo deny check bans licenses sources`
- Debug + Release matrix build/test
- BookEngine DLL build
- FFI smoke test (`ffi_smoke`)

