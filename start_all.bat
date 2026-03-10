@echo off
title Kraken-Hybrad9 Launcher
echo ============================================
echo  KRAKEN-HYBRAD9 — Starting All Services
echo ============================================

REM ── Kill any stale processes ─────────────────
echo [1/4] Cleaning up old processes...
taskkill /F /IM llama-server.exe >nul 2>&1
taskkill /F /IM hybrid_kraken_core.exe >nul 2>&1
timeout /t 2 /nobreak >nul

REM ── Start Nemotron 9B — Port 8081 ───────────
echo [2/4] Starting Nemotron-Nano-9B on port 8081...
start "Nemo-9B [8081]" /min "C:\Users\kitti\Desktop\llama-server\llama-server.exe" ^
  --model "C:\Users\kitti\blobs\sha256-95ee1b5339df8b4c735564cc2297572c972f181222cced619e726ba428adfc83" ^
  --port 8081 ^
  --ctx-size 8192 ^
  --n-gpu-layers 999 ^
  --flash-attn on ^
  --log-file "C:\Users\kitti\Desktop\llama-server\llama_nemotron9b.log"

REM ── Start OpenReasoning 1.5B — Port 8082 ────
echo [3/4] Starting OpenReasoning-1.5B on port 8082...
start "Quant-1.5B [8082]" /min "C:\Users\kitti\Desktop\llama-server\llama-server.exe" ^
  --model "C:\Users\kitti\.lmstudio\models\lmstudio-community\OpenReasoning-Nemotron-1.5B-GGUF\OpenReasoning-Nemotron-1.5B-Q8_0.gguf" ^
  --port 8082 ^
  --ctx-size 4096 ^
  --n-gpu-layers 999 ^
  --flash-attn on ^
  --log-file "C:\Users\kitti\Desktop\llama-server\llama_qwen1_5b.log"

REM ── Wait for Nemotron to be ready ───────────
echo [4/4] Waiting for AI servers to be ready...
:wait_8081
timeout /t 3 /nobreak >nul
curl -s http://127.0.0.1:8081/health >nul 2>&1
if errorlevel 1 (
    echo   ... waiting for Nemotron 9B [8081]
    goto wait_8081
)
echo   [OK] Nemotron 9B ready!

:wait_8082
timeout /t 2 /nobreak >nul
curl -s http://127.0.0.1:8082/health >nul 2>&1
if errorlevel 1 (
    echo   ... waiting for Quant 1.5B [8082]
    goto wait_8082
)
echo   [OK] Quant 1.5B ready!

REM ── Start NPU Signal Bridge ─────────────────
echo Starting NPU Signal Bridge (Intel AI Boost)...
start "NPU-Bridge" /min python "C:\Users\kitti\Projects\kraken-hybrad9\npu_bridge.py"
timeout /t 3 /nobreak >nul

REM ── Start the Bot ────────────────────────────
echo.
echo  Starting Kraken-Hybrad9 bot...
start "Kraken-Bot" /min /D "C:\Users\kitti\Projects\kraken-hybrad9" ^
  powershell -WindowStyle Minimized -Command ^
  "& '.\rust_core\target\release\hybrid_kraken_core.exe' *>> '.\logs\bot.log'"

echo.
echo ============================================
echo  All services running in background.
echo  Check taskbar for minimized windows.
echo  Logs: Desktop\llama-server\*.log
echo  Bot:  Projects\kraken-hybrad9\logs\bot.log
echo ============================================
timeout /t 5 /nobreak >nul
