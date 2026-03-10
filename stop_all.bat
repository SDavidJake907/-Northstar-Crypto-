@echo off
title Kraken-Hybrad9 Stop
echo Stopping all Kraken-Hybrad9 services...
taskkill /F /IM hybrid_kraken_core.exe >nul 2>&1 && echo  [OK] Bot stopped
taskkill /F /IM llama-server.exe >nul 2>&1        && echo  [OK] AI servers stopped
echo Done.
timeout /t 2 /nobreak >nul
