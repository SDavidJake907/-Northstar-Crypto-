$ErrorActionPreference = "Stop"

$projectRoot = if ($env:KRAKEN_PROJECT_ROOT) { $env:KRAKEN_PROJECT_ROOT } else { (Split-Path -Parent $PSScriptRoot) }
$llamaDir = if ($env:LLAMA_DIR) { $env:LLAMA_DIR } else { "C:\Users\kitti\Desktop\llama-server" }

$entryModel = if ($env:LLAMA_GPU_MODEL) { $env:LLAMA_GPU_MODEL } else { "C:\Users\kitti\blobs\sha256-2049f5674b1e92b4464e5729975c9689fcfbf0b0e4443ccf10b5339f370f9a54" }
$exitModel  = if ($env:LLAMA_CPU_MODEL) { $env:LLAMA_CPU_MODEL } else { "C:\Users\kitti\blobs\sha256-3e4cb14174460404e7a233e531675303b2fbf7749c02f91864fe311ab6344e4f" }

$entryPort = 8081
$exitPort  = 8082
$healthPort = 9091

function Test-Port($port) {
    $conn = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue
    return $null -ne $conn
}

function Ensure-Entry() {
    if (-not (Test-Port $entryPort)) {
        Write-Host "Entry AI down, restarting..."
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd `"$llamaDir`"; .\llama-server.exe --model `"$entryModel`" --port $entryPort --ctx-size 8192 --n-gpu-layers 99 -fa on --no-mmap"
    }
}

function Ensure-Exit() {
    if (-not (Test-Port $exitPort)) {
        Write-Host "Exit AI down, restarting..."
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd `"$llamaDir`"; .\llama-server.exe --model `"$exitModel`" --port $exitPort --ctx-size 4096 --n-gpu-layers 0 --no-mmap"
    }
}

function Ensure-Engine() {
    if (-not (Test-Port $healthPort)) {
        Write-Host "Engine down, restarting..."
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd `"$projectRoot`"; .\rust_core\target\release\hybrid_kraken_core.exe"
    }
}

while ($true) {
    Ensure-Entry
    Start-Sleep -Seconds 1
    Ensure-Exit
    Start-Sleep -Seconds 1
    Ensure-Engine
    Start-Sleep -Seconds 30
}
