$ErrorActionPreference = "Stop"

$projectRoot = if ($env:KRAKEN_PROJECT_ROOT) { $env:KRAKEN_PROJECT_ROOT } else { (Split-Path -Parent $PSScriptRoot) }
$projectRoot = (Resolve-Path $projectRoot).Path
$env:KRAKEN_PROJECT_ROOT = $projectRoot
$env:ENV_PATH = Join-Path $projectRoot ".env"
Set-Location $projectRoot
$llamaDir = if ($env:LLAMA_DIR) { $env:LLAMA_DIR } else { "C:\Users\kitti\Desktop\llama-server" }
$logDir = Join-Path $projectRoot "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
if (-not $env:MODE) { $env:MODE = "paper" }

function Wait-Port {
  param(
    [string]$TargetHost = "127.0.0.1",
    [int]$Port,
    [int]$TimeoutSec = 240
  )
  $start = Get-Date
  while ((Get-Date) -lt $start.AddSeconds($TimeoutSec)) {
    try {
      $client = New-Object System.Net.Sockets.TcpClient
      $iar = $client.BeginConnect($TargetHost, $Port, $null, $null)
      if ($iar.AsyncWaitHandle.WaitOne(500)) {
        $client.EndConnect($iar) | Out-Null
        $client.Close()
        return $true
      }
      $client.Close()
    } catch {
      # ignore and retry
    }
    Start-Sleep -Milliseconds 500
  }
  return $false
}

# Model blobs
$entryModel = if ($env:LLAMA_GPU_MODEL) { $env:LLAMA_GPU_MODEL } else { "C:\Users\kitti\blobs\sha256-2049f5674b1e92b4464e5729975c9689fcfbf0b0e4443ccf10b5339f370f9a54" }
$exitModel  = if ($env:LLAMA_CPU_MODEL) { $env:LLAMA_CPU_MODEL } else { "C:\Users\kitti\blobs\qwen2.5-1.5b-instruct-q4_k_m.gguf" }

# Ports
$entryPort = 8081
$exitPort  = 8082
$healthPort = 9091

Write-Host "Starting AI entry server on $entryPort..."
Start-Process -NoNewWindow -FilePath (Join-Path $llamaDir "llama-server.exe") -WorkingDirectory $llamaDir `
  -ArgumentList "--model", $entryModel, "--port", "$entryPort", "--ctx-size", "2048", "--n-gpu-layers", "99", "-fa", "on" `
  -RedirectStandardOutput (Join-Path $logDir "llama_8081.out.log") `
  -RedirectStandardError (Join-Path $logDir "llama_8081.err.log")

Start-Sleep -Seconds 2

Write-Host "Starting AI exit server on $exitPort..."
Start-Process -NoNewWindow -FilePath (Join-Path $llamaDir "llama-server.exe") -WorkingDirectory $llamaDir `
  -ArgumentList "--model", $exitModel, "--port", "$exitPort", "--ctx-size", "2048", "--n-gpu-layers", "0" `
  -RedirectStandardOutput (Join-Path $logDir "llama_8082.out.log") `
  -RedirectStandardError (Join-Path $logDir "llama_8082.err.log")

Write-Host "Waiting for AI entry server on $entryPort..."
$entryReady = Wait-Port -Port $entryPort -TimeoutSec 300
Write-Host "Waiting for AI exit server on $exitPort..."
$exitReady = Wait-Port -Port $exitPort -TimeoutSec 300
if (-not $entryReady) { Write-Error "AI entry server not reachable on $entryPort. Aborting engine start."; exit 1 }
if (-not $exitReady) { Write-Error "AI exit server not reachable on $exitPort. Aborting engine start."; exit 1 }

Write-Host "Starting engine on $healthPort..."
Start-Process -NoNewWindow -FilePath (Join-Path $projectRoot "rust_core\\target\\release\\hybrid_kraken_core.exe") -WorkingDirectory $projectRoot `
  -RedirectStandardOutput (Join-Path $logDir "core.out.log") `
  -RedirectStandardError (Join-Path $logDir "core.err.log")

Write-Host "Stack start issued. Logs: $logDir"
