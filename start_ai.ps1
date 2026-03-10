$llamaDir = if ($env:LLAMA_DIR) { $env:LLAMA_DIR } else { "C:\Users\kitti\Desktop\llama-server" }
$llama = Join-Path $llamaDir "llama-server.exe"

# Model path - set LLAMA_GPU_MODEL env var to override
$gpuModel = if ($env:LLAMA_GPU_MODEL) { $env:LLAMA_GPU_MODEL } else { "C:\Users\kitti\.ollama\models\blobs\sha256-2049f5674b1e92b4464e5729975c9689fcfbf0b0e4443ccf10b5339f370f9a54" }

$gpuLayers = if ($env:LLAMA_GPU_LAYERS) { $env:LLAMA_GPU_LAYERS } else { "auto" }
$faFlag    = if ($env:LLAMA_FA) { @("-fa", $env:LLAMA_FA) } else { @("-fa", "auto") }

# RTX 5060 Ti 16GB: qwen2.5:14b ~9.5GB VRAM, ctx 8192 KV ~2GB, fits with headroom
$gpuArgs = @(
    "--model",        $gpuModel,
    "--port",         "8081",
    "--ctx-size",     "8192",
    "--n-gpu-layers", $gpuLayers
)
$gpuArgs += $faFlag

$logDir = Join-Path $PSScriptRoot "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

Start-Process -NoNewWindow -FilePath $llama -WorkingDirectory $llamaDir `
    -ArgumentList $gpuArgs `
    -RedirectStandardOutput (Join-Path $logDir "llama_8081.out.log") `
    -RedirectStandardError  (Join-Path $logDir "llama_8081.err.log")

Write-Host "Started llama-server on 8081 (GPU only). Logs in $logDir" -ForegroundColor Green
Write-Host "Model: $gpuModel" -ForegroundColor Cyan
