$ErrorActionPreference = "Stop"

$root = if ($env:KRAKEN_PROJECT_ROOT) { $env:KRAKEN_PROJECT_ROOT } else { (Split-Path -Parent $PSScriptRoot) }
$archiveDir = Join-Path $root "data\archive"
if (-not (Test-Path $archiveDir)) {
    New-Item -ItemType Directory -Path $archiveDir | Out-Null
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFiles = @(
    "core.log",
    "core.err",
    "engine_output.log",
    "engine_error.log",
    "llama_8081.out.log",
    "llama_8081.err.log",
    "llama_8082.out.log",
    "llama_8082.err.log"
)

foreach ($name in $logFiles) {
    $path = Join-Path $root $name
    if (Test-Path $path) {
        $dest = Join-Path $archiveDir ("{0}.{1}" -f $name, $timestamp)
        Copy-Item $path $dest -Force
        Clear-Content $path
        Write-Host "Rotated $name -> $dest"
    }
}
