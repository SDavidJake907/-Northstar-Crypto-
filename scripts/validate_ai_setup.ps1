param(
    [string]$ProjectRoot = ""
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = Split-Path -Parent $PSScriptRoot
}

function Get-EnvMap([string]$envPath) {
    $map = @{}
    if (-not (Test-Path -LiteralPath $envPath)) { return $map }
    foreach ($line in Get-Content -LiteralPath $envPath) {
        $trim = $line.Trim()
        if (-not $trim -or $trim.StartsWith('#')) { continue }
        $idx = $trim.IndexOf('=')
        if ($idx -lt 1) { continue }
        $k = $trim.Substring(0,$idx).Trim()
        $v = $trim.Substring($idx+1).Trim().Trim('"')
        $map[$k] = $v
    }
    return $map
}

function To-Abs([string]$root,[string]$p) {
    if ([string]::IsNullOrWhiteSpace($p)) { return $p }
    if ([System.IO.Path]::IsPathRooted($p)) { return $p }
    return Join-Path $root $p
}

$envPath = Join-Path $ProjectRoot '.env'
$envMap = Get-EnvMap $envPath

$checks = @()
$checks += [PSCustomObject]@{ Key='USE_CPP_GPU'; Value=$envMap['USE_CPP_GPU']; Ok=($envMap['USE_CPP_GPU'] -in @('1','true','True')) }
$checks += [PSCustomObject]@{ Key='USE_CPP_NPU'; Value=$envMap['USE_CPP_NPU']; Ok=($envMap['USE_CPP_NPU'] -in @('1','true','True')) }
$checks += [PSCustomObject]@{ Key='AI_GPU_PASSES'; Value=$envMap['AI_GPU_PASSES']; Ok=([int]($envMap['AI_GPU_PASSES'] -as [int]) -ge 2) }
$checks += [PSCustomObject]@{ Key='AI_NPU_FINAL_DECISION'; Value=$envMap['AI_NPU_FINAL_DECISION']; Ok=($envMap['AI_NPU_FINAL_DECISION'] -in @('1','true','True')) }

$paths = @(
    @{ Name='GPU DLL'; Path=(To-Abs $ProjectRoot $envMap['GPU_CPP_DLL']) },
    @{ Name='NPU DLL'; Path=(To-Abs $ProjectRoot $envMap['NPU_CPP_DLL']) },
    @{ Name='DeepSeek GPU model A'; Path=(To-Abs $ProjectRoot $envMap['DEEPSEEK_GGUF']) },
    @{ Name='DeepSeek GPU model B'; Path=(To-Abs $ProjectRoot $envMap['QWEN_GGUF']) },
    @{ Name='Qwen NPU model dir'; Path=(To-Abs $ProjectRoot $envMap['NPU_CPP_MODEL_DIR']) }
)

Write-Host "AI Setup Validation" -ForegroundColor Cyan
Write-Host "Project: $ProjectRoot"
Write-Host ""
Write-Host "Flags:" -ForegroundColor Yellow
foreach ($c in $checks) {
    $status = if ($c.Ok) { 'OK' } else { 'MISSING/INVALID' }
    Write-Host (" - {0}={1} [{2}]" -f $c.Key, $c.Value, $status)
}
Write-Host ""
Write-Host "Paths:" -ForegroundColor Yellow
$missing = @()
foreach ($p in $paths) {
    $exists = Test-Path -LiteralPath $p.Path
    $status = if ($exists) { 'OK' } else { 'MISSING' }
    Write-Host (" - {0}: {1} [{2}]" -f $p.Name, $p.Path, $status)
    if (-not $exists) { $missing += $p.Name }
}

$badFlags = $checks | Where-Object { -not $_.Ok }
Write-Host ""
if ($badFlags.Count -eq 0 -and $missing.Count -eq 0) {
    Write-Host "Result: READY" -ForegroundColor Green
    exit 0
}

Write-Host "Result: NOT READY" -ForegroundColor Red
if ($badFlags.Count -gt 0) {
    Write-Host " - Fix flags: $($badFlags.Key -join ', ')"
}
if ($missing.Count -gt 0) {
    Write-Host " - Missing paths: $($missing -join ', ')"
}
exit 1

