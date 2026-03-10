param(
    [string]$Root = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command rg -ErrorAction SilentlyContinue)) {
    Write-Host "[FATAL] ripgrep (rg) not found. Install ripgrep to use this hook."
    exit 2
}

$pattern = "(api[_-]?key|api[_-]?secret|secret|token|password|AKIA|BEGIN PRIVATE KEY|-----BEGIN|xox[pbar]-|sk-[A-Za-z0-9]{20,})"

$args = @(
    "-n",
    "--hidden",
    "--glob", "!.env*",
    "--glob", "!data/**",
    "--glob", "!npu_journal/**",
    "--glob", "!**/target/**",
    "--glob", "!**/__pycache__/**",
    $pattern,
    $Root
)

Write-Host "[pre-commit] Scanning for secrets..."
$matches = & rg @args
if ($LASTEXITCODE -eq 0 -and $matches) {
    Write-Host "[FATAL] Potential secrets detected:"
    $matches | ForEach-Object { Write-Host $_ }
    exit 1
}

Write-Host "[pre-commit] OK"
exit 0
