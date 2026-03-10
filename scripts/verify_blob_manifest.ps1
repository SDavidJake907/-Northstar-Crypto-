param(
    [string]$Root = ".",
    [string]$Manifest = ".\blob_manifest.sha256"
)

$ErrorActionPreference = "Stop"

$rootPath = (Resolve-Path $Root).Path
$manifestPath = if ([System.IO.Path]::IsPathRooted($Manifest)) {
    $Manifest
} else {
    Join-Path $rootPath $Manifest
}

if (-not (Test-Path $manifestPath)) {
    throw "Manifest not found: $manifestPath"
}

$failures = New-Object System.Collections.Generic.List[string]

Get-Content $manifestPath | ForEach-Object {
    if ([string]::IsNullOrWhiteSpace($_)) { return }
    $parts = $_ -split "\s+", 2
    if ($parts.Count -ne 2) {
        $failures.Add("Malformed manifest line: $_")
        return
    }
    $expected = $parts[0].Trim()
    $rel = $parts[1].Trim()
    $full = Join-Path $rootPath $rel
    if (-not (Test-Path $full)) {
        $failures.Add("Missing file: $rel")
        return
    }
    $actual = (Get-FileHash $full -Algorithm SHA256).Hash
    if ($actual -ne $expected) {
        $failures.Add("Hash mismatch: $rel")
    }
}

if ($failures.Count -gt 0) {
    $failures | ForEach-Object { Write-Host $_ -ForegroundColor Red }
    throw "Manifest verification failed with $($failures.Count) issue(s)."
}

Write-Host "Manifest verification passed." -ForegroundColor Green
