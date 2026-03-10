param(
    [string]$Root = "."
)

$ErrorActionPreference = "Stop"

$patterns = @(
    "curl",
    "wget",
    "Invoke-WebRequest",
    "powershell",
    "cmd.exe",
    "bash",
    "ExternalProject_Add",
    "FetchContent",
    "git clone"
)

$rootPath = (Resolve-Path $Root).Path
$files = Get-ChildItem -Path $rootPath -Recurse -File |
    Where-Object {
        $_.FullName -notmatch "\\target\\" -and
        $_.FullName -notmatch "\\build(_ci|_safety_check)?\\" -and
        $_.FullName -notmatch "\\.git\\"
    }

$results = @()
foreach ($file in $files) {
    foreach ($p in $patterns) {
        $hits = Select-String -Path $file.FullName -Pattern $p -SimpleMatch -ErrorAction SilentlyContinue
        if ($hits) {
            $results += $hits
        }
    }
}

if (-not $results) {
    Write-Host "No matching supply-chain execution patterns found."
    exit 0
}

$results |
    Select-Object Path, LineNumber, Line |
    Sort-Object Path, LineNumber |
    Format-Table -AutoSize
