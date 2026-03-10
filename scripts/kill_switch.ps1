param(
    [ValidateSet("on", "off", "status")]
    [string]$Mode = "status",
    [string]$EnvPath = ".env"
)

$ErrorActionPreference = "Stop"

function Read-EnvLines {
    param([string]$Path)
    if (Test-Path $Path) {
        return [System.Collections.Generic.List[string]](Get-Content $Path)
    }
    return [System.Collections.Generic.List[string]]::new()
}

function Set-EnvKey {
    param(
        [System.Collections.Generic.List[string]]$Lines,
        [string]$Key,
        [string]$Value
    )
    $prefix = "$Key="
    $updated = $false
    for ($i = 0; $i -lt $Lines.Count; $i++) {
        if ($Lines[$i] -like "$prefix*") {
            $Lines[$i] = "$prefix$Value"
            $updated = $true
            break
        }
    }
    if (-not $updated) {
        $Lines.Add("$prefix$Value")
    }
}

$lines = Read-EnvLines -Path $EnvPath

if ($Mode -eq "status") {
    $line = $lines | Where-Object { $_ -like "MANUAL_KILL_SWITCH=*" } | Select-Object -First 1
    if ($line) {
        Write-Host $line
    } else {
        Write-Host "MANUAL_KILL_SWITCH=0 (default)"
    }
    exit 0
}

if ($Mode -eq "on") {
    Set-EnvKey -Lines $lines -Key "MANUAL_KILL_SWITCH" -Value "1"
    $newValue = "1"
} else {
    Set-EnvKey -Lines $lines -Key "MANUAL_KILL_SWITCH" -Value "0"
    $newValue = "0"
}

$lines | Set-Content -Path $EnvPath -Encoding UTF8
Write-Host "Updated $EnvPath: MANUAL_KILL_SWITCH=$newValue"
