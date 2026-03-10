param(
    [int]$IntervalSeconds = 30
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$root = Split-Path -Parent $scriptDir
Set-Location $root

$proc = $null
$push = $null

function Stop-Services {
    if ($push -and !$push.HasExited) {
        $push.Kill()
    }
    if ($proc -and !$proc.HasExited) {
        $proc.Kill()
    }
}

trap {
    Write-Host "Stopping Nemo services..."
    Stop-Services
    break
}

Write-Host "Starting Nemo WS service (foreground)..."
$proc = Start-Process python -ArgumentList "scripts/nemo_ws_service.py" -NoNewWindow -PassThru

Write-Host "Waiting for data/positions.json..."
while (-not (Test-Path "$root\data\positions.json")) {
    Start-Sleep -Seconds 1
}

Write-Host "Starting push_positions loop (foreground)..."
$push = Start-Process python -ArgumentList "scripts/push_positions.py","--loop","--interval",$IntervalSeconds -NoNewWindow -PassThru

Write-Host "Services running. Press Ctrl+C to stop."
while ($true) {
    Start-Sleep -Seconds 5
    if (($proc -and $proc.HasExited) -or ($push -and $push.HasExited)) {
        Write-Host "One of the subprocesses exited; shutting down."
        break
    }
}

Stop-Services
