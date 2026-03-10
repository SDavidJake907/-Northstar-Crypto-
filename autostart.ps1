# autostart.ps1 — Keeps hybrid_kraken_core.exe running permanently.
# If the bot dies, this restarts it automatically after a short cooldown.
# Run this script once; it loops forever.

$BotExe   = "rust_core\target\release\hybrid_kraken_core.exe"
$WorkDir  = "C:\Users\kitti\Projects\kraken-hybrad9"
$LogFile  = "logs\bot_out.log"
$KillFile = "data\manual_kill_switch.json"

Set-Location $WorkDir

function Is-KillSwitchOn {
    try {
        $k = Get-Content $KillFile -Raw | ConvertFrom-Json
        return $k.kill -eq $true
    } catch { return $false }
}

Write-Host "[AUTOSTART] Bot watchdog started. Press Ctrl+C to stop."

while ($true) {
    # Don't restart if kill switch is on
    if (Is-KillSwitchOn) {
        Write-Host "[AUTOSTART] Kill switch ON — waiting..."
        Start-Sleep -Seconds 5
        continue
    }

    # Start the bot
    Write-Host "[AUTOSTART] Starting bot at $(Get-Date -Format 'HH:mm:ss')..."
    $proc = Start-Process -FilePath $BotExe `
        -WorkingDirectory $WorkDir `
        -RedirectStandardOutput $LogFile `
        -WindowStyle Hidden `
        -PassThru

    Write-Host "[AUTOSTART] Bot PID: $($proc.Id)"

    # Wait for it to exit
    $proc.WaitForExit()
    $code = $proc.ExitCode

    Write-Host "[AUTOSTART] Bot exited with code $code at $(Get-Date -Format 'HH:mm:ss')"

    # If kill switch turned on while running, don't restart
    if (Is-KillSwitchOn) {
        Write-Host "[AUTOSTART] Kill switch ON — not restarting."
        continue
    }

    Write-Host "[AUTOSTART] Restarting in 10 seconds..."
    Start-Sleep -Seconds 10
}
