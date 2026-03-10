param(
    [string]$Path = "data/heartbeat_trader.json",
    [int]$IntervalMs = 1500
)

$ErrorActionPreference = "Stop"
Write-Host "Tailing heartbeat: $Path (Ctrl+C to stop)"

while ($true) {
    if (Test-Path $Path) {
        try {
            $hb = Get-Content $Path -Raw | ConvertFrom-Json
            $ts = Get-Date -Format "HH:mm:ss"
            $tick = $hb.tick
            $openPos = $hb.positions.open
            $pending = $hb.positions.pending
            $aiCalls = $hb.meters_60s.ai_calls
            $rejects = $hb.meters_60s.entry_reject_total
            $laneUnhealthy = $hb.lane_health.unhealthy_count
            $laneBlock = $hb.lane_health.block_entries
            $manualKill = $hb.manual_kill_switch.active
            Write-Host "[$ts] tick=$tick pos=$openPos pending=$pending ai60=$aiCalls reject60=$rejects lanes_bad=$laneUnhealthy block=$laneBlock manual_kill=$manualKill"
        } catch {
            Write-Host "[warn] failed to parse heartbeat JSON"
        }
    } else {
        Write-Host "[wait] heartbeat not found: $Path"
    }
    Start-Sleep -Milliseconds $IntervalMs
}

