param(
    [string]$HeartbeatPath = "data/heartbeat_trader.json",
    [string]$WatchdogPath = "data/watchdog_status.json",
    [int]$WarnHeartbeatAgeSec = 20
)

$ErrorActionPreference = "Stop"

function Get-JsonFile {
    param([string]$Path)
    if (-not (Test-Path $Path)) { return $null }
    try {
        return (Get-Content $Path -Raw | ConvertFrom-Json)
    } catch {
        return $null
    }
}

$hb = Get-JsonFile -Path $HeartbeatPath
$wd = Get-JsonFile -Path $WatchdogPath

$now = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
$hbAge = $null
if ($hb -and $hb.ts) {
    $hbAge = [Math]::Max(0, [int]([double]$now - [double]$hb.ts))
}

$status = "OK"
if (-not $hb) { $status = "NO_HEARTBEAT" }
elseif ($hbAge -ge $WarnHeartbeatAgeSec) { $status = "STALE_HEARTBEAT" }
elseif ($wd -and $wd.ok -eq $false) { $status = "WATCHDOG_ALERT" }

Write-Host "Status: $status"
Write-Host "Heartbeat: $HeartbeatPath"
Write-Host "Watchdog:  $WatchdogPath"
if ($hbAge -ne $null) { Write-Host "HeartbeatAgeSec: $hbAge" }

if ($hb) {
    $tick = $hb.tick
    $openPos = $hb.positions.open
    $pending = $hb.positions.pending
    $entryReject = $hb.meters_60s.entry_reject_total
    $aiCalls = $hb.meters_60s.ai_calls
    $laneUnhealthy = $hb.lane_health.unhealthy_count
    $laneBlock = $hb.lane_health.block_entries
    $manualKill = $hb.manual_kill_switch.active
    $restThrottle = $hb.rest.throttle_hits_total
    $restBackoff = $hb.rest.backoff_hits_total

    Write-Host "Tick: $tick"
    Write-Host "Positions: open=$openPos pending=$pending"
    Write-Host "AI: calls_60s=$aiCalls entry_reject_60s=$entryReject"
    Write-Host "LaneHealth: unhealthy=$laneUnhealthy block_entries=$laneBlock"
    Write-Host "ManualKillSwitch: $manualKill"
    Write-Host "REST: throttle_hits_total=$restThrottle backoff_hits_total=$restBackoff"
}

if ($wd) {
    Write-Host "WatchdogOK: $($wd.ok)"
    if ($wd.fail_count -ne $null) { Write-Host "WatchdogFailCount: $($wd.fail_count)" }
    if ($wd.checked_count -ne $null) { Write-Host "WatchdogChecked: $($wd.checked_count)" }
    if ($wd.failing_paths -and $wd.failing_paths.Count -gt 0) {
        Write-Host "WatchdogFailingPaths:"
        $wd.failing_paths | ForEach-Object { Write-Host "  - $_" }
    }
}

if ($hb -and $hb.watchdog) {
    Write-Host "HB.Watchdog: ok=$($hb.watchdog.ok) block_entries=$($hb.watchdog.block_entries) kill_on_fail=$($hb.watchdog.kill_on_fail)"
}
if ($hb -and $hb.hardware_guard) {
    Write-Host "HB.HardwareGuard: level=$($hb.hardware_guard.level) block_entries=$($hb.hardware_guard.block_entries) ai_p95_ms=$($hb.hardware_guard.ai_latency_p95_ms) ai_parallel=$($hb.hardware_guard.effective_ai_parallel)"
}
