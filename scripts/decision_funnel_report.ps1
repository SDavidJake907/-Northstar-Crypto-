$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$logPath = Join-Path $root "logs\bot.log"
$reportPath = Join-Path $root "docs\decision_funnel_report.md"

$tailLines = 5000
$lines = Get-Content $logPath -Tail $tailLines -ErrorAction Stop

$counts = [ordered]@{
  ticks = 0
  engine_lines = 0
  entry_skip = 0
  enter_live = 0
  enter_paper = 0
  ai_override = 0
  math_advisory = 0
  threshold_advisory = 0
  hf_conf_advisory = 0
  sentiment_block = 0
  loss_cooldown = 0
  btc_crash = 0
}

$passedSum = 0
$warnSum = 0
$hardSum = 0

$rejectReasons = [ordered]@{
  liq_low = 0
  vol_low = 0
  spread_high = 0
  slip_high = 0
  fill_missing = 0
  slip_missing = 0
  illiquid = 0
  atr_spike = 0
  score_low = 0
}

$engineRegex = [regex]"(\d+) passed gates, (\d+) gate-warned, (\d+) hard-rejected"
$gateDetailRegex = [regex]"hard_reject=\[(.*?)\]\s+gate_reject=\[(.*?)\]\s+warnings=\[(.*?)\]"
$gateTopRegex = [regex]"hard=\[(.*?)\]\s+reject=\[(.*?)\]\s+warn=\[(.*?)\]"
$decisionSummaryRegex = [regex]"math BUY=(\d+)\s+SELL=(\d+)\s+HOLD=(\d+)"
$entryRejectRegex = [regex]"top=(.*)"
$entrySummaryRegex = [regex]"candidates=(\d+)\s+entry_candidates=(\d+)"
$detailHard = @{}
$detailReject = @{}
$detailWarn = @{}
$gateTopHard = @{}
$gateTopReject = @{}
$gateTopWarn = @{}
$decisionSummary = @()
$entryRejectTop = @()
$entrySummary = @()

function Add-Counts {
  param([hashtable]$map, [string]$csv)
  if (-not $csv) { return }
  foreach ($part in $csv.Split(",")) {
    $p = $part.Trim()
    if (-not $p) { continue }
    $kv = $p.Split("=", 2)
    if ($kv.Count -ne 2) { continue }
    $k = $kv[0].Trim()
    $v = 0
    [void][int]::TryParse($kv[1].Trim(), [ref]$v)
    if ($map.ContainsKey($k)) { $map[$k] += $v } else { $map[$k] = $v }
  }
}

foreach ($line in $lines) {
  if ($line.Contains("[TICK] completed")) { $counts.ticks++ }
  if ($line.Contains("[ENGINE]")) {
    $m = $engineRegex.Match($line)
    if ($m.Success) {
      $counts.engine_lines++
      $passedSum += [int]$m.Groups[1].Value
      $warnSum += [int]$m.Groups[2].Value
      $hardSum += [int]$m.Groups[3].Value
    }
  }
  if ($line.Contains("[ENTRY-SKIP]")) { $counts.entry_skip++ }
  if ($line.Contains("[ENTER-")) { $counts.enter_live++ }
  if ($line.Contains("[PAPER-ENTER]")) { $counts.enter_paper++ }
  if ($line.Contains("[AI-OVERRIDE]")) { $counts.ai_override++ }
  if ($line.Contains("[MATH-ADVISORY]")) { $counts.math_advisory++ }
  if ($line.Contains("[THRESHOLD-ADVISORY]")) { $counts.threshold_advisory++ }
  if ($line.Contains("[HF-CONF-ADVISORY]")) { $counts.hf_conf_advisory++ }

  $lower = $line.ToLower()
  if ($lower.Contains("sentiment") -and $lower.Contains("gate") -and $lower.Contains("block")) {
    $counts.sentiment_block++
  }
  if ($line.Contains("[LOSS-COOLDOWN]")) { $counts.loss_cooldown++ }
  if ($line.Contains("BTC crash")) { $counts.btc_crash++ }

  foreach ($k in @($rejectReasons.Keys)) {
    if ($line.Contains($k)) { $rejectReasons[$k]++ }
  }

  if ($line.Contains("[GATE-DETAIL]")) {
    $gm = $gateDetailRegex.Match($line)
    if ($gm.Success) {
      Add-Counts -map $detailHard -csv $gm.Groups[1].Value
      Add-Counts -map $detailReject -csv $gm.Groups[2].Value
      Add-Counts -map $detailWarn -csv $gm.Groups[3].Value
    }
  }
  if ($line.Contains("[GATE-TOP]")) {
    $gt = $gateTopRegex.Match($line)
    if ($gt.Success) {
      Add-Counts -map $gateTopHard -csv $gt.Groups[1].Value
      Add-Counts -map $gateTopReject -csv $gt.Groups[2].Value
      Add-Counts -map $gateTopWarn -csv $gt.Groups[3].Value
    }
  }
  if ($line.Contains("[DECISION-SUMMARY]")) {
    $dm = $decisionSummaryRegex.Match($line)
    if ($dm.Success) {
      $decisionSummary += "BUY=$($dm.Groups[1].Value) SELL=$($dm.Groups[2].Value) HOLD=$($dm.Groups[3].Value)"
    }
  }
  if ($line.Contains("[ENTRY-REJECT]")) {
    $em = $entryRejectRegex.Match($line)
    if ($em.Success) { $entryRejectTop += $em.Groups[1].Value.Trim() }
  }
  if ($line.Contains("[ENTRY-SUMMARY]")) {
    $es = $entrySummaryRegex.Match($line)
    if ($es.Success) { $entrySummary += "candidates=$($es.Groups[1].Value) entry_candidates=$($es.Groups[2].Value)" }
  }
}

$avgPassed = if ($counts.engine_lines -gt 0) { [Math]::Round($passedSum / $counts.engine_lines, 2) } else { 0 }
$avgWarn = if ($counts.engine_lines -gt 0) { [Math]::Round($warnSum / $counts.engine_lines, 2) } else { 0 }
$avgHard = if ($counts.engine_lines -gt 0) { [Math]::Round($hardSum / $counts.engine_lines, 2) } else { 0 }

$report = @()
$report += "# Decision Funnel Report"
$report += ""
$report += "Source log: $logPath"
$report += ""
$report += "## Summary"
$report += "Ticks: $($counts.ticks)"
$report += "Engine snapshots: $($counts.engine_lines)"
$report += "Entries (live): $($counts.enter_live)"
$report += "Entries (paper): $($counts.enter_paper)"
$report += "Entry skips: $($counts.entry_skip)"
$report += "AI overrides: $($counts.ai_override)"
$report += ""
$report += "## Gate Flow Averages (per engine snapshot)"
$report += "Passed gates: $avgPassed"
$report += "Gate warned: $avgWarn"
$report += "Hard rejected: $avgHard"
$report += ""
$report += "## Advisory Counts"
$report += "Math advisory: $($counts.math_advisory)"
$report += "Threshold advisory: $($counts.threshold_advisory)"
$report += "HF conf advisory: $($counts.hf_conf_advisory)"
$report += "Sentiment gate blocks: $($counts.sentiment_block)"
$report += "Loss cooldown blocks: $($counts.loss_cooldown)"
$report += "BTC crash blocks: $($counts.btc_crash)"
$report += ""
$report += "## Gate Rejection Reasons (best-effort from logs)"
foreach ($k in ($rejectReasons.GetEnumerator() | Sort-Object Value -Descending)) {
  $report += "$($k.Key): $($k.Value)"
}

$report += ""
$report += "## Gate Detail Summary (from [GATE-DETAIL])"
if ($detailHard.Count -eq 0 -and $detailReject.Count -eq 0 -and $detailWarn.Count -eq 0) {
  $report += "No [GATE-DETAIL] lines found."
} else {
  if ($detailHard.Count -gt 0) {
    $report += "Hard rejects:"
    foreach ($k in ($detailHard.GetEnumerator() | Sort-Object Value -Descending)) {
      $report += "  $($k.Key): $($k.Value)"
    }
  }
  if ($detailReject.Count -gt 0) {
    $report += "Gate rejects:"
    foreach ($k in ($detailReject.GetEnumerator() | Sort-Object Value -Descending)) {
      $report += "  $($k.Key): $($k.Value)"
    }
  }
  if ($detailWarn.Count -gt 0) {
    $report += "Warnings:"
    foreach ($k in ($detailWarn.GetEnumerator() | Sort-Object Value -Descending)) {
      $report += "  $($k.Key): $($k.Value)"
    }
  }
}

$report += ""
$report += "## Gate Top Summary (from [GATE-TOP])"
if ($gateTopHard.Count -eq 0 -and $gateTopReject.Count -eq 0 -and $gateTopWarn.Count -eq 0) {
  $report += "No [GATE-TOP] lines found."
} else {
  if ($gateTopHard.Count -gt 0) {
    $report += "Hard (aggregated):"
    foreach ($k in ($gateTopHard.GetEnumerator() | Sort-Object Value -Descending)) { $report += "  $($k.Key): $($k.Value)" }
  }
  if ($gateTopReject.Count -gt 0) {
    $report += "Reject (aggregated):"
    foreach ($k in ($gateTopReject.GetEnumerator() | Sort-Object Value -Descending)) { $report += "  $($k.Key): $($k.Value)" }
  }
  if ($gateTopWarn.Count -gt 0) {
    $report += "Warn (aggregated):"
    foreach ($k in ($gateTopWarn.GetEnumerator() | Sort-Object Value -Descending)) { $report += "  $($k.Key): $($k.Value)" }
  }
}

$report += ""
$report += "## Decision Summary Samples (from [DECISION-SUMMARY])"
if ($decisionSummary.Count -eq 0) { $report += "No [DECISION-SUMMARY] lines found." }
else { $report += ($decisionSummary | Select-Object -Last 5) }

$report += ""
$report += "## Entry Summary Samples (from [ENTRY-SUMMARY] / [ENTRY-REJECT])"
if ($entrySummary.Count -eq 0 -and $entryRejectTop.Count -eq 0) {
  $report += "No [ENTRY-SUMMARY]/[ENTRY-REJECT] lines found."
} else {
  if ($entrySummary.Count -gt 0) {
    $report += "Entry summary:"
    $report += ($entrySummary | Select-Object -Last 5)
  }
  if ($entryRejectTop.Count -gt 0) {
    $report += "Entry reject top:"
    $report += ($entryRejectTop | Select-Object -Last 5)
  }
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $reportPath) | Out-Null
$report | Set-Content -Path $reportPath -Encoding UTF8

Write-Host "Wrote report to $reportPath"
