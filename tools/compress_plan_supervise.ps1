# Keeps the compression plan (tools/compress_plan.cpp) running until it reports done (Windows PowerShell 5.1 /
# PowerShell 7): restarts it when it dies, and kills and restarts it when its heartbeat goes stale. Jobs resume
# from their checkpoints and finished jobs are skipped. The POSIX version is compress_plan_supervise.sh.
#
#   powershell -ExecutionPolicy Bypass -File tools\compress_plan_supervise.ps1 [PHASES]
#
# Environment: everything compress_plan.cpp reads (FORGE_BIN, COMPRESS_RUNS, COMPRESS_MEM_LIMIT, ...), and
#   COMPRESS_PLAN, COMPRESS_MAX_RESTARTS (1000), COMPRESS_RESTART_DELAY (60 s), COMPRESS_HEARTBEAT_STALE (30 min; 0 = never)
# Exit status: 0 when the plan is done, 75 when another plan instance holds the lock, 1 when it gave up.
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$PlanArgs)

function EnvOr([string]$k, [string]$d) { $v = [Environment]::GetEnvironmentVariable($k); if ([string]::IsNullOrEmpty($v)) { $d } else { $v } }

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$forge = EnvOr 'FORGE_BIN' 'forge'
$plan = EnvOr 'COMPRESS_PLAN' (Join-Path $here 'compress_plan.cpp')
$runs = EnvOr 'COMPRESS_RUNS' (Join-Path $env:USERPROFILE 'runs')
$maxRestarts = [int](EnvOr 'COMPRESS_MAX_RESTARTS' '1000')
$delay = [int](EnvOr 'COMPRESS_RESTART_DELAY' '60')
$stale = [int](EnvOr 'COMPRESS_HEARTBEAT_STALE' '30')
$env:FORGE_BIN = $forge
$env:COMPRESS_RUNS = $runs
New-Item -ItemType Directory -Force -Path $runs | Out-Null
$hb = Join-Path $runs 'plan.heartbeat'
$done = Join-Path $runs 'plan.done'
$log = Join-Path $runs 'supervise.log'

function Say([string]$m) { $line = "[{0}] supervise: {1}" -f (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ'), $m; Write-Host $line; Add-Content -Path $log -Value $line }

$restarts = 0
while ($true) {
    Say "starting the plan (restart $restarts)"
    $argList = @('script', "`"$plan`"") + ($PlanArgs | ForEach-Object { "`"$_`"" })
    $p = Start-Process -FilePath $forge -ArgumentList $argList -NoNewWindow -PassThru
    while (-not $p.WaitForExit(30000)) {
        if ($stale -gt 0 -and (Test-Path $hb)) {
            $age = (Get-Date) - (Get-Item $hb).LastWriteTime
            if ($age.TotalMinutes -gt $stale) {
                Say ("heartbeat older than {0} min: killing the plan (pid {1})" -f $stale, $p.Id)
                # the whole tree: the plan and the forge compress / eval job it runs
                & taskkill.exe /PID $p.Id /T /F | Out-Null
            }
        }
    }
    $rc = $p.ExitCode
    if ($rc -eq 0 -and (Test-Path $done)) { Say 'the plan is done'; exit 0 }
    if ($rc -eq 75) { Say "another plan instance is running (lock $runs\plan.lock)"; exit 75 }
    $restarts++
    if ($restarts -gt $maxRestarts) { Say "giving up after $maxRestarts restarts (last exit $rc)"; exit 1 }
    Say "the plan exited with status $rc; restarting in ${delay}s"
    Start-Sleep -Seconds $delay
}
