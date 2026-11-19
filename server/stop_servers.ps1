<#
.SYNOPSIS
    Stop the CLARITY* FastAPI backend and Streamlit dashboard started by start_servers.ps1.

.DESCRIPTION
    Terminates the PIDs recorded in .run/, then frees the backend and UI ports
    (8000 and 8501 by default) in case the servers were started from separate
    terminals. Safe to run when nothing is running.

.EXAMPLE
    .\server\stop_servers.ps1

.EXAMPLE
    .\server\stop_servers.ps1 -Ports 8000,8600
#>
[CmdletBinding()]
param(
    [int[]]$Ports = @(8000, 8501)
)

$ErrorActionPreference = "Stop"

$serverDir  = $PSScriptRoot
$projectDir = Split-Path -Parent $serverDir
Set-Location $projectDir

$runDir = Join-Path $projectDir ".run"

function Stop-Tree {
    param([int]$ProcessId)
    $proc = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if (-not $proc) { return $false }
    # Streamlit/uvicorn can spawn children; stop the whole tree quietly.
    & taskkill /PID $ProcessId /T /F 2>$null | Out-Null
    return $true
}

$stopped = @()

foreach ($pidFile in @("backend.pid", "ui.pid")) {
    $path = Join-Path $runDir $pidFile
    if (Test-Path $path) {
        $procId = (Get-Content $path -ErrorAction SilentlyContinue | Select-Object -First 1)
        if ($procId -and $procId -match '^\d+$') {
            if (Stop-Tree -ProcessId ([int]$procId)) {
                $stopped += "$pidFile ($procId)"
            }
        }
        Remove-Item $path -Force -ErrorAction SilentlyContinue
    }
}

foreach ($port in $Ports) {
    try {
        $listeners = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction Stop
        foreach ($listener in $listeners) {
            if ($listener.OwningProcess -and $listener.OwningProcess -ne $PID) {
                if (Stop-Tree -ProcessId ([int]$listener.OwningProcess)) {
                    $stopped += "port $port (PID $($listener.OwningProcess))"
                }
            }
        }
    } catch {
        # Nothing listening on this port.
    }
}

Remove-Item (Join-Path $runDir "ports.txt") -Force -ErrorAction SilentlyContinue

if ($stopped.Count -gt 0) {
    Write-Host "[ok] Stopped: $($stopped -join ', ')" -ForegroundColor Green
} else {
    Write-Host "[ok] No CLARITY* servers were running." -ForegroundColor Green
}
