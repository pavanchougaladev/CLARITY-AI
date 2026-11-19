<#
.SYNOPSIS
    Start the CLARITY* suite on Windows: FastAPI backend (port 8000) + Streamlit UI (port 8501).

.DESCRIPTION
    Frees the two ports, launches uvicorn and Streamlit from the project virtual
    environment, waits for both health endpoints, records the PIDs in .run/ and
    opens the dashboard in the default browser.

    Stop everything with:  .\server\stop_servers.ps1

.PARAMETER BackendPort
    Port for the FastAPI backend. Default 8000.

.PARAMETER UiPort
    Port for the Streamlit dashboard. Default 8501.

.PARAMETER NoBrowser
    Do not open the dashboard automatically.

.EXAMPLE
    .\server\start_servers.ps1

.EXAMPLE
    .\server\start_servers.ps1 -UiPort 8600 -NoBrowser
#>
[CmdletBinding()]
param(
    [int]$BackendPort = 8000,
    [int]$UiPort = 8501,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"

$serverDir  = $PSScriptRoot
$projectDir = Split-Path -Parent $serverDir
Set-Location $projectDir

# ---- Locate the interpreter (prefer the project venv) ------------------------
$python = Join-Path $projectDir ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $cmd) {
        throw "Python not found. Create the virtual environment first: py -3.12 -m venv .venv"
    }
    $python = $cmd.Source
    Write-Host "[warn] .venv not found; using $python" -ForegroundColor Yellow
}

# server/ hosts api/ and src/, so both directories must be importable.
$env:PYTHONPATH = "$serverDir;$projectDir"
$env:PYTHONUNBUFFERED = "1"

$runDir = Join-Path $projectDir ".run"
$logDir = Join-Path $projectDir "logs"
New-Item -ItemType Directory -Force -Path $runDir, $logDir | Out-Null

function Stop-Port {
    param([int]$Port)
    try {
        $listeners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop
        foreach ($listener in $listeners) {
            if ($listener.OwningProcess -and $listener.OwningProcess -ne $PID) {
                Stop-Process -Id $listener.OwningProcess -Force -ErrorAction SilentlyContinue
            }
        }
    } catch {
        # Nothing listening on this port - that is the normal case.
    }
}

function Wait-Http {
    param([string]$Url, [int]$TimeoutSeconds = 60)
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $response = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 2
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) { return $true }
        } catch {
            Start-Sleep -Milliseconds 750
        }
    }
    return $false
}

Write-Host "==================================================================" -ForegroundColor DarkGray
Write-Host " CLARITY* - Chandrayaan Lunar AI Registration & 3D Mapping Suite" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor DarkGray
Write-Host " Project : $projectDir"
Write-Host " Python  : $python"

# ---- Free the ports ----------------------------------------------------------
Stop-Port $BackendPort
Stop-Port $UiPort
Start-Sleep -Milliseconds 400

# ---- 1/2 FastAPI backend -----------------------------------------------------
Write-Host "[1/2] Starting FastAPI backend on port $BackendPort ..." -ForegroundColor Yellow
$backend = Start-Process -FilePath $python `
    -ArgumentList @("-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "$BackendPort") `
    -WorkingDirectory $serverDir `
    -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $logDir "backend.out.log") `
    -RedirectStandardError  (Join-Path $logDir "backend.err.log")

$healthUrl = "http://127.0.0.1:$BackendPort/api/v1/health"
if (Wait-Http -Url $healthUrl -TimeoutSeconds 60) {
    Write-Host "      OK  API online  -> http://localhost:$BackendPort/docs" -ForegroundColor Green
} else {
    Write-Host "      WARN backend did not answer $healthUrl" -ForegroundColor Yellow
    Write-Host "           see logs\backend.err.log" -ForegroundColor Yellow
}

# ---- 2/2 Streamlit UI --------------------------------------------------------
Write-Host "[2/2] Starting Streamlit dashboard on port $UiPort ..." -ForegroundColor Yellow
$ui = Start-Process -FilePath $python `
    -ArgumentList @("-m", "streamlit", "run", "client/app.py",
                    "--server.port", "$UiPort",
                    "--server.headless", "true",
                    "--browser.gatherUsageStats", "false") `
    -WorkingDirectory $projectDir `
    -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $logDir "ui.out.log") `
    -RedirectStandardError  (Join-Path $logDir "ui.err.log")

$uiHealth = "http://127.0.0.1:$UiPort/_stcore/health"
if (Wait-Http -Url $uiHealth -TimeoutSeconds 90) {
    Write-Host "      OK  Dashboard online -> http://localhost:$UiPort" -ForegroundColor Green
} else {
    Write-Host "      WARN dashboard did not answer $uiHealth" -ForegroundColor Yellow
    Write-Host "           see logs\ui.err.log" -ForegroundColor Yellow
}

# ---- Bookkeeping -------------------------------------------------------------
Set-Content -Path (Join-Path $runDir "backend.pid") -Value $backend.Id -Encoding ascii
Set-Content -Path (Join-Path $runDir "ui.pid")      -Value $ui.Id      -Encoding ascii
Set-Content -Path (Join-Path $runDir "ports.txt")   -Value @($BackendPort, $UiPort) -Encoding ascii

Write-Host "------------------------------------------------------------------" -ForegroundColor DarkGray
Write-Host " Dashboard : http://localhost:$UiPort"
Write-Host " API docs  : http://localhost:$BackendPort/docs"
Write-Host " Stop with : .\server\stop_servers.ps1"
Write-Host " Logs      : $logDir"
Write-Host "------------------------------------------------------------------" -ForegroundColor DarkGray

if (-not $NoBrowser) {
    Start-Process "http://localhost:$UiPort"
}
