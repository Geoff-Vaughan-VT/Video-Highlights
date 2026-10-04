<#
.SYNOPSIS
  Windows native runner for NVIDIA RTX rigs: .venv + CUDA torch, API + queue worker.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\run_native_windows.ps1
  powershell -ExecutionPolicy Bypass -File scripts\run_native_windows.ps1 -Inline
  powershell -ExecutionPolicy Bypass -File scripts\run_native_windows.ps1 -TorchIndex cpu -Port 8080

.NOTES
  Prereqs: Python 3.11 (py launcher), ffmpeg on PATH (winget install Gyan.FFmpeg),
  a current NVIDIA driver. Data lives in .\data, weights in .\models.
#>
param(
    [string]$TorchIndex = "",          # cu128 (default with NVIDIA), cpu, or a full URL
    [int]$Port = 0,
    [switch]$Inline,                   # API runs jobs in-process, no worker
    [switch]$SkipInstall
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

function Log($msg) { Write-Host "[vh] $msg" }

# .env (KEY=VALUE) without overriding variables already set in the session.
$envFile = Join-Path $Repo ".env"
if (Test-Path $envFile) {
    foreach ($line in Get-Content $envFile) {
        if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)=(.*)$' -and -not (Test-Path "Env:$($Matches[1])")) {
            Set-Item -Path "Env:$($Matches[1])" -Value $Matches[2]
        }
    }
}

# GPU + torch index.
$hasNvidia = $false
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    & nvidia-smi -L
    if ($LASTEXITCODE -eq 0) { $hasNvidia = $true }
}
if (-not $TorchIndex) { $TorchIndex = $(if ($hasNvidia) { "cu128" } else { "cpu" }) }
if (-not $hasNvidia) { Log "WARNING: nvidia-smi not found/failed; installing CPU torch (processing will be slow)" }
if ($TorchIndex -notmatch '^https?://') { $TorchIndex = "https://download.pytorch.org/whl/$TorchIndex" }

# ffmpeg.
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue) -and -not $env:VH_FFMPEG -and -not $env:VH_FFMPEG_BIN) {
    throw "ffmpeg not found. Install with: winget install Gyan.FFmpeg (then reopen the terminal)"
}
if (Get-Command ffmpeg -ErrorAction SilentlyContinue) {
    $encoders = & ffmpeg -hide_banner -encoders 2>$null | Out-String
    if ($encoders -match 'h264_nvenc') { Log "ffmpeg has h264_nvenc" } else { Log "WARNING: ffmpeg lacks h264_nvenc; renders use libx264" }
}

# venv + dependencies (re-installed only when requirements or the index change).
$venvPy = Join-Path $Repo ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    Log "creating .venv (Python 3.11)"
    if (Get-Command py -ErrorAction SilentlyContinue) { & py -3.11 -m venv .venv }
    if (-not (Test-Path $venvPy)) { & python -m venv .venv }
    if (-not (Test-Path $venvPy)) { throw "Could not create .venv; install Python 3.11 from python.org" }
}
$stampFile = Join-Path $Repo ".venv\.vh-install-stamp"
$reqHash = (Get-FileHash requirements-base.txt).Hash + (Get-FileHash requirements-ml.txt).Hash
$want = "$reqHash-$TorchIndex"
$have = if (Test-Path $stampFile) { Get-Content $stampFile -Raw } else { "" }
if (-not $SkipInstall -and $have.Trim() -ne $want) {
    & $venvPy -m pip install --upgrade pip wheel
    Log "installing torch from $TorchIndex"
    & $venvPy -m pip install torch torchvision --index-url $TorchIndex
    if ($LASTEXITCODE -ne 0) { throw "torch install failed" }
    & $venvPy -m pip install -r requirements-ml.txt
    if ($LASTEXITCODE -ne 0) { throw "requirements install failed" }
    Set-Content -Path $stampFile -Value $want
} else { Log "dependencies up to date" }

# Runtime defaults (forward slashes keep SQLite URLs valid).
$data = if ($env:VH_DATA_DIR) { $env:VH_DATA_DIR } else { (Join-Path $Repo "data") -replace '\\', '/' }
if (-not $env:VH_DB_URL) { $env:VH_DB_URL = "sqlite:///$data/db/video_highlights.db" }
if (-not $env:VH_OUTPUT_ROOT) { $env:VH_OUTPUT_ROOT = "$data/outputs" }
if (-not $env:VH_LOCAL_STORAGE_ROOT) { $env:VH_LOCAL_STORAGE_ROOT = "$data/storage" }
if (-not $env:VH_MODEL_DIR) { $env:VH_MODEL_DIR = ((Join-Path $Repo "models") -replace '\\', '/') }
if (-not $env:VH_DEVICE) { $env:VH_DEVICE = "auto" }
if (-not $env:VH_SKIP_USER_MANAGEMENT) { $env:VH_SKIP_USER_MANAGEMENT = "true" }
$env:VH_JOB_EXECUTION_MODE = $(if ($Inline) { "inline" } elseif ($env:VH_JOB_EXECUTION_MODE) { $env:VH_JOB_EXECUTION_MODE } else { "queue" })
if ($Port -eq 0) { $Port = $(if ($env:VH_API_PORT) { [int]$env:VH_API_PORT } else { 8000 }) }
foreach ($d in @("$data/db", $env:VH_OUTPUT_ROOT, $env:VH_LOCAL_STORAGE_ROOT, $env:VH_MODEL_DIR)) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
}

$models = if ($env:VH_MODELS) { $env:VH_MODELS -split '\s+' } else { @("yolov8n.pt", "yolov8s.pt") }
& $venvPy scripts\download_models.py --model-dir $env:VH_MODEL_DIR --allow-missing @models
& $venvPy -c "from backend.services.gpu_status import get_gpu_status as g; s=g(); print('[vh] device=%s hardware_class=%s encoder=%s hwaccel=%s' % (s['recommended_device'], s['hardware_class'], s['recommended_encoder'], s['recommended_hwaccel']))"

# API, then the worker once /v1/health answers.
$api = Start-Process -FilePath $venvPy -ArgumentList @("-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "$Port") -NoNewWindow -PassThru
$worker = $null
try {
    $ready = $false
    for ($i = 0; $i -lt 60 -and -not $ready; $i++) {
        Start-Sleep -Seconds 1
        if ($api.HasExited) { throw "API exited with code $($api.ExitCode)" }
        try { $ready = (Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:$Port/v1/health" -TimeoutSec 2).StatusCode -eq 200 } catch { }
    }
    if (-not $ready) { throw "API did not become healthy on port $Port" }
    if ($env:VH_JOB_EXECUTION_MODE -eq "queue") {
        $worker = Start-Process -FilePath $venvPy -ArgumentList @("-m", "backend.worker") -NoNewWindow -PassThru
    }
    Log "Studio: http://localhost:$Port   (mode: $($env:VH_JOB_EXECUTION_MODE), data: $data). Ctrl+C to stop."
    while (-not $api.HasExited -and ($null -eq $worker -or -not $worker.HasExited)) { Start-Sleep -Seconds 2 }
} finally {
    foreach ($p in @($worker, $api)) { if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue } }
}
