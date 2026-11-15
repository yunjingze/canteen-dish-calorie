<#
.SYNOPSIS
  Serve the canteen-dish demo to the whole internet with a Cloudflare quick tunnel.
.DESCRIPTION
  Starts the local FastAPI demo (bound to 127.0.0.1) and opens a temporary
  https URL of the form https://something.trycloudflare.com that works from any
  network - mobile data, another campus Wi-Fi, a friend's phone. No account, no
  domain, no firewall change: cloudflared dials out to Cloudflare and traffic
  comes back through that connection.
  The URL is temporary and changes every time the tunnel restarts. Keep this
  window open; Ctrl+C closes the tunnel (and the server if we started it).
  Usage:
      powershell -ExecutionPolicy Bypass -File scripts/start_public.ps1
      powershell -ExecutionPolicy Bypass -File scripts/start_public.ps1 -Port 8100
      powershell -ExecutionPolicy Bypass -File scripts/start_public.ps1 -NoServer
#>
param(
    [int]$Port = 8000,
    [string]$Ckpt = "results/runs/resnet18_last_stage_aug/best.pt",
    [string]$MixedCkpt = "results/runs/resnet18_multilabel_aug/best.pt",
    [string]$WideCkpt = "results/runs/resnet18_food101/best.pt",
    [switch]$NoServer,
    [switch]$SkipDownload
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv/Scripts/python.exe"
if (-not (Test-Path $py)) { $py = Join-Path (Split-Path -Parent $root) ".venv/Scripts/python.exe" }
if (-not (Test-Path $py)) { throw "Python venv not found; see README section 2" }

function Test-Health([int]$p) {
    try {
        $r = Invoke-RestMethod -Uri "http://127.0.0.1:$p/api/health" -TimeoutSec 2
        return [bool]$r.ok
    } catch { return $false }
}

$serverProc = $null
if (-not $NoServer -and -not (Test-Health $Port)) {
    $env:FOODCLS_HOST = "127.0.0.1"
    $env:PORT = "$Port"
    $env:FOODCLS_CKPT = (Join-Path $root $Ckpt)
    if (Test-Path (Join-Path $root $MixedCkpt)) { $env:FOODCLS_MIXED_CKPT = (Join-Path $root $MixedCkpt) }
    if (Test-Path (Join-Path $root $WideCkpt)) { $env:FOODCLS_WIDE_CKPT = (Join-Path $root $WideCkpt) }
    Write-Host "  Starting the demo server on port $Port ..." -ForegroundColor Cyan
    $serverProc = Start-Process -FilePath $py -ArgumentList "-m","foodcls.app" -WorkingDirectory (Join-Path $root "src") -PassThru -WindowStyle Hidden
    $ok = $false
    for ($i = 0; $i -lt 90; $i++) {
        Start-Sleep -Milliseconds 1000
        if (Test-Health $Port) { $ok = $true; break }
    }
    if (-not $ok) {
        if ($serverProc -and -not $serverProc.HasExited) { Stop-Process -Id $serverProc.Id -Force }
        throw "server did not become healthy on port $Port"
    }
}

$cf = Join-Path $root "tools/cloudflared.exe"
if (-not (Test-Path $cf)) {
    if ($SkipDownload) { throw "cloudflared not found at $cf" }
    New-Item -ItemType Directory -Force -Path (Join-Path $root "tools") | Out-Null
    Write-Host "  Downloading cloudflared (about 50 MB, one time) ..." -ForegroundColor Cyan
    $url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
    Invoke-WebRequest -Uri $url -OutFile $cf -TimeoutSec 300 -UseBasicParsing
}

$log = Join-Path $env:TEMP ("cloudflared_" + [guid]::NewGuid().ToString("N") + ".log")
$tunnel = Start-Process -FilePath $cf -ArgumentList "tunnel","--url","http://127.0.0.1:$Port","--no-autoupdate" -RedirectStandardError $log -RedirectStandardOutput ($log + ".out") -PassThru -WindowStyle Hidden

try {
    $public = $null
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Milliseconds 1000
        if (Test-Path $log) {
            $text = Get-Content $log -Raw -ErrorAction SilentlyContinue
            if ($text -match "https://[a-z0-9-]+\.trycloudflare\.com") { $public = $Matches[0]; break }
        }
        if ($tunnel.HasExited) { break }
    }
    if (-not $public) { throw "tunnel did not report a public URL (see $log)" }

    Write-Host ""
    Write-Host "  Public URL (works on any network, no same-Wi-Fi needed):" -ForegroundColor Green
    Write-Host "    $public" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Share it, or open it on a phone and choose 'Add to Home Screen'." -ForegroundColor Gray
    Write-Host "  The link is temporary: it changes each time this script restarts." -ForegroundColor Yellow
    Write-Host "  Anyone with the link can use the demo (no login). Press Ctrl+C to stop." -ForegroundColor Yellow
    Write-Host ""

    while (-not $tunnel.HasExited) { Start-Sleep -Seconds 2 }
} finally {
    if ($tunnel -and -not $tunnel.HasExited) { Stop-Process -Id $tunnel.Id -Force }
    if ($serverProc -and -not $serverProc.HasExited) { Stop-Process -Id $serverProc.Id -Force }
    Remove-Item -Force $log, ($log + ".out") -ErrorAction SilentlyContinue
}
