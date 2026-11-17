<#
.SYNOPSIS
  Start the canteen-dish web demo and print every URL others can use.
.DESCRIPTION
  Binds 0.0.0.0 so phones/laptops on the same Wi-Fi can open the page.
  Usage:
      powershell -ExecutionPolicy Bypass -File scripts\start_demo.ps1
      powershell -ExecutionPolicy Bypass -File scripts\start_demo.ps1 -Port 8080 -Open
      powershell -ExecutionPolicy Bypass -File scripts\start_demo.ps1 -Qr   # 生成手机扫码图并打开
#>
param(
    [int]$Port = 8000,
    [string]$BindAddress = "0.0.0.0",
    [string]$Ckpt = "results/runs/resnet18_last_stage_aug_f101n/best.pt",
    [string]$MixedCkpt = "results/runs/resnet18_multilabel_plate40/best.pt",
    [string]$WideCkpt = "results/runs/resnet18_food101_cn116/best.pt",
    [switch]$Open,
    [switch]$Qr,
    [switch]$NoFirewallCheck
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = Join-Path (Split-Path -Parent $root) ".venv\Scripts\python.exe"
}
if (-not (Test-Path $python)) { throw "Python venv not found; see README section 2" }

$ckptPath = if ([System.IO.Path]::IsPathRooted($Ckpt)) { $Ckpt } else { Join-Path $root $Ckpt }
if (-not (Test-Path $ckptPath)) { throw "Checkpoint not found: $ckptPath" }

$env:FOODCLS_CKPT = $ckptPath
$mixedPath = if ([System.IO.Path]::IsPathRooted($MixedCkpt)) { $MixedCkpt } else { Join-Path $root $MixedCkpt }
if (Test-Path $mixedPath) {
    $env:FOODCLS_MIXED_CKPT = $mixedPath
} else {
    Write-Host "  Mixed-plate model not found; the mixed tab stays disabled." -ForegroundColor DarkGray
}
$widePath = if ([System.IO.Path]::IsPathRooted($WideCkpt)) { $WideCkpt } else { Join-Path $root $WideCkpt }
if (Test-Path $widePath) {
    $env:FOODCLS_WIDE_CKPT = $widePath
} else {
    Write-Host "  116-class model not found; that tab stays disabled." -ForegroundColor DarkGray
}
$env:FOODCLS_HOST = $BindAddress
$env:PORT = "$Port"

Write-Host ""
Write-Host "  Server starting - the address to share is printed below." -ForegroundColor Cyan
Write-Host "  Keep this window open while others use the page (Ctrl+C to stop)." -ForegroundColor Yellow
Write-Host ""

if (-not $NoFirewallCheck) {
    $rule = Get-NetFirewallRule -DisplayName "FoodDemo $Port" -ErrorAction SilentlyContinue
    if (-not $rule) {
        Write-Host "  If a phone cannot open the page, run once in an ADMIN PowerShell:" -ForegroundColor Yellow
        Write-Host "    New-NetFirewallRule -DisplayName 'FoodDemo $Port' -Direction Inbound -Protocol TCP -LocalPort $Port -Action Allow"
        Write-Host ""
    }
}

if ($Qr) {
    # A big scannable code for the projector; the desktop page also shows a small one.
    $qrCode = @"
import pathlib, socket, segno
ips = [i[4][0] for i in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
       if not i[4][0].startswith(("127.", "169.254."))]
ip = ips[0] if ips else "127.0.0.1"
url = f"http://{ip}:$Port"
out = pathlib.Path(r"$root") / "dist" / "phone_qr.png"
out.parent.mkdir(parents=True, exist_ok=True)
segno.make(url, error="m").save(out, scale=9, border=3, dark="#1b2430", light="#ffffff")
print(url)
print(out)
"@
    $qrScript = Join-Path $env:TEMP "foodcls_qr.py"
    Set-Content -Path $qrScript -Value $qrCode -Encoding UTF8
    $lines = & $python $qrScript
    if ($lines -and $lines.Count -ge 2) {
        Write-Host "  Phone QR : $($lines[0])  ->  $($lines[1])" -ForegroundColor Cyan
        Start-Process $lines[1]
    }
}

if ($Open) { Start-Process "http://127.0.0.1:$Port" }
Push-Location (Join-Path $root "src")
try {
    & $python -m foodcls.app
} finally {
    Pop-Location
}
