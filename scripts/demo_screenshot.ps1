# Start the demo server, capture headless screenshots of the web UI, stop the server.
param(
  [string]$OutDir = "results\figures",
  [ValidateSet("single", "mixed", "wide")][string]$Mode = "single",
  [string]$Sample = "",
  [int]$Grams = 0,
  [int]$Port = 8000,
  [string]$Ckpt = "results/runs/resnet18_last_stage_aug/best.pt",
  [string]$WideCkpt = "results/runs/resnet18_food101_cn116/best.pt",
  [string]$Name = "demo_ui.png",
  [string]$Extra = "",
  [int]$Height = 830
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
Set-Location $root

$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = Join-Path (Split-Path -Parent $root) ".venv\Scripts\python.exe" }
if (-not (Test-Path $py)) { $py = "python" }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$env:FOODCLS_CKPT = $Ckpt
$env:FOODCLS_WIDE_CKPT = $WideCkpt
$env:PORT = "$Port"
$proc = Start-Process -FilePath $py -ArgumentList "-m","foodcls.app" -WorkingDirectory (Join-Path $root "src") -PassThru -WindowStyle Hidden
try {
  $ok = $false
  for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Milliseconds 1000
    try {
      $r = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/health" -TimeoutSec 2
      if ($r.ok) { $ok = $true; break }
    } catch { }
  }
  if (-not $ok) { throw "server did not become healthy" }

  if (-not $Sample) {
    $listUrl = if ($Mode -eq "mixed") { "/api/mixed/samples" } elseif ($Mode -eq "wide") { "/api/wide/samples" } else { "/api/samples" }
    $s = Invoke-RestMethod -Uri "http://127.0.0.1:$Port$listUrl"
    $Sample = $s.samples[0].name
  }
  Write-Host "mode = $Mode, sample = $Sample"

  $edge = "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
  $key = if ($Mode -eq "mixed") { "msample" } elseif ($Mode -eq "wide") { "wsample" } else { "sample" }
  $url = "http://127.0.0.1:$Port/?$key=" + [uri]::EscapeDataString($Sample)
  if ($Grams -gt 0) { $url += "&g=$Grams" }
  if ($Extra) { $url += $Extra }
  $shot = Join-Path $root (Join-Path $OutDir $Name)
  # a fresh profile is required, otherwise the running Edge instance swallows the flags
  $ud = Join-Path $env:TEMP ("edge_demo_" + [guid]::NewGuid().ToString("N"))
  $prev = $ErrorActionPreference
  $ErrorActionPreference = "Continue"   # Edge writes harmless warnings to stderr
  & $edge --headless=new --disable-gpu --no-first-run --hide-scrollbars `
      --user-data-dir="$ud" --force-device-scale-factor=2 `
      --window-size=1360,$Height --virtual-time-budget=20000 --run-all-compositor-stages-before-draw `
      --screenshot="$shot" "$url" 2>$null | Out-Null
  $ErrorActionPreference = $prev
  Remove-Item -Recurse -Force $ud -ErrorAction SilentlyContinue
  Start-Sleep -Seconds 2
  if (Test-Path $shot) { Write-Host "saved $shot" } else { throw "screenshot not created" }
} finally {
  if ($proc -and -not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
}
