<#
.SYNOPSIS
  Package the demo so someone else can run it on their own Windows PC.
.DESCRIPTION
  Produces dist\FoodDemo.zip containing the code, the trained checkpoint,
  a handful of demo images and a launcher. The receiver only needs Python 3.10+
  and to install requirements.txt (~2 GB of PyTorch, downloaded on their side).
#>
param(
    [string]$OutDir = "dist",
    [int]$Samples = 40
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$stage = Join-Path $root "$OutDir\FoodDemo"
$zip = Join-Path $root "$OutDir\FoodDemo.zip"

if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory -Path $stage -Force | Out-Null

Copy-Item (Join-Path $root "src") (Join-Path $stage "src") -Recurse
Copy-Item (Join-Path $root "requirements.txt") $stage
Copy-Item (Join-Path $root "scripts\start_demo.ps1") (Join-Path $stage "start_demo.ps1")
Copy-Item (Join-Path $root "scripts\start_public.ps1") (Join-Path $stage "start_public.ps1")
foreach ($s in @("setup_cloudflare_tunnel.ps1", "start_permanent.ps1", "start_funnel.ps1", "start_ngrok.ps1")) {
    Copy-Item (Join-Path $root "scripts\$s") (Join-Path $stage $s)
}
if (Test-Path (Join-Path $root "docs")) {
    Copy-Item (Join-Path $root "docs") (Join-Path $stage "docs") -Recurse
}
New-Item -ItemType Directory -Path (Join-Path $stage "results\runs\resnet18_last_stage_aug") -Force | Out-Null
Copy-Item (Join-Path $root "results\runs\resnet18_last_stage_aug\best.pt") (Join-Path $stage "results\runs\resnet18_last_stage_aug\best.pt")
# without this the stage silently falls back to T=1.0 and every probability,
# confidence bar and 0.32 gate decision differs from the measured model
$mainCal = Join-Path $root "results\runs\resnet18_last_stage_aug\calibration.json"
if (Test-Path $mainCal) { Copy-Item $mainCal (Join-Path $stage "results\runs\resnet18_last_stage_aug\calibration.json") }

# keep the bundle clean: no bytecode caches, no TorchInductor scratch directories
$junk = Get-ChildItem (Join-Path $stage "src") -Recurse -Directory |
    Where-Object { $_.Name -eq "__pycache__" -or $_.Name -like "torchinductor_*" }
if ($junk) { $junk | Remove-Item -Recurse -Force }

# multi-label head for mixed trays (optional: the single-dish tab works without it)
$mixedRun = Join-Path $root "results\runs\resnet18_multilabel_aug"
if (Test-Path (Join-Path $mixedRun "best.pt")) {
    New-Item -ItemType Directory -Path (Join-Path $stage "results\runs\resnet18_multilabel_aug") -Force | Out-Null
    Copy-Item (Join-Path $mixedRun "best.pt") (Join-Path $stage "results\runs\resnet18_multilabel_aug\best.pt")
    $mcal = Join-Path $mixedRun "calibration.json"
    if (Test-Path $mcal) { Copy-Item $mcal (Join-Path $stage "results\runs\resnet18_multilabel_aug\calibration.json") }
    Write-Host "bundled multi-label checkpoint"
}
$mixedPlates = Join-Path $root "data\splits\mixed"
if (Test-Path $mixedPlates) {
    Copy-Item $mixedPlates (Join-Path $stage "data\splits\mixed") -Recurse
    Write-Host "bundled mixed demo plates"
}

# chain nutrition table for the brand reference panel (optional: page hides it when absent)
$brandMenu = Join-Path $root "data\external\brand_menu.json"
if (Test-Path $brandMenu) {
    $bdst = Join-Path $stage "data\external"
    New-Item -ItemType Directory -Path $bdst -Force | Out-Null
    Copy-Item $brandMenu (Join-Path $bdst "brand_menu.json")
    Write-Host "bundled chain nutrition table"
}

# 116-class head: Food-101 + 15 Chinese dishes (optional: the 20-class course protocol works without it)
$wideRun = Join-Path $root "results\runs\resnet18_food101_cn116"
if (Test-Path (Join-Path $wideRun "best.pt")) {
    New-Item -ItemType Directory -Path (Join-Path $stage "results\runs\resnet18_food101_cn116") -Force | Out-Null
    Copy-Item (Join-Path $wideRun "best.pt") (Join-Path $stage "results\runs\resnet18_food101_cn116\best.pt")
    $cal = Join-Path $wideRun "calibration.json"
    if (Test-Path $cal) { Copy-Item $cal (Join-Path $stage "results\runs\resnet18_food101_cn116\calibration.json") }
    Write-Host "bundled 116-class checkpoint"
}
$wideSamples = Join-Path $root "data\wide\splits\test"
if (Test-Path $wideSamples) {
    $j = 0
    foreach ($cls in Get-ChildItem $wideSamples -Directory) {
        $dst = Join-Path $stage "data\wide\splits\test\$($cls.Name)"
        New-Item -ItemType Directory -Path $dst -Force | Out-Null
        $cls.GetFiles("*.jpg") | Select-Object -First 1 | ForEach-Object {
            Copy-Item $_.FullName $dst; $j++
        }
    }
    Write-Host "bundled $j wide sample images"
}

# 20-dish canteen head (optional: the other tabs work without it)
$cnRun = Join-Path $root "results\runs\cn241_full"
if (Test-Path (Join-Path $cnRun "best.pt")) {
    New-Item -ItemType Directory -Path (Join-Path $stage "results\runs\cn241_full") -Force | Out-Null
    Copy-Item (Join-Path $cnRun "best.pt") (Join-Path $stage "results\runs\cn241_full\best.pt")
    $ccal = Join-Path $cnRun "calibration.json"
    if (Test-Path $ccal) { Copy-Item $ccal (Join-Path $stage "results\runs\cn241_full\calibration.json") }
    Write-Host "bundled 20-dish canteen checkpoint"
}
$cnSamples = Join-Path $root "data\splits\cn241_full\test"
if (Test-Path $cnSamples) {
    $k = 0
    foreach ($cls in Get-ChildItem $cnSamples -Directory) {
        $dst = Join-Path $stage "data\splits\cn241_full\test\$($cls.Name)"
        New-Item -ItemType Directory -Path $dst -Force | Out-Null
        $cls.GetFiles("*.jpg") | Select-Object -First 2 | ForEach-Object {
            Copy-Item $_.FullName $dst; $k++
        }
    }
    Write-Host "bundled $k canteen sample images"
}

# the WeChat mini-program talks to the same FastAPI server
$mp = Join-Path $root "miniprogram"
if (Test-Path $mp) {
    Copy-Item $mp (Join-Path $stage "miniprogram") -Recurse
    Write-Host "bundled mini-program"
}

# a small sample gallery so the page is not empty on a fresh machine
$src = Join-Path $root "data\splits\canteen"
if (Test-Path $src) {
    $i = 0
    foreach ($cls in Get-ChildItem $src -Directory) {
        $dst = Join-Path $stage "data\splits\canteen\$($cls.Name)"
        New-Item -ItemType Directory -Path $dst -Force | Out-Null
        $cls.GetFiles("*.jpg") | Select-Object -First 2 | ForEach-Object {
            Copy-Item $_.FullName $dst; $i++
        }
    }
    Write-Host "bundled $i sample images"
}

$readme = @'
Campus canteen dish recognition - standalone demo
=================================================
1. Install Python 3.10+ for Windows (python.org), then in this folder:
     python -m venv .venv
     .venv\Scripts\python.exe -m pip install -r requirements.txt
   (PyTorch is ~2 GB and is downloaded here; a CUDA GPU is optional.)
2. Start the page:
     powershell -ExecutionPolicy Bypass -File start_demo.ps1 -Open
3. Open http://127.0.0.1:8000 on this PC, or http://<this-pc-ip>:8000
   from a phone on the same Wi-Fi.
The page returns Top-3 dishes with an estimated calorie range; demo images
are in data\splits\canteen and are not used for training.
Switch to the "mixed plate" tab to detect 2-3 dishes on one tray (for example
fries + chicken wings), set a portion or weight for each part, and read the
summed calorie range; mixed demo plates live in data\splits\mixed.
   Scores below 10% are hidden, and if nothing passes its class threshold the
   page says so instead of inventing a dish, so an odd angle may need a retry.
   Photos of dishes outside the 20 supported classes (a whole roast chicken,
   hot pot, home-style stir fry) are flagged as "probably not one of these"
   instead of being given a confident label.
   A third tab, "all cuisines - 116 classes", switches to a model trained on the
   full Food-101 dataset plus 15 Chinese dishes (70k images, 5.8x more than the
   course split; Peking duck, paella, bibimbap, pho, mapo tofu, kung pao chicken).
   It is broader but each class has fewer examples, so it is less accurate on
   canteen photos; the 20-class model remains the course model.
   Wide demo images live in data\wide\splits\test.
   A fourth tab, "canteen home dishes - 20", switches to a model trained on 20
   everyday Chinese dishes from CNFOOD-241 (tomato and egg, hot and sour potato
   strips, braised pork, mapo-style stir fries...), which is what a campus
   canteen actually serves. It knows only those 20 dishes and says so for
   anything else; switch back to the 20-class tab for the course menu.
   Canteen demo images live in data\splits\cn241_full\test.
   Each result also shows a "chain reference": the measured per-serving calories
   of the closest McDonald's China menu item (Big Mac 513 kcal, medium fries
   279 kcal, ...), scraped from the official nutrition calculator. The panel
   below the results lists all 89 items with search and category filters. These
   are references only and never change the range computed above.

4. On a phone: open http://<this-pc-ip>:8000 from the same Wi-Fi, or scan the QR
   code shown at the top right of the page. "Add to home screen" gives an
   app-like icon; taking a photo goes straight to the model.
5. WeChat mini-program: import the miniprogram\ folder in WeChat DevTools and set
   BASE_URL in miniprogram\config.js. It calls the same /api endpoints, so the
   server above must be reachable (HTTPS + a filed domain for a real device).
6. Share it over the internet (optional, no same-Wi-Fi needed):
     powershell -ExecutionPolicy Bypass -File start_public.ps1
   The script starts the server, downloads cloudflared.exe on first use (~50 MB)
   and prints a temporary https://xxx.trycloudflare.com link that opens from any
   network, including mobile data. The link changes every restart and has no
   login, so share it only with people you trust and Ctrl+C when done.
7. Want a permanent, fixed URL instead of the temporary one? Open docs\
   (two Chinese guides: permanent address, mini-program step by step). It covers
   three routes: Cloudflare named tunnel with your own domain, Tailscale Funnel,
   and a free ngrok static domain. Helpers: setup_cloudflare_tunnel.ps1 +
   start_permanent.ps1, start_funnel.ps1, start_ngrok.ps1.
'@
Set-Content -Path (Join-Path $stage "README.txt") -Value $readme -Encoding UTF8

if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $zip -CompressionLevel Optimal
$mb = [math]::Round((Get-Item $zip).Length / 1MB, 1)
Write-Host "packaged -> $zip ($mb MB)"
