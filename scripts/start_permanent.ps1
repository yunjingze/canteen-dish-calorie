<#
.SYNOPSIS
    用 Cloudflare Named Tunnel 把本项目发布到一个固定地址。

.DESCRIPTION
    先跑过 scripts\setup_cloudflare_tunnel.ps1（生成 tools\cloudflared_config.yml）。
    本脚本会：检查 8000 端口 -> 没服务就后台拉起 FastAPI -> 前台运行隧道。
    保持这个窗口开着就是“在线”，Ctrl+C 只停隧道，后端继续跑。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\start_permanent.ps1 -Hostname food.example.com
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Hostname,

    [int]$Port = 8000,

    [string]$Ckpt = "",
    [string]$MixedCkpt = "",
    [string]$WideCkpt = "",

    [switch]$NoServer
)

$ErrorActionPreference = "Stop"
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }

$root   = Split-Path -Parent $PSScriptRoot
$cf     = Join-Path $root "tools\cloudflared.exe"
$cfg    = Join-Path $root "tools\cloudflared_config.yml"
$src    = Join-Path $root "src"
$python = Join-Path (Split-Path -Parent $root) ".venv\Scripts\python.exe"

function Test-Port([int]$p) {
    $client = New-Object System.Net.Sockets.TcpClient
    try { $client.Connect("127.0.0.1", $p); return $true }
    catch { return $false }
    finally { $client.Dispose() }
}

if (-not (Test-Path $cf)) { throw "缺少 $cf，请先运行 scripts\setup_cloudflare_tunnel.ps1 -Hostname $Hostname" }
if (-not (Test-Path $cfg)) { throw "缺少 $cfg，请先运行 scripts\setup_cloudflare_tunnel.ps1 -Hostname $Hostname" }

if (-not $NoServer -and -not (Test-Port $Port)) {
    if (-not (Test-Path $python)) { throw "找不到虚拟环境 Python：$python" }
    Write-Host "==> 后端未启动，正在后台拉起（127.0.0.1:$Port）" -ForegroundColor Cyan
    $env:FOODCLS_HOST = "127.0.0.1"
    $env:PORT = "$Port"
    $env:PYTHONUNBUFFERED = "1"
    if ($Ckpt)      { $env:FOODCLS_CKPT = $Ckpt }
    if ($MixedCkpt) { $env:FOODCLS_MIXED_CKPT = $MixedCkpt }
    if ($WideCkpt)  { $env:FOODCLS_WIDE_CKPT = $WideCkpt }
    Start-Process -FilePath $python -ArgumentList "-m", "foodcls.app" -WorkingDirectory $src -WindowStyle Hidden | Out-Null
    $ready = $false
    for ($i = 0; $i -lt 120; $i++) {
        Start-Sleep -Milliseconds 500
        if (Test-Port $Port) { $ready = $true; break }
    }
    if (-not $ready) { throw "后端 60 秒内没有监听 $Port；先手动运行 scripts\start_demo.ps1 看报错" }
    Write-Host "    后端已就绪" -ForegroundColor Green
}

Write-Host ""
Write-Host "==> 固定地址：https://$Hostname" -ForegroundColor Green
Write-Host "    Ctrl+C 停止隧道（后端进程会继续运行）" -ForegroundColor DarkGray
Write-Host ""

& $cf tunnel --config $cfg --no-autoupdate run
$code = $LASTEXITCODE
Write-Host ""
Write-Host "隧道已退出（退出码 $code）。常见原因：凭据/配置不对、Cloudflare 上隧道被删。" -ForegroundColor Yellow
Write-Host "排查：.\tools\cloudflared.exe tunnel list ; .\tools\cloudflared.exe tunnel --config tools\cloudflared_config.yml ingress validate" -ForegroundColor DarkGray
exit $code
