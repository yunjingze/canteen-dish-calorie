<#
.SYNOPSIS
    用 Tailscale Funnel 发布本项目（不买域名，地址形如 https://机器名.tailxxxx.ts.net）。

.DESCRIPTION
    准备：
      1) 安装并登录 Tailscale：winget install --id tailscale.tailscale -e
      2) Tailscale 管理后台 -> DNS -> HTTPS Certificates -> Enable
    地址随“机器名 + tailnet”固定，重启后不变。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\start_funnel.ps1 -Port 8000

.EXAMPLE
    # 关闭
    powershell -ExecutionPolicy Bypass -File scripts\start_funnel.ps1 -Reset
#>
[CmdletBinding()]
param(
    [int]$Port = 8000,

    [switch]$Reset,

    [switch]$NoServer
)

$ErrorActionPreference = "Stop"
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }

function Test-Port([int]$p) {
    $client = New-Object System.Net.Sockets.TcpClient
    try { $client.Connect("127.0.0.1", $p); return $true }
    catch { return $false }
    finally { $client.Dispose() }
}

$ts = Get-Command tailscale -ErrorAction SilentlyContinue
if (-not $ts) {
    $fallback = "C:\Program Files\Tailscale\tailscale.exe"
    if (Test-Path $fallback) { $ts = $fallback }
}
if (-not $ts) {
    Write-Host "没有找到 tailscale。先安装并登录：" -ForegroundColor Yellow
    Write-Host "  winget install --id tailscale.tailscale -e"
    Write-Host "然后打开 Tailscale 用账号登录（免费个人版即可）。"
    exit 1
}
$tsExe = if ($ts -is [string]) { $ts } else { $ts.Source }

if ($Reset) {
    & $tsExe funnel reset
    Write-Host "已关闭 Funnel。" -ForegroundColor Green
    exit 0
}

if (-not $NoServer -and -not (Test-Port $Port)) {
    Write-Host ("警告：本地 $Port 端口没有服务，请先运行 scripts\start_demo.ps1，否则页面 502。") -ForegroundColor Yellow
}

Write-Host "==> 发布 127.0.0.1:$Port 到 Tailscale Funnel" -ForegroundColor Cyan
& $tsExe funnel --bg $Port
if ($LASTEXITCODE -ne 0) {
    Write-Host "Funnel 启用失败。常见原因：管理后台没开 HTTPS Certificates，或没登录。" -ForegroundColor Yellow
    exit $LASTEXITCODE
}

$status = (& $tsExe funnel status 2>&1 | Out-String)
Write-Host $status.Trim()
if ($status -match 'https://[^\s]+') {
    Write-Host ""
    Write-Host ("==> 固定地址：" + $Matches[0]) -ForegroundColor Green
    Write-Host "    关闭：powershell -ExecutionPolicy Bypass -File scripts\start_funnel.ps1 -Reset" -ForegroundColor DarkGray
} else {
    Write-Host "没从 status 里解析到地址，请手动运行：tailscale funnel status" -ForegroundColor Yellow
}
