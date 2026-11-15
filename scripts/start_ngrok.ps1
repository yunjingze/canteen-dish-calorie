<#
.SYNOPSIS
    用 ngrok 免费固定域名发布本项目（最快，但有流量配额与首访提示页）。

.DESCRIPTION
    准备：注册 https://dashboard.ngrok.com/signup ，复制 Your Authtoken，先执行
      .\tools\ngrok.exe config add-authtoken <token>
    再在 Dashboard -> Universal Gateway -> Domains 领一个免费固定域名 xxx.ngrok-free.app。
    本脚本首次运行会自动下载 tools\ngrok.exe（约 12 MB）。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\start_ngrok.ps1 -Domain xxx.ngrok-free.app
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Domain,

    [int]$Port = 8000,

    [switch]$NoServer
)

$ErrorActionPreference = "Stop"
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }

$root  = Split-Path -Parent $PSScriptRoot
$tools = Join-Path $root "tools"
$ngrok = Join-Path $tools "ngrok.exe"

function Test-Port([int]$p) {
    $client = New-Object System.Net.Sockets.TcpClient
    try { $client.Connect("127.0.0.1", $p); return $true }
    catch { return $false }
    finally { $client.Dispose() }
}

if (-not (Test-Path $ngrok)) {
    Write-Host "==> 下载 ngrok.exe（约 12 MB）" -ForegroundColor Cyan
    New-Item -ItemType Directory -Force -Path $tools | Out-Null
    $zip = Join-Path $tools "ngrok.zip"
    Invoke-WebRequest -UseBasicParsing -Uri "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-windows-amd64.zip" -OutFile $zip
    Expand-Archive -Path $zip -DestinationPath $tools -Force
    Remove-Item $zip -Force
    Write-Host "    已下载 $ngrok" -ForegroundColor Green
}

$hasToken = $false
foreach ($f in @((Join-Path $env:LOCALAPPDATA "ngrok\ngrok.yml"), (Join-Path $env:USERPROFILE ".ngrok2\ngrok.yml"))) {
    if ((Test-Path $f) -and ((Get-Content $f -Raw) -match "authtoken")) { $hasToken = $true }
}
if (-not $hasToken) {
    Write-Host ""
    Write-Host "还没有配置 ngrok authtoken。先做这两步：" -ForegroundColor Yellow
    Write-Host "  1) 打开 https://dashboard.ngrok.com/get-started/your-authtoken 复制 token"
    Write-Host ("  2) 运行：.\tools\ngrok.exe config add-authtoken " + "<你的token>")
    Write-Host "然后重新执行本脚本。"
    exit 1
}

if (-not $NoServer -and -not (Test-Port $Port)) {
    Write-Host ("警告：本地 $Port 端口没有服务，页面会 502。先运行 scripts\start_demo.ps1 再启动 ngrok。") -ForegroundColor Yellow
}

Write-Host ""
Write-Host ("==> 固定地址：https://" + $Domain) -ForegroundColor Green
Write-Host "    免费版限制：1 GB/月；浏览器首次访问有提示页（程序调用可加请求头 ngrok-skip-browser-warning: 1）" -ForegroundColor DarkGray
Write-Host ""

& $ngrok http $Port "--domain=$Domain" --log=stdout
exit $LASTEXITCODE
