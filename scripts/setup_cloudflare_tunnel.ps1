<#
.SYNOPSIS
    一次性配置 Cloudflare Named Tunnel：创建隧道、绑定域名、生成 ingress 配置。

.DESCRIPTION
    前提：域名已托管到 Cloudflare（NS 指向 Cloudflare），并且已经运行过
    .\tools\cloudflared.exe tunnel login（或让本脚本代跑，会弹浏览器）。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\setup_cloudflare_tunnel.ps1 -Hostname food.example.com

.EXAMPLE
    # 顺便注册“登录 Windows 后自动启动”
    powershell -ExecutionPolicy Bypass -File scripts\setup_cloudflare_tunnel.ps1 -Hostname food.example.com -RegisterAutostart
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Hostname,

    [string]$TunnelName = "fooddemo",

    [int]$Port = 8000,

    [switch]$SkipLogin,

    [switch]$RegisterAutostart
)

$ErrorActionPreference = "Stop"
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }

$root    = Split-Path -Parent $PSScriptRoot
$tools   = Join-Path $root "tools"
$cf      = Join-Path $tools "cloudflared.exe"
$cfg     = Join-Path $tools "cloudflared_config.yml"
$homeDir = [Environment]::GetFolderPath("UserProfile")
$certDir = Join-Path $homeDir ".cloudflared"
$cert    = Join-Path $certDir "cert.pem"

function Step([string]$text) { Write-Host ""; Write-Host "==> $text" -ForegroundColor Cyan }
function Ok([string]$text)   { Write-Host "    $text" -ForegroundColor Green }
function Warn([string]$text) { Write-Host "    $text" -ForegroundColor Yellow }

if ($Hostname -match '^https?://') {
    throw "Hostname 不要带协议：写 food.example.com，不要写 https://food.example.com"
}
if ($Hostname -notmatch '^[A-Za-z0-9]([A-Za-z0-9.-]*[A-Za-z0-9])?\.[A-Za-z]{2,}$') {
    throw "Hostname 格式不对：$Hostname（应形如 food.example.com）"
}

# 1. cloudflared
if (-not (Test-Path $cf)) {
    Step "下载 cloudflared.exe"
    New-Item -ItemType Directory -Force -Path $tools | Out-Null
    $url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
    Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $cf
    Ok "已下载 $cf"
}

# 2. 登录（生成 cert.pem，用来管理 DNS）
if (-not $SkipLogin -and -not (Test-Path $cert)) {
    Step "登录 Cloudflare（浏览器里选中你的域名，点 Authorize）"
    & $cf tunnel login
    if ($LASTEXITCODE -ne 0) { throw "cloudflared tunnel login 失败（退出码 $LASTEXITCODE）" }
}
if (-not (Test-Path $cert)) {
    throw "找不到 $cert。请先运行 .\tools\cloudflared.exe tunnel login，或去掉 -SkipLogin 重试"
}
Ok "登录凭据：$cert"

# 3. 查找或创建隧道
Step "查找或创建隧道 $TunnelName"
$tunnelId = $null
$listText = (& $cf tunnel list 2>$null | Out-String)
foreach ($line in ($listText -split "`r?`n")) {
    if ($line -match '^(\S+)\s+([0-9a-fA-F-]{36})\b') {
        if ($Matches[1] -eq $TunnelName) { $tunnelId = $Matches[2]; break }
    }
}

if (-not $tunnelId) {
    $out = (& $cf tunnel create $TunnelName 2>&1 | Out-String)
    if ($out -match '([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})') {
        $tunnelId = $Matches[1]
    }
    if (-not $tunnelId) { throw "创建隧道失败：$out" }
    Ok "已创建 $TunnelName ($tunnelId)"
} else {
    Ok "复用已有隧道 $TunnelName ($tunnelId)"
}

$cred = Join-Path $certDir "$tunnelId.json"
if (-not (Test-Path $cred)) {
    throw "缺少凭据文件 $cred —— 这条隧道可能是在别的电脑/账户下创建的，请换一个 -TunnelName"
}
Ok "凭据：$cred"

# 4. DNS
Step "把 $Hostname 指向隧道（CNAME）"
& $cf tunnel route dns --overwrite-dns $TunnelName $Hostname
if ($LASTEXITCODE -ne 0) {
    Warn "route dns 退出码 $LASTEXITCODE —— 如果这个域名已经手动配过 CNAME 可以忽略"
}

# 5. ingress 配置（无 BOM 的 UTF-8，避免 YAML 解析问题）
Step "写入 $cfg"
$yaml = @(
    "tunnel: $tunnelId",
    "credentials-file: $cred",
    "",
    "ingress:",
    "  - hostname: $Hostname",
    "    service: http://127.0.0.1:$Port",
    "  - service: http_status:404",
    ""
) -join "`n"
[System.IO.File]::WriteAllText($cfg, $yaml, (New-Object System.Text.UTF8Encoding($false)))
Ok "配置已写入"

Step "校验配置"
$val = (& $cf tunnel --config $cfg ingress validate 2>&1 | Out-String)
if ($val.Trim()) { Write-Host $val.Trim() }
if ($LASTEXITCODE -ne 0) { throw "ingress 校验失败，请检查 $cfg" }

# 6. 可选：开机自启
if ($RegisterAutostart) {
    Step "注册登录自启（任务计划程序：FoodDemoPermanent）"
    $runner = Join-Path $PSScriptRoot "start_permanent.ps1"
    $arg = "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File ""$runner"" -Hostname $Hostname -Port $Port"
    $action  = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arg
    $trigger = New-ScheduledTaskTrigger -AtLogOn
    Register-ScheduledTask -TaskName "FoodDemoPermanent" -Action $action -Trigger $trigger -Description "Food demo: FastAPI + Cloudflare named tunnel" -Force | Out-Null
    Ok "已注册；下次登录 Windows 会自动启动（想删除：Unregister-ScheduledTask -TaskName FoodDemoPermanent）"
}

# 7. 汇总
Write-Host ""
Write-Host "配置完成。下一步：" -ForegroundColor Cyan
Write-Host "  1) 启动：powershell -ExecutionPolicy Bypass -File scripts\start_permanent.ps1 -Hostname $Hostname"
Write-Host ("  2) 验证：浏览器打开 https://" + $Hostname + "/api/health （应返回 ok:true）")
Write-Host "  3) 手机用移动数据打开同一地址，能看到页面即成功"
Write-Host ""
Write-Host "提示：电脑关机/睡眠时地址会 502；公网无鉴权，长期用建议加 Cloudflare Access（见 docs\永久公网地址.md A7）" -ForegroundColor DarkGray
