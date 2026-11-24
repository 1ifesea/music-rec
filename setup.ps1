# 一次性环境准备：装依赖 -> 造数据 -> 建库 -> 提特征 -> 训练
# 用法：.\setup.ps1            （默认 600 首合成数据）
#       .\setup.ps1 -Count 1000

param(
    [int]$Count = 600,
    [string]$IndexUrl = "https://pypi.tuna.tsinghua.edu.cn/simple"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

if (-not (Test-Path -LiteralPath ".venv\Scripts\python.exe")) {
    Write-Host "[1/6] 创建虚拟环境" -ForegroundColor Cyan
    py -m venv .venv
} else {
    Write-Host "[1/6] 虚拟环境已存在，跳过" -ForegroundColor Cyan
}

$py = ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath ".env")) {
    Write-Host "请先复制 .env.example 为 .env 并填写 MySQL 密码，然后重新执行本脚本。" -ForegroundColor Yellow
    exit 1
}

Write-Host "[2/6] 安装依赖（首次约 3~5 分钟）" -ForegroundColor Cyan
& $py -m pip install --upgrade pip --quiet
& $py -m pip install -r requirements.txt -i $IndexUrl

Write-Host "[3/6] 生成数据集（$Count 首）" -ForegroundColor Cyan
& $py scripts\prepare_data.py --count $Count

Write-Host "[4/6] 建库并导入歌曲 + 生成模拟行为" -ForegroundColor Cyan
& $py scripts\load_tracks.py --reset
& $py scripts\seed_simulated.py --reset

Write-Host "[5/6] 批量提取多模态特征" -ForegroundColor Cyan
& $py scripts\build_features.py --force

Write-Host "[6/6] 训练推荐模型" -ForegroundColor Cyan
& $py scripts\train.py

Write-Host "全部完成。运行 .\start.ps1 启动服务，浏览器打开 http://127.0.0.1:8000" -ForegroundColor Green
