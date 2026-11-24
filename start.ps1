# 启动服务（Windows PowerShell）
# 用法：右键 -> 使用 PowerShell 运行；或在终端里执行 .\start.ps1

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

if (-not (Test-Path -LiteralPath ".venv\Scripts\python.exe")) {
    Write-Host "没有找到 .venv，请先执行 setup.ps1 完成环境安装。" -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Path -LiteralPath ".env")) {
    Write-Host "没有找到 .env，请复制 .env.example 为 .env 并填写数据库密码。" -ForegroundColor Yellow
    exit 1
}

Write-Host "服务启动中：http://127.0.0.1:8000" -ForegroundColor Green
& ".venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
