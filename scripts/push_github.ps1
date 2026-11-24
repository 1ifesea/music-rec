# 把本项目推送到 GitHub——不依赖本机安装 git，全部走 GitHub REST API。
#
# 用法（在项目根目录执行）：
#   .\scripts\push_github.ps1 -Repo "你的用户名/music-rec" -Token "ghp_xxx"
#
# Token 需要勾选的权限（fine-grained PAT）：
#   Repository permissions -> Contents: Read and write
#   如果要让脚本自动建仓库，还需要 Administration: Read and write（或先用网页把空仓库建好）
#
# 已有仓库时脚本会覆盖 main 分支的最新提交（等价 git push --force），
# 所以第一次用之前请确认这个仓库是你自己的空仓库，别指向已有内容的仓库。

param(
    [Parameter(Mandatory = $true)][string]$Repo,        # owner/name
    [Parameter(Mandatory = $true)][string]$Token,       # Personal Access Token
    [string]$Branch = "main",
    [string]$Message = "初始化：多模态音乐个性化推荐系统",
    [switch]$CreateRepo                                 # 仓库不存在时是否自动创建（私有）
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$api = "https://api.github.com"
$headers = @{
    Authorization        = "Bearer $Token"
    Accept               = "application/vnd.github+json"
    "X-GitHub-Api-Version" = "2022-11-28"
    "User-Agent"         = "musicrec-push-script"
}

function Invoke-GH([string]$Method, [string]$Path, $Body = $null) {
    $uri = if ($Path.StartsWith("http")) { $Path } else { "$api$Path" }
    $params = @{
        Method      = $Method
        Uri         = $uri
        Headers     = $headers
        ContentType = "application/json; charset=utf-8"
    }
    if ($Body -ne $null) {
        $params.Body = ([Text.Encoding]::UTF8.GetBytes(($Body | ConvertTo-Json -Depth 12 -Compress)))
    }
    try {
        return Invoke-RestMethod @params
    } catch [System.Net.WebException] {
        $resp = $_.Exception.Response
        $code = [int]$resp.StatusCode
        $reader = New-Object IO.StreamReader($resp.GetResponseStream())
        $detail = $reader.ReadToEnd()
        throw "GitHub API $Method $Path 失败：HTTP $code`n$detail"
    }
}

# ---------- 1. 校验 token 与仓库 ----------
Write-Host "[1/6] 校验 token ..."
$me = Invoke-GH "GET" "/user"
Write-Host "      已登录：$($me.login)"

$owner, $name = $Repo.Split("/")
if ($owner -ne $me.login) {
    Write-Warning "仓库 owner ($owner) 和当前 token 用户 ($($me.login)) 不一致，只有该组织/用户下的仓库才有权限，请自行确认。"
}

$repoExists = $true
try {
    Invoke-GH "GET" "/repos/$Repo" | Out-Null
    Write-Host "      仓库已存在：$Repo"
} catch {
    $repoExists = $false
}

if (-not $repoExists) {
    if (-not $CreateRepo) {
        throw "仓库 $Repo 不存在。要么先在网页上建一个空仓库，要么加 -CreateRepo 让脚本自动创建。"
    }
    Write-Host "[2/6] 创建私有仓库 $name ..."
    Invoke-GH "POST" "/user/repos" @{ name = $name; private = $true; auto_init = $false } | Out-Null
    Start-Sleep -Seconds 2
} else {
    Write-Host "[2/6] 复用已有仓库"
}

# ---------- 3. 收集要上传的文件 ----------
# 与 .gitignore 保持一致的排除规则：虚拟环境、缓存、密码文件、音频与特征产物、IDE 目录
$excludeDirs = @("\.venv\", "\.git\", "\.idea\", "\__pycache__\", "\.pytest_cache\", "\node_modules\")
$excludeFiles = @("\.env$", "\.pyc$", "tracks\.csv$", "features\.npz$", "mmgate\.pt$",
                  "item_ids\.npy$", "metrics\.json$", "text_encoder\.joblib$", "\.log$")

$files = Get-ChildItem -LiteralPath $root -Recurse -File | Where-Object {
    $rel = "/" + ($_.FullName.Substring($root.Length + 1).Replace("\", "/"))
    $skip = $false
    foreach ($d in $excludeDirs) { if ($rel -like "*$d*") { $skip = $true } }
    if (-not $skip) {
        foreach ($f in $excludeFiles) {
            if ($rel -like "*$f") {
                # web 里的 vue 运行时是必须提交的依赖，不能按名称排掉
                if (-not ($rel -like "*/web/assets/vue.esm-browser.prod.js")) { $skip = $true }
            }
        }
    }
    -not $skip
}

$tree = @()
$skipped = @()
Write-Host "[3/6] 收集文件：$($files.Count) 个"
foreach ($f in $files) {
    $relPath = $f.FullName.Substring($root.Length + 1).Replace("\", "/")
    if ($f.Length -gt 20MB) {
        $skipped += "$relPath ($([math]::Round($f.Length / 1MB, 1)) MB)"
        continue
    }
    $b64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes($f.FullName))
    $blob = Invoke-GH "POST" "/repos/$Repo/git/blobs" @{ content = $b64; encoding = "base64" }
    $tree += @{ path = $relPath; mode = "100644"; type = "blob"; sha = $blob.sha }
    Write-Host "      + $relPath"
}
if ($skipped.Count -gt 0) {
    Write-Warning ("以下文件超过 20MB 已跳过（GitHub 单文件限制 100MB）：`n" + ($skipped -join "`n"))
}

# ---------- 4. 建 tree ----------
Write-Host "[4/6] 创建 tree ..."
$newTree = Invoke-GH "POST" "/repos/$Repo/git/trees" @{ tree = $tree }

# ---------- 5. 建 commit ----------
Write-Host "[5/6] 创建 commit ..."
$parentSha = $null
try {
    $ref = Invoke-GH "GET" "/repos/$Repo/git/ref/heads/$Branch"
    $parentSha = $ref.object.sha
    Write-Host "      已有 $Branch 分支，作为父提交：$($parentSha.Substring(0,7))"
} catch {
    Write-Host "      仓库为空，本次是首个提交"
}

$commitBody = @{ message = $Message; tree = $newTree.sha }
if ($parentSha) { $commitBody.parents = @($parentSha) }
$commit = Invoke-GH "POST" "/repos/$Repo/git/commits" $commitBody

# ---------- 6. 更新分支 ----------
Write-Host "[6/6] 更新 $Branch 分支 ..."
if ($parentSha) {
    Invoke-GH "PATCH" "/repos/$Repo/git/refs/heads/$Branch" @{ sha = $commit.sha; force = $true } | Out-Null
} else {
    Invoke-GH "POST" "/repos/$Repo/git/refs" @{ ref = "refs/heads/$Branch"; sha = $commit.sha } | Out-Null
}

Write-Host ""
# 顺带把默认分支指向 $Branch（新建仓库若沿用 master 会看起来"空"，这一步保证打开仓库就能看到代码）
try { Invoke-GH "PATCH" "/repos/$Repo" @{ default_branch = $Branch } | Out-Null } catch { }
Write-Host "完成：https://github.com/$Repo/tree/$Branch"
Write-Host "提交：$($commit.sha.Substring(0,7))  文件：$($tree.Count) 个"
