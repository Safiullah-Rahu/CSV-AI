# Push CSV-AI v2 to https://github.com/Safiullah-Rahu/CSV-AI
#
# Run from inside this folder in PowerShell:
#     cd "A:\Claude Work\Git Repo CSV AI\Github Repo"
#     powershell -ExecutionPolicy Bypass -File .\push-to-github.ps1
#
# Prereqs: Git installed, and either a Personal Access Token / SSH key
# configured for github.com (Git will prompt the first time).

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "=== CSV-AI v2 → GitHub ===" -ForegroundColor Cyan
Write-Host ""

# 1. Remove the broken .git folder that the session couldn't clean up.
if (Test-Path ".git") {
    Write-Host "Removing existing .git folder..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force ".git"
}

# 2. Fresh repo on the main branch.
git init -b main | Out-Null
git config user.email "safiullahrahu1555@gmail.com"
git config user.name  "Safiullah Rahu"

# 3. Add the GitHub remote.
git remote add origin https://github.com/Safiullah-Rahu/CSV-AI.git

# 4. Stage and commit everything.
git add .
git commit -m "v2: modernized -- modular architecture, multi-provider LLM, modern UI"

# 5. Push to a NEW BRANCH (your main is protected by GitHub).
#    Merge into main on GitHub afterward by opening a Pull Request.
Write-Host ""
Write-Host "Pushing to branch 'v2-modernization' (main is protected)." -ForegroundColor Yellow
Write-Host ""
git push -u origin main:v2-modernization --force

Write-Host ""
Write-Host "Done." -ForegroundColor Green
Write-Host "Open a PR here:" -ForegroundColor Cyan
Write-Host "  https://github.com/Safiullah-Rahu/CSV-AI/compare/main...v2-modernization" -ForegroundColor Cyan
