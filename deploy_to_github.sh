#!/usr/bin/env bash
# ==============================================================================
# GitHub Deployment Automation & Pre-Commit Security Audit Script
# ==============================================================================

set -euo pipefail

echo "=========================================================="
echo "🛡️  SecOps AutoML: GitHub Deployment Automation"
echo "=========================================================="

# 1. Initialize Git repository if not present
if [ ! -d ".git" ]; then
    echo "[+] Initializing new Git repository..."
    git init
    git branch -M main
else
    echo "[*] Existing Git repository detected."
fi

# 2. Pre-Commit Security Checks
echo "[+] Running Pre-Commit Security Verifications..."

# Check that .env is ignored
if git check-ignore -q .env 2>/dev/null; then
    echo "    ✅ .env is safely ignored by .gitignore"
else
    echo "    ⚠️ WARNING: .env is NOT ignored! Updating .gitignore..."
    echo ".env" >> .gitignore
fi

# Stage files for inspection
git add .

# Scan staged files for leaked API keys (AIza..., sk-..., etc.)
echo "[+] Scanning staged files for exposed API credentials..."
LEAK_COUNT=0

if git diff --cached | grep -E -i "(AIza[0-9A-Za-z-_]{35}|sk-[a-zA-Z0-9]{32,}|GEMINI_API_KEY=[a-zA-Z0-9_\-]{10,})" | grep -v "\.env\.example" >/dev/null 2>&1; then
    echo "    🚨 ERROR: Potential API key or credential pattern detected in staged files!"
    echo "    Review your staged files with 'git diff --cached' before committing."
    exit 1
else
    echo "    ✅ No exposed API keys detected in staged changes."
fi

# 3. Commit changes
COMMIT_MSG="feat: initial release of Adversarial-Resistant SecOps AutoML & XAI Engine"
echo "[+] Committing staged files with message: '${COMMIT_MSG}'"

if git diff --cached --quiet; then
    echo "    [*] Working tree clean, nothing to commit."
else
    git commit -m "${COMMIT_MSG}"
    echo "    ✅ Commit successful."
fi

# 4. Remote Configuration and Push
echo ""
echo "=========================================================="
echo "🚀 Remote Push Configuration"
echo "=========================================================="

REMOTE_URL="${1:-}"

if [ -z "${REMOTE_URL}" ]; then
    echo "Please provide your GitHub repository remote URL (or press Enter to skip push):"
    read -r -p "GitHub Remote URL (e.g. git@github.com:user/repo.git): " INPUT_REMOTE
    REMOTE_URL="${INPUT_REMOTE}"
fi

if [ -n "${REMOTE_URL}" ]; then
    if git remote | grep -q "^origin$"; then
        echo "[*] Updating existing 'origin' remote to ${REMOTE_URL}..."
        git remote set-url origin "${REMOTE_URL}"
    else
        echo "[+] Adding 'origin' remote: ${REMOTE_URL}..."
        git remote add origin "${REMOTE_URL}"
    fi

    echo "[+] Pushing 'main' branch to origin..."
    git push -u origin main
    echo "✅ Successfully deployed to GitHub: ${REMOTE_URL}"
else
    echo "[*] Remote URL not provided. Files are safely committed locally to branch 'main'."
    echo "To push manually later, run:"
    echo "    git remote add origin <YOUR_GITHUB_REPO_URL>"
    echo "    git push -u origin main"
fi
