#!/usr/bin/env bash
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
export PATH="$PROJECT_DIR/.bin:$PATH"

REPO_NAME="autonomous-pothole-lane-navigator"
REPO_DESC="AI-Enhanced Autonomous Navigator for Real-Time Pothole Sensing and Lane Precision using YOLOv11, Canny Edge Detection, and Hough Line Transform"

echo "=================================================================="
echo " GitHub Repository Creator & Automated Push Tool"
echo " Target Profile: https://github.com/lokeshreddyambati"
echo " Repository Name: $REPO_NAME"
echo "=================================================================="

# Check if Personal Access Token is passed as argument or environment variable
if [ -n "$1" ]; then
    GH_TOKEN="$1"
fi

if [ -n "$GH_TOKEN" ]; then
    echo "[+] Using provided GitHub Personal Access Token..."
    echo "$GH_TOKEN" | gh auth login --with-token
fi

# Check authentication status
if ! gh auth status &>/dev/null; then
    echo "[!] GitHub authentication is required to create a repository on your account."
    echo "[*] Launching fast 10-second web login..."
    gh auth login -h github.com -p https -w
fi

echo "[+] Authenticated as: $(gh api user --jq .login 2>/dev/null || echo 'lokeshreddyambati')"

# Check if remote repository already exists, otherwise create it
if gh repo view "lokeshreddyambati/$REPO_NAME" &>/dev/null; then
    echo "[*] Repository already exists on GitHub. Setting remote origin..."
    git remote remove origin 2>/dev/null || true
    git remote add origin "https://github.com/lokeshreddyambati/$REPO_NAME.git"
    echo "[+] Pushing code to main branch..."
    git push -u origin main
else
    echo "[+] Creating new public repository '$REPO_NAME' on GitHub..."
    gh repo create "$REPO_NAME" --public --source="$PROJECT_DIR" --remote=origin --push --description "$REPO_DESC"
fi

echo "=================================================================="
echo "🎉 SUCCESS! Your project has been pushed to GitHub:"
echo "👉 https://github.com/lokeshreddyambati/$REPO_NAME"
echo "=================================================================="
