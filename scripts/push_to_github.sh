#!/usr/bin/env bash
# Push the clean release branch to GitHub (force overwrite remote).
# Run on AutoDL: bash scripts/push_to_github.sh
set -euo pipefail

REPO=/root/autodl-tmp/Weather-pred
cd "$REPO"

# --- optional: academic proxy (set your HTTP port, or leave empty) ---
PROXY_PORT="${PROXY_PORT:-}"
if [[ -n "$PROXY_PORT" ]]; then
  export http_proxy="http://127.0.0.1:${PROXY_PORT}"
  export https_proxy="http://127.0.0.1:${PROXY_PORT}"
  git config http.proxy  "http://127.0.0.1:${PROXY_PORT}"
  git config https.proxy "http://127.0.0.1:${PROXY_PORT}"
fi

# --- GitHub token: create at https://github.com/settings/tokens (scope: repo) ---
if [[ -z "${GITHUB_TOKEN:-}" ]]; then
  echo "Set your token first, e.g.:"
  echo "  export GITHUB_TOKEN=ghp_xxxxxxxx"
  exit 1
fi

git remote set-url origin "https://Ltangwang:${GITHUB_TOKEN}@github.com/Ltangwang/Weather-pred.git"

echo "Local HEAD:"
git log -1 --oneline

# master -> GitHub (use master:main if default branch on GitHub is main)
BRANCH_REMOTE="${BRANCH_REMOTE:-master}"
if [[ "$BRANCH_REMOTE" == "main" ]]; then
  git push --force origin master:main
else
  git push --force origin master
fi

# Remove token from remote URL after push
git remote set-url origin https://github.com/Ltangwang/Weather-pred.git

echo "Done. Check: https://github.com/Ltangwang/Weather-pred"
