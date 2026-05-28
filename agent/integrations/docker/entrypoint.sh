#!/usr/bin/env bash
set -e

SECRETS_DIR="/tmp/open-swe/secrets"

mkdir -p "$SECRETS_DIR"
chmod 700 "$SECRETS_DIR"

if [ -f "$SECRETS_DIR/GH_TOKEN" ]; then
  GH_TOKEN=$(cat "$SECRETS_DIR/GH_TOKEN")

  git config --global credential.helper store
  echo "https://x-access-token:${GH_TOKEN}@github.com" \
    > ~/.git-credentials
  chmod 600 ~/.git-credentials
fi

if [ -f "$SECRETS_DIR/GITHUB_PROXY_URL" ]; then
  HTTPS_PROXY=$(cat "$SECRETS_DIR/GITHUB_PROXY_URL")
  export HTTPS_PROXY
  export HTTP_PROXY="$HTTPS_PROXY"
  git config --global http.proxy "$HTTPS_PROXY"
fi

mkdir -p /tmp/open-swe
touch /tmp/open-swe/ready

exec "$@"
