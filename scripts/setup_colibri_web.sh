#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
COLIBRI=${1:-"$ROOT/third_party/colibri"}
WEB="$COLIBRI/web"

if [ ! -f "$WEB/package-lock.json" ]; then
  echo "setup_colibri_web: Colibri checkout is missing: $COLIBRI" >&2
  echo "run ./scripts/setup_colibri_m5.sh first" >&2
  exit 2
fi
if ! command -v npm >/dev/null 2>&1; then
  echo "setup_colibri_web: npm is required" >&2
  exit 2
fi

npm --prefix "$WEB" ci
npm --prefix "$WEB" run build

echo "Colibri web UI ready: $WEB/dist/index.html"
