#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
TARGET=${1:-"$ROOT/third_party/colibri"}
COMMIT=81f08a09e5651ce52616dc720f68810f9021c0be
PATCH="$ROOT/patches/colibri-m5-cache-route.patch"

if [ -e "$TARGET" ]; then
  echo "setup_colibri_m5: target already exists: $TARGET" >&2
  exit 2
fi

mkdir -p "$(dirname -- "$TARGET")"
git clone https://github.com/JustVugg/colibri.git "$TARGET"
git -C "$TARGET" checkout --detach "$COMMIT"
git -C "$TARGET" apply --check "$PATCH"
git -C "$TARGET" apply "$PATCH"

make -C "$TARGET/c" METAL=1 ARCH=native
make -C "$TARGET/c" metal-test

echo "Colibri M5 CLI ready: $TARGET/c/coli"
