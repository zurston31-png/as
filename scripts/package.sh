#!/usr/bin/env bash
# Build a clean, runnable zip of the bot.
#
# Uses `git archive` rather than zipping the working directory, so the
# bundle contains exactly what is committed and nothing else. That matters
# more than convenience here: a working directory carries .env, the
# runtime database, backups and __pycache__, and a bundle built by zipping
# it would ship live secrets to whoever received it.
#
#   ./scripts/package.sh            -> dist/memecoin-bot-<short-sha>.zip
#   ./scripts/package.sh /tmp/out   -> writes into /tmp/out instead
set -euo pipefail

cd "$(dirname "$0")/.."
OUT_DIR="${1:-dist}"
mkdir -p "$OUT_DIR"

SHA="$(git rev-parse --short HEAD)"
STAMP="$(date -u +%Y%m%d-%H%M)"
NAME="memecoin-bot-${STAMP}-${SHA}"
ZIP="${OUT_DIR}/${NAME}.zip"

rm -f "$ZIP"
# --prefix so the zip expands into its own directory rather than spraying
# files across whatever the user extracted it in.
git archive --format=zip --prefix="${NAME}/" -o "$ZIP" HEAD

echo "$ZIP"
