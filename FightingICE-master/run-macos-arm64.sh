#!/usr/bin/env bash
set -euo pipefail

# Always run from the directory that contains this script (so relative paths like ./lib work)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# This script launches the *actual* FightingICE game (not the Py4J stub gateway).
# You must have FightingICE.jar in this folder.
if [ ! -f "FightingICE.jar" ]; then
  echo "ERROR: FightingICE.jar not found in $(pwd)"
  echo "Your current folder only has dependency jars (./lib/*), but not the game jar."
  echo "Fix: download the FightingICE/DareFightingICE release package and copy FightingICE.jar here,"
  echo "or update this script to point to where your FightingICE.jar is located."
  exit 1
fi

PORT="${PORT:-31415}"

java -XstartOnFirstThread \
  -cp "FightingICE.jar:./lib/*:./lib/lwjgl/*:./lib/lwjgl/natives/macos/arm64/*:./lib/grpc/*" \
  Main --limithp 400 400 --grey-bg --pyftg-mode --port "$PORT" "$@"
