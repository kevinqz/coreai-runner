#!/usr/bin/env bash
# Build + expose coreai-runner. Requires macOS 27 SDK (Core AI framework).
set -euo pipefail
cd "$(dirname "$0")/.."
sdk=$(xcrun --sdk macosx --show-sdk-version 2>/dev/null || echo 0)
if [ "${sdk%%.*}" -lt 27 ] 2>/dev/null; then
  echo "⚠️  macOS 27 SDK required (found ${sdk}). Core AI's framework ships with macOS 27+." >&2
  echo "   On macOS 26: use ComfyUI-CoreAI/tools/mock_runner.py or the CoreAIAppleText node." >&2
  exit 1
fi
swift build -c release
bin="$PWD/.build/release/coreai-runner"
echo "✓ built ${bin}"
echo
echo "Point ComfyUI-CoreAI at it:"
echo "  export COREAI_RUNNER_PATH=\"${bin}\""
