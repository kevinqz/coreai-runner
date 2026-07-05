# Running coreai-runner (real Core AI inference)

The runner loads `.aimodel` bundles through Apple's **Core AI** runtime — the
`CoreAI` **system framework**, which ships **only with macOS 27+**. On macOS 26,
`swift build` fails with `error: no such module 'CoreAI'`. This is an OS-level wall,
not a code issue: CoreAIKit's `Package.swift` declares `platforms: [.macOS("27.0")]`.

## Requirements
- Apple Silicon Mac on **macOS 27+** (developer beta or GA)
- Xcode 26+ with the macOS 27 SDK — check: `xcrun --sdk macosx --show-sdk-version` (≥ 27)

## One command
```bash
scripts/run.sh          # verifies the SDK, builds -c release, prints the export line
```
Manual equivalent:
```bash
swift build -c release
export COREAI_RUNNER_PATH="$PWD/.build/release/coreai-runner"
```

## Wire into ComfyUI-CoreAI
```bash
export COREAI_RUNNER_PATH=/path/to/coreai-runner/.build/release/coreai-runner
# start ComfyUI — the nodes now use the REAL binary instead of the mock.
# unset it (or point at ComfyUI-CoreAI/tools/mock_runner.py) to return to the mock.
```

## On macOS 26 (today, no Core AI framework)
- `.aimodel` vision models (depth/detection/VLM/SAM/FLUX) cannot execute here.
- Exercise the full pipeline with `tools/mock_runner.py` (placeholder outputs), or run
  **real on-device text** via the `CoreAIAppleText` node (Apple FoundationModels).
