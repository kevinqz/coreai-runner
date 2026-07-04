# coreai-runner

> Swift binary that loads Apple Core AI models (`.aimodel`) and serves
> inference over HTTP on a Unix domain socket. The shared runtime engine
> for all Core AI consumers.

**Status:** Architecture phase.

## What is this?

A pre-compiled Swift binary that wraps [CoreAIKit](https://github.com/john-rocky/coreai-kit)
behind a simple HTTP API. It loads `.aimodel` packages, runs inference on
Neural Engine / GPU / CPU, and returns results — all over a local Unix socket.

## Consumers

| Consumer | How it uses coreai-runner |
|----------|--------------------------|
| **[ComfyUI-CoreAI](https://github.com/kevinqz/ComfyUI-CoreAI)** | Downloads pre-compiled binary, spawns as subprocess |
| **[coreai-server](https://github.com/kevinqz/coreai-server)** | Imports as SPM package, wraps with TCP HTTP + Bonjour |
| **Ditto** | SPM import (future) |
| **CLI / curl** | Runs binary standalone |

## API

```
GET  /v1/health              → device info, loaded models, thermal state
GET  /v1/models?capability=  → models from catalog (filtered, cached)
POST /v1/predict             → single inference (image in, result out)
POST /v1/models/{id}/load    → download + load model into memory
POST /v1/models/{id}/unload  → release model from memory
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full wire protocol.

## Requirements

- Apple Silicon Mac (M1+)
- macOS 27.0+
- Xcode 26+ (build only — consumers download pre-compiled binary)

## License

Apache-2.0
