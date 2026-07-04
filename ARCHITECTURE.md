# Core AI Ecosystem — Full Architecture

> The complete map of all projects, their relationships, and responsibilities.
> Private repos are owned by `kevinqz`. Public repos are the existing ecosystem.

---

## Bird's Eye View

```
                    DISCOVERY                    RUNTIME                     CONSUMERS
                    ─────────                    ────────                    ─────────

           ┌─────────────────────┐
           │  coreai-catalog     │
           │  (público, feito)   │
           │  82 modelos         │
           │  API JSON           │
           └────────┬────────────┘
                    │ model metadata (capabilities, devices, benchmarks, HF URLs)
                    │
     ┌──────────────┼──────────────────────────────────────────┐
     │              │                                          │
     ▼              ▼                                          ▼
┌─────────────┐  ┌─────────────────────┐              ┌──────────────────┐
│ CONVERSION  │  │     RUNTIME LAYER   │              │    CONSUMERS     │
│             │  │                     │              │                  │
│ coreai-zoo  │  │  coreai-runner ◄═══│═════════════│ ComfyUI-CoreAI   │
│ (público)   │  │  (PRIVADO, novo)    │              │ (PRIVADO, novo)  │
│             │  │  Swift binary       │              │ Python nodes     │
│ converts    │  │  + CoreAIKit SPM   │              │                  │
│ HF models   │  │  + adapters         │              ├──────────────────┤
│ to .aimodel │  │                     │◄────────────│ Ditto            │
│             │  │  ┌──────────────┐   │  SPM dep     │ (PRIVADO)        │
│             │  │  │ coreai-server│   │              │ iOS transmutation│
│             │  │  │ (PRIVADO,    │   │              ├──────────────────┤
│             │  │  │  futuro)     │   │              │ Future apps      │
│             │  │  │ LAN server   │   │              │                  │
│             │  │  └──────────────┘   │              └──────────────────┘
│             │  │                     │
└─────────────┘  └─────────────────────┘
```

---

## Project Roster

### Existing Public Projects (not ours to build)

| Project | Owner | Role |
|---------|-------|------|
| **coreai-catalog** | kevinsaltarelli (público) | Intelligence layer — 82 models, API, benchmarks. THE source of truth for "what exists, where to download, what it can do." |
| **CoreAIKit** | john-rocky/coreai-kit | Swift SPM runtime — typed pipelines (DepthEstimator, ObjectDetector, KitVisionModel, ChatSession, etc.). Links patched CoreAILM engine. |
| **coreai-model-zoo** | john-rocky/coreai-model-zoo | Conversion — 32 model cards, 60+ scripts, PORTING.md. Converts HF models to `.aimodel`. |
| **[coreai-fabric]** | kevinqz (planejado) | HF org própria para hosting/conversão de `.aimodel` independentes do Zoo. |

### New Private Projects (kevinqz)

| Project | Repo | Role | Status |
|---------|------|------|--------|
| **coreai-runner** | `kevinqz/coreai-runner` | Swift binary + package. Loads `.aimodel`, serves inference over HTTP (Unix socket). Embeds CoreAIKit. Adapter pattern. | Architecture phase |
| **ComfyUI-CoreAI** | `kevinqz/ComfyUI-CoreAI` | ComfyUI custom node. Python thin client over coreai-runner subprocess. Vision nodes: depth, SAM, detection, VLM, image-gen. | Architecture phase |
| **coreai-server** | `kevinqz/coreai-server` | LAN inference server. Wraps coreai-runner with TCP HTTP + Bonjour discovery. iPhone/iPad/Mac as neural co-processors. | Future |

---

## Architecture: The 3-Layer Stack

```
┌─────────────────────────────────────────────────────────────┐
│  LAYER 3 — CONSUMERS (app-specific, one per use case)       │
│                                                             │
│  ComfyUI-CoreAI    Ditto (iOS)    Future Apps    curl       │
│  (Python)          (Swift)        (Swift/Py)     (shell)    │
└───────┬────────────────┬──────────────┬───────────┬─────────┘
        │ subprocess     │ SPM import   │ SPM       │ HTTP
        │ (Unix socket)  │              │           │
        ▼                ▼              ▼           │
┌──────────────────────────────────────────────┐   │
│  LAYER 2 — RUNTIME (shared, one binary)      │   │
│                                              │   │
│  coreai-runner                               │   │
│  ├── HTTP server (Hummingbird, Unix socket)  │   │
│  ├── Model cache (actor-isolated, LRU)       │   │
│  ├── Catalog client (API + 5min cache)       │   │
│  └── Adapters:                               │   │
│      ├── DepthAdapter    → DepthEstimator    │   │
│      ├── DetectionAdapter → ObjectDetector   │   │
│      ├── VLMAdapter      → KitVisionModel    │   │
│      ├── DiffusionAdapter → Flux2Pipeline    │   │
│      └── SegmenterAdapter → ImageSegmenter   │   │
│                                              │   │
│  Links: CoreAIKit (SPM)                      │   │
│    ├── CoreAIKitVision (GraphModel, etc.)    │   │
│    ├── CoreAILM (patched pipelined engine)   │   │
│    └── swift-transformers (HF tokenizer)     │   │
└──────────────────┬───────────────────────────┘   │
                   │                               │
     ┌─────────────┼────────────────┐              │
     │             │                │              │
     ▼             ▼                ▼              ▼
┌──────────────────────────────────────────────────────┐
│  LAYER 1 — FOUNDATION (existing ecosystem, not ours) │
│                                                      │
│  coreai-catalog (API)     CoreAIKit (SPM)            │
│  coreai-model-zoo         Apple Core AI framework    │
│  coreai-fabric (HF)       Apple Silicon (ANE/GPU/CPU)│
└──────────────────────────────────────────────────────┘
```

---

## Dependency Graph (precise)

```
                    ┌──────────────────────┐
                    │  coreai-catalog API  │
                    │  (HTTP JSON, público)│
                    └──────────┬───────────┘
                               │
              ┌────────────────┼─────────────────┐
              │ fetch metadata │                  │
              ▼                ▼                  ▼
    ┌──────────────┐  ┌──────────────┐   ┌──────────────┐
    │coreai-runner │  │ComfyUI-CoreAI│   │ coreai-server│
    │  (Swift)     │  │  (Python)    │   │   (Swift)    │
    └──────┬───────┘  └──────┬───────┘   └──────┬───────┘
           │                 │                   │
           │ SPM dep         │ subprocess        │ SPM dep
           │ (compile-time)  │ (runtime)         │ (compile-time)
           │                 │                   │
    ┌──────▼───────┐  ┌─────▼────────┐          │
    │  CoreAIKit   │  │ bridge.py    │          │
    │  (SPM)       │  │ manages proc │          │
    └──────┬───────┘  └──────────────┘          │
           │                                   │
           ├── CoreAIKitVision (GraphModel)     │
           ├── CoreAILM (patched engine)        │
           ├── swift-transformers              │
           └── system CoreAI framework          │
                                               │
    ┌─────────────────────────────────────────┘
    │
    │ coreai-server wraps coreai-runner with:
    │   ├── TCP HTTP (instead of Unix socket)
    │   ├── Bonjour/mDNS discovery
    │   ├── App UI (macOS/iOS)
    │   └── Multi-device coordination (future)
    │
    ▼
  LAN clients connect to coreai-server
  via Bonjour-discovered TCP endpoint
```

---

## How the Pieces Fit (Dependency Rules)

### coreai-runner (the shared engine)

**What it IS:**
- A Swift package (`CoreAIRunner`) that can be imported as SPM
- A standalone binary (`coreai-runner`) that serves HTTP over Unix socket
- The single source of truth for "how to load and run a Core AI model"

**What it is NOT:**
- Not a LAN server (that's coreai-server)
- Not a ComfyUI plugin (that's ComfyUI-CoreAI)
- Not an iOS app (that's Ditto)

**Consumers:**

| Consumer | How it uses coreai-runner |
|----------|--------------------------|
| **ComfyUI-CoreAI** | Downloads pre-compiled binary, spawns as subprocess, talks HTTP over Unix socket |
| **coreai-server** | Imports as SPM package, wraps with TCP HTTP + Bonjour, adds app UI |
| **Ditto** | Imports as SPM package (future), calls adapter APIs directly in-process |
| **CLI / scripts** | Runs binary standalone, pipes JSON via curl |

### ComfyUI-CoreAI (the ComfyUI plugin)

**What it IS:**
- A Python package (`comfyui_coreai`) with ComfyUI node definitions
- A bridge that manages the coreai-runner subprocess lifecycle
- A catalog client that populates model dropdowns dynamically

**What it is NOT:**
- Not the inference engine (that's coreai-runner)
- Not a Swift project (pure Python)

```
comfy node install ComfyUI-CoreAI
  → pip installs Python package
  → first predict() call downloads coreai-runner binary
  → binary spawned as subprocess on Unix socket
  → user never sees the Swift side
```

### coreai-server (the LAN inference server — FUTURE)

**What it IS:**
- A macOS/iOS app that wraps coreai-runner
- Exposes inference over TCP HTTP on the LAN (not Unix socket)
- Discovers and is discoverable via Bonjour/mDNS
- The "Ollama for Core AI vision" — but for multimodal, not LLMs

**What it is NOT:**
- Not competing with Ollama/Exo (different runtime, different models)
- Not needed for ComfyUI (ComfyUI uses the embedded runner, not the server)
- Not an LLM server (text LLMs are out of scope — Exo/Ollama own that)

**When to build it:**
After ComfyUI-CoreAI v1 is working. The server is a thin wrapper — if
coreai-runner is well-architected, coreai-server is mostly UI + networking.

---

## What Lives Where (Repository Contents)

### `kevinqz/coreai-runner`

```
coreai-runner/
├── Package.swift                 # SPM manifest (links CoreAIKit)
├── Sources/
│   ├── CoreAIRunner/             # Library target (importable by other Swift apps)
│   │   ├── CoreAIRunner.swift    # Public API facade
│   │   ├── Server/
│   │   │   ├── HTTPServer.swift  # Hummingbird Unix socket server
│   │   │   ├── Routes.swift      # /v1/health, /v1/models, /v1/predict
│   │   │   └── Codables.swift    # JSON request/response types
│   │   ├── Cache/
│   │   │   └── ModelCache.swift  # Actor-isolated LRU model manager
│   │   ├── Catalog/
│   │   │   └── CatalogClient.swift # Fetches coreai-catalog API (cached)
│   │   ├── Adapters/             # Thin wrappers over CoreAIKit types
│   │   │   ├── ModelAdapter.swift   # Protocol
│   │   │   ├── DepthAdapter.swift
│   │   │   ├── DetectionAdapter.swift
│   │   │   ├── VLMAdapter.swift
│   │   │   ├── DiffusionAdapter.swift
│   │   │   └── SegmenterAdapter.swift
│   │   └── Utils/
│   │       ├── ImageIO.swift
│   │       ├── DeviceInfo.swift
│   │       └── Logging.swift
│   └── coreai-runner-cli/        # Executable target (the binary)
│       └── main.swift            # Entry point, arg parse, launches server
├── Tests/
│   └── CoreAIRunnerTests/
│       └── ...
├── .github/workflows/
│   └── build.yml                 # Build binary → GitHub Release
├── README.md
└── LICENSE (Apache-2.0)
```

### `kevinqz/ComfyUI-CoreAI`

```
ComfyUI-CoreAI/
├── comfyui_coreai/               # Python package
│   ├── __init__.py               # ComfyUI node registration
│   ├── bridge.py                 # Subprocess lifecycle + HTTP client
│   ├── catalog.py                # Catalog API client (cached)
│   ├── image_utils.py            # ComfyUI tensor ↔ PNG file
│   ├── nodes/
│   │   ├── depth.py
│   │   ├── segmentation.py
│   │   ├── detection.py
│   │   ├── vlm.py
│   │   ├── image_gen.py
│   │   └── loader.py
│   ├── bin/                      # Downloaded binary (gitignored)
│   └── install.py                # comfy-cli post-install hook
├── ARCHITECTURE.md               # (already written)
├── docs/
│   └── OPEN_QUESTIONS_RESOLVED.md
├── README.md
├── pyproject.toml
└── LICENSE (Apache-2.0)
```

### `kevinqz/coreai-server` (FUTURE)

```
coreai-server/
├── Package.swift                 # SPM (depends on coreai-runner)
├── Sources/
│   ├── CoreAIServer/
│   │   ├── LANServer.swift       # TCP HTTP + Bonjour/mDNS
│   │   ├── DeviceDiscovery.swift # Find other coreai-server instances
│   │   └── MultiDeviceRouter.swift # (future) route to best device
│   └── coreai-server-app/        # macOS/iOS app
│       ├── App.swift             # SwiftUI entry
│       ├── Views/                # Model picker, status, settings
│       └── ...
├── README.md
└── LICENSE (Apache-2.0)
```

---

## Build & Release Pipeline

```
coreai-runner tag push (v1.0.0)
  │
  ├── GitHub Actions (macos-15 runner, Apple Silicon)
  │     swift build -c release
  │     strip binary
  │     upload to GitHub Release: coreai-runner-arm64-macos
  │
  └── ComfyUI-CoreAI install.py
        downloads binary from Release
        caches in package bin/

coreai-server (future)
  imports coreai-runner as SPM
  builds as standalone macOS/iOS app
  distributed via DMG / TestFlight
```

---

## Build Order (Dependency-Safe)

```
Phase 1: coreai-runner
  ├── SPM package structure
  ├── CoreAIKit integration
  ├── HTTP server (Hummingbird, Unix socket)
  ├── 3 adapters: Depth, Detection, VLM
  ├── Catalog API client
  ├── Binary build via GitHub Actions
  └── Tests: health, predict (depth), model listing

Phase 2: ComfyUI-CoreAI
  ├── bridge.py (subprocess lifecycle)
  ├── install.py (binary auto-download)
  ├── 3 nodes: Depth, Detection, VLM
  ├── Catalog dropdowns
  ├── Smoke test: comfy node install → generate
  └── Release: pip install

Phase 3: coreai-runner v1.1
  ├── Diffusion adapter (FLUX.2)
  ├── Segmenter adapter (SAM 3)
  ├── LRU model cache with sticky models
  └── Compute-unit routing (ANE for eligible models)

Phase 4: ComfyUI-CoreAI v1.1
  ├── SAM node (promptable segmentation)
  ├── FLUX.2 node (image generation)
  ├── CLIP embedding node
  └── Benchmark suite (Core AI vs PyTorch/MPS)

Phase 5: coreai-server (future)
  ├── TCP HTTP wrapper
  ├── Bonjour discovery
  ├── macOS app
  ├── iOS app
  └── Multi-device routing
```

---

## Non-Goals (explicitly excluded)

| What | Why |
|------|-----|
| Text LLM server | Ollama/Exo/LM Studio dominate. No value-add. |
| Model conversion | That's coreai-zoo + coreai-fabric. Not our job. |
| ComfyUI replacement | We're nodes inside ComfyUI, not a competitor. |
| Windows/Linux support | Core AI is Apple-only. By definition. |
| Cloud inference | On-device only. Private Cloud Compute is Apple's lane. |
