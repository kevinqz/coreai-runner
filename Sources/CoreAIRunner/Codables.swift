// Codables.swift — JSON types for the HTTP wire protocol.
// These are the Codable structs decoded from / encoded to HTTP request/response bodies.

import Foundation
import Hummingbird

// MARK: - Health

public struct HealthResponse: ResponseCodable, Sendable {
    public let status: String
    public let device: String
    public let chip: String
    public let memoryTotalGB: Double
    public let memoryAvailableGB: Double
    public let macosVersion: String
    public let coreaiVersion: String
    public let loadedModels: [String]
    public let thermalState: String

    // snake_case wire keys — SotA convention (matches coreai-catalog, ComfyUI,
    // HF). Swift keeps idiomatic camelCase properties.
    enum CodingKeys: String, CodingKey {
        case status, device, chip
        case memoryTotalGB = "memory_total_gb"
        case memoryAvailableGB = "memory_available_gb"
        case macosVersion = "macos_version"
        case coreaiVersion = "coreai_version"
        case loadedModels = "loaded_models"
        case thermalState = "thermal_state"
    }

    public init(
        status: String,
        device: String,
        chip: String,
        memoryTotalGB: Double,
        memoryAvailableGB: Double,
        macosVersion: String,
        coreaiVersion: String,
        loadedModels: [String],
        thermalState: String
    ) {
        self.status = status
        self.device = device
        self.chip = chip
        self.memoryTotalGB = memoryTotalGB
        self.memoryAvailableGB = memoryAvailableGB
        self.macosVersion = macosVersion
        self.coreaiVersion = coreaiVersion
        self.loadedModels = loadedModels
        self.thermalState = thermalState
    }
}

// MARK: - Model listing

public struct ModelListResponse: ResponseCodable, Sendable {
    public let models: [ModelEntry]

    public struct ModelEntry: Codable, Sendable {
        public let id: String
        public let name: String
        public let family: String
        public let capability: String
        public let sizeMB: Double?
        public let precision: String?
        public let license: String?
        public let commercialUse: String?
        public let deviceSupport: [String]
        public let installed: Bool
        public let loaded: Bool
        public let benchmarkMs: Double?
        public let runner: String?

        enum CodingKeys: String, CodingKey {
            case id, name, family, capability, precision, license, installed, loaded, runner
            case sizeMB = "size_mb"
            case commercialUse = "commercial_use"
            case deviceSupport = "device_support"
            case benchmarkMs = "benchmark_ms"
        }

        public init(
            id: String,
            name: String,
            family: String,
            capability: String,
            sizeMB: Double?,
            precision: String?,
            license: String?,
            commercialUse: String?,
            deviceSupport: [String],
            installed: Bool,
            loaded: Bool,
            benchmarkMs: Double?,
            runner: String?
        ) {
            self.id = id
            self.name = name
            self.family = family
            self.capability = capability
            self.sizeMB = sizeMB
            self.precision = precision
            self.license = license
            self.commercialUse = commercialUse
            self.deviceSupport = deviceSupport
            self.installed = installed
            self.loaded = loaded
            self.benchmarkMs = benchmarkMs
            self.runner = runner
        }
    }
}

// MARK: - Predict request/response

public struct PredictRequest: Codable, Sendable {
    public let modelID: String
    public let input: PredictInput
    public let options: PredictOptions?
    /// Runtime kind: "vision" (default, image-based inference) or "action" (robot policy
    /// — observation with images/state/task → action chunk). Spec §19.1.
    public let runtimeKind: String?

    public struct PredictInput: Codable, Sendable {
        public let imagePath: String?
        public let prompt: String?
        public let maxTokens: Int?
        public let temperature: Double?
        public let scoreThreshold: Float?
        public let points: [AdapterInput.PointPrompt]?
        public let boxes: [AdapterInput.BoxPrompt]?
        public let textPrompt: String?
        // Action policy observation (runtime_kind=action): images are paths, state is a float array.
        public let observation: ActionObservation?

        enum CodingKeys: String, CodingKey {
            case prompt, temperature, points, boxes, observation
            case imagePath = "image_path"
            case maxTokens = "max_tokens"
            case scoreThreshold = "score_threshold"
            case textPrompt = "text_prompt"
        }
    }

    public struct PredictOptions: Codable, Sendable {
        public let computeUnit: AdapterInput.ComputeUnitPreference?

        enum CodingKeys: String, CodingKey {
            case computeUnit = "compute_unit"
        }
    }

    enum CodingKeys: String, CodingKey {
        case modelID = "model_id"
        case input
        case options
        case runtimeKind = "runtime_kind"
    }
}

/// Observation for a robot policy action request (spec §19.1). Images are file paths to
/// temp-written frames; state is the robot's proprioceptive joint positions; task is an
/// optional language instruction.
public struct ActionObservation: Codable, Sendable {
    /// Keyed by LeRobot feature name: "observation.images.wrist", "observation.images.front", etc.
    public let images: [String: String]?
    public let state: [Double]?
    public let task: String?

    enum CodingKeys: String, CodingKey {
        case images, state, task
    }
}

public struct PredictResponse: ResponseCodable, Sendable {
    public let modelID: String
    public let output: PredictOutput
    public let timing: AdapterOutput.Timing
    /// Action chunk for robot policies (runtime_kind=action, spec §19.1). [[Float]] —
    /// shape [chunk_size, action_dim]. Absent for vision/text models.
    public let action: [[Double]]?
    public let actionFeatures: ActionFeatures?

    public struct PredictOutput: Codable, Sendable {
        public let kind: String
        public let outputPath: String?
        public let text: String?
        public let detections: [AdapterOutput.DetectionResult]?
        public let maskPaths: [AdapterOutput.MaskResult]?

        enum CodingKeys: String, CodingKey {
            case kind, text, detections
            case outputPath = "output_path"
            case maskPaths = "mask_paths"
        }
    }

    /// Action metadata (spec §19.1 response).
    public struct ActionFeatures: Codable, Sendable {
        public let shape: [Int]?
        public let representation: String?

        enum CodingKeys: String, CodingKey {
            case shape, representation
        }
    }

    enum CodingKeys: String, CodingKey {
        case modelID = "model_id"
        case output
        case timing
        case action
        case actionFeatures = "action_features"
    }

    // Existing init for vision/text (no action fields).
    public init(modelID: String, output: PredictOutput, timing: AdapterOutput.Timing) {
        self.modelID = modelID
        self.output = output
        self.timing = timing
        self.action = nil
        self.actionFeatures = nil
    }

    // Action init (spec §19.1).
    public init(modelID: String, action: [[Double]], actionFeatures: ActionFeatures?,
                timing: AdapterOutput.Timing) {
        self.modelID = modelID
        self.output = PredictOutput(kind: "action", outputPath: nil, text: nil, detections: nil, maskPaths: nil)
        self.timing = timing
        self.action = action
        self.actionFeatures = actionFeatures
    }
}

// MARK: - Load / Unload

public struct LoadRequest: Codable, Sendable {
    public let force: Bool?
}

public struct LoadResponse: ResponseCodable, Sendable {
    public let modelID: String
    public let status: String  // "loaded" | "already_loaded" | "downloading"
    public let sizeMB: Double?

    enum CodingKeys: String, CodingKey {
        case modelID = "model_id"
        case status
        case sizeMB = "size_mb"
    }
}

// MARK: - Model status + download progress

public struct DownloadStatus: Codable, Sendable {
    public let modelID: String
    public let fraction: Double
    public let state: State

    public enum State: String, Codable, Sendable {
        case queued, downloading, completed, failed
    }

    enum CodingKeys: String, CodingKey {
        case modelID = "model_id"
        case fraction, state
    }

    public init(modelID: String, fraction: Double, state: State) {
        self.modelID = modelID
        self.fraction = fraction
        self.state = state
    }
}

public struct ModelStatusResponse: ResponseCodable, Sendable {
    public let modelID: String
    public let installed: Bool
    public let loaded: Bool
    public let download: DownloadStatus?
    /// Compilation mode: "aot" (.aimodelc, ahead-of-time compiled), "jit" (.aimodel, on-device
    /// specialization), or nil if unknown. On iOS, JIT bundles >4B risk jetsam kills — this field
    /// lets callers warn the user proactively (the Zoo hard-codes AOT-only for iOS; we surface it).
    public let compilation: String?
    /// Active engine variant: "coreai-sequential" or "coreai-pipelined" (LLM models only).
    /// nil for non-LLM models or when not loaded.
    public let engineVariant: String?
    /// Graph-role awareness for split policies (spec §19.3). Each graph has a name, functional
    /// role, and loaded state. Absent for single-graph models.
    public let graphs: [GraphInfo]?
    /// Host-loop requirements for action policies (spec §19.3). Absent for non-action models.
    public let hostLoop: HostLoopInfo?

    public struct GraphInfo: Codable, Sendable {
        public let name: String       // e.g. "encode", "denoise_step"
        public let role: String       // e.g. "context_encoder", "denoise_step"
        public let loaded: Bool

        public init(name: String, role: String, loaded: Bool) {
            self.name = name
            self.role = role
            self.loaded = loaded
        }
    }

    public struct HostLoopInfo: Codable, Sendable {
        public let required: Bool
        public let supported: Bool  // runner supports host_loop (always true via capabilities)

        public init(required: Bool, supported: Bool = true) {
            self.required = required
            self.supported = supported
        }
    }

    enum CodingKeys: String, CodingKey {
        case modelID = "model_id"
        case installed, loaded, download, compilation
        case engineVariant = "engine_variant"
        case graphs
        case hostLoop = "host_loop"
    }

    public init(modelID: String, installed: Bool, loaded: Bool,
                download: DownloadStatus? = nil, compilation: String? = nil,
                engineVariant: String? = nil,
                graphs: [GraphInfo]? = nil, hostLoop: HostLoopInfo? = nil) {
        self.modelID = modelID
        self.installed = installed
        self.loaded = loaded
        self.download = download
        self.compilation = compilation
        self.engineVariant = engineVariant
        self.graphs = graphs
        self.hostLoop = hostLoop
    }
}

// MARK: - Capabilities (spec §19.2)

/// Advertises which runtime kinds the runner supports. Lets clients (lerobot-coreai, ComfyUI)
/// discover features without guessing.
public struct CapabilitiesResponse: ResponseCodable, Sendable {
    public let runtime: String = "coreai-runner"
    /// Explicit protocol version (RFC-0400 §3.4 / coreai-interop). Clients negotiate on
    /// this, not on the absence of a field.
    public let protocolVersion: String = "coreai-runner.v2"
    public let supports: Supports
    /// RFC-0400 §3.3 / coreai-interop runner-capabilities.v2: the v2 envelope REQUIRES
    /// `action_batching` and `inference_state`. Values mirror the interop `split` profile
    /// (fixtures/valid/runner-capabilities.split.json) — the truthful shape for this runner:
    /// action inference is unsupported (action=false), policies are split/multi-graph, and
    /// inference is stateless (prefix cache is a transparent optimization, not session state).
    public let actionBatching = ActionBatching()
    public let inferenceState = InferenceState()

    public struct Supports: Codable, Sendable {
        public let llm: Bool = true
        public let vlm: Bool = true
        // RFC-0400 §3.1: action inference returns 501, so `action` MUST advertise false
        // until a real ActionAdapter + host loop lands (lerobot-coreai v0.2 scope).
        public let action: Bool = false
        public let rewardModel: Bool = false // future
        public let multiGraph: Bool = true   // split policies (encode + denoise_step)
        // RFC-0400 §3.2: no conformance case executes a host loop yet, so `host_loop`
        // MUST advertise false until one does.
        public let hostLoop: Bool = false
        public let prefixCache: Bool = true  // KV reuse for multi-turn chat
    }

    public struct ActionBatching: Codable, Sendable {
        // action=false ⇒ no action batching. `split_and_stack` documents the intended mode
        // when an ActionAdapter lands; `independent` = per-slot isolation.
        public let supported: Bool = false
        public let semantics: String = "split_and_stack"
        public let slotIsolation: String = "independent"

        enum CodingKeys: String, CodingKey {
            case supported, semantics
            case slotIsolation = "slot_isolation"
        }
    }

    public struct InferenceState: Codable, Sendable {
        // Stateless per request; the prefix cache is a transparent KV optimization, not
        // durable session state, so no session ids and nothing to reset.
        public let scope: String = "stateless"
        public let supportsSessionIds: Bool = false
        public let resetScope: String = "none"

        enum CodingKeys: String, CodingKey {
            case scope
            case supportsSessionIds = "supports_session_ids"
            case resetScope = "reset_scope"
        }
    }

    enum CodingKeys: String, CodingKey {
        case runtime
        case protocolVersion = "protocol_version"
        case supports
        case actionBatching = "action_batching"
        case inferenceState = "inference_state"
    }

    public init() {
        self.supports = Supports()
    }
}

// MARK: - Error response

public struct ErrorResponse: ResponseCodable, Sendable {
    public let error: ErrorDetail

    public struct ErrorDetail: Codable, Sendable {
        public let code: String
        public let message: String
        public let modelID: String?

        enum CodingKeys: String, CodingKey {
            case code
            case message
            case modelID = "model_id"
        }
    }

    public init(code: String, message: String, modelID: String? = nil) {
        self.error = ErrorDetail(code: code, message: message, modelID: modelID)
    }
}

// MARK: - Chat completions (OpenAI-compatible)

public struct ChatCompletionRequest: Codable, Sendable {
    public let model: String
    public let messages: [Message]
    public let maxTokens: Int?
    public let temperature: Double?

    public struct Message: Codable, Sendable {
        public let role: String      // "user", "assistant", "system"
        public let content: String
    }

    enum CodingKeys: String, CodingKey {
        case model, messages
        case maxTokens = "max_tokens"
        case temperature
    }
}

public struct ChatCompletionResponse: ResponseCodable, Sendable {
    public let id: String
    public let object: String = "chat.completion"
    public let model: String
    public let choices: [Choice]

    public struct Choice: Codable, Sendable {
        public let index: Int
        public let message: Message
        public let finishReason: String

        enum CodingKeys: String, CodingKey {
            case index, message
            case finishReason = "finish_reason"
        }
    }

    public struct Message: Codable, Sendable {
        public let role: String
        public let content: String
    }

    public let usage: Usage

    public struct Usage: Codable, Sendable {
        public let promptTokens: Int
        public let completionTokens: Int
        public let totalTokens: Int

        enum CodingKeys: String, CodingKey {
            case promptTokens = "prompt_tokens"
            case completionTokens = "completion_tokens"
            case totalTokens = "total_tokens"
        }
    }

    public init(id: String, model: String, choices: [Choice], usage: Usage) {
        self.id = id
        self.model = model
        self.choices = choices
        self.usage = usage
    }
}

// MARK: - Chat streaming (SSE chunks)

public struct ChatStreamChunk: ResponseCodable, Sendable {
    public let id: String = "chatcmn-stream"
    public let object: String = "chat.completion.chunk"
    public let model: String = ""
    public let choices: [Choice]
    public let stats: StreamStats?

    public struct Choice: Codable, Sendable {
        public let index: Int
        public let delta: Delta
        public let finishReason: String?

        public init(index: Int, delta: Delta, finishReason: String?) {
            self.index = index
            self.delta = delta
            self.finishReason = finishReason
        }

        enum CodingKeys: String, CodingKey {
            case index, delta
            case finishReason = "finish_reason"
        }
    }

    public struct Delta: Codable, Sendable {
        public let content: String

        public init(content: String) {
            self.content = content
        }
    }

    public struct StreamStats: Codable, Sendable {
        public let promptTokens: Int
        public let reusedPromptTokens: Int
        public var generatedTokens: Int?
        public var tokensPerSecond: Double?

        public init(
            promptTokens: Int, reusedPromptTokens: Int,
            generatedTokens: Int? = nil, tokensPerSecond: Double? = nil
        ) {
            self.promptTokens = promptTokens
            self.reusedPromptTokens = reusedPromptTokens
            self.generatedTokens = generatedTokens
            self.tokensPerSecond = tokensPerSecond
        }

        enum CodingKeys: String, CodingKey {
            case promptTokens = "prompt_tokens"
            case reusedPromptTokens = "reused_prompt_tokens"
            case generatedTokens = "generated_tokens"
            case tokensPerSecond = "tokens_per_second"
        }
    }

    public init(choices: [Choice], stats: StreamStats? = nil) {
        self.choices = choices
        self.stats = stats
    }

    public init(error message: String) {
        self.choices = []
        self.stats = nil
    }
}

enum ChatStreamError: Error {
    case streamError
}
