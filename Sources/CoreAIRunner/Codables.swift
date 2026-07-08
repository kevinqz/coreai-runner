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

    public struct PredictInput: Codable, Sendable {
        public let imagePath: String?
        public let prompt: String?
        public let maxTokens: Int?
        public let temperature: Double?
        public let scoreThreshold: Float?
        public let points: [AdapterInput.PointPrompt]?
        public let boxes: [AdapterInput.BoxPrompt]?
        public let textPrompt: String?

        enum CodingKeys: String, CodingKey {
            case prompt, temperature, points, boxes
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
    }
}

public struct PredictResponse: ResponseCodable, Sendable {
    public let modelID: String
    public let output: PredictOutput
    public let timing: AdapterOutput.Timing

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

    enum CodingKeys: String, CodingKey {
        case modelID = "model_id"
        case output
        case timing
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

    enum CodingKeys: String, CodingKey {
        case modelID = "model_id"
        case installed, loaded, download
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
