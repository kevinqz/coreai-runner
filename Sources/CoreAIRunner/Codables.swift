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
            case imagePath = "image_path"
            case prompt
            case maxTokens = "max_tokens"
            case temperature
            case scoreThreshold = "score_threshold"
            case points
            case boxes
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
