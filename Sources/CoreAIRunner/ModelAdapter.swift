// ModelAdapter.swift — uniform async interface that every model family conforms to.
// Adapters wrap CoreAIKit typed pipelines, translating file paths + JSON ↔ Swift.
//
// The adapter does NOT own model lifecycle — ModelCache handles load/unload.
// The adapter receives an already-loaded CoreAIKit object and runs inference.

import CoreGraphics
import Foundation

// MARK: - Input / Output

/// Decoded from the POST /v1/predict JSON body.
public struct AdapterInput: Sendable {
    public let modelID: String
    public let imagePath: String?
    public let prompt: String?
    public let maxTokens: Int?
    public let temperature: Double?
    public let scoreThreshold: Float?
    public let points: [PointPrompt]?
    public let boxes: [BoxPrompt]?
    public let textPrompt: String?
    public let computeUnit: ComputeUnitPreference

    public struct PointPrompt: Sendable, Codable {
        public let x: Float
        public let y: Float
        public let label: String  // "foreground" | "background"
    }

    public struct BoxPrompt: Sendable, Codable {
        public let x1: Float
        public let y1: Float
        public let x2: Float
        public let y2: Float
    }

    public enum ComputeUnitPreference: String, Sendable, Codable {
        case auto, gpu, neuralEngine, cpu
    }

    public init(
        modelID: String,
        imagePath: String? = nil,
        prompt: String? = nil,
        maxTokens: Int? = nil,
        temperature: Double? = nil,
        scoreThreshold: Float? = nil,
        points: [PointPrompt]? = nil,
        boxes: [BoxPrompt]? = nil,
        textPrompt: String? = nil,
        computeUnit: ComputeUnitPreference = .auto
    ) {
        self.modelID = modelID
        self.imagePath = imagePath
        self.prompt = prompt
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.scoreThreshold = scoreThreshold
        self.points = points
        self.boxes = boxes
        self.textPrompt = textPrompt
        self.computeUnit = computeUnit
    }
}

/// One inference result, written as the POST /v1/predict response body.
public struct AdapterOutput: Sendable {
    public enum Kind: String, Sendable {
        case depthMap
        case detections
        case masks
        case text
        case image
    }

    public let modelID: String
    public let kind: Kind
    /// Path to the primary output file (PNG for images, text for VLM).
    public let outputPath: String?
    /// Text output (VLM, captioning).
    public let text: String?
    /// Detection results (JSON-serializable).
    public let detections: [DetectionResult]?
    /// Segmentation masks (paths to individual mask PNGs).
    public let maskPaths: [MaskResult]?
    /// Timing breakdown.
    public let timing: Timing

    public struct DetectionResult: Sendable, Codable {
        public let label: String
        public let score: Float
        public let bbox: [Float]  // [x1, y1, x2, y2] normalized 0-1
    }

    public struct MaskResult: Sendable, Codable {
        public let maskPath: String
        public let score: Float
        public let bbox: [Float]  // [x1, y1, x2, y2] normalized 0-1

        // snake_case wire keys — the SotA convention for this API (matches
        // coreai-catalog, ComfyUI, HF). Swift keeps idiomatic camelCase props.
        enum CodingKeys: String, CodingKey {
            case maskPath = "mask_path"
            case score
            case bbox
        }
    }

    public struct Timing: Sendable, Codable {
        public let loadMs: Double
        public let preprocessMs: Double
        public let inferenceMs: Double
        public let postprocessMs: Double
        public let totalMs: Double
        public let computeUnitUsed: String  // "GPU" | "ANE" | "CPU"

        enum CodingKeys: String, CodingKey {
            case loadMs = "load_ms"
            case preprocessMs = "preprocess_ms"
            case inferenceMs = "inference_ms"
            case postprocessMs = "postprocess_ms"
            case totalMs = "total_ms"
            case computeUnitUsed = "compute_unit_used"
        }
    }

    public init(
        modelID: String,
        kind: Kind,
        outputPath: String? = nil,
        text: String? = nil,
        detections: [DetectionResult]? = nil,
        maskPaths: [MaskResult]? = nil,
        timing: Timing
    ) {
        self.modelID = modelID
        self.kind = kind
        self.outputPath = outputPath
        self.text = text
        self.detections = detections
        self.maskPaths = maskPaths
        self.timing = timing
    }
}

// MARK: - Adapter Protocol

/// Uniform inference interface. Each adapter wraps a loaded CoreAIKit object.
///
/// Lifecycle:
/// 1. `ModelCache.load(modelID:)` downloads the bundle and creates the
///    CoreAIKit pipeline (DepthEstimator, ObjectDetector, etc.).
/// 2. The cache stores the adapter in an LRU table.
/// 3. On predict, the cache returns the adapter; the route handler calls
///    `adapter.predict(input)`.
public protocol ModelAdapter: Sendable {
    var modelID: String { get }
    func predict(_ input: AdapterInput) async throws -> AdapterOutput
}

// MARK: - Errors

public enum CoreAIRunnerError: Error, LocalizedError {
    case modelNotInstalled(modelID: String)
    case modelLoadFailed(modelID: String, detail: String)
    case inferenceFailed(modelID: String, detail: String)
    case unsupportedDevice(modelID: String, reason: String)
    case patchRequired(modelID: String)
    case memoryInsufficient(modelID: String, requiredMB: Int, availableMB: Int)
    case timeout(modelID: String, seconds: Double)
    case invalidInput(detail: String)
    case missingInput(String)
    case catalogError(detail: String)

    public var errorDescription: String? {
        switch self {
        case .modelNotInstalled(let id):
            return "Model '\(id)' is not installed. Use POST /v1/models/\(id)/load to download."
        case .modelLoadFailed(let id, let detail):
            return "Failed to load model '\(id)': \(detail)"
        case .inferenceFailed(let id, let detail):
            return "Inference failed for model '\(id)': \(detail)"
        case .unsupportedDevice(let id, let reason):
            return "Model '\(id)' is not supported on this device: \(reason)"
        case .patchRequired(let id):
            return "Model '\(id)' requires an engine patch that is not available."
        case .memoryInsufficient(let id, let req, let avail):
            return "Insufficient memory for model '\(id)': needs \(req) MB, \(avail) MB available."
        case .timeout(let id, let seconds):
            return "Inference timed out for model '\(id)' after \(seconds) seconds."
        case .invalidInput(let detail):
            return "Invalid input: \(detail)"
        case .missingInput(let name):
            return "Missing required input: \(name)"
        case .catalogError(let detail):
            return "Catalog error: \(detail)"
        }
    }
}
