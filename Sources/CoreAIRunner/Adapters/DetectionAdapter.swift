// DetectionAdapter.swift — wraps CoreAIKitVision.ObjectDetector behind ModelAdapter.
//
// Input: PNG file path → CGImage → ObjectDetector.detect() → [Detection]
// Output: JSON detections (label, score, bbox)
//
// RF-DETR models need no NMS (DETR family: sigmoid + threshold).
// YOLOX-S is a dense detector — ObjectDetector handles obj·cls + NMS host-side.
// Supports split deployment (backbone→ANE, head→GPU) when split bundles exist.
//
// Detection COCO classes (91 categories with gaps) are built into ObjectDetector.

import CoreGraphics
import Foundation
import CoreAIKitVision

public struct DetectionAdapter: ModelAdapter {
    public let modelID: String
    private let detector: ObjectDetector

    /// Wrap an already-loaded ObjectDetector.
    public init(modelID: String, detector: ObjectDetector) {
        self.modelID = modelID
        self.detector = detector
    }

    /// Download (if needed) and load via CoreAIKit's catalog-aware initializer.
    /// Auto-detects split models (backbone + head) for RF-DETR.
    public init(
        catalog id: String,
        store: ModelStore = .default,
        computeUnits: GraphModel.ComputeUnits = .gpu,
        downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws {
        self.modelID = id
        self.detector = try await ObjectDetector(
            catalog: id,
            store: store,
            computeUnits: computeUnits,
            downloadProgress: downloadProgress)
    }

    public func predict(_ input: AdapterInput) async throws -> AdapterOutput {
        let startTime = Date()

        guard let imagePath = input.imagePath else {
            throw CoreAIRunnerError.missingInput("imagePath")
        }

        // Load image
        let t0 = Date()
        let cgImage = try ImageIO.loadCGImage(from: imagePath)
        let loadMs = Date().timeIntervalSince(t0) * 1000

        // Inference
        let t1 = Date()
        let threshold = input.scoreThreshold ?? 0.5
        let detections = try await detector.detect(
            in: cgImage,
            scoreThreshold: threshold,
            maxDetections: 100
        )
        let inferenceMs = Date().timeIntervalSince(t1) * 1000

        // Convert to output
        let t2 = Date()
        let results: [AdapterOutput.DetectionResult] = detections.map { det in
            AdapterOutput.DetectionResult(
                label: det.label,
                score: det.score,
                bbox: [
                    Float(det.box.origin.x),
                    Float(det.box.origin.y),
                    Float(det.box.origin.x + det.box.width),
                    Float(det.box.origin.y + det.box.height),
                ]
            )
        }
        let postprocessMs = Date().timeIntervalSince(t2) * 1000

        let totalMs = Date().timeIntervalSince(startTime) * 1000

        return AdapterOutput(
            modelID: modelID,
            kind: .detections,
            detections: results,
            timing: AdapterOutput.Timing(
                loadMs: loadMs,
                preprocessMs: 0,
                inferenceMs: inferenceMs,
                postprocessMs: postprocessMs,
                totalMs: totalMs,
                computeUnitUsed: "GPU"
            )
        )
    }
}
