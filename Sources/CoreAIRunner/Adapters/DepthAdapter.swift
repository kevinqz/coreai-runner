// DepthAdapter.swift — wraps CoreAIKitVision.DepthEstimator behind ModelAdapter.
//
// Input: PNG file path → CGImage → DepthEstimator.estimateDepth() → DepthMap
// Output: grayscale depth PNG saved to /tmp
//
// The DepthEstimator auto-detects the image input name and output name from
// the graph descriptor. ImageNet normalization is folded in-graph (feed RAW
// [0,1] RGB). Compute unit defaults to GPU but can be overridden to ANE.

import CoreGraphics
import Foundation
import CoreAIKitVision

public struct DepthAdapter: ModelAdapter {
    public let modelID: String
    private let estimator: DepthEstimator

    /// Wrap an already-loaded DepthEstimator.
    public init(modelID: String, estimator: DepthEstimator) {
        self.modelID = modelID
        self.estimator = estimator
    }

    /// Download (if needed) and load via CoreAIKit's catalog-aware initializer.
    public init(
        catalog id: String,
        store: ModelStore = .default,
        computeUnits: GraphModel.ComputeUnits = .gpu,
        downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws {
        self.modelID = id
        self.estimator = try await DepthEstimator(
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
        let depthMap = try await estimator.estimateDepth(for: cgImage)
        let inferenceMs = Date().timeIntervalSince(t1) * 1000

        // Save as grayscale PNG
        let t2 = Date()
        let outputPath = try ImageIO.saveGrayscalePNG(
            width: depthMap.width,
            height: depthMap.height,
            values: depthMap.values,
            prefix: "coreai_depth_\(modelID)"
        )
        let postprocessMs = Date().timeIntervalSince(t2) * 1000

        let totalMs = Date().timeIntervalSince(startTime) * 1000

        return AdapterOutput(
            modelID: modelID,
            kind: .depthMap,
            outputPath: outputPath,
            timing: AdapterOutput.Timing(
                loadMs: loadMs,
                preprocessMs: 0,  // preprocessing is inside estimateDepth
                inferenceMs: inferenceMs,
                postprocessMs: postprocessMs,
                totalMs: totalMs,
                computeUnitUsed: "GPU"
            )
        )
    }
}
