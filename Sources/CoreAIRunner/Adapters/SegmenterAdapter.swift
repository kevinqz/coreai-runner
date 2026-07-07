// SegmenterAdapter.swift — wraps CoreAIImageSegmenter (system framework) behind ModelAdapter.
//
// SAM 3 (text-prompt, open-vocabulary segmentation) via Apple's official runtime.
// The segmenter bundle is a directory containing metadata.json + .aimodel + tokenizer/.
//
// Input: PNG image path + text prompt ("cat", "the red car", etc.)
// Output: individual mask PNGs + bboxes + scores
//
// The segmenter is text-prompt: give it a phrase, not a fixed class list.
// Score = sigmoid(pred_logit) × sigmoid(presence_logit).
//
// iOS note: needs AOT-compiled .aimodelc bundle (JIT crashes on device).
// macOS: JIT .aimodel works fine.

#if canImport(CoreAIImageSegmenter)
import CoreGraphics
import Foundation
import CoreAIImageSegmenter

public struct SegmenterAdapter: ModelAdapter {
    public let modelID: String
    private let segmenter: ImageSegmenter

    /// Wrap an already-loaded ImageSegmenter.
    public init(modelID: String, segmenter: ImageSegmenter) {
        self.modelID = modelID
        self.segmenter = segmenter
    }

    /// Load from a local bundle directory (containing metadata.json + .aimodel + tokenizer/).
    public init(modelID: String, bundleDir: String) async throws {
        self.modelID = modelID
        let seg = try await ImageSegmenter(resourcesAt: bundleDir)
        try await seg.warmup()
        self.segmenter = seg
    }

    public func predict(_ input: AdapterInput) async throws -> AdapterOutput {
        let startTime = Date()

        guard let imagePath = input.imagePath else {
            throw CoreAIRunnerError.missingInput("imagePath")
        }
        // text prompt is required for SAM 3 (open-vocabulary)
        let prompt = input.textPrompt ?? input.prompt ?? ""
        guard !prompt.isEmpty else {
            throw CoreAIRunnerError.missingInput("textPrompt (or prompt)")
        }

        // Load image
        let t0 = Date()
        let cgImage = try ImageIO.loadCGImage(from: imagePath)
        let loadMs = Date().timeIntervalSince(t0) * 1000

        // Segmentation
        let t1 = Date()
        let params = SegmentationParameters(
            maskThreshold: 0.5,
            maxSegments: 12
        )
        let response = try await segmenter.segment(
            image: cgImage, prompt: prompt, parameters: params)
        let inferenceMs = Date().timeIntervalSince(t1) * 1000

        // Save each mask as a separate PNG
        let t2 = Date()
        let masks: [AdapterOutput.MaskResult] = try response.segments.enumerated().map { idx, seg in
            let maskPath = try saveBinaryMask(
                seg.mask, width: seg.maskWidth, height: seg.maskHeight,
                prefix: "coreai_mask_\(modelID)_\(idx)"
            )
            return AdapterOutput.MaskResult(
                maskPath: maskPath,
                score: seg.score,
                bbox: [
                    Float(seg.box.minX) / Float(cgImage.width),
                    Float(seg.box.minY) / Float(cgImage.height),
                    Float(seg.box.maxX) / Float(cgImage.width),
                    Float(seg.box.maxY) / Float(cgImage.height),
                ]
            )
        }
        let postprocessMs = Date().timeIntervalSince(t2) * 1000
        let totalMs = Date().timeIntervalSince(startTime) * 1000

        return AdapterOutput(
            modelID: modelID,
            kind: .masks,
            maskPaths: masks,
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

    /// Save a binary mask ([Bool]) as a white-on-black PNG for ComfyUI consumption.
    private func saveBinaryMask(
        _ mask: [Bool], width: Int, height: Int,
        prefix: String
    ) throws -> String {
        var bytes = [UInt8](repeating: 0, count: width * height)
        for i in 0..<mask.count where mask[i] {
            bytes[i] = 255
        }
        let data = Data(bytes)
        guard let provider = CGDataProvider(data: data as CFData) else {
            throw CoreAIRunnerError.inferenceFailed(
                modelID: modelID, detail: "cannot create mask data provider")
        }
        guard let cgImage = CGImage(
            width: width, height: height,
            bitsPerComponent: 8, bitsPerPixel: 8,
            bytesPerRow: width,
            space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
            provider: provider,
            decode: nil, shouldInterpolate: false,
            intent: .defaultIntent
        ) else {
            throw CoreAIRunnerError.inferenceFailed(
                modelID: modelID, detail: "cannot create mask CGImage")
        }
        return try ImageIO.savePNG(cgImage, prefix: prefix)
    }
}
#endif
