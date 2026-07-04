// VLMAdapter.swift — wraps CoreAIKit.KitVisionModel behind ModelAdapter.
//
// Input: PNG file path + text prompt → CGImage + Prompt → LanguageModelSession.respond()
// Output: generated text
//
// Qwen3-VL (2B/4B/8B) and Holo2-4B use the patched pipelined engine (static-inputs
// hook). The patch is already baked into CoreAIKit's CoreAILM SPM dependency.
//
// The vision tower runs ONCE per image (~60-80ms on Mac) and image embeddings are
// injected into the text decoder via MTLBuffers. Multi-turn about the same image
// holds the session — each /v1/predict creates a fresh session for now.
//
// IMPORTANT: VLM is NOT available on all devices.
//   - Qwen3-VL 2B: iPhone + Mac (2.3 GB)
//   - Qwen3-VL 4B: iPhone (thermally limited) + Mac (4.7 GB)
//   - Qwen3-VL 8B: Mac only (8.7 GB)

import CoreGraphics
import Foundation
import CoreAIKit
import FoundationModels

public struct VLMAdapter: ModelAdapter {
    public let modelID: String
    private let model: KitVisionModel
    private let session: LanguageModelSession

    /// Wrap an already-loaded KitVisionModel.
    public init(modelID: String, model: KitVisionModel) {
        self.modelID = modelID
        self.model = model
        self.session = LanguageModelSession(model: model)
    }

    /// Download (if needed) and load via CoreAIKit's catalog-aware initializer.
    public init(
        catalog id: String,
        store: ModelStore = .default,
        downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws {
        let vlm = try await KitVisionModel(catalog: id, store: store, downloadProgress: downloadProgress)
        self.modelID = id
        self.model = vlm
        self.session = LanguageModelSession(model: vlm)
    }

    public func predict(_ input: AdapterInput) async throws -> AdapterOutput {
        let startTime = Date()

        guard let imagePath = input.imagePath else {
            throw CoreAIRunnerError.missingInput("imagePath")
        }
        guard let prompt = input.prompt else {
            throw CoreAIRunnerError.missingInput("prompt")
        }

        // Load image
        let t0 = Date()
        let cgImage = try ImageIO.loadCGImage(from: imagePath)
        let loadMs = Date().timeIntervalSince(t0) * 1000

        // Inference: send image + prompt to the VLM session
        let t1 = Date()
        let reply = try await session.respond(to: Prompt {
            prompt
            Attachment(cgImage)
        })
        let inferenceMs = Date().timeIntervalSince(t1) * 1000

        let totalMs = Date().timeIntervalSince(startTime) * 1000

        return AdapterOutput(
            modelID: modelID,
            kind: .text,
            text: reply.content,
            timing: AdapterOutput.Timing(
                loadMs: loadMs,
                preprocessMs: 0,
                inferenceMs: inferenceMs,
                postprocessMs: 0,
                totalMs: totalMs,
                computeUnitUsed: "GPU"
            )
        )
    }
}
