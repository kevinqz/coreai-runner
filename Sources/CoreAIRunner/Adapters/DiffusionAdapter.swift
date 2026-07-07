// DiffusionAdapter.swift — wraps CoreAIDiffusionPipeline (system framework) behind ModelAdapter.
//
// FLUX.2 klein 4B (text-to-image) via Apple's official runtime.
// The bundle is a directory containing multiple .aimodel components:
//   TextEncoder.aimodel, Transformer.aimodel, VAEDecoder.aimodel, VAEEncoder.aimodel
//   + tokenizer/ + vae_bn_mean.npy + vae_bn_var.npy + metadata.json
//
// PipelineDescriptor auto-detects the type from metadata.json and resolves
// components by convention name. One path, zero manual component juggling.
//
// macOS-only at 4B params (exceeds iOS per-process memory limit).
// 4-step distilled, guidance 1.0, discreteFlow scheduler.

#if canImport(CoreAIDiffusionPipeline)
import CoreAI
import CoreAIDiffusionPipeline
import CoreGraphics
import Foundation
import ImageIO

public struct DiffusionAdapter: ModelAdapter {
    public let modelID: String
    private let pipeline: any DiffusionPipeline
    private let descriptor: PipelineDescriptor

    /// Wrap an already-loaded pipeline.
    public init(modelID: String, pipeline: any DiffusionPipeline, descriptor: PipelineDescriptor) {
        self.modelID = modelID
        self.pipeline = pipeline
        self.descriptor = descriptor
    }

    /// Load from a local bundle directory.
    /// Uses PipelineDescriptor.resolve() for auto-detection.
    public init(modelID: String, bundleDir: URL) async throws {
        self.modelID = modelID
        let desc = try await PipelineDescriptor.resolve(at: bundleDir, config: .auto)

        let built: any DiffusionPipeline
        switch desc.type {
        case .some(.flux2):
            built = try await Flux2Pipeline(from: bundleDir, config: .auto, mode: .auto)
        case .some(.stableDiffusion3):
            built = try await SD3Pipeline(from: bundleDir, config: .auto)
        default:
            built = try await StableDiffusionPipeline.load(from: bundleDir, config: .auto)
        }

        self.descriptor = desc
        self.pipeline = built
    }

    public func predict(_ input: AdapterInput) async throws -> AdapterOutput {
        let startTime = Date()

        guard let prompt = input.prompt else {
            throw CoreAIRunnerError.missingInput("prompt")
        }

        // Build pipeline configuration
        // FLUX.2: 4-step distilled, guidance 1.0, discreteFlow scheduler
        let steps = 4
        let guidance: Float = 1.0

        let scheduler: SchedulerType =
            (descriptor.type == .flux2 || descriptor.type == .stableDiffusion3)
            ? .discreteFlow : .dpmSolverMultistep

        let config = PipelineConfiguration(
            prompt: prompt,
            negativePrompt: "",
            seed: UInt32.random(in: 0...UInt32.max),
            stepCount: steps,
            guidanceScale: guidance,
            schedulerType: scheduler,
            encoderScaleFactor: descriptor.encoderScaleFactor ?? 0.18215,
            decoderScaleFactor: descriptor.decoderScaleFactor ?? 0.18215,
            decoderShiftFactor: descriptor.decoderShiftFactor ?? 0.0,
            decodeResolution: .auto,
            lazyModelLoading: true
        )

        // Generate
        let t1 = Date()
        let result = try await pipeline.generateImages(configuration: config) { _ in
            return true  // never cancel mid-generation
        }
        let inferenceMs = Date().timeIntervalSince(t1) * 1000

        guard let cgImage = result.images.first else {
            throw CoreAIRunnerError.inferenceFailed(
                modelID: modelID, detail: "diffusion pipeline produced no images")
        }

        // Save generated image
        let t2 = Date()
        let outputPath = try ImageIO.savePNG(cgImage, prefix: "coreai_gen_\(modelID)")
        let postprocessMs = Date().timeIntervalSince(t2) * 1000
        let totalMs = Date().timeIntervalSince(startTime) * 1000

        return AdapterOutput(
            modelID: modelID,
            kind: .image,
            outputPath: outputPath,
            timing: AdapterOutput.Timing(
                loadMs: 0,
                preprocessMs: 0,
                inferenceMs: inferenceMs,
                postprocessMs: postprocessMs,
                totalMs: totalMs,
                computeUnitUsed: "GPU"
            )
        )
    }
}
#endif
