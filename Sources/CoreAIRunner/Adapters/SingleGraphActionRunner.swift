// SingleGraphActionRunner.swift — the single-graph counterpart to ActionAdapter (Gate G).
//
// Some VLA action experts export as ONE stateless denoise graph (e.g. fastwam-libero,
// molmoact2-libero: `action_denoise_step(latents_action, timestep, context, vk, vv) -> velocity`)
// rather than the chained embed+block programs that `ActionAdapter` drives (lingbot-vla-v2).
// The runner's action machinery is architecture-agnostic (RFC-0800 Gate G "no core special
// cases"): the same raw-CoreAI NDArray plumbing + host Euler flow-matching loop drives either
// shape. This runner covers the single-graph case:
//
//   x_t = noise
//   for step in 0..<numSteps (dt = -1/N, time = 1 + step*dt):
//       velocity = graph(noiseInput: x_t, timestepInput: time, <conditioning...>)
//       x_t += dt * velocity
//   return x_t
//
// Conditioning (context/vk/vv — the VLM representation) is host-supplied, as with ActionAdapter.

import CoreAI
import Foundation

public final class SingleGraphActionRunner: @unchecked Sendable {
    public let modelID: String
    public let numSteps: Int
    private let fn: InferenceFunction
    private let desc: InferenceFunctionDescriptor
    private let noiseInput: String
    private let timestepInput: String
    private let velocityOutput: String

    /// Loads a single-graph action `.aimodel`.
    public init(
        modelID: String, bundleAt url: URL, function: String,
        noiseInput: String, timestepInput: String, velocityOutput: String,
        numSteps: Int
    ) async throws {
        self.modelID = modelID
        self.numSteps = numSteps
        self.noiseInput = noiseInput
        self.timestepInput = timestepInput
        self.velocityOutput = velocityOutput
        let model = try await AIModel(
            contentsOf: url, options: SpecializationOptions(preferredComputeUnitKind: .gpu))
        guard let d = model.functionDescriptor(for: function) else {
            throw ActionAdapterError.programNotFound(function)
        }
        guard let f = try model.loadFunction(named: function) else {
            throw ActionAdapterError.programNotFound(function)
        }
        self.desc = d
        self.fn = f
    }

    private func declShape(_ name: String) -> [Int] {
        if case .ndArray(let d) = desc.inputDescriptor(of: name) { return d.shape }
        return []
    }

    private func makeInput(_ name: String, _ input: PolicyInput) throws -> NDArray {
        guard case .ndArray(let d) = desc.inputDescriptor(of: name) else {
            throw ActionAdapterError.dtypeUnsupported(name, "not an ndArray input")
        }
        let resolved = d.resolvingDynamicDimensions(d.shape)
        var array = NDArray(descriptor: resolved)
        switch (resolved.scalarType, input) {
        case (.float16, .floats(let v)):
            var view = array.mutableView(as: Float16.self); view.copyElements(fromContentsOf: v.map(Float16.init))
        case (.float32, .floats(let v)):
            var view = array.mutableView(as: Float.self); view.copyElements(fromContentsOf: v)
        case (.int32, .floats(let v)):
            var view = array.mutableView(as: Int32.self); view.copyElements(fromContentsOf: v.map { Int32($0) })
        case (.int32, .ints(let v)):
            var view = array.mutableView(as: Int32.self); view.copyElements(fromContentsOf: v)
        case (.bool, .bools(let v)):
            var view = array.mutableView(as: Bool.self); view.copyElements(fromContentsOf: v)
        default:
            throw ActionAdapterError.dtypeUnsupported(name, "\(resolved.scalarType) vs host tensor")
        }
        return array
    }

    private func readFloats(_ array: NDArray) -> [Float] {
        let count = array.shape.reduce(1, *)
        switch array.scalarType {
        case .float16:
            return array.view(as: Float16.self).withUnsafePointer { p, _, _ in (0..<count).map { Float(p[$0]) } }
        case .float32:
            return array.view(as: Float.self).withUnsafePointer { p, _, _ in Array(UnsafeBufferPointer(start: p, count: count)) }
        default:
            return []
        }
    }

    /// Run the Euler flow-matching sampler. `conditioning` supplies every graph input other
    /// than the noise/timestep (e.g. context/vk/vv); shapes come from the graph descriptor.
    public func sampleActionChunk(
        noiseInit: [Float], conditioning: [String: PolicyInput]
    ) async throws -> [Float] {
        var xt = noiseInit
        let dt = -1.0 / Float(numSteps)
        for step in 0..<numSteps {
            let time = 1.0 + Float(step) * dt
            var feeds: [String: NDArray] = [
                noiseInput: try makeInput(noiseInput, .floats(xt)),
                timestepInput: try makeInput(timestepInput, .floats([time])),
            ]
            for (name, value) in conditioning {
                feeds[name] = try makeInput(name, value)
            }
            var out = try await fn.run(inputs: feeds)
            guard let vel = out.remove(velocityOutput)?.ndArray else {
                throw ActionAdapterError.missingOutput(velocityOutput)
            }
            let velocity = readFloats(vel)
            for i in 0..<xt.count { xt[i] += dt * velocity[i] }
        }
        return xt
    }
}
