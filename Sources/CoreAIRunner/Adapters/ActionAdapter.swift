// ActionAdapter.swift — run a real LeRobot VLA/robot policy on-device via CoreAI.
//
// The runner's first robot-policy executor (RFC-0400 §3.1 / RFC-0800 Gate G "Policy Breadth").
// Targets the `lingbot-vla-v2` "chained-programs" flow-matching action expert: a 36-layer
// dense-MoE expert split (ANE single-program limit) into embed + block0/1/2, chained on the
// host through an Euler flow-matching denoise loop:
//
//   x_t = noise
//   for step in 0..<numSteps:            // dt = -1/numSteps ; time = 1 + step*dt
//       h,cond,cos,sin = embed(state, x_t, time, position_ids)
//       h = block0(h,cond,cos,sin, prefix_k[0:12],  prefix_v[0:12],  attn_mask)
//       h = block1(h,cond,cos,sin, prefix_k[12:24], prefix_v[12:24], attn_mask)
//       velocity = block2(h,cond,cos,sin, prefix_k[24:36], prefix_v[24:36], attn_mask)
//       x_t += dt * velocity
//   return x_t                            // action chunk [1, chunk, action_dim]
//
// The Qwen3-VL prefix KV (prefix_k/prefix_v) is HOST-SUPPLIED — the VLM prefill is a separate
// stage (manifest: "host prefills the Qwen3-VL backbone with collect_layer_kv_states=True").
// This adapter is the EXPERT executor; it takes the conditioning + observation state and
// returns the action chunk. Built on the raw CoreAI API (not CoreAIKitVision.GraphModel)
// because attn_mask is a bool tensor, which the GraphModel/TensorValue wrapper does not carry.

import CoreAI
import Foundation

public enum ActionAdapterError: Error, Sendable {
    case programNotFound(String)
    case badBundle(String)
    case dtypeUnsupported(String, String)
    case missingOutput(String)
}

/// One host-side tensor, tagged by source element type; converted to the graph's declared
/// scalar type at feed time (Float→Float16 where the graph wants float16, etc.).
public enum PolicyInput: Sendable {
    case floats([Float])
    case ints([Int32])
    case bools([Bool])
}

public final class ActionAdapter: @unchecked Sendable {
    private struct Program {
        let name: String
        let fn: InferenceFunction
        let desc: InferenceFunctionDescriptor
    }

    public let modelID: String
    public let numSteps: Int
    public let chunk: Int
    public let actionDim: Int
    public let stateDim: Int
    public let numLayers: Int

    private let embed: Program
    private let blocks: [(program: Program, layerLo: Int)]

    // prefix-KV + attn-mask geometry, read from the block0 descriptor at init.
    private let perBlockLayers: Int
    private let prefixSeqLen: Int
    private let nKV: Int
    private let headDim: Int
    private let maskRows: Int
    private let maskCols: Int
    private let posLen: Int

    /// Loads the chained-programs bundle from a local directory (the `.aimodel` dir with a
    /// `programs/` subfolder + `manifest.json`).
    public init(
        modelID: String, bundleAt bundleURL: URL
    ) async throws {
        self.modelID = modelID

        // Read the chained-programs manifest for the sampler shape (no hardcoded magic numbers).
        let manifestURL = bundleURL.appendingPathComponent("manifest.json")
        guard let mdata = try? Data(contentsOf: manifestURL),
              let m = try? JSONSerialization.jsonObject(with: mdata) as? [String: Any]
        else { throw ActionAdapterError.badBundle("missing/invalid manifest.json at \(manifestURL.path)") }
        self.numSteps = (m["num_denoise_steps"] as? Int) ?? 4
        self.chunk = (m["chunk_size"] as? Int) ?? 32
        self.actionDim = (m["action_dim"] as? Int) ?? 55
        self.numLayers = ((m["prefix_kv"] as? [String: Any])?["num_layers"] as? Int) ?? 36

        func load(_ p: String) async throws -> Program {
            let url = bundleURL.appendingPathComponent("programs/\(p).aimodel")
            let model = try await AIModel(
                contentsOf: url, options: SpecializationOptions(preferredComputeUnitKind: .gpu))
            guard let desc = model.functionDescriptor(for: p) else {
                throw ActionAdapterError.programNotFound(p)
            }
            guard let fn = try model.loadFunction(named: p) else {
                throw ActionAdapterError.programNotFound(p)
            }
            return Program(name: p, fn: fn, desc: desc)
        }

        self.embed = try await load("embed")
        let blockNames = ["block0", "block1", "block2"]
        self.perBlockLayers = numLayers / blockNames.count
        var loaded: [(Program, Int)] = []
        for (i, name) in blockNames.enumerated() {
            loaded.append((try await load(name), i * perBlockLayers))
        }
        self.blocks = loaded.map { (program: $0.0, layerLo: $0.1) }

        // Derive geometry from declared shapes (embed.state, block0.prefix_k, block0.attn_mask).
        func shape(_ prog: Program, _ input: String) -> [Int] {
            if case .ndArray(let d) = prog.desc.inputDescriptor(of: input) { return d.shape }
            return []
        }
        let stateShape = shape(embed, "state")            // [1, stateDim]
        self.stateDim = stateShape.count >= 2 ? stateShape[1] : actionDim
        let posShape = shape(embed, "position_ids")       // [1, posLen]
        self.posLen = posShape.count >= 2 ? posShape[1] : (chunk + 1)
        let pk = shape(blocks[0].program, "prefix_k")     // [perBlockLayers, 1, prefixSeq, nKV, head]
        self.prefixSeqLen = pk.count >= 5 ? pk[2] : 32
        self.nKV = pk.count >= 5 ? pk[3] : 8
        self.headDim = pk.count >= 5 ? pk[4] : 128
        let am = shape(blocks[0].program, "attn_mask")     // [1, rows, cols]
        self.maskRows = am.count >= 3 ? am[1] : (chunk + 1)
        self.maskCols = am.count >= 3 ? am[2] : 65
    }

    /// Build an NDArray for a named graph input, converting the host tensor to the graph's
    /// declared scalar type. `shape` resolves any dynamic dimensions.
    private func makeInput(_ prog: Program, _ name: String, _ input: PolicyInput, shape: [Int]) throws -> NDArray {
        guard case .ndArray(let d) = prog.desc.inputDescriptor(of: name) else {
            throw ActionAdapterError.dtypeUnsupported(name, "not an ndArray input")
        }
        let resolved = d.resolvingDynamicDimensions(shape)
        var array = NDArray(descriptor: resolved)
        switch (resolved.scalarType, input) {
        case (.float16, .floats(let v)):
            var view = array.mutableView(as: Float16.self); view.copyElements(fromContentsOf: v.map(Float16.init))
        case (.float32, .floats(let v)):
            var view = array.mutableView(as: Float.self); view.copyElements(fromContentsOf: v)
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

    /// Run the flow-matching Euler sampler and return the action chunk, row-major
    /// `[chunk * actionDim]` (shape `[1, chunk, actionDim]`).
    ///
    /// - prefixK/prefixV: host-supplied Qwen3-VL KV, row-major `[numLayers,1,prefixSeq,nKV,head]`.
    /// - attnMask: row-major `[1, maskRows, maskCols]` (true = attend).
    public func sampleActionChunk(
        state: [Float], noisyInit: [Float], positionIds: [Int32],
        prefixK: [Float], prefixV: [Float], attnMask: [Bool]
    ) async throws -> [Float] {
        var xt = noisyInit
        let dt = -1.0 / Float(numSteps)
        let kSliceLen = prefixK.count / blocks.count
        let vSliceLen = prefixV.count / blocks.count

        for step in 0..<numSteps {
            let time = 1.0 + Float(step) * dt
            var e = try await embed.fn.run(inputs: [
                "state": try makeInput(embed, "state", .floats(state), shape: [1, stateDim]),
                "noisy_actions": try makeInput(embed, "noisy_actions", .floats(xt), shape: [1, chunk, actionDim]),
                "timestep": try makeInput(embed, "timestep", .floats([time]), shape: [1]),
                "position_ids": try makeInput(embed, "position_ids", .ints(positionIds), shape: [1, posLen]),
            ])
            guard let h0 = e.remove("h")?.ndArray, let cond = e.remove("cond")?.ndArray,
                  let cos = e.remove("cos")?.ndArray, let sin = e.remove("sin")?.ndArray
            else { throw ActionAdapterError.missingOutput("embed outputs") }

            var h = h0
            for (idx, blk) in blocks.enumerated() {
                let kSlice = Array(prefixK[(idx * kSliceLen)..<((idx + 1) * kSliceLen)])
                let vSlice = Array(prefixV[(idx * vSliceLen)..<((idx + 1) * vSliceLen)])
                let feeds: [String: NDArray] = [
                    "h": h, "cond": cond, "cos": cos, "sin": sin,
                    "prefix_k": try makeInput(blk.program, "prefix_k", .floats(kSlice),
                                              shape: [perBlockLayers, 1, prefixSeqLen, nKV, headDim]),
                    "prefix_v": try makeInput(blk.program, "prefix_v", .floats(vSlice),
                                              shape: [perBlockLayers, 1, prefixSeqLen, nKV, headDim]),
                    "attn_mask": try makeInput(blk.program, "attn_mask", .bools(attnMask),
                                               shape: [1, maskRows, maskCols]),
                ]
                var out = try await blk.program.fn.run(inputs: feeds)
                if blk.program.name == "block2" {
                    guard let vel = out.remove("velocity")?.ndArray else {
                        throw ActionAdapterError.missingOutput("velocity")
                    }
                    let velocity = readFloats(vel)
                    for i in 0..<xt.count { xt[i] += dt * velocity[i] }
                } else {
                    guard let hOut = out.remove("h_out")?.ndArray else {
                        throw ActionAdapterError.missingOutput("h_out")
                    }
                    h = hOut
                }
            }
        }
        return xt
    }
}
