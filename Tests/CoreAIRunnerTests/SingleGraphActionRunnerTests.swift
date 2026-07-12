// SingleGraphActionRunnerTests — cross-language parity for the single-graph VLA path (Gate G
// breadth). Drives the fastwam-libero single-graph action expert through the Swift
// SingleGraphActionRunner and asserts the action chunk matches the Python coreai.runtime
// reference (scratchpad/fastwam_dump_fixture.py → $FASTWAM_FIXTURE). Opt-in via env.

import Foundation
import Testing
@testable import CoreAIRunner

private struct FwFixture {
    let dir: URL
    let manifest: [String: Any]
    init?(_ path: String) {
        dir = URL(fileURLWithPath: path)
        guard let d = try? Data(contentsOf: dir.appendingPathComponent("manifest.json")),
              let m = try? JSONSerialization.jsonObject(with: d) as? [String: Any] else { return nil }
        manifest = m
    }
    func floats(_ name: String) -> [Float] {
        let t = (manifest["tensors"] as! [String: Any])[name] as! [String: Any]
        let data = try! Data(contentsOf: dir.appendingPathComponent(t["file"] as! String))
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }
}

@Test(.enabled(if: ProcessInfo.processInfo.environment["FASTWAM_BUNDLE"] != nil
                 && ProcessInfo.processInfo.environment["FASTWAM_FIXTURE"] != nil))
func singleGraphRunnerMatchesPythonReference() async throws {
    let env = ProcessInfo.processInfo.environment
    let bundle = URL(fileURLWithPath: env["FASTWAM_BUNDLE"]!)
    let fx = FwFixture(env["FASTWAM_FIXTURE"]!)!

    let runner = try await SingleGraphActionRunner(
        modelID: "fastwam-libero", bundleAt: bundle, function: "action_denoise_step",
        noiseInput: "latents_action", timestepInput: "timestep", velocityOutput: "velocity",
        numSteps: (fx.manifest["num_steps"] as? Int) ?? 4)

    let chunk = try await runner.sampleActionChunk(
        noiseInit: fx.floats("latents_action_init"),
        conditioning: [
            "context": .floats(fx.floats("context")),
            "vk": .floats(fx.floats("vk")),
            "vv": .floats(fx.floats("vv")),
        ])

    let ref = fx.floats("ref_chunk")
    #expect(chunk.count == ref.count)
    var dot = 0.0, na = 0.0, nb = 0.0, maxAbs = 0.0
    for i in 0..<min(chunk.count, ref.count) {
        let a = Double(chunk[i]), b = Double(ref[i])
        dot += a * b; na += a * a; nb += b * b
        maxAbs = max(maxAbs, abs(a - b))
    }
    let cos = dot / (na.squareRoot() * nb.squareRoot() + 1e-12)
    print("SingleGraphActionRunner parity: cosine=\(cos) maxAbsDiff=\(maxAbs) n=\(chunk.count)")
    #expect(cos > 0.999, "single-graph action chunk must match the Python reference (cosine \(cos))")
    #expect(maxAbs < 0.05, "max abs diff \(maxAbs) too large")
}
