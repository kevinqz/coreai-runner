// ActionAdapterLingbotTests — cross-language parity for the robot-policy executor (Gate G).
// Loads the seeded inputs + reference action chunk produced by the Python coreai.runtime probe
// (scratchpad/lingbot_dump_fixture.py → $LINGBOT_FIXTURE), drives the SAME lingbot-vla-v2
// chained expert through the Swift ActionAdapter's flow-matching sampler, and asserts the
// Swift action chunk matches the Python reference (chunk cosine + max abs diff). Opt-in:
// runs only when LINGBOT_BUNDLE (the .aimodel dir) and LINGBOT_FIXTURE are set; CI (no macOS 27
// SDK, no bundle) skips it.

import Foundation
import Testing
@testable import CoreAIRunner

private struct Fixture {
    let dir: URL
    let manifest: [String: Any]
    init?(_ path: String) {
        dir = URL(fileURLWithPath: path)
        guard let d = try? Data(contentsOf: dir.appendingPathComponent("manifest.json")),
              let m = try? JSONSerialization.jsonObject(with: d) as? [String: Any] else { return nil }
        manifest = m
    }
    private func file(_ name: String) -> URL {
        let t = (manifest["tensors"] as! [String: Any])[name] as! [String: Any]
        return dir.appendingPathComponent(t["file"] as! String)
    }
    func floats(_ name: String) -> [Float] {
        let data = try! Data(contentsOf: file(name))
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }
    func ints(_ name: String) -> [Int32] {
        let data = try! Data(contentsOf: file(name))
        return data.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }
    }
    func bools(_ name: String) -> [Bool] {
        let data = try! Data(contentsOf: file(name))
        return data.map { $0 != 0 }
    }
}

@Test(.enabled(if: ProcessInfo.processInfo.environment["LINGBOT_BUNDLE"] != nil
                 && ProcessInfo.processInfo.environment["LINGBOT_FIXTURE"] != nil))
func actionAdapterMatchesPythonReference() async throws {
    let env = ProcessInfo.processInfo.environment
    let bundle = URL(fileURLWithPath: env["LINGBOT_BUNDLE"]!)
    let fx = Fixture(env["LINGBOT_FIXTURE"]!)!

    let adapter = try await ActionAdapter(modelID: "lingbot-vla-v2", bundleAt: bundle)

    let chunk = try await adapter.sampleActionChunk(
        state: fx.floats("state"),
        noisyInit: fx.floats("noisy_actions_init"),
        positionIds: fx.ints("position_ids"),
        prefixK: fx.floats("prefix_k"),
        prefixV: fx.floats("prefix_v"),
        attnMask: fx.bools("attn_mask"))

    let ref = fx.floats("ref_chunk")
    #expect(chunk.count == ref.count)

    // cosine similarity + max abs diff between Swift and Python reference chunks.
    var dot = 0.0, na = 0.0, nb = 0.0, maxAbs = 0.0
    for i in 0..<min(chunk.count, ref.count) {
        let a = Double(chunk[i]), b = Double(ref[i])
        dot += a * b; na += a * a; nb += b * b
        maxAbs = max(maxAbs, abs(a - b))
    }
    let cos = dot / (na.squareRoot() * nb.squareRoot() + 1e-12)
    print("ActionAdapter parity: cosine=\(cos) maxAbsDiff=\(maxAbs) n=\(chunk.count)")
    #expect(cos > 0.999, "Swift action chunk must match the Python reference (cosine \(cos))")
    #expect(maxAbs < 0.05, "max abs diff \(maxAbs) too large")
}
