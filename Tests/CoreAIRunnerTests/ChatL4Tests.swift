// ChatL4Tests.swift — on-device conformance (L4) via the LLM path: the real Swift runner
// (ChatAdapter → CoreAILanguageModels engine → CoreAI) loads a REAL `.aimodel` chat bundle
// and generates text on real Apple hardware. Opt-in: runs only when QWEN_BUNDLE points at a
// local LanguageBundle dir (metadata.json kind=llm + .aimodel + tokenizer/). Normal CI lacks
// both the macOS 27 SDK and the bundle, so it skips.
//
// Reproduce:
//   QWEN_BUNDLE=/path/to/int8 \
//   DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer swift test --filter chat

import Foundation
import Testing
import CoreAIRunner

private let _chatL4Enabled = ProcessInfo.processInfo.environment["QWEN_BUNDLE"] != nil

@Test(.enabled(if: _chatL4Enabled))
func chatExecutesRealAimodelOnDevice() async throws {
    let bundleDir = ProcessInfo.processInfo.environment["QWEN_BUNDLE"]!

    // Real Swift runner adapter loads a REAL .aimodel LLM bundle from disk (no catalog/network).
    let adapter = ChatAdapter(modelID: "qwen2.5-0.5b-instruct-int8", bundleDir: bundleDir)

    // Real CoreAI inference on real hardware — greedy for determinism.
    let out = try await adapter.predict(
        AdapterInput(
            modelID: "qwen2.5-0.5b-instruct-int8",
            prompt: "What is the capital of France? Reply with only the city name.",
            maxTokens: 24,
            temperature: 0.0))

    let text = out.text ?? ""
    print("L4 chat generated: \(text.debugDescription)")

    // The model actually ran and produced tokens — that is the L4 claim.
    #expect(!text.isEmpty)
    // Sanity: a coherent Qwen2.5-0.5B answers this correctly.
    #expect(text.lowercased().contains("paris"))
}
