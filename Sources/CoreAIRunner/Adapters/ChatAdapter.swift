// ChatAdapter.swift — LLM inference adapter using Apple's CoreAILanguageModels engine.
//
// This adapter bridges the ModelAdapter protocol to the EngineFactory +
// InferenceEngine pipeline from john-rocky/coreai-models. It supports:
//
//   - Automatic engine selection (auto-detect from model structure)
//   - Cross-turn prefix cache (KV reuse) — up to 101× TTFT speedup at 4k ctx
//   - Streaming token output via AsyncStream
//   - Incremental detokenization for O(n) display
//   - Chat template application via swift-transformers
//
// The adapter owns the engine + tokenizer lifecycle. ModelCache handles
// download via ModelStore and calls createAdapter.

import CoreAILanguageModels
import CoreAIShared
import Foundation
import Tokenizers

public actor ChatAdapter: ModelAdapter {
    public let modelID: String

    private let bundleDir: String

    // Engine state — populated by loadEngine(), nil until then.
    private var engine: (any InferenceEngine)?
    private var tokenizer: (any Tokenizer)?
    private var kvTokens: [Int32] = []           // exact token seq held in engine KV
    // Which engine variant is active — surfaced in stats/status.
    private var activeVariant: String = "unknown"

    // Prefix cache: on by default. A/B switch: COREAI_RUNNER_NO_PREFIX_CACHE=1 forces the
    // old reset()+full re-prefill path (ported from the zoo's CHATMAC_NO_PREFIX_CACHE).
    private let prefixCacheEnabled =
        ProcessInfo.processInfo.environment["COREAI_RUNNER_NO_PREFIX_CACHE"] == nil

    // Engine variant fallback (ported from the zoo's ChatEngine.load): the factory's auto-detect
    // maps every "dynamic" structure to pipelined, which SIGTRAPs in GrowingLogitsBuffer for the
    // common sequential bundles. So we force coreai-sequential first, and fall back to
    // coreai-pipelined only when the sequential engine rejects the bundle's extra SSM states.
    private static let launchChunkThreshold =
        ProcessInfo.processInfo.environment["COREAI_CHUNK_THRESHOLD"]
    private static func restoreLaunchChunkThreshold() {
        if let v = launchChunkThreshold { setenv("COREAI_CHUNK_THRESHOLD", v, 1) }
        else { unsetenv("COREAI_CHUNK_THRESHOLD") }
    }

    public init(modelID: String, bundleDir: String) {
        self.modelID = modelID
        self.bundleDir = bundleDir
    }

    // MARK: - Load

    private func loadEngine() async throws {
        if engine != nil { return }

        let bundle = try LanguageBundle(from: bundleDir)
        let modelURL = try bundle.requireModelURL(for: ModelBundle.ComponentKey.main)

        // Build ModelConfig from the bundle's LanguageConfig
        let config = ModelConfig(
            name: bundle.name,
            tokenizer: bundle.tokenizer,
            vocabSize: bundle.vocabSize,
            maxContextLength: bundle.maxContextLength,
            source: ModelSource(hfModelId: bundle.name, modelDefinition: nil),
            serializedModel: [bundle.modelAssetPath],
            function: bundle.language.functionMap?.name(for: "main") ?? "main"
        )
        let configData = try JSONEncoder().encode(config)

        // Load tokenizer in parallel with engine creation.
        async let tokenizerResult = bundle.loadTokenizer()

        // Engine variant selection (ported from the zoo's ChatEngine.load):
        // The factory's auto-detect maps every "dynamic" structure to the GPU pipelined variant,
        // whose logits path asserts in GrowingLogitsBuffer for them (SIGTRAP on load). The
        // coreai-sequential variant drives these bundles correctly. Hybrid decode-pipelined
        // bundles (qwen3.5 family, Granite, Ornith) carry extra fixed-shape SSM states the
        // sequential engine rejects — those fall through to pipelined via the catch.
        Self.restoreLaunchChunkThreshold()

        let loadedEngine: any InferenceEngine
        let variant: String
        do {
            loadedEngine = try await EngineFactory.createEngine(
                config: configData, modelURL: modelURL,
                options: EngineOptions(variant: "coreai-sequential")
            )
            variant = "coreai-sequential"
        } catch let error as InferenceRuntimeError {
            // Hybrid decode-pipelined bundles carry extra fixed-shape SSM states the sequential
            // engine rejects. Fall back to pipelined with per-token prefill (chunk threshold 1).
            let detail = String(describing: error)
            if detail.contains("Expected 2 states") && !bundle.name.contains("gather") {
                setenv("COREAI_CHUNK_THRESHOLD", "1", 1)
                loadedEngine = try await EngineFactory.createEngine(
                    config: configData, modelURL: modelURL,
                    options: EngineOptions(variant: "coreai-pipelined")
                )
                variant = "coreai-pipelined"
            } else {
                throw error
            }
        }

        self.engine = loadedEngine
        self.activeVariant = variant
        self.tokenizer = try await tokenizerResult
    }

    // MARK: - Predict (non-streaming, returns full text)

    public func predict(_ input: AdapterInput) async throws -> AdapterOutput {
        try await loadEngine()
        guard let engine, let tokenizer else {
            throw CoreAIRunnerError.modelLoadFailed(modelID: modelID, detail: "engine not loaded")
        }

        let prompt = input.prompt ?? input.textPrompt ?? ""
        guard !prompt.isEmpty else {
            throw CoreAIRunnerError.missingInput("prompt or text_prompt")
        }

        let maxTokens = input.maxTokens ?? 256
        let temp = input.temperature ?? 0.7

        // Apply chat template (single-turn: user message)
        let history: [[String: any Sendable]] = [
            ["role": "user", "content": prompt]
        ]

        let t0 = SuspendingClock.now
        let fullTokens = try await tokenizer.applyChatTemplate(messages: history).map(Int32.init)

        // Prefix reuse: trim KV to longest common prefix
        let reused = await trimAndFeedAsync(engine: engine, fullTokens: fullTokens)

        let sampling = SamplingConfiguration(temperature: temp, topK: 40, topP: 0.9)
        let options = InferenceOptions(maxTokens: maxTokens)

        var generatedText = ""
        var genCount = 0
        var incremental = IncrementalDetokenizer()

        let stream = try engine.generate(
            with: engine.prefixReuseFeedsFullSequence ? fullTokens : Array(fullTokens[reused...]),
            samplingConfiguration: sampling,
            inferenceOptions: options
        )

        let eosID = tokenizer.eosTokenId.flatMap { Int32($0) }

        for try await output in stream {
            if let eosID, output.tokenId == eosID { break }
            let piece = incremental.feed(Int(output.tokenId), tokenizer: tokenizer)
            generatedText += piece
            genCount += 1
            kvTokens.append(output.tokenId)
        }

        generatedText += incremental.flush()

        let t1 = SuspendingClock.now
        let duration = t1 - t0
        let inferenceMs = Double(duration.components.seconds) * 1000
            + Double(duration.components.attoseconds) / 1e15

        return AdapterOutput(
            modelID: modelID,
            kind: .text,
            text: generatedText,
            timing: AdapterOutput.Timing(
                loadMs: 0, preprocessMs: 0,
                inferenceMs: inferenceMs,
                postprocessMs: 0, totalMs: inferenceMs,
                computeUnitUsed: "GPU"
            )
        )
    }

    // MARK: - Streaming

    /// Stream chat completion tokens. Called by the SSE endpoint.
    public func streamChat(
        messages: [[String: String]],
        maxTokens: Int,
        temperature: Double
    ) async throws -> AsyncStream<ChatStreamEvent> {
        try await loadEngine()
        guard let engine, let tokenizer else {
            throw CoreAIRunnerError.modelLoadFailed(modelID: modelID, detail: "engine not loaded")
        }

        // Tokenize + trim BEFORE entering the AsyncStream (actor-isolated)
        let sendableHistory = messages.map { msg -> [String: any Sendable] in
            ["role": msg["role"] ?? "user", "content": msg["content"] ?? ""]
        }
        let fullTokens = try await tokenizer.applyChatTemplate(messages: sendableHistory).map(Int32.init)
        let reused = await trimAndFeedAsync(engine: engine, fullTokens: fullTokens)
        let promptTokenCount = fullTokens.count
        let eosID = tokenizer.eosTokenId.flatMap { Int32($0) }
        let feedsFull = engine.prefixReuseFeedsFullSequence
        let inputTokens = feedsFull ? fullTokens : Array(fullTokens[reused...])

        let sampling = SamplingConfiguration(temperature: temperature, topK: 40, topP: 0.9)
        let options = InferenceOptions(maxTokens: maxTokens)

        return AsyncStream { continuation in
            Task {
                do {
                    continuation.yield(.stats(ChatStats(
                        promptTokens: promptTokenCount, reusedPromptTokens: reused
                    )))

                    let stream = try engine.generate(
                        with: inputTokens,
                        samplingConfiguration: sampling,
                        inferenceOptions: options
                    )

                    var incremental = IncrementalDetokenizer()
                    var genCount = 0
                    let t0 = SuspendingClock.now

                    for try await output in stream {
                        if let eosID, output.tokenId == eosID { break }
                        let piece = incremental.feed(Int(output.tokenId), tokenizer: tokenizer)
                        if !piece.isEmpty {
                            continuation.yield(.token(piece))
                        }
                        genCount += 1
                    }

                    let finalPiece = incremental.flush()
                    if !finalPiece.isEmpty { continuation.yield(.token(finalPiece)) }

                    let t1 = SuspendingClock.now
                    let duration = t1 - t0
                    let seconds = Double(duration.components.seconds)
                        + Double(duration.components.attoseconds) / 1e15
                    let tokPerSec = genCount > 0 && seconds > 0
                        ? Double(genCount) / seconds : 0

                    continuation.yield(.done(ChatStats(
                        promptTokens: promptTokenCount,
                        reusedPromptTokens: reused,
                        generatedTokens: genCount,
                        tokensPerSecond: tokPerSec
                    )))
                } catch {
                    continuation.yield(.error(error.localizedDescription))
                }
                continuation.finish()
            }
        }
    }

    // MARK: - Prefix reuse helpers

    private func trimAndFeedAsync(engine: any InferenceEngine, fullTokens: [Int32]) async -> Int {
        // Cap at full.count-1 so at least one token is always prefilled (the engine needs a
        // query step to produce the next-token logits). Without this cap, if the entire prompt
        // is cached from a prior turn, the engine has no query step and stalls.
        // Disabled entirely when prefixCacheEnabled is false (env toggle for A/B testing).
        let want = prefixCacheEnabled
            ? min(Self.commonPrefixLength(fullTokens, kvTokens), max(0, fullTokens.count - 1))
            : 0
        guard want > 0 else {
            try? await engine.reset()
            kvTokens = fullTokens
            return 0
        }
        let retained = await engine.trimKVCache(to: want)
        if retained >= 0 {
            kvTokens = fullTokens
            return retained
        } else {
            // Engine can't rewind (SSM hybrids: qwen3.5, RWKV-7) — full reset
            try? await engine.reset()
            kvTokens = fullTokens
            return 0
        }
    }

    private static func commonPrefixLength(_ a: [Int32], _ b: [Int32]) -> Int {
        let n = min(a.count, b.count)
        var i = 0
        while i < n, a[i] == b[i] { i += 1 }
        return i
    }

    // MARK: - Unload

    func unload() async {
        if let engine {
            try? await engine.reset()
        }
        engine = nil
        tokenizer = nil
        kvTokens = []
    }
}

// MARK: - Streaming types

public enum ChatStreamEvent: Sendable {
    case token(String)
    case stats(ChatStats)
    case done(ChatStats)
    case error(String)
}

public struct ChatStats: Sendable, Codable {
    public var promptTokens: Int = 0
    public var reusedPromptTokens: Int = 0
    public var generatedTokens: Int = 0
    public var tokensPerSecond: Double = 0
}

// MARK: - Incremental Detokenization (O(n) streaming display)
//
// Full re-decode per token is O(n²). Instead, decode only a small tail cache
// and fold into stableText at safe boundaries. A trailing U+FFFD means a UTF-8
// byte sequence is still split across tokens — keep the cache open.

struct IncrementalDetokenizer {
    private var tailTokens: [Int] = []
    private var stableText = ""

    init() {}

    mutating func feed(_ tokenId: Int, tokenizer: any Tokenizer) -> String {
        tailTokens.append(tokenId)
        guard let decoded = try? tokenizer.decode(tokens: tailTokens) else { return "" }

        // Fold at safe boundary (no trailing replacement char = no split UTF-8)
        guard !decoded.hasSuffix("\u{FFFD}") else {
            return ""  // hold — char split across tokens
        }
        let prev = stableText
        stableText = decoded
        tailTokens = []
        return String(decoded.dropFirst(prev.count))
    }

    mutating func flush() -> String {
        // stableText is already emitted incrementally via feed().
        // Nothing remains to flush in normal operation.
        return ""
    }
}
