// RunnerServer.swift — Hummingbird 2.x HTTP server over a Unix domain socket.
//
// This is the runtime surface: ComfyUI-CoreAI's bridge.py connects here,
// sends POST /v1/predict, and receives results. The server manages model
// lifecycle via ModelCache and delegates inference to adapters.
//
// The server is designed to be embedded:
//   - In coreai-runner CLI: Unix socket at /tmp/coreai-runner.sock
//   - In coreai-server (future): TCP HTTP on 0.0.0.0:PORT + Bonjour
//
// Hummingbird 2.x notes:
//   - Route handlers return `Response` directly (typed explicitly to unify
//     branches that produce different ResponseGenerator types).
//   - Use `EditedResponse(status:headers:response:)` to customise status/headers.
//   - `RouterRequestContext` requires both `coreContext` and `routerContext`.
//   - `HTTPResponseStatus` → `HTTPResponse.Status` (from HTTPTypes).

import Foundation
import Hummingbird
import HummingbirdCore
import HummingbirdRouter
import HTTPTypes
import Logging

public struct RunnerServer: Sendable {
    public let socketPath: String
    public let logger: Logger

    private let cache: ModelCache
    private let catalog: CatalogClient

    public init(
        socketPath: String = "/tmp/coreai-runner.sock",
        cache: ModelCache = .shared,
        catalog: CatalogClient = .shared,
        logger: Logger? = nil
    ) {
        self.socketPath = socketPath
        self.cache = cache
        self.catalog = catalog
        self.logger = logger ?? Logger(label: "coreai-runner")
    }

    /// Build and start the HTTP server. Blocks until the server stops.
    public func run() async throws {
        // Clean up stale socket file
        if FileManager.default.fileExists(atPath: socketPath) {
            try? FileManager.default.removeItem(atPath: socketPath)
        }

        let router = buildRouter()

        var app = Application(
            router: router,
            configuration: .init(
                address: .unixDomainSocket(path: socketPath),
                serverName: "coreai-runner"
            ),
            logger: logger
        )

        // Write a ready-signal file so the spawning process knows we're up.
        // This is how bridge.py knows the socket is listening without polling.
        try? writeReadyFile()

        app.beforeServerStarts { [socketPath = self.socketPath, logger = self.logger] in
            logger.info("coreai-runner listening on \(socketPath)")
        }

        // Graceful shutdown is handled by ServiceGroup (SIGTERM/SIGINT).
        // Hummingbird 2.x removed afterServerShutdown; the socket + ready
        // files are cleaned up on next start (stale-socket guard above).

        try await app.runService()
    }

    // MARK: - Router

    private func buildRouter() -> Router<CoreAIRunnerContext> {
        let router = Router(context: CoreAIRunnerContext.self)

        // GET /v1/health — device info, loaded models, thermal state
        router.get("v1/health") { _, _ in
            let info = DeviceInfo.current()
            let loaded = await self.cache.loadedModelIDs()

            return try EditedResponse(
                status: .ok,
                headers: [.contentType: "application/json"],
                response: HealthResponse(
                    status: "healthy",
                    device: info.deviceName,
                    chip: info.chipName,
                    memoryTotalGB: info.memoryTotalGB,
                    memoryAvailableGB: info.memoryAvailableGB,
                    macosVersion: info.macosVersion,
                    coreaiVersion: info.coreaiVersion,
                    loadedModels: loaded,
                    thermalState: info.thermalState
                )
            )
        }

        // GET /v1/capabilities — runtime feature advertisement (spec §19.2).
        // Lets clients (lerobot-coreai, ComfyUI) discover what the runner supports.
        router.get("v1/capabilities") { request, context in
            try EditedResponse(
                status: .ok,
                headers: [.contentType: "application/json"],
                response: CapabilitiesResponse()
            ).response(from: request, context: context)
        }

        // GET /v1/models — list models (optionally filtered by capability)
        router.get("v1/models") { request, _ in
            let query = request.uri.query
            let capability = self.extractQueryParam(query, name: "capability")

            let entries = await self.catalog.listModels(capability: capability)
            let loadedIDs = await self.cache.loadedModelIDs()

            let installedIDs = await self.cache.installedModelIDs()

            let modelEntries: [ModelListResponse.ModelEntry] = entries.map { entry in
                ModelListResponse.ModelEntry(
                    id: entry.id,
                    name: entry.name,
                    family: entry.family ?? "",
                    capability: entry.capabilities.first ?? "",
                    sizeMB: entry.size?.sizeInMB,
                    precision: entry.size?.precision,
                    license: entry.license?.name,
                    commercialUse: entry.license?.commercialUse,
                    deviceSupport: self.deviceSupportArray(entry.deviceSupport),
                    installed: installedIDs.contains(entry.id),
                    loaded: loadedIDs.contains(entry.id),
                    benchmarkMs: nil,
                    runner: entry.runtime?.runner
                )
            }

            return try EditedResponse(
                status: .ok,
                headers: [.contentType: "application/json"],
                response: ModelListResponse(models: modelEntries)
            )
        }

        // POST /v1/models/:model_id/load — download + load model
        router.post("v1/models/:model_id/load") { request, context -> Response in
            guard let modelID = try? context.parameters.require("model_id", as: String.self) else {
                return try EditedResponse(
                    status: .badRequest,
                    headers: [.contentType: "application/json"],
                    response: ErrorResponse(code: "INVALID_INPUT", message: "model_id is required")
                ).response(from: request, context: context)
            }

            do {
                _ = try await self.cache.load(modelID)
                return try EditedResponse(
                    status: .ok,
                    headers: [.contentType: "application/json"],
                    response: LoadResponse(modelID: modelID, status: "loaded", sizeMB: nil)
                ).response(from: request, context: context)
            } catch let error as CoreAIRunnerError {
                let (status, code) = self.mapError(error)
                return try EditedResponse(
                    status: status,
                    headers: [.contentType: "application/json"],
                    response: ErrorResponse(code: code, message: error.localizedDescription, modelID: modelID)
                ).response(from: request, context: context)
            } catch {
                return try EditedResponse(
                    status: .internalServerError,
                    headers: [.contentType: "application/json"],
                    response: ErrorResponse(code: "MODEL_LOAD_FAILED", message: "\(error)", modelID: modelID)
                ).response(from: request, context: context)
            }
        }

        // POST /v1/models/:model_id/unload — release model from memory
        router.post("v1/models/:model_id/unload") { request, context -> Response in
            guard let modelID = try? context.parameters.require("model_id", as: String.self) else {
                return try EditedResponse(
                    status: .badRequest,
                    headers: [.contentType: "application/json"],
                    response: ErrorResponse(code: "INVALID_INPUT", message: "model_id is required")
                ).response(from: request, context: context)
            }
            await self.cache.unload(modelID)
            return try EditedResponse(
                status: .ok,
                headers: [.contentType: "application/json"],
                response: LoadResponse(modelID: modelID, status: "unloaded", sizeMB: nil)
            ).response(from: request, context: context)
        }

        // GET /v1/models/:model_id/status — installed, loaded, download progress
        router.get("v1/models/:model_id/status") { request, context -> Response in
            guard let modelID = try? context.parameters.require("model_id", as: String.self) else {
                return try EditedResponse(
                    status: .badRequest,
                    headers: [.contentType: "application/json"],
                    response: ErrorResponse(code: "INVALID_INPUT", message: "model_id is required")
                ).response(from: request, context: context)
            }
            let loaded = await self.cache.isLoaded(modelID)
            let download = await self.cache.getDownloadStatus(modelID)
            let installedIDs = await self.cache.installedModelIDs()
            let compilation = await self.cache.bundleCompilation(for: modelID)
            return try EditedResponse(
                status: .ok,
                headers: [.contentType: "application/json"],
                response: ModelStatusResponse(
                    modelID: modelID,
                    installed: installedIDs.contains(modelID),
                    loaded: loaded,
                    download: download,
                    compilation: compilation
                )
            ).response(from: request, context: context)
        }

        // POST /v1/chat/completions — OpenAI-compatible chat (non-streaming)
        router.post("v1/chat/completions") { request, context -> Response in
            let chatRequest: ChatCompletionRequest
            do {
                chatRequest = try await request.decode(as: ChatCompletionRequest.self, context: context)
            } catch {
                return try EditedResponse(
                    status: .badRequest,
                    headers: [.contentType: "application/json"],
                    response: ErrorResponse(
                        code: "INVALID_INPUT", message: "Cannot decode request body: \(error)"
                    )
                ).response(from: request, context: context)
            }

            let modelID = chatRequest.model

            // Load model (lazy)
            let adapter: any ModelAdapter
            do {
                adapter = try await self.cache.getAdapter(modelID)
            } catch {
                do {
                    adapter = try await self.cache.load(modelID)
                } catch let loadError as CoreAIRunnerError {
                    let (status, code) = self.mapError(loadError)
                    return try EditedResponse(
                        status: status,
                        headers: [.contentType: "application/json"],
                        response: ErrorResponse(
                            code: code, message: loadError.localizedDescription, modelID: modelID
                        )
                    ).response(from: request, context: context)
                } catch {
                    return try EditedResponse(
                        status: .internalServerError,
                        headers: [.contentType: "application/json"],
                        response: ErrorResponse(
                            code: "MODEL_LOAD_FAILED", message: "\(error)", modelID: modelID
                        )
                    ).response(from: request, context: context)
                }
            }

            guard let chatAdapter = adapter as? ChatAdapter else {
                return try EditedResponse(
                    status: .badRequest,
                    headers: [.contentType: "application/json"],
                    response: ErrorResponse(
                        code: "NOT_A_CHAT_MODEL",
                        message: "Model '\(modelID)' does not support chat. Use POST /v1/predict instead.",
                        modelID: modelID
                    )
                ).response(from: request, context: context)
            }

            // Build messages array from request
            let messages = chatRequest.messages.map { msg in
                ["role": msg.role, "content": msg.content]
            }
            let maxTokens = chatRequest.maxTokens ?? 256
            let temp = chatRequest.temperature ?? 0.7

            do {
                let output = try await chatAdapter.predict(AdapterInput(
                    modelID: modelID,
                    prompt: chatRequest.messages.last?.content,
                    maxTokens: maxTokens,
                    temperature: temp
                ))

                return try EditedResponse(
                    status: .ok,
                    headers: [.contentType: "application/json"],
                    response: ChatCompletionResponse(
                        id: "chatcmn-\(UUID().uuidString.prefix(8))",
                        model: modelID,
                        choices: [ChatCompletionResponse.Choice(
                            index: 0,
                            message: ChatCompletionResponse.Message(
                                role: "assistant",
                                content: output.text ?? ""
                            ),
                            finishReason: "stop"
                        )],
                        usage: ChatCompletionResponse.Usage(
                            promptTokens: 0,
                            completionTokens: 0,
                            totalTokens: 0
                        )
                    )
                ).response(from: request, context: context)
            } catch let error as CoreAIRunnerError {
                let (status, code) = self.mapError(error)
                return try EditedResponse(
                    status: status,
                    headers: [.contentType: "application/json"],
                    response: ErrorResponse(
                        code: code, message: error.localizedDescription, modelID: modelID
                    )
                ).response(from: request, context: context)
            } catch {
                return try EditedResponse(
                    status: .internalServerError,
                    headers: [.contentType: "application/json"],
                    response: ErrorResponse(
                        code: "INFERENCE_FAILED", message: "\(error)", modelID: modelID
                    )
                ).response(from: request, context: context)
            }
        }

        // POST /v1/chat/stream — SSE streaming chat (OpenAI-compatible)
        router.post("v1/chat/stream") { request, context -> Response in
            let chatRequest: ChatCompletionRequest
            do {
                chatRequest = try await request.decode(as: ChatCompletionRequest.self, context: context)
            } catch {
                return try EditedResponse(
                    status: .badRequest,
                    headers: [.contentType: "application/json"],
                    response: ErrorResponse(
                        code: "INVALID_INPUT", message: "Cannot decode request body: \(error)"
                    )
                ).response(from: request, context: context)
            }

            let modelID = chatRequest.model

            // Load model (lazy)
            let adapter: any ModelAdapter
            do {
                adapter = try await self.cache.getAdapter(modelID)
            } catch {
                do {
                    adapter = try await self.cache.load(modelID)
                } catch let loadError as CoreAIRunnerError {
                    let (status, code) = self.mapError(loadError)
                    return try EditedResponse(
                        status: status,
                        headers: [.contentType: "application/json"],
                        response: ErrorResponse(
                            code: code, message: loadError.localizedDescription, modelID: modelID
                        )
                    ).response(from: request, context: context)
                } catch {
                    return try EditedResponse(
                        status: .internalServerError,
                        headers: [.contentType: "application/json"],
                        response: ErrorResponse(
                            code: "MODEL_LOAD_FAILED", message: "\(error)", modelID: modelID
                        )
                    ).response(from: request, context: context)
                }
            }

            guard let chatAdapter = adapter as? ChatAdapter else {
                return try EditedResponse(
                    status: .badRequest,
                    headers: [.contentType: "application/json"],
                    response: ErrorResponse(
                        code: "NOT_A_CHAT_MODEL",
                        message: "Model '\(modelID)' does not support chat streaming.",
                        modelID: modelID
                    )
                ).response(from: request, context: context)
            }

            let messages = chatRequest.messages.map { msg in
                ["role": msg.role, "content": msg.content]
            }
            let maxTokens = chatRequest.maxTokens ?? 256
            let temp = chatRequest.temperature ?? 0.7

            // Build SSE stream
            let eventStream = try await chatAdapter.streamChat(
                messages: messages,
                maxTokens: maxTokens,
                temperature: temp
            )

            // Convert ChatStreamEvent → SSE-formatted ByteBuffer chunks
            let sseStream = AsyncThrowingStream<ByteBuffer, Error> { continuation in
                Task {
                    do {
                        for try await event in eventStream {
                            let sseData: Data
                            switch event {
                            case .token(let piece):
                                let payload = ChatStreamChunk(choices: [ChatStreamChunk.Choice(
                                    index: 0,
                                    delta: ChatStreamChunk.Delta(content: piece),
                                    finishReason: nil
                                )])
                                sseData = try JSONEncoder().encode(payload)
                            case .stats(let stats):
                                let payload = ChatStreamChunk(
                                    choices: [],
                                    stats: ChatStreamChunk.StreamStats(
                                        promptTokens: stats.promptTokens,
                                        reusedPromptTokens: stats.reusedPromptTokens
                                    )
                                )
                                sseData = try JSONEncoder().encode(payload)
                            case .done(let stats):
                                let payload = ChatStreamChunk(
                                    choices: [ChatStreamChunk.Choice(
                                        index: 0,
                                        delta: ChatStreamChunk.Delta(content: ""),
                                        finishReason: "stop"
                                    )],
                                    stats: ChatStreamChunk.StreamStats(
                                        promptTokens: stats.promptTokens,
                                        reusedPromptTokens: stats.reusedPromptTokens,
                                        generatedTokens: stats.generatedTokens,
                                        tokensPerSecond: stats.tokensPerSecond
                                    )
                                )
                                sseData = try JSONEncoder().encode(payload)
                            case .error(let message):
                                let payload = ChatStreamChunk(error: message)
                                sseData = try JSONEncoder().encode(payload)
                            }

                            var buffer = ByteBuffer()
                            buffer.writeString("data: ")
                            buffer.writeBytes(sseData)
                            buffer.writeString("\n\n")

                            continuation.yield(buffer)

                            if case .done = event {
                                var doneBuffer = ByteBuffer()
                                doneBuffer.writeString("data: [DONE]\n\n")
                                continuation.yield(doneBuffer)
                                continuation.finish()
                                return
                            }
                            if case .error = event {
                                continuation.finish(throwing: ChatStreamError.streamError)
                                return
                            }
                        }
                        continuation.finish()
                    } catch {
                        continuation.finish(throwing: error)
                    }
                }
            }

            let body = ResponseBody(asyncSequence: sseStream)
            return Response(
                status: .ok,
                headers: [
                    .contentType: "text/event-stream",
                    .cacheControl: "no-cache",
                    .connection: "keep-alive",
                ],
                body: body
            )
        }

        // POST /v1/predict — run inference
        router.post("v1/predict") { request, context -> Response in
            let predictRequest: PredictRequest
            do {
                predictRequest = try await request.decode(as: PredictRequest.self, context: context)
            } catch {
                return try EditedResponse(
                    status: .badRequest,
                    headers: [.contentType: "application/json"],
                    response: ErrorResponse(
                        code: "INVALID_INPUT", message: "Cannot decode request body: \(error)"
                    )
                ).response(from: request, context: context)
            }

            let modelID = predictRequest.modelID

            // Action policy path (spec §19.1): runtime_kind=action means robot policy inference.
            // The request carries an observation (images/state/task) and expects an action chunk.
            // NOTE: Action inference requires an ActionAdapter + host-loop sampler, which is the
            // lerobot-coreai v0.2 scope. Until then, we return a clear 501 so clients can detect
            // support without crashing.
            if predictRequest.runtimeKind == "action" {
                return try EditedResponse(
                    status: .notImplemented,
                    headers: [.contentType: "application/json"],
                    response: ErrorResponse(
                        code: "ACTION_RUNTIME_NOT_AVAILABLE",
                        message: "runtime_kind=action requires the action adapter (lerobot-coreai v0.2). "
                               + "The runner can load action artifacts and report their graph/host_loop "
                               + "structure, but host-driven action sampling is not yet implemented.",
                        modelID: modelID
                    )
                ).response(from: request, context: context)
            }

            let input = AdapterInput(
                modelID: modelID,
                imagePath: predictRequest.input.imagePath,
                prompt: predictRequest.input.prompt,
                maxTokens: predictRequest.input.maxTokens,
                temperature: predictRequest.input.temperature,
                scoreThreshold: predictRequest.input.scoreThreshold,
                points: predictRequest.input.points,
                boxes: predictRequest.input.boxes,
                textPrompt: predictRequest.input.textPrompt,
                computeUnit: predictRequest.options?.computeUnit ?? .auto
            )

            // Get adapter (loads model if needed via lazy load)
            let adapter: any ModelAdapter
            do {
                adapter = try await self.cache.getAdapter(modelID)
            } catch let error as CoreAIRunnerError {
                // Try lazy-load on modelNotInstalled
                if case .modelNotInstalled = error {
                    do {
                        adapter = try await self.cache.load(modelID)
                    } catch let loadError as CoreAIRunnerError {
                        let (status, code) = self.mapError(loadError)
                        return try EditedResponse(
                            status: status,
                            headers: [.contentType: "application/json"],
                            response: ErrorResponse(
                                code: code, message: loadError.localizedDescription, modelID: modelID
                            )
                        ).response(from: request, context: context)
                    } catch {
                        return try EditedResponse(
                            status: .internalServerError,
                            headers: [.contentType: "application/json"],
                            response: ErrorResponse(
                                code: "MODEL_LOAD_FAILED", message: "\(error)", modelID: modelID
                            )
                        ).response(from: request, context: context)
                    }
                } else {
                    let (status, code) = self.mapError(error)
                    return try EditedResponse(
                        status: status,
                        headers: [.contentType: "application/json"],
                        response: ErrorResponse(
                            code: code, message: error.localizedDescription, modelID: modelID
                        )
                    ).response(from: request, context: context)
                }
            }

            // Run inference
            do {
                let output = try await adapter.predict(input)

                return try EditedResponse(
                    status: .ok,
                    headers: [.contentType: "application/json"],
                    response: PredictResponse(
                        modelID: modelID,
                        output: PredictResponse.PredictOutput(
                            kind: output.kind.rawValue,
                            outputPath: output.outputPath,
                            text: output.text,
                            detections: output.detections,
                            maskPaths: output.maskPaths
                        ),
                        timing: output.timing
                    )
                ).response(from: request, context: context)
            } catch let error as CoreAIRunnerError {
                let (status, code) = self.mapError(error)
                return try EditedResponse(
                    status: status,
                    headers: [.contentType: "application/json"],
                    response: ErrorResponse(
                        code: code, message: error.localizedDescription, modelID: modelID
                    )
                ).response(from: request, context: context)
            } catch {
                return try EditedResponse(
                    status: .internalServerError,
                    headers: [.contentType: "application/json"],
                    response: ErrorResponse(
                        code: "INFERENCE_FAILED", message: "\(error)", modelID: modelID
                    )
                ).response(from: request, context: context)
            }
        }

        return router
    }

    // MARK: - Helpers

    private func writeReadyFile() throws {
        let readyPath = socketPath + ".ready"
        // Write the process PID so the Python bridge can verify the process
        // that created the .ready file is still alive (stale file detection).
        let pid = ProcessInfo.processInfo.processIdentifier
        try "\(pid)".write(toFile: readyPath, atomically: true, encoding: .utf8)
    }

    private func extractQueryParam(_ query: String?, name: String) -> String? {
        guard let query else { return nil }
        let pairs = query.split(separator: "&")
        for pair in pairs {
            let kv = pair.split(separator: "=", maxSplits: 1)
            if kv.count == 2 && kv[0] == name {
                return String(kv[1])
            }
        }
        return nil
    }

    private func deviceSupportArray(_ support: CatalogModelEntry.DeviceSupport?) -> [String] {
        guard let support else { return [] }
        var devices: [String] = []
        if support.iphone == true { devices.append("iphone") }
        if support.ipad == true { devices.append("ipad") }
        if support.mac == true { devices.append("mac") }
        return devices
    }

    private func mapError(_ error: CoreAIRunnerError) -> (HTTPResponse.Status, String) {
        switch error {
        case .modelNotInstalled: return (.notFound, "MODEL_NOT_INSTALLED")
        case .modelLoadFailed: return (.internalServerError, "MODEL_LOAD_FAILED")
        case .inferenceFailed: return (.internalServerError, "INFERENCE_FAILED")
        case .unsupportedDevice: return (.forbidden, "UNSUPPORTED_DEVICE")
        case .patchRequired: return (.internalServerError, "PATCH_REQUIRED")
        case .memoryInsufficient: return (.internalServerError, "MEMORY_INSUFFICIENT")
        case .timeout: return (.requestTimeout, "TIMEOUT")
        case .invalidInput, .missingInput: return (.badRequest, "INVALID_INPUT")
        case .catalogError: return (.serviceUnavailable, "CATALOG_ERROR")
        }
    }
}

/// Hummingbird 2.x request context.
/// RouterRequestContext requires both `coreContext` and `routerContext`.
struct CoreAIRunnerContext: RouterRequestContext {
    var coreContext: CoreRequestContextStorage
    var routerContext: RouterBuilderContext

    init(source: Source) {
        self.coreContext = .init(source: source)
        self.routerContext = .init()
    }
}
