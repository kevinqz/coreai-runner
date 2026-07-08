// ModelCache.swift — actor-isolated model manager with LRU eviction.
//
// Thread-safe model lifecycle:
//   NOT_INSTALLED → DOWNLOADING → INSTALLED → LOADING → READY → (evict) → INSTALLED
//
// LRU eviction: when memory pressure is high (thermal state or available RAM
// threshold), the least-recently-used model is unloaded. Models marked as
// "sticky" are never evicted (workflow-critical models).
//
// Concurrent access: the actor serializes load/unload operations. Multiple
// /v1/predict calls on the SAME model run concurrently (the adapter is
// Sendable). Two DIFFERENT models loading simultaneously is safe — the
// Foundation Models framework supports independent InferenceFunctions.

import Foundation
import CoreAIKit
import CoreAIKitVision

public actor ModelCache {

    public static let shared = ModelCache()

    /// One entry per loaded model.
    private struct Entry {
        let adapter: any ModelAdapter
        var lastUsed: Date
        var sticky: Bool
    }

    private var entries: [String: Entry] = [:]
    private var downloadTasks: [String: Task<any ModelAdapter, Error>] = [:]
    private let store: ModelStore
    private let downloader: ParallelModelDownloader
    private let catalog: CatalogClient

    // Eviction thresholds
    private let minAvailableGB: Double = 2.0  // evict when less than 2GB free
    private let maxThermalLevel: Int = 1      // evict at .fair or worse

    public init(store: ModelStore = .default, catalog: CatalogClient = .shared) {
        self.store = store
        // Parallel downloader shares the store's directory so bundles land at the exact cache
        // path ModelStore expects (<directory>/<repo>/<rev>/<path>/). A store.download() after the
        // parallel download finds the bundle present and returns it immediately (fast-path).
        self.downloader = ParallelModelDownloader(directory: store.directory)
        self.catalog = catalog
    }

    // MARK: - Get (predict path)

    /// Returns the adapter for a loaded model, or throws if not loaded.
    public func getAdapter(_ modelID: String) throws -> any ModelAdapter {
        guard let entry = entries[modelID] else {
            throw CoreAIRunnerError.modelNotInstalled(modelID: modelID)
        }
        return entry.adapter
    }

    /// Returns true if the model is currently loaded in memory.
    public func isLoaded(_ modelID: String) -> Bool {
        entries[modelID] != nil
    }

    // MARK: - Load

    /// Downloads (if needed) and loads a model. Returns the adapter.
    /// Concurrent calls for the same model join the in-flight download.
    public func load(
        _ modelID: String,
        catalog: CatalogClient = .shared,
        sticky: Bool = false,
        progress: (@Sendable (Double) -> Void)? = nil
    ) async throws -> any ModelAdapter {

        // Already loaded?
        if var entry = entries[modelID] {
            entry.lastUsed = Date()
            entries[modelID] = entry
            return entry.adapter
        }

        // Already downloading?
        if let task = downloadTasks[modelID] {
            return try await task.value
        }

        // Start loading
        let task = Task<(any ModelAdapter), Error> { [weak self] in
            guard let self else { throw CoreAIRunnerError.modelLoadFailed(
                modelID: modelID, detail: "cache deallocated") }
            guard let entry = await catalog.getModel(id: modelID) else {
                throw CoreAIRunnerError.modelLoadFailed(
                    modelID: modelID, detail: "model not found in the catalog")
            }
            // Wire progress callback to track download status
            let adapter = try await self.createAdapter(
                for: entry, modelID: modelID,
                progress: { fraction in
                    progress?(fraction)
                    Task { await self.updateDownloadProgress(modelID, DownloadStatus(
                        modelID: modelID,
                        fraction: fraction,
                        state: fraction >= 1.0 ? .completed : .downloading
                    ))}
                }
            )
            await self.setLoaded(modelID: modelID, adapter: adapter, sticky: sticky)
            await self.clearDownloadProgress(modelID)
            return adapter
        }
        downloadTasks[modelID] = task
        defer { downloadTasks[modelID] = nil }
        return try await task.value
    }

    private func setLoaded(modelID: String, adapter: any ModelAdapter, sticky: Bool) {
        entries[modelID] = Entry(
            adapter: adapter,
            lastUsed: Date(),
            sticky: sticky
        )
        // Check memory after loading
        Task { await self.evictIfNeeded() }
    }

    // MARK: - Unload

    public func unload(_ modelID: String) async {
        entries[modelID] = nil
    }

    public func unloadAll() async {
        entries.removeAll()
    }

    // MARK: - LRU eviction

    /// Evicts the least-recently-used non-sticky model if memory/thermal is tight.
    private func evictIfNeeded() async {
        let info = DeviceInfo.current()
        let thermalLevel = ProcessInfo.processInfo.thermalState.rawValue

        guard info.memoryAvailableGB < minAvailableGB || thermalLevel > maxThermalLevel else {
            return  // no pressure
        }

        // Find LRU non-sticky
        let candidates = entries
            .filter { !$0.value.sticky }
            .sorted { $0.value.lastUsed < $1.value.lastUsed }

        if let victim = candidates.first {
            entries[victim.key] = nil
        }
    }

    /// Mark a model as sticky (never evict during this session).
    public func setSticky(_ modelID: String, sticky: Bool) {
        entries[modelID]?.sticky = sticky
    }

    // MARK: - List loaded

    public func loadedModelIDs() -> [String] {
        Array(entries.keys)
    }

    // MARK: - Installed status (downloaded on disk, not necessarily loaded)

    /// Returns catalog IDs for models whose bundles exist on disk.
    /// Maps by checking each catalog entry's HF repo against the ModelStore.
    public func installedModelIDs() async -> Set<String> {
        let entries = await catalog.listModels()
        guard !entries.isEmpty else { return [] }

        var installed: Set<String> = []
        for entry in entries {
            if let hf = entry.provenance?.huggingface {
                let repo = "\(hf.owner)/\(hf.repo)"
                let revision = hf.revision ?? "main"
                let modelID = ModelID(repo, path: nil, revision: revision)
                if store.localURL(for: modelID) != nil {
                    installed.insert(entry.id)
                }
            }
        }
        return installed
    }

    // MARK: - Download progress tracking

    private var downloadProgress: [String: DownloadStatus] = [:]

    public func updateDownloadProgress(_ modelID: String, _ status: DownloadStatus) {
        downloadProgress[modelID] = status
    }

    public func getDownloadStatus(_ modelID: String) -> DownloadStatus? {
        downloadProgress[modelID]
    }

    public func clearDownloadProgress(_ modelID: String) {
        downloadProgress.removeValue(forKey: modelID)
    }

    // MARK: - Adapter creation

    private func createAdapter(
        for entry: CatalogModelEntry,
        modelID: String,
        progress: (@Sendable (Double) -> Void)?
    ) async throws -> any ModelAdapter {
        // Determine model type from capabilities
        let capabilities = entry.capabilities

        if capabilities.contains("chat") || capabilities.contains("text-generation") {
            // LLM via CoreAILanguageModels EngineFactory.
            // Parallel-chunked download (8 connections, 16 MB chunks, cross-launch resume) — the
            // multi-GB main.mlirb payload saturates a single HF/CDN stream, so fanning out is
            // several times faster than ModelStore's sequential loop. The parallel downloader
            // pre-populates the ModelStore cache path; store.download() then returns it (fast-path).
            let modelID_hf = CoreAIKit.ModelID(
                entry.provenance?.huggingface?.owner ?? "", path: nil)
            try await downloader.download(
                modelID_hf, progress: { p in progress?(p) })
            let url = try await store.download(modelID_hf, progress: nil)
            return try await ChatAdapter(modelID: modelID, bundleDir: url.path)
        }

        if capabilities.contains("monocular-depth") {
            return try await DepthAdapter(catalog: modelID, store: store, downloadProgress: { p in
                progress?(p.fraction)
            })
        }

        if capabilities.contains("object-detection") || capabilities.contains("instance-segmentation") {
            return try await DetectionAdapter(catalog: modelID, store: store, downloadProgress: { p in
                progress?(p.fraction)
            })
        }

        if capabilities.contains("vision-language") {
            return try await VLMAdapter(catalog: modelID, store: store, downloadProgress: { p in
                progress?(p.fraction)
            })
        }

        if capabilities.contains("promptable-segmentation") {
            #if canImport(CoreAIImageSegmenter)
            // SAM 3 via CoreAIImageSegmenter (system framework, text-prompt).
            // The segmenter bundle needs a directory with metadata.json + .aimodel + tokenizer/.
            // Parallel-chunked download pre-populates the cache; store.download() returns fast-path.
            let modelID_hf = CoreAIKit.ModelID(
                entry.provenance?.huggingface?.owner ?? "", path: nil)
            try await downloader.download(
                modelID_hf, progress: { p in progress?(p) })
            let url = try await store.download(modelID_hf, progress: nil)
            return try await SegmenterAdapter(modelID: modelID, bundleDir: url.path)
            #else
            throw CoreAIRunnerError.modelLoadFailed(
                modelID: modelID,
                detail: "promptable-segmentation (SAM 3) needs the CoreAIImageSegmenter framework, absent in this SDK (the macOS 27 beta ships CoreAI core only). Build against an SDK that includes it.")
            #endif
        }

        if capabilities.contains("image-generation") {
            #if canImport(CoreAIDiffusionPipeline)
            // FLUX.2 / Z-Image via CoreAIDiffusionPipeline (system framework).
            // Multi-component bundle (TextEncoder + Transformer + VAE + tokenizer).
            // PipelineDescriptor auto-detects from metadata.json.
            // Parallel-chunked download pre-populates the cache; store.download() returns fast-path.
            let modelID_hf = CoreAIKit.ModelID(
                entry.provenance?.huggingface?.owner ?? "", path: nil)
            try await downloader.download(
                modelID_hf, progress: { p in progress?(p) })
            let url = try await store.download(modelID_hf, progress: nil)
            return try await DiffusionAdapter(modelID: modelID, bundleDir: url)
            #else
            throw CoreAIRunnerError.modelLoadFailed(
                modelID: modelID,
                detail: "image-generation (FLUX.2) needs the CoreAIDiffusionPipeline framework, absent in this SDK (the macOS 27 beta ships CoreAI core only). Build against an SDK that includes it.")
            #endif
        }

        throw CoreAIRunnerError.modelLoadFailed(
            modelID: modelID,
            detail: "Unsupported capability set: \(capabilities). Supported: monocular-depth, object-detection, instance-segmentation, vision-language, promptable-segmentation, image-generation."
        )
    }
}
