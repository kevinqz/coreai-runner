// ParallelModelDownloader.swift — range-chunked parallel model bundle delivery.
//
// Replaces the stock CoreAIKit.ModelStore sequential download for the bundles the runner
// downloads directly (LLM, SAM, FLUX.2). Those share one multi-GB payload file (the
// `main.mlirb`), and a single HF/CDN stream is bandwidth-capped, so 8 parallel range chunks
// are several times faster than the one-stream-per-file loop ModelStore.performDownload runs.
//
// The design is ported from the zoo's AppShared/ModelDownloader.swift (proven in production),
// adapted for a headless actor (no SwiftUI ObservableObject, no main-actor isolation):
//   • Parallel: every file >chunkSize is split into byte-range chunks pulled CONCURRENTLY
//     (up to maxConnections), each written straight at its offset in the staging file
//     (no concatenation pass, disk use stays at 1×, peak RAM = chunkSize × maxConnections).
//   • Resumable across launches: a sidecar bitmap (one byte per chunk) records done chunks;
//     a chunk's bit is set only AFTER its data is written and closed (bit==1 ⟹ bytes present),
//     so a quit/crash costs at most the one in-flight chunk, never the whole bundle.
//
// Integration: this downloader pre-populates the ModelStore cache layout
// (<directory>/<repo>/<revision>/<resolvedPath>/), so a subsequent ModelStore.download()
// finds the bundle present and returns it immediately (fast-path). ModelStore stays the
// source of truth for the on-disk layout — this just fills it faster.

import Foundation
import CoreAIKit

public actor ParallelModelDownloader {

    // MARK: - Tunables (mirrors the zoo, tuned for headless server use)

    /// Number of independent `URLSession`s in the pool = independent TCP connections streaming
    /// chunks at once. A single `URLSession` multiplexes all its tasks onto ONE HTTP/2 connection
    /// per host, so `httpMaximumConnectionsPerHost` is ignored under H2 and you get ~one stream.
    /// One session per slot makes aggregate throughput scale with N. 8 is a safe default.
    private nonisolated static let maxConnections = 8

    /// Files larger than this are split into byte-range chunks of this size. 16 MiB keeps peak RAM
    /// (chunkSize × maxConnections ≈ 96 MB) modest while advancing progress smoothly.
    private nonisolated static let chunkSize: Int64 = 16 * 1024 * 1024

    /// Per-chunk retry budget. A failed chunk re-fetches only its own ≤chunkSize slice; each retry
    /// re-hits the HF resolve URL, so it re-rolls onto a possibly-different (live) CDN.
    private nonisolated static let maxChunkRetries = 6

    /// Hidden directory under the staging root holding per-file completed-chunk bitmaps.
    private nonisolated static let progressDirName = ".dl-progress"

    /// Re-applies the Range header across the HF→CDN 302 redirect. Without it, a redirect that
    /// dropped Range would answer 200 with the WHOLE file and `data(for:)` would buffer ~30 GB.
    private nonisolated static let redirector = RangePreservingRedirector()

    // MARK: - State

    /// Same root as ModelStore.directory — bundles land at <directory>/<repo>/<rev>/<path>/.
    public nonisolated let directory: URL

    /// Optional custom session configuration for testing (e.g. URLProtocol-based mock CDN).
    /// When nil, the production pool (8 independent `URLSessionConfiguration.default` sessions) is used.
    private nonisolated let testSessionConfig: URLSessionConfiguration?

    /// Single-flight guard: two overlapping downloads of the same model open duplicate connections
    /// that fight over the same byte ranges and stall at ~0 (-1005 "network connection lost").
    private var inflight: [ModelID: Task<URL, Error>] = [:]

    public init(directory: URL) {
        self.directory = directory
        self.testSessionConfig = nil
    }

    /// Rooted at Application Support/CoreAIKit/Models (same default as ModelStore).
    public init() {
        let base = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask)[0]
        self.directory = base.appendingPathComponent("CoreAIKit/Models", isDirectory: true)
        self.testSessionConfig = nil
    }

    /// Test initializer: uses a custom session configuration (e.g. with a URLProtocol mock registered).
    init(directory: URL, sessionConfig: URLSessionConfiguration) {
        self.directory = directory
        self.testSessionConfig = sessionConfig
    }

    // MARK: - Public API

    /// Downloads the bundle for `model` into the ModelStore cache layout, returning the bundle root.
    /// If the bundle is already present (complete), returns immediately. Concurrent calls for the
    /// same model join the in-flight download.
    /// `progress` reports a 0...1 fraction (not a DownloadProgress struct, which is internal to
    /// CoreAIKit and can't be constructed outside it).
    @discardableResult
    public func download(
        _ model: ModelID,
        progress: (@Sendable (Double) -> Void)? = nil
    ) async throws -> URL {
        let final = directory.appendingPathComponent(Self.cacheSubpath(for: model), isDirectory: true)
        // Fast-path: already complete on disk.
        if FileManager.default.fileExists(atPath: final.path) { return final }
        // Single-flight: join an in-flight download of the same model.
        if let task = inflight[model] { return try await task.value }

        let task = Task<URL, Error> { try await self.performDownload(model, progress: progress) }
        inflight[model] = task
        defer { inflight[model] = nil }
        return try await task.value
    }

    /// Local bundle root for a model, or nil if not downloaded. Mirrors ModelStore.localURL so
    /// callers can check presence without touching the store.
    public nonisolated func localURL(for model: ModelID) -> URL? {
        let url = directory.appendingPathComponent(Self.cacheSubpath(for: model), isDirectory: true)
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    // MARK: - Download orchestration

    private func performDownload(
        _ model: ModelID,
        progress: (@Sendable (Double) -> Void)?
    ) async throws -> URL {
        let files = try await listFiles(
            repo: model.repo, revision: model.revision, path: model.resolvedPath)
        let totalBytes = files.reduce(0) { $0 + $1.size }

        let fm = FileManager.default
        let final = directory.appendingPathComponent(Self.cacheSubpath(for: model), isDirectory: true)
        let parent = final.deletingLastPathComponent()
        try fm.createDirectory(at: parent, withIntermediateDirectories: true)

        // Stage the whole bundle, then one atomic rename into place (same volume).
        let staging = parent.appendingPathComponent(".staging-\(final.lastPathComponent)")
        let progressRoot = staging.appendingPathComponent(Self.progressDirName)

        // Keep an existing staging tree so a re-download (crash/resume) can continue into it.
        try fm.createDirectory(at: staging, withIntermediateDirectories: true)
        try fm.createDirectory(at: progressRoot, withIntermediateDirectories: true)
        Self.excludeFromBackup(staging)

        // Build the segment plan across all files, loading prior bitmaps for resume.
        var segments: [Segment] = []
        var fileRemaining: [Int] = []   // missing chunks per global fileIndex
        var completedBytes: Int64 = 0
        var fileIndex = 0
        for f in files {
            let destFile = staging.appendingPathComponent(f.relativePath)
            let bits = progressRoot.appendingPathComponent(f.relativePath + ".bits")
            try fm.createDirectory(at: destFile.deletingLastPathComponent(), withIntermediateDirectories: true)
            try fm.createDirectory(at: bits.deletingLastPathComponent(), withIntermediateDirectories: true)

            let geo = Self.chunkGeometry(f.size)
            // Trust a prior partial only if its bitmap length matches AND the staging file exists.
            var done = [UInt8](repeating: 0, count: geo.count)
            if let saved = try? Data(contentsOf: bits), saved.count == geo.count,
               fm.fileExists(atPath: destFile.path) {
                done = [UInt8](saved)
            } else {
                fm.createFile(atPath: destFile.path, contents: nil)
                try Data(count: geo.count).write(to: bits)
            }

            var filePending = 0
            for (ci, g) in geo.enumerated() {
                if done[ci] != 0 {
                    completedBytes += g.length
                } else {
                    segments.append(Segment(
                        fileIndex: fileIndex, url: f.url, dest: destFile,
                        offset: g.offset, length: g.length, ranged: g.ranged,
                        chunkIndex: ci, bits: bits))
                    filePending += 1
                }
            }
            fileRemaining.append(filePending)
            fileIndex += 1
        }

        let gate = ProgressGate()
        Self.reportProgress(progress, gate: gate, completed: completedBytes, total: totalBytes)

        // Bounded fan-out: keep maxConnections chunks in flight, refilling as each lands.
        // Each slot is pinned to its own pool session (= its own connection); when a slot's chunk
        // lands, the next chunk reuses that same session so the connection persists.
        let pool = Self.makeSessionPool(testConfig: testSessionConfig)
        defer { pool.forEach { $0.invalidateAndCancel() } }

        if !segments.isEmpty {
            try await withThrowingTaskGroup(of: (Segment, Int).self) { group in
                var iterator = segments.makeIterator()
                var inFlight = 0
                var slot = 0
                for _ in 0..<pool.count {
                    guard let seg = iterator.next() else { break }
                    let s = slot; slot += 1
                    let sess = pool[s]
                    group.addTask { try await Self.fetchChunk(seg, via: sess, deadline: 30); return (seg, s) }
                    inFlight += 1
                }
                while inFlight > 0 {
                    let (seg, freed) = try await group.next()!
                    inFlight -= 1
                    completedBytes += seg.length
                    Self.reportProgress(progress, gate: gate, completed: completedBytes, total: totalBytes)
                    if let next = iterator.next() {
                        let sess = pool[freed]
                        group.addTask { try await Self.fetchChunk(next, via: sess, deadline: 30); return (next, freed) }
                        inFlight += 1
                    }
                }
            }
        }

        // Atomic placement: staging → final. The runtime must never see a partial bundle.
        try? fm.removeItem(at: final)
        try fm.moveItem(at: staging, to: final)
        Self.excludeFromBackup(final)

        progress?(1.0)
        return final
    }

    // MARK: - Chunk geometry & fetch (nonisolated — pure compute / static IO)

    /// Byte-range geometry for a file. Files ≤ chunkSize get one whole-file segment (plain GET,
    /// no Range header); larger files are split into contiguous ranged chunks.
    /// Exposed as internal for unit testing (testable import).
    nonisolated static func chunkGeometry(_ size: Int64) -> [(offset: Int64, length: Int64, ranged: Bool)] {
        guard size > chunkSize else { return [(0, max(size, 0), false)] }
        var out: [(Int64, Int64, Bool)] = []
        var off: Int64 = 0
        while off < size {
            let len = min(chunkSize, size - off)
            out.append((off, len, true))
            off += len
        }
        return out
    }

    /// Fetch one chunk, write it at its offset in the staging file, then mark its bitmap bit.
    /// The data write+close happens BEFORE the bit is set so a crash can never leave bit==1 over
    /// missing bytes (a re-download just rewrites the same slice). Retries re-fetch only this chunk.
    private nonisolated static func fetchChunk(
        _ seg: Segment, via session: URLSession, deadline: UInt64
    ) async throws {
        var attempt = 0
        while true {
            do {
                var req = URLRequest(url: seg.url)
                if seg.ranged {
                    req.setValue("bytes=\(seg.offset)-\(seg.offset + seg.length - 1)",
                                 forHTTPHeaderField: "Range")
                }
                let (data, resp) = try await Self.dataWithDeadline(req, via: session, seconds: deadline)
                let code = (resp as? HTTPURLResponse)?.statusCode ?? -1
                // A ranged GET must answer 206; a whole-file GET 200. Anything else (e.g. the CDN
                // ignored Range and sent 200) would mis-place bytes, so fail loudly instead.
                guard seg.ranged ? code == 206 : code == 200 else {
                    throw Self.err("HTTP \(code) for \(seg.url.lastPathComponent)")
                }
                let fh = try FileHandle(forWritingTo: seg.dest)
                do {
                    try fh.seek(toOffset: UInt64(seg.offset))
                    try fh.write(contentsOf: data)
                    try fh.close()
                } catch { try? fh.close(); throw error }
                // Bytes are durable in the OS file now → record the chunk as done.
                let bh = try FileHandle(forWritingTo: seg.bits)
                do {
                    try bh.seek(toOffset: UInt64(seg.chunkIndex))
                    try bh.write(contentsOf: Data([1]))
                    try bh.close()
                } catch { try? bh.close(); throw error }
                return
            } catch {
                if Task.isCancelled { throw error }     // group is tearing down — don't retry
                attempt += 1
                if attempt > maxChunkRetries { throw error }
                try? await Task.sleep(nanoseconds: UInt64(attempt) * 500_000_000)  // 0.5s, 1s, …
            }
        }
    }

    /// A wall-clock deadline around one GET (redirect + body). `timeoutIntervalForRequest` is only
    /// an IDLE timer that resets on every received byte, so a connection degrading to a crawl never
    /// trips it and the download wedges. This races the transfer against a hard timeout and cancels
    /// it, so a stalled chunk fails fast and the retry re-opens a fresh connection.
    private nonisolated static func dataWithDeadline(
        _ req: URLRequest, via session: URLSession, seconds: UInt64
    ) async throws -> (Data, URLResponse) {
        try await withThrowingTaskGroup(of: (Data, URLResponse).self) { group in
            group.addTask { try await session.data(for: req, delegate: redirector) }
            group.addTask {
                try await Task.sleep(nanoseconds: seconds * 1_000_000_000)
                throw Self.err("chunk stalled > \(seconds)s")
            }
            defer { group.cancelAll() }                 // cancel the loser (transfer or timer)
            guard let first = try await group.next() else { throw Self.err("no chunk result") }
            return first
        }
    }

    // MARK: - Session pool

    /// A pool of independent `URLSession`s = independent TCP connections. Each slot gets its own
    /// long-lived session so its connection persists across chunks assigned to that slot.
    /// When `testSessionConfig` is provided (testing), the pool shares that config (so a URLProtocol
    /// mock intercepts); production builds use fresh `.default` configs per slot.
    private nonisolated static func makeSessionPool(testConfig: URLSessionConfiguration? = nil) -> [URLSession] {
        let cfg: URLSessionConfiguration
        if let testConfig {
            cfg = testConfig
        } else {
            cfg = URLSessionConfiguration.default
            cfg.httpMaximumConnectionsPerHost = 1
            // waitsForConnectivity MUST be false: on a dead CDN connection, a true value makes the
            // retry block forever "waiting for connectivity" (request timeout doesn't run while
            // waiting) → the download hangs silently at 0. False makes a dead connection fail fast.
            cfg.waitsForConnectivity = false
            cfg.timeoutIntervalForRequest = 25      // IDLE timer (resets on each packet)
            cfg.timeoutIntervalForResource = 7 * 24 * 60 * 60
        }
        return (0..<max(1, maxConnections)).map { _ in URLSession(configuration: cfg) }
    }

    // MARK: - HF tree listing (mirrors CoreAIKit HubClient, which is internal)

    struct PlannedFile: Sendable {
        let url: URL
        let relativePath: String
        let size: Int64
    }

    private struct TreeEntry: Decodable {
        let type: String
        let path: String
        let size: Int64?
        let lfs: LFS?
        struct LFS: Decodable { let size: Int64? }
    }

    /// Enumerates the files under `path` in the repo at the given revision via the HF tree API.
    /// Mirrors CoreAIKit's HubClient.listFiles (which is module-internal and not callable here).
    /// Uses `testSessionConfig` when provided (testing) so a URLProtocol mock intercepts the listing
    /// too; production uses `URLSession.shared`.
    private nonisolated func listFiles(
        repo: String, revision: String, path: String
    ) async throws -> [PlannedFile] {
        // Empty path = repo root (flat bundle layout): no trailing slash, or the API 404s.
        let treePath = path.isEmpty ? "" : "/\(path)"
        guard let api = URL(string:
            "https://huggingface.co/api/models/\(repo)/tree/\(revision)\(treePath)?recursive=true")
        else {
            throw Self.err("bad repo/path: \(repo)/\(path)")
        }
        let session: URLSession = if let cfg = testSessionConfig {
            URLSession(configuration: cfg)
        } else {
            URLSession.shared
        }
        defer { if testSessionConfig != nil { session.finishTasksAndInvalidate() } }
        let (data, resp) = try await session.data(from: api)
        guard (resp as? HTTPURLResponse)?.statusCode == 200 else {
            throw Self.err("HF tree API \( (resp as? HTTPURLResponse)?.statusCode ?? -1) for \(repo)/\(path)")
        }
        let entries = try JSONDecoder().decode([TreeEntry].self, from: data)
        let prefix = path.isEmpty ? "" : (path.hasSuffix("/") ? path : path + "/")
        return try entries.filter { $0.type == "file" }.map { e in
            let rel = (!path.isEmpty && e.path == path)
                ? (e.path as NSString).lastPathComponent
                : String(e.path.dropFirst(prefix.count))
            guard let url = URL(string: "https://huggingface.co/\(repo)/resolve/\(revision)/\(e.path)")
            else { throw Self.err("bad file URL: \(e.path)") }
            return PlannedFile(url: url, relativePath: rel, size: e.lfs?.size ?? e.size ?? 0)
        }
    }

    // MARK: - Path & progress helpers

    /// Replicates ModelID.cacheSubpath (which is internal to CoreAIKit). The on-disk layout is
    /// deterministic: <repo>/<revision>/<resolvedPath>, where resolvedPath is the platform default
    /// ("macos" / "ios") when path is nil.
    private nonisolated static func cacheSubpath(for model: ModelID) -> String {
        "\(model.repo)/\(model.revision)/\(model.resolvedPath)"
    }

    private nonisolated static func reportProgress(
        _ progress: (@Sendable (Double) -> Void)?,
        gate: ProgressGate,
        completed: Int64, total: Int64
    ) {
        guard let progress, total > 0 else { return }
        let f = Double(completed) / Double(total)
        guard gate.pass(f) else { return }
        progress(min(f, 1))
    }

    private nonisolated static func excludeFromBackup(_ url: URL) {
        var v = URLResourceValues()
        v.isExcludedFromBackup = true
        var u = url
        try? u.setResourceValues(v)
    }

    private nonisolated static func err(_ msg: String) -> Error {
        NSError(domain: "ParallelModelDownloader", code: 1,
                userInfo: [NSLocalizedDescriptionKey: msg])
    }
}

// MARK: - Segment

/// One byte-range of one file: where to GET it, where to write it, and which bitmap bit marks it
/// done. Files at or below chunkSize are a single un-ranged segment (a plain whole-file GET).
private struct Segment: Sendable {
    let fileIndex: Int      // which file (for per-file completion accounting)
    let url: URL
    let dest: URL           // staging file this chunk writes into
    let offset: Int64       // byte offset within the file / dest
    let length: Int64
    let ranged: Bool        // send a Range header (false = whole small file)
    let chunkIndex: Int     // index into the file's bitmap
    let bits: URL           // sidecar bitmap file (one byte per chunk)
}

// MARK: - ProgressGate

/// Rate-limits progress callbacks to visible changes (~0.002 of the total), matching ModelStore.
private final class ProgressGate: @unchecked Sendable {
    private let lock = NSLock()
    private var last = -1.0
    func pass(_ fraction: Double) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard fraction - last >= 0.002 || fraction >= 1 else { return false }
        last = fraction
        return true
    }
}

// MARK: - RangePreservingRedirector

/// Carries the `Range` header onto the redirected request. HF `resolve/...` 302-redirects to the
/// CDN; URLSession usually copies headers across a redirect, but if it ever dropped Range the CDN
/// would send the full file (200) and the chunk's `data(for:)` would buffer the whole ~30 GB.
/// This guarantees the redirected GET stays a 206 partial.
private final class RangePreservingRedirector: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
    func urlSession(
        _ session: URLSession, task: URLSessionTask,
        willPerformHTTPRedirection response: HTTPURLResponse, newRequest request: URLRequest,
        completionHandler: @escaping (URLRequest?) -> Void
    ) {
        var req = request
        if let range = task.originalRequest?.value(forHTTPHeaderField: "Range") {
            req.setValue(range, forHTTPHeaderField: "Range")
        }
        completionHandler(req)
    }
}
