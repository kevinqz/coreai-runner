// DownloadTests.swift — tests for ParallelModelDownloader.
//
// Tests the pure-logic parts (chunk geometry) and an end-to-end download over a mock URLProtocol
// that serves byte-range requests from an in-memory buffer (no network needed).

import Testing
import Foundation
import CoreAIKit
@testable import CoreAIRunner

// MARK: - Chunk geometry (pure logic)

@Test func chunkGeometry_smallFile_isSingleWholeFileSegment() {
    // A file smaller than chunkSize (16 MiB) should be one un-ranged segment.
    let geo = ParallelModelDownloader.chunkGeometry(1_000_000)
    #expect(geo.count == 1)
    #expect(geo[0].offset == 0)
    #expect(geo[0].length == 1_000_000)
    #expect(geo[0].ranged == false)
}

@Test func chunkGeometry_exactChunkSize_isSingleWholeFileSegment() {
    // A file exactly at chunkSize is NOT split (guard is `size > chunkSize`).
    let chunkSize = 16 * 1024 * 1024
    let geo = ParallelModelDownloader.chunkGeometry(Int64(chunkSize))
    #expect(geo.count == 1)
    #expect(geo[0].ranged == false)
}

@Test func chunkGeometry_largeFile_isSplitIntoRangedChunks() {
    // A 40 MiB file → 3 chunks: 16 + 16 + 8 MiB, all ranged.
    let chunkSize: Int64 = 16 * 1024 * 1024
    let total: Int64 = 40 * 1024 * 1024
    let geo = ParallelModelDownloader.chunkGeometry(total)
    #expect(geo.count == 3)

    // First two chunks are full chunkSize, last is the remainder.
    #expect(geo[0].offset == 0)
    #expect(geo[0].length == chunkSize)
    #expect(geo[0].ranged == true)

    #expect(geo[1].offset == chunkSize)
    #expect(geo[1].length == chunkSize)
    #expect(geo[1].ranged == true)

    let remainder = total - 2 * chunkSize
    #expect(geo[2].offset == 2 * chunkSize)
    #expect(geo[2].length == remainder)
    #expect(geo[2].ranged == true)

    // Offsets + lengths must tile the file exactly (no gaps, no overlaps).
    #expect(geo.reduce(Int64(0)) { $0 + $1.length } == total)
}

@Test func chunkGeometry_emptyFile_isSingleZeroLengthSegment() {
    let geo = ParallelModelDownloader.chunkGeometry(0)
    #expect(geo.count == 1)
    #expect(geo[0].offset == 0)
    #expect(geo[0].length == 0)
    #expect(geo[0].ranged == false)
}

// MARK: - End-to-end download via mock URLProtocol

/// Helper: creates a session config wired to the MockHFCDN URLProtocol, so the downloader's
/// internal sessions hit our in-memory mock instead of the network.
private func mockSessionConfig() -> URLSessionConfiguration {
    let cfg = URLSessionConfiguration.default
    cfg.protocolClasses = [MockHFCDN.self] + (cfg.protocolClasses ?? [])
    cfg.httpMaximumConnectionsPerHost = 1
    return cfg
}

@Test func parallelDownload_writesCompleteBundle() async throws {
    // Two-file "bundle": a small tokenizer.json (19 B) and a larger payload (35 MiB, 3 chunks).
    // The mock URLProtocol serves range requests from an in-memory buffer, so no network is needed.
    let payload = Data((0..<35 * 1024 * 1024).map { _ in UInt8.random(in: 0...255) })
    let tokenizer = Data("[{\"hello\":\"world\"}]".utf8)

    MockHFCDN.shared = MockHFCDN()
    MockHFCDN.shared!.files["test-org/test-model/macos/payload.bin"] = payload
    MockHFCDN.shared!.files["test-org/test-model/macos/tokenizer.json"] = tokenizer

    let tmp = FileManager.default.temporaryDirectory
        .appendingPathComponent("dl-test-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: tmp) }

    let downloader = ParallelModelDownloader(directory: tmp, sessionConfig: mockSessionConfig())
    let url = try await downloader.download(
        ModelID("test-org/test-model", path: nil, revision: "main"))

    // The bundle root should exist and contain both files.
    #expect(FileManager.default.fileExists(atPath: url.path))

    let writtenPayload = try Data(contentsOf: url.appendingPathComponent("payload.bin"))
    let writtenTokenizer = try Data(contentsOf: url.appendingPathComponent("tokenizer.json"))

    // Byte-exact fidelity: the written payload must equal the source.
    #expect(writtenPayload == payload)
    #expect(writtenTokenizer == tokenizer)
}

@Test func parallelDownload_resumeAfterPartialDownload() async throws {
    // Simulate a crash mid-download: pre-populate the staging dir with a partial bitmap (chunk 0
    // done, chunks 1 and 2 missing), then verify a re-download fetches only the missing chunks
    // and produces a complete, correct file.
    let payload = Data((0..<35 * 1024 * 1024).map { _ in UInt8.random(in: 0...255) })

    MockHFCDN.shared = MockHFCDN()
    MockHFCDN.shared!.files["test-org/resume-model/macos/payload.bin"] = payload

    let tmp = FileManager.default.temporaryDirectory
        .appendingPathComponent("dl-resume-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: tmp) }

    // Manually set up the staging tree as if a prior run wrote the full file but only marked
    // chunk 0 as done in the bitmap, then crashed before marking chunks 1–2.
    let cacheSubpath = "test-org/resume-model/main/macos"
    let final = tmp.appendingPathComponent(cacheSubpath, isDirectory: true)
    let parent = final.deletingLastPathComponent()
    let staging = parent.appendingPathComponent(".staging-\(final.lastPathComponent)")
    let progressRoot = staging.appendingPathComponent(".dl-progress")
    try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: progressRoot, withIntermediateDirectories: true)

    // Write chunk 0 (the first 16 MiB) with the CORRECT bytes — as if a prior run downloaded it
    // successfully before crashing. Chunks 1 and 2 are zero-filled (missing); the resume should
    // download only those and overwrite the zeros.
    let stagingFile = staging.appendingPathComponent("payload.bin")
    let chunk0 = payload.prefix(16 * 1024 * 1024)
    var stagingData = Data(chunk0)
    stagingData.append(Data(repeating: 0x00, count: payload.count - chunk0.count))
    FileManager.default.createFile(atPath: stagingFile.path, contents: stagingData)

    // Bitmap: 3 chunks, only chunk 0 done (0x01). Chunks 1 and 2 are 0x00 (missing).
    let bits = progressRoot.appendingPathComponent("payload.bin.bits")
    try Data([1, 0, 0]).write(to: bits)

    // Now download — it should resume, fetching chunks 1 and 2 and overwriting their bytes.
    let downloader = ParallelModelDownloader(directory: tmp, sessionConfig: mockSessionConfig())
    let url = try await downloader.download(
        ModelID("test-org/resume-model", path: nil, revision: "main"))

    let written = try Data(contentsOf: url.appendingPathComponent("payload.bin"))
    #expect(written == payload)   // full fidelity after resume
}

@Test func parallelDownload_alreadyPresent_returnsFastPath() async throws {
    // If the bundle is already complete, download() should return immediately without hitting
    // the network (the mock has no files registered, so any HTTP attempt would fail).
    let tmp = FileManager.default.temporaryDirectory
        .appendingPathComponent("dl-fast-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: tmp) }

    // Pre-create the final bundle dir with a metadata.json so it looks complete.
    let cacheSubpath = "test-org/cached-model/main/macos"
    let final = tmp.appendingPathComponent(cacheSubpath, isDirectory: true)
    try FileManager.default.createDirectory(at: final, withIntermediateDirectories: true)
    try Data("{\"version\":1}".utf8).write(to: final.appendingPathComponent("metadata.json"))

    // Don't register any mock files — the fast path must not touch the network.
    MockHFCDN.shared = MockHFCDN()

    let downloader = ParallelModelDownloader(directory: tmp, sessionConfig: mockSessionConfig())
    let url = try await downloader.download(
        ModelID("test-org/cached-model", path: nil, revision: "main"))

    #expect(url == final)
    #expect(FileManager.default.fileExists(atPath: url.appendingPathComponent("metadata.json").path))
}

// MARK: - MockHFCDN URLProtocol

/// A URLProtocol subclass that serves byte-range requests from an in-memory buffer, mimicking a
/// HF CDN that supports `Range` headers. Files are keyed by "<repo>/<name>". Also answers the HF
/// tree API for file listing. Uses a shared singleton so the URLProtocol instances (which are
/// created and destroyed by URLSession) can access the registered files.
final class MockHFCDN: URLProtocol, @unchecked Sendable {
    /// Shared store: "<org>/<name>" → file data. Set by the test before invoking the downloader.
    nonisolated(unsafe) static var shared: MockHFCDN?

    var files: [String: Data] = [:]

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func stopLoading() {}

    override func startLoading() {
        guard let url = request.url else {
            client?.urlProtocol(self, didFailWithError: URLError(.badURL))
            return
        }
        guard let store = MockHFCDN.shared else {
            client?.urlProtocol(self, didFailWithError: URLError(.badURL))
            return
        }

        // --- Tree API: /api/models/<org>/<name>/tree/<rev>/<path>?recursive=true ---
        if url.path.hasPrefix("/api/models/") {
            // Extract repo from the path: /api/models/<org>/<name>/tree/<rev>/<path>
            let comps = url.path.split(separator: "/", omittingEmptySubsequences: true).map(String.init)
            // ["api","models","<org>","<name>","tree","<rev>","<path>", ...]
            guard comps.count >= 4 else {
                let resp = HTTPURLResponse(url: url, statusCode: 404, httpVersion: "1.1",
                                           headerFields: nil)!
                client?.urlProtocol(self, didReceive: resp, cacheStoragePolicy: .notAllowed)
                client?.urlProtocolDidFinishLoading(self)
                return
            }
            let repo = "\(comps[2])/\(comps[3])"
            // The tree path is everything AFTER "<api>/<models>/<org>/<name>/tree/<rev>/".
            // comps[6...] = the subtree path (e.g. "macos"); comps[5] = the revision.
            let treePath = comps.count >= 7 ? comps[6...].joined(separator: "/") : ""
            var entries: [[String: Any]] = []
            for key in store.files.keys where key.hasPrefix(repo + "/") {
                let fullPath = String(key.dropFirst(repo.count + 1))   // e.g. "macos/payload.bin"
                // Only include files under the requested tree path.
                if treePath.isEmpty || fullPath == treePath || fullPath.hasPrefix(treePath + "/") {
                    entries.append([
                        "type": "file",
                        "path": fullPath,
                        "size": store.files[key]!.count,
                        "lfs": ["size": store.files[key]!.count],
                    ])
                }
            }
            let body = try! JSONSerialization.data(withJSONObject: entries)
            let resp = HTTPURLResponse(url: url, statusCode: 200, httpVersion: "1.1",
                                       headerFields: ["Content-Type": "application/json"])!
            client?.urlProtocol(self, didReceive: resp, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: body)
            client?.urlProtocolDidFinishLoading(self)
            return
        }

        // --- Resolve (file download): /<org>/<name>/resolve/<rev>/<path>/<file> ---
        let pathComps = url.path.split(separator: "/", omittingEmptySubsequences: true).map(String.init)
        // ["<org>", "<name>", "resolve", "<rev>", "<path>", "<file>"] (path may be multi-segment)
        guard pathComps.count >= 5, pathComps[2] == "resolve" else {
            client?.urlProtocol(self, didFailWithError: URLError(.fileDoesNotExist))
            return
        }
        let repo = "\(pathComps[0])/\(pathComps[1])"
        // The path segments between "resolve/<rev>/" and the end are the file path inside the repo.
        // For our mock the key is always "<repo>/<lastSegment>" (flat layout).
        let fileSegments = Array(pathComps[4...])
        let filePath = fileSegments.joined(separator: "/")
        let key = "\(repo)/\(filePath)"

        guard let fullData = store.files[key] else {
            let resp = HTTPURLResponse(url: url, statusCode: 404, httpVersion: "1.1",
                                       headerFields: nil)!
            client?.urlProtocol(self, didReceive: resp, cacheStoragePolicy: .notAllowed)
            client?.urlProtocolDidFinishLoading(self)
            return
        }

        let rangeHeader = request.value(forHTTPHeaderField: "Range")
        if let rangeHeader {
            // Parse "bytes=<start>-<end>"
            let nums = rangeHeader
                .replacingOccurrences(of: "bytes=", with: "")
                .split(separator: "-")
                .compactMap { Int($0) }
            guard nums.count == 2 else {
                client?.urlProtocol(self, didFailWithError: URLError(.badURL))
                return
            }
            let start = nums[0], end = nums[1]
            let chunk = fullData.subdata(in: start..<(end + 1))
            let resp = HTTPURLResponse(url: url, statusCode: 206, httpVersion: "1.1", headerFields: [
                "Content-Range": "bytes \(start)-\(end)/\(fullData.count)",
                "Content-Length": "\(chunk.count)",
            ])!
            client?.urlProtocol(self, didReceive: resp, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: chunk)
            client?.urlProtocolDidFinishLoading(self)
        } else {
            // Whole-file GET (200).
            let resp = HTTPURLResponse(url: url, statusCode: 200, httpVersion: "1.1", headerFields: [
                "Content-Length": "\(fullData.count)",
            ])!
            client?.urlProtocol(self, didReceive: resp, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: fullData)
            client?.urlProtocolDidFinishLoading(self)
        }
    }
}
