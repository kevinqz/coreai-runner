// CatalogClient.swift — fetches model metadata from the coreai-catalog API.
//
// The catalog is the intelligence layer: it knows what models exist, what
// capabilities they have, where to download them, and what benchmark numbers
// to show in the UI. We fetch lazily and cache for 5 minutes to avoid hitting
// the API on every node render.
//
// API base: https://coreai-catalog.nousresearch.com/v1
// (graceful fallback to nil if the API is unreachable — installed models
//  still work without the catalog.)

import Foundation
import Logging

private let catalogLogger = Logger(label: "coreai-runner.catalog")

fileprivate func logError(_ msg: String) {
    catalogLogger.error(.init(stringLiteral: msg))
}

public actor CatalogClient {

    public static let shared = CatalogClient()

    // The catalog API. Override for testing.
    private let apiBase: URL
    private let session: URLSession
    private var cache: [String: (entry: CatalogModelEntry, expiresAt: Date)] = [:]
    private let cacheTTL: TimeInterval = 300  // 5 minutes

    // Bulk list cache (separate from per-model cache).
    private var listCache: [CatalogModelEntry]?
    private var listCacheExpiresAt: Date?

    public init(apiBase: String = "https://raw.githubusercontent.com/kevinqz/coreai-catalog/main/dist/") {
        self.apiBase = URL(string: apiBase)!
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 10
        config.timeoutIntervalForResource = 30
        self.session = URLSession(configuration: config)
    }

    // MARK: - Single model

    public func getModel(id: String) async -> CatalogModelEntry? {
        // Check cache
        if let cached = cache[id], cached.expiresAt > Date() {
            return cached.entry
        }

        // Fetch full catalog and find model (catalog is static JSON, not a REST API)
        guard let url = URL(string: "catalog.json", relativeTo: apiBase) else { return nil }
        let responseData: Data
        do {
            let (data, response) = try await session.data(from: url)
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                return cache[id]?.entry
            }
            responseData = data
        } catch {
            return cache[id]?.entry
        }

        // Parse { "models": [...] }
        let models: [CatalogModelEntry]
        do {
            let wrapper = try JSONDecoder().decode(CatalogListResponse.self, from: responseData)
            models = wrapper.models
        } catch {
            // Log the decode error so it's visible in the runner console
            logError("Catalog decode failed: \(error)")
            return cache[id]?.entry
        }

        // Cache ALL models from this fetch (populates individual cache entries)
        for m in models {
            cache[m.id] = (m, Date().addingTimeInterval(cacheTTL))
        }

        return cache[id]?.entry
    }

    // MARK: - List models

    public func listModels(capability: String? = nil, device: String? = nil) async -> [CatalogModelEntry] {
        // Check bulk cache (only when no filters — filtered queries always fetch fresh)
        if capability == nil && device == nil,
           let cached = listCache, let expiresAt = listCacheExpiresAt, expiresAt > Date() {
            return cached
        }

        guard let url = URL(string: "catalog.json", relativeTo: apiBase) else {
            logError("Catalog URL construction failed")
            return listCache ?? []
        }
        let responseData: Data
        do {
            let (data, response) = try await session.data(from: url)
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                logError("Catalog fetch: HTTP \((response as? HTTPURLResponse)?.statusCode ?? -1)")
                return listCache ?? []
            }
            responseData = data
        } catch {
            logError("Catalog fetch error: \(error)")
            return listCache ?? []
        }

        // The catalog returns { "models": [...] }
        let models: [CatalogModelEntry]
        do {
            let wrapper = try JSONDecoder().decode(CatalogListResponse.self, from: responseData)
            models = wrapper.models
        } catch {
            logError("Catalog decode failed (listModels): \(error)")
            return listCache ?? []
        }

        // Apply client-side filters
        var filtered = models
        if let capability {
            filtered = filtered.filter { $0.capabilities.contains(capability) }
        }
        if let device {
            switch device {
            case "mac":
                filtered = filtered.filter { $0.deviceSupport?.mac == true || $0.deviceSupport?.macOnly == true }
            case "iphone":
                filtered = filtered.filter { $0.deviceSupport?.iphone == true }
            case "ipad":
                filtered = filtered.filter { $0.deviceSupport?.ipad == true }
            default:
                break
            }
        }

        // Cache the full list (unfiltered only)
        if capability == nil && device == nil {
            listCache = models
            listCacheExpiresAt = Date().addingTimeInterval(cacheTTL)
        }

        return filtered
    }

    // MARK: - Invalidate

    public func invalidate() {
        cache.removeAll()
        listCache = nil
        listCacheExpiresAt = nil
    }
}

// MARK: - Catalog model types

public struct CatalogModelEntry: Codable, Sendable, Hashable {
    public let id: String
    public let name: String
    public let family: String?
    public let capabilities: [String]
    public let modalities: Modalities?
    public let size: ModelSize?
    public let runtime: ModelRuntime?
    public let deviceSupport: DeviceSupport?
    public let license: License?
    public let readinessScore: Int?
    public let provenance: Provenance?

    enum CodingKeys: String, CodingKey {
        case id, name, family, capabilities, modalities, size, runtime
        case deviceSupport = "device_support"
        case license, readinessScore = "readiness_score"
        case provenance
    }

    public struct Modalities: Codable, Sendable, Hashable {
        public let input: [String]?
        public let output: [String]?
    }

    public struct ModelSize: Codable, Sendable, Hashable {
        public let parameters: String?
        public let precision: String?
        public let artifactSize: String?

        enum CodingKeys: String, CodingKey {
            case parameters, precision
            case artifactSize = "artifact_size"
        }

        /// Parse artifact size string like "54.5MB" to a Double in MB.
        public var sizeInMB: Double? {
            guard let artifactSize else { return nil }
            let cleaned = artifactSize
                .replacingOccurrences(of: "MB", with: "")
                .replacingOccurrences(of: "GB", with: "")
                .trimmingCharacters(in: .whitespaces)
            guard let value = Double(cleaned) else { return nil }
            return artifactSize.uppercased().contains("GB") ? value * 1024 : value
        }
    }

    public struct ModelRuntime: Codable, Sendable, Hashable {
        public let runtimeName: String?
        public let runner: String?
        public let stockRuntime: Bool?
        public let patchRequired: Bool?
        public let tokenizerRequired: Bool?
        public let processorRequired: Bool?
        public let aotRequired: Bool?

        enum CodingKeys: String, CodingKey {
            case runtimeName = "runtime_name"
            case runner
            case stockRuntime = "stock_runtime"
            case patchRequired = "patch_required"
            case tokenizerRequired = "tokenizer_required"
            case processorRequired = "processor_required"
            case aotRequired = "aot_required"
        }

        // Custom decoder: catalog uses "yes"/"no"/"unknown" (strings) for
        // some boolean fields alongside true/false/null.
        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            runtimeName = try c.decodeIfPresent(String.self, forKey: .runtimeName)
            runner = try c.decodeIfPresent(String.self, forKey: .runner)
            stockRuntime = try c.decodeFlexibleBool(forKey: .stockRuntime)
            patchRequired = try c.decodeFlexibleBool(forKey: .patchRequired)
            tokenizerRequired = try c.decodeFlexibleBool(forKey: .tokenizerRequired)
            processorRequired = try c.decodeFlexibleBool(forKey: .processorRequired)
            aotRequired = try c.decodeFlexibleBool(forKey: .aotRequired)
        }
    }

    public struct DeviceSupport: Codable, Sendable, Hashable {
        public let iphone: Bool?
        public let ipad: Bool?  // can be null or "unknown" in catalog
        public let mac: Bool?
        public let macOnly: Bool?

        enum CodingKeys: String, CodingKey {
            case iphone, ipad, mac
            case macOnly = "mac_only"
        }

        // Custom decoder: catalog uses "unknown" (string) for some fields
        // alongside true/false/null. Decode any non-bool as nil.
        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            iphone = try c.decodeFlexibleBool(forKey: .iphone)
            ipad = try c.decodeFlexibleBool(forKey: .ipad)
            mac = try c.decodeFlexibleBool(forKey: .mac)
            macOnly = try c.decodeFlexibleBool(forKey: .macOnly)
        }

        public func encode(to encoder: Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encodeIfPresent(iphone, forKey: .iphone)
            try c.encodeIfPresent(ipad, forKey: .ipad)
            try c.encodeIfPresent(mac, forKey: .mac)
            try c.encodeIfPresent(macOnly, forKey: .macOnly)
        }
    }

    public struct License: Codable, Sendable, Hashable {
        public let name: String?
        public let commercialUse: String?

        enum CodingKeys: String, CodingKey {
            case name
            case commercialUse = "commercial_use"
        }
    }

    public struct Provenance: Codable, Sendable, Hashable {
        public let huggingface: HuggingFace?

        public struct HuggingFace: Codable, Sendable, Hashable {
            public let owner: String
            public let repo: String
            public let url: String?
            public let revision: String?
            public let files: [HFFile]?

            public struct HFFile: Codable, Sendable, Hashable {
                public let path: String
                public let sizeBytes: Int?

                enum CodingKeys: String, CodingKey {
                    case path
                    case sizeBytes = "size_bytes"
                }
            }
        }
    }
}

struct CatalogListResponse: Codable, Sendable {
    let models: [CatalogModelEntry]
}

// MARK: - Flexible Bool decoding

/// The catalog uses "unknown" (string) for some device_support fields
/// alongside true/false/null. This extension decodes any non-bool value as nil.
// Shared across DeviceSupport and ModelRuntime
extension KeyedDecodingContainer {
    func decodeFlexibleBool(forKey key: Key) throws -> Bool? {
        if let value = try? decodeIfPresent(Bool.self, forKey: key) {
            return value
        }
        return nil
    }
}
