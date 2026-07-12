// HealthTests.swift — smoke tests for the HTTP server and adapter protocol.
//
// These tests verify the server infrastructure without requiring actual
// models (which need a real Apple Silicon device with downloaded bundles).
// Model-level tests run on-device via the CoreAIKit test suite.

import Testing
import Foundation
@testable import CoreAIRunner

@Test func healthResponseCodable() async throws {
    let health = HealthResponse(
        status: "healthy",
        device: "MacBook Pro",
        chip: "Apple M4 Pro",
        memoryTotalGB: 48.0,
        memoryAvailableGB: 31.2,
        macosVersion: "26.6",
        coreaiVersion: "26.6",
        loadedModels: ["depth-anything-3-small"],
        thermalState: "nominal"
    )

    let encoded = try JSONEncoder().encode(health)
    let decoded = try JSONDecoder().decode(HealthResponse.self, from: encoded)

    #expect(decoded.status == "healthy")
    #expect(decoded.chip == "Apple M4 Pro")
    #expect(decoded.loadedModels == ["depth-anything-3-small"])
}

@Test func predictRequestDecoding() async throws {
    let json = """
    {
        "model_id": "depth-anything-3-small",
        "input": {
            "image_path": "/tmp/test.png"
        },
        "options": {
            "compute_unit": "neuralEngine"
        }
    }
    """.data(using: .utf8)!

    let request = try JSONDecoder().decode(PredictRequest.self, from: json)

    #expect(request.modelID == "depth-anything-3-small")
    #expect(request.input.imagePath == "/tmp/test.png")
    #expect(request.options?.computeUnit == .neuralEngine)
}

@Test func predictRequestWithPrompts() async throws {
    let json = """
    {
        "model_id": "qwen3-vl-2b",
        "input": {
            "image_path": "/tmp/photo.jpg",
            "prompt": "Describe this image.",
            "max_tokens": 200,
            "temperature": 0.7
        }
    }
    """.data(using: .utf8)!

    let request = try JSONDecoder().decode(PredictRequest.self, from: json)

    #expect(request.modelID == "qwen3-vl-2b")
    #expect(request.input.prompt == "Describe this image.")
    #expect(request.input.maxTokens == 200)
    #expect(request.input.temperature == 0.7)
}

@Test func errorResponseEncoding() async throws {
    let error = ErrorResponse(
        code: "MODEL_NOT_INSTALLED",
        message: "Model 'depth-anything-3-small' is not installed.",
        modelID: "depth-anything-3-small"
    )

    let encoded = try JSONEncoder().encode(error)
    let decoded = try JSONDecoder().decode(ErrorResponse.self, from: encoded)

    #expect(decoded.error.code == "MODEL_NOT_INSTALLED")
    #expect(decoded.error.modelID == "depth-anything-3-small")
}

@Test func adapterInputComputeUnitDefault() {
    let input = AdapterInput(modelID: "test", imagePath: "/tmp/x.png")
    #expect(input.computeUnit == .auto)
}

@Test func catalogEntryDecoding() throws {
    let json = """
    {
        "id": "depth-anything-3-small",
        "name": "Depth Anything 3 Small",
        "family": "Depth Anything",
        "capabilities": ["monocular-depth"],
        "size": {
            "parameters": "small",
            "precision": "fp16",
            "artifact_size": "54.5MB"
        },
        "runtime": {
            "runner": "CoreAIRunner",
            "stock_runtime": true,
            "patch_required": false,
            "processor_required": true
        },
        "device_support": {
            "iphone": true,
            "mac": true
        },
        "license": {
            "name": "Apache-2.0",
            "commercial_use": "likely"
        },
        "readiness_score": 93
    }
    """.data(using: .utf8)!

    let entry = try JSONDecoder().decode(CatalogModelEntry.self, from: json)

    #expect(entry.id == "depth-anything-3-small")
    #expect(entry.capabilities == ["monocular-depth"])
    #expect(entry.size?.precision == "fp16")
    #expect(entry.size?.sizeInMB == 54.5)
    #expect(entry.runtime?.runner == "CoreAIRunner")
    #expect(entry.runtime?.patchRequired == false)
    #expect(entry.deviceSupport?.iphone == true)
    #expect(entry.license?.name == "Apache-2.0")
}

@Test func catalogEntrySizeInGB() throws {
    let json = """
    {
        "id": "test",
        "name": "Big Model",
        "capabilities": [],
        "size": {"artifact_size": "4GB"}
    }
    """.data(using: .utf8)!

    let entry = try JSONDecoder().decode(CatalogModelEntry.self, from: json)
    #expect(entry.size?.sizeInMB == 4096.0)
}

@Test func capabilitiesResponseCodable() async throws {
    let caps = CapabilitiesResponse()
    let encoded = try JSONEncoder().encode(caps)
    let decoded = try JSONDecoder().decode(CapabilitiesResponse.self, from: encoded)

    #expect(decoded.runtime == "coreai-runner")
    #expect(decoded.protocolVersion == "coreai-runner.v2")   // RFC-0400 §3.4
    #expect(decoded.supports.llm == true)
    // RFC-0400 §3.1/§3.2: action inference returns 501 and no host-loop conformance
    // case runs yet, so both advertise false — truthful capabilities.
    #expect(decoded.supports.action == false)
    #expect(decoded.supports.hostLoop == false)
    #expect(decoded.supports.prefixCache == true)
}

@Test func modelStatusResponseWithCompilation() async throws {
    let status = ModelStatusResponse(
        modelID: "test-model",
        installed: true,
        loaded: false,
        compilation: "aot",
        engineVariant: "coreai-sequential"
    )
    let encoded = try JSONEncoder().encode(status)
    let decoded = try JSONDecoder().decode(ModelStatusResponse.self, from: encoded)

    #expect(decoded.modelID == "test-model")
    #expect(decoded.compilation == "aot")
    #expect(decoded.engineVariant == "coreai-sequential")
}
