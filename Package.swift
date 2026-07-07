// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "coreai-runner",
    platforms: [.macOS("27.0")],
    products: [
        // Library target: importable by coreai-server, Ditto, etc.
        .library(name: "CoreAIRunner", targets: ["CoreAIRunner"]),
        // Executable: standalone binary used by ComfyUI-CoreAI as subprocess.
        .executable(name: "coreai-runner", targets: ["CoreAIRunnerCLI"]),
    ],
    dependencies: [
        // Typed pipelines over Core AI: DepthEstimator, ObjectDetector,
        // KitVisionModel, ChatSession. Links the patched pipelined engine
        // (CoreAILM from john-rocky/coreai-models v0.1.2-zoo).
        .package(url: "https://github.com/john-rocky/coreai-kit", branch: "main"),

        // Lightweight HTTP server with Unix domain socket support.
        .package(url: "https://github.com/hummingbird-project/hummingbird", from: "2.0.0"),

        // Structured logging (Apple standard).
        .package(url: "https://github.com/apple/swift-log", from: "1.5.0"),
    ],
    targets: [
        .target(
            name: "CoreAIRunner",
            dependencies: [
                .product(name: "CoreAIKit", package: "coreai-kit"),
                .product(name: "CoreAIKitVision", package: "coreai-kit"),
                .product(name: "Hummingbird", package: "hummingbird"),
                .product(name: "HummingbirdCore", package: "hummingbird"),
                .product(name: "HummingbirdRouter", package: "hummingbird"),
                .product(name: "Logging", package: "swift-log"),
            ]
            // SAM 3 (CoreAIImageSegmenter) + FLUX.2 (CoreAIDiffusionPipeline) are NOT
            // linked explicitly: their adapters `#if canImport` them and Swift autolinks
            // each system framework only when the SDK actually ships it. The macOS 27
            // beta SDK carries the CoreAI core runtime but not these two pipeline
            // frameworks yet; explicit -framework flags would fail the link today.
        ),
        .executableTarget(
            name: "CoreAIRunnerCLI",
            dependencies: ["CoreAIRunner"]
        ),
        .testTarget(
            name: "CoreAIRunnerTests",
            dependencies: ["CoreAIRunner"]
        ),
    ]
)
