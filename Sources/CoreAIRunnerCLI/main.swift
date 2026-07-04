// main.swift — entry point for the coreai-runner binary.
//
// Usage:
//   coreai-runner [--socket /tmp/coreai-runner.sock] [--log-level info]
//
// The binary binds to a Unix domain socket and serves HTTP until SIGTERM/SIGINT.
// ComfyUI-CoreAI's bridge.py spawns this process and connects to the socket.

import CoreAIRunner
import Foundation
import Logging

@main
struct CoreAIRunnerCLI {
    static func main() async {
        var socketPath = "/tmp/coreai-runner.sock"
        var logLevel: Logger.Level = .info

        // Parse arguments
        var i = 1
        let args = CommandLine.arguments
        while i < args.count {
            switch args[i] {
            case "--socket", "-s":
                i += 1
                if i < args.count { socketPath = args[i] }
            case "--log-level":
                i += 1
                if i < args.count, let level = Logger.Level(rawValue: args[i]) {
                    logLevel = level
                }
            case "--help", "-h":
                printHelp()
                exit(0)
            case "--version", "-v":
                printVersion()
                exit(0)
            default:
                break
            }
            i += 1
        }

        // Configure logging
        var logger = Logger(label: "coreai-runner")
        logger.logLevel = logLevel

        logger.info("Starting coreai-runner")
        logger.info("Socket: \(socketPath)")
        logger.info("Log level: \(logLevel)")

        // Install signal handlers for clean shutdown
        let server = RunnerServer(socketPath: socketPath, logger: logger)

        // SIGTERM / SIGINT → graceful exit
        let signalQueue = DispatchQueue(label: "coreai-runner.signals")
        let signalSource = DispatchSource.makeSignalSource(signal: SIGTERM, queue: signalQueue)
        signalSource.setEventHandler {
            logger.info("Received SIGTERM, shutting down...")
            exit(0)
        }
        signal(SIGTERM, SIG_IGN)
        signalSource.resume()

        let intSource = DispatchSource.makeSignalSource(signal: SIGINT, queue: signalQueue)
        intSource.setEventHandler {
            logger.info("Received SIGINT, shutting down...")
            exit(0)
        }
        signal(SIGINT, SIG_IGN)
        intSource.resume()

        // Start server (blocks)
        do {
            try await server.run()
        } catch {
            logger.error("Server failed: \(error)")
            // Clean up socket
            try? FileManager.default.removeItem(atPath: socketPath)
            exit(1)
        }
    }

    static func printHelp() {
        print("""
        coreai-runner — Core AI model inference server

        USAGE:
            coreai-runner [OPTIONS]

        OPTIONS:
            --socket <path>     Unix domain socket path (default: /tmp/coreai-runner.sock)
            --log-level <level> Log level: trace, debug, info, warning, error (default: info)
            -h, --help          Show this help
            -v, --version       Show version

        ENDPOINTS:
            GET  /v1/health              Device info, loaded models, thermal state
            GET  /v1/models              List available models (optional ?capability=X)
            POST /v1/predict             Run inference
            POST /v1/models/:id/load     Download and load a model
            POST /v1/models/:id/unload   Unload a model from memory

        EXAMPLE:
            coreai-runner --socket /tmp/coreai-runner.sock --log-level debug
            curl --unix-socket /tmp/coreai-runner.sock http://localhost/v1/health
        """)
    }

    static func printVersion() {
        print("coreai-runner 1.0.0-dev")
    }
}
