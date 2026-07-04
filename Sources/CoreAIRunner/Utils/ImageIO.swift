// ImageIO.swift — file-based image transfer between Python (ComfyUI) and Swift (CoreAIKit).
//
// ComfyUI tensors (CHW float32) are saved as PNG by bridge.py. The runner
// reads them as CGImage, runs inference, writes results as PNG. The bottleneck
// is inference (15ms–17s), not file I/O (~2ms for a 1024×1024 PNG).

import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

public enum ImageIO {

    /// Load a PNG/JPEG from disk as a CGImage.
    public static func loadCGImage(from path: String) throws -> CGImage {
        guard let url = URL(string: "file://\(path)") else {
            throw CoreAIRunnerError.invalidInput(detail: "invalid image path: \(path)")
        }
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else {
            throw CoreAIRunnerError.invalidInput(detail: "cannot create image source: \(path)")
        }
        guard let image = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
            throw CoreAIRunnerError.invalidInput(detail: "cannot decode image: \(path)")
        }
        return image
    }

    /// Save a CGImage as PNG to a temp file. Returns the absolute path.
    @discardableResult
    public static func savePNG(_ image: CGImage, prefix: String = "coreai_out") throws -> String {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("\(prefix)_\(UUID().uuidString).png")
        guard let dest = CGImageDestinationCreateWithURL(
            url as CFURL, UTType.png.identifier as CFString, 1, nil
        ) else {
            throw CoreAIRunnerError.inferenceFailed(
                modelID: "", detail: "cannot create PNG destination")
        }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else {
            throw CoreAIRunnerError.inferenceFailed(
                modelID: "", detail: "cannot finalize PNG write")
        }
        return url.path
    }

    /// Save raw grayscale float values as a normalized grayscale PNG. Returns the path.
    @discardableResult
    public static func saveGrayscalePNG(
        width: Int, height: Int, values: [Float], prefix: String = "coreai_depth"
    ) throws -> String {
        guard !values.isEmpty, width > 0, height > 0 else {
            throw CoreAIRunnerError.invalidInput(detail: "empty or zero-dimension depth map")
        }
        var lo = Float.greatestFiniteMagnitude
        var hi = -Float.greatestFiniteMagnitude
        for v in values where v.isFinite {
            lo = min(lo, v)
            hi = max(hi, v)
        }
        let range = hi - lo
        var bytes = [UInt8](repeating: 0, count: width * height)
        if range > 0 {
            for i in 0..<bytes.count {
                bytes[i] = UInt8(max(0, min(255, (values[i] - lo) / range * 255)))
            }
        }
        let data = Data(bytes)
        guard let provider = CGDataProvider(data: data as CFData) else {
            throw CoreAIRunnerError.inferenceFailed(
                modelID: "", detail: "cannot create data provider for depth PNG")
        }
        guard let cgImage = CGImage(
            width: width, height: height,
            bitsPerComponent: 8, bitsPerPixel: 8,
            bytesPerRow: width,
            space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
            provider: provider,
            decode: nil, shouldInterpolate: false,
            intent: .defaultIntent
        ) else {
            throw CoreAIRunnerError.inferenceFailed(
                modelID: "", detail: "cannot create grayscale CGImage")
        }
        return try savePNG(cgImage, prefix: prefix)
    }
}
