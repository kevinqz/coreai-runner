// DetectionL4Tests.swift — on-device conformance (L4): the real Swift runner code path
// (DetectionAdapter → CoreAIKitVision.ObjectDetector → CoreAI) executes a REAL `.aimodel`
// on real Apple hardware and returns detections. Opt-in: only runs when RFDETR_BUNDLE (a
// local rf-detr `.aimodel` bundle) and RFDETR_IMAGE (a test image) are set, so normal CI —
// which lacks both the macOS 27 SDK and the model bundle — skips it.
//
// Reproduce:
//   RFDETR_BUNDLE=/path/to/rfdetr-nano_float32.aimodel \
//   RFDETR_IMAGE=/path/to/demo_coco_cats.jpg \
//   DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer swift test

import Foundation
import Testing
import CoreAIRunner

private let _l4Enabled = ProcessInfo.processInfo.environment["RFDETR_BUNDLE"] != nil
    && ProcessInfo.processInfo.environment["RFDETR_IMAGE"] != nil

@Test(.enabled(if: _l4Enabled))
func rfdetrNanoExecutesRealAimodelOnDevice() async throws {
    let env = ProcessInfo.processInfo.environment
    let bundle = URL(fileURLWithPath: env["RFDETR_BUNDLE"]!)
    let imagePath = env["RFDETR_IMAGE"]!

    // Real Swift runner adapter loads a REAL .aimodel from disk (no catalog/network).
    let adapter = try await DetectionAdapter(modelID: "rf-detr-nano", bundleAt: bundle)

    // Real CoreAI inference on real hardware.
    let out = try await adapter.predict(
        AdapterInput(modelID: "rf-detr-nano", imagePath: imagePath, scoreThreshold: 0.5))

    let dets = out.detections ?? []
    print("L4 rf-detr-nano detections: \(dets.count)")
    for d in dets {
        print(String(format: "  %@  score=%.3f  bbox=[%.1f, %.1f, %.1f, %.1f]",
                     d.label, d.score, d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3]))
    }

    // The model actually ran and produced structured detections — that is the L4 claim.
    #expect(!dets.isEmpty)
    for d in dets {
        #expect(d.score >= 0.5)
        #expect(d.bbox.count == 4)
    }
}
