// DeviceInfo.swift — hardware introspection for health checks and device gating.
//
// Reports chip name, memory, thermal state, and OS version — the data the
// /v1/health endpoint returns so clients can make informed decisions.

import Foundation
import ProcessInfo

#if canImport(UIKit)
import UIKit
#endif

public struct DeviceInfo: Sendable {
    public let deviceName: String       // "MacBook Pro" / "iPhone 17 Pro"
    public let chipName: String          // "Apple M4 Pro" / "A19 Pro"
    public let memoryTotalGB: Double
    public let macosVersion: String      // "26.6" (also used for iOS)
    public let coreaiVersion: String
    public let thermalState: String      // "nominal" | "fair" | "serious" | "critical"

    /// Gather device info at call time (thermal state changes during a session).
    public static func current() -> DeviceInfo {
        let processInfo = ProcessInfo.processInfo

        let deviceName: String
        let chipName: String
        let memoryTotalGB: Double

        #if os(macOS)
        deviceName = macDeviceName()
        chipName = macChipName()
        memoryTotalGB = Double(processInfo.physicalMemory) / 1_073_741_824.0
        #elseif canImport(UIKit)
        deviceName = UIDevice.current.name
        chipName = iosChipName()
        memoryTotalGB = Double(processInfo.physicalMemory) / 1_073_741_824.0
        #else
        deviceName = "Unknown"
        chipName = "Unknown"
        memoryTotalGB = Double(processInfo.physicalMemory) / 1_073_741_824.0
        #endif

        let osVersion = "\(processInfo.operatingSystemVersion.majorVersion).\(processInfo.operatingSystemVersion.minorVersion)"

        return DeviceInfo(
            deviceName: deviceName,
            chipName: chipName,
            memoryTotalGB: memoryTotalGB,
            macosVersion: osVersion,
            coreaiVersion: osVersion,  // Core AI ships with the OS
            thermalState: thermalStateString(processInfo.thermalState)
        )
    }

    /// Available memory in GB (approximate — uses virtual memory stats).
    public var memoryAvailableGB: Double {
        var vmStats = vm_statistics64_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &vmStats) { ptr -> kern_return_t in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return 0 }
        let pageSize = Double(vm_kernel_page_size)
        let free = Double(vmStats.free_count) * pageSize / 1_073_741_824.0
        return free
    }

    // MARK: - Private helpers

    private static func thermalStateString(_ state: ProcessInfo.ThermalState) -> String {
        switch state {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    #if os(macOS)
    private static func macDeviceName() -> String {
        // Use sysctl to get the marketing name (e.g. "MacBook Pro")
        let model = sysctlString("hw.model") ?? "Mac"
        return model
    }

    private static func macChipName() -> String {
        // machdep.cpu.brand_string gives "Apple M4 Pro" on Apple Silicon
        return sysctlString("machdep.cpu.brand_string") ?? "Apple Silicon"
    }
    #endif

    #if canImport(UIKit)
    private static func iosChipName() -> String {
        let machine = sysctlString("hw.machine") ?? ""
        // Map common machine identifiers to marketing names
        // (Not exhaustive — fallback to the raw identifier)
        let chipMap: [String: String] = [
            "iPhone17,1": "A18 Pro",
            "iPhone17,2": "A18 Pro",
            "iPhone17,3": "A18",
            "iPhone17,4": "A18 Pro",
            "iPhone17,5": "A18 Pro",
        ]
        return chipMap[machine] ?? machine
    }
    #endif

    private static func sysctlString(_ name: String) -> String? {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return nil }
        var buffer = [CChar](repeating: 0, count: size)
        guard sysctlbyname(name, &buffer, &size, nil, 0) == 0 else { return nil }
        return String(cString: buffer)
    }
}
