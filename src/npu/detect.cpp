// NPU discovery from sysfs: the DRM accel subsystem lists every compute accelerator as
// /sys/class/accel/accelN, its driver and PCI ids under device/. Reads files only, so it works
// without XRT or Level Zero installed.

#include "npu/npu.h"

#include <algorithm>
#include <filesystem>
#include <fstream>

namespace vulkanizer::npu {

namespace {

namespace fs = std::filesystem;

uint32_t readHex(const fs::path& file) {
    std::ifstream in(file);
    std::string s;
    if (!(in >> s)) return 0;
    try {
        return uint32_t(std::stoul(s, nullptr, 16));
    } catch (...) {
        return 0;
    }
}

// The PCI ids known to us: AMD 1502 Phoenix / Hawk Point (XDNA, "NPU1"), 17f0 Strix / Krackan /
// Strix Halo (XDNA2, "NPU4" and later); Intel 7d1d Meteor Lake, ad1d Arrow Lake (NPU 3720),
// 643e Lunar Lake (NPU 4000).
std::string generation(uint32_t vendor, uint32_t device) {
    if (vendor == 0x1022) {
        if (device == 0x1502) return "XDNA1";
        if (device == 0x17f0) return "XDNA2";
    }
    if (vendor == 0x8086) {
        if (device == 0x7d1d || device == 0xad1d) return "NPU3720";
        if (device == 0x643e) return "NPU4000";
    }
    return "";
}

}  // namespace

const char* vendorName(Vendor v) {
    switch (v) {
    case Vendor::AmdXdna: return "AMD XDNA";
    case Vendor::IntelVpu: return "Intel NPU";
    case Vendor::Unknown: break;
    }
    return "unknown";
}

std::vector<DeviceInfo> detect(const std::string& sysfsRoot) {
    std::vector<DeviceInfo> out;
    std::error_code ec;
    const fs::path cls = fs::path(sysfsRoot) / "class" / "accel";
    for (const fs::directory_entry& e : fs::directory_iterator(cls, ec)) {
        const std::string name = e.path().filename().string();
        if (name.rfind("accel", 0) != 0) continue;
        DeviceInfo d;
        d.accel = name;
        d.node = "/dev/accel/" + name;
        const fs::path dev = e.path() / "device";
        d.driver = fs::read_symlink(dev / "driver", ec).filename().string();
        if (ec) d.driver.clear(), ec.clear();
        d.pciVendor = readHex(dev / "vendor");
        d.pciDevice = readHex(dev / "device");
        d.vendor = d.driver == "amdxdna" ? Vendor::AmdXdna : d.driver == "intel_vpu" ? Vendor::IntelVpu : Vendor::Unknown;
        d.generation = generation(d.pciVendor, d.pciDevice);
        out.push_back(std::move(d));
    }
    std::sort(out.begin(), out.end(), [](const DeviceInfo& a, const DeviceInfo& b) {
        return a.accel.size() != b.accel.size() ? a.accel.size() < b.accel.size() : a.accel < b.accel;
    });
    return out;
}

}  // namespace vulkanizer::npu
