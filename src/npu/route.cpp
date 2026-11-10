// The offload decision: the environment switches and the policy, without any device access.

#include "npu/npu.h"

#include "common/log.h"

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

namespace vulkanizer::npu {

namespace {

int64_t envInt(const char* name, int64_t fallback) {
    const char* v = std::getenv(name);
    if (!v || !*v) return fallback;
    char* end = nullptr;
    const long long x = std::strtoll(v, &end, 0);
    return end && *end == 0 && x >= 0 ? int64_t(x) : fallback;
}

double envDouble(const char* name, double fallback) {
    const char* v = std::getenv(name);
    if (!v || !*v) return fallback;
    char* end = nullptr;
    const double x = std::strtod(v, &end);
    return end && *end == 0 && x >= 1 ? x : fallback;
}

}  // namespace

Mode parseMode(const char* value) {
    if (!value || !*value || !std::strcmp(value, "0") || !std::strcmp(value, "off")) return Mode::Off;
    if (!std::strcmp(value, "auto") || !std::strcmp(value, "1")) return Mode::Auto;
    if (!std::strcmp(value, "force")) return Mode::Force;
    VULKANIZER_WARN("VULKANIZER_NPU=%s: expected off, auto or force; NPU offload stays off", value);
    return Mode::Off;
}

Mode mode() {
    static std::mutex mutex;
    static std::string last;
    static Mode m = Mode::Off;
    static bool parsed = false;
    const char* v = std::getenv("VULKANIZER_NPU");
    std::lock_guard<std::mutex> lock(mutex);
    if (!parsed || last != (v ? v : "")) {
        last = v ? v : "";
        m = parseMode(v);
        parsed = true;
    }
    return m;
}

Policy policy() {
    static const Policy p = [] {
        Policy q;
        q.minMnk = envInt("VULKANIZER_NPU_MIN_MNK", q.minMnk);
        q.maxMnk = envInt("VULKANIZER_NPU_MAX_MNK", q.maxMnk);
        q.minDim = envInt("VULKANIZER_NPU_MIN_DIM", q.minDim);
        q.maxPadding = envDouble("VULKANIZER_NPU_MAX_PADDING", q.maxPadding);
        return q;
    }();
    return p;
}

Target route(const GemmShape& s, Mode mode, const Policy& policy, int64_t volume) {
    if (mode == Mode::Off || volume <= 0) return Target::Gpu;
    if (s.m <= 0 || s.n <= 0 || s.k <= 0 || s.batch <= 0) return Target::Gpu;
    if (mode == Mode::Force) return Target::Npu;
    // Auto: small and matrix-vector shaped products stay on the GPU, where the launch is
    // cheaper than the NPU round trip, and so do large ones (the GPU is faster there) and
    // operands the host reads slowly.
    if (!s.cached) return Target::Gpu;
    if (s.m < policy.minDim || s.n < policy.minDim) return Target::Gpu;
    // A kernel much larger than the shape (zero padding) wastes the NPU's time.
    if (double(volume) > policy.maxPadding * double(s.m) * double(s.n) * double(s.k)) return Target::Gpu;
    const double mnk = double(s.m) * double(s.n) * double(s.k) * double(s.batch);
    return mnk >= double(policy.minMnk) && mnk <= double(policy.maxMnk) ? Target::Npu : Target::Gpu;
}

}  // namespace vulkanizer::npu
