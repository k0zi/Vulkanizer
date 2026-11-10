// The offload decision: the environment switches and the policy, without any device access.

#include "npu/npu.h"

#include "common/log.h"

#include <cstdlib>
#include <cstring>

namespace vulkanizer::npu {

namespace {

int64_t envInt(const char* name, int64_t fallback) {
    const char* v = std::getenv(name);
    if (!v || !*v) return fallback;
    char* end = nullptr;
    const long long x = std::strtoll(v, &end, 0);
    return end && *end == 0 && x >= 0 ? int64_t(x) : fallback;
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
    static const Mode m = parseMode(std::getenv("VULKANIZER_NPU"));
    return m;
}

Policy policy() {
    static const Policy p = [] {
        Policy q;
        q.minMnk = envInt("VULKANIZER_NPU_MIN_MNK", q.minMnk);
        q.minDim = envInt("VULKANIZER_NPU_MIN_DIM", q.minDim);
        return q;
    }();
    return p;
}

Target route(const GemmShape& s, Mode mode, const Policy& policy, bool supported) {
    if (mode == Mode::Off || !supported) return Target::Gpu;
    if (s.m <= 0 || s.n <= 0 || s.k <= 0 || s.batch <= 0) return Target::Gpu;
    if (mode == Mode::Force) return Target::Npu;
    // Auto: small and matrix-vector shaped products stay on the GPU, where the launch is
    // cheaper than the NPU round trip.
    if (s.m < policy.minDim || s.n < policy.minDim) return Target::Gpu;
    const double mnk = double(s.m) * double(s.n) * double(s.k) * double(s.batch);
    return mnk >= double(policy.minMnk) ? Target::Npu : Target::Gpu;
}

}  // namespace vulkanizer::npu
