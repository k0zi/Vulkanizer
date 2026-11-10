// Backend selection and the host reference backend.

#include "npu/host.h"
#include "npu/npu.h"

#include "common/log.h"

#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

namespace vulkanizer::npu {

namespace {

// The GEMM on the CPU through the same packing and epilogue as the NPU backends: tests the
// cuBLAS hook, the host pointers and the layout conversion on machines without an NPU.
class CpuBackend final : public Backend {
public:
    const char* name() const override { return "cpu"; }
    bool supports(const GemmShape&) const override { return true; }

    bool gemm(const GemmArgs& g) override {
        const GemmShape& s = g.shape;
        std::lock_guard<std::mutex> lock(mutex_);
        lhs_.resize(size_t(s.n * s.k));
        rhs_.resize(size_t(s.k * s.m));
        out_.resize(size_t(s.n * s.m));
        const size_t sa = dtypeSize(s.a), sb = dtypeSize(s.b), sc = dtypeSize(s.c);
        for (int64_t i = 0; i < s.batch; ++i) {
            const auto* a = static_cast<const char*>(g.a) + i * g.strideA * int64_t(sa);
            const auto* b = static_cast<const char*>(g.b) + i * g.strideB * int64_t(sb);
            auto* c = static_cast<char*>(g.c) + i * g.strideC * int64_t(sc);
            packTransposed(b, s.b, g.ldb, s.transB, s.k, s.n, lhs_.data(), DType::F32, s.k);  // op(B)^T, n x k
            packTransposed(a, s.a, g.lda, s.transA, s.m, s.k, rhs_.data(), DType::F32, s.m);  // op(A)^T, k x m
            std::fill(out_.begin(), out_.end(), 0.0f);
            for (int64_t r = 0; r < s.n; ++r)
                for (int64_t p = 0; p < s.k; ++p) {
                    const float l = lhs_[size_t(r * s.k + p)];
                    const float* rr = &rhs_[size_t(p * s.m)];
                    float* o = &out_[size_t(r * s.m)];
                    for (int64_t q = 0; q < s.m; ++q) o[q] += l * rr[q];
                }
            epilogue(out_.data(), DType::F32, s.m, g.alpha, g.beta, c, s.c, g.ldc, s.m, s.n);
        }
        return true;
    }

private:
    std::mutex mutex_;
    std::vector<float> lhs_, rhs_, out_;
};

std::unique_ptr<Backend> create() {
    if (mode() == Mode::Off) return nullptr;
    const char* env = std::getenv("VULKANIZER_NPU_BACKEND");
    const std::string choice = env && *env ? env : "auto";
    if (choice == "cpu") {
        VULKANIZER_INFO("npu: cpu reference backend");
        return createCpuBackend();
    }
    if (choice != "auto" && choice != "xdna") {
        VULKANIZER_WARN("VULKANIZER_NPU_BACKEND=%s: expected auto, xdna or cpu", choice.c_str());
        return nullptr;
    }
    for (const DeviceInfo& d : detect()) {
        VULKANIZER_INFO("npu: %s %s (%s, %04x:%04x, %s)", vendorName(d.vendor), d.generation.c_str(), d.node.c_str(),
                        d.pciVendor, d.pciDevice, d.driver.c_str());
        if (d.vendor == Vendor::AmdXdna) {
#ifdef VULKANIZER_NPU_XRT
            if (auto b = createXdnaBackend(d)) return b;
#else
            VULKANIZER_WARN("npu: %s found, but Vulkanizer was built without XRT (-DVULKANIZER_NPU_XRT=ON)", d.node.c_str());
#endif
        } else if (d.vendor == Vendor::IntelVpu && choice == "auto") {
            VULKANIZER_WARN("npu: %s: no Intel NPU backend yet (plan.md 7.)", d.node.c_str());
        }
    }
    VULKANIZER_INFO("npu: no usable NPU; everything runs on the GPU");
    return nullptr;
}

}  // namespace

std::unique_ptr<Backend> createCpuBackend() { return std::make_unique<CpuBackend>(); }

Backend* backend() {
    static const std::unique_ptr<Backend> b = create();
    return b.get();
}

}  // namespace vulkanizer::npu
