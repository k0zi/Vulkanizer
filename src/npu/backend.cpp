// Backend selection and the host reference backend.

#include "npu/host.h"
#include "npu/npu.h"

#include "common/log.h"

#include <algorithm>
#include <array>
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
    int64_t volume(const GemmShape& s) const override { return s.m * s.n * s.k; }

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

    // Zero copy on the host: f32 buffers of exactly the shape (lhs n x k, rhs k x m, result n x m),
    // grown as needed. Buffers handed out earlier stay valid: they are kept until the backend goes.
    bool canStage() const override { return true; }

    bool stage(const GemmShape& s, Staging& out) override {
        std::lock_guard<std::mutex> lock(mutex_);
        const size_t lhs = size_t(s.n * s.k) * 4, rhs = size_t(s.k * s.m) * 4, result = size_t(s.n * s.m) * 4;
        if (staged_.empty() || staged_.back()[0].size() < lhs || staged_.back()[1].size() < rhs ||
            staged_.back()[2].size() < result)
            staged_.push_back({HostPages(lhs), HostPages(rhs), HostPages(result)});
        const auto& b = staged_.back();
        out = {DType::F32, DType::F32, s.n, s.k, s.m, b[0].data(), b[1].data(), b[2].data(), b[0].size(), b[1].size(),
               b[2].size()};
        return true;
    }

    bool runStaged(const Staging& st) override {
        // VULKANIZER_NPU_CPU_FAIL_AT=n (tests): the n-th run fails, for the GPU fallback of a batch's rest
        if (const char* f = std::getenv("VULKANIZER_NPU_CPU_FAIL_AT"); f && ++runs_ == std::atoi(f)) return false;
        const auto* l = static_cast<const float*>(st.lhs);
        const auto* r = static_cast<const float*>(st.rhs);
        auto* o = static_cast<float*>(st.result);
        for (int64_t i = 0; i < st.m; ++i) {
            float* row = o + i * st.n;
            std::fill(row, row + st.n, 0.0f);
            for (int64_t p = 0; p < st.k; ++p) {
                const float x = l[i * st.k + p];
                const float* rr = r + p * st.n;
                for (int64_t j = 0; j < st.n; ++j) row[j] += x * rr[j];
            }
        }
        return true;
    }

private:
    std::mutex mutex_;
    std::vector<float> lhs_, rhs_, out_;
    std::vector<std::array<HostPages, 3>> staged_;
    int runs_ = 0;
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
