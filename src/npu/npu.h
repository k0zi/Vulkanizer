#pragma once

// NPU offload (plan.md 7.): fixed-function work that an NPU accelerates (for now the cuBLAS GEMMs)
// goes to the machine's NPU instead of the Vulkan device when one is present and the policy
// says it pays off. Everything else, and every machine without an NPU, stays on the GPU path.
//
// The layers: detect() finds the NPUs in sysfs (no driver library needed), route() is the pure
// decision, and a Backend runs the work (xdna: AMD XDNA through XRT, built with
// -DVULKANIZER_NPU_XRT=ON; cpu: a host reference that exercises the same path in the tests).

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vulkanizer::npu {

// VULKANIZER_NPU: off (default until measured on NPU hardware), auto (the policy decides), force
// (everything the backend supports, regardless of size).
enum class Mode { Off, Auto, Force };
Mode parseMode(const char* value);
Mode mode();  // parsed once from VULKANIZER_NPU

enum class Vendor { AmdXdna, IntelVpu, Unknown };

struct DeviceInfo {
    Vendor vendor = Vendor::Unknown;
    std::string accel;   // accel0
    std::string node;    // /dev/accel/accel0
    std::string driver;  // amdxdna, intel_vpu
    uint32_t pciVendor = 0, pciDevice = 0;
    std::string generation;  // XDNA1, XDNA2, NPU3720, ... ("" if the PCI id is not known)
};

// The NPUs under <sysfsRoot>/class/accel (DRM accel subsystem: amdxdna, intel_vpu). A root other
// than /sys is for tests.
std::vector<DeviceInfo> detect(const std::string& sysfsRoot = "/sys");
const char* vendorName(Vendor v);

enum class DType { F32, F16, BF16 };

// One cuBLAS GEMM (column-major, like the API): C = alpha op(A) op(B) + beta C, op(A) m x k.
struct GemmShape {
    bool transA = false, transB = false;
    int64_t m = 0, n = 0, k = 0, batch = 1;
    DType a = DType::F32, b = DType::F32, c = DType::F32;
    // The caller accepts bf16 products for f32 inputs (CUBLAS_COMPUTE_32F_FAST_16BF).
    bool reducedPrecision = false;
};

struct GemmArgs {
    GemmShape shape;
    float alpha = 1, beta = 0;
    const void* a = nullptr;
    int64_t lda = 0, strideA = 0;
    const void* b = nullptr;
    int64_t ldb = 0, strideB = 0;
    void* c = nullptr;
    int64_t ldc = 0, strideC = 0;
};

// Thresholds of the auto mode. VULKANIZER_NPU_MIN_MNK: m * n * k * batch at least this (default
// 2^27, a 512^3 GEMM); VULKANIZER_NPU_MIN_DIM: m and n at least this (default 64; matrix-vector
// products stay on the GPU).
struct Policy {
    int64_t minMnk = int64_t(1) << 27;
    int64_t minDim = 64;
};
Policy policy();  // parsed once from the environment

enum class Target { Gpu, Npu };
// The decision for one GEMM; `supported`: the backend can run this shape.
Target route(const GemmShape& shape, Mode mode, const Policy& policy, bool supported);

class Backend {
public:
    virtual ~Backend() = default;
    virtual const char* name() const = 0;
    virtual bool supports(const GemmShape& shape) const = 0;
    // Runs the GEMM on host-addressable memory and returns when C is written; false: not done
    // (C untouched), the caller falls back to the GPU.
    virtual bool gemm(const GemmArgs& args) = 0;
};

// The process's backend, created on first use: nullptr with VULKANIZER_NPU=off, without an NPU
// or without a backend for it. VULKANIZER_NPU_BACKEND picks one: auto (default: by the detected
// NPU), xdna, cpu (host reference, for tests; needs no NPU).
Backend* backend();

std::unique_ptr<Backend> createCpuBackend();
#ifdef VULKANIZER_NPU_XRT
std::unique_ptr<Backend> createXdnaBackend(const DeviceInfo& device);
#endif

}  // namespace vulkanizer::npu
