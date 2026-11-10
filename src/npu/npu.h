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
#include <mutex>
#include <string>
#include <vector>

namespace vulkanizer::npu {

// VULKANIZER_NPU: off (default until measured on NPU hardware), auto (the policy decides), force
// (everything the backend supports, regardless of size).
enum class Mode { Off, Auto, Force };
Mode parseMode(const char* value);
Mode mode();  // from VULKANIZER_NPU, re-read when it changes (a process may switch it, e.g. the benchmark)

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
    // A, B and C are in cached host memory; false: write-combined (device memory on RADV), whose
    // host reads made the copying path slower than the GPU on every measured shape (plan.md 7.2).
    bool cached = true;
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

// Thresholds of the auto mode, from the measurements in plan.md 7.2 (copying path, XDNA2 vs Radeon
// 890M): the NPU won from cached memory between 256^3 and 512^3, not above. VULKANIZER_NPU_MIN_MNK /
// _MAX_MNK: m * n * k * batch in this range (default 2^24 .. 2^27); VULKANIZER_NPU_MIN_DIM: m and n
// at least this (default 64; matrix-vector products stay on the GPU).
struct Policy {
    int64_t minMnk = int64_t(1) << 24;
    int64_t maxMnk = int64_t(1) << 27;
    int64_t minDim = 64;
    // VULKANIZER_NPU_MAX_PADDING: the backend's padded m * n * k (Backend::volume) at most this many
    // times the shape's (default 2).
    double maxPadding = 2;
};
Policy policy();  // parsed once from the environment

enum class Target { Gpu, Npu };
// The decision for one GEMM; `volume`: the backend's Backend::volume for it (0: not supported).
Target route(const GemmShape& shape, Mode mode, const Policy& policy, int64_t volume);

// Where the time of one gemm() went, in seconds: packing the operands into the device's layout and
// type (incl. zero padding and the copy to the device), the device run, and writing C (copy back,
// alpha / beta).
// Zero-copy operation (plan.md 7.3): the kernel's own buffers, in host memory that the NPU and the GPU
// both map (anonymous pages: the NPU driver takes them as user pointer buffers, Vulkan through
// VK_EXT_external_memory_host). The GPU packs the operands into lhs / rhs (row-major lhs m x k, rhs
// k x n of type `in`, zero padded) and reads the result (m x n of type `out`); for C^T =
// op(B)^T op(A)^T, lhs takes op(B)^T and rhs op(A)^T (see host.h).
struct Staging {
    DType in = DType::BF16, out = DType::F32;
    int64_t m = 0, k = 0, n = 0;
    void *lhs = nullptr, *rhs = nullptr, *result = nullptr;
    size_t lhsBytes = 0, rhsBytes = 0, resultBytes = 0;  // page multiples
};

struct GemmTimes {
    double pack = 0, run = 0, epilogue = 0;
};

class Backend {
public:
    virtual ~Backend() = default;
    virtual const char* name() const = 0;
    // The m * n * k the backend computes per batch entry for this shape, padding included; 0: the
    // shape is not supported.
    virtual int64_t volume(const GemmShape& shape) const = 0;
    bool supports(const GemmShape& shape) const { return volume(shape) > 0; }
    // The breakdown of the last successful gemm() or runStaged() (zeros if the backend does not measure).
    virtual GemmTimes lastTimes() const { return {}; }
    // Zero copy: whether the backend has stage() / runStaged(); stage() picks the kernel for one batch
    // entry of `shape` and returns its buffers (they stay valid for the backend's lifetime), runStaged()
    // runs it on what the caller put into them. The caller serializes stage..runStaged..reading out
    // against other users of the backend through lockStaging().
    virtual bool canStage() const { return false; }
    virtual bool stage(const GemmShape& shape, Staging& out) { return false; }
    virtual bool runStaged(const Staging& staging) { return false; }
    std::mutex& lockStaging() { return stagingMutex_; }
    // Runs the GEMM on host-addressable memory and returns when C is written; false: not done
    // (C untouched), the caller falls back to the GPU.
    virtual bool gemm(const GemmArgs& args) = 0;

private:
    std::mutex stagingMutex_;
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
