// GEMM on the GPU (Vulkanizer cuBLAS) vs the NPU backend (plan.md 7.2), bf16 x bf16 -> f32.
//   vulkanizer-npu-bench                      the default shapes
//   vulkanizer-npu-bench NN 1024 1024 1024 1  one shape: op(A) op(B) m n k batch (cuBLAS, column-major)
//
// GPU: cublasGemmEx / cublasGemmStridedBatchedEx on device memory, timed to cuCtxSynchronize.
// NPU: the backend's gemm() on the host addresses of the same buffers, as the cuBLAS hook does
// (write-combined on RADV: slow host reads), and again on cached pinned host memory; the NPU time is
// split into pack (layout, bf16, padding, copy in), run and epilogue (copy out, alpha / beta). GFLOP/s and
// gpu/np use the faster of the two NPU memories; relerr: max |C_npu - C_gpu| / max |C_gpu|.
// "pad" is the kernel's m * n * k over the shape's. Times are medians, in ms.

#include <cublas_v2.h>
#include <cuda.h>
#include <vulkanizer/vulkanizer_ext.h>

#include "npu/npu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

namespace npu = vulkanizer::npu;

void check(CUresult r, const char* what) {
    if (r == CUDA_SUCCESS) return;
    const char* name = nullptr;
    cuGetErrorName(r, &name);
    std::fprintf(stderr, "%s failed: %s\n", what, name ? name : "?");
    std::exit(1);
}
void check(cublasStatus_t r, const char* what) {
    if (r == CUBLAS_STATUS_SUCCESS) return;
    std::fprintf(stderr, "%s failed: cuBLAS status %d\n", what, int(r));
    std::exit(1);
}
#define CK(call) check((call), #call)

double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0 : v[v.size() / 2];
}

uint16_t toBf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7fff + ((u >> 16) & 1);
    return uint16_t(u >> 16);
}

struct Case {
    std::string label;
    bool transA, transB;
    int m, n, k, batch;
};

struct NpuRun {
    double total = 0;
    npu::GemmTimes t;
    bool ok = false;
};

// Repeats `body` until it ran at least `minRuns` times and for `minSeconds`.
template <class F>
std::vector<double> repeat(F body, int minRuns = 3, double minSeconds = 0.3) {
    std::vector<double> times;
    const double start = now();
    while (int(times.size()) < minRuns || now() - start < minSeconds) {
        const double t0 = now();
        body();
        times.push_back(now() - t0);
        if (times.size() >= 200) break;
    }
    return times;
}

NpuRun runNpu(npu::Backend& backend, const npu::GemmArgs& args) {
    NpuRun r;
    if (!backend.gemm(args)) return r;  // warm-up: loads the kernel
    std::vector<double> pack, run, epi;
    const std::vector<double> total = repeat([&] {
        if (!backend.gemm(args)) return;
        const npu::GemmTimes t = backend.lastTimes();
        pack.push_back(t.pack), run.push_back(t.run), epi.push_back(t.epilogue);
    });
    if (pack.size() != total.size()) return r;
    r.ok = true;
    r.total = median(total);
    r.t = {median(pack), median(run), median(epi)};
    return r;
}

}  // namespace

int main(int argc, char** argv) {
    setenv("VULKANIZER_NPU", "off", 1);  // the cuBLAS calls below are the GPU side
    std::vector<Case> cases;
    if (argc == 6) {
        const std::string ops = argv[1];
        cases.push_back({"custom", ops[0] == 'T', ops.size() > 1 && ops[1] == 'T', std::atoi(argv[2]), std::atoi(argv[3]),
                         std::atoi(argv[4]), std::atoi(argv[5])});
    } else if (argc != 1) {
        std::fprintf(stderr, "usage: %s [NN|NT|TN|TT m n k batch]\n", argv[0]);
        return 2;
    } else {
        for (int s : {256, 512, 1024, 2048, 4096}) cases.push_back({"square", false, false, s, s, s, 1});
        // nn.Linear: C (out x tokens) = W^T x, k = in features
        cases.push_back({"proj 2048->2048, 512 tok", true, false, 2048, 512, 2048, 1});
        cases.push_back({"proj 4096->4096, 512 tok", true, false, 4096, 512, 4096, 1});
        cases.push_back({"proj 4096->4096, 2048 tok", true, false, 4096, 2048, 4096, 1});
        // attention, 32 heads, 512 tokens, head dim 128: S = K^T Q, O = V P
        cases.push_back({"attn QK^T 32 heads", true, false, 512, 512, 128, 32});
        cases.push_back({"attn PV 32 heads", false, false, 128, 512, 512, 32});
    }

    std::vector<npu::DeviceInfo> devices = npu::detect();
    if (devices.empty()) {
        std::fprintf(stderr, "no NPU found\n");
        return 1;
    }
#ifdef VULKANIZER_NPU_XRT
    std::unique_ptr<npu::Backend> backend = npu::createXdnaBackend(devices[0]);
#else
    std::unique_ptr<npu::Backend> backend;
#endif
    if (!backend) {
        std::fprintf(stderr, "no NPU backend (build with -DVULKANIZER_NPU_XRT=ON, kernels in VULKANIZER_NPU_KERNELS)\n");
        return 1;
    }

    CK(cuInit(0));
    CUdevice dev;
    CK(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CK(cuDevicePrimaryCtxRetain(&ctx, dev));
    CK(cuCtxSetCurrent(ctx));
    char devName[256] = {};
    CK(cuDeviceGetName(devName, sizeof devName, dev));
    cublasHandle_t handle;
    CK(cublasCreate(&handle));

    std::printf("GPU: %s, NPU: %s %s (%s); bf16 x bf16 -> f32, median ms\n", devName, npu::vendorName(devices[0].vendor),
                devices[0].generation.c_str(), backend->name());
    std::printf("%-27s %-4s %5s %5s %5s %3s %5s | %8s %7s | %8s %7s | %8s %7s %7s %7s | %7s %6s %8s\n", "case", "op", "m",
                "n", "k", "b", "pad", "gpu", "GFLOP/s", "npu(wc)", "pack", "npu(pin)", "pack", "run", "epi", "GFLOP/s",
                "gpu/np", "relerr");

    std::mt19937 rng(1);
    std::uniform_real_distribution<float> dist(-1, 1);
    for (const Case& c : cases) {
        const int64_t rowsA = c.transA ? c.k : c.m, colsA = c.transA ? c.m : c.k;
        const int64_t rowsB = c.transB ? c.n : c.k, colsB = c.transB ? c.k : c.n;
        const int lda = int(rowsA), ldb = int(rowsB), ldc = c.m;
        const int64_t sA = rowsA * colsA, sB = rowsB * colsB, sC = int64_t(c.m) * c.n;
        std::vector<uint16_t> a(sA * c.batch), b(sB * c.batch);
        for (uint16_t& x : a) x = toBf16(dist(rng));
        for (uint16_t& x : b) x = toBf16(dist(rng));
        const size_t bytesA = a.size() * 2, bytesB = b.size() * 2, bytesC = size_t(sC * c.batch) * 4;
        CUdeviceptr dA, dB, dC;
        CK(cuMemAlloc(&dA, bytesA));
        CK(cuMemAlloc(&dB, bytesB));
        CK(cuMemAlloc(&dC, bytesC));
        CK(cuMemcpyHtoD(dA, a.data(), bytesA));
        CK(cuMemcpyHtoD(dB, b.data(), bytesB));
        CK(cuMemsetD8(dC, 0, bytesC));

        const float alpha = 1, beta = 0;
        const auto gpu = [&] {
            const cublasOperation_t ta = c.transA ? CUBLAS_OP_T : CUBLAS_OP_N, tb = c.transB ? CUBLAS_OP_T : CUBLAS_OP_N;
            if (c.batch == 1)
                CK(cublasGemmEx(handle, ta, tb, c.m, c.n, c.k, &alpha, reinterpret_cast<void*>(dA), CUDA_R_16BF, lda,
                                reinterpret_cast<void*>(dB), CUDA_R_16BF, ldb, &beta, reinterpret_cast<void*>(dC),
                                CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
            else
                CK(cublasGemmStridedBatchedEx(handle, ta, tb, c.m, c.n, c.k, &alpha, reinterpret_cast<void*>(dA),
                                              CUDA_R_16BF, lda, sA, reinterpret_cast<void*>(dB), CUDA_R_16BF, ldb, sB,
                                              &beta, reinterpret_cast<void*>(dC), CUDA_R_32F, ldc, sC, c.batch,
                                              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
            CK(cuCtxSynchronize());
        };
        gpu();
        const double gpuTime = median(repeat(gpu));
        std::vector<float> gpuC(sC * c.batch);
        CK(cuMemcpyDtoH(gpuC.data(), dC, bytesC));

        npu::GemmArgs args;
        args.shape = {c.transA, c.transB, c.m, c.n, c.k, c.batch, npu::DType::BF16, npu::DType::BF16, npu::DType::F32, false};
        args.alpha = alpha, args.beta = beta;
        args.lda = lda, args.ldb = ldb, args.ldc = ldc;
        args.strideA = sA, args.strideB = sB, args.strideC = sC;
        const int64_t volume = backend->volume(args.shape);
        const double pad = double(volume) / (double(c.m) * c.n * c.k);
        const double flop = 2.0 * c.m * c.n * c.k * c.batch;

        NpuRun wc, pinned;
        double relErr = -1;
        if (volume > 0) {
            int cached = 0;
            void *hA, *hB, *hC;
            CK(vulkanizerGetHostPointer(dA, bytesA, &hA, &cached));
            CK(vulkanizerGetHostPointer(dB, bytesB, &hB, &cached));
            CK(vulkanizerGetHostPointer(dC, bytesC, &hC, &cached));
            args.a = hA, args.b = hB, args.c = hC;
            wc = runNpu(*backend, args);
            if (wc.ok) {
                std::vector<float> npuC(sC * c.batch);
                std::memcpy(npuC.data(), hC, bytesC);
                double maxErr = 0, maxRef = 0;
                for (size_t i = 0; i < npuC.size(); ++i) {
                    maxErr = std::max(maxErr, double(std::fabs(npuC[i] - gpuC[i])));
                    maxRef = std::max(maxRef, double(std::fabs(gpuC[i])));
                }
                relErr = maxRef > 0 ? maxErr / maxRef : maxErr;
            }
            void *pA, *pB, *pC;
            CK(cuMemAllocHost(&pA, bytesA));
            CK(cuMemAllocHost(&pB, bytesB));
            CK(cuMemAllocHost(&pC, bytesC));
            std::memcpy(pA, a.data(), bytesA);
            std::memcpy(pB, b.data(), bytesB);
            args.a = pA, args.b = pB, args.c = pC;
            pinned = runNpu(*backend, args);
            CK(cuMemFreeHost(pA));
            CK(cuMemFreeHost(pB));
            CK(cuMemFreeHost(pC));
        }
        CK(cuMemFree(dA));
        CK(cuMemFree(dB));
        CK(cuMemFree(dC));

        std::printf("%-27s %c%c   %5d %5d %5d %3d ", c.label.c_str(), c.transA ? 'T' : 'N', c.transB ? 'T' : 'N', c.m,
                    c.n, c.k, c.batch);
        if (volume > 0)
            std::printf("%5.1f ", pad);
        else
            std::printf("%5s ", "-");
        std::printf("| %8.3f %7.0f | ", gpuTime * 1e3, flop / gpuTime / 1e9);
        if (!wc.ok) {
            std::printf("%s\n", volume > 0 ? "npu failed" : "no npu kernel");
            continue;
        }
        const double best = std::min(wc.total, pinned.ok ? pinned.total : wc.total);
        std::printf("%8.3f %7.3f | ", wc.total * 1e3, wc.t.pack * 1e3);
        if (pinned.ok)
            std::printf("%8.3f %7.3f %7.3f %7.3f | ", pinned.total * 1e3, pinned.t.pack * 1e3, pinned.t.run * 1e3,
                        pinned.t.epilogue * 1e3);
        else
            std::printf("%8s %7s %7s %7s | ", "-", "-", "-", "-");
        std::printf("%7.0f %6.2f %8.1e\n", flop / best / 1e9, gpuTime / best, relErr);
    }
    cublasDestroy(handle);
    return 0;
}
