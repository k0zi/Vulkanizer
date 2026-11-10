// Transfer bandwidth and allocation rate through the CUDA Driver API (like NVIDIA's bandwidthTest).
//   vulkanizer-bandwidth [MiB]   copy size, default 256

#include <cuda.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

void check(CUresult r, const char* what) {
    if (r == CUDA_SUCCESS) return;
    const char* name = nullptr;
    cuGetErrorName(r, &name);
    std::fprintf(stderr, "%s failed: %s\n", what, name ? name : "?");
    std::exit(1);
}
#define CK(call) check((call), #call)

double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Best of a few runs of `body`, in GB/s for `bytes` moved per run.
template <class F>
double rate(size_t bytes, F body) {
    double best = 1e30;
    for (int i = 0; i < 5; ++i) {
        const double t0 = now();
        body();
        CK(cuCtxSynchronize());
        best = std::min(best, now() - t0);
    }
    return double(bytes) / best / 1e9;
}

}  // namespace

int main(int argc, char** argv) {
    const size_t bytes = size_t(argc > 1 ? std::atoi(argv[1]) : 256) << 20;
    CK(cuInit(0));
    CUdevice dev;
    CK(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CK(cuDevicePrimaryCtxRetain(&ctx, dev));
    CK(cuCtxSetCurrent(ctx));

    CUdeviceptr d0, d1;
    CK(cuMemAlloc(&d0, bytes));
    CK(cuMemAlloc(&d1, bytes));
    std::vector<char> pageable(bytes, 1);
    void* pinned = nullptr;
    CK(cuMemAllocHost(&pinned, bytes));
    std::memset(pinned, 1, bytes);

    std::printf("%zu MiB copies, GB/s (best of 5)\n", bytes >> 20);
    std::printf("  HtoD pageable  %8.2f\n", rate(bytes, [&] { CK(cuMemcpyHtoD(d0, pageable.data(), bytes)); }));
    std::printf("  HtoD pinned    %8.2f\n", rate(bytes, [&] { CK(cuMemcpyHtoD(d0, pinned, bytes)); }));
    std::printf("  DtoH pageable  %8.2f\n", rate(bytes, [&] { CK(cuMemcpyDtoH(pageable.data(), d0, bytes)); }));
    std::printf("  DtoH pinned    %8.2f\n", rate(bytes, [&] { CK(cuMemcpyDtoH(pinned, d0, bytes)); }));
    std::printf("  DtoD           %8.2f\n", rate(2 * bytes, [&] { CK(cuMemcpyDtoD(d1, d0, bytes)); }));

    // Allocation churn: the pattern of frameworks without a caching allocator.
    for (size_t size : {size_t(4) << 10, size_t(1) << 20, size_t(16) << 20}) {
        constexpr int kCount = 200;
        std::vector<CUdeviceptr> ptrs(kCount);
        const double t0 = now();
        for (auto& p : ptrs) CK(cuMemAlloc(&p, size));
        for (auto p : ptrs) CK(cuMemFree(p));
        std::printf("  alloc+free %5zu KiB  %8.1f us\n", size >> 10, (now() - t0) / kCount * 1e6);
    }

    CK(cuMemFreeHost(pinned));
    CK(cuMemFree(d1));
    CK(cuMemFree(d0));
    CK(cuDevicePrimaryCtxRelease(dev));
    return 0;
}
