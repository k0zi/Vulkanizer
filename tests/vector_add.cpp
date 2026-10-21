// End-to-end: an unmodified Driver API program launching an nvcc-style PTX kernel.

#include "test_util.h"

#include <cmath>
#include <vector>

int main() {
    CHECK_CU(cuInit(0));
    CUdevice dev;
    CHECK_CU(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CHECK_CU(cuCtxCreate(&ctx, 0, dev));

    CUmodule mod;
    CHECK_CU(cuModuleLoad(&mod, VULKANIZER_TEST_DATA_DIR "/vector_add.ptx"));
    CUfunction fn;
    CHECK_CU(cuModuleGetFunction(&fn, mod, "vecAdd"));

    int n = 1'000'003;  // deliberately not a multiple of the block size
    std::vector<float> a(n), b(n), c(n);
    for (int i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = 0.5f * static_cast<float>(i);
    }

    CUdeviceptr da, db, dc;
    CHECK_CU(cuMemAlloc(&da, n * sizeof(float)));
    CHECK_CU(cuMemAlloc(&db, n * sizeof(float)));
    CHECK_CU(cuMemAlloc(&dc, n * sizeof(float)));
    CHECK_CU(cuMemcpyHtoD(da, a.data(), n * sizeof(float)));
    CHECK_CU(cuMemcpyHtoD(db, b.data(), n * sizeof(float)));
    CHECK_CU(cuMemsetD32(dc, 0, n));

    const unsigned block = 256, grid = (n + block - 1) / block;
    void* args[] = {&da, &db, &dc, &n};
    CHECK_CU(cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, nullptr, args, nullptr));
    CHECK_CU(cuCtxSynchronize());
    CHECK_CU(cuMemcpyDtoH(c.data(), dc, n * sizeof(float)));

    for (int i = 0; i < n; ++i) {
        if (std::fabs(c[i] - 1.5f * static_cast<float>(i)) > 1e-3f * (1 + i)) {
            std::fprintf(stderr, "mismatch at %d: %f\n", i, c[i]);
            return 1;
        }
    }

    CHECK_CU(cuMemFree(da));
    CHECK_CU(cuMemFree(db));
    CHECK_CU(cuMemFree(dc));
    CHECK_CU(cuModuleUnload(mod));
    CHECK_CU(cuCtxDestroy(ctx));
    std::printf("OK\n");
    return 0;
}
