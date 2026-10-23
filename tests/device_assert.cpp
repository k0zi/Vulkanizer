// Device-side assert() (data/assert.ptx): CUDA_ERROR_ASSERT at the synchronization after the failing
// launch, sticky for later launches, copies and queries, cleared by cuDevicePrimaryCtxReset.

#include "test_util.h"

#include <vector>

namespace {

CUresult runCheck(CUmodule mod, CUdeviceptr data, int n) {
    CUfunction f;
    CHECK_CU(cuModuleGetFunction(&f, mod, "check"));
    void* args[] = {&data, &n};
    return cuLaunchKernel(f, 4, 1, 1, 16, 1, 1, 0, nullptr, args, nullptr);
}

}  // namespace

int main() {
    CHECK_CU(cuInit(0));
    CUdevice dev;
    CHECK_CU(cuDeviceGet(&dev, 0));
    for (int round = 0; round < 2; ++round) {  // the second round runs after the reset
        CUcontext ctx;
        CHECK_CU(cuDevicePrimaryCtxRetain(&ctx, dev));
        CHECK_CU(cuCtxSetCurrent(ctx));
        CUmodule mod;
        CHECK_CU(cuModuleLoad(&mod, VULKANIZER_TEST_DATA_DIR "/assert.ptx"));
        std::vector<int> host(64, -1);
        CUdeviceptr data;
        CHECK_CU(cuMemAlloc(&data, host.size() * sizeof(int)));
        CHECK_CU(cuMemcpyHtoD(data, host.data(), host.size() * sizeof(int)));

        CHECK_CU(runCheck(mod, data, 64));  // nothing asserts
        CHECK_CU(cuCtxSynchronize());
        CHECK_CU(cuStreamQuery(nullptr));

        host[2 * 16 + 5] = 1;  // thread 37 checks 37 < 30
        CHECK_CU(cuMemcpyHtoD(data, host.data(), host.size() * sizeof(int)));
        CHECK_CU(runCheck(mod, data, 30));
        EXPECT_CU(CUDA_ERROR_ASSERT, cuCtxSynchronize());
        EXPECT_CU(CUDA_ERROR_ASSERT, cuStreamSynchronize(nullptr));
        EXPECT_CU(CUDA_ERROR_ASSERT, cuStreamQuery(nullptr));
        EXPECT_CU(CUDA_ERROR_ASSERT, cuMemcpyDtoH(host.data(), data, sizeof(int)));
        EXPECT_CU(CUDA_ERROR_ASSERT, runCheck(mod, data, 64));
        const char* name = nullptr;
        CHECK_CU(cuGetErrorName(CUDA_ERROR_ASSERT, &name));
        EXPECT(name && std::string(name) == "CUDA_ERROR_ASSERT");

        CHECK_CU(cuDevicePrimaryCtxRelease(dev));
        CHECK_CU(cuDevicePrimaryCtxReset(dev));
    }
    std::printf("OK\n");
    return 0;
}
