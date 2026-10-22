// cuInit, device enumeration, attributes and entry point lookup.

#include "test_util.h"

int main() {
    int count = -1;
    EXPECT_CU(CUDA_ERROR_NOT_INITIALIZED, cuDeviceGetCount(&count));
    EXPECT_CU(CUDA_ERROR_INVALID_VALUE, cuInit(1));
    CHECK_CU(cuInit(0));
    CHECK_CU(cuInit(0));  // idempotent

    int version = 0;
    CHECK_CU(cuDriverGetVersion(&version));
    EXPECT(version >= 12000);

    CHECK_CU(cuDeviceGetCount(&count));
    EXPECT(count > 0);

    CUdevice dev;
    EXPECT_CU(CUDA_ERROR_INVALID_DEVICE, cuDeviceGet(&dev, count));
    CHECK_CU(cuDeviceGet(&dev, 0));

    char name[256];
    CHECK_CU(cuDeviceGetName(name, sizeof(name), dev));
    size_t totalMem = 0;
    CHECK_CU(cuDeviceTotalMem(&totalMem, dev));
    EXPECT(totalMem > 0);

    int major = 0, minor = 0, warp = 0, maxThreads = 0, smem = 0, sms = 0;
    CHECK_CU(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
    CHECK_CU(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));
    CHECK_CU(cuDeviceGetAttribute(&warp, CU_DEVICE_ATTRIBUTE_WARP_SIZE, dev));
    CHECK_CU(cuDeviceGetAttribute(&maxThreads, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, dev));
    CHECK_CU(cuDeviceGetAttribute(&smem, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, dev));
    CHECK_CU(cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
    EXPECT(warp == 32);
    EXPECT(maxThreads >= 256);
    EXPECT(major >= 3);

    CUuuid uuid;
    CHECK_CU(cuDeviceGetUuid(&uuid, dev));

    std::printf("device 0: %s\n", name);
    std::printf("  compute capability %d.%d, %d SMs, warp %d\n", major, minor, sms, warp);
    std::printf("  %zu MiB global memory, %d KiB shared/block, %d threads/block\n", totalMem >> 20, smem / 1024,
                maxThreads);

    // The CUDA runtime resolves all driver entry points this way.
    void* fn = nullptr;
    CUdriverProcAddressQueryResult status;
    CHECK_CU(cuGetProcAddress("cuMemAlloc", &fn, 12040, CU_GET_PROC_ADDRESS_DEFAULT, &status));
    EXPECT(fn == reinterpret_cast<void*>(&cuMemAlloc_v2));
    EXPECT(status == CU_GET_PROC_ADDRESS_SUCCESS);
    CHECK_CU(cuGetProcAddress("cuInit", &fn, 12040, CU_GET_PROC_ADDRESS_DEFAULT, &status));
    EXPECT(fn == reinterpret_cast<void*>(&cuInit));
    EXPECT_CU(CUDA_ERROR_NOT_FOUND,
              cuGetProcAddress("cuDoesNotExist", &fn, 12040, CU_GET_PROC_ADDRESS_DEFAULT, &status));
    EXPECT(status == CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND);
    // ABI variants by the caller's CUDA version: CUDA 13 passes a context to cuCtxSynchronize
    CHECK_CU(cuGetProcAddress("cuCtxSynchronize", &fn, 12090, CU_GET_PROC_ADDRESS_DEFAULT, &status));
    EXPECT(fn == reinterpret_cast<void*>(&cuCtxSynchronize));
    CHECK_CU(cuGetProcAddress("cuCtxSynchronize", &fn, 13000, CU_GET_PROC_ADDRESS_DEFAULT, &status));
    EXPECT(fn == reinterpret_cast<void*>(&cuCtxSynchronize_v2));
    CHECK_CU(cuGetProcAddress("cuCtxCreate", &fn, 13000, CU_GET_PROC_ADDRESS_DEFAULT, &status));
    EXPECT(fn == reinterpret_cast<void*>(&cuCtxCreate_v4));
    CHECK_CU(cuGetProcAddress("cuCtxCreate", &fn, 11000, CU_GET_PROC_ADDRESS_DEFAULT, &status));
    EXPECT(fn == reinterpret_cast<void*>(&cuCtxCreate_v2));
    CHECK_CU(cuGetProcAddress("cuEventElapsedTime", &fn, 12000, CU_GET_PROC_ADDRESS_DEFAULT, &status));
    EXPECT(fn == reinterpret_cast<void*>(&cuEventElapsedTime));
    CUcontext primary;  // the CUDA 13 variants work on a given context, current or not
    CHECK_CU(cuDevicePrimaryCtxRetain(&primary, dev));
    CUdevice viaV2 = -1;
    CHECK_CU(cuCtxGetDevice_v2(&viaV2, primary));
    EXPECT(viaV2 == dev);
    CHECK_CU(cuCtxSynchronize_v2(primary));
    EXPECT_CU(CUDA_ERROR_INVALID_CONTEXT, cuCtxGetDevice_v2(&viaV2, nullptr));  // none is current
    CHECK_CU(cuDevicePrimaryCtxRelease(dev));
    // a variant Vulkanizer lacks is not found rather than an older signature
    EXPECT_CU(CUDA_ERROR_NOT_FOUND,
              cuGetProcAddress("cuMemAdvise", &fn, 13000, CU_GET_PROC_ADDRESS_DEFAULT, &status));

    const char* str = nullptr;
    CHECK_CU(cuGetErrorString(CUDA_ERROR_OUT_OF_MEMORY, &str));
    EXPECT(str != nullptr);

    std::printf("OK\n");
    return 0;
}
