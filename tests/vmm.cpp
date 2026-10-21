// Virtual memory management (reserve / create / map across the buffers of a large reservation),
// stream-ordered allocation and memory pools.

#include "test_util.h"

#include <cmath>
#include <cstring>
#include <vector>

static void testPools() {
    CUmemoryPool def = nullptr, cur = nullptr;
    CHECK_CU(cuDeviceGetDefaultMemPool(&def, 0));
    CHECK_CU(cuDeviceGetMemPool(&cur, 0));
    EXPECT(def && def == cur);

    CUdeviceptr p = 0;
    CHECK_CU(cuMemAllocAsync(&p, 1 << 20, nullptr));
    CHECK_CU(cuMemsetD32Async(p, 0x01020304, (1 << 20) / 4, nullptr));
    cuuint64_t used = 0;
    CHECK_CU(cuMemPoolGetAttribute(def, CU_MEMPOOL_ATTR_USED_MEM_CURRENT, &used));
    EXPECT(used == 1 << 20);
    uint32_t word = 0;
    CHECK_CU(cuMemcpyDtoH(&word, p + 4096, 4));
    EXPECT(word == 0x01020304);
    CHECK_CU(cuMemFreeAsync(p, nullptr));  // deferred: no wait
    CHECK_CU(cuMemPoolGetAttribute(def, CU_MEMPOOL_ATTR_USED_MEM_CURRENT, &used));
    EXPECT(used == 0);
    EXPECT_CU(CUDA_ERROR_INVALID_VALUE, cuMemFreeAsync(p, nullptr));

    CUmemPoolProps props{};
    props.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    props.location = {CU_MEM_LOCATION_TYPE_DEVICE, 0};
    CUmemoryPool pool;
    CHECK_CU(cuMemPoolCreate(&pool, &props));
    cuuint64_t threshold = ~0ull;
    CHECK_CU(cuMemPoolSetAttribute(pool, CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, &threshold));
    CHECK_CU(cuMemAllocFromPoolAsync(&p, 4096, pool, nullptr));
    CHECK_CU(cuMemPoolGetAttribute(pool, CU_MEMPOOL_ATTR_USED_MEM_HIGH, &used));
    EXPECT(used == 4096);
    CHECK_CU(cuDeviceSetMemPool(0, pool));
    CHECK_CU(cuMemPoolDestroy(pool));  // the outstanding allocation stays valid
    CHECK_CU(cuDeviceGetMemPool(&cur, 0));
    EXPECT(cur == def);
    CHECK_CU(cuMemFree(p));

    // the entry points programs resolve for the per-thread default stream
    void* fn = nullptr;
    CHECK_CU(cuGetProcAddress("cuMemAllocAsync", &fn, 12000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM, nullptr));
    void* ptsz = nullptr;
    CHECK_CU(cuGetProcAddress("cuMemAllocAsync_ptsz", &ptsz, 12000, 0, nullptr));
    EXPECT(fn && fn == ptsz);
}

static void testVmm(CUfunction vecAdd) {
    int supported = 0;
    CHECK_CU(cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, 0));
    if (!supported) {
        std::printf("no virtual memory management on this device, skipped\n");
        return;
    }
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location = {CU_MEM_LOCATION_TYPE_DEVICE, 0};
    size_t g = 0;
    CHECK_CU(cuMemGetAllocationGranularity(&g, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    EXPECT(g >= (2u << 20) && (g & (g - 1)) == 0);

    // 12 GB of address space: several buffers where maxBufferSize is 4 GB. The window around the
    // 4 GB mark crosses the first buffer boundary (the buffers are a little under 4 GB).
    const size_t reserved = size_t(12) << 30, window = 16 * g;
    CUdeviceptr base = 0;
    CUresult r = cuMemAddressReserve(&base, reserved, 0, 0, 0);
    size_t at = (size_t(4) << 30) - window / 2;
    if (r == CUDA_ERROR_OUT_OF_MEMORY) {  // no address capture replay: one buffer only
        std::printf("large reservations unsupported, testing a small one\n");
        CHECK_CU(cuMemAddressReserve(&base, 64 * g, 0, 0, 0));
        at = 8 * g;
    } else {
        CHECK_CU(r);
    }
    EXPECT(base % g == 0);
    const CUdeviceptr win = base + at;

    // two physical allocations, mapped side by side
    CUmemGenericAllocationHandle h1, h2;
    CHECK_CU(cuMemCreate(&h1, window / 2, &prop, 0));
    CHECK_CU(cuMemCreate(&h2, window / 2, &prop, 0));
    CHECK_CU(cuMemMap(win, window / 2, 0, h1, 0));
    CHECK_CU(cuMemMap(win + window / 2, window / 2, 0, h2, 0));
    EXPECT_CU(CUDA_ERROR_ALREADY_MAPPED, cuMemMap(win, g, 0, h2, 0));
    CHECK_CU(cuMemRelease(h2));  // stays alive while mapped
    CUmemAccessDesc access{};
    access.location = prop.location;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CHECK_CU(cuMemSetAccess(win, window, &access, 1));

    // host copies across the boundary
    const size_t n = window / 4;
    std::vector<float> a(n), back(n);
    for (size_t i = 0; i < n; ++i) a[i] = float(i % 100000);
    CHECK_CU(cuMemcpyHtoD(win, a.data(), window));
    CHECK_CU(cuMemcpyDtoH(back.data(), win, window));
    EXPECT(std::memcmp(a.data(), back.data(), window) == 0);
    CUdeviceptr rangeBase = 0;
    size_t rangeSize = 0;
    CHECK_CU(cuMemGetAddressRange(&rangeBase, &rangeSize, win + window / 2 + 100));
    EXPECT(rangeBase == win + window / 2 && rangeSize == window / 2);

    // a kernel over the whole window: c = a + b with a = b = the window, c a plain buffer
    CUdeviceptr c;
    CHECK_CU(cuMemAlloc(&c, window));
    int count = int(n);
    void* args[] = {const_cast<CUdeviceptr*>(&win), const_cast<CUdeviceptr*>(&win), &c, &count};
    CHECK_CU(cuLaunchKernel(vecAdd, (count + 255) / 256, 1, 1, 256, 1, 1, 0, nullptr, args, nullptr));
    CHECK_CU(cuMemcpyDtoD(win, c, window));  // and back into the window
    CHECK_CU(cuMemsetD8(win + window - 2, 0, 2));
    CHECK_CU(cuMemcpyDtoH(back.data(), win, window));
    for (size_t i = 0; i + 1 < n; ++i)
        if (back[i] != 2 * a[i]) {
            std::fprintf(stderr, "mismatch at %zu: %f\n", i, back[i]);
            std::exit(1);
        }
    CHECK_CU(cuMemFree(c));

    CUmemGenericAllocationHandle retained;
    CHECK_CU(cuMemRetainAllocationHandle(&retained, reinterpret_cast<void*>(win + 10)));
    EXPECT(retained == h1);
    CHECK_CU(cuMemRelease(retained));
    EXPECT_CU(CUDA_ERROR_INVALID_VALUE, cuMemUnmap(win + g, g));  // not a whole mapping
    CHECK_CU(cuMemUnmap(win, window));                             // both mappings at once
    CHECK_CU(cuMemRelease(h1));
    EXPECT_CU(CUDA_ERROR_INVALID_VALUE, cuMemRelease(h2));  // freed with its last mapping
    CHECK_CU(cuMemAddressFree(base, r == CUDA_SUCCESS ? reserved : 64 * g));

    // host registration
    std::vector<char> host(1 << 20);
    CHECK_CU(cuMemHostRegister(host.data(), host.size() / 2, 0));
    EXPECT_CU(CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED, cuMemHostRegister(host.data() + 100, 100, 0));
    CHECK_CU(cuMemHostRegister(host.data() + host.size() / 2, host.size() / 2, 0));
    CHECK_CU(cuMemHostUnregister(host.data()));
    CHECK_CU(cuMemHostUnregister(host.data() + host.size() / 2));
    EXPECT_CU(CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED, cuMemHostUnregister(host.data()));
}

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

    testPools();
    testVmm(fn);

    CHECK_CU(cuModuleUnload(mod));
    CHECK_CU(cuCtxDestroy(ctx));
    std::printf("OK\n");
    return 0;
}
