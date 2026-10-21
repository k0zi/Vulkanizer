// Contexts, allocation, copies and memsets.

#include "test_util.h"

#include <cstring>
#include <vector>

int main() {
    CHECK_CU(cuInit(0));
    CUdevice dev;
    CHECK_CU(cuDeviceGet(&dev, 0));

    CUdeviceptr p = 0;
    EXPECT_CU(CUDA_ERROR_INVALID_CONTEXT, cuMemAlloc(&p, 16));

    CUcontext ctx;
    CHECK_CU(cuCtxCreate(&ctx, 0, dev));
    CUcontext current = nullptr;
    CHECK_CU(cuCtxGetCurrent(&current));
    EXPECT(current == ctx);

    constexpr size_t N = 1 << 20;
    std::vector<uint32_t> host(N), back(N);
    for (size_t i = 0; i < N; ++i) host[i] = static_cast<uint32_t>(i * 2654435761u);

    CUdeviceptr a, b;
    EXPECT_CU(CUDA_ERROR_INVALID_VALUE, cuMemAlloc(&a, 0));
    CHECK_CU(cuMemAlloc(&a, N * 4));
    CHECK_CU(cuMemAlloc(&b, N * 4));
    EXPECT(a != 0 && b != 0 && a != b);
    EXPECT(a % 256 == 0 && b % 256 == 0);

    // HtoD -> DtoD -> DtoH round trip
    CHECK_CU(cuMemcpyHtoD(a, host.data(), N * 4));
    CHECK_CU(cuMemcpyDtoD(b, a, N * 4));
    CHECK_CU(cuMemcpyDtoH(back.data(), b, N * 4));
    EXPECT(std::memcmp(host.data(), back.data(), N * 4) == 0);

    // Interior pointers and range checks
    CUdeviceptr base = 0;
    size_t size = 0;
    CHECK_CU(cuMemGetAddressRange(&base, &size, a + 1000));
    EXPECT(base == a && size == N * 4);
    uint32_t word = 0;
    CHECK_CU(cuMemcpyDtoH(&word, a + 40, 4));
    EXPECT(word == host[10]);
    EXPECT_CU(CUDA_ERROR_INVALID_VALUE, cuMemcpyDtoH(back.data(), a + 8, N * 4));

    // Unified cuMemcpy: host <-> device
    std::vector<uint32_t> small(64, 7);
    CHECK_CU(cuMemcpy(a, reinterpret_cast<CUdeviceptr>(small.data()), 64 * 4));
    CHECK_CU(cuMemcpy(reinterpret_cast<CUdeviceptr>(back.data()), a, 64 * 4));
    EXPECT(back[0] == 7 && back[63] == 7);

    // Memsets
    CHECK_CU(cuMemsetD32(a, 0xdeadbeef, N));
    CHECK_CU(cuMemcpyDtoH(back.data(), a, N * 4));
    for (size_t i = 0; i < N; ++i) EXPECT(back[i] == 0xdeadbeef);
    CHECK_CU(cuMemsetD8(a, 0xab, 4096));
    CHECK_CU(cuMemsetD16(a + 4096, 0x1234, 2));
    CHECK_CU(cuMemcpyDtoH(back.data(), a, 8192));
    EXPECT(back[0] == 0xabababab && back[1023] == 0xabababab);
    EXPECT(back[1024] == 0x12341234 && back[1025] == 0xdeadbeef);

    size_t freeMem = 0, totalMem = 0;
    CHECK_CU(cuMemGetInfo(&freeMem, &totalMem));
    EXPECT(freeMem <= totalMem && freeMem > 0);

    // Streams and events (synchronous for now)
    CUstream stream;
    CUevent start, stop;
    CHECK_CU(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    CHECK_CU(cuEventCreate(&start, CU_EVENT_DEFAULT));
    CHECK_CU(cuEventCreate(&stop, CU_EVENT_DEFAULT));
    CHECK_CU(cuEventRecord(start, stream));
    CHECK_CU(cuMemcpyHtoDAsync(b, host.data(), N * 4, stream));
    CHECK_CU(cuEventRecord(stop, stream));
    CHECK_CU(cuStreamSynchronize(stream));
    float ms = -1;
    CHECK_CU(cuEventElapsedTime(&ms, start, stop));
    EXPECT(ms >= 0);
    CHECK_CU(cuEventDestroy(start));
    CHECK_CU(cuEventDestroy(stop));
    CHECK_CU(cuStreamDestroy(stream));
    EXPECT_CU(CUDA_ERROR_INVALID_HANDLE, cuStreamSynchronize(stream));

    CHECK_CU(cuMemFree(a));
    CHECK_CU(cuMemFree(b));
    EXPECT_CU(CUDA_ERROR_INVALID_VALUE, cuMemFree(a));
    CHECK_CU(cuCtxSynchronize());
    CHECK_CU(cuCtxDestroy(ctx));
    CHECK_CU(cuCtxGetCurrent(&current));
    EXPECT(current == nullptr);

    std::printf("OK\n");
    return 0;
}
