// Several devices (plan.md 8.): unique addresses per device (unified addressing), asynchronous peer
// copies in queue order on both devices, cross-device event waits that do not block the caller.
// Needs two devices: ctest runs it with VULKANIZER_DUPLICATE_DEVICES=2 (one GPU twice, multigpu)
// and on the machine's own devices (multigpu_devices, skipped with fewer than two).

#include "test_util.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

struct Dev {
    CUcontext ctx = nullptr;
    CUmodule mod = nullptr;
    CUfunction vecAdd = nullptr;
    CUstream stream = nullptr;
};

void use(const Dev& d) { CHECK_CU(cuCtxSetCurrent(d.ctx)); }

int ordinalOf(CUdeviceptr p) {
    int ordinal = -1;
    CHECK_CU(cuPointerGetAttribute(&ordinal, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, p));
    return ordinal;
}

void launchAdd(const Dev& d, CUdeviceptr a, CUdeviceptr b, CUdeviceptr c, int n) {
    void* args[] = {&a, &b, &c, &n};
    CHECK_CU(cuLaunchKernel(d.vecAdd, (n + 255) / 256, 1, 1, 256, 1, 1, 0, d.stream, args, nullptr));
}

std::vector<float> download(const Dev& d, CUdeviceptr p, size_t n) {
    use(d);
    std::vector<float> out(n);
    CHECK_CU(cuStreamSynchronize(d.stream));
    CHECK_CU(cuMemcpyDtoH(out.data(), p, n * sizeof(float)));
    return out;
}

}  // namespace

int main() {
    // small staging, so that the large copy below reuses its slots (llvmpipe's exportable memory
    // stops at ~64 MiB per allocation)
    setenv("VULKANIZER_PEER_STAGING_MB", "16", 0);
    CHECK_CU(cuInit(0));
    int count = 0;
    CHECK_CU(cuDeviceGetCount(&count));
    if (count < 2) {
        std::printf("SKIP: %d device(s)\n", count);
        return 0;
    }
    Dev dev[2];
    for (int i = 0; i < 2; ++i) {
        CUdevice d;
        CHECK_CU(cuDeviceGet(&d, i));
        CHECK_CU(cuCtxCreate(&dev[i].ctx, 0, d));
        CHECK_CU(cuModuleLoad(&dev[i].mod, VULKANIZER_TEST_DATA_DIR "/vector_add.ptx"));
        CHECK_CU(cuModuleGetFunction(&dev[i].vecAdd, dev[i].mod, "vecAdd"));
        CHECK_CU(cuStreamCreate(&dev[i].stream, CU_STREAM_NON_BLOCKING));
    }

    // Unique addresses: the same allocation sequence on both devices gives different pointers, and
    // a pointer's device is found from either context.
    constexpr int n = 1 << 20;
    constexpr size_t bytes = n * sizeof(float);
    CUdeviceptr a[2], b[2], c[2];
    for (int i = 0; i < 2; ++i) {
        use(dev[i]);
        CHECK_CU(cuMemAlloc(&a[i], bytes));
        CHECK_CU(cuMemAlloc(&b[i], bytes));
        CHECK_CU(cuMemAlloc(&c[i], bytes));
    }
    for (CUdeviceptr p : {a[0], b[0], c[0]})
        for (CUdeviceptr q : {a[1], b[1], c[1]}) EXPECT(p + bytes <= q || q + bytes <= p);
    for (int i = 0; i < 2; ++i) {
        use(dev[i]);
        EXPECT(ordinalOf(a[0]) == 0 && ordinalOf(c[0] + 100) == 0);
        EXPECT(ordinalOf(a[1]) == 1 && ordinalOf(c[1] + 100) == 1);
    }

    // c0 = a0 + b0 on device 0; peer copy to device 1 behind an event wait; c1 = c0' + b1 on device 1.
    std::vector<float> ha(n), hb(n);
    for (int i = 0; i < n; ++i) ha[i] = float(i % 1000), hb[i] = 0.25f * float(i % 77);
    for (int i = 0; i < 2; ++i) {
        use(dev[i]);
        CHECK_CU(cuMemcpyHtoD(a[i], ha.data(), bytes));
        CHECK_CU(cuMemcpyHtoD(b[i], hb.data(), bytes));
    }
    use(dev[0]);
    launchAdd(dev[0], a[0], b[0], c[0], n);
    // more work after it on device 0, so that the event is still pending below
    for (int k = 0; k < 64; ++k) launchAdd(dev[0], a[0], b[0], b[0], n);
    CUevent done0;
    CHECK_CU(cuEventCreate(&done0, CU_EVENT_DISABLE_TIMING));
    CHECK_CU(cuEventRecord(done0, dev[0].stream));
    use(dev[1]);
    const auto t0 = std::chrono::steady_clock::now();
    CHECK_CU(cuStreamWaitEvent(dev[1].stream, done0, 0));
    CHECK_CU(cuMemcpyPeerAsync(a[1], dev[1].ctx, c[0], dev[0].ctx, bytes, dev[1].stream));
    launchAdd(dev[1], a[1], b[1], c[1], n);
    const double callMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const CUresult pending = cuEventQuery(done0);
    std::printf("cross-device wait + peer copy + launch: %.2f ms (device 0 %s)\n", callMs,
                pending == CUDA_ERROR_NOT_READY ? "still busy" : "already done");
    std::vector<float> got = download(dev[1], c[1], n);
    for (int i = 0; i < n; ++i)
        if (got[i] != ha[i] + 2 * hb[i]) {
            std::fprintf(stderr, "peer copy then kernel: [%d] = %f, expected %f\n", i, got[i], ha[i] + 2 * hb[i]);
            return 1;
        }

    // Unified copies between the devices' pointers, from either current context (cuMemcpy and
    // cuMemcpyAsync with the source on the other device).
    use(dev[0]);
    CHECK_CU(cuMemcpy(a[0], c[1], bytes));  // device 1 -> 0, synchronous
    got = download(dev[0], a[0], n);
    EXPECT(std::memcmp(got.data(), download(dev[1], c[1], n).data(), bytes) == 0);
    use(dev[1]);
    CHECK_CU(cuMemsetD32Async(b[1], 0, n, dev[1].stream));
    CHECK_CU(cuMemcpyAsync(b[1] + 4096, a[0] + 8, bytes - 8192, dev[1].stream));  // interior, unaligned
    got = download(dev[1], b[1], n);
    std::vector<float> want = download(dev[0], a[0], n);
    EXPECT(got[0] == 0.0f && got[1023] == 0.0f && got[n - 1] == 0.0f);
    EXPECT(std::memcmp(got.data() + 1024, want.data() + 2, bytes - 8192) == 0);

    // Pitched rows between the devices: cuMemcpy3DPeerAsync (device 0 -> 1), and back with a plain
    // cuMemcpy2DAsync (unified addressing), into a third layout.
    {
        constexpr size_t width = 1000, rows = 300, srcPitch = 4096, dstPitch = 2048, backPitch = 1024;
        std::vector<uint8_t> image(srcPitch * rows);
        for (size_t i = 0; i < image.size(); ++i) image[i] = uint8_t(i * 7 + i / 4096);
        CUdeviceptr s0, d1, back0;
        use(dev[0]);
        CHECK_CU(cuMemAlloc(&s0, srcPitch * rows));
        CHECK_CU(cuMemAlloc(&back0, backPitch * rows));
        CHECK_CU(cuMemcpyHtoD(s0, image.data(), image.size()));
        use(dev[1]);
        CHECK_CU(cuMemAlloc(&d1, dstPitch * rows));
        CHECK_CU(cuMemsetD8(d1, 0xEE, dstPitch * rows));
        CUDA_MEMCPY3D_PEER p{};
        p.srcMemoryType = CU_MEMORYTYPE_DEVICE, p.srcDevice = s0 + 24, p.srcContext = dev[0].ctx, p.srcPitch = srcPitch;
        p.dstMemoryType = CU_MEMORYTYPE_DEVICE, p.dstDevice = d1, p.dstContext = dev[1].ctx, p.dstPitch = dstPitch;
        p.WidthInBytes = width, p.Height = rows, p.Depth = 1;
        CHECK_CU(cuMemcpy3DPeerAsync(&p, dev[1].stream));
        CUDA_MEMCPY2D q{};
        q.srcMemoryType = CU_MEMORYTYPE_UNIFIED, q.srcDevice = d1, q.srcPitch = dstPitch;
        q.dstMemoryType = CU_MEMORYTYPE_UNIFIED, q.dstDevice = back0, q.dstPitch = backPitch;
        q.WidthInBytes = width, q.Height = rows;
        CHECK_CU(cuMemcpy2DAsync(&q, dev[1].stream));
        CHECK_CU(cuStreamSynchronize(dev[1].stream));
        use(dev[0]);
        CHECK_CU(cuCtxSynchronize());
        std::vector<uint8_t> got1(dstPitch * rows), got0(backPitch * rows);
        CHECK_CU(cuMemcpyDtoH(got0.data(), back0, got0.size()));
        use(dev[1]);
        CHECK_CU(cuMemcpyDtoH(got1.data(), d1, got1.size()));
        for (size_t y = 0; y < rows; ++y) {
            EXPECT(std::memcmp(&got1[y * dstPitch], &image[y * srcPitch + 24], width) == 0);
            EXPECT(got1[y * dstPitch + width] == 0xEE);
            EXPECT(std::memcmp(&got0[y * backPitch], &image[y * srcPitch + 24], width) == 0);
        }
        CHECK_CU(cuMemFree(d1));
        use(dev[0]);
        CHECK_CU(cuMemFree(s0));
        CHECK_CU(cuMemFree(back0));
    }

    // A copy larger than the staging memory (its slots are reused), both directions.
    constexpr size_t bigN = size_t(16) << 20;  // 64 MiB: four times the staging set above (slots reused)
    CUdeviceptr big[2];
    std::vector<uint32_t> pattern(bigN);
    for (size_t i = 0; i < bigN; ++i) pattern[i] = uint32_t(i * 2654435761u);
    for (int i = 0; i < 2; ++i) {
        use(dev[i]);
        CHECK_CU(cuMemAlloc(&big[i], bigN * 4));
    }
    use(dev[0]);
    CHECK_CU(cuMemcpyHtoD(big[0], pattern.data(), bigN * 4));
    const auto t1 = std::chrono::steady_clock::now();
    CHECK_CU(cuMemcpyPeer(big[1], dev[1].ctx, big[0], dev[0].ctx, bigN * 4));
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    std::printf("peer copy of %zu MiB: %.1f ms, %.2f GB/s\n", bigN * 4 >> 20, s * 1e3, bigN * 4 / s * 1e-9);
    use(dev[1]);
    std::vector<uint32_t> back(bigN);
    CHECK_CU(cuMemcpyDtoH(back.data(), big[1], bigN * 4));
    EXPECT(back == pattern);
    use(dev[0]);
    CHECK_CU(cuMemsetD32(big[0], 0, bigN));
    use(dev[1]);
    CHECK_CU(cuMemcpyPeerAsync(big[0], dev[0].ctx, big[1], dev[1].ctx, bigN * 4, dev[1].stream));
    use(dev[0]);
    CHECK_CU(cuCtxSynchronize());  // device 0's queue holds the copy's second half
    std::fill(back.begin(), back.end(), 0);
    CHECK_CU(cuMemcpyDtoH(back.data(), big[0], bigN * 4));
    EXPECT(back == pattern);

    // Freeing another device's memory from the current one (unified addressing, as llama.cpp does).
    use(dev[1]);
    CHECK_CU(cuMemFree(big[0]));
    EXPECT_CU(CUDA_ERROR_INVALID_VALUE, cuMemFree(big[0]));
    use(dev[0]);
    CHECK_CU(cuMemFree(big[1]));

    for (int i = 0; i < 2; ++i) {
        use(dev[i]);
        for (CUdeviceptr p : {a[i], b[i], c[i]}) CHECK_CU(cuMemFree(p));
        CHECK_CU(cuStreamDestroy(dev[i].stream));
        CHECK_CU(cuModuleUnload(dev[i].mod));
    }
    CHECK_CU(cuEventDestroy(done0));
    for (Dev& d : dev) CHECK_CU(cuCtxDestroy(d.ctx));
    std::printf("OK\n");
    return 0;
}
