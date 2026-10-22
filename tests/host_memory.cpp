// Host memory the device uses at the host pointer: managed (cuMemAllocManaged), pinned
// (cuMemHostAlloc) and registered (cuMemHostRegister) memory, read and written by kernels and
// by the CPU through the same pointer.

#include "test_util.h"

#include <cstring>
#include <vector>

namespace {

const char* kPtx = R"(
.version 7.0
.target sm_75
.address_size 64

// y[i] = y[i] * a + b
.visible .entry axpb(.param .u64 py, .param .u32 n, .param .f32 a, .param .f32 b)
{
    .reg .b32 %r<6>;
    .reg .f32 %f<5>;
    .reg .b64 %rd<4>;
    .reg .pred %p;
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.s32 %r4, %r1, %r2, %r3;
    ld.param.u32 %r5, [n];
    setp.ge.u32 %p, %r4, %r5;
    @%p bra DONE;
    ld.param.u64 %rd1, [py];
    mul.wide.u32 %rd2, %r4, 4;
    add.s64 %rd3, %rd1, %rd2;
    ld.global.f32 %f1, [%rd3];
    ld.param.f32 %f2, [a];
    ld.param.f32 %f3, [b];
    fma.rn.f32 %f4, %f1, %f2, %f3;
    st.global.f32 [%rd3], %f4;
DONE:
    ret;
}
)";

CUfunction g_axpb;

void axpb(CUdeviceptr y, unsigned n, float a, float b) {
    void* args[] = {&y, &n, &a, &b};
    CHECK_CU(cuLaunchKernel(g_axpb, (n + 127) / 128, 1, 1, 128, 1, 1, 0, nullptr, args, nullptr));
    CHECK_CU(cuCtxSynchronize());
}

bool all(const float* v, unsigned n, float want) {
    for (unsigned i = 0; i < n; ++i)
        if (v[i] != want) {
            std::fprintf(stderr, "element %u: %g, want %g\n", i, v[i], want);
            return false;
        }
    return true;
}

template <class T>
T pointerAttribute(CUpointer_attribute attr, CUdeviceptr p) {
    T v{};
    CHECK_CU(cuPointerGetAttribute(&v, attr, p));
    return v;
}

}  // namespace

int main() {
    CHECK_CU(cuInit(0));
    CUdevice dev;
    CHECK_CU(cuDeviceGet(&dev, 0));
    int managedSupported = 0, mapSupported = 0;
    CHECK_CU(cuDeviceGetAttribute(&managedSupported, CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY, dev));
    CHECK_CU(cuDeviceGetAttribute(&mapSupported, CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, dev));
    if (!managedSupported || !mapSupported) {
        std::printf("SKIP: the device cannot import host memory\n");
        return 0;
    }
    CUcontext ctx;
    CHECK_CU(cuCtxCreate(&ctx, 0, dev));
    CUmodule mod;
    CHECK_CU(cuModuleLoadData(&mod, kPtx));
    CHECK_CU(cuModuleGetFunction(&g_axpb, mod, "axpb"));

    // Managed: the CPU writes, a kernel updates in place, the CPU reads, all through one pointer.
    const unsigned n = 100000;
    CUdeviceptr m;
    CHECK_CU(cuMemAllocManaged(&m, n * 4, CU_MEM_ATTACH_GLOBAL));
    float* mh = reinterpret_cast<float*>(m);
    for (unsigned i = 0; i < n; ++i) mh[i] = 2.0f;
    CHECK_CU(cuMemAdvise(m, n * 4, CU_MEM_ADVISE_SET_READ_MOSTLY, dev));
    CHECK_CU(cuMemPrefetchAsync(m, n * 4, dev, nullptr));
    axpb(m, n, 3.0f, 1.0f);
    EXPECT(all(mh, n, 7.0f));
    EXPECT(pointerAttribute<int>(CU_POINTER_ATTRIBUTE_IS_MANAGED, m) == 1);
    EXPECT(pointerAttribute<unsigned>(CU_POINTER_ATTRIBUTE_MEMORY_TYPE, m) == CU_MEMORYTYPE_DEVICE);
    EXPECT(pointerAttribute<CUdeviceptr>(CU_POINTER_ATTRIBUTE_DEVICE_POINTER, m + 8) == m + 8);
    unsigned oneBits;
    const float one = 1.0f;
    std::memcpy(&oneBits, &one, 4);
    CHECK_CU(cuMemsetD32(m, oneBits, n));  // driver operations work on it like on device memory
    EXPECT(all(mh, n, 1.0f));
    std::vector<float> copy(n);
    CHECK_CU(cuMemcpyDtoH(copy.data(), m, n * 4));
    EXPECT(all(copy.data(), n, 1.0f));
    CHECK_CU(cuMemFree(m));

    // Pinned memory, used by the kernel without a copy.
    void* pinned;
    CHECK_CU(cuMemHostAlloc(&pinned, n * 4, CU_MEMHOSTALLOC_DEVICEMAP));
    float* ph = static_cast<float*>(pinned);
    for (unsigned i = 0; i < n; ++i) ph[i] = float(i);
    CUdeviceptr pd;
    CHECK_CU(cuMemHostGetDevicePointer(&pd, pinned, 0));
    EXPECT(pd == CUdeviceptr(pinned));
    axpb(pd, n, 1.0f, 0.5f);
    bool ok = true;
    for (unsigned i = 0; i < n && ok; ++i) ok = ph[i] == float(i) + 0.5f;
    EXPECT(ok);
    EXPECT(pointerAttribute<unsigned>(CU_POINTER_ATTRIBUTE_MEMORY_TYPE, pd) == CU_MEMORYTYPE_HOST);
    unsigned flags = 0;
    CHECK_CU(cuMemHostGetFlags(&flags, pinned));
    EXPECT(flags & CU_MEMHOSTALLOC_DEVICEMAP);
    CHECK_CU(cuMemFreeHost(pinned));

    // Registered memory from malloc, not page-aligned: the device pointer is the host pointer.
    std::vector<float> plain(n + 37, 4.0f);
    float* reg = plain.data() + 37;
    CHECK_CU(cuMemHostRegister(reg, n * 4, CU_MEMHOSTREGISTER_DEVICEMAP));
    CUdeviceptr rd;
    CHECK_CU(cuMemHostGetDevicePointer(&rd, reg, 0));
    EXPECT(rd == CUdeviceptr(reg));
    axpb(rd, n, 0.5f, 0.0f);
    EXPECT(all(reg, n, 2.0f) && plain[36] == 4.0f);  // the rest of the page untouched
    // A second registration within the same pages stays host-only (the pages are taken).
    std::vector<float> small(16);
    float* neighbour = reg + n;  // not registered: plain's tail beyond the first registration
    CHECK_CU(cuMemHostUnregister(reg));
    CHECK_CU(cuMemHostRegister(plain.data(), 4, 0));
    CHECK_CU(cuMemHostRegister(plain.data() + 1, 4, 0));
    CUdeviceptr none;
    EXPECT_CU(CUDA_ERROR_INVALID_VALUE, cuMemHostGetDevicePointer(&none, plain.data() + 1, 0));
    CHECK_CU(cuMemHostUnregister(plain.data()));
    CHECK_CU(cuMemHostUnregister(plain.data() + 1));
    EXPECT_CU(CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED, cuMemHostUnregister(neighbour));
    (void)small;

    CHECK_CU(cuCtxDestroy(ctx));
    std::printf("OK\n");
    return 0;
}
