// CUDA 12 libraries (cuLibrary*, cuKernel*) and JIT linking (cuLink*): a library loaded without a
// current context, its kernels enumerated and launched as CUkernels, a global read back; two PTX
// modules linked (one calls a function the other defines, through an .extern declaration) and the
// linked image loaded as a module and as a library.

#include "test_util.h"

#include <cstring>
#include <string>
#include <vector>

namespace {

const char* kLibraryPtx = R"(
.version 7.0
.target sm_75
.address_size 64

.global .align 4 .u32 counter = 41;

// x[i] = x[i] * 2 + 1
.visible .entry twice(.param .u64 px, .param .u32 n)
{
    .reg .b32 %r<6>;
    .reg .b64 %rd<4>;
    .reg .pred %p;
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.s32 %r4, %r1, %r2, %r3;
    ld.param.u32 %r5, [n];
    setp.ge.u32 %p, %r4, %r5;
    @%p bra DONE;
    ld.param.u64 %rd1, [px];
    mul.wide.u32 %rd2, %r4, 4;
    add.s64 %rd3, %rd1, %rd2;
    ld.global.u32 %r1, [%rd3];
    mad.lo.s32 %r1, %r1, 2, 1;
    st.global.u32 [%rd3], %r1;
DONE:
    ret;
}

.visible .entry bump()
{
    .reg .b32 %r<2>;
    ld.global.u32 %r1, [counter];
    add.u32 %r1, %r1, 1;
    st.global.u32 [counter], %r1;
    ret;
}
)";

// The caller: square() is declared here and defined in kHelperPtx.
const char* kCallerPtx = R"(
.version 7.0
.target sm_75
.address_size 64

.extern .func (.param .b32 result) square(.param .b32 x);

.visible .entry squares(.param .u64 px)
{
    .reg .b32 %r<4>;
    .reg .b64 %rd<4>;
    mov.u32 %r1, %tid.x;
    ld.param.u64 %rd1, [px];
    mul.wide.u32 %rd2, %r1, 4;
    add.s64 %rd3, %rd1, %rd2;
    {
        .param .b32 arg;
        .param .b32 ret;
        st.param.b32 [arg], %r1;
        call.uni (ret), square, (arg);
        ld.param.b32 %r2, [ret];
    }
    st.global.u32 [%rd3], %r2;
    ret;
}
)";

const char* kHelperPtx = R"(
.version 7.0
.target sm_75
.address_size 64

.visible .func (.param .b32 result) square(.param .b32 x)
{
    .reg .b32 %r<3>;
    ld.param.b32 %r1, [x];
    mul.lo.s32 %r2, %r1, %r1;
    st.param.b32 [result], %r2;
    ret;
}
)";

}  // namespace

int main() {
    CHECK_CU(cuInit(0));

    // A library loaded before any context exists: the first device's primary context gets it.
    CUlibrary lib;
    CHECK_CU(cuLibraryLoadData(&lib, kLibraryPtx, nullptr, nullptr, 0, nullptr, nullptr, 0));
    CUdevice dev;
    CHECK_CU(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CHECK_CU(cuDevicePrimaryCtxRetain(&ctx, dev));
    CHECK_CU(cuCtxSetCurrent(ctx));
    unsigned count = 0;
    CHECK_CU(cuLibraryGetKernelCount(&count, lib));
    EXPECT(count == 2);
    std::vector<CUkernel> kernels(count);
    CHECK_CU(cuLibraryEnumerateKernels(kernels.data(), count, lib));
    std::vector<std::string> names;
    for (CUkernel k : kernels) {
        const char* name = nullptr;
        CHECK_CU(cuKernelGetName(&name, k));
        names.push_back(name);
        CUlibrary owner;
        CHECK_CU(cuKernelGetLibrary(&owner, k));
        EXPECT(owner == lib);
    }
    EXPECT(names.size() == 2 && ((names[0] == "twice" && names[1] == "bump") || (names[0] == "bump" && names[1] == "twice")));
    CUkernel twice, bump;
    CHECK_CU(cuLibraryGetKernel(&twice, lib, "twice"));
    CHECK_CU(cuLibraryGetKernel(&bump, lib, "bump"));
    EXPECT_CU(CUDA_ERROR_NOT_FOUND, cuLibraryGetKernel(&bump, lib, "missing"));
    size_t offset = 0, size = 0;
    CHECK_CU(cuKernelGetParamInfo(twice, 1, &offset, &size));
    EXPECT(offset == 8 && size == 4);
    int maxThreads = 0;
    CHECK_CU(cuKernelGetAttribute(&maxThreads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, twice, dev));
    EXPECT(maxThreads > 0);

    const unsigned n = 1000;
    std::vector<uint32_t> host(n);
    for (unsigned i = 0; i < n; ++i) host[i] = i;
    CUdeviceptr x;
    CHECK_CU(cuMemAlloc(&x, n * 4));
    CHECK_CU(cuMemcpyHtoD(x, host.data(), n * 4));
    void* args[] = {&x, const_cast<unsigned*>(&n)};
    // a CUkernel launches as a CUfunction; cuKernelGetFunction gives the same function
    CHECK_CU(cuLaunchKernel(reinterpret_cast<CUfunction>(twice), (n + 127) / 128, 1, 1, 128, 1, 1, 0, nullptr, args, nullptr));
    CUfunction f;
    CHECK_CU(cuKernelGetFunction(&f, bump));
    CHECK_CU(cuLaunchKernel(f, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr));
    CHECK_CU(cuMemcpyDtoH(host.data(), x, n * 4));
    bool ok = true;
    for (unsigned i = 0; i < n; ++i) ok &= host[i] == 2 * i + 1;
    EXPECT(ok);
    CUdeviceptr counter;
    size_t counterBytes = 0;
    CHECK_CU(cuLibraryGetGlobal(&counter, &counterBytes, lib, "counter"));
    uint32_t value = 0;
    CHECK_CU(cuMemcpyDtoH(&value, counter, 4));
    EXPECT(counterBytes == 4 && value == 42);
    CUmodule mod;
    CHECK_CU(cuLibraryGetModule(&mod, lib));
    unsigned functions = 0;
    CHECK_CU(cuModuleGetFunctionCount(&functions, mod));
    EXPECT(functions == 2);
    CHECK_CU(cuLibraryUnload(lib));

    // Linking: the caller and the helper into one PTX image
    char errorLog[256] = "untouched";
    CUjit_option options[] = {CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
    void* values[] = {errorLog, reinterpret_cast<void*>(uintptr_t(sizeof errorLog))};
    CUlinkState link;
    CHECK_CU(cuLinkCreate(2, options, values, &link));
    EXPECT(errorLog[0] == 0);
    CHECK_CU(cuLinkAddData(link, CU_JIT_INPUT_PTX, const_cast<char*>(kCallerPtx), std::strlen(kCallerPtx) + 1, "caller",
                           0, nullptr, nullptr));
    CHECK_CU(cuLinkAddData(link, CU_JIT_INPUT_PTX, const_cast<char*>(kHelperPtx), std::strlen(kHelperPtx) + 1, "helper",
                           0, nullptr, nullptr));
    const unsigned char cubin[] = {0x7f, 'E', 'L', 'F', 2, 1, 1, 0};
    EXPECT_CU(CUDA_ERROR_NO_BINARY_FOR_GPU, cuLinkAddData(link, CU_JIT_INPUT_CUBIN, const_cast<unsigned char*>(cubin),
                                                          sizeof cubin, "native", 0, nullptr, nullptr));
    EXPECT(std::strstr(errorLog, "native") != nullptr);
    void* image = nullptr;
    size_t imageSize = 0;
    CHECK_CU(cuLinkComplete(link, &image, &imageSize));
    EXPECT(image && imageSize > 1);
    CUmodule linked;
    CHECK_CU(cuModuleLoadData(&linked, image));
    CUlibrary linkedLib;
    CHECK_CU(cuLibraryLoadData(&linkedLib, image, nullptr, nullptr, 0, nullptr, nullptr, 0));
    CHECK_CU(cuLinkDestroy(link));  // the module and library keep their own copies
    CUfunction squares;
    CHECK_CU(cuModuleGetFunction(&squares, linked, "squares"));
    void* sargs[] = {&x};
    CHECK_CU(cuLaunchKernel(squares, 1, 1, 1, 64, 1, 1, 0, nullptr, sargs, nullptr));
    CHECK_CU(cuMemcpyDtoH(host.data(), x, 64 * 4));
    ok = true;
    for (unsigned i = 0; i < 64; ++i) ok &= host[i] == i * i;
    EXPECT(ok);
    CHECK_CU(cuLibraryUnload(linkedLib));
    CHECK_CU(cuModuleUnload(linked));
    CHECK_CU(cuMemFree(x));
    CHECK_CU(cuDevicePrimaryCtxRelease(dev));
    std::printf("OK\n");
    return 0;
}
