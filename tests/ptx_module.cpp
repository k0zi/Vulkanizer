// PTX module loading, kernel lookup and parameter layout.

#include "test_util.h"

#include <cstring>

static const char* kLayoutPtx = R"(
.version 8.0
.target sm_75
.address_size 64

.global .align 4 .b8 table[64] = {1, 2, 3, 4};
.extern .shared .align 16 .b8 dynSmem[];

.func (.param .b32 retval) helper(.param .b32 x)
{
    ret;
}

.visible .entry layout(
    .param .u32 layout_param_0,
    .param .align 8 .b8 layout_param_1[16],
    .param .u8 layout_param_2,
    .param .u64 .ptr .global .align 16 layout_param_3
)
.maxntid 128, 1, 1
{
    { .reg .b32 nested; }
    ret;
}

.entry noParams()
{
    ret;
}
)";

int main() {
    CHECK_CU(cuInit(0));
    CUdevice dev;
    CHECK_CU(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CHECK_CU(cuDevicePrimaryCtxRetain(&ctx, dev));
    CHECK_CU(cuCtxSetCurrent(ctx));

    // vecAdd from a file, the way nvcc would emit it
    CUmodule mod;
    CHECK_CU(cuModuleLoad(&mod, VULKANIZER_TEST_DATA_DIR "/vector_add.ptx"));
    CUfunction fn;
    CHECK_CU(cuModuleGetFunction(&fn, mod, "vecAdd"));
    EXPECT_CU(CUDA_ERROR_NOT_FOUND, cuModuleGetFunction(&fn, mod, "nope"));
    CHECK_CU(cuModuleGetFunction(&fn, mod, "vecAdd"));
    const size_t expectedOffsets[] = {0, 8, 16, 24}, expectedSizes[] = {8, 8, 8, 4};
    for (size_t i = 0; i < 4; ++i) {
        size_t offset = 0, size = 0;
        CHECK_CU(cuFuncGetParamInfo(fn, i, &offset, &size));
        EXPECT(offset == expectedOffsets[i] && size == expectedSizes[i]);
    }
    EXPECT_CU(CUDA_ERROR_INVALID_VALUE, cuFuncGetParamInfo(fn, 4, nullptr, nullptr));
    int ptxVersion = 0;
    CHECK_CU(cuFuncGetAttribute(&ptxVersion, CU_FUNC_ATTRIBUTE_PTX_VERSION, fn));
    EXPECT(ptxVersion == 52);  // the virtual architecture (.target sm_52), not the ISA version
    CHECK_CU(cuModuleUnload(mod));
    EXPECT_CU(CUDA_ERROR_INVALID_HANDLE, cuModuleGetFunction(&fn, mod, "vecAdd"));

    // Alignment rules, module variables, .func skipping, tuning directives
    CHECK_CU(cuModuleLoadData(&mod, kLayoutPtx));
    CHECK_CU(cuModuleGetFunction(&fn, mod, "layout"));
    const size_t layoutOffsets[] = {0, 8, 24, 32}, layoutSizes[] = {4, 16, 1, 8};
    for (size_t i = 0; i < 4; ++i) {
        size_t offset = 0, size = 0;
        CHECK_CU(cuFuncGetParamInfo(fn, i, &offset, &size));
        EXPECT(offset == layoutOffsets[i] && size == layoutSizes[i]);
    }
    int maxThreads = 0;
    CHECK_CU(cuFuncGetAttribute(&maxThreads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, fn));
    EXPECT(maxThreads == 128);
    CHECK_CU(cuModuleGetFunction(&fn, mod, "noParams"));
    EXPECT_CU(CUDA_ERROR_NOT_FOUND, cuModuleGetFunction(&fn, mod, "helper"));
    CHECK_CU(cuModuleUnload(mod));

    // Malformed PTX reports through the JIT error log
    char log[256] = {};
    CUjit_option opts[] = {CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
    void* vals[] = {log, reinterpret_cast<void*>(sizeof(log))};
    EXPECT_CU(CUDA_ERROR_INVALID_PTX,
              cuModuleLoadDataEx(&mod, ".version 7.0\n.entry broken(\n", 2, opts, vals));
    EXPECT(std::strstr(log, "line") != nullptr);
    std::printf("expected parse error: %s\n", log);

    // A kernel the compiler cannot translate does not fail the module, only its launch
    static const char* kUnsupportedPtx = R"(
.version 7.8
.target sm_52
.address_size 64
.visible .entry fine() { ret; }
.visible .entry unsupported() { .reg .b32 r; vulkanizer_no_such_op.b32 r, 1; ret; }
)";
    std::memset(log, 0, sizeof log);
    CHECK_CU(cuModuleLoadDataEx(&mod, kUnsupportedPtx, 2, opts, vals));
    EXPECT(std::strstr(log, "unsupported") != nullptr && std::strstr(log, "vulkanizer_no_such_op") != nullptr);
    std::printf("expected translation error: %s", log);
    CHECK_CU(cuModuleGetFunction(&fn, mod, "fine"));
    CHECK_CU(cuLaunchKernel(fn, 1, 1, 1, 32, 1, 1, 0, nullptr, nullptr, nullptr));
    CHECK_CU(cuModuleGetFunction(&fn, mod, "unsupported"));
    EXPECT_CU(CUDA_ERROR_NOT_SUPPORTED, cuLaunchKernel(fn, 1, 1, 1, 32, 1, 1, 0, nullptr, nullptr, nullptr));
    CHECK_CU(cuCtxSynchronize());
    CHECK_CU(cuModuleUnload(mod));

    // Native SASS (cubin) cannot be translated
    const char elf[] = "\x7f" "ELF\x02\x01\x01";
    EXPECT_CU(CUDA_ERROR_NO_BINARY_FOR_GPU, cuModuleLoadData(&mod, elf));

    CHECK_CU(cuDevicePrimaryCtxRelease(dev));
    std::printf("OK\n");
    return 0;
}
