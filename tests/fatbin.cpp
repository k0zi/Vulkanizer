// Fat binaries: PTX selection, LZ4-compressed entries, the __fatBinC_Wrapper_t form and
// cubin-only images. Test images come from data/make_fatbin.py.

#include "test_util.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

struct Wrapper {  // __fatBinC_Wrapper_t
    uint32_t magic;
    uint32_t version;
    const void* data;
    const void* filename;
};

std::vector<uint64_t> readImage(const char* name) {
    const std::string bytes = readTextFile(std::string(VULKANIZER_TEST_DATA_DIR "/") + name);
    std::vector<uint64_t> image((bytes.size() + 7) / 8);  // 8-byte aligned like nvcc's embedded data
    std::memcpy(image.data(), bytes.data(), bytes.size());
    return image;
}

// Loads vecAdd from the module and checks a small launch.
void runVecAdd(CUmodule mod) {
    CUfunction fn;
    CHECK_CU(cuModuleGetFunction(&fn, mod, "vecAdd"));
    int n = 1000;
    std::vector<float> a(n), b(n), c(n);
    for (int i = 0; i < n; ++i) a[i] = float(i), b[i] = 2.0f;
    CUdeviceptr da, db, dc;
    CHECK_CU(cuMemAlloc(&da, n * sizeof(float)));
    CHECK_CU(cuMemAlloc(&db, n * sizeof(float)));
    CHECK_CU(cuMemAlloc(&dc, n * sizeof(float)));
    CHECK_CU(cuMemcpyHtoD(da, a.data(), n * sizeof(float)));
    CHECK_CU(cuMemcpyHtoD(db, b.data(), n * sizeof(float)));
    void* args[] = {&da, &db, &dc, &n};
    CHECK_CU(cuLaunchKernel(fn, (n + 127) / 128, 1, 1, 128, 1, 1, 0, nullptr, args, nullptr));
    CHECK_CU(cuMemcpyDtoH(c.data(), dc, n * sizeof(float)));
    for (int i = 0; i < n; ++i) EXPECT(c[i] == float(i) + 2.0f);
    CHECK_CU(cuMemFree(da));
    CHECK_CU(cuMemFree(db));
    CHECK_CU(cuMemFree(dc));
}

}  // namespace

int main() {
    CHECK_CU(cuInit(0));
    CUdevice dev;
    CHECK_CU(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CHECK_CU(cuCtxCreate(&ctx, 0, dev));
    CUmodule mod;

    // cubin + LZ4-compressed sm_52 PTX + sm_90 PTX: the sm_52 one must be picked for CC 7.5
    const std::vector<uint64_t> lz4 = readImage("vector_add_lz4.fatbin");
    CHECK_CU(cuModuleLoadData(&mod, lz4.data()));
    runVecAdd(mod);
    CHECK_CU(cuModuleUnload(mod));

    // the same through the wrapper struct, as __cudaRegisterFatBinary receives it
    const Wrapper wrapper{0x466243B1, 1, lz4.data(), nullptr};
    CHECK_CU(cuModuleLoadFatBinary(&mod, &wrapper));
    runVecAdd(mod);
    CHECK_CU(cuModuleUnload(mod));

    // uncompressed entries, newer one first
    const std::vector<uint64_t> plain = readImage("vector_add_plain.fatbin");
    CHECK_CU(cuModuleLoadData(&mod, plain.data()));
    runVecAdd(mod);
    CHECK_CU(cuModuleUnload(mod));

    // only native code: nothing to translate
    const std::vector<uint64_t> cubins = readImage("cubin_only.fatbin");
    EXPECT_CU(CUDA_ERROR_NO_BINARY_FOR_GPU, cuModuleLoadData(&mod, cubins.data()));

    CHECK_CU(cuCtxDestroy(ctx));
    std::printf("OK\n");
    return 0;
}
