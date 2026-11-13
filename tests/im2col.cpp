// PyTorch's im2col_kernel as Vulkanizer's own shader (driver/im2col.cpp): stub kernels with the names of
// PyTorch's (elements of 2, 4 and 8 bytes), launched with its arguments, against a reference on the CPU.
// Strides, padding, dilation; output rows that are and are not a multiple of the 16-byte stores.

#include "test_util.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

const char* kNames[] = {
    "_ZN2at6native13im2col_kernelIN3c108BFloat16EEEvlPKT_llllllllllllPS4_",  // 2 bytes
    "_ZN2at6native13im2col_kernelIfEEvlPKT_llllllllllllPS2_",                // 4 bytes
    "_ZN2at6native13im2col_kernelIdEEvlPKT_llllllllllllPS2_",                // 8 bytes
};

struct Case {
    long c, h, w, kh, kw, ph, pw, sh, sw, dh, dw;
};

template <typename T>
bool run(CUfunction fn, const Case& k) {
    long hc = (k.h + 2 * k.ph - k.dh * (k.kh - 1) - 1) / k.sh + 1;
    long wc = (k.w + 2 * k.pw - k.dw * (k.kw - 1) - 1) / k.sw + 1;
    std::vector<T> im(k.c * k.h * k.w), col(k.c * k.kh * k.kw * hc * wc), ref(col.size());
    for (size_t i = 0; i < im.size(); ++i) im[i] = T(i + 1);
    for (long c = 0; c < k.c; ++c)
        for (long i = 0; i < k.kh; ++i)
            for (long j = 0; j < k.kw; ++j)
                for (long y = 0; y < hc; ++y)
                    for (long x = 0; x < wc; ++x) {
                        const long h = y * k.sh - k.ph + i * k.dh, w = x * k.sw - k.pw + j * k.dw;
                        ref[(((c * k.kh + i) * k.kw + j) * hc + y) * wc + x] =
                            h >= 0 && h < k.h && w >= 0 && w < k.w ? im[(c * k.h + h) * k.w + w] : T(0);
                    }
    CUdeviceptr dim, dcol;
    CHECK_CU(cuMemAlloc(&dim, im.size() * sizeof(T)));
    CHECK_CU(cuMemAlloc(&dcol, col.size() * sizeof(T)));
    CHECK_CU(cuMemcpyHtoD(dim, im.data(), im.size() * sizeof(T)));
    CHECK_CU(cuMemsetD8(dcol, 0xff, col.size() * sizeof(T)));
    long n = k.c * hc * wc, h = k.h, w = k.w, kh = k.kh, kw = k.kw, ph = k.ph, pw = k.pw, sh = k.sh, sw = k.sw,
         dh = k.dh, dw = k.dw;
    void* args[] = {&n, &dim, &h, &w, &kh, &kw, &ph, &pw, &sh, &sw, &dh, &dw, &hc, &wc, &dcol};
    CHECK_CU(cuLaunchKernel(fn, unsigned((n + 511) / 512), 1, 1, 512, 1, 1, 0, nullptr, args, nullptr));
    CHECK_CU(cuMemcpyDtoH(col.data(), dcol, col.size() * sizeof(T)));
    CHECK_CU(cuMemFree(dim));
    CHECK_CU(cuMemFree(dcol));
    const bool ok = std::memcmp(col.data(), ref.data(), col.size() * sizeof(T)) == 0;
    std::printf("%zu-byte c%ld %ldx%ld k%ldx%ld p%ld,%ld s%ld,%ld d%ld,%ld: %s\n", sizeof(T), k.c, k.h, k.w, k.kh, k.kw,
                k.ph, k.pw, k.sh, k.sw, k.dh, k.dw, ok ? "ok" : "FAILED");
    return ok;
}

}  // namespace

int main() {
    CHECK_CU(cuInit(0));
    CUdevice dev;
    CHECK_CU(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CHECK_CU(cuDevicePrimaryCtxRetain(&ctx, dev));
    CHECK_CU(cuCtxSetCurrent(ctx));
    std::string ptx = ".version 8.0\n.target sm_75\n.address_size 64\n";
    for (const char* name : kNames) {
        ptx += std::string(".visible .entry ") + name + "(.param .u64 a0, .param .u64 a1";
        for (int i = 2; i < 15; ++i) ptx += ", .param .u64 a" + std::to_string(i);
        ptx += ") { ret; }\n";
    }
    CUmodule mod;
    CHECK_CU(cuModuleLoadData(&mod, ptx.c_str()));
    CUfunction fn[3];
    for (int i = 0; i < 3; ++i) CHECK_CU(cuModuleGetFunction(&fn[i], mod, kNames[i]));

    const Case cases[] = {
        {3, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1},   // output rows of 16: whole 16-byte stores
        {2, 9, 11, 3, 3, 1, 1, 1, 1, 1, 1},    // rows of 11
        {3, 37, 29, 5, 3, 2, 1, 2, 2, 1, 1},   // strides
        {2, 33, 35, 3, 3, 2, 2, 1, 1, 2, 2},   // dilation
        {4, 20, 24, 1, 1, 0, 0, 1, 1, 1, 1},   // 1 x 1
    };
    bool ok = true;
    for (const Case& c : cases) {
        ok &= run<uint16_t>(fn[0], c);
        ok &= run<float>(fn[1], c);
        ok &= run<double>(fn[2], c);
    }
    CHECK_CU(cuModuleUnload(mod));
    CHECK_CU(cuDevicePrimaryCtxRelease(dev));
    if (!ok) return 1;
    std::printf("OK\n");
    return 0;
}
