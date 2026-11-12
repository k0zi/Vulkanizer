// f32 -> f16 conversion (cvt.rn.f16.f32 and the packed cvt.rn.f16x2.f32) against a host reference that
// rounds to nearest even: every f16 value, every midpoint between two neighbouring f16 values (ties)
// and the floats right next to it, and a spread of 2^22 f32 bit patterns (subnormals, overflow, inf,
// NaN). Runs with the GPU's conversion where the device has round-to-nearest-even float controls,
// and with VULKANIZER_SOFT_F16_CVT=1 (the integer code) as a second ctest entry.

#include "test_util.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

static const char* kPtx = R"(
.version 7.0
.target sm_70
.address_size 64

.visible .entry cvt_f16(.param .u64 in, .param .u64 out, .param .u64 outPacked, .param .u32 n)
{
    .reg .pred %p<2>;
    .reg .b32 %r<8>;
    .reg .f32 %f<3>;
    .reg .b16 %h<2>;
    .reg .b64 %rd<10>;
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.s32 %r1, %r1, %r2, %r3;
    ld.param.u32 %r2, [n];
    setp.ge.u32 %p1, %r1, %r2;
    @%p1 bra DONE;
    ld.param.u64 %rd1, [in];
    ld.param.u64 %rd2, [out];
    ld.param.u64 %rd3, [outPacked];
    cvta.to.global.u64 %rd1, %rd1;
    cvta.to.global.u64 %rd2, %rd2;
    cvta.to.global.u64 %rd3, %rd3;
    mul.wide.u32 %rd4, %r1, 4;
    add.s64 %rd5, %rd1, %rd4;
    ld.global.f32 %f1, [%rd5];
    cvt.rn.f16.f32 %h1, %f1;
    mul.wide.u32 %rd6, %r1, 2;
    add.s64 %rd7, %rd2, %rd6;
    st.global.b16 [%rd7], %h1;
    // packed: this value in the upper half, its negation in the lower one
    neg.f32 %f2, %f1;
    cvt.rn.f16x2.f32 %r4, %f1, %f2;
    add.s64 %rd8, %rd3, %rd4;
    st.global.b32 [%rd8], %r4;
DONE:
    ret;
}
)";

static uint16_t referenceHalf(float f) {  // round to nearest even; NaN -> some NaN (checked as a class)
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t sign = (u >> 16) & 0x8000u, a = u & 0x7FFFFFFFu;
    if (a > 0x7F800000u) return uint16_t(sign | 0x7E00u);
    const double v = std::fabs(double(f));
    // scale to units of the f16 spacing at this magnitude, round to nearest even
    int e;
    std::frexp(v, &e);  // v = m * 2^e, m in [0.5, 1)
    const int exp = std::max(e - 1, -14);  // f16 exponent of v (subnormals: -14)
    const double ulp = std::ldexp(1.0, exp - 10);
    double q = v / ulp;
    const double fl = std::floor(q), rem = q - fl;
    q = rem > 0.5 || (rem == 0.5 && std::fmod(fl, 2.0) != 0.0) ? fl + 1 : fl;
    const double r = q * ulp;
    if (r >= 65520.0 || v >= 65520.0) return uint16_t(sign | 0x7C00u);
    // encode r
    if (r < std::ldexp(1.0, -14)) return uint16_t(sign | uint32_t(r / std::ldexp(1.0, -24)));
    int re;
    const double rm = std::frexp(r, &re);  // r = rm * 2^re
    const uint32_t hexp = uint32_t(re - 1 + 15), mant = uint32_t((rm * 2 - 1) * 1024);
    return uint16_t(sign | hexp << 10 | mant);
}

static bool isHalfNan(uint16_t h) { return (h & 0x7C00u) == 0x7C00u && (h & 0x3FFu) != 0; }

static float halfToFloat(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000u) << 16, e = (h >> 10) & 0x1F, m = h & 0x3FF;
    float v;
    if (e == 0) v = std::ldexp(float(m), -24);
    else if (e == 31) v = m ? NAN : INFINITY;
    else v = std::ldexp(float(m | 0x400), int(e) - 25);
    uint32_t u;
    std::memcpy(&u, &v, 4);
    u |= sign;
    std::memcpy(&v, &u, 4);
    return v;
}

int main() {
    CHECK_CU(cuInit(0));
    CUdevice dev;
    CHECK_CU(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CHECK_CU(cuCtxCreate(&ctx, 0, dev));
    CUmodule mod;
    CHECK_CU(cuModuleLoadData(&mod, kPtx));
    CUfunction fn;
    CHECK_CU(cuModuleGetFunction(&fn, mod, "cvt_f16"));

    std::vector<float> in;
    for (uint32_t h = 0; h < 0x10000; ++h) {  // every half, the midpoints to the next and their neighbours
        if (isHalfNan(uint16_t(h)) || (h & 0x7FFFu) >= 0x7C00u) continue;
        const float x = halfToFloat(uint16_t(h));
        in.push_back(x);
        if ((h & 0x7FFFu) == 0x7BFFu) {  // above the largest finite half: the overflow boundary 65520
            for (float t : {65519.996f, 65520.0f, 65520.004f}) in.push_back(h & 0x8000u ? -t : t);
            continue;
        }
        const float y = halfToFloat(uint16_t(h + 1));
        const float mid = float((double(x) + double(y)) / 2);  // exact in f32
        in.push_back(mid);
        in.push_back(std::nextafter(mid, x));
        in.push_back(std::nextafter(mid, y));
    }
    for (uint64_t i = 0; i < (uint64_t(1) << 32); i += 1031) {  // a spread of all bit patterns
        uint32_t u = uint32_t(i);
        float f;
        std::memcpy(&f, &u, 4);
        in.push_back(f);
    }
    const uint32_t n = uint32_t(in.size());

    CUdeviceptr dIn, dOut, dPacked;
    CHECK_CU(cuMemAlloc(&dIn, size_t(n) * 4));
    CHECK_CU(cuMemAlloc(&dOut, size_t(n) * 2));
    CHECK_CU(cuMemAlloc(&dPacked, size_t(n) * 4));
    CHECK_CU(cuMemcpyHtoD(dIn, in.data(), size_t(n) * 4));
    void* args[] = {&dIn, &dOut, &dPacked, const_cast<uint32_t*>(&n)};
    CHECK_CU(cuLaunchKernel(fn, (n + 255) / 256, 1, 1, 256, 1, 1, 0, nullptr, args, nullptr));
    std::vector<uint16_t> out(n);
    std::vector<uint32_t> packed(n);
    CHECK_CU(cuMemcpyDtoH(out.data(), dOut, size_t(n) * 2));
    CHECK_CU(cuMemcpyDtoH(packed.data(), dPacked, size_t(n) * 4));

    size_t bad = 0;
    auto check = [&](const char* what, float x, uint16_t got) {
        const uint16_t want = referenceHalf(x);
        const bool ok = isHalfNan(want) ? isHalfNan(got) : got == want;
        if (!ok && ++bad <= 10) {
            uint32_t u;
            std::memcpy(&u, &x, 4);
            std::printf("%s: f32 0x%08x (%g) -> 0x%04x, expected 0x%04x\n", what, u, double(x), got, want);
        }
    };
    for (uint32_t i = 0; i < n; ++i) {
        check("cvt.rn.f16.f32", in[i], out[i]);
        check("cvt.rn.f16x2.f32 (hi)", in[i], uint16_t(packed[i] >> 16));
        check("cvt.rn.f16x2.f32 (lo)", -in[i], uint16_t(packed[i]));
    }
    std::printf("%u values, %zu mismatches\n", n, bad);
    CHECK_CU(cuCtxDestroy(ctx));
    if (bad) return 1;
    std::printf("OK\n");
    return 0;
}
