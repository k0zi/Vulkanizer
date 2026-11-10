#include "npu/host.h"

#include <cstring>

namespace vulkanizer::npu {

size_t dtypeSize(DType t) { return t == DType::F32 ? 4 : 2; }

float halfToFloat(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000) << 16, exp = (h >> 10) & 0x1F, mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0x1F) bits = sign | 0x7F800000 | (mant << 13);
    else if (exp) bits = sign | ((exp + 112) << 23) | (mant << 13);
    else if (!mant) bits = sign;
    else {  // subnormal
        int e = -1;
        uint32_t m = mant;
        do { ++e; m <<= 1; } while (!(m & 0x400));
        bits = sign | ((112 - e) << 23) | ((m & 0x3FF) << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

uint16_t floatToHalf(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t sign = (u >> 16) & 0x8000;
    const uint32_t abs = u & 0x7FFFFFFF;
    if (abs > 0x7F800000) return uint16_t(sign | 0x7E00);  // NaN
    if (abs >= 0x477FF000) return uint16_t(sign | 0x7C00);  // rounds to infinity
    if (abs < 0x38800000) {  // subnormal or zero: shift the mantissa with its implicit bit
        if (abs < 0x33000000) return uint16_t(sign);
        const uint32_t e = abs >> 23, mant = (abs & 0x7FFFFF) | 0x800000;
        const uint32_t shift = 126 - e;
        uint32_t v = mant >> shift;
        const uint32_t rest = mant & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rest > half || (rest == half && (v & 1))) ++v;
        return uint16_t(sign | v);
    }
    uint32_t v = ((abs - 0x38000000) >> 13);
    const uint32_t rest = abs & 0x1FFF;
    if (rest > 0x1000 || (rest == 0x1000 && (v & 1))) ++v;
    return uint16_t(sign | v);
}

float bf16ToFloat(uint16_t h) {
    const uint32_t bits = uint32_t(h) << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

uint16_t floatToBf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7FFFFFFF) > 0x7F800000) return uint16_t((u >> 16) | 0x40);  // quiet NaN
    u += 0x7FFF + ((u >> 16) & 1);
    return uint16_t(u >> 16);
}

float load(const void* base, DType t, int64_t index) {
    switch (t) {
    case DType::F32: return static_cast<const float*>(base)[index];
    case DType::F16: return halfToFloat(static_cast<const uint16_t*>(base)[index]);
    case DType::BF16: return bf16ToFloat(static_cast<const uint16_t*>(base)[index]);
    }
    return 0;
}

void store(void* base, DType t, int64_t index, float v) {
    switch (t) {
    case DType::F32: static_cast<float*>(base)[index] = v; return;
    case DType::F16: static_cast<uint16_t*>(base)[index] = floatToHalf(v); return;
    case DType::BF16: static_cast<uint16_t*>(base)[index] = floatToBf16(v); return;
    }
}

void packTransposed(const void* x, DType xt, int64_t ld, bool trans, int64_t rows, int64_t cols, void* dst, DType dt,
                    int64_t dstLd) {
    for (int64_t c = 0; c < cols; ++c)
        for (int64_t r = 0; r < rows; ++r)
            store(dst, dt, c * dstLd + r, load(x, xt, trans ? c + r * ld : r + c * ld));
}

void epilogue(const void* d, DType dt, int64_t dLd, float alpha, float beta, void* c, DType ct, int64_t ldc, int64_t m,
              int64_t n) {
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = 0; i < m; ++i) {
            float v = alpha * load(d, dt, j * dLd + i);
            if (beta != 0) v += beta * load(c, ct, i + j * ldc);
            store(c, ct, i + j * ldc, v);
        }
}

}  // namespace vulkanizer::npu
