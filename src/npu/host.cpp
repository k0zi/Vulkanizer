#include "npu/host.h"

#include <algorithm>
#include <cstring>
#include <vector>

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

namespace {

// Converts n elements from src (type st) to dst (type dt), dst elements dstStride apart.
void convert(const char* src, DType st, int64_t n, char* dst, DType dt, int64_t dstStride) {
    const size_t de = dtypeSize(dt);
    if (st == dt && dstStride == 1) {
        std::memcpy(dst, src, size_t(n) * de);
        return;
    }
    if (st == dt && de == 2) {
        const auto* s = reinterpret_cast<const uint16_t*>(src);
        auto* d = reinterpret_cast<uint16_t*>(dst);
        for (int64_t i = 0; i < n; ++i) d[i * dstStride] = s[i];
        return;
    }
    if (st == DType::F32 && dt == DType::BF16) {
        const auto* s = reinterpret_cast<const float*>(src);
        auto* d = reinterpret_cast<uint16_t*>(dst);
        for (int64_t i = 0; i < n; ++i) d[i * dstStride] = floatToBf16(s[i]);
        return;
    }
    for (int64_t i = 0; i < n; ++i) store(dst, dt, i * dstStride, load(src, st, i));
}

}  // namespace

// The source is read in whole contiguous runs (memcpy into a cached scratch buffer first): device
// memory is often write-combined on the host, where element-wise reads are ~100x slower.
void packTransposed(const void* x, DType xt, int64_t ld, bool trans, int64_t rows, int64_t cols, void* dst, DType dt,
                    int64_t dstLd) {
    const size_t xe = dtypeSize(xt), de = dtypeSize(dt);
    const char* src = static_cast<const char*>(x);
    char* out = static_cast<char*>(dst);
    if (!trans) {  // column c of X is contiguous, and so is row c of dst
        std::vector<char> tmp(size_t(rows) * xe);
        for (int64_t c = 0; c < cols; ++c) {
            const char* col = src + size_t(c * ld) * xe;
            if (xt == dt) {
                std::memcpy(out + size_t(c * dstLd) * de, col, size_t(rows) * xe);
                continue;
            }
            std::memcpy(tmp.data(), col, tmp.size());
            convert(tmp.data(), xt, rows, out + size_t(c * dstLd) * de, dt, 1);
        }
        return;
    }
    // op(X)[r][c] = X[c + r * ld]: row r of op(X) is a contiguous run of X; blocks of kBlock of them
    // are read, then written out kBlock contiguous elements of each dst row at a time.
    constexpr int64_t kBlock = 64;
    std::vector<char> tmp(size_t(kBlock * cols) * xe);
    for (int64_t r0 = 0; r0 < rows; r0 += kBlock) {
        const int64_t nr = std::min(kBlock, rows - r0);
        for (int64_t r = 0; r < nr; ++r)
            std::memcpy(tmp.data() + size_t(r * cols) * xe, src + size_t((r0 + r) * ld) * xe, size_t(cols) * xe);
        if (xt == dt && xe == 2) {  // transpose kBlock x kBlock tiles: both sides stay in L1
            const auto* t = reinterpret_cast<const uint16_t*>(tmp.data());
            auto* o = reinterpret_cast<uint16_t*>(out);
            for (int64_t c0 = 0; c0 < cols; c0 += kBlock)
                for (int64_t c = c0; c < std::min(cols, c0 + kBlock); ++c)
                    for (int64_t r = 0; r < nr; ++r) o[c * dstLd + r0 + r] = t[r * cols + c];
            continue;
        }
        for (int64_t c = 0; c < cols; ++c)
            for (int64_t r = 0; r < nr; ++r)
                store(out, dt, c * dstLd + r0 + r, load(tmp.data(), xt, r * cols + c));
    }
}

void zeroPadding(void* dst, size_t elem, int64_t rows, int64_t cols, int64_t ld, int64_t allRows) {
    char* d = static_cast<char*>(dst);
    if (cols < ld)
        for (int64_t r = 0; r < rows; ++r) std::memset(d + size_t(r * ld + cols) * elem, 0, size_t(ld - cols) * elem);
    if (rows < allRows) std::memset(d + size_t(rows * ld) * elem, 0, size_t((allRows - rows) * ld) * elem);
}

void epilogue(const void* d, DType dt, int64_t dLd, float alpha, float beta, void* c, DType ct, int64_t ldc, int64_t m,
              int64_t n) {
    const size_t de = dtypeSize(dt), ce = dtypeSize(ct);
    if (beta == 0 && dt == ct && alpha == 1) {
        for (int64_t j = 0; j < n; ++j)
            std::memcpy(static_cast<char*>(c) + size_t(j * ldc) * ce, static_cast<const char*>(d) + size_t(j * dLd) * de,
                        size_t(m) * ce);
        return;
    }
    if (beta == 0 && dt == DType::F32 && ct == DType::F32) {
        for (int64_t j = 0; j < n; ++j) {
            const float* s = static_cast<const float*>(d) + j * dLd;
            float* o = static_cast<float*>(c) + j * ldc;
            for (int64_t i = 0; i < m; ++i) o[i] = alpha * s[i];
        }
        return;
    }
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = 0; i < m; ++i) {
            float v = alpha * load(d, dt, j * dLd + i);
            if (beta != 0) v += beta * load(c, ct, i + j * ldc);
            store(c, ct, i + j * ldc, v);
        }
}

}  // namespace vulkanizer::npu
