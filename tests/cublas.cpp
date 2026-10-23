// Vulkanizer's libcublas GEMM family against a CPU reference.

#include "test_util.h"

#include <cublasLt.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

namespace {

int g_failures = 0;

#define CHECK_BLAS(call)                                                                                   \
    do {                                                                                                   \
        cublasStatus_t s_ = (call);                                                                        \
        if (s_ != CUBLAS_STATUS_SUCCESS) {                                                                 \
            std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #call, cublasGetStatusName(s_)); \
            std::exit(1);                                                                                  \
        }                                                                                                  \
    } while (0)

uint16_t toHalf(float f) {  // round to nearest even
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t sign = (u >> 16) & 0x8000, a = u & 0x7FFFFFFF;
    if (a >= 0x477FF000) return uint16_t(sign | 0x7C00);
    if (a < 0x38800000) return uint16_t(sign | uint32_t(std::lround(std::ldexp(double(std::fabs(f)), 24))));
    const uint32_t x = a - (112u << 23);
    uint32_t h = x >> 13;
    const uint32_t rem = x & 0x1FFF;
    if (rem > 0x1000 || (rem == 0x1000 && (h & 1))) ++h;
    return uint16_t(sign | h);
}
float fromHalf(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000) << 16, e = (h >> 10) & 0x1F, m = h & 0x3FF;
    float f = e ? std::ldexp(1.0f + float(m) / 1024.0f, int(e) - 15) : std::ldexp(float(m), -24);
    return sign ? -f : f;
}
uint16_t toBf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return uint16_t((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}
float fromBf16(uint16_t b) {
    const uint32_t u = uint32_t(b) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// A matrix stored as `type`, plus the float values it actually holds.
struct Matrix {
    cudaDataType type;
    std::vector<float> values;
    std::vector<uint8_t> bytes;
    CUdeviceptr dev = 0;

    Matrix(cudaDataType t, size_t count, uint32_t seed) : type(t), values(count) {
        const size_t esize = t == CUDA_R_32F ? 4 : 2;
        bytes.resize(count * esize);
        for (size_t i = 0; i < count; ++i) {
            seed = seed * 1664525u + 1013904223u;
            float v = float(int(seed >> 24) - 128) / 64.0f;
            if (t == CUDA_R_16F) v = fromHalf(toHalf(v));
            if (t == CUDA_R_16BF) v = fromBf16(toBf16(v));
            values[i] = v;
            if (t == CUDA_R_32F) std::memcpy(&bytes[i * 4], &v, 4);
            else {
                const uint16_t h = t == CUDA_R_16F ? toHalf(v) : toBf16(v);
                std::memcpy(&bytes[i * 2], &h, 2);
            }
        }
        CHECK_CU(cuMemAlloc(&dev, std::max<size_t>(bytes.size(), 4)));  // k = 0 leaves A and B empty
        if (!bytes.empty()) CHECK_CU(cuMemcpyHtoD(dev, bytes.data(), bytes.size()));
    }
    ~Matrix() { cuMemFree(dev); }

    std::vector<float> download() {
        CHECK_CU(cuCtxSynchronize());
        if (!bytes.empty()) CHECK_CU(cuMemcpyDtoH(bytes.data(), dev, bytes.size()));
        std::vector<float> out(values.size());
        for (size_t i = 0; i < out.size(); ++i) {
            if (type == CUDA_R_32F) std::memcpy(&out[i], &bytes[i * 4], 4);
            else {
                uint16_t h;
                std::memcpy(&h, &bytes[i * 2], 2);
                out[i] = type == CUDA_R_16F ? fromHalf(h) : fromBf16(h);
            }
        }
        return out;
    }
};

struct Case {
    const char* name;
    cublasOperation_t ta, tb;
    int m, n, k;
    cudaDataType typeA, typeB, typeC;
    float alpha, beta;
    int batch;  // >1: strided batched; <-1: pointer-array batched with -batch entries
    cublasComputeType_t compute = CUBLAS_COMPUTE_32F;
    bool aligned = false;  // leading dimensions multiples of 8 (cuBLAS's vectorized f16 kernels)
};

void runCase(cublasHandle_t h, const Case& c) {
    const int batch = c.batch < 0 ? -c.batch : c.batch;
    const int rowsA = c.ta == CUBLAS_OP_N ? c.m : c.k, colsA = c.ta == CUBLAS_OP_N ? c.k : c.m;
    const int rowsB = c.tb == CUBLAS_OP_N ? c.k : c.n, colsB = c.tb == CUBLAS_OP_N ? c.n : c.k;
    const auto pad = [&](int rows, int extra) { return c.aligned ? (rows + extra + 7) / 8 * 8 : rows + extra; };
    const int lda = pad(rowsA, 3), ldb = pad(rowsB, 1), ldc = c.m + 2;  // padded leading dimensions
    const long long sA = (long long)lda * colsA, sB = (long long)ldb * colsB, sC = (long long)ldc * c.n;
    Matrix A(c.typeA, size_t(sA) * batch, 1), B(c.typeB, size_t(sB) * batch, 2), C(c.typeC, size_t(sC) * batch, 3);

    std::vector<float> want = C.values, wantAbs(C.values.size());  // wantAbs: |alpha| sum |a b|
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < c.n; ++j)
            for (int i = 0; i < c.m; ++i) {
                double acc = 0, absAcc = 0;
                for (int l = 0; l < c.k; ++l) {
                    const float a = c.ta == CUBLAS_OP_N ? A.values[b * sA + i + l * lda] : A.values[b * sA + l + i * lda];
                    const float bv = c.tb == CUBLAS_OP_N ? B.values[b * sB + l + j * ldb] : B.values[b * sB + j + l * ldb];
                    acc += double(a) * double(bv);
                    absAcc += std::fabs(double(a) * double(bv));
                }
                wantAbs[b * sC + i + j * ldc] = float(std::fabs(c.alpha) * absAcc);
                float& out = want[b * sC + i + j * ldc];
                out = float(c.alpha * acc + double(c.beta) * out);
                if (c.typeC == CUDA_R_16F) out = fromHalf(toHalf(out));
                if (c.typeC == CUDA_R_16BF) out = fromBf16(toBf16(out));
            }

    const auto pa = reinterpret_cast<const void*>(A.dev), pb = reinterpret_cast<const void*>(B.dev);
    void* pc = reinterpret_cast<void*>(C.dev);
    // alpha/beta have the compute type's scalar type
    const uint16_t alphaH = toHalf(c.alpha), betaH = toHalf(c.beta);
    const bool halfScalars = c.compute == CUBLAS_COMPUTE_16F;
    const void* alpha = halfScalars ? static_cast<const void*>(&alphaH) : &c.alpha;
    const void* beta = halfScalars ? static_cast<const void*>(&betaH) : &c.beta;
    const bool plainF32 = c.typeA == CUDA_R_32F && c.typeB == CUDA_R_32F && c.typeC == CUDA_R_32F;
    if (c.batch == 1 && plainF32) {
        CHECK_BLAS(cublasSgemm(h, c.ta, c.tb, c.m, c.n, c.k, &c.alpha, (const float*)pa, lda, (const float*)pb, ldb,
                               &c.beta, (float*)pc, ldc));
    } else if (c.batch == 1) {
        CHECK_BLAS(cublasGemmEx(h, c.ta, c.tb, c.m, c.n, c.k, alpha, pa, c.typeA, lda, pb, c.typeB, ldb, beta, pc,
                                c.typeC, ldc, c.compute, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else if (c.batch > 1) {
        CHECK_BLAS(cublasGemmStridedBatchedEx(h, c.ta, c.tb, c.m, c.n, c.k, alpha, pa, c.typeA, lda, sA, pb, c.typeB,
                                              ldb, sB, beta, pc, c.typeC, ldc, sC, batch, c.compute,
                                              CUBLAS_GEMM_DEFAULT));
    } else {
        const size_t esA = c.typeA == CUDA_R_32F ? 4 : 2, esB = c.typeB == CUDA_R_32F ? 4 : 2,
                     esC = c.typeC == CUDA_R_32F ? 4 : 2;
        std::vector<uint64_t> ptrs;
        for (int b = 0; b < batch; ++b) ptrs.push_back(A.dev + b * sA * esA);
        for (int b = 0; b < batch; ++b) ptrs.push_back(B.dev + b * sB * esB);
        for (int b = 0; b < batch; ++b) ptrs.push_back(C.dev + b * sC * esC);
        CUdeviceptr dptrs;
        CHECK_CU(cuMemAlloc(&dptrs, ptrs.size() * 8));
        CHECK_CU(cuMemcpyHtoD(dptrs, ptrs.data(), ptrs.size() * 8));
        auto arr = reinterpret_cast<const void**>(dptrs);
        CHECK_BLAS(cublasGemmBatchedEx(h, c.ta, c.tb, c.m, c.n, c.k, &c.alpha, arr, c.typeA, lda, arr + batch, c.typeB,
                                       ldb, &c.beta, const_cast<void**>(arr + 2 * batch), c.typeC, ldc, batch,
                                       CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        CHECK_CU(cuCtxSynchronize());
        CHECK_CU(cuMemFree(dptrs));
    }

    const std::vector<float> got = C.download();
    // f32 accumulation in a different order than the double reference; half outputs round once.
    const float relTol = c.typeC == CUDA_R_32F ? 1e-5f : c.typeC == CUDA_R_16F ? 2e-3f : 1.6e-2f;
    // Compute 16F (and 32F_FAST_16F) may round products and partial sums to f16 (2^-11 each).
    const bool f16Math = c.compute == CUBLAS_COMPUTE_16F || c.compute == CUBLAS_COMPUTE_32F_FAST_16F;
    for (size_t i = 0; i < got.size(); ++i) {
        const float tol = relTol * std::fmax(1.0f, std::fabs(want[i])) * std::sqrt(float(c.k) + 1.0f) +
                          (f16Math ? std::ldexp(wantAbs[i], -10) : 0.0f);
        if (!(std::fabs(got[i] - want[i]) <= tol)) {
            std::fprintf(stderr, "  %s: C[%zu] = %g, want %g\n", c.name, i, got[i], want[i]);
            std::printf("%-28s FAILED\n", c.name);
            ++g_failures;
            return;
        }
    }
    std::printf("%-28s ok\n", c.name);
}

// Every side/fill/transpose/diagonal combination, checked through the residual.
void testTrsm(cublasHandle_t h) {
    const int m = 7, n = 5, batch = 3;
    int combo = 0;
    for (cublasSideMode_t side : {CUBLAS_SIDE_LEFT, CUBLAS_SIDE_RIGHT})
        for (cublasFillMode_t uplo : {CUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_UPPER})
            for (cublasOperation_t trans : {CUBLAS_OP_N, CUBLAS_OP_T})
                for (cublasDiagType_t diag : {CUBLAS_DIAG_NON_UNIT, CUBLAS_DIAG_UNIT}) {
                    const int order = side == CUBLAS_SIDE_LEFT ? m : n, lda = order + 1, ldb = m + 2;
                    const float alpha = 1.5f;
                    std::vector<float> a(size_t(lda) * order * batch), b(size_t(ldb) * n * batch);
                    uint32_t seed = 77 + combo++;
                    auto rnd = [&] { return float(int((seed = seed * 1664525u + 1013904223u) >> 24) - 128) / 512.0f; };
                    for (int t = 0; t < batch; ++t) {
                        for (int c = 0; c < order; ++c)
                            for (int r = 0; r < order; ++r) {
                                const bool inTriangle = uplo == CUBLAS_FILL_MODE_LOWER ? r >= c : r <= c;
                                // the other triangle holds garbage that must be ignored
                                a[t * lda * order + r + c * lda] = r == c ? 2.0f + float(r) * 0.1f : inTriangle ? rnd() : 1e6f;
                            }
                        for (int i = 0; i < ldb * n; ++i) b[t * ldb * n + i] = rnd() * 4.0f;
                    }
                    CUdeviceptr da, db, ptrs;
                    CHECK_CU(cuMemAlloc(&da, a.size() * 4));
                    CHECK_CU(cuMemAlloc(&db, b.size() * 4));
                    CHECK_CU(cuMemAlloc(&ptrs, 2 * batch * 8));
                    CHECK_CU(cuMemcpyHtoD(da, a.data(), a.size() * 4));
                    CHECK_CU(cuMemcpyHtoD(db, b.data(), b.size() * 4));
                    std::vector<uint64_t> p;
                    for (int t = 0; t < batch; ++t) p.push_back(da + uint64_t(t) * lda * order * 4);
                    for (int t = 0; t < batch; ++t) p.push_back(db + uint64_t(t) * ldb * n * 4);
                    CHECK_CU(cuMemcpyHtoD(ptrs, p.data(), p.size() * 8));
                    auto arr = reinterpret_cast<float**>(ptrs);
                    CHECK_BLAS(cublasStrsmBatched(h, side, uplo, trans, diag, m, n, &alpha, (const float* const*)arr, lda,
                                                  arr + batch, ldb, batch));
                    CHECK_CU(cuCtxSynchronize());
                    std::vector<float> x(b.size());
                    CHECK_CU(cuMemcpyDtoH(x.data(), db, x.size() * 4));
                    CHECK_CU(cuMemFree(da));
                    CHECK_CU(cuMemFree(db));
                    CHECK_CU(cuMemFree(ptrs));

                    // op(A)[r][c] with the unused triangle zero and an optional unit diagonal
                    auto opA = [&](int t, int r, int c) -> double {
                        const int sr = trans == CUBLAS_OP_N ? r : c, sc = trans == CUBLAS_OP_N ? c : r;
                        if (sr == sc) return diag == CUBLAS_DIAG_UNIT ? 1.0 : a[t * lda * order + sr + sc * lda];
                        const bool inTriangle = uplo == CUBLAS_FILL_MODE_LOWER ? sr > sc : sr < sc;
                        return inTriangle ? a[t * lda * order + sr + sc * lda] : 0.0;
                    };
                    double worst = 0;
                    for (int t = 0; t < batch; ++t)
                        for (int i = 0; i < m; ++i)
                            for (int j = 0; j < n; ++j) {
                                double lhs = 0;
                                if (side == CUBLAS_SIDE_LEFT)
                                    for (int k = 0; k < m; ++k) lhs += opA(t, i, k) * x[t * ldb * n + k + j * ldb];
                                else
                                    for (int k = 0; k < n; ++k) lhs += x[t * ldb * n + i + k * ldb] * opA(t, k, j);
                                worst = std::max(worst, std::fabs(lhs - alpha * b[t * ldb * n + i + j * ldb]));
                            }
                    const bool ok = worst < 1e-4;
                    std::printf("strsmBatched %s %s %s %-8s %s\n", side == CUBLAS_SIDE_LEFT ? "L" : "R",
                                uplo == CUBLAS_FILL_MODE_LOWER ? "lo" : "up", trans == CUBLAS_OP_N ? "N" : "T",
                                diag == CUBLAS_DIAG_UNIT ? "unit" : "nonunit", ok ? "ok" : "FAILED");
                    if (!ok) {
                        std::fprintf(stderr, "  residual %g\n", worst);
                        ++g_failures;
                    }
                }
}

// cublasDgemm (single, strided and pointer-array batched) against an f64 CPU reference; skipped
// where the device has no f64.
void testDgemm(cublasHandle_t h) {
    struct D {
        const char* name;
        cublasOperation_t ta, tb;
        int m, n, k, batch;  // batch < 0: pointer arrays
        double alpha, beta;
    };
    const cublasOperation_t N = CUBLAS_OP_N, T = CUBLAS_OP_T;
    const D cases[] = {{"dgemm NN", N, N, 37, 21, 50, 1, 1.0, 0.0},  {"dgemm TN beta", T, N, 16, 33, 17, 1, -0.5, 2.0},
                       {"dgemm NT", N, T, 5, 1, 40, 1, 1.0, 0.0},     {"dgemm TT", T, T, 1, 30, 9, 1, 2.0, 1.0},
                       {"dgemm strided", N, T, 12, 14, 8, 4, 1.0, 0.0}, {"dgemm pointers", T, N, 9, 7, 30, -3, 1.0, 0.5}};
    for (const D& c : cases) {
        const int batch = std::abs(c.batch), rowsA = c.ta == N ? c.m : c.k, colsA = c.ta == N ? c.k : c.m;
        const int rowsB = c.tb == N ? c.k : c.n, colsB = c.tb == N ? c.n : c.k;
        const size_t sa = size_t(rowsA) * colsA, sb = size_t(rowsB) * colsB, sc = size_t(c.m) * c.n;
        std::vector<double> a(sa * batch), b(sb * batch), cc(sc * batch);
        uint32_t seed = 99;
        auto rnd = [&] { return double(int((seed = seed * 1664525u + 1013904223u) >> 8) - (1 << 23)) / (1 << 23); };
        for (double& x : a) x = rnd();
        for (double& x : b) x = rnd();
        for (double& x : cc) x = rnd();
        CUdeviceptr da, db, dc, ptrs;
        CHECK_CU(cuMemAlloc(&da, a.size() * 8));
        CHECK_CU(cuMemAlloc(&db, b.size() * 8));
        CHECK_CU(cuMemAlloc(&dc, cc.size() * 8));
        CHECK_CU(cuMemAlloc(&ptrs, 3 * batch * 8));
        CHECK_CU(cuMemcpyHtoD(da, a.data(), a.size() * 8));
        CHECK_CU(cuMemcpyHtoD(db, b.data(), b.size() * 8));
        CHECK_CU(cuMemcpyHtoD(dc, cc.data(), cc.size() * 8));
        std::vector<uint64_t> p;
        for (int t = 0; t < batch; ++t) p.push_back(da + t * sa * 8);
        for (int t = 0; t < batch; ++t) p.push_back(db + t * sb * 8);
        for (int t = 0; t < batch; ++t) p.push_back(dc + t * sc * 8);
        CHECK_CU(cuMemcpyHtoD(ptrs, p.data(), p.size() * 8));
        cublasStatus_t st;
        if (c.batch < 0) {
            auto arr = reinterpret_cast<double**>(ptrs);
            st = cublasDgemmBatched(h, c.ta, c.tb, c.m, c.n, c.k, &c.alpha, (const double* const*)arr, rowsA,
                                    (const double* const*)arr + batch, rowsB, &c.beta, arr + 2 * batch, c.m, batch);
        } else if (c.batch > 1) {
            st = cublasDgemmStridedBatched(h, c.ta, c.tb, c.m, c.n, c.k, &c.alpha, (const double*)da, rowsA, sa,
                                           (const double*)db, rowsB, sb, &c.beta, (double*)dc, c.m, sc, batch);
        } else {
            st = cublasDgemm(h, c.ta, c.tb, c.m, c.n, c.k, &c.alpha, (const double*)da, rowsA, (const double*)db, rowsB,
                             &c.beta, (double*)dc, c.m);
        }
        if (st == CUBLAS_STATUS_NOT_SUPPORTED) {
            std::printf("%-24s skipped (no f64)\n", c.name);
        } else {
            CHECK_BLAS(st);
            CHECK_CU(cuCtxSynchronize());
            std::vector<double> got(cc.size());
            CHECK_CU(cuMemcpyDtoH(got.data(), dc, got.size() * 8));
            double worst = 0;
            for (int t = 0; t < batch; ++t)
                for (int i = 0; i < c.m; ++i)
                    for (int j = 0; j < c.n; ++j) {
                        double s = 0;
                        for (int l = 0; l < c.k; ++l) {
                            const double x = c.ta == N ? a[t * sa + i + size_t(l) * rowsA] : a[t * sa + l + size_t(i) * rowsA];
                            const double y = c.tb == N ? b[t * sb + l + size_t(j) * rowsB] : b[t * sb + j + size_t(l) * rowsB];
                            s += x * y;
                        }
                        const double want = c.alpha * s + c.beta * cc[t * sc + i + size_t(j) * c.m];
                        worst = std::max(worst, std::fabs(want - got[t * sc + i + size_t(j) * c.m]));
                    }
            const bool ok = worst < 1e-12;
            std::printf("%-24s %s\n", c.name, ok ? "ok" : "FAILED");
            if (!ok) {
                std::fprintf(stderr, "  max error %g\n", worst);
                ++g_failures;
            }
        }
        cuMemFree(da), cuMemFree(db), cuMemFree(dc), cuMemFree(ptrs);
    }
}

}  // namespace

// cublasSgemv with both transpositions and vector strides.
void testSgemv(cublasHandle_t h) {
    bool ok = true;
    for (const cublasOperation_t trans : {CUBLAS_OP_N, CUBLAS_OP_T}) {
        const int m = 37, n = 53, lda = 40, incx = 2, incy = 3;
        const int rows = trans == CUBLAS_OP_N ? m : n, cols = trans == CUBLAS_OP_N ? n : m;
        Matrix A(CUDA_R_32F, size_t(lda) * n, 21), x(CUDA_R_32F, size_t(cols) * incx, 22),
            y(CUDA_R_32F, size_t(rows) * incy, 23);
        const float alpha = 1.5f, beta = -0.5f;
        if (trans == CUBLAS_OP_N)
            CHECK_BLAS(cublasSgemv(h, trans, m, n, &alpha, reinterpret_cast<const float*>(A.dev), lda,
                                   reinterpret_cast<const float*>(x.dev), incx, &beta, reinterpret_cast<float*>(y.dev),
                                   incy));
        else  // the 64-bit form (cuBLAS 12+, what cu13x PyTorch builds call)
            CHECK_BLAS(cublasSgemv_v2_64(h, trans, m, n, &alpha, reinterpret_cast<const float*>(A.dev), lda,
                                         reinterpret_cast<const float*>(x.dev), incx, &beta,
                                         reinterpret_cast<float*>(y.dev), incy));
        const std::vector<float> got = y.download();
        for (int i = 0; i < rows; ++i) {
            double acc = 0;
            for (int l = 0; l < cols; ++l) {
                const float a = trans == CUBLAS_OP_N ? A.values[i + l * lda] : A.values[l + i * lda];
                acc += double(a) * double(x.values[l * incx]);
            }
            const float want = float(alpha * acc + beta * y.values[i * incy]);
            if (std::fabs(got[i * incy] - want) > 1e-4f * std::fmax(1.0f, std::fabs(want))) {
                std::fprintf(stderr, "  sgemv %c: y[%d] = %g, want %g\n", trans == CUBLAS_OP_N ? 'N' : 'T', i,
                             got[i * incy], want);
                ok = false;
                break;
            }
            // elements between the strided ones are untouched
            if (i + 1 < rows && got[i * incy + 1] != y.values[i * incy + 1]) {
                std::fprintf(stderr, "  sgemv: wrote between strided elements\n");
                ok = false;
                break;
            }
        }
    }
    // sizes beyond 32 bits are refused rather than truncated
    const float one = 1.0f;
    if (cublasSgemv_v2_64(h, CUBLAS_OP_N, int64_t(1) << 32, 1, &one, nullptr, int64_t(1) << 32, nullptr, 1, &one,
                          nullptr, 1) != CUBLAS_STATUS_NOT_SUPPORTED) {
        std::fprintf(stderr, "  sgemv_64: a 2^32 row count was not refused\n");
        ok = false;
    }
    std::printf("%-28s %s\n", "sgemv", ok ? "ok" : "FAILED");
    if (!ok) ++g_failures;
}

// cublasLtMatmul: D = epilogue(alpha op(A) op(B) + beta C) with D != C, bias, ReLU / GELU, batches.
void testLt(cublasHandle_t h) {
    struct LtCase {
        const char* name;
        cublasOperation_t ta, tb;
        int m, n, k, batch;
        cudaDataType type;
        cublasLtEpilogue_t epilogue;
        float alpha, beta;
    };
    const LtCase cases[] = {
        {"lt TN bias", CUBLAS_OP_T, CUBLAS_OP_N, 70, 33, 40, 1, CUDA_R_32F, CUBLASLT_EPILOGUE_BIAS, 1.0f, 0.0f},
        {"lt NN relu bias beta", CUBLAS_OP_N, CUBLAS_OP_N, 50, 20, 64, 1, CUDA_R_32F, CUBLASLT_EPILOGUE_RELU_BIAS,
         0.5f, 1.5f},
        {"lt TN gelu bias f16", CUBLAS_OP_T, CUBLAS_OP_N, 64, 48, 32, 1, CUDA_R_16F, CUBLASLT_EPILOGUE_GELU_BIAS,
         1.0f, 0.0f},
        {"lt NT default batched", CUBLAS_OP_N, CUBLAS_OP_T, 30, 17, 25, 3, CUDA_R_32F, CUBLASLT_EPILOGUE_DEFAULT,
         1.0f, -1.0f},
    };
    for (const LtCase& c : cases) {
        const int rowsA = c.ta == CUBLAS_OP_N ? c.m : c.k, colsA = c.ta == CUBLAS_OP_N ? c.k : c.m;
        const int rowsB = c.tb == CUBLAS_OP_N ? c.k : c.n, colsB = c.tb == CUBLAS_OP_N ? c.n : c.k;
        const long long sA = (long long)rowsA * colsA, sB = (long long)rowsB * colsB, sC = (long long)c.m * c.n;
        Matrix A(c.type, size_t(sA) * c.batch, 31), B(c.type, size_t(sB) * c.batch, 32),
            C(c.type, size_t(sC) * c.batch, 33), D(c.type, size_t(sC) * c.batch, 34), bias(c.type, c.m, 35);
        cublasLtMatmulDesc_t desc;
        CHECK_BLAS(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        const int32_t ta = c.ta, tb = c.tb;
        const uint32_t epilogue = c.epilogue;
        const void* biasPtr = reinterpret_cast<const void*>(bias.dev);
        CHECK_BLAS(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta));
        CHECK_BLAS(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb));
        CHECK_BLAS(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof epilogue));
        CHECK_BLAS(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &biasPtr, sizeof biasPtr));
        cublasLtMatrixLayout_t la, lb, lc, ld;
        CHECK_BLAS(cublasLtMatrixLayoutCreate(&la, c.type, rowsA, colsA, rowsA));
        CHECK_BLAS(cublasLtMatrixLayoutCreate(&lb, c.type, rowsB, colsB, rowsB));
        CHECK_BLAS(cublasLtMatrixLayoutCreate(&lc, c.type, c.m, c.n, c.m));
        CHECK_BLAS(cublasLtMatrixLayoutCreate(&ld, c.type, c.m, c.n, c.m));
        const int32_t batch = c.batch;
        const std::pair<cublasLtMatrixLayout_t, long long> strides[] = {{la, sA}, {lb, sB}, {lc, sC}, {ld, sC}};
        for (const auto& [l, stride] : strides) {
            const int64_t s64 = stride;
            CHECK_BLAS(cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof batch));
            CHECK_BLAS(cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &s64,
                                                        sizeof s64));
        }
        cublasLtMatmulPreference_t pref;
        CHECK_BLAS(cublasLtMatmulPreferenceCreate(&pref));
        const uint64_t ws = 1 << 20;
        CHECK_BLAS(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof ws));
        cublasLtMatmulHeuristicResult_t heuristic{};
        int found = 0;
        CHECK_BLAS(cublasLtMatmulAlgoGetHeuristic(h, desc, la, lb, lc, ld, pref, 1, &heuristic, &found));
        bool ok = found == 1;
        CHECK_BLAS(cublasLtMatmul(h, desc, &c.alpha, reinterpret_cast<const void*>(A.dev), la,
                                  reinterpret_cast<const void*>(B.dev), lb, &c.beta,
                                  reinterpret_cast<const void*>(C.dev), lc, reinterpret_cast<void*>(D.dev), ld,
                                  &heuristic.algo, nullptr, 0, nullptr));
        const std::vector<float> got = D.download(), gotC = C.download();
        const bool half = c.type == CUDA_R_16F;
        for (int b = 0; b < c.batch && ok; ++b)
            for (int j = 0; j < c.n && ok; ++j)
                for (int i = 0; i < c.m && ok; ++i) {
                    double acc = 0;
                    for (int l = 0; l < c.k; ++l) {
                        const float a = c.ta == CUBLAS_OP_N ? A.values[b * sA + i + l * rowsA]
                                                            : A.values[b * sA + l + i * rowsA];
                        const float bv = c.tb == CUBLAS_OP_N ? B.values[b * sB + l + j * rowsB]
                                                             : B.values[b * sB + j + l * rowsB];
                        acc += double(a) * double(bv);
                    }
                    double v = c.alpha * acc + c.beta * C.values[b * sC + i + j * c.m];
                    if (half) v = fromHalf(toHalf(float(v)));  // D is rounded once after the GEMM
                    const uint32_t e = c.epilogue;
                    if (e & CUBLASLT_EPILOGUE_BIAS) v += bias.values[i];
                    if ((e & ~CUBLASLT_EPILOGUE_BIAS) == CUBLASLT_EPILOGUE_RELU) v = std::max(v, 0.0);
                    if ((e & ~CUBLASLT_EPILOGUE_BIAS) == CUBLASLT_EPILOGUE_GELU)
                        v = 0.5 * v * (1.0 + std::tanh(0.7978845608 * (v + 0.044715 * v * v * v)));
                    const float want = half ? fromHalf(toHalf(float(v))) : float(v);
                    const float tol = (half ? 4e-3f : 1e-5f) * std::fmax(1.0f, std::fabs(want)) * std::sqrt(float(c.k));
                    if (!(std::fabs(got[b * sC + i + j * c.m] - want) <= tol)) {
                        std::fprintf(stderr, "  %s: D[%d,%d,%d] = %g, want %g\n", c.name, b, i, j,
                                     got[b * sC + i + j * c.m], want);
                        ok = false;
                    }
                }
        ok = ok && gotC == C.values;  // C is only read
        cublasLtMatmulPreferenceDestroy(pref);
        for (auto l : {la, lb, lc, ld}) cublasLtMatrixLayoutDestroy(l);
        cublasLtMatmulDescDestroy(desc);
        std::printf("%-28s %s\n", c.name, ok ? "ok" : "FAILED");
        if (!ok) ++g_failures;
    }
}

// cublasGemmGroupedBatchedEx: groups of pointer-array problems with their own shapes and scalars.
void testGrouped(cublasHandle_t h) {
    struct Group {
        int m, n, k, count;
        float alpha, beta;
    };
    const Group groups[] = {{40, 7, 33, 2, 1.0f, 0.0f}, {17, 5, 64, 1, -0.5f, 1.0f}, {64, 1, 48, 3, 2.0f, 0.0f}};
    std::vector<std::unique_ptr<Matrix>> as, bs, cs;
    std::vector<uint64_t> pa, pb, pc;
    std::vector<cublasOperation_t> ta, tb;
    std::vector<int> ms, ns, ks, ldas, ldbs, ldcs, sizes;
    std::vector<float> alphas, betas;
    uint32_t seed = 11;
    for (const Group& g : groups) {
        for (int i = 0; i < g.count; ++i) {
            as.push_back(std::make_unique<Matrix>(CUDA_R_16F, size_t(g.k) * g.m, seed++));
            bs.push_back(std::make_unique<Matrix>(CUDA_R_16F, size_t(g.k) * g.n, seed++));
            cs.push_back(std::make_unique<Matrix>(CUDA_R_32F, size_t(g.m) * g.n, seed++));
            pa.push_back(as.back()->dev), pb.push_back(bs.back()->dev), pc.push_back(cs.back()->dev);
        }
        ta.push_back(CUBLAS_OP_T), tb.push_back(CUBLAS_OP_N);
        ms.push_back(g.m), ns.push_back(g.n), ks.push_back(g.k);
        ldas.push_back(g.k), ldbs.push_back(g.k), ldcs.push_back(g.m);
        sizes.push_back(g.count), alphas.push_back(g.alpha), betas.push_back(g.beta);
    }
    std::vector<uint64_t> ptrs = pa;
    ptrs.insert(ptrs.end(), pb.begin(), pb.end());
    ptrs.insert(ptrs.end(), pc.begin(), pc.end());
    CUdeviceptr dptrs;
    CHECK_CU(cuMemAlloc(&dptrs, ptrs.size() * 8));
    CHECK_CU(cuMemcpyHtoD(dptrs, ptrs.data(), ptrs.size() * 8));
    const size_t total = pa.size();
    auto arr = reinterpret_cast<const void**>(dptrs);
    CHECK_BLAS(cublasGemmGroupedBatchedEx(h, ta.data(), tb.data(), ms.data(), ns.data(), ks.data(), alphas.data(), arr,
                                          CUDA_R_16F, ldas.data(), arr + total, CUDA_R_16F, ldbs.data(), betas.data(),
                                          const_cast<void**>(arr + 2 * total), CUDA_R_32F, ldcs.data(), 3, sizes.data(),
                                          CUBLAS_COMPUTE_32F_FAST_16F));
    bool ok = true;
    size_t p = 0;
    for (const Group& g : groups)
        for (int i = 0; i < g.count; ++i, ++p) {
            const std::vector<float> got = cs[p]->download();
            for (int c = 0; c < g.n && ok; ++c)
                for (int r = 0; r < g.m && ok; ++r) {
                    double acc = 0;
                    for (int l = 0; l < g.k; ++l)
                        acc += double(as[p]->values[l + r * g.k]) * double(bs[p]->values[l + c * g.k]);
                    const float want = float(g.alpha * acc + g.beta * cs[p]->values[r + c * g.m]);
                    if (std::fabs(got[r + c * g.m] - want) > 2e-3f * std::fmax(1.0f, std::fabs(want)) * 8.0f) {
                        std::fprintf(stderr, "  grouped: problem %zu C[%d,%d] = %g, want %g\n", p, r, c,
                                     got[r + c * g.m], want);
                        ok = false;
                    }
                }
        }
    CHECK_CU(cuMemFree(dptrs));
    std::printf("%-28s %s\n", "gemmGroupedBatchedEx", ok ? "ok" : "FAILED");
    if (!ok) ++g_failures;
}

int main() {
    CHECK_CU(cuInit(0));
    CUdevice dev;
    CHECK_CU(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CHECK_CU(cuCtxCreate(&ctx, 0, dev));
    cublasHandle_t h;
    CHECK_BLAS(cublasCreate(&h));
    CHECK_BLAS(cublasSetMathMode(h, CUBLAS_TF32_TENSOR_OP_MATH));

    const cublasOperation_t N = CUBLAS_OP_N, T = CUBLAS_OP_T;
    const Case cases[] = {
        {"sgemm NN", N, N, 70, 33, 129, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, 1.0f, 0.0f, 1},
        {"sgemm TN", T, N, 128, 64, 64, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, 0.5f, 0.0f, 1},
        {"sgemm NT beta", N, T, 65, 65, 17, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, -1.0f, 2.0f, 1},
        {"sgemm TT", T, T, 3, 200, 300, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, 1.0f, 1.0f, 1},
        {"gemmEx f16 -> f32", T, N, 96, 40, 256, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F, 1.0f, 0.0f, 1},
        {"gemmEx f16 -> f16", T, N, 64, 64, 64, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, 1.0f, 0.0f, 1},
        {"gemmEx f16 tn fast", T, N, 200, 137, 72, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, 0.5f, -1.0f, 1,
         CUBLAS_COMPUTE_32F, true},
        {"gemmEx f16 tn fast f32 C", T, N, 130, 129, 37, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F, 1.0f, 0.25f, 1,
         CUBLAS_COMPUTE_32F, true},
        {"gemmEx f16 tn 16F", T, N, 257, 130, 300, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, 1.0f, 0.5f, 1,
         CUBLAS_COMPUTE_16F, true},
        {"gemmEx f16 tn 16F f32 C", T, N, 64, 200, 45, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F, -2.0f, 0.0f, 1,
         CUBLAS_COMPUTE_32F_FAST_16F, true},
        {"strided f16 tn 16F", T, N, 128, 64, 128, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, 1.0f, 0.0f, 7,
         CUBLAS_COMPUTE_16F, true},
        {"gemmEx f16 tn odd ld", T, N, 130, 129, 37, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F, 1.0f, 0.25f, 1},
        {"gemmEx bf16 -> f32", N, N, 50, 70, 90, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_32F, 1.0f, 0.0f, 1},
        {"gemmEx f16 x f32", T, N, 33, 31, 100, CUDA_R_16F, CUDA_R_32F, CUDA_R_32F, 1.0f, 0.0f, 1},
        {"strided batched f16", T, N, 64, 16, 128, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, 1.0f, 0.0f, 12},
        {"pointer batched f32", T, N, 40, 24, 64, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, 1.0f, 0.0f, -5},
        {"gemv NN f32", N, N, 300, 1, 77, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, 1.0f, 0.0f, 1},
        {"gemv TN f32 beta", T, N, 130, 1, 1000, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, 0.5f, -2.0f, 1},
        {"gemv NT f16", N, T, 65, 1, 200, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, 1.0f, 0.0f, 1},
        {"gemv TT bf16 -> f32", T, T, 70, 1, 129, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_32F, 1.0f, 1.0f, 1},
        {"gemv m=1 NN", N, N, 1, 333, 90, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, 1.0f, 0.5f, 1},
        {"gemv m=1 TT f16", T, T, 1, 70, 257, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F, -1.0f, 0.0f, 1},
        {"gemv m=1 NT", N, T, 1, 40, 64, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, 1.0f, 0.0f, 1},
        {"gemv m=1 TN f16 x f32", T, N, 1, 50, 31, CUDA_R_16F, CUDA_R_32F, CUDA_R_32F, 2.0f, 0.0f, 1},
        {"gemv 1x1", T, N, 1, 1, 500, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, 1.0f, 0.0f, 1},
        {"gemv strided batched f16", T, N, 129, 1, 128, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, 1.0f, 0.0f, 9,
         CUBLAS_COMPUTE_16F},
        {"gemv f16 aligned rows", T, N, 100, 1, 203, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F, 1.0f, 0.5f, 3,
         CUBLAS_COMPUTE_32F, true},
        {"gemv f16 long rows", T, N, 33, 1, 3000, CUDA_R_16F, CUDA_R_32F, CUDA_R_16F, 1.0f, 0.0f, 1,
         CUBLAS_COMPUTE_32F, true},
        {"gemv pointer batched", T, N, 100, 1, 64, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, 1.0f, 0.0f, -6},
        {"gemv m=1 pointer batched", N, N, 1, 100, 64, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, 1.0f, 0.0f, -4},
        {"k = 0 scales C", N, N, 10, 10, 0, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, 1.0f, 3.0f, 1},
    };
    for (const Case& c : cases) runCase(h, c);
    testTrsm(h);
    testDgemm(h);
    testGrouped(h);
    testSgemv(h);
    testLt(h);

    CHECK_BLAS(cublasDestroy(h));
    CHECK_CU(cuCtxDestroy(ctx));
    return g_failures ? 1 : 0;
}
