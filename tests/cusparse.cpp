// Vulkanizer's libcusparse against CPU references: SpMV and SpMM (CSR with 32- and 64-bit indices,
// COO, transposed A and B, both dense orders, batched), SDDMM, sparse triangular solves (SpSV,
// SpSM, both triangles, transposed, unit diagonal), csrgeam2, SpGEMM, COO to CSR and sorting; f32
// and f64 (where the device has it).

#include "test_util.h"

#include <cusparse/cusparse.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#define CHECK_SP(call)                                                                             \
    do {                                                                                           \
        cusparseStatus_t s_ = (call);                                                              \
        if (s_ != CUSPARSE_STATUS_SUCCESS) {                                                       \
            std::fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__, #call,              \
                         cusparseGetErrorName(s_));                                                \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

namespace {

std::mt19937 rng(5);
int g_failures = 0;
bool g_f64 = false;

void report(const std::string& name, double err, double tol) {
    const bool ok = err <= tol;
    std::printf("  %-44s max error %.2e%s\n", name.c_str(), err, ok ? "" : "  FAILED");
    if (!ok) ++g_failures;
}

// A random sparse matrix on the host (CSR, sorted columns, zero-based).
struct Sparse {
    int rows = 0, cols = 0;
    std::vector<int64_t> offsets, columns;
    std::vector<double> values;
    int nnz() const { return int(columns.size()); }
    double at(int i, int j) const {
        for (int64_t t = offsets[size_t(i)]; t < offsets[size_t(i + 1)]; ++t)
            if (columns[size_t(t)] == j) return values[size_t(t)];
        return 0;
    }
};

Sparse randomSparse(int rows, int cols, double density, int triangle = 0 /* 1 lower, 2 upper */) {
    std::uniform_real_distribution<double> u(-1, 1), pick(0, 1);
    Sparse s;
    s.rows = rows, s.cols = cols;
    s.offsets.push_back(0);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            const bool diagonal = triangle && i == j;
            if (triangle == 1 && j > i) continue;
            if (triangle == 2 && j < i) continue;
            if (diagonal || pick(rng) < density) {
                s.columns.push_back(j);
                s.values.push_back(diagonal ? 2 + pick(rng) : u(rng));
            }
        }
        s.offsets.push_back(int64_t(s.columns.size()));
    }
    return s;
}

std::vector<double> randomDense(size_t count) {
    std::uniform_real_distribution<double> u(-1, 1);
    std::vector<double> v(count);
    for (double& x : v) x = u(rng);
    return v;
}

// Device arrays of a given element size.
CUdeviceptr uploadIdx(const std::vector<int64_t>& v, bool wide, int64_t add = 0) {
    CUdeviceptr d;
    const size_t size = std::max<size_t>(v.size(), 1) * (wide ? 8 : 4);
    CHECK_CU(cuMemAlloc(&d, size));
    if (wide) {
        std::vector<int64_t> w(v);
        for (auto& x : w) x += add;
        if (!w.empty()) CHECK_CU(cuMemcpyHtoD(d, w.data(), w.size() * 8));
    } else {
        std::vector<int32_t> w(v.begin(), v.end());
        for (auto& x : w) x += int32_t(add);
        if (!w.empty()) CHECK_CU(cuMemcpyHtoD(d, w.data(), w.size() * 4));
    }
    return d;
}
CUdeviceptr uploadVal(const std::vector<double>& v, bool f64) {
    CUdeviceptr d;
    CHECK_CU(cuMemAlloc(&d, std::max<size_t>(v.size(), 1) * (f64 ? 8 : 4)));
    if (v.empty()) return d;
    if (f64) CHECK_CU(cuMemcpyHtoD(d, v.data(), v.size() * 8));
    else {
        std::vector<float> f(v.begin(), v.end());
        CHECK_CU(cuMemcpyHtoD(d, f.data(), f.size() * 4));
    }
    return d;
}
std::vector<double> downloadVal(CUdeviceptr d, size_t count, bool f64) {
    std::vector<double> v(count);
    if (f64) CHECK_CU(cuMemcpyDtoH(v.data(), d, count * 8));
    else {
        std::vector<float> f(count);
        CHECK_CU(cuMemcpyDtoH(f.data(), d, count * 4));
        std::copy(f.begin(), f.end(), v.begin());
    }
    return v;
}
std::vector<int> downloadInt(CUdeviceptr d, size_t count) {
    std::vector<int> v(count);
    if (count) CHECK_CU(cuMemcpyDtoH(v.data(), d, count * 4));
    return v;
}

double maxDiff(const std::vector<double>& a, const std::vector<double>& b) {
    double e = 0;
    for (size_t i = 0; i < a.size(); ++i) e = std::max(e, std::fabs(a[i] - b[i]));
    return e;
}

// Element (i, j) of a dense rows x cols matrix with leading dimension ld.
size_t at(int i, int j, int ld, bool rowMajor) { return rowMajor ? size_t(i) * ld + j : i + size_t(j) * ld; }

struct Device {
    std::vector<CUdeviceptr> ptrs;
    CUdeviceptr keep(CUdeviceptr p) { return ptrs.push_back(p), p; }
    ~Device() {
        for (CUdeviceptr p : ptrs) cuMemFree(p);
    }
};

// C = alpha op(A) op(B) + beta C (SpMM; SpMV for n = 1 through vectors).
void testSpmm(cusparseHandle_t h, const std::string& name, bool f64, int m, int k, int n, bool transA, bool transB,
              bool rowMajor, bool wide, bool coo, int batch, bool vector) {
    const cudaDataType type = f64 ? CUDA_R_64F : CUDA_R_32F;
    const int aRows = transA ? k : m, aCols = transA ? m : k;
    std::vector<Sparse> as;
    for (int b = 0; b < batch; ++b) as.push_back(randomSparse(aRows, aCols, 0.3));
    // batch entries share the sparsity pattern size: equal nnz is not needed, entries are padded
    int maxNnz = 0;
    for (auto& a : as) maxNnz = std::max(maxNnz, a.nnz());
    std::vector<int64_t> offsets, columns, rowsCoo;
    std::vector<double> values;
    for (auto& a : as) {
        for (int64_t o : a.offsets) offsets.push_back(o);
        for (int t = 0; t < maxNnz; ++t) {
            columns.push_back(t < a.nnz() ? a.columns[size_t(t)] : 0);
            values.push_back(t < a.nnz() ? a.values[size_t(t)] : 0);
        }
        for (int i = 0; i < a.rows; ++i)
            for (int64_t t = a.offsets[size_t(i)]; t < a.offsets[size_t(i + 1)]; ++t) rowsCoo.push_back(i);
    }
    const int bRows = transB ? n : k, bCols = transB ? k : n;
    const int ldb = rowMajor ? bCols : bRows, ldc = rowMajor ? n : m;
    const size_t bSize = size_t(bRows) * bCols, cSize = size_t(m) * n;
    const std::vector<double> bh = randomDense(bSize * batch);
    std::vector<double> ch = randomDense(cSize * batch);
    const double alpha = 1.5, beta = vector ? 0.0 : -0.5;
    Device dev;
    const CUdeviceptr dOff = dev.keep(uploadIdx(coo ? rowsCoo : offsets, wide));
    const CUdeviceptr dCol = dev.keep(uploadIdx(columns, wide)), dVal = dev.keep(uploadVal(values, f64));
    const CUdeviceptr dB = dev.keep(uploadVal(bh, f64)), dC = dev.keep(uploadVal(ch, f64));
    const cusparseIndexType_t it = wide ? CUSPARSE_INDEX_64I : CUSPARSE_INDEX_32I;
    cusparseSpMatDescr_t A;
    if (coo) {
        CHECK_SP(cusparseCreateCoo(&A, aRows, aCols, as[0].nnz(), (void*)dOff, (void*)dCol, (void*)dVal, it,
                                   CUSPARSE_INDEX_BASE_ZERO, type));
    } else {
        CHECK_SP(cusparseCreateCsr(&A, aRows, aCols, as[0].nnz(), (void*)dOff, (void*)dCol, (void*)dVal, it, it,
                                   CUSPARSE_INDEX_BASE_ZERO, type));
        if (batch > 1) CHECK_SP(cusparseCsrSetStridedBatch(A, batch, aRows + 1, maxNnz));
    }
    const auto opA = transA ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
    const auto opB = transB ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
    const float alpha32 = float(alpha), beta32 = float(beta);
    const void* pa = f64 ? (const void*)&alpha : (const void*)&alpha32;
    const void* pb = f64 ? (const void*)&beta : (const void*)&beta32;
    size_t size = 0;
    if (vector) {
        cusparseDnVecDescr_t x, y;
        CHECK_SP(cusparseCreateDnVec(&x, k, (void*)dB, type));
        CHECK_SP(cusparseCreateDnVec(&y, m, (void*)dC, type));
        CHECK_SP(cusparseSpMV_bufferSize(h, opA, pa, A, x, pb, y, type, 0, &size));
        const CUdeviceptr work = size ? dev.keep([&] { CUdeviceptr w; CHECK_CU(cuMemAlloc(&w, size)); return w; }()) : 0;
        CHECK_SP(cusparseSpMV(h, opA, pa, A, x, pb, y, type, 0, (void*)work));
        cusparseDestroyDnVec(x), cusparseDestroyDnVec(y);
    } else {
        cusparseDnMatDescr_t B, C;
        const auto order = rowMajor ? CUSPARSE_ORDER_ROW : CUSPARSE_ORDER_COL;
        CHECK_SP(cusparseCreateDnMat(&B, bRows, bCols, ldb, (void*)dB, type, order));
        CHECK_SP(cusparseCreateDnMat(&C, m, n, ldc, (void*)dC, type, order));
        if (batch > 1) {
            CHECK_SP(cusparseDnMatSetStridedBatch(B, batch, int64_t(bSize)));
            CHECK_SP(cusparseDnMatSetStridedBatch(C, batch, int64_t(cSize)));
        }
        CHECK_SP(cusparseSpMM_bufferSize(h, opA, opB, pa, A, B, pb, C, type, 0, &size));
        const CUdeviceptr work = size ? dev.keep([&] { CUdeviceptr w; CHECK_CU(cuMemAlloc(&w, size)); return w; }()) : 0;
        CHECK_SP(cusparseSpMM(h, opA, opB, pa, A, B, pb, C, type, 0, (void*)work));
        cusparseDestroyDnMat(B), cusparseDestroyDnMat(C);
    }
    cusparseDestroySpMat(A);
    const std::vector<double> got = downloadVal(dC, ch.size(), f64);
    std::vector<double> want = ch;
    for (int b = 0; b < batch; ++b)
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j) {
                double s = 0;
                for (int l = 0; l < k; ++l) {
                    const double av = transA ? as[size_t(b)].at(l, i) : as[size_t(b)].at(i, l);
                    const double bv = bh[b * bSize + (transB ? at(j, l, ldb, rowMajor) : at(l, j, ldb, rowMajor))];
                    s += av * bv;
                }
                double& c = want[b * cSize + at(i, j, ldc, rowMajor)];
                c = alpha * s + beta * c;
            }
    report(name, maxDiff(got, want), f64 ? 1e-12 : 1e-4);
}

void testSddmm(cusparseHandle_t h, bool f64, int m, int k, int n, bool transA, bool rowMajor) {
    const cudaDataType type = f64 ? CUDA_R_64F : CUDA_R_32F;
    Sparse c = randomSparse(m, n, 0.25);
    const int aRows = transA ? k : m, aCols = transA ? m : k;
    const int lda = rowMajor ? aCols : aRows, ldb = rowMajor ? n : k;
    const std::vector<double> ah = randomDense(size_t(m) * k), bh = randomDense(size_t(k) * n);
    Device dev;
    const CUdeviceptr dOff = dev.keep(uploadIdx(c.offsets, false)), dCol = dev.keep(uploadIdx(c.columns, false));
    const CUdeviceptr dVal = dev.keep(uploadVal(c.values, f64)), dA = dev.keep(uploadVal(ah, f64));
    const CUdeviceptr dB = dev.keep(uploadVal(bh, f64));
    cusparseSpMatDescr_t C;
    cusparseDnMatDescr_t A, B;
    const auto order = rowMajor ? CUSPARSE_ORDER_ROW : CUSPARSE_ORDER_COL;
    CHECK_SP(cusparseCreateCsr(&C, m, n, c.nnz(), (void*)dOff, (void*)dCol, (void*)dVal, CUSPARSE_INDEX_32I,
                               CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, type));
    CHECK_SP(cusparseCreateDnMat(&A, aRows, aCols, lda, (void*)dA, type, order));
    CHECK_SP(cusparseCreateDnMat(&B, k, n, ldb, (void*)dB, type, order));
    const double alpha = 2, beta = 0.5;
    const float a32 = 2, b32 = 0.5f;
    const auto opA = transA ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
    CHECK_SP(cusparseSDDMM(h, opA, CUSPARSE_OPERATION_NON_TRANSPOSE, f64 ? (const void*)&alpha : &a32, A, B,
                           f64 ? (const void*)&beta : &b32, C, type, 0, nullptr));
    cusparseDestroySpMat(C), cusparseDestroyDnMat(A), cusparseDestroyDnMat(B);
    const std::vector<double> got = downloadVal(dVal, size_t(c.nnz()), f64);
    std::vector<double> want(c.values);
    for (int i = 0; i < m; ++i)
        for (int64_t t = c.offsets[size_t(i)]; t < c.offsets[size_t(i + 1)]; ++t) {
            const int j = int(c.columns[size_t(t)]);
            double s = 0;
            for (int l = 0; l < k; ++l)
                s += ah[transA ? at(l, i, lda, rowMajor) : at(i, l, lda, rowMajor)] * bh[at(l, j, ldb, rowMajor)];
            want[size_t(t)] = alpha * s + beta * want[size_t(t)];
        }
    report(std::string(f64 ? "D" : "S") + "SDDMM " + std::to_string(m) + "x" + std::to_string(n) + " k=" +
               std::to_string(k) + (transA ? " A^T" : "") + (rowMajor ? " row-major" : ""),
           maxDiff(got, want), f64 ? 1e-12 : 1e-4);
}

// op(A) Y = alpha X with A triangular (the other triangle holds entries to ignore).
void testSolve(cusparseHandle_t h, bool f64, int n, int nrhs, bool upper, bool trans, bool unit, bool coo) {
    const cudaDataType type = f64 ? CUDA_R_64F : CUDA_R_32F;
    Sparse tri = randomSparse(n, n, 0.2, upper ? 2 : 1), noise = randomSparse(n, n, 0.1, upper ? 1 : 2);
    // the full stored matrix: the triangle plus strictly-other-triangle noise
    Sparse a;
    a.rows = a.cols = n;
    a.offsets.push_back(0);
    std::vector<int64_t> rowsCoo;
    for (int i = 0; i < n; ++i) {
        std::vector<std::pair<int64_t, double>> row;
        for (int64_t t = tri.offsets[size_t(i)]; t < tri.offsets[size_t(i + 1)]; ++t)
            row.push_back({tri.columns[size_t(t)], tri.values[size_t(t)]});
        for (int64_t t = noise.offsets[size_t(i)]; t < noise.offsets[size_t(i + 1)]; ++t)
            if (noise.columns[size_t(t)] != i) row.push_back({noise.columns[size_t(t)], 100.0});
        std::sort(row.begin(), row.end());
        for (auto& [j, v] : row) a.columns.push_back(j), a.values.push_back(v), rowsCoo.push_back(i);
        a.offsets.push_back(int64_t(a.columns.size()));
    }
    const std::vector<double> xh = randomDense(size_t(n) * nrhs);
    Device dev;
    const CUdeviceptr dOff = dev.keep(uploadIdx(coo ? rowsCoo : a.offsets, false));
    const CUdeviceptr dCol = dev.keep(uploadIdx(a.columns, false)), dVal = dev.keep(uploadVal(a.values, f64));
    const CUdeviceptr dX = dev.keep(uploadVal(xh, f64)), dY = dev.keep(uploadVal(std::vector<double>(xh.size()), f64));
    cusparseSpMatDescr_t A;
    if (coo)
        CHECK_SP(cusparseCreateCoo(&A, n, n, a.nnz(), (void*)dOff, (void*)dCol, (void*)dVal, CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_BASE_ZERO, type));
    else
        CHECK_SP(cusparseCreateCsr(&A, n, n, a.nnz(), (void*)dOff, (void*)dCol, (void*)dVal, CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, type));
    cusparseFillMode_t fill = upper ? CUSPARSE_FILL_MODE_UPPER : CUSPARSE_FILL_MODE_LOWER;
    cusparseDiagType_t diag = unit ? CUSPARSE_DIAG_TYPE_UNIT : CUSPARSE_DIAG_TYPE_NON_UNIT;
    CHECK_SP(cusparseSpMatSetAttribute(A, CUSPARSE_SPMAT_FILL_MODE, &fill, sizeof fill));
    CHECK_SP(cusparseSpMatSetAttribute(A, CUSPARSE_SPMAT_DIAG_TYPE, &diag, sizeof diag));
    const double alpha = 2;
    const float a32 = 2;
    const void* pa = f64 ? (const void*)&alpha : &a32;
    const auto op = trans ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
    size_t size = 0;
    if (nrhs == 1) {
        cusparseDnVecDescr_t x, y;
        cusparseSpSVDescr_t d;
        CHECK_SP(cusparseCreateDnVec(&x, n, (void*)dX, type));
        CHECK_SP(cusparseCreateDnVec(&y, n, (void*)dY, type));
        CHECK_SP(cusparseSpSV_createDescr(&d));
        CHECK_SP(cusparseSpSV_bufferSize(h, op, pa, A, x, y, type, 0, d, &size));
        const CUdeviceptr work = size ? dev.keep([&] { CUdeviceptr w; CHECK_CU(cuMemAlloc(&w, size)); return w; }()) : 0;
        CHECK_SP(cusparseSpSV_analysis(h, op, pa, A, x, y, type, 0, d, (void*)work));
        CHECK_SP(cusparseSpSV_solve(h, op, pa, A, x, y, type, 0, d));
        cusparseSpSV_destroyDescr(d), cusparseDestroyDnVec(x), cusparseDestroyDnVec(y);
    } else {
        cusparseDnMatDescr_t x, y;
        cusparseSpSMDescr_t d;
        CHECK_SP(cusparseCreateDnMat(&x, n, nrhs, n, (void*)dX, type, CUSPARSE_ORDER_COL));
        CHECK_SP(cusparseCreateDnMat(&y, n, nrhs, n, (void*)dY, type, CUSPARSE_ORDER_COL));
        CHECK_SP(cusparseSpSM_createDescr(&d));
        CHECK_SP(cusparseSpSM_bufferSize(h, op, CUSPARSE_OPERATION_NON_TRANSPOSE, pa, A, x, y, type, 0, d, &size));
        const CUdeviceptr work = size ? dev.keep([&] { CUdeviceptr w; CHECK_CU(cuMemAlloc(&w, size)); return w; }()) : 0;
        CHECK_SP(cusparseSpSM_analysis(h, op, CUSPARSE_OPERATION_NON_TRANSPOSE, pa, A, x, y, type, 0, d, (void*)work));
        CHECK_SP(cusparseSpSM_solve(h, op, CUSPARSE_OPERATION_NON_TRANSPOSE, pa, A, x, y, type, 0, d));
        cusparseSpSM_destroyDescr(d), cusparseDestroyDnMat(x), cusparseDestroyDnMat(y);
    }
    cusparseDestroySpMat(A);
    // the residual op(T) Y - alpha X with T the triangle (unit diagonal if asked)
    const std::vector<double> y = downloadVal(dY, xh.size(), f64);
    double worst = 0;
    for (int c = 0; c < nrhs; ++c)
        for (int i = 0; i < n; ++i) {
            double s = 0;
            for (int j = 0; j < n; ++j) {
                const int r = trans ? j : i, q = trans ? i : j;  // T(r, q) = op(T)(i, j)
                double t = (upper ? q >= r : q <= r) ? tri.at(r, q) : 0;
                if (r == q && unit) t = 1;
                s += t * y[size_t(c) * n + j];
            }
            worst = std::max(worst, std::fabs(s - alpha * xh[size_t(c) * n + i]));
        }
    report(std::string(f64 ? "D" : "S") + (nrhs == 1 ? "SpSV " : "SpSM ") + std::to_string(n) + (upper ? " upper" : " lower") +
               (trans ? " A^T" : "") + (unit ? " unit" : "") + (coo ? " COO" : "") +
               (nrhs > 1 ? " nrhs=" + std::to_string(nrhs) : ""),
           worst, f64 ? 1e-11 : 1e-3);
}

void testGeam(cusparseHandle_t h, bool f64, int m, int n, bool oneBased) {
    const Sparse a = randomSparse(m, n, 0.3), b = randomSparse(m, n, 0.3);
    const int base = oneBased;
    Device dev;
    const CUdeviceptr aOff = dev.keep(uploadIdx(a.offsets, false, base)), aCol = dev.keep(uploadIdx(a.columns, false, base));
    const CUdeviceptr bOff = dev.keep(uploadIdx(b.offsets, false, base)), bCol = dev.keep(uploadIdx(b.columns, false, base));
    const CUdeviceptr aVal = dev.keep(uploadVal(a.values, f64)), bVal = dev.keep(uploadVal(b.values, f64));
    cusparseMatDescr_t d;
    CHECK_SP(cusparseCreateMatDescr(&d));
    CHECK_SP(cusparseSetMatIndexBase(d, oneBased ? CUSPARSE_INDEX_BASE_ONE : CUSPARSE_INDEX_BASE_ZERO));
    CUdeviceptr cOff;
    CHECK_CU(cuMemAlloc(&cOff, size_t(m + 1) * 4));
    dev.keep(cOff);
    const double alpha = 1.5, beta = -2;
    const float a32 = 1.5f, b32 = -2;
    size_t size = 0;
    if (f64)
        CHECK_SP(cusparseDcsrgeam2_bufferSizeExt(h, m, n, &alpha, d, a.nnz(), (const double*)aVal, (const int*)aOff,
                                                 (const int*)aCol, &beta, d, b.nnz(), (const double*)bVal,
                                                 (const int*)bOff, (const int*)bCol, d, nullptr, (const int*)cOff,
                                                 nullptr, &size));
    else
        CHECK_SP(cusparseScsrgeam2_bufferSizeExt(h, m, n, &a32, d, a.nnz(), (const float*)aVal, (const int*)aOff,
                                                 (const int*)aCol, &b32, d, b.nnz(), (const float*)bVal,
                                                 (const int*)bOff, (const int*)bCol, d, nullptr, (const int*)cOff,
                                                 nullptr, &size));
    CUdeviceptr work;
    CHECK_CU(cuMemAlloc(&work, size));
    dev.keep(work);
    int nnzC = -1;
    CHECK_SP(cusparseXcsrgeam2Nnz(h, m, n, d, a.nnz(), (const int*)aOff, (const int*)aCol, d, b.nnz(),
                                  (const int*)bOff, (const int*)bCol, d, (int*)cOff, &nnzC, (void*)work));
    CUdeviceptr cCol, cVal;
    CHECK_CU(cuMemAlloc(&cCol, size_t(std::max(nnzC, 1)) * 4));
    CHECK_CU(cuMemAlloc(&cVal, size_t(std::max(nnzC, 1)) * 8));
    dev.keep(cCol), dev.keep(cVal);
    if (f64)
        CHECK_SP(cusparseDcsrgeam2(h, m, n, &alpha, d, a.nnz(), (const double*)aVal, (const int*)aOff,
                                   (const int*)aCol, &beta, d, b.nnz(), (const double*)bVal, (const int*)bOff,
                                   (const int*)bCol, d, (double*)cVal, (int*)cOff, (int*)cCol, (void*)work));
    else
        CHECK_SP(cusparseScsrgeam2(h, m, n, &a32, d, a.nnz(), (const float*)aVal, (const int*)aOff, (const int*)aCol,
                                   &b32, d, b.nnz(), (const float*)bVal, (const int*)bOff, (const int*)bCol, d,
                                   (float*)cVal, (int*)cOff, (int*)cCol, (void*)work));
    cusparseDestroyMatDescr(d);
    const std::vector<int> off = downloadInt(cOff, size_t(m + 1)), col = downloadInt(cCol, size_t(nnzC));
    const std::vector<double> val = downloadVal(cVal, size_t(nnzC), f64);
    // dense comparison, and the structure: the union of both patterns
    double worst = 0;
    int expectedNnz = 0;
    for (int i = 0; i < m; ++i) {
        std::vector<double> row(size_t(n), 0);
        for (int t = off[size_t(i)] - base; t < off[size_t(i + 1)] - base; ++t) row[size_t(col[size_t(t)] - base)] += val[size_t(t)];
        for (int j = 0; j < n; ++j) {
            const bool present = a.at(i, j) != 0 || b.at(i, j) != 0;
            expectedNnz += present;
            worst = std::max(worst, std::fabs(row[size_t(j)] - (alpha * a.at(i, j) + beta * b.at(i, j))));
        }
    }
    if (nnzC != expectedNnz) worst = 1e9;
    report(std::string(f64 ? "D" : "S") + "csrgeam2 " + std::to_string(m) + "x" + std::to_string(n) +
               (oneBased ? " one-based" : "") + " nnz " + std::to_string(nnzC),
           worst, f64 ? 1e-12 : 1e-5);
}

void testSpgemm(cusparseHandle_t h, bool f64, int m, int k, int n) {
    const cudaDataType type = f64 ? CUDA_R_64F : CUDA_R_32F;
    const Sparse a = randomSparse(m, k, 0.2), b = randomSparse(k, n, 0.2);
    Device dev;
    const CUdeviceptr aOff = dev.keep(uploadIdx(a.offsets, true)), aCol = dev.keep(uploadIdx(a.columns, true));
    const CUdeviceptr bOff = dev.keep(uploadIdx(b.offsets, true)), bCol = dev.keep(uploadIdx(b.columns, true));
    const CUdeviceptr aVal = dev.keep(uploadVal(a.values, f64)), bVal = dev.keep(uploadVal(b.values, f64));
    CUdeviceptr cOff;
    CHECK_CU(cuMemAlloc(&cOff, size_t(m + 1) * 8));
    dev.keep(cOff);
    cusparseSpMatDescr_t A, B, C;
    const auto it = CUSPARSE_INDEX_64I;
    CHECK_SP(cusparseCreateCsr(&A, m, k, a.nnz(), (void*)aOff, (void*)aCol, (void*)aVal, it, it, CUSPARSE_INDEX_BASE_ZERO, type));
    CHECK_SP(cusparseCreateCsr(&B, k, n, b.nnz(), (void*)bOff, (void*)bCol, (void*)bVal, it, it, CUSPARSE_INDEX_BASE_ZERO, type));
    CHECK_SP(cusparseCreateCsr(&C, m, n, 0, (void*)cOff, nullptr, nullptr, it, it, CUSPARSE_INDEX_BASE_ZERO, type));
    cusparseSpGEMMDescr_t d;
    CHECK_SP(cusparseSpGEMM_createDescr(&d));
    const double alpha = 0.5, beta = 0;
    const float a32 = 0.5f, b32 = 0;
    const void* pa = f64 ? (const void*)&alpha : &a32;
    const void* pb = f64 ? (const void*)&beta : &b32;
    const auto N = CUSPARSE_OPERATION_NON_TRANSPOSE;
    size_t s1 = 0, s2 = 0;
    CHECK_SP(cusparseSpGEMM_workEstimation(h, N, N, pa, A, B, pb, C, type, 0, d, &s1, nullptr));
    CUdeviceptr w1, w2;
    CHECK_CU(cuMemAlloc(&w1, std::max<size_t>(s1, 1)));
    dev.keep(w1);
    CHECK_SP(cusparseSpGEMM_workEstimation(h, N, N, pa, A, B, pb, C, type, 0, d, &s1, (void*)w1));
    CHECK_SP(cusparseSpGEMM_compute(h, N, N, pa, A, B, pb, C, type, 0, d, &s2, nullptr));
    CHECK_CU(cuMemAlloc(&w2, std::max<size_t>(s2, 1)));
    dev.keep(w2);
    CHECK_SP(cusparseSpGEMM_compute(h, N, N, pa, A, B, pb, C, type, 0, d, &s2, (void*)w2));
    int64_t rows, cols, nnz;
    CHECK_SP(cusparseSpMatGetSize(C, &rows, &cols, &nnz));
    CUdeviceptr cCol, cVal;
    CHECK_CU(cuMemAlloc(&cCol, size_t(std::max<int64_t>(nnz, 1)) * 8));
    CHECK_CU(cuMemAlloc(&cVal, size_t(std::max<int64_t>(nnz, 1)) * 8));
    dev.keep(cCol), dev.keep(cVal);
    CHECK_SP(cusparseCsrSetPointers(C, (void*)cOff, (void*)cCol, (void*)cVal));
    CHECK_SP(cusparseSpGEMM_copy(h, N, N, pa, A, B, pb, C, type, 0, d));
    cusparseSpGEMM_destroyDescr(d), cusparseDestroySpMat(A), cusparseDestroySpMat(B), cusparseDestroySpMat(C);
    std::vector<int64_t> off(static_cast<size_t>(m + 1)), col(static_cast<size_t>(nnz));
    CHECK_CU(cuMemcpyDtoH(off.data(), cOff, off.size() * 8));
    if (nnz) CHECK_CU(cuMemcpyDtoH(col.data(), cCol, col.size() * 8));
    const std::vector<double> val = downloadVal(cVal, size_t(nnz), f64);
    double worst = 0;
    for (int i = 0; i < m; ++i) {
        std::vector<double> row(size_t(n), 0);
        for (int64_t t = off[size_t(i)]; t < off[size_t(i + 1)]; ++t) row[size_t(col[size_t(t)])] += val[size_t(t)];
        for (int j = 0; j < n; ++j) {
            double s = 0;
            for (int l = 0; l < k; ++l) s += a.at(i, l) * b.at(l, j);
            worst = std::max(worst, std::fabs(row[size_t(j)] - alpha * s));
        }
    }
    report(std::string(f64 ? "D" : "S") + "SpGEMM " + std::to_string(m) + "x" + std::to_string(k) + " * " +
               std::to_string(k) + "x" + std::to_string(n) + " nnz " + std::to_string(nnz),
           worst, f64 ? 1e-12 : 1e-5);
}

void testConversions(cusparseHandle_t h) {
    // COO (sorted rows, some empty) to CSR, one-based
    const std::vector<int64_t> rows = {1, 1, 2, 4, 4, 4, 6};
    Device dev;
    const CUdeviceptr dRows = dev.keep(uploadIdx(rows, false));
    CUdeviceptr dOff;
    CHECK_CU(cuMemAlloc(&dOff, 7 * 4));
    dev.keep(dOff);
    CHECK_SP(cusparseXcoo2csr(h, (const int*)dRows, 7, 6, (int*)dOff, CUSPARSE_INDEX_BASE_ONE));
    const std::vector<int> off = downloadInt(dOff, 7), wantOff = {1, 3, 4, 4, 7, 7, 8};
    report("Xcoo2csr one-based", off == wantOff ? 0 : 1, 0);
    // sort an unsorted COO by row then column, with the permutation
    const std::vector<int64_t> r = {3, 0, 2, 0, 3, 1}, c = {1, 4, 0, 2, 0, 3};
    const CUdeviceptr dr = dev.keep(uploadIdx(r, false)), dc = dev.keep(uploadIdx(c, false));
    CUdeviceptr dp, work;
    CHECK_CU(cuMemAlloc(&dp, 6 * 4));
    dev.keep(dp);
    CHECK_SP(cusparseCreateIdentityPermutation(h, 6, (int*)dp));
    size_t size = 0;
    CHECK_SP(cusparseXcoosort_bufferSizeExt(h, 4, 5, 6, (const int*)dr, (const int*)dc, &size));
    CHECK_CU(cuMemAlloc(&work, std::max<size_t>(size, 1)));
    dev.keep(work);
    CHECK_SP(cusparseXcoosortByRow(h, 4, 5, 6, (int*)dr, (int*)dc, (int*)dp, (void*)work));
    const std::vector<int> sr = downloadInt(dr, 6), sc = downloadInt(dc, 6), sp = downloadInt(dp, 6);
    const bool cooOk = sr == std::vector<int>{0, 0, 1, 2, 3, 3} && sc == std::vector<int>{2, 4, 3, 0, 0, 1} &&
                       sp == std::vector<int>{3, 1, 5, 2, 4, 0};
    report("CreateIdentityPermutation + XcoosortByRow", cooOk ? 0 : 1, 0);
    // sort the columns of each CSR row
    const std::vector<int64_t> o = {0, 3, 3, 5}, cc = {4, 0, 2, 1, 0};
    const CUdeviceptr dO = dev.keep(uploadIdx(o, false)), dC = dev.keep(uploadIdx(cc, false));
    CHECK_SP(cusparseCreateIdentityPermutation(h, 5, (int*)dp));
    cusparseMatDescr_t d;
    CHECK_SP(cusparseCreateMatDescr(&d));
    CHECK_SP(cusparseXcsrsort(h, 3, 5, 5, d, (const int*)dO, (int*)dC, (int*)dp, (void*)work));
    cusparseDestroyMatDescr(d);
    const std::vector<int> c2 = downloadInt(dC, 5), p2 = downloadInt(dp, 5);
    report("Xcsrsort", c2 == std::vector<int>{0, 2, 4, 0, 1} && p2 == std::vector<int>{1, 2, 0, 4, 3} ? 0 : 1, 0);
}

}  // namespace

int main() {
    CHECK_CU(cuInit(0));
    CUdevice dev;
    CHECK_CU(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CHECK_CU(cuDevicePrimaryCtxRetain(&ctx, dev));
    CHECK_CU(cuCtxSetCurrent(ctx));
    cusparseHandle_t h;
    CHECK_SP(cusparseCreate(&h));
    {
        cusparseSpMatDescr_t probe;  // f64 needs shaderFloat64: SpMV reports NOT_SUPPORTED otherwise
        CHECK_SP(cusparseCreateCsr(&probe, 1, 1, 0, nullptr, nullptr, nullptr, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
        cusparseDnVecDescr_t v;
        CHECK_SP(cusparseCreateDnVec(&v, 1, nullptr, CUDA_R_64F));
        const double one = 1;
        size_t size;
        g_f64 = cusparseSpMV_bufferSize(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, probe, v, &one, v, CUDA_R_64F, 0,
                                        &size) == CUSPARSE_STATUS_SUCCESS;
        cusparseDestroySpMat(probe), cusparseDestroyDnVec(v);
    }
    for (bool f64 : {false, true}) {
        if (f64 && !g_f64) {
            std::printf("  (no f64 on this device)\n");
            break;
        }
        const std::string t = f64 ? "D" : "S";
        testSpmm(h, t + "SpMV 37x29", f64, 37, 29, 1, false, false, false, false, false, 1, true);
        testSpmm(h, t + "SpMV A^T 37x29 64-bit indices", f64, 37, 29, 1, true, false, false, true, false, 1, true);
        testSpmm(h, t + "SpMM 40x30 * 30x7", f64, 40, 30, 7, false, false, false, false, false, 1, false);
        testSpmm(h, t + "SpMM row-major B^T", f64, 33, 20, 9, false, true, true, false, false, 1, false);
        testSpmm(h, t + "SpMM A^T", f64, 25, 31, 6, true, false, false, false, false, 1, false);
        testSpmm(h, t + "SpMM COO 64-bit", f64, 28, 19, 5, false, false, true, true, true, 1, false);
        testSpmm(h, t + "SpMM COO A^T", f64, 21, 17, 4, true, false, false, false, true, 1, false);
        testSpmm(h, t + "SpMM batched x4", f64, 16, 12, 5, false, false, false, false, false, 4, false);
        testSddmm(h, f64, 18, 7, 21, false, false);
        testSddmm(h, f64, 15, 9, 11, true, true);
        for (bool upper : {false, true})
            for (bool trans : {false, true}) testSolve(h, f64, 60, 1, upper, trans, false, false);
        testSolve(h, f64, 45, 1, false, false, true, false);
        testSolve(h, f64, 50, 3, true, false, false, true);
        testSolve(h, f64, 40, 2, false, true, true, false);
        testGeam(h, f64, 23, 19, false);
        testGeam(h, f64, 17, 25, true);
        testSpgemm(h, f64, 20, 15, 18);
    }
    testConversions(h);
    CHECK_SP(cusparseDestroy(h));
    CHECK_CU(cuDevicePrimaryCtxRelease(dev));
    std::printf(g_failures ? "FAILED\n" : "OK\n");
    return g_failures ? 1 : 0;
}
