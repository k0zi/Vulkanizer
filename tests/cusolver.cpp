// Vulkanizer's libcusolver: LU (getrf/getrs) and Cholesky (potrf/potrs, also batched and through the
// 64-bit API), f32 and f64, checked by the residual of the solved system on the CPU; singular and
// non positive definite matrices report their info. QR and symmetric eigenproblems by their
// defining identities.

#include "test_util.h"

#include <cusolver/cusolverDn.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#define CHECK_SOLVER(call)                                                                         \
    do {                                                                                           \
        cusolverStatus_t s_ = (call);                                                              \
        if (s_ != CUSOLVER_STATUS_SUCCESS) {                                                       \
            std::fprintf(stderr, "%s:%d: %s failed: %d\n", __FILE__, __LINE__, #call, int(s_));    \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

namespace {

std::mt19937 rng(7);

template <class T>
std::vector<T> randomMatrix(int rows, int cols) {
    std::uniform_real_distribution<double> d(-1, 1);
    std::vector<T> v(size_t(rows) * cols);
    for (T& x : v) x = T(d(rng));
    return v;
}

// A symmetric positive definite matrix: M M^T + n I.
template <class T>
std::vector<T> spdMatrix(int n) {
    const std::vector<T> m = randomMatrix<T>(n, n);
    std::vector<T> a(size_t(n) * n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            double s = i == j ? n : 0;
            for (int k = 0; k < n; ++k) s += double(m[i + size_t(k) * n]) * m[j + size_t(k) * n];
            a[i + size_t(j) * n] = T(s);
        }
    return a;
}

template <class T>
CUdeviceptr upload(const std::vector<T>& v) {
    CUdeviceptr d;
    CHECK_CU(cuMemAlloc(&d, v.size() * sizeof(T)));
    CHECK_CU(cuMemcpyHtoD(d, v.data(), v.size() * sizeof(T)));
    return d;
}

template <class T>
std::vector<T> download(CUdeviceptr d, size_t count) {
    std::vector<T> v(count);
    CHECK_CU(cuMemcpyDtoH(v.data(), d, count * sizeof(T)));
    return v;
}

int readInt(CUdeviceptr d) {
    int v = -1;
    CHECK_CU(cuMemcpyDtoH(&v, d, sizeof v));
    return v;
}

// max |op(A) X - B| / max |B|
template <class T>
double residual(const std::vector<T>& a, bool trans, const std::vector<T>& x, const std::vector<T>& b, int n, int nrhs) {
    double worst = 0, scale = 0;
    for (int c = 0; c < nrhs; ++c)
        for (int i = 0; i < n; ++i) {
            double s = 0;
            for (int k = 0; k < n; ++k)
                s += double(trans ? a[k + size_t(i) * n] : a[i + size_t(k) * n]) * x[k + size_t(c) * n];
            worst = std::max(worst, std::fabs(s - b[i + size_t(c) * n]));
            scale = std::max(scale, std::fabs(double(b[i + size_t(c) * n])));
        }
    return worst / scale;
}

template <class T>
cusolverStatus_t getrf(cusolverDnHandle_t h, int n, CUdeviceptr a, CUdeviceptr ipiv, CUdeviceptr info) {
    if constexpr (sizeof(T) == 4) return cusolverDnSgetrf(h, n, n, (float*)a, n, nullptr, (int*)ipiv, (int*)info);
    else return cusolverDnDgetrf(h, n, n, (double*)a, n, nullptr, (int*)ipiv, (int*)info);
}
template <class T>
cusolverStatus_t getrs(cusolverDnHandle_t h, bool trans, int n, int nrhs, CUdeviceptr a, CUdeviceptr ipiv,
                       CUdeviceptr b, CUdeviceptr info) {
    const cublasOperation_t op = trans ? CUBLAS_OP_T : CUBLAS_OP_N;
    if constexpr (sizeof(T) == 4)
        return cusolverDnSgetrs(h, op, n, nrhs, (const float*)a, n, (const int*)ipiv, (float*)b, n, (int*)info);
    else return cusolverDnDgetrs(h, op, n, nrhs, (const double*)a, n, (const int*)ipiv, (double*)b, n, (int*)info);
}

template <class T>
bool testLu(cusolverDnHandle_t h, int n, int nrhs, bool trans, bool pivoting, double tolerance) {
    std::vector<T> a = randomMatrix<T>(n, n);
    if (!pivoting)  // diagonally dominant: stable without row interchanges
        for (int i = 0; i < n; ++i) a[i + size_t(i) * n] += T(n);
    const std::vector<T> b = randomMatrix<T>(n, nrhs);
    const CUdeviceptr da = upload(a), db = upload(b);
    CUdeviceptr ipiv = 0, info;
    if (pivoting) CHECK_CU(cuMemAlloc(&ipiv, n * sizeof(int)));
    CHECK_CU(cuMemAlloc(&info, sizeof(int)));
    CHECK_SOLVER(getrf<T>(h, n, da, ipiv, info));
    const int factorInfo = readInt(info);
    CHECK_SOLVER(getrs<T>(h, trans, n, nrhs, da, ipiv, db, info));
    const double r = residual(a, trans, download<T>(db, b.size()), b, n, n ? nrhs : 0);
    const bool ok = factorInfo == 0 && readInt(info) == 0 && r < tolerance;
    std::printf("  %s getrf/getrs n=%d nrhs=%d%s%s: residual %.2e%s\n", sizeof(T) == 4 ? "S" : "D", n, nrhs,
                trans ? " trans" : "", pivoting ? "" : " no pivoting", r, ok ? "" : "  FAILED");
    cuMemFree(da), cuMemFree(db), cuMemFree(info);
    if (ipiv) cuMemFree(ipiv);
    return ok;
}

template <class T>
bool testCholesky(cusolverDnHandle_t h, int n, int nrhs, bool upper, double tolerance) {
    const std::vector<T> a = spdMatrix<T>(n), b = randomMatrix<T>(n, nrhs);
    const CUdeviceptr da = upload(a), db = upload(b);
    CUdeviceptr info;
    CHECK_CU(cuMemAlloc(&info, sizeof(int)));
    const cublasFillMode_t uplo = upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
    if constexpr (sizeof(T) == 4) {
        CHECK_SOLVER(cusolverDnSpotrf(h, uplo, n, (float*)da, n, nullptr, 0, (int*)info));
        CHECK_SOLVER(cusolverDnSpotrs(h, uplo, n, nrhs, (const float*)da, n, (float*)db, n, (int*)info));
    } else {  // through the 64-bit API
        CHECK_SOLVER(cusolverDnXpotrf(h, nullptr, uplo, n, CUDA_R_64F, (void*)da, n, CUDA_R_64F, nullptr, 0, nullptr,
                                      0, (int*)info));
        CHECK_SOLVER(cusolverDnXpotrs(h, nullptr, uplo, n, nrhs, CUDA_R_64F, (const void*)da, n, CUDA_R_64F,
                                      (void*)db, n, (int*)info));
    }
    const double r = residual(a, false, download<T>(db, b.size()), b, n, nrhs);
    const bool ok = readInt(info) == 0 && r < tolerance;
    std::printf("  %s potrf/potrs n=%d nrhs=%d %s: residual %.2e%s\n", sizeof(T) == 4 ? "S" : "D (64-bit API)", n, nrhs,
                upper ? "upper" : "lower", r, ok ? "" : "  FAILED");
    cuMemFree(da), cuMemFree(db), cuMemFree(info);
    return ok;
}

// Batched Cholesky through device arrays of pointers.
bool testCholeskyBatched(cusolverDnHandle_t h, int n, int batch) {
    std::vector<std::vector<float>> a(batch), b(batch);
    std::vector<CUdeviceptr> da(batch), db(batch);
    for (int i = 0; i < batch; ++i) {
        a[i] = spdMatrix<float>(n);
        b[i] = randomMatrix<float>(n, 1);
        da[i] = upload(a[i]);
        db[i] = upload(b[i]);
    }
    const CUdeviceptr aArray = upload(da), bArray = upload(db);
    CUdeviceptr info, solveInfo;
    CHECK_CU(cuMemAlloc(&info, batch * sizeof(int)));
    CHECK_CU(cuMemAlloc(&solveInfo, sizeof(int)));
    CHECK_SOLVER(cusolverDnSpotrfBatched(h, CUBLAS_FILL_MODE_LOWER, n, (float**)aArray, n, (int*)info, batch));
    CHECK_SOLVER(cusolverDnSpotrsBatched(h, CUBLAS_FILL_MODE_LOWER, n, 1, (float**)aArray, n, (float**)bArray, n,
                                         (int*)solveInfo, batch));
    const std::vector<int> infos = download<int>(info, batch);
    double worst = 0;
    bool ok = readInt(solveInfo) == 0;
    for (int i = 0; i < batch; ++i) {
        ok = ok && infos[i] == 0;
        worst = std::max(worst, residual(a[i], false, download<float>(db[i], n), b[i], n, 1));
        cuMemFree(da[i]), cuMemFree(db[i]);
    }
    ok = ok && worst < 1e-4;
    std::printf("  S potrfBatched/potrsBatched n=%d batch=%d: residual %.2e%s\n", n, batch, worst, ok ? "" : "  FAILED");
    cuMemFree(aArray), cuMemFree(bArray), cuMemFree(info), cuMemFree(solveInfo);
    return ok;
}

// geqrf + orgqr: Q^T Q = I and Q R = A for an m x n A (m >= n); ormqr: Q^T b against the explicit Q.
template <class T>
bool testQr(cusolverDnHandle_t h, int m, int n, double tolerance) {
    const std::vector<T> a = randomMatrix<T>(m, n), b = randomMatrix<T>(m, 2);
    const CUdeviceptr da = upload(a), db = upload(b);
    CUdeviceptr tau, info;
    CHECK_CU(cuMemAlloc(&tau, n * sizeof(T)));
    CHECK_CU(cuMemAlloc(&info, sizeof(int)));
    if constexpr (sizeof(T) == 4) {
        CHECK_SOLVER(cusolverDnSgeqrf(h, m, n, (float*)da, m, (float*)tau, nullptr, 0, (int*)info));
        CHECK_SOLVER(cusolverDnSormqr(h, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, m, 2, n, (const float*)da, m, (const float*)tau,
                                      (float*)db, m, nullptr, 0, (int*)info));
    } else {  // through the 64-bit API
        CHECK_SOLVER(cusolverDnXgeqrf(h, nullptr, m, n, CUDA_R_64F, (void*)da, m, CUDA_R_64F, (void*)tau, CUDA_R_64F,
                                      nullptr, 0, nullptr, 0, (int*)info));
        CHECK_SOLVER(cusolverDnDormqr(h, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, m, 2, n, (const double*)da, m,
                                      (const double*)tau, (double*)db, m, nullptr, 0, (int*)info));
    }
    const std::vector<T> factored = download<T>(da, a.size()), qtb = download<T>(db, b.size());
    if constexpr (sizeof(T) == 4)
        CHECK_SOLVER(cusolverDnSorgqr(h, m, n, n, (float*)da, m, (const float*)tau, nullptr, 0, (int*)info));
    else CHECK_SOLVER(cusolverDnDorgqr(h, m, n, n, (double*)da, m, (const double*)tau, nullptr, 0, (int*)info));
    const std::vector<T> q = download<T>(da, a.size());
    double orth = 0, recon = 0, apply = 0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {  // (Q^T Q)(i, j)
            double s = 0;
            for (int r = 0; r < m; ++r) s += double(q[r + size_t(i) * m]) * q[r + size_t(j) * m];
            orth = std::max(orth, std::fabs(s - (i == j)));
        }
    for (int r = 0; r < m; ++r)
        for (int j = 0; j < n; ++j) {  // (Q R)(r, j), R = the upper triangle of the factored A
            double s = 0;
            for (int i = 0; i <= j; ++i) s += double(q[r + size_t(i) * m]) * factored[i + size_t(j) * m];
            recon = std::max(recon, std::fabs(s - a[r + size_t(j) * m]));
        }
    for (int c = 0; c < 2; ++c)
        for (int i = 0; i < n; ++i) {  // (Q^T b)(i) for the first n rows
            double s = 0;
            for (int r = 0; r < m; ++r) s += double(q[r + size_t(i) * m]) * b[r + size_t(c) * m];
            apply = std::max(apply, std::fabs(s - qtb[i + size_t(c) * m]));
        }
    const bool ok = readInt(info) == 0 && orth < tolerance && recon < tolerance && apply < tolerance;
    std::printf("  %s geqrf/orgqr/ormqr %dx%d: |Q^T Q - I| %.2e, |QR - A| %.2e, |ormqr - Q^T b| %.2e%s\n",
                sizeof(T) == 4 ? "S" : "D (64-bit geqrf)", m, n, orth, recon, apply, ok ? "" : "  FAILED");
    cuMemFree(da), cuMemFree(db), cuMemFree(tau), cuMemFree(info);
    return ok;
}

// Symmetric eigenproblems of `batch` random n x n matrices, given by their lower or upper triangle
// (the other one holds garbage): f32 through syevjBatched, f64 through the 64-bit Xsyevd (one
// matrix) or XsyevBatched; checked by |A V - V diag(W)| / |A|, |V^T V - I|, the ascending order,
// and the eigenvalues alone (NOVECTOR) against the ones with vectors.
template <class T>
bool testEig(cusolverDnHandle_t h, int n, int batch, bool upper, double tolerance) {
    std::vector<T> a = randomMatrix<T>(n, n * batch), garbage = a;
    const size_t nn = size_t(n) * n;
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                const T v = a[b * nn + std::max(i, j) + size_t(std::min(i, j)) * n];  // the lower triangle's value
                a[b * nn + i + size_t(j) * n] = v;
                if (upper ? i <= j : i >= j) garbage[b * nn + i + size_t(j) * n] = v;
                else garbage[b * nn + i + size_t(j) * n] = T(1e3);
            }
    const cublasFillMode_t uplo = upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
    std::vector<T> w[2], v;
    for (int vectors = 1; vectors >= 0; --vectors) {
        const cusolverEigMode_t jobz = vectors ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
        const CUdeviceptr da = upload(garbage);
        CUdeviceptr dw, info, work;
        CHECK_CU(cuMemAlloc(&dw, size_t(n) * batch * sizeof(T)));
        CHECK_CU(cuMemAlloc(&info, batch * sizeof(int)));
        if constexpr (sizeof(T) == 4) {
            syevjInfo_t params;
            CHECK_SOLVER(cusolverDnCreateSyevjInfo(&params));
            int lwork = 0;
            CHECK_SOLVER(cusolverDnSsyevjBatched_bufferSize(h, jobz, uplo, n, (const float*)da, n, (const float*)dw,
                                                            &lwork, params, batch));
            CHECK_CU(cuMemAlloc(&work, size_t(lwork) * sizeof(T)));
            CHECK_SOLVER(cusolverDnSsyevjBatched(h, jobz, uplo, n, (float*)da, n, (float*)dw, (float*)work, lwork,
                                                 (int*)info, params, batch));
            CHECK_SOLVER(cusolverDnDestroySyevjInfo(params));
        } else {
            size_t device = 0, host = 0;
            if (batch == 1) {
                CHECK_SOLVER(cusolverDnXsyevd_bufferSize(h, nullptr, jobz, uplo, n, CUDA_R_64F, (const void*)da, n,
                                                         CUDA_R_64F, (const void*)dw, CUDA_R_64F, &device, &host));
                CHECK_CU(cuMemAlloc(&work, device));
                CHECK_SOLVER(cusolverDnXsyevd(h, nullptr, jobz, uplo, n, CUDA_R_64F, (void*)da, n, CUDA_R_64F,
                                              (void*)dw, CUDA_R_64F, (void*)work, device, nullptr, 0, (int*)info));
            } else {
                CHECK_SOLVER(cusolverDnXsyevBatched_bufferSize(h, nullptr, jobz, uplo, n, CUDA_R_64F, (const void*)da,
                                                               n, CUDA_R_64F, (const void*)dw, CUDA_R_64F, &device,
                                                               &host, batch));
                CHECK_CU(cuMemAlloc(&work, device));
                CHECK_SOLVER(cusolverDnXsyevBatched(h, nullptr, jobz, uplo, n, CUDA_R_64F, (void*)da, n, CUDA_R_64F,
                                                    (void*)dw, CUDA_R_64F, (void*)work, device, nullptr, 0,
                                                    (int*)info, batch));
            }
        }
        w[vectors] = download<T>(dw, size_t(n) * batch);
        if (vectors) v = download<T>(da, a.size());
        for (int b = 0; b < batch; ++b)
            if (readInt(info + b * sizeof(int)) != 0) return std::printf("  eig: info %d FAILED\n", b), false;
        cuMemFree(da), cuMemFree(dw), cuMemFree(info), cuMemFree(work);
    }
    double res = 0, orth = 0, values = 0, scale = 0;
    bool sorted = true;
    for (int b = 0; b < batch; ++b) {
        const T* A = &a[b * nn];
        const T* V = &v[b * nn];
        const T* W = &w[1][size_t(b) * n];
        for (size_t i = 0; i < nn; ++i) scale = std::max(scale, std::fabs(double(A[i])));
        for (int j = 0; j < n; ++j) {
            if (j && W[j] < W[j - 1]) sorted = false;
            values = std::max(values, std::fabs(double(W[j]) - w[0][size_t(b) * n + j]));
            for (int i = 0; i < n; ++i) {
                double av = 0, vv = 0;
                for (int k = 0; k < n; ++k) {
                    av += double(A[i + size_t(k) * n]) * V[k + size_t(j) * n];
                    vv += double(V[k + size_t(i) * n]) * V[k + size_t(j) * n];
                }
                res = std::max(res, std::fabs(av - double(V[i + size_t(j) * n]) * W[j]));
                orth = std::max(orth, std::fabs(vv - (i == j)));
            }
        }
    }
    res /= scale;
    values /= scale;
    const bool ok = sorted && res < tolerance && orth < tolerance && values < tolerance;
    std::printf("  %s %d x %dx%d %s: |AV - VW| %.2e, |V^T V - I| %.2e, |W - W(novector)| %.2e%s%s\n",
                sizeof(T) == 4 ? "SsyevjBatched" : batch == 1 ? "Xsyevd" : "XsyevBatched", batch, n, n,
                upper ? "upper" : "lower", res, orth, values, sorted ? "" : ", unsorted", ok ? "" : "  FAILED");
    return ok;
}

enum class Svd { Gesvdj, GesvdjBatched, Gesvd };

// A = U diag(S) V^T of `batch` random m x n matrices (`rank` < min(m, n): a product of random
// m x rank and rank x n factors), checked by |U S V^T - A| / |A|, the orthonormality of U and V
// (all of them, also the completion for full and for zero singular values), the descending order,
// and the singular values alone (NOVECTOR, or gesvd's jobs 'N') against the ones with vectors.
template <class T>
bool testSvd(cusolverDnHandle_t h, Svd api, int m, int n, int batch, bool full, int rank, double tolerance) {
    const int k = std::min(m, n), uc = full ? m : k, vc = full || api == Svd::Gesvd ? n : k;
    const size_t mn = size_t(m) * n;
    std::vector<T> a = randomMatrix<T>(m, n * batch);
    if (rank < k) {
        for (int b = 0; b < batch; ++b) {
            const std::vector<T> x = randomMatrix<T>(m, rank), y = randomMatrix<T>(rank, n);
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < m; ++i) {
                    double s = 0;
                    for (int l = 0; l < rank; ++l) s += double(x[i + size_t(l) * m]) * y[l + size_t(j) * rank];
                    a[b * mn + i + size_t(j) * m] = T(s);
                }
        }
    }
    std::vector<T> sv[2], u, v;
    for (int vectors = 1; vectors >= 0; --vectors) {
        const CUdeviceptr da = upload(a);
        CUdeviceptr ds, du, dv, info, work;
        CHECK_CU(cuMemAlloc(&ds, size_t(k) * batch * sizeof(T)));
        CHECK_CU(cuMemAlloc(&du, size_t(m) * uc * batch * sizeof(T)));
        CHECK_CU(cuMemAlloc(&dv, size_t(n) * vc * batch * sizeof(T)));
        CHECK_CU(cuMemAlloc(&info, batch * sizeof(int)));
        const cusolverEigMode_t jobz = vectors ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
        T *A = (T*)da, *S = (T*)ds, *U = (T*)du, *V = (T*)dv;
        int lwork = 0;
        gesvdjInfo_t params;
        CHECK_SOLVER(cusolverDnCreateGesvdjInfo(&params));
        if (api == Svd::Gesvdj) {
            if constexpr (sizeof(T) == 4)
                CHECK_SOLVER(cusolverDnSgesvdj_bufferSize(h, jobz, !full, m, n, A, m, S, U, m, V, n, &lwork, params));
            else CHECK_SOLVER(cusolverDnDgesvdj_bufferSize(h, jobz, !full, m, n, A, m, S, U, m, V, n, &lwork, params));
        } else if (api == Svd::GesvdjBatched) {
            if constexpr (sizeof(T) == 4)
                CHECK_SOLVER(cusolverDnSgesvdjBatched_bufferSize(h, jobz, m, n, A, m, S, U, m, V, n, &lwork, params,
                                                                 batch));
            else CHECK_SOLVER(cusolverDnDgesvdjBatched_bufferSize(h, jobz, m, n, A, m, S, U, m, V, n, &lwork, params,
                                                                  batch));
        } else {
            if constexpr (sizeof(T) == 4) CHECK_SOLVER(cusolverDnSgesvd_bufferSize(h, m, n, &lwork));
            else CHECK_SOLVER(cusolverDnDgesvd_bufferSize(h, m, n, &lwork));
        }
        CHECK_CU(cuMemAlloc(&work, size_t(lwork) * sizeof(T)));
        T* W = (T*)work;
        int* I = (int*)info;
        const signed char jobu = vectors ? (full ? 'A' : 'S') : 'N', jobvt = vectors ? 'A' : 'N';
        if constexpr (sizeof(T) == 4) {
            if (api == Svd::Gesvdj)
                CHECK_SOLVER(cusolverDnSgesvdj(h, jobz, !full, m, n, A, m, S, U, m, V, n, W, lwork, I, params));
            else if (api == Svd::GesvdjBatched)
                CHECK_SOLVER(cusolverDnSgesvdjBatched(h, jobz, m, n, A, m, S, U, m, V, n, W, lwork, I, params, batch));
            else CHECK_SOLVER(cusolverDnSgesvd(h, jobu, jobvt, m, n, A, m, S, U, m, V, n, W, lwork, nullptr, I));
        } else {
            if (api == Svd::Gesvdj)
                CHECK_SOLVER(cusolverDnDgesvdj(h, jobz, !full, m, n, A, m, S, U, m, V, n, W, lwork, I, params));
            else if (api == Svd::GesvdjBatched)
                CHECK_SOLVER(cusolverDnDgesvdjBatched(h, jobz, m, n, A, m, S, U, m, V, n, W, lwork, I, params, batch));
            else CHECK_SOLVER(cusolverDnDgesvd(h, jobu, jobvt, m, n, A, m, S, U, m, V, n, W, lwork, nullptr, I));
        }
        CHECK_SOLVER(cusolverDnDestroyGesvdjInfo(params));
        sv[vectors] = download<T>(ds, size_t(k) * batch);
        if (vectors) u = download<T>(du, size_t(m) * uc * batch), v = download<T>(dv, size_t(n) * vc * batch);
        for (int b = 0; b < batch; ++b)
            if (readInt(info + b * sizeof(int)) != 0) return std::printf("  svd: info %d FAILED\n", b), false;
        cuMemFree(da), cuMemFree(ds), cuMemFree(du), cuMemFree(dv), cuMemFree(info), cuMemFree(work);
    }
    const bool vt = api == Svd::Gesvd;
    double recon = 0, orthU = 0, orthV = 0, values = 0, scale = 0;
    bool sorted = true;
    for (int b = 0; b < batch; ++b) {
        const T* A = &a[b * mn];
        const T* U = &u[size_t(b) * m * uc];
        const T* V = &v[size_t(b) * n * vc];
        const T* S = &sv[1][size_t(b) * k];
        const auto vel = [&](int i, int j) { return double(vt ? V[j + size_t(i) * n] : V[i + size_t(j) * n]); };
        for (size_t i = 0; i < mn; ++i) scale = std::max(scale, std::fabs(double(A[i])));
        for (int j = 0; j < k; ++j) {
            if (S[j] < 0 || (j && S[j] > S[j - 1])) sorted = false;
            values = std::max(values, std::fabs(double(S[j]) - sv[0][size_t(b) * k + j]));
        }
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) {
                double s = 0;
                for (int l = 0; l < k; ++l) s += double(U[i + size_t(l) * m]) * S[l] * vel(j, l);
                recon = std::max(recon, std::fabs(s - A[i + size_t(j) * m]));
            }
        for (int i = 0; i < uc; ++i)
            for (int j = 0; j < uc; ++j) {
                double s = 0;
                for (int l = 0; l < m; ++l) s += double(U[l + size_t(i) * m]) * U[l + size_t(j) * m];
                orthU = std::max(orthU, std::fabs(s - (i == j)));
            }
        for (int i = 0; i < vc; ++i)
            for (int j = 0; j < vc; ++j) {
                double s = 0;
                for (int l = 0; l < n; ++l) s += vel(l, i) * vel(l, j);
                orthV = std::max(orthV, std::fabs(s - (i == j)));
            }
    }
    recon /= scale;
    values /= scale;
    const bool ok = sorted && recon < tolerance && orthU < tolerance && orthV < tolerance && values < tolerance;
    const char* name = api == Svd::Gesvdj ? "gesvdj" : api == Svd::GesvdjBatched ? "gesvdjBatched" : "gesvd";
    std::printf("  %s%s %d x %dx%d%s%s: |USV^T - A| %.2e, |U^T U - I| %.2e, |V^T V - I| %.2e, |S - S(novector)| "
                "%.2e%s%s\n",
                sizeof(T) == 4 ? "S" : "D", name, batch, m, n, full ? " full" : "",
                rank < k ? (" rank " + std::to_string(rank)).c_str() : "", recon, orthU, orthV, values,
                sorted ? "" : ", unsorted", ok ? "" : "  FAILED");
    return ok;
}

bool testInfo(cusolverDnHandle_t h) {
    // singular: the third column is zero, so the third pivot is exactly zero
    const std::vector<float> singular = {2, 1, 0, 1, 4, 0, 0, 0, 0};
    // not positive definite: the second leading minor is negative
    const std::vector<float> indefinite = {1, 2, 0, 2, 1, 0, 0, 0, 1};
    const CUdeviceptr s = upload(singular), d = upload(indefinite);
    CUdeviceptr ipiv, info;
    CHECK_CU(cuMemAlloc(&ipiv, 3 * sizeof(int)));
    CHECK_CU(cuMemAlloc(&info, sizeof(int)));
    CHECK_SOLVER(cusolverDnSgetrf(h, 3, 3, (float*)s, 3, nullptr, (int*)ipiv, (int*)info));
    const int luInfo = readInt(info);
    CHECK_SOLVER(cusolverDnSpotrf(h, CUBLAS_FILL_MODE_LOWER, 3, (float*)d, 3, nullptr, 0, (int*)info));
    const int cholInfo = readInt(info);
    const bool ok = luInfo == 3 && cholInfo == 2;
    std::printf("  info: singular LU %d (want 3), indefinite Cholesky %d (want 2)%s\n", luInfo, cholInfo,
                ok ? "" : "  FAILED");
    cuMemFree(s), cuMemFree(d), cuMemFree(ipiv), cuMemFree(info);
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
    cusolverDnHandle_t h;
    CHECK_SOLVER(cusolverDnCreate(&h));
    bool ok = true;
    ok &= testLu<float>(h, 1, 1, false, true, 1e-5);
    ok &= testLu<float>(h, 17, 3, false, true, 1e-4);
    ok &= testLu<float>(h, 17, 3, true, true, 1e-4);
    ok &= testLu<float>(h, 40, 2, false, false, 1e-4);
    ok &= testLu<float>(h, 300, 4, false, true, 1e-3);
    ok &= testCholesky<float>(h, 33, 2, false, 1e-4);
    ok &= testCholesky<float>(h, 33, 2, true, 1e-4);
    ok &= testCholeskyBatched(h, 8, 50);
    ok &= testInfo(h);
    ok &= testQr<float>(h, 30, 12, 1e-4);
    ok &= testQr<float>(h, 9, 9, 1e-4);
    ok &= testEig<float>(h, 1, 2, false, 1e-5);
    ok &= testEig<float>(h, 7, 5, false, 1e-5);
    ok &= testEig<float>(h, 32, 3, true, 5e-5);
    ok &= testEig<float>(h, 100, 1, false, 1e-4);
    ok &= testEig<float>(h, 300, 1, true, 1e-4);  // one larger matrix: the tridiagonal path
    ok &= testSvd<float>(h, Svd::Gesvdj, 20, 7, 1, false, 7, 1e-5);
    ok &= testSvd<float>(h, Svd::Gesvdj, 20, 7, 1, true, 7, 1e-5);
    ok &= testSvd<float>(h, Svd::Gesvdj, 6, 15, 1, true, 6, 1e-5);
    ok &= testSvd<float>(h, Svd::Gesvdj, 6, 15, 1, false, 6, 1e-5);
    ok &= testSvd<float>(h, Svd::GesvdjBatched, 8, 8, 5, true, 8, 1e-5);
    ok &= testSvd<float>(h, Svd::Gesvd, 12, 5, 1, true, 5, 1e-5);
    ok &= testSvd<float>(h, Svd::Gesvd, 12, 5, 1, false, 5, 1e-5);
    ok &= testSvd<float>(h, Svd::Gesvdj, 10, 6, 1, true, 3, 1e-5);
    ok &= testSvd<float>(h, Svd::Gesvdj, 1, 1, 1, true, 1, 1e-6);
    ok &= testSvd<float>(h, Svd::Gesvdj, 90, 70, 1, false, 70, 1e-4);
    int dev64 = 0;
    {
        CUdeviceptr probe;  // f64 kernels need shaderFloat64: the library reports NOT_SUPPORTED otherwise
        CHECK_CU(cuMemAlloc(&probe, 64));
        dev64 = cusolverDnDgetrf(h, 1, 1, (double*)probe, 1, nullptr, nullptr, (int*)(probe + 32)) ==
                CUSOLVER_STATUS_SUCCESS;
        cuMemFree(probe);
    }
    if (dev64) {
        ok &= testLu<double>(h, 25, 3, false, true, 1e-12);
        ok &= testLu<double>(h, 25, 3, true, true, 1e-12);
        ok &= testCholesky<double>(h, 21, 2, false, 1e-12);
        ok &= testCholesky<double>(h, 21, 2, true, 1e-12);
        ok &= testQr<double>(h, 26, 11, 1e-12);
        ok &= testEig<double>(h, 45, 1, true, 1e-12);
        ok &= testEig<double>(h, 128, 1, false, 1e-12);
        ok &= testEig<double>(h, 9, 4, false, 1e-12);
        ok &= testSvd<double>(h, Svd::Gesvdj, 17, 9, 1, true, 9, 1e-12);
        ok &= testSvd<double>(h, Svd::GesvdjBatched, 5, 9, 3, true, 4, 1e-12);
        ok &= testSvd<double>(h, Svd::Gesvd, 9, 9, 1, false, 9, 1e-12);
    } else {
        std::printf("  (no f64 on this device)\n");
    }
    CHECK_SOLVER(cusolverDnDestroy(h));
    CHECK_CU(cuDevicePrimaryCtxRelease(dev));
    std::printf(ok ? "OK\n" : "FAILED\n");
    return ok ? 0 : 1;
}
