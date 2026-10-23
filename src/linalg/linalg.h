#pragma once

// Dense factorizations and solves on Vulkanizer's own kernels (shaders/*.comp), shared by
// libcusolver and libcublas's batched factorizations. Column-major matrices, f32 or f64; every
// operation is queued on `stream` of the current context and takes a batch of matrices given
// either as a device array of pointers or as a base address with a stride. Pivots and info are
// 1-based, as in LAPACK.

#include <cuda.h>

#include <cstdint>

namespace vulkanizer::linalg {

struct Matrices {
    CUdeviceptr base = 0;    // first matrix, or a device array of matrix pointers
    bool pointerArray = false;
    int64_t stride = 0;      // elements between matrices (strided form)
    int ld = 0;
};

// The current context's device runs f64 kernels.
bool hasFloat64();

// LU with partial pivoting of m x n matrices (getrf); pivots: min(m, n) per matrix, `pivotStride`
// apart, or 0 for no pivoting; info: one int per matrix (0, or the first zero pivot).
CUresult getrf(CUstream stream, bool f64, int m, int n, const Matrices& a, CUdeviceptr pivots, int64_t pivotStride,
               CUdeviceptr info, int batch);
// Solves A X = B (or A^T X = B) with getrf's factors of the n x n A, in place on the n x nrhs B.
CUresult getrs(CUstream stream, bool f64, bool trans, int n, int nrhs, const Matrices& a, CUdeviceptr pivots,
               int64_t pivotStride, const Matrices& b, int batch);
// Cholesky A = L L^T (lower) or U^T U (upper) of n x n matrices; info: 0, or k + 1 if the leading
// minor of order k + 1 is not positive definite.
CUresult potrf(CUstream stream, bool f64, bool upper, int n, const Matrices& a, CUdeviceptr info, int batch);
// Solves A X = B with potrf's factor, in place on the n x nrhs B.
CUresult potrs(CUstream stream, bool f64, bool upper, int n, int nrhs, const Matrices& a, const Matrices& b, int batch);

// Householder QR of m x n matrices (geqrf): R on and above the diagonal, the reflectors below it,
// their scalars in tau (min(m, n) per matrix, `tauStride` apart).
// tauArray: `tau` is a device array of pointers, one per matrix.
CUresult geqrf(CUstream stream, bool f64, int m, int n, const Matrices& a, CUdeviceptr tau, int64_t tauStride, int batch,
               bool tauArray = false);
// The m x n Q = H(0) ... H(k-1) of geqrf's first k reflectors, in place of A (orgqr).
CUresult orgqr(CUstream stream, bool f64, int m, int n, int k, const Matrices& a, CUdeviceptr tau, int64_t tauStride,
               int batch);
// C = op(Q) C (left) or C op(Q) (right) for the m x n C, Q from geqrf's first k reflectors in A (ormqr).
CUresult ormqr(CUstream stream, bool f64, bool right, bool trans, int m, int n, int k, const Matrices& a,
               CUdeviceptr tau, int64_t tauStride, const Matrices& c, int batch);

// Triangular solve in place on the m x n B (trsm): op(A) X = alpha B (left, A m x m) or X op(A) = alpha B
// (right, A n x n), A upper or lower triangular, optionally with a unit diagonal.
CUresult trsm(CUstream stream, bool f64, bool right, bool upper, bool trans, bool unit, int m, int n, double alpha,
              const Matrices& a, const Matrices& b, int batch);

// Eigenvalues (ascending, n per matrix into w, `wStride` apart) and, if `vectors`, orthonormal
// eigenvectors (in place of A, column i for eigenvalue i) of symmetric n x n matrices given by
// their lower or upper triangle (syevd), by cyclic Jacobi. work: syevjWorkspace(n) elements per
// matrix, `workStride` apart.
int64_t syevjWorkspace(int n);
CUresult syevj(CUstream stream, bool f64, bool upper, bool vectors, int n, const Matrices& a, CUdeviceptr w,
               int64_t wStride, CUdeviceptr work, int64_t workStride, int batch);

// Singular value decomposition A = U diag(S) V^T of m x n matrices by one-sided Jacobi (gesvdj):
// the min(m, n) singular values descending into s (`sStride` apart), and if wanted the vectors:
// U m x m (full) or m x min(m, n), V n x n (full) or n x min(m, n), or V^T (vt, m >= n only)
// instead of V. A is overwritten. work: gesvdjWorkspace(m, n) elements per matrix.
int64_t gesvdjWorkspace(int m, int n);
CUresult gesvdj(CUstream stream, bool f64, int m, int n, const Matrices& a, CUdeviceptr s, int64_t sStride,
                bool wantU, const Matrices& u, bool wantV, const Matrices& v, bool vt, bool full, CUdeviceptr work,
                int64_t workStride, int batch);

// C = alpha op(A) op(B) + beta C, C m x n (for cuBLAS's f64 GEMMs; C is not read if beta = 0).
CUresult gemm(CUstream stream, bool f64, bool transA, bool transB, int m, int n, int k, double alpha,
              const Matrices& a, const Matrices& b, double beta, const Matrices& c, int batch);

}  // namespace vulkanizer::linalg
