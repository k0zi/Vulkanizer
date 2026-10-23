// Vulkanizer's libcusolver: the dense factorizations and solves PyTorch's torch.linalg uses, on
// Vulkanizer's own kernels (src/linalg), queued on the handle's stream like any CUDA kernel.
// Only the eigenvalue solvers need a workspace (the other buffer size queries report 0). The rest
// of the API reports CUSOLVER_STATUS_NOT_SUPPORTED (cusolver_stubs.cpp).

#include <cusolver/cusolverDn.h>
#include <cuda.h>

#include "common/log.h"
#include "linalg/linalg.h"

using namespace vulkanizer;

struct cusolverDnContext {
    CUstream stream = nullptr;
};

struct cusolverDnParams {};

namespace {

cusolverStatus_t status(CUresult r) {
    if (r == CUDA_SUCCESS) return CUSOLVER_STATUS_SUCCESS;
    VULKANIZER_WARN("cuSOLVER: kernel launch failed (%d)", int(r));
    return CUSOLVER_STATUS_EXECUTION_FAILED;
}

// f64 needs shaderFloat64.
bool precisionSupported(bool f64) { return !f64 || linalg::hasFloat64(); }

linalg::Matrices single(const void* a, int ld) { return {CUdeviceptr(a), false, 0, ld}; }
linalg::Matrices array(const void* a, int ld) { return {CUdeviceptr(a), true, 0, ld}; }

cusolverStatus_t getrf(cusolverDnHandle_t h, bool f64, int m, int n, void* A, int lda, int* ipiv, int* info) {
    if (!h) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (m < 0 || n < 0 || lda < (m > 1 ? m : 1)) return CUSOLVER_STATUS_INVALID_VALUE;
    if (!precisionSupported(f64)) return CUSOLVER_STATUS_NOT_SUPPORTED;
    return status(linalg::getrf(h->stream, f64, m, n, single(A, lda), CUdeviceptr(ipiv), 0, CUdeviceptr(info), 1));
}

cusolverStatus_t getrs(cusolverDnHandle_t h, bool f64, cublasOperation_t trans, int n, int nrhs, const void* A, int lda,
                       const int* ipiv, void* B, int ldb, int* info) {
    if (!h) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (n < 0 || nrhs < 0 || lda < (n > 1 ? n : 1) || ldb < (n > 1 ? n : 1)) return CUSOLVER_STATUS_INVALID_VALUE;
    if (trans == CUBLAS_OP_C) trans = CUBLAS_OP_T;  // real types: the conjugate transpose is the transpose
    if (!precisionSupported(f64)) return CUSOLVER_STATUS_NOT_SUPPORTED;
    if (CUresult r = cuMemsetD32Async(CUdeviceptr(info), 0, 1, h->stream)) return status(r);
    return status(linalg::getrs(h->stream, f64, trans == CUBLAS_OP_T, n, nrhs, single(A, lda), CUdeviceptr(ipiv), 0,
                                single(B, ldb), 1));
}

cusolverStatus_t potrf(cusolverDnHandle_t h, bool f64, cublasFillMode_t uplo, int n, void* A, int lda, int* info,
                       int batch, bool arrays) {
    if (!h) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (n < 0 || lda < (n > 1 ? n : 1) || batch < 0 || (uplo != CUBLAS_FILL_MODE_LOWER && uplo != CUBLAS_FILL_MODE_UPPER))
        return CUSOLVER_STATUS_INVALID_VALUE;
    if (!precisionSupported(f64)) return CUSOLVER_STATUS_NOT_SUPPORTED;
    return status(linalg::potrf(h->stream, f64, uplo == CUBLAS_FILL_MODE_UPPER, n, arrays ? array(A, lda) : single(A, lda),
                                CUdeviceptr(info), batch));
}

cusolverStatus_t potrs(cusolverDnHandle_t h, bool f64, cublasFillMode_t uplo, int n, int nrhs, const void* A, int lda,
                       void* B, int ldb, int* info, int batch, bool arrays) {
    if (!h) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (n < 0 || nrhs < 0 || lda < (n > 1 ? n : 1) || ldb < (n > 1 ? n : 1) || batch < 0 ||
        (uplo != CUBLAS_FILL_MODE_LOWER && uplo != CUBLAS_FILL_MODE_UPPER))
        return CUSOLVER_STATUS_INVALID_VALUE;
    if (!precisionSupported(f64)) return CUSOLVER_STATUS_NOT_SUPPORTED;
    if (CUresult r = cuMemsetD32Async(CUdeviceptr(info), 0, 1, h->stream)) return status(r);
    return status(linalg::potrs(h->stream, f64, uplo == CUBLAS_FILL_MODE_UPPER, n, nrhs,
                                arrays ? array(A, lda) : single(A, lda), arrays ? array(B, ldb) : single(B, ldb), batch));
}

// The 64-bit API's data types: real f32 or f64 only.
bool realType(cudaDataType t, bool& f64) {
    if (t == CUDA_R_32F) return f64 = false, true;
    if (t == CUDA_R_64F) return f64 = true, true;
    return false;
}

}  // namespace

extern "C" {

cusolverStatus_t cusolverDnCreate(cusolverDnHandle_t* handle) {
    VULKANIZER_TRACE_API();
    if (!handle) return CUSOLVER_STATUS_INVALID_VALUE;
    if (cuInit(0) != CUDA_SUCCESS) return CUSOLVER_STATUS_NOT_INITIALIZED;
    *handle = new cusolverDnContext();
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDestroy(cusolverDnHandle_t handle) {
    VULKANIZER_TRACE_API();
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    delete handle;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSetStream(cusolverDnHandle_t handle, cudaStream_t stream) {
    VULKANIZER_TRACE_API();
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    handle->stream = reinterpret_cast<CUstream>(stream);
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnGetStream(cusolverDnHandle_t handle, cudaStream_t* stream) {
    VULKANIZER_TRACE_API();
    if (!handle || !stream) return CUSOLVER_STATUS_NOT_INITIALIZED;
    *stream = reinterpret_cast<cudaStream_t>(handle->stream);
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnGetVersion(int* version) {
    if (!version) return CUSOLVER_STATUS_INVALID_VALUE;
    *version = VULKANIZER_CUSOLVER_VERSION;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCreateParams(cusolverDnParams_t* params) {
    if (!params) return CUSOLVER_STATUS_INVALID_VALUE;
    *params = new cusolverDnParams();
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDestroyParams(cusolverDnParams_t params) {
    delete params;
    return CUSOLVER_STATUS_SUCCESS;
}

// ---- LU ---------------------------------------------------------------------------------------

cusolverStatus_t cusolverDnSgetrf_bufferSize(cusolverDnHandle_t handle, int, int, float*, int, int* lwork) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (!lwork) return CUSOLVER_STATUS_INVALID_VALUE;
    *lwork = 0;
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnDgetrf_bufferSize(cusolverDnHandle_t handle, int m, int n, double*, int lda, int* lwork) {
    return cusolverDnSgetrf_bufferSize(handle, m, n, nullptr, lda, lwork);
}
cusolverStatus_t cusolverDnSgetrf(cusolverDnHandle_t handle, int m, int n, float* A, int lda, float*, int* devIpiv,
                                  int* devInfo) {
    VULKANIZER_TRACE_API();
    return getrf(handle, false, m, n, A, lda, devIpiv, devInfo);
}
cusolverStatus_t cusolverDnDgetrf(cusolverDnHandle_t handle, int m, int n, double* A, int lda, double*, int* devIpiv,
                                  int* devInfo) {
    VULKANIZER_TRACE_API();
    return getrf(handle, true, m, n, A, lda, devIpiv, devInfo);
}
cusolverStatus_t cusolverDnSgetrs(cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const float* A,
                                  int lda, const int* devIpiv, float* B, int ldb, int* devInfo) {
    VULKANIZER_TRACE_API();
    return getrs(handle, false, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo);
}
cusolverStatus_t cusolverDnDgetrs(cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const double* A,
                                  int lda, const int* devIpiv, double* B, int ldb, int* devInfo) {
    VULKANIZER_TRACE_API();
    return getrs(handle, true, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo);
}

// ---- Cholesky ---------------------------------------------------------------------------------

cusolverStatus_t cusolverDnSpotrf_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t, int, float*, int, int* lwork) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (!lwork) return CUSOLVER_STATUS_INVALID_VALUE;
    *lwork = 0;
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnDpotrf_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double*, int lda,
                                             int* lwork) {
    return cusolverDnSpotrf_bufferSize(handle, uplo, n, nullptr, lda, lwork);
}
cusolverStatus_t cusolverDnSpotrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float* A, int lda, float*,
                                  int, int* devInfo) {
    VULKANIZER_TRACE_API();
    return potrf(handle, false, uplo, n, A, lda, devInfo, 1, false);
}
cusolverStatus_t cusolverDnDpotrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double* A, int lda, double*,
                                  int, int* devInfo) {
    VULKANIZER_TRACE_API();
    return potrf(handle, true, uplo, n, A, lda, devInfo, 1, false);
}
cusolverStatus_t cusolverDnSpotrs(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, const float* A,
                                  int lda, float* B, int ldb, int* devInfo) {
    VULKANIZER_TRACE_API();
    return potrs(handle, false, uplo, n, nrhs, A, lda, B, ldb, devInfo, 1, false);
}
cusolverStatus_t cusolverDnDpotrs(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, const double* A,
                                  int lda, double* B, int ldb, int* devInfo) {
    VULKANIZER_TRACE_API();
    return potrs(handle, true, uplo, n, nrhs, A, lda, B, ldb, devInfo, 1, false);
}
cusolverStatus_t cusolverDnSpotrfBatched(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float* Aarray[],
                                         int lda, int* infoArray, int batchSize) {
    VULKANIZER_TRACE_API();
    return potrf(handle, false, uplo, n, Aarray, lda, infoArray, batchSize, true);
}
cusolverStatus_t cusolverDnDpotrfBatched(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double* Aarray[],
                                         int lda, int* infoArray, int batchSize) {
    VULKANIZER_TRACE_API();
    return potrf(handle, true, uplo, n, Aarray, lda, infoArray, batchSize, true);
}
cusolverStatus_t cusolverDnSpotrsBatched(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs,
                                         float* Aarray[], int lda, float* Barray[], int ldb, int* info, int batchSize) {
    VULKANIZER_TRACE_API();
    return potrs(handle, false, uplo, n, nrhs, Aarray, lda, Barray, ldb, info, batchSize, true);
}
cusolverStatus_t cusolverDnDpotrsBatched(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs,
                                         double* Aarray[], int lda, double* Barray[], int ldb, int* info,
                                         int batchSize) {
    VULKANIZER_TRACE_API();
    return potrs(handle, true, uplo, n, nrhs, Aarray, lda, Barray, ldb, info, batchSize, true);
}

// ---- 64-bit API -------------------------------------------------------------------------------

cusolverStatus_t cusolverDnXpotrf_bufferSize(cusolverDnHandle_t handle, cusolverDnParams_t, cublasFillMode_t, int64_t,
                                             cudaDataType dataTypeA, const void*, int64_t, cudaDataType,
                                             size_t* workspaceInBytesOnDevice, size_t* workspaceInBytesOnHost) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    bool f64;
    if (!realType(dataTypeA, f64)) return CUSOLVER_STATUS_NOT_SUPPORTED;
    if (workspaceInBytesOnDevice) *workspaceInBytesOnDevice = 0;
    if (workspaceInBytesOnHost) *workspaceInBytesOnHost = 0;
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnXpotrf(cusolverDnHandle_t handle, cusolverDnParams_t, cublasFillMode_t uplo, int64_t n,
                                  cudaDataType dataTypeA, void* A, int64_t lda, cudaDataType, void*, size_t, void*,
                                  size_t, int* info) {
    VULKANIZER_TRACE_API();
    bool f64;
    if (!realType(dataTypeA, f64)) return CUSOLVER_STATUS_NOT_SUPPORTED;
    if (n > INT32_MAX || lda > INT32_MAX) return CUSOLVER_STATUS_NOT_SUPPORTED;
    return potrf(handle, f64, uplo, int(n), A, int(lda), info, 1, false);
}
cusolverStatus_t cusolverDnXpotrs(cusolverDnHandle_t handle, cusolverDnParams_t, cublasFillMode_t uplo, int64_t n,
                                  int64_t nrhs, cudaDataType dataTypeA, const void* A, int64_t lda,
                                  cudaDataType dataTypeB, void* B, int64_t ldb, int* info) {
    VULKANIZER_TRACE_API();
    bool f64, f64b;
    if (!realType(dataTypeA, f64) || !realType(dataTypeB, f64b) || f64 != f64b) return CUSOLVER_STATUS_NOT_SUPPORTED;
    if (n > INT32_MAX || nrhs > INT32_MAX || lda > INT32_MAX || ldb > INT32_MAX) return CUSOLVER_STATUS_NOT_SUPPORTED;
    return potrs(handle, f64, uplo, int(n), int(nrhs), A, int(lda), B, int(ldb), info, 1, false);
}

}  // extern "C"

// ---- QR ---------------------------------------------------------------------------------------

namespace {

cusolverStatus_t geqrf(cusolverDnHandle_t h, bool f64, int m, int n, void* A, int lda, void* tau, int* info) {
    if (!h) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (m < 0 || n < 0 || lda < (m > 1 ? m : 1)) return CUSOLVER_STATUS_INVALID_VALUE;
    if (!precisionSupported(f64)) return CUSOLVER_STATUS_NOT_SUPPORTED;
    if (info) {
        if (CUresult r = cuMemsetD32Async(CUdeviceptr(info), 0, 1, h->stream)) return status(r);
    }
    return status(linalg::geqrf(h->stream, f64, m, n, single(A, lda), CUdeviceptr(tau), 0, 1));
}

cusolverStatus_t orgqr(cusolverDnHandle_t h, bool f64, int m, int n, int k, void* A, int lda, const void* tau, int* info) {
    if (!h) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (m < 0 || n < 0 || n > m || k < 0 || k > n || lda < (m > 1 ? m : 1)) return CUSOLVER_STATUS_INVALID_VALUE;
    if (!precisionSupported(f64)) return CUSOLVER_STATUS_NOT_SUPPORTED;
    if (info) {
        if (CUresult r = cuMemsetD32Async(CUdeviceptr(info), 0, 1, h->stream)) return status(r);
    }
    return status(linalg::orgqr(h->stream, f64, m, n, k, single(A, lda), CUdeviceptr(tau), 0, 1));
}

cusolverStatus_t ormqr(cusolverDnHandle_t h, bool f64, cublasSideMode_t side, cublasOperation_t trans, int m, int n,
                       int k, const void* A, int lda, const void* tau, void* C, int ldc, int* info) {
    if (!h) return CUSOLVER_STATUS_NOT_INITIALIZED;
    const bool right = side == CUBLAS_SIDE_RIGHT;
    const int order = right ? n : m;  // Q is order x order
    if (m < 0 || n < 0 || k < 0 || k > order || lda < (order > 1 ? order : 1) || ldc < (m > 1 ? m : 1))
        return CUSOLVER_STATUS_INVALID_VALUE;
    if (!precisionSupported(f64)) return CUSOLVER_STATUS_NOT_SUPPORTED;
    if (info) {
        if (CUresult r = cuMemsetD32Async(CUdeviceptr(info), 0, 1, h->stream)) return status(r);
    }
    return status(linalg::ormqr(h->stream, f64, right, trans != CUBLAS_OP_N, m, n, k, single(A, lda), CUdeviceptr(tau),
                                0, single(C, ldc), 1));
}

cusolverStatus_t noWorkspace(cusolverDnHandle_t h, int* lwork) {
    if (!h) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (!lwork) return CUSOLVER_STATUS_INVALID_VALUE;
    *lwork = 0;
    return CUSOLVER_STATUS_SUCCESS;
}

}  // namespace

extern "C" {

cusolverStatus_t cusolverDnSgeqrf_bufferSize(cusolverDnHandle_t handle, int, int, float*, int, int* lwork) {
    return noWorkspace(handle, lwork);
}
cusolverStatus_t cusolverDnDgeqrf_bufferSize(cusolverDnHandle_t handle, int, int, double*, int, int* lwork) {
    return noWorkspace(handle, lwork);
}
cusolverStatus_t cusolverDnSgeqrf(cusolverDnHandle_t handle, int m, int n, float* A, int lda, float* tau, float*, int,
                                  int* devInfo) {
    VULKANIZER_TRACE_API();
    return geqrf(handle, false, m, n, A, lda, tau, devInfo);
}
cusolverStatus_t cusolverDnDgeqrf(cusolverDnHandle_t handle, int m, int n, double* A, int lda, double* tau, double*,
                                  int, int* devInfo) {
    VULKANIZER_TRACE_API();
    return geqrf(handle, true, m, n, A, lda, tau, devInfo);
}
cusolverStatus_t cusolverDnSorgqr_bufferSize(cusolverDnHandle_t handle, int, int, int, const float*, int, const float*,
                                             int* lwork) {
    return noWorkspace(handle, lwork);
}
cusolverStatus_t cusolverDnDorgqr_bufferSize(cusolverDnHandle_t handle, int, int, int, const double*, int, const double*,
                                             int* lwork) {
    return noWorkspace(handle, lwork);
}
cusolverStatus_t cusolverDnSorgqr(cusolverDnHandle_t handle, int m, int n, int k, float* A, int lda, const float* tau,
                                  float*, int, int* devInfo) {
    VULKANIZER_TRACE_API();
    return orgqr(handle, false, m, n, k, A, lda, tau, devInfo);
}
cusolverStatus_t cusolverDnDorgqr(cusolverDnHandle_t handle, int m, int n, int k, double* A, int lda, const double* tau,
                                  double*, int, int* devInfo) {
    VULKANIZER_TRACE_API();
    return orgqr(handle, true, m, n, k, A, lda, tau, devInfo);
}
cusolverStatus_t cusolverDnSormqr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t, cublasOperation_t, int, int,
                                             int, const float*, int, const float*, const float*, int, int* lwork) {
    return noWorkspace(handle, lwork);
}
cusolverStatus_t cusolverDnDormqr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t, cublasOperation_t, int, int,
                                             int, const double*, int, const double*, const double*, int, int* lwork) {
    return noWorkspace(handle, lwork);
}
cusolverStatus_t cusolverDnSormqr(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m,
                                  int n, int k, const float* A, int lda, const float* tau, float* C, int ldc, float*,
                                  int, int* devInfo) {
    VULKANIZER_TRACE_API();
    return ormqr(handle, false, side, trans, m, n, k, A, lda, tau, C, ldc, devInfo);
}
cusolverStatus_t cusolverDnDormqr(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m,
                                  int n, int k, const double* A, int lda, const double* tau, double* C, int ldc,
                                  double*, int, int* devInfo) {
    VULKANIZER_TRACE_API();
    return ormqr(handle, true, side, trans, m, n, k, A, lda, tau, C, ldc, devInfo);
}

cusolverStatus_t cusolverDnXgeqrf_bufferSize(cusolverDnHandle_t handle, cusolverDnParams_t, int64_t, int64_t,
                                             cudaDataType dataTypeA, const void*, int64_t, cudaDataType, const void*,
                                             cudaDataType, size_t* workspaceInBytesOnDevice,
                                             size_t* workspaceInBytesOnHost) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    bool f64;
    if (!realType(dataTypeA, f64)) return CUSOLVER_STATUS_NOT_SUPPORTED;
    if (workspaceInBytesOnDevice) *workspaceInBytesOnDevice = 0;
    if (workspaceInBytesOnHost) *workspaceInBytesOnHost = 0;
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnXgeqrf(cusolverDnHandle_t handle, cusolverDnParams_t, int64_t m, int64_t n,
                                  cudaDataType dataTypeA, void* A, int64_t lda, cudaDataType dataTypeTau, void* tau,
                                  cudaDataType, void*, size_t, void*, size_t, int* info) {
    VULKANIZER_TRACE_API();
    bool f64, f64tau;
    if (!realType(dataTypeA, f64) || !realType(dataTypeTau, f64tau) || f64 != f64tau) return CUSOLVER_STATUS_NOT_SUPPORTED;
    if (m > INT32_MAX || n > INT32_MAX || lda > INT32_MAX) return CUSOLVER_STATUS_NOT_SUPPORTED;
    return geqrf(handle, f64, int(m), int(n), A, int(lda), tau, info);
}

}  // extern "C"

// ---- Symmetric eigenvalue problems ------------------------------------------------------------

// The Jacobi parameters are accepted but not needed: the iteration runs a fixed number of sweeps
// and always sorts the eigenvalues.
struct syevjInfo {
    double tolerance = 0;
    int maxSweeps = 100;
    int sortEig = 1;
};

namespace {

// Every variant (syevd, syevj, their batched and 64-bit forms): workspace sizes in elements.
int64_t syevWorkspace(int64_t n, int64_t batch) { return linalg::syevjWorkspace(int(n)) * batch; }

cusolverStatus_t syevWorkspaceQuery(cusolverDnHandle_t h, int n, int lda, int batch, int* lwork) {
    if (!h) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (!lwork || n < 0 || lda < (n > 1 ? n : 1) || batch < 0) return CUSOLVER_STATUS_INVALID_VALUE;
    const int64_t size = syevWorkspace(n, batch);
    if (size > INT32_MAX) return CUSOLVER_STATUS_NOT_SUPPORTED;
    *lwork = int(size);
    return CUSOLVER_STATUS_SUCCESS;
}

// A: batch matrices n x n, lda * n apart; W: n eigenvalues per matrix; info: one int per matrix.
cusolverStatus_t syev(cusolverDnHandle_t h, bool f64, cusolverEigMode_t jobz, cublasFillMode_t uplo, int64_t n,
                      void* A, int64_t lda, void* W, void* work, int64_t lworkElements, int* info, int64_t batch) {
    if (!h) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (n < 0 || lda < (n > 1 ? n : 1) || batch < 0 || (uplo != CUBLAS_FILL_MODE_LOWER && uplo != CUBLAS_FILL_MODE_UPPER) ||
        (jobz != CUSOLVER_EIG_MODE_NOVECTOR && jobz != CUSOLVER_EIG_MODE_VECTOR))
        return CUSOLVER_STATUS_INVALID_VALUE;
    if (n > INT32_MAX || lda > INT32_MAX || batch > INT32_MAX) return CUSOLVER_STATUS_NOT_SUPPORTED;
    if (lworkElements < syevWorkspace(n, batch) || (!work && n > 0 && batch > 0)) return CUSOLVER_STATUS_INVALID_VALUE;
    if (!precisionSupported(f64)) return CUSOLVER_STATUS_NOT_SUPPORTED;
    if (info) {
        if (CUresult r = cuMemsetD32Async(CUdeviceptr(info), 0, size_t(batch), h->stream)) return status(r);
    }
    linalg::Matrices a{CUdeviceptr(A), false, lda * n, int(lda)};
    return status(linalg::syevj(h->stream, f64, uplo == CUBLAS_FILL_MODE_UPPER, jobz == CUSOLVER_EIG_MODE_VECTOR, int(n),
                                a, CUdeviceptr(W), n, CUdeviceptr(work), linalg::syevjWorkspace(int(n)), int(batch)));
}

// The 64-bit API: A and W of one real type, workspace sizes in bytes.
cusolverStatus_t xsyevTypes(cudaDataType dataTypeA, cudaDataType dataTypeW, bool& f64) {
    bool f64w;
    if (!realType(dataTypeA, f64) || !realType(dataTypeW, f64w) || f64 != f64w) return CUSOLVER_STATUS_NOT_SUPPORTED;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t xsyevWorkspaceQuery(cusolverDnHandle_t h, cudaDataType dataTypeA, cudaDataType dataTypeW, int64_t n,
                                     int64_t batch, size_t* device, size_t* host) {
    if (!h) return CUSOLVER_STATUS_NOT_INITIALIZED;
    bool f64;
    if (cusolverStatus_t s = xsyevTypes(dataTypeA, dataTypeW, f64)) return s;
    if (n < 0 || batch < 0) return CUSOLVER_STATUS_INVALID_VALUE;
    if (n > INT32_MAX) return CUSOLVER_STATUS_NOT_SUPPORTED;
    if (device) *device = size_t(syevWorkspace(n, batch)) * (f64 ? 8 : 4);
    if (host) *host = 0;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t xsyev(cusolverDnHandle_t h, cusolverEigMode_t jobz, cublasFillMode_t uplo, int64_t n,
                       cudaDataType dataTypeA, void* A, int64_t lda, cudaDataType dataTypeW, void* W, void* work,
                       size_t workBytes, int* info, int64_t batch) {
    bool f64;
    if (cusolverStatus_t s = xsyevTypes(dataTypeA, dataTypeW, f64)) return s;
    return syev(h, f64, jobz, uplo, n, A, lda, W, work, int64_t(workBytes / (f64 ? 8 : 4)), info, batch);
}

}  // namespace

extern "C" {

cusolverStatus_t cusolverDnCreateSyevjInfo(syevjInfo_t* info) {
    if (!info) return CUSOLVER_STATUS_INVALID_VALUE;
    *info = new syevjInfo();
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnDestroySyevjInfo(syevjInfo_t info) {
    delete info;
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnXsyevjSetTolerance(syevjInfo_t info, double tolerance) {
    if (!info) return CUSOLVER_STATUS_INVALID_VALUE;
    info->tolerance = tolerance;
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnXsyevjSetMaxSweeps(syevjInfo_t info, int maxSweeps) {
    if (!info) return CUSOLVER_STATUS_INVALID_VALUE;
    info->maxSweeps = maxSweeps;
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnXsyevjSetSortEig(syevjInfo_t info, int sortEig) {
    if (!info) return CUSOLVER_STATUS_INVALID_VALUE;
    info->sortEig = sortEig;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSsyevd_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t, cublasFillMode_t, int n,
                                             const float*, int lda, const float*, int* lwork) {
    return syevWorkspaceQuery(handle, n, lda, 1, lwork);
}
cusolverStatus_t cusolverDnDsyevd_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t, cublasFillMode_t, int n,
                                             const double*, int lda, const double*, int* lwork) {
    return syevWorkspaceQuery(handle, n, lda, 1, lwork);
}
cusolverStatus_t cusolverDnSsyevd(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n,
                                  float* A, int lda, float* W, float* work, int lwork, int* devInfo) {
    VULKANIZER_TRACE_API();
    return syev(handle, false, jobz, uplo, n, A, lda, W, work, lwork, devInfo, 1);
}
cusolverStatus_t cusolverDnDsyevd(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n,
                                  double* A, int lda, double* W, double* work, int lwork, int* devInfo) {
    VULKANIZER_TRACE_API();
    return syev(handle, true, jobz, uplo, n, A, lda, W, work, lwork, devInfo, 1);
}
cusolverStatus_t cusolverDnSsyevj_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t, cublasFillMode_t, int n,
                                             const float*, int lda, const float*, int* lwork, syevjInfo_t) {
    return syevWorkspaceQuery(handle, n, lda, 1, lwork);
}
cusolverStatus_t cusolverDnDsyevj_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t, cublasFillMode_t, int n,
                                             const double*, int lda, const double*, int* lwork, syevjInfo_t) {
    return syevWorkspaceQuery(handle, n, lda, 1, lwork);
}
cusolverStatus_t cusolverDnSsyevj(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n,
                                  float* A, int lda, float* W, float* work, int lwork, int* info, syevjInfo_t) {
    VULKANIZER_TRACE_API();
    return syev(handle, false, jobz, uplo, n, A, lda, W, work, lwork, info, 1);
}
cusolverStatus_t cusolverDnDsyevj(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n,
                                  double* A, int lda, double* W, double* work, int lwork, int* info, syevjInfo_t) {
    VULKANIZER_TRACE_API();
    return syev(handle, true, jobz, uplo, n, A, lda, W, work, lwork, info, 1);
}
cusolverStatus_t cusolverDnSsyevjBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t, cublasFillMode_t,
                                                    int n, const float*, int lda, const float*, int* lwork,
                                                    syevjInfo_t, int batchSize) {
    return syevWorkspaceQuery(handle, n, lda, batchSize, lwork);
}
cusolverStatus_t cusolverDnDsyevjBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t, cublasFillMode_t,
                                                    int n, const double*, int lda, const double*, int* lwork,
                                                    syevjInfo_t, int batchSize) {
    return syevWorkspaceQuery(handle, n, lda, batchSize, lwork);
}
cusolverStatus_t cusolverDnSsyevjBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo,
                                         int n, float* A, int lda, float* W, float* work, int lwork, int* info,
                                         syevjInfo_t, int batchSize) {
    VULKANIZER_TRACE_API();
    return syev(handle, false, jobz, uplo, n, A, lda, W, work, lwork, info, batchSize);
}
cusolverStatus_t cusolverDnDsyevjBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo,
                                         int n, double* A, int lda, double* W, double* work, int lwork, int* info,
                                         syevjInfo_t, int batchSize) {
    VULKANIZER_TRACE_API();
    return syev(handle, true, jobz, uplo, n, A, lda, W, work, lwork, info, batchSize);
}
cusolverStatus_t cusolverDnXsyevd_bufferSize(cusolverDnHandle_t handle, cusolverDnParams_t, cusolverEigMode_t,
                                             cublasFillMode_t, int64_t n, cudaDataType dataTypeA, const void*, int64_t,
                                             cudaDataType dataTypeW, const void*, cudaDataType,
                                             size_t* workspaceInBytesOnDevice, size_t* workspaceInBytesOnHost) {
    return xsyevWorkspaceQuery(handle, dataTypeA, dataTypeW, n, 1, workspaceInBytesOnDevice, workspaceInBytesOnHost);
}
cusolverStatus_t cusolverDnXsyevd(cusolverDnHandle_t handle, cusolverDnParams_t, cusolverEigMode_t jobz,
                                  cublasFillMode_t uplo, int64_t n, cudaDataType dataTypeA, void* A, int64_t lda,
                                  cudaDataType dataTypeW, void* W, cudaDataType, void* bufferOnDevice,
                                  size_t workspaceInBytesOnDevice, void*, size_t, int* info) {
    VULKANIZER_TRACE_API();
    return xsyev(handle, jobz, uplo, n, dataTypeA, A, lda, dataTypeW, W, bufferOnDevice, workspaceInBytesOnDevice,
                 info, 1);
}
cusolverStatus_t cusolverDnXsyevBatched_bufferSize(cusolverDnHandle_t handle, cusolverDnParams_t, cusolverEigMode_t,
                                                   cublasFillMode_t, int64_t n, cudaDataType dataTypeA, const void*,
                                                   int64_t, cudaDataType dataTypeW, const void*, cudaDataType,
                                                   size_t* workspaceInBytesOnDevice, size_t* workspaceInBytesOnHost,
                                                   int64_t batchSize) {
    return xsyevWorkspaceQuery(handle, dataTypeA, dataTypeW, n, batchSize, workspaceInBytesOnDevice,
                               workspaceInBytesOnHost);
}
cusolverStatus_t cusolverDnXsyevBatched(cusolverDnHandle_t handle, cusolverDnParams_t, cusolverEigMode_t jobz,
                                        cublasFillMode_t uplo, int64_t n, cudaDataType dataTypeA, void* A,
                                        int64_t lda, cudaDataType dataTypeW, void* W, cudaDataType,
                                        void* bufferOnDevice, size_t workspaceInBytesOnDevice, void*, size_t,
                                        int* info, int64_t batchSize) {
    VULKANIZER_TRACE_API();
    return xsyev(handle, jobz, uplo, n, dataTypeA, A, lda, dataTypeW, W, bufferOnDevice, workspaceInBytesOnDevice, info,
                 batchSize);
}

}  // extern "C"

// ---- Singular value decomposition -------------------------------------------------------------

// As syevjInfo: accepted, not needed.
struct gesvdjInfo {
    double tolerance = 0;
    int maxSweeps = 100;
    int sortEig = 1;
};

namespace {

cusolverStatus_t svdWorkspaceQuery(cusolverDnHandle_t h, int m, int n, int batch, int* lwork) {
    if (!h) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (!lwork || m < 0 || n < 0 || batch < 0) return CUSOLVER_STATUS_INVALID_VALUE;
    const int64_t size = linalg::gesvdjWorkspace(m, n) * batch;
    if (size > INT32_MAX) return CUSOLVER_STATUS_NOT_SUPPORTED;
    *lwork = int(size);
    return CUSOLVER_STATUS_SUCCESS;
}

// A: batch matrices m x n, lda * n apart; S: min(m, n) per matrix; U: m x m (full) or
// m x min(m, n), ldu * (its columns) apart; V (or V^T, vt) n x n (full) or n x min(m, n).
cusolverStatus_t svd(cusolverDnHandle_t h, bool f64, bool wantU, bool wantV, bool full, bool vt, int m, int n, void* A,
                     int lda, void* S, void* U, int ldu, void* V, int ldv, void* work, int64_t lwork, int* info,
                     int batch) {
    if (!h) return CUSOLVER_STATUS_NOT_INITIALIZED;
    const int k = m < n ? m : n;
    const int uColumns = full ? m : k, vColumns = full || vt ? n : k;  // V^T: n x n (gesvd, m >= n)
    if (m < 0 || n < 0 || batch < 0 || lda < (m > 1 ? m : 1) || (wantU && (!U || ldu < (m > 1 ? m : 1))) ||
        (wantV && (!V || ldv < (n > 1 ? n : 1))))
        return CUSOLVER_STATUS_INVALID_VALUE;
    if (vt && wantV && m < n) return CUSOLVER_STATUS_INVALID_VALUE;  // gesvd: m >= n only
    if (lwork < linalg::gesvdjWorkspace(m, n) * batch || (!work && k > 0 && batch > 0))
        return CUSOLVER_STATUS_INVALID_VALUE;
    if (!precisionSupported(f64)) return CUSOLVER_STATUS_NOT_SUPPORTED;
    if (info) {
        if (CUresult r = cuMemsetD32Async(CUdeviceptr(info), 0, size_t(batch), h->stream)) return status(r);
    }
    const linalg::Matrices a{CUdeviceptr(A), false, int64_t(lda) * n, lda};
    const linalg::Matrices u{CUdeviceptr(U), false, int64_t(ldu) * uColumns, ldu};
    const linalg::Matrices v{CUdeviceptr(V), false, int64_t(ldv) * vColumns, ldv};
    return status(linalg::gesvdj(h->stream, f64, m, n, a, CUdeviceptr(S), k, wantU, u, wantV, v, vt, full,
                                 CUdeviceptr(work), linalg::gesvdjWorkspace(m, n), batch));
}

}  // namespace

extern "C" {

cusolverStatus_t cusolverDnCreateGesvdjInfo(gesvdjInfo_t* info) {
    if (!info) return CUSOLVER_STATUS_INVALID_VALUE;
    *info = new gesvdjInfo();
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnDestroyGesvdjInfo(gesvdjInfo_t info) {
    delete info;
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnXgesvdjSetTolerance(gesvdjInfo_t info, double tolerance) {
    if (!info) return CUSOLVER_STATUS_INVALID_VALUE;
    info->tolerance = tolerance;
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnXgesvdjSetMaxSweeps(gesvdjInfo_t info, int maxSweeps) {
    if (!info) return CUSOLVER_STATUS_INVALID_VALUE;
    info->maxSweeps = maxSweeps;
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnXgesvdjSetSortEig(gesvdjInfo_t info, int sortSvd) {
    if (!info) return CUSOLVER_STATUS_INVALID_VALUE;
    info->sortEig = sortSvd;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSgesvdj_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t, int, int m, int n,
                                              const float*, int, const float*, const float*, int, const float*, int,
                                              int* lwork, gesvdjInfo_t) {
    return svdWorkspaceQuery(handle, m, n, 1, lwork);
}
cusolverStatus_t cusolverDnDgesvdj_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t, int, int m, int n,
                                              const double*, int, const double*, const double*, int, const double*,
                                              int, int* lwork, gesvdjInfo_t) {
    return svdWorkspaceQuery(handle, m, n, 1, lwork);
}
cusolverStatus_t cusolverDnSgesvdj(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, float* A,
                                   int lda, float* S, float* U, int ldu, float* V, int ldv, float* work, int lwork,
                                   int* info, gesvdjInfo_t) {
    VULKANIZER_TRACE_API();
    const bool vectors = jobz == CUSOLVER_EIG_MODE_VECTOR;
    return svd(handle, false, vectors, vectors, !econ, false, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, 1);
}
cusolverStatus_t cusolverDnDgesvdj(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n,
                                   double* A, int lda, double* S, double* U, int ldu, double* V, int ldv, double* work,
                                   int lwork, int* info, gesvdjInfo_t) {
    VULKANIZER_TRACE_API();
    const bool vectors = jobz == CUSOLVER_EIG_MODE_VECTOR;
    return svd(handle, true, vectors, vectors, !econ, false, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, 1);
}
cusolverStatus_t cusolverDnSgesvdjBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t, int m, int n,
                                                     const float*, int, const float*, const float*, int, const float*,
                                                     int, int* lwork, gesvdjInfo_t, int batchSize) {
    return svdWorkspaceQuery(handle, m, n, batchSize, lwork);
}
cusolverStatus_t cusolverDnDgesvdjBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t, int m, int n,
                                                     const double*, int, const double*, const double*, int,
                                                     const double*, int, int* lwork, gesvdjInfo_t, int batchSize) {
    return svdWorkspaceQuery(handle, m, n, batchSize, lwork);
}
cusolverStatus_t cusolverDnSgesvdjBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, float* A,
                                          int lda, float* S, float* U, int ldu, float* V, int ldv, float* work,
                                          int lwork, int* info, gesvdjInfo_t, int batchSize) {
    VULKANIZER_TRACE_API();
    const bool vectors = jobz == CUSOLVER_EIG_MODE_VECTOR;
    return svd(handle, false, vectors, vectors, true, false, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info,
               batchSize);
}
cusolverStatus_t cusolverDnDgesvdjBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, double* A,
                                          int lda, double* S, double* U, int ldu, double* V, int ldv, double* work,
                                          int lwork, int* info, gesvdjInfo_t, int batchSize) {
    VULKANIZER_TRACE_API();
    const bool vectors = jobz == CUSOLVER_EIG_MODE_VECTOR;
    return svd(handle, true, vectors, vectors, true, false, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info,
               batchSize);
}

cusolverStatus_t cusolverDnSgesvd_bufferSize(cusolverDnHandle_t handle, int m, int n, int* lwork) {
    return svdWorkspaceQuery(handle, m, n, 1, lwork);
}
cusolverStatus_t cusolverDnDgesvd_bufferSize(cusolverDnHandle_t handle, int m, int n, int* lwork) {
    return svdWorkspaceQuery(handle, m, n, 1, lwork);
}

}  // extern "C"

namespace {

// gesvd's jobu and jobvt: 'A' (all columns), 'S' (the first min(m, n)) or 'N' (none); 'O'
// (overwriting A) is not supported.
cusolverStatus_t gesvd(cusolverDnHandle_t h, bool f64, signed char jobu, signed char jobvt, int m, int n, void* A,
                       int lda, void* S, void* U, int ldu, void* VT, int ldvt, void* work, int lwork, int* info) {
    const auto valid = [](signed char job) { return job == 'A' || job == 'S' || job == 'N'; };
    if (jobu == 'O' || jobvt == 'O') return CUSOLVER_STATUS_NOT_SUPPORTED;
    if (!valid(jobu) || !valid(jobvt) || m < n) return CUSOLVER_STATUS_INVALID_VALUE;
    // m >= n: V^T is n x n either way
    return svd(h, f64, jobu != 'N', jobvt != 'N', jobu == 'A', true, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork,
               info, 1);
}

}  // namespace

extern "C" {

cusolverStatus_t cusolverDnSgesvd(cusolverDnHandle_t handle, signed char jobu, signed char jobvt, int m, int n,
                                  float* A, int lda, float* S, float* U, int ldu, float* VT, int ldvt, float* work,
                                  int lwork, float*, int* info) {
    VULKANIZER_TRACE_API();
    return gesvd(handle, false, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, info);
}
cusolverStatus_t cusolverDnDgesvd(cusolverDnHandle_t handle, signed char jobu, signed char jobvt, int m, int n,
                                  double* A, int lda, double* S, double* U, int ldu, double* VT, int ldvt,
                                  double* work, int lwork, double*, int* info) {
    VULKANIZER_TRACE_API();
    return gesvd(handle, true, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, info);
}

}  // extern "C"
