#pragma once

// Clean-room declarations of the cuSOLVER dense API subset implemented by Vulkanizer's
// libcusolver.so, written from the public cuSOLVER documentation. Enum values follow the
// documented ABI; the fill mode, operation and data types are cuBLAS's (cublas_v2.h).

#include <cublas_v2.h>

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CUSOLVER_STATUS_SUCCESS = 0,
    CUSOLVER_STATUS_NOT_INITIALIZED = 1,
    CUSOLVER_STATUS_ALLOC_FAILED = 2,
    CUSOLVER_STATUS_INVALID_VALUE = 3,
    CUSOLVER_STATUS_ARCH_MISMATCH = 4,
    CUSOLVER_STATUS_MAPPING_ERROR = 5,
    CUSOLVER_STATUS_EXECUTION_FAILED = 6,
    CUSOLVER_STATUS_INTERNAL_ERROR = 7,
    CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 8,
    CUSOLVER_STATUS_NOT_SUPPORTED = 9,
} cusolverStatus_t;

typedef enum { CUSOLVER_EIG_MODE_NOVECTOR = 0, CUSOLVER_EIG_MODE_VECTOR = 1 } cusolverEigMode_t;

typedef struct cusolverDnContext* cusolverDnHandle_t;
typedef struct cusolverDnParams* cusolverDnParams_t;
typedef struct syevjInfo* syevjInfo_t;
typedef struct gesvdjInfo* gesvdjInfo_t;

cusolverStatus_t cusolverDnCreate(cusolverDnHandle_t* handle);
cusolverStatus_t cusolverDnDestroy(cusolverDnHandle_t handle);
cusolverStatus_t cusolverDnSetStream(cusolverDnHandle_t handle, cudaStream_t stream);
cusolverStatus_t cusolverDnGetStream(cusolverDnHandle_t handle, cudaStream_t* stream);
cusolverStatus_t cusolverDnGetVersion(int* version);
cusolverStatus_t cusolverDnCreateParams(cusolverDnParams_t* params);
cusolverStatus_t cusolverDnDestroyParams(cusolverDnParams_t params);

// LU with partial pivoting and its solve
cusolverStatus_t cusolverDnSgetrf_bufferSize(cusolverDnHandle_t handle, int m, int n, float* A, int lda, int* lwork);
cusolverStatus_t cusolverDnDgetrf_bufferSize(cusolverDnHandle_t handle, int m, int n, double* A, int lda, int* lwork);
cusolverStatus_t cusolverDnSgetrf(cusolverDnHandle_t handle, int m, int n, float* A, int lda, float* workspace,
                                  int* devIpiv, int* devInfo);
cusolverStatus_t cusolverDnDgetrf(cusolverDnHandle_t handle, int m, int n, double* A, int lda, double* workspace,
                                  int* devIpiv, int* devInfo);
cusolverStatus_t cusolverDnSgetrs(cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const float* A,
                                  int lda, const int* devIpiv, float* B, int ldb, int* devInfo);
cusolverStatus_t cusolverDnDgetrs(cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const double* A,
                                  int lda, const int* devIpiv, double* B, int ldb, int* devInfo);

// Cholesky and its solve
cusolverStatus_t cusolverDnSpotrf_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float* A, int lda,
                                             int* lwork);
cusolverStatus_t cusolverDnDpotrf_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double* A, int lda,
                                             int* lwork);
cusolverStatus_t cusolverDnSpotrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float* A, int lda,
                                  float* workspace, int lwork, int* devInfo);
cusolverStatus_t cusolverDnDpotrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double* A, int lda,
                                  double* workspace, int lwork, int* devInfo);
cusolverStatus_t cusolverDnSpotrs(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, const float* A,
                                  int lda, float* B, int ldb, int* devInfo);
cusolverStatus_t cusolverDnDpotrs(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, const double* A,
                                  int lda, double* B, int ldb, int* devInfo);
cusolverStatus_t cusolverDnSpotrfBatched(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float* Aarray[],
                                         int lda, int* infoArray, int batchSize);
cusolverStatus_t cusolverDnDpotrfBatched(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double* Aarray[],
                                         int lda, int* infoArray, int batchSize);
cusolverStatus_t cusolverDnSpotrsBatched(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs,
                                         float* Aarray[], int lda, float* Barray[], int ldb, int* info, int batchSize);
cusolverStatus_t cusolverDnDpotrsBatched(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs,
                                         double* Aarray[], int lda, double* Barray[], int ldb, int* info,
                                         int batchSize);

// 64-bit API (cusolverDnX*): the data type selects f32 or f64
cusolverStatus_t cusolverDnXpotrf_bufferSize(cusolverDnHandle_t handle, cusolverDnParams_t params,
                                             cublasFillMode_t uplo, int64_t n, cudaDataType dataTypeA, const void* A,
                                             int64_t lda, cudaDataType computeType, size_t* workspaceInBytesOnDevice,
                                             size_t* workspaceInBytesOnHost);
cusolverStatus_t cusolverDnXpotrf(cusolverDnHandle_t handle, cusolverDnParams_t params, cublasFillMode_t uplo,
                                  int64_t n, cudaDataType dataTypeA, void* A, int64_t lda, cudaDataType computeType,
                                  void* bufferOnDevice, size_t workspaceInBytesOnDevice, void* bufferOnHost,
                                  size_t workspaceInBytesOnHost, int* info);
cusolverStatus_t cusolverDnXpotrs(cusolverDnHandle_t handle, cusolverDnParams_t params, cublasFillMode_t uplo,
                                  int64_t n, int64_t nrhs, cudaDataType dataTypeA, const void* A, int64_t lda,
                                  cudaDataType dataTypeB, void* B, int64_t ldb, int* info);

// Householder QR, the explicit Q and products with Q
cusolverStatus_t cusolverDnSgeqrf_bufferSize(cusolverDnHandle_t handle, int m, int n, float* A, int lda, int* lwork);
cusolverStatus_t cusolverDnDgeqrf_bufferSize(cusolverDnHandle_t handle, int m, int n, double* A, int lda, int* lwork);
cusolverStatus_t cusolverDnSgeqrf(cusolverDnHandle_t handle, int m, int n, float* A, int lda, float* tau,
                                  float* workspace, int lwork, int* devInfo);
cusolverStatus_t cusolverDnDgeqrf(cusolverDnHandle_t handle, int m, int n, double* A, int lda, double* tau,
                                  double* workspace, int lwork, int* devInfo);
cusolverStatus_t cusolverDnSorgqr(cusolverDnHandle_t handle, int m, int n, int k, float* A, int lda, const float* tau,
                                  float* work, int lwork, int* devInfo);
cusolverStatus_t cusolverDnDorgqr(cusolverDnHandle_t handle, int m, int n, int k, double* A, int lda,
                                  const double* tau, double* work, int lwork, int* devInfo);
cusolverStatus_t cusolverDnSormqr(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m,
                                  int n, int k, const float* A, int lda, const float* tau, float* C, int ldc,
                                  float* work, int lwork, int* devInfo);
cusolverStatus_t cusolverDnDormqr(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m,
                                  int n, int k, const double* A, int lda, const double* tau, double* C, int ldc,
                                  double* work, int lwork, int* devInfo);
cusolverStatus_t cusolverDnXgeqrf(cusolverDnHandle_t handle, cusolverDnParams_t params, int64_t m, int64_t n,
                                  cudaDataType dataTypeA, void* A, int64_t lda, cudaDataType dataTypeTau, void* tau,
                                  cudaDataType computeType, void* bufferOnDevice, size_t workspaceInBytesOnDevice,
                                  void* bufferOnHost, size_t workspaceInBytesOnHost, int* info);

// Symmetric eigenvalue problems: eigenvalues ascending in W, eigenvectors in place of A
cusolverStatus_t cusolverDnCreateSyevjInfo(syevjInfo_t* info);
cusolverStatus_t cusolverDnDestroySyevjInfo(syevjInfo_t info);
cusolverStatus_t cusolverDnXsyevjSetTolerance(syevjInfo_t info, double tolerance);
cusolverStatus_t cusolverDnXsyevjSetMaxSweeps(syevjInfo_t info, int maxSweeps);
cusolverStatus_t cusolverDnXsyevjSetSortEig(syevjInfo_t info, int sortEig);
cusolverStatus_t cusolverDnSsyevd_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo,
                                             int n, const float* A, int lda, const float* W, int* lwork);
cusolverStatus_t cusolverDnDsyevd_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo,
                                             int n, const double* A, int lda, const double* W, int* lwork);
cusolverStatus_t cusolverDnSsyevd(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n,
                                  float* A, int lda, float* W, float* work, int lwork, int* devInfo);
cusolverStatus_t cusolverDnDsyevd(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n,
                                  double* A, int lda, double* W, double* work, int lwork, int* devInfo);
cusolverStatus_t cusolverDnSsyevj_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo,
                                             int n, const float* A, int lda, const float* W, int* lwork,
                                             syevjInfo_t params);
cusolverStatus_t cusolverDnDsyevj_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo,
                                             int n, const double* A, int lda, const double* W, int* lwork,
                                             syevjInfo_t params);
cusolverStatus_t cusolverDnSsyevj(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n,
                                  float* A, int lda, float* W, float* work, int lwork, int* info, syevjInfo_t params);
cusolverStatus_t cusolverDnDsyevj(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n,
                                  double* A, int lda, double* W, double* work, int lwork, int* info,
                                  syevjInfo_t params);
cusolverStatus_t cusolverDnSsyevjBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
                                                    cublasFillMode_t uplo, int n, const float* A, int lda,
                                                    const float* W, int* lwork, syevjInfo_t params, int batchSize);
cusolverStatus_t cusolverDnDsyevjBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
                                                    cublasFillMode_t uplo, int n, const double* A, int lda,
                                                    const double* W, int* lwork, syevjInfo_t params, int batchSize);
cusolverStatus_t cusolverDnSsyevjBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo,
                                         int n, float* A, int lda, float* W, float* work, int lwork, int* info,
                                         syevjInfo_t params, int batchSize);
cusolverStatus_t cusolverDnDsyevjBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo,
                                         int n, double* A, int lda, double* W, double* work, int lwork, int* info,
                                         syevjInfo_t params, int batchSize);
cusolverStatus_t cusolverDnXsyevd_bufferSize(cusolverDnHandle_t handle, cusolverDnParams_t params,
                                             cusolverEigMode_t jobz, cublasFillMode_t uplo, int64_t n,
                                             cudaDataType dataTypeA, const void* A, int64_t lda,
                                             cudaDataType dataTypeW, const void* W, cudaDataType computeType,
                                             size_t* workspaceInBytesOnDevice, size_t* workspaceInBytesOnHost);
cusolverStatus_t cusolverDnXsyevd(cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverEigMode_t jobz,
                                  cublasFillMode_t uplo, int64_t n, cudaDataType dataTypeA, void* A, int64_t lda,
                                  cudaDataType dataTypeW, void* W, cudaDataType computeType, void* bufferOnDevice,
                                  size_t workspaceInBytesOnDevice, void* bufferOnHost, size_t workspaceInBytesOnHost,
                                  int* info);
cusolverStatus_t cusolverDnXsyevBatched_bufferSize(cusolverDnHandle_t handle, cusolverDnParams_t params,
                                                   cusolverEigMode_t jobz, cublasFillMode_t uplo, int64_t n,
                                                   cudaDataType dataTypeA, const void* A, int64_t lda,
                                                   cudaDataType dataTypeW, const void* W, cudaDataType computeType,
                                                   size_t* workspaceInBytesOnDevice, size_t* workspaceInBytesOnHost,
                                                   int64_t batchSize);
cusolverStatus_t cusolverDnXsyevBatched(cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverEigMode_t jobz,
                                        cublasFillMode_t uplo, int64_t n, cudaDataType dataTypeA, void* A,
                                        int64_t lda, cudaDataType dataTypeW, void* W, cudaDataType computeType,
                                        void* bufferOnDevice, size_t workspaceInBytesOnDevice, void* bufferOnHost,
                                        size_t workspaceInBytesOnHost, int* info, int64_t batchSize);

// Singular value decomposition A = U diag(S) V^T (gesvd: V^T), singular values descending
cusolverStatus_t cusolverDnCreateGesvdjInfo(gesvdjInfo_t* info);
cusolverStatus_t cusolverDnDestroyGesvdjInfo(gesvdjInfo_t info);
cusolverStatus_t cusolverDnXgesvdjSetTolerance(gesvdjInfo_t info, double tolerance);
cusolverStatus_t cusolverDnXgesvdjSetMaxSweeps(gesvdjInfo_t info, int maxSweeps);
cusolverStatus_t cusolverDnXgesvdjSetSortEig(gesvdjInfo_t info, int sortSvd);
cusolverStatus_t cusolverDnSgesvdj_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m,
                                              int n, const float* A, int lda, const float* S, const float* U, int ldu,
                                              const float* V, int ldv, int* lwork, gesvdjInfo_t params);
cusolverStatus_t cusolverDnDgesvdj_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m,
                                              int n, const double* A, int lda, const double* S, const double* U,
                                              int ldu, const double* V, int ldv, int* lwork, gesvdjInfo_t params);
cusolverStatus_t cusolverDnSgesvdj(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, float* A,
                                   int lda, float* S, float* U, int ldu, float* V, int ldv, float* work, int lwork,
                                   int* info, gesvdjInfo_t params);
cusolverStatus_t cusolverDnDgesvdj(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n,
                                   double* A, int lda, double* S, double* U, int ldu, double* V, int ldv, double* work,
                                   int lwork, int* info, gesvdjInfo_t params);
cusolverStatus_t cusolverDnSgesvdjBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n,
                                                     const float* A, int lda, const float* S, const float* U, int ldu,
                                                     const float* V, int ldv, int* lwork, gesvdjInfo_t params,
                                                     int batchSize);
cusolverStatus_t cusolverDnDgesvdjBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n,
                                                     const double* A, int lda, const double* S, const double* U,
                                                     int ldu, const double* V, int ldv, int* lwork,
                                                     gesvdjInfo_t params, int batchSize);
cusolverStatus_t cusolverDnSgesvdjBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, float* A,
                                          int lda, float* S, float* U, int ldu, float* V, int ldv, float* work,
                                          int lwork, int* info, gesvdjInfo_t params, int batchSize);
cusolverStatus_t cusolverDnDgesvdjBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, double* A,
                                          int lda, double* S, double* U, int ldu, double* V, int ldv, double* work,
                                          int lwork, int* info, gesvdjInfo_t params, int batchSize);
cusolverStatus_t cusolverDnSgesvd_bufferSize(cusolverDnHandle_t handle, int m, int n, int* lwork);
cusolverStatus_t cusolverDnDgesvd_bufferSize(cusolverDnHandle_t handle, int m, int n, int* lwork);
cusolverStatus_t cusolverDnSgesvd(cusolverDnHandle_t handle, signed char jobu, signed char jobvt, int m, int n,
                                  float* A, int lda, float* S, float* U, int ldu, float* VT, int ldvt, float* work,
                                  int lwork, float* rwork, int* info);
cusolverStatus_t cusolverDnDgesvd(cusolverDnHandle_t handle, signed char jobu, signed char jobvt, int m, int n,
                                  double* A, int lda, double* S, double* U, int ldu, double* VT, int ldvt,
                                  double* work, int lwork, double* rwork, int* info);

#ifdef __cplusplus
}
#endif
