#pragma once

// Clean-room declarations of the cuBLAS API subset implemented by Vulkanizer's libcublas.so,
// written from the public cuBLAS documentation. Enum values follow the documented ABI.

#if __has_include(<cuda_runtime_api.h>)
#include <cuda_runtime_api.h>
#else
typedef struct CUstream_st* cudaStream_t;  // building without the CUDA toolkit (Vulkanizer itself)
#endif
#include <stddef.h>
#include <stdint.h>

#if __has_include(<library_types.h>)
#include <library_types.h>
#else
// Data and library property types (normally from the CUDA toolkit's library_types.h).
typedef enum cudaDataType_t {
    CUDA_R_16F = 2,
    CUDA_C_16F = 6,
    CUDA_R_16BF = 14,
    CUDA_C_16BF = 15,
    CUDA_R_32F = 0,
    CUDA_C_32F = 4,
    CUDA_R_64F = 1,
    CUDA_C_64F = 5,
    CUDA_R_8I = 3,
    CUDA_C_8I = 7,
    CUDA_R_8U = 8,
    CUDA_C_8U = 9,
    CUDA_R_32I = 10,
    CUDA_C_32I = 11,
    CUDA_R_32U = 12,
    CUDA_C_32U = 13,
} cudaDataType;
typedef enum libraryPropertyType_t { MAJOR_VERSION, MINOR_VERSION, PATCH_LEVEL } libraryPropertyType;
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define CUBLAS_VER_MAJOR 12
#define CUBLAS_VER_MINOR 9
#define CUBLAS_VER_PATCH 0
#define CUBLAS_VERSION (CUBLAS_VER_MAJOR * 10000 + CUBLAS_VER_MINOR * 100 + CUBLAS_VER_PATCH)

#ifndef CUBLASAPI
#define CUBLASAPI
#endif
#ifndef CUBLASWINAPI
#define CUBLASWINAPI
#endif

typedef enum {
    CUBLAS_STATUS_SUCCESS = 0,
    CUBLAS_STATUS_NOT_INITIALIZED = 1,
    CUBLAS_STATUS_ALLOC_FAILED = 3,
    CUBLAS_STATUS_INVALID_VALUE = 7,
    CUBLAS_STATUS_ARCH_MISMATCH = 8,
    CUBLAS_STATUS_MAPPING_ERROR = 11,
    CUBLAS_STATUS_EXECUTION_FAILED = 13,
    CUBLAS_STATUS_INTERNAL_ERROR = 14,
    CUBLAS_STATUS_NOT_SUPPORTED = 15,
    CUBLAS_STATUS_LICENSE_ERROR = 16
} cublasStatus_t;

typedef enum { CUBLAS_FILL_MODE_LOWER = 0, CUBLAS_FILL_MODE_UPPER = 1, CUBLAS_FILL_MODE_FULL = 2 } cublasFillMode_t;
typedef enum { CUBLAS_DIAG_NON_UNIT = 0, CUBLAS_DIAG_UNIT = 1 } cublasDiagType_t;
typedef enum { CUBLAS_SIDE_LEFT = 0, CUBLAS_SIDE_RIGHT = 1 } cublasSideMode_t;
typedef enum {
    CUBLAS_OP_N = 0,
    CUBLAS_OP_T = 1,
    CUBLAS_OP_C = 2,
    CUBLAS_OP_HERMITAN = 2,
    CUBLAS_OP_CONJG = 3
} cublasOperation_t;
typedef enum { CUBLAS_POINTER_MODE_HOST = 0, CUBLAS_POINTER_MODE_DEVICE = 1 } cublasPointerMode_t;
typedef enum { CUBLAS_ATOMICS_NOT_ALLOWED = 0, CUBLAS_ATOMICS_ALLOWED = 1 } cublasAtomicsMode_t;

typedef enum {
    CUBLAS_GEMM_DFALT = -1,
    CUBLAS_GEMM_DEFAULT = -1,
    CUBLAS_GEMM_ALGO0 = 0,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP = 99,
    CUBLAS_GEMM_DFALT_TENSOR_OP = 99,
    CUBLAS_GEMM_ALGO0_TENSOR_OP = 100
} cublasGemmAlgo_t;

typedef enum {
    CUBLAS_DEFAULT_MATH = 0,
    CUBLAS_TENSOR_OP_MATH = 1,
    CUBLAS_PEDANTIC_MATH = 2,
    CUBLAS_TF32_TENSOR_OP_MATH = 3,
    CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION = 16
} cublasMath_t;

typedef enum {
    CUBLAS_COMPUTE_16F = 64,
    CUBLAS_COMPUTE_16F_PEDANTIC = 65,
    CUBLAS_COMPUTE_32F = 68,
    CUBLAS_COMPUTE_32F_PEDANTIC = 69,
    CUBLAS_COMPUTE_32F_FAST_16F = 74,
    CUBLAS_COMPUTE_32F_FAST_16BF = 75,
    CUBLAS_COMPUTE_32F_FAST_TF32 = 77,
    CUBLAS_COMPUTE_64F = 70,
    CUBLAS_COMPUTE_64F_PEDANTIC = 71,
    CUBLAS_COMPUTE_32I = 72,
    CUBLAS_COMPUTE_32I_PEDANTIC = 73
} cublasComputeType_t;

typedef struct cublasContext* cublasHandle_t;

cublasStatus_t CUBLASWINAPI cublasCreate_v2(cublasHandle_t* handle);
cublasStatus_t CUBLASWINAPI cublasDestroy_v2(cublasHandle_t handle);
cublasStatus_t CUBLASWINAPI cublasGetVersion_v2(cublasHandle_t handle, int* version);
cublasStatus_t CUBLASWINAPI cublasGetProperty(libraryPropertyType type, int* value);
size_t CUBLASWINAPI cublasGetCudartVersion(void);
cublasStatus_t CUBLASWINAPI cublasSetStream_v2(cublasHandle_t handle, cudaStream_t streamId);
cublasStatus_t CUBLASWINAPI cublasGetStream_v2(cublasHandle_t handle, cudaStream_t* streamId);
cublasStatus_t CUBLASWINAPI cublasSetWorkspace_v2(cublasHandle_t handle, void* workspace, size_t workspaceSizeInBytes);
cublasStatus_t CUBLASWINAPI cublasSetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t mode);
cublasStatus_t CUBLASWINAPI cublasGetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t* mode);
cublasStatus_t CUBLASWINAPI cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode);
cublasStatus_t CUBLASWINAPI cublasGetMathMode(cublasHandle_t handle, cublasMath_t* mode);
cublasStatus_t CUBLASWINAPI cublasSetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t mode);
cublasStatus_t CUBLASWINAPI cublasGetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t* mode);
const char* CUBLASWINAPI cublasGetStatusName(cublasStatus_t status);
const char* CUBLASWINAPI cublasGetStatusString(cublasStatus_t status);
cublasStatus_t CUBLASWINAPI cublasLoggerConfigure(int logIsOn, int logToStdOut, int logToStdErr,
                                                  const char* logFileName);

cublasStatus_t CUBLASWINAPI cublasSgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                                           int m, int n, int k, const float* alpha, const float* A, int lda,
                                           const float* B, int ldb, const float* beta, float* C, int ldc);
cublasStatus_t CUBLASWINAPI cublasSgemmBatched(cublasHandle_t handle, cublasOperation_t transa,
                                               cublasOperation_t transb, int m, int n, int k, const float* alpha,
                                               const float* const Aarray[], int lda, const float* const Barray[],
                                               int ldb, const float* beta, float* const Carray[], int ldc,
                                               int batchCount);
cublasStatus_t CUBLASWINAPI cublasSgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa,
                                                      cublasOperation_t transb, int m, int n, int k,
                                                      const float* alpha, const float* A, int lda,
                                                      long long int strideA, const float* B, int ldb,
                                                      long long int strideB, const float* beta, float* C, int ldc,
                                                      long long int strideC, int batchCount);
cublasStatus_t CUBLASWINAPI cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                                         int m, int n, int k, const void* alpha, const void* A, cudaDataType Atype,
                                         int lda, const void* B, cudaDataType Btype, int ldb, const void* beta,
                                         void* C, cudaDataType Ctype, int ldc, cublasComputeType_t computeType,
                                         cublasGemmAlgo_t algo);
cublasStatus_t CUBLASWINAPI cublasGemmBatchedEx(cublasHandle_t handle, cublasOperation_t transa,
                                                cublasOperation_t transb, int m, int n, int k, const void* alpha,
                                                const void* const Aarray[], cudaDataType Atype, int lda,
                                                const void* const Barray[], cudaDataType Btype, int ldb,
                                                const void* beta, void* const Carray[], cudaDataType Ctype, int ldc,
                                                int batchCount, cublasComputeType_t computeType,
                                                cublasGemmAlgo_t algo);
cublasStatus_t CUBLASWINAPI cublasGemmStridedBatchedEx(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
    const void* alpha, const void* A, cudaDataType Atype, int lda, long long int strideA, const void* B,
    cudaDataType Btype, int ldb, long long int strideB, const void* beta, void* C, cudaDataType Ctype, int ldc,
    long long int strideC, int batchCount, cublasComputeType_t computeType, cublasGemmAlgo_t algo);
cublasStatus_t CUBLASWINAPI cublasSgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n,
                                           const float* alpha, const float* A, int lda, const float* x, int incx,
                                           const float* beta, float* y, int incy);
// 64-bit forms (cuBLAS 12+): the same, with int64_t sizes; the sizes must still fit an int here.
cublasStatus_t CUBLASWINAPI cublasSgemm_v2_64(cublasHandle_t handle, cublasOperation_t transa,
                                              cublasOperation_t transb, int64_t m, int64_t n, int64_t k,
                                              const float* alpha, const float* A, int64_t lda, const float* B,
                                              int64_t ldb, const float* beta, float* C, int64_t ldc);
cublasStatus_t CUBLASWINAPI cublasSgemmStridedBatched_64(cublasHandle_t handle, cublasOperation_t transa,
                                                         cublasOperation_t transb, int64_t m, int64_t n, int64_t k,
                                                         const float* alpha, const float* A, int64_t lda,
                                                         long long int strideA, const float* B, int64_t ldb,
                                                         long long int strideB, const float* beta, float* C,
                                                         int64_t ldc, long long int strideC, int64_t batchCount);
cublasStatus_t CUBLASWINAPI cublasSgemv_v2_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n,
                                              const float* alpha, const float* A, int64_t lda, const float* x,
                                              int64_t incx, const float* beta, float* y, int64_t incy);
cublasStatus_t CUBLASWINAPI cublasSgemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                                          int m, int n, int k, const float* alpha, const void* A, cudaDataType Atype,
                                          int lda, const void* B, cudaDataType Btype, int ldb, const float* beta,
                                          void* C, cudaDataType Ctype, int ldc);
cublasStatus_t CUBLASWINAPI cublasGemmGroupedBatchedEx(
    cublasHandle_t handle, const cublasOperation_t transa_array[], const cublasOperation_t transb_array[],
    const int m_array[], const int n_array[], const int k_array[], const void* alpha_array,
    const void* const Aarray[], cudaDataType Atype, const int lda_array[], const void* const Barray[],
    cudaDataType Btype, const int ldb_array[], const void* beta_array, void* const Carray[], cudaDataType Ctype,
    const int ldc_array[], int group_count, const int group_size[], cublasComputeType_t computeType);
cublasStatus_t CUBLASWINAPI cublasStrsmBatched(cublasHandle_t handle, cublasSideMode_t side,
                                               cublasFillMode_t uplo, cublasOperation_t trans,
                                               cublasDiagType_t diag, int m, int n, const float* alpha,
                                               const float* const A[], int lda, float* const B[], int ldb,
                                               int batchCount);

#define cublasCreate cublasCreate_v2
#define cublasDestroy cublasDestroy_v2
#define cublasGetVersion cublasGetVersion_v2
#define cublasSetStream cublasSetStream_v2
#define cublasGetStream cublasGetStream_v2
#define cublasSetWorkspace cublasSetWorkspace_v2
#define cublasSetPointerMode cublasSetPointerMode_v2
#define cublasGetPointerMode cublasGetPointerMode_v2
#define cublasSgemm cublasSgemm_v2
#define cublasSgemv cublasSgemv_v2

#ifdef __cplusplus
}
#endif
