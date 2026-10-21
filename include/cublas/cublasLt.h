#pragma once

// Clean-room declarations of the cuBLASLt API subset implemented by Vulkanizer's libcublasLt.so,
// written from the public cuBLAS documentation. Enum values follow the documented ABI
// (tests/cublaslt_abi checks them against the official header when it is available).

#include <cublas_v2.h>

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// A cuBLAS handle doubles as a cuBLASLt handle (applications such as PyTorch rely on that).
typedef struct cublasContext* cublasLtHandle_t;

// Descriptors: pointers to Vulkanizer's own structures.
typedef struct cublasLtMatmulDescOpaque_st* cublasLtMatmulDesc_t;
typedef struct cublasLtMatrixLayoutOpaque_st* cublasLtMatrixLayout_t;
typedef struct cublasLtMatmulPreferenceOpaque_st* cublasLtMatmulPreference_t;

typedef struct {
    uint64_t data[8];
} cublasLtMatmulAlgo_t;

typedef struct {
    cublasLtMatmulAlgo_t algo;
    size_t workspaceSize;
    cublasStatus_t state;
    float wavesCount;
    int reserved[4];
} cublasLtMatmulHeuristicResult_t;

typedef enum {
    CUBLASLT_ORDER_COL = 0,
    CUBLASLT_ORDER_ROW = 1,
} cublasLtOrder_t;

typedef enum {
    CUBLASLT_MATRIX_LAYOUT_TYPE = 0,
    CUBLASLT_MATRIX_LAYOUT_ORDER = 1,
    CUBLASLT_MATRIX_LAYOUT_ROWS = 2,
    CUBLASLT_MATRIX_LAYOUT_COLS = 3,
    CUBLASLT_MATRIX_LAYOUT_LD = 4,
    CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT = 5,
    CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET = 6,
    CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET = 7,
} cublasLtMatrixLayoutAttribute_t;

typedef enum {
    CUBLASLT_POINTER_MODE_HOST = CUBLAS_POINTER_MODE_HOST,
    CUBLASLT_POINTER_MODE_DEVICE = CUBLAS_POINTER_MODE_DEVICE,
    CUBLASLT_POINTER_MODE_DEVICE_VECTOR = 2,
    CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO = 3,
    CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST = 4,
} cublasLtPointerMode_t;

typedef enum {
    CUBLASLT_MATMUL_DESC_COMPUTE_TYPE = 0,
    CUBLASLT_MATMUL_DESC_SCALE_TYPE = 1,
    CUBLASLT_MATMUL_DESC_POINTER_MODE = 2,
    CUBLASLT_MATMUL_DESC_TRANSA = 3,
    CUBLASLT_MATMUL_DESC_TRANSB = 4,
    CUBLASLT_MATMUL_DESC_TRANSC = 5,
    CUBLASLT_MATMUL_DESC_FILL_MODE = 6,
    CUBLASLT_MATMUL_DESC_EPILOGUE = 7,
    CUBLASLT_MATMUL_DESC_BIAS_POINTER = 8,
    CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE = 10,
    CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER = 11,
    CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD = 12,
    CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE = 13,
    CUBLASLT_MATMUL_DESC_ALPHA_VECTOR_BATCH_STRIDE = 14,
    CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET = 15,
    CUBLASLT_MATMUL_DESC_A_SCALE_POINTER = 17,
    CUBLASLT_MATMUL_DESC_B_SCALE_POINTER = 18,
    CUBLASLT_MATMUL_DESC_C_SCALE_POINTER = 19,
    CUBLASLT_MATMUL_DESC_D_SCALE_POINTER = 20,
    CUBLASLT_MATMUL_DESC_AMAX_D_POINTER = 21,
    CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE = 22,
    CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER = 23,
    CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_AMAX_POINTER = 24,
    CUBLASLT_MATMUL_DESC_FAST_ACCUM = 25,
    CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE = 26,
} cublasLtMatmulDescAttributes_t;

typedef enum {
    CUBLASLT_EPILOGUE_DEFAULT = 1,
    CUBLASLT_EPILOGUE_RELU = 2,
    CUBLASLT_EPILOGUE_RELU_AUX = CUBLASLT_EPILOGUE_RELU | 128,
    CUBLASLT_EPILOGUE_BIAS = 4,
    CUBLASLT_EPILOGUE_RELU_BIAS = CUBLASLT_EPILOGUE_RELU | CUBLASLT_EPILOGUE_BIAS,
    CUBLASLT_EPILOGUE_RELU_AUX_BIAS = CUBLASLT_EPILOGUE_RELU_AUX | CUBLASLT_EPILOGUE_BIAS,
    CUBLASLT_EPILOGUE_DRELU = 8 | 128,
    CUBLASLT_EPILOGUE_DRELU_BGRAD = CUBLASLT_EPILOGUE_DRELU | 16,
    CUBLASLT_EPILOGUE_GELU = 32,
    CUBLASLT_EPILOGUE_GELU_AUX = CUBLASLT_EPILOGUE_GELU | 128,
    CUBLASLT_EPILOGUE_GELU_BIAS = CUBLASLT_EPILOGUE_GELU | CUBLASLT_EPILOGUE_BIAS,
    CUBLASLT_EPILOGUE_GELU_AUX_BIAS = CUBLASLT_EPILOGUE_GELU_AUX | CUBLASLT_EPILOGUE_BIAS,
    CUBLASLT_EPILOGUE_DGELU = 64 | 128,
    CUBLASLT_EPILOGUE_DGELU_BGRAD = CUBLASLT_EPILOGUE_DGELU | 16,
    CUBLASLT_EPILOGUE_BGRADA = 256,
    CUBLASLT_EPILOGUE_BGRADB = 512,
} cublasLtEpilogue_t;

typedef enum {
    CUBLASLT_MATMUL_PREF_SEARCH_MODE = 0,
    CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 1,
    CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK = 3,
    CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES = 5,
    CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES = 6,
    CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES = 7,
    CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES = 8,
    CUBLASLT_MATMUL_PREF_MAX_WAVES_COUNT = 9,
    CUBLASLT_MATMUL_PREF_IMPL_MASK = 12,
} cublasLtMatmulPreferenceAttributes_t;

cublasStatus_t cublasLtCreate(cublasLtHandle_t* lightHandle);
cublasStatus_t cublasLtDestroy(cublasLtHandle_t lightHandle);
size_t cublasLtGetVersion(void);
size_t cublasLtGetCudartVersion(void);

cublasStatus_t cublasLtMatmulDescCreate(cublasLtMatmulDesc_t* matmulDesc, cublasComputeType_t computeType,
                                        cudaDataType scaleType);
cublasStatus_t cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t matmulDesc);
cublasStatus_t cublasLtMatmulDescSetAttribute(cublasLtMatmulDesc_t matmulDesc, cublasLtMatmulDescAttributes_t attr,
                                              const void* buf, size_t sizeInBytes);
cublasStatus_t cublasLtMatmulDescGetAttribute(cublasLtMatmulDesc_t matmulDesc, cublasLtMatmulDescAttributes_t attr,
                                              void* buf, size_t sizeInBytes, size_t* sizeWritten);

cublasStatus_t cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t* matLayout, cudaDataType type, uint64_t rows,
                                          uint64_t cols, int64_t ld);
cublasStatus_t cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t matLayout);
cublasStatus_t cublasLtMatrixLayoutSetAttribute(cublasLtMatrixLayout_t matLayout,
                                                cublasLtMatrixLayoutAttribute_t attr, const void* buf,
                                                size_t sizeInBytes);
cublasStatus_t cublasLtMatrixLayoutGetAttribute(cublasLtMatrixLayout_t matLayout,
                                                cublasLtMatrixLayoutAttribute_t attr, void* buf, size_t sizeInBytes,
                                                size_t* sizeWritten);

cublasStatus_t cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t* pref);
cublasStatus_t cublasLtMatmulPreferenceDestroy(cublasLtMatmulPreference_t pref);
cublasStatus_t cublasLtMatmulPreferenceSetAttribute(cublasLtMatmulPreference_t pref,
                                                    cublasLtMatmulPreferenceAttributes_t attr, const void* buf,
                                                    size_t sizeInBytes);
cublasStatus_t cublasLtMatmulPreferenceGetAttribute(cublasLtMatmulPreference_t pref,
                                                    cublasLtMatmulPreferenceAttributes_t attr, void* buf,
                                                    size_t sizeInBytes, size_t* sizeWritten);

cublasStatus_t cublasLtMatmulAlgoGetHeuristic(cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t operationDesc,
                                              cublasLtMatrixLayout_t Adesc, cublasLtMatrixLayout_t Bdesc,
                                              cublasLtMatrixLayout_t Cdesc, cublasLtMatrixLayout_t Ddesc,
                                              cublasLtMatmulPreference_t preference, int requestedAlgoCount,
                                              cublasLtMatmulHeuristicResult_t heuristicResultsArray[],
                                              int* returnAlgoCount);

// D = epilogue(alpha * op(A) * op(B) + beta * C)
cublasStatus_t cublasLtMatmul(cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t computeDesc, const void* alpha,
                              const void* A, cublasLtMatrixLayout_t Adesc, const void* B,
                              cublasLtMatrixLayout_t Bdesc, const void* beta, const void* C,
                              cublasLtMatrixLayout_t Cdesc, void* D, cublasLtMatrixLayout_t Ddesc,
                              const cublasLtMatmulAlgo_t* algo, void* workspace, size_t workspaceSizeInBytes,
                              cudaStream_t stream);

#ifdef __cplusplus
}
#endif
