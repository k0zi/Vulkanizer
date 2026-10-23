#pragma once

// Clean-room declarations of the cuSPARSE API subset implemented by Vulkanizer's libcusparse.so,
// written from the public cuSPARSE documentation. Enum values follow the documented ABI; the data
// types are cuBLAS's (cublas_v2.h).

#include <cublas_v2.h>

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CUSPARSE_STATUS_SUCCESS = 0,
    CUSPARSE_STATUS_NOT_INITIALIZED = 1,
    CUSPARSE_STATUS_ALLOC_FAILED = 2,
    CUSPARSE_STATUS_INVALID_VALUE = 3,
    CUSPARSE_STATUS_ARCH_MISMATCH = 4,
    CUSPARSE_STATUS_MAPPING_ERROR = 5,
    CUSPARSE_STATUS_EXECUTION_FAILED = 6,
    CUSPARSE_STATUS_INTERNAL_ERROR = 7,
    CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 8,
    CUSPARSE_STATUS_ZERO_PIVOT = 9,
    CUSPARSE_STATUS_NOT_SUPPORTED = 10,
    CUSPARSE_STATUS_INSUFFICIENT_RESOURCES = 11,
} cusparseStatus_t;

typedef enum { CUSPARSE_POINTER_MODE_HOST = 0, CUSPARSE_POINTER_MODE_DEVICE = 1 } cusparsePointerMode_t;
typedef enum {
    CUSPARSE_OPERATION_NON_TRANSPOSE = 0,
    CUSPARSE_OPERATION_TRANSPOSE = 1,
    CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE = 2,
} cusparseOperation_t;
typedef enum { CUSPARSE_INDEX_BASE_ZERO = 0, CUSPARSE_INDEX_BASE_ONE = 1 } cusparseIndexBase_t;
typedef enum {
    CUSPARSE_MATRIX_TYPE_GENERAL = 0,
    CUSPARSE_MATRIX_TYPE_SYMMETRIC = 1,
    CUSPARSE_MATRIX_TYPE_HERMITIAN = 2,
    CUSPARSE_MATRIX_TYPE_TRIANGULAR = 3,
} cusparseMatrixType_t;
typedef enum { CUSPARSE_FILL_MODE_LOWER = 0, CUSPARSE_FILL_MODE_UPPER = 1 } cusparseFillMode_t;
typedef enum { CUSPARSE_DIAG_TYPE_NON_UNIT = 0, CUSPARSE_DIAG_TYPE_UNIT = 1 } cusparseDiagType_t;
typedef enum { CUSPARSE_ORDER_COL = 1, CUSPARSE_ORDER_ROW = 2 } cusparseOrder_t;
typedef enum { CUSPARSE_INDEX_16U = 1, CUSPARSE_INDEX_32I = 2, CUSPARSE_INDEX_64I = 3 } cusparseIndexType_t;
typedef enum {
    CUSPARSE_FORMAT_CSR = 1,
    CUSPARSE_FORMAT_CSC = 2,
    CUSPARSE_FORMAT_COO = 3,
    CUSPARSE_FORMAT_BLOCKED_ELL = 5,
    CUSPARSE_FORMAT_BSR = 6,
    CUSPARSE_FORMAT_SLICED_ELLPACK = 7,
} cusparseFormat_t;
typedef enum { CUSPARSE_SPMAT_FILL_MODE = 0, CUSPARSE_SPMAT_DIAG_TYPE = 1 } cusparseSpMatAttribute_t;

// Algorithm choices: accepted, not needed (one implementation each).
typedef int cusparseSpMVAlg_t;
typedef int cusparseSpMMAlg_t;
typedef int cusparseSDDMMAlg_t;
typedef int cusparseSpGEMMAlg_t;
typedef int cusparseSpSVAlg_t;
typedef int cusparseSpSMAlg_t;

typedef struct cusparseContext* cusparseHandle_t;
typedef struct cusparseMatDescr* cusparseMatDescr_t;
typedef struct cusparseSpMatDescr* cusparseSpMatDescr_t;
typedef struct cusparseSpMatDescr const* cusparseConstSpMatDescr_t;
typedef struct cusparseDnMatDescr* cusparseDnMatDescr_t;
typedef struct cusparseDnMatDescr const* cusparseConstDnMatDescr_t;
typedef struct cusparseDnVecDescr* cusparseDnVecDescr_t;
typedef struct cusparseDnVecDescr const* cusparseConstDnVecDescr_t;
typedef struct cusparseSpGEMMDescr* cusparseSpGEMMDescr_t;
typedef struct cusparseSpSVDescr* cusparseSpSVDescr_t;
typedef struct cusparseSpSMDescr* cusparseSpSMDescr_t;

cusparseStatus_t cusparseCreate(cusparseHandle_t* handle);
cusparseStatus_t cusparseDestroy(cusparseHandle_t handle);
cusparseStatus_t cusparseSetStream(cusparseHandle_t handle, cudaStream_t streamId);
cusparseStatus_t cusparseGetStream(cusparseHandle_t handle, cudaStream_t* streamId);
cusparseStatus_t cusparseSetPointerMode(cusparseHandle_t handle, cusparsePointerMode_t mode);
cusparseStatus_t cusparseGetPointerMode(cusparseHandle_t handle, cusparsePointerMode_t* mode);
cusparseStatus_t cusparseGetVersion(cusparseHandle_t handle, int* version);
cusparseStatus_t cusparseGetProperty(libraryPropertyType type, int* value);
const char* cusparseGetErrorName(cusparseStatus_t status);
const char* cusparseGetErrorString(cusparseStatus_t status);

// Legacy matrix descriptors (csrgeam2, sorting)
cusparseStatus_t cusparseCreateMatDescr(cusparseMatDescr_t* descrA);
cusparseStatus_t cusparseDestroyMatDescr(cusparseMatDescr_t descrA);
cusparseStatus_t cusparseSetMatType(cusparseMatDescr_t descrA, cusparseMatrixType_t type);
cusparseStatus_t cusparseSetMatIndexBase(cusparseMatDescr_t descrA, cusparseIndexBase_t base);
cusparseStatus_t cusparseSetMatFillMode(cusparseMatDescr_t descrA, cusparseFillMode_t fillMode);
cusparseStatus_t cusparseSetMatDiagType(cusparseMatDescr_t descrA, cusparseDiagType_t diagType);
cusparseMatrixType_t cusparseGetMatType(const cusparseMatDescr_t descrA);
cusparseIndexBase_t cusparseGetMatIndexBase(const cusparseMatDescr_t descrA);

// Generic API descriptors
cusparseStatus_t cusparseCreateCsr(cusparseSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz,
                                   void* csrRowOffsets, void* csrColInd, void* csrValues,
                                   cusparseIndexType_t csrRowOffsetsType, cusparseIndexType_t csrColIndType,
                                   cusparseIndexBase_t idxBase, cudaDataType valueType);
cusparseStatus_t cusparseCreateCoo(cusparseSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz,
                                   void* cooRowInd, void* cooColInd, void* cooValues, cusparseIndexType_t cooIdxType,
                                   cusparseIndexBase_t idxBase, cudaDataType valueType);
cusparseStatus_t cusparseDestroySpMat(cusparseConstSpMatDescr_t spMatDescr);
cusparseStatus_t cusparseCsrSetPointers(cusparseSpMatDescr_t spMatDescr, void* csrRowOffsets, void* csrColInd,
                                        void* csrValues);
cusparseStatus_t cusparseCsrSetStridedBatch(cusparseSpMatDescr_t spMatDescr, int batchCount,
                                            int64_t offsetsBatchStride, int64_t columnsValuesBatchStride);
cusparseStatus_t cusparseCooSetStridedBatch(cusparseSpMatDescr_t spMatDescr, int batchCount, int64_t batchStride);
cusparseStatus_t cusparseSpMatGetSize(cusparseConstSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols,
                                      int64_t* nnz);
cusparseStatus_t cusparseSpMatSetAttribute(cusparseSpMatDescr_t spMatDescr, cusparseSpMatAttribute_t attribute,
                                           void* data, size_t dataSize);
cusparseStatus_t cusparseCreateDnMat(cusparseDnMatDescr_t* dnMatDescr, int64_t rows, int64_t cols, int64_t ld,
                                     void* values, cudaDataType valueType, cusparseOrder_t order);
cusparseStatus_t cusparseDestroyDnMat(cusparseConstDnMatDescr_t dnMatDescr);
cusparseStatus_t cusparseDnMatSetStridedBatch(cusparseDnMatDescr_t dnMatDescr, int batchCount, int64_t batchStride);
cusparseStatus_t cusparseCreateDnVec(cusparseDnVecDescr_t* dnVecDescr, int64_t size, void* values,
                                     cudaDataType valueType);
cusparseStatus_t cusparseDestroyDnVec(cusparseConstDnVecDescr_t dnVecDescr);

// y = alpha op(A) x + beta y
cusparseStatus_t cusparseSpMV_bufferSize(cusparseHandle_t handle, cusparseOperation_t opA, const void* alpha,
                                         cusparseConstSpMatDescr_t matA, cusparseConstDnVecDescr_t vecX,
                                         const void* beta, cusparseDnVecDescr_t vecY, cudaDataType computeType,
                                         cusparseSpMVAlg_t alg, size_t* bufferSize);
cusparseStatus_t cusparseSpMV(cusparseHandle_t handle, cusparseOperation_t opA, const void* alpha,
                              cusparseConstSpMatDescr_t matA, cusparseConstDnVecDescr_t vecX, const void* beta,
                              cusparseDnVecDescr_t vecY, cudaDataType computeType, cusparseSpMVAlg_t alg,
                              void* externalBuffer);
// C = alpha op(A) op(B) + beta C
cusparseStatus_t cusparseSpMM_bufferSize(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB,
                                         const void* alpha, cusparseConstSpMatDescr_t matA,
                                         cusparseConstDnMatDescr_t matB, const void* beta, cusparseDnMatDescr_t matC,
                                         cudaDataType computeType, cusparseSpMMAlg_t alg, size_t* bufferSize);
cusparseStatus_t cusparseSpMM(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB,
                              const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstDnMatDescr_t matB,
                              const void* beta, cusparseDnMatDescr_t matC, cudaDataType computeType,
                              cusparseSpMMAlg_t alg, void* externalBuffer);
// C = alpha (op(A) op(B)) .* spy(C) + beta C, C sparse (CSR)
cusparseStatus_t cusparseSDDMM_bufferSize(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB,
                                          const void* alpha, cusparseConstDnMatDescr_t matA,
                                          cusparseConstDnMatDescr_t matB, const void* beta, cusparseSpMatDescr_t matC,
                                          cudaDataType computeType, cusparseSDDMMAlg_t alg, size_t* bufferSize);
cusparseStatus_t cusparseSDDMM_preprocess(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB,
                                          const void* alpha, cusparseConstDnMatDescr_t matA,
                                          cusparseConstDnMatDescr_t matB, const void* beta, cusparseSpMatDescr_t matC,
                                          cudaDataType computeType, cusparseSDDMMAlg_t alg, void* externalBuffer);
cusparseStatus_t cusparseSDDMM(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB,
                               const void* alpha, cusparseConstDnMatDescr_t matA, cusparseConstDnMatDescr_t matB,
                               const void* beta, cusparseSpMatDescr_t matC, cudaDataType computeType,
                               cusparseSDDMMAlg_t alg, void* externalBuffer);
// C = alpha op(A) op(B), all CSR
cusparseStatus_t cusparseSpGEMM_createDescr(cusparseSpGEMMDescr_t* descr);
cusparseStatus_t cusparseSpGEMM_destroyDescr(cusparseSpGEMMDescr_t descr);
cusparseStatus_t cusparseSpGEMM_workEstimation(cusparseHandle_t handle, cusparseOperation_t opA,
                                               cusparseOperation_t opB, const void* alpha,
                                               cusparseConstSpMatDescr_t matA, cusparseConstSpMatDescr_t matB,
                                               const void* beta, cusparseSpMatDescr_t matC, cudaDataType computeType,
                                               cusparseSpGEMMAlg_t alg, cusparseSpGEMMDescr_t spgemmDescr,
                                               size_t* bufferSize1, void* externalBuffer1);
cusparseStatus_t cusparseSpGEMM_compute(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB,
                                        const void* alpha, cusparseConstSpMatDescr_t matA,
                                        cusparseConstSpMatDescr_t matB, const void* beta, cusparseSpMatDescr_t matC,
                                        cudaDataType computeType, cusparseSpGEMMAlg_t alg,
                                        cusparseSpGEMMDescr_t spgemmDescr, size_t* bufferSize2,
                                        void* externalBuffer2);
cusparseStatus_t cusparseSpGEMM_copy(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB,
                                     const void* alpha, cusparseConstSpMatDescr_t matA,
                                     cusparseConstSpMatDescr_t matB, const void* beta, cusparseSpMatDescr_t matC,
                                     cudaDataType computeType, cusparseSpGEMMAlg_t alg,
                                     cusparseSpGEMMDescr_t spgemmDescr);
// Triangular solves: op(A) Y = alpha X (SpSV: vectors, SpSM: matrices), A CSR with its fill mode and diagonal type
cusparseStatus_t cusparseSpSV_createDescr(cusparseSpSVDescr_t* descr);
cusparseStatus_t cusparseSpSV_destroyDescr(cusparseSpSVDescr_t descr);
cusparseStatus_t cusparseSpSV_bufferSize(cusparseHandle_t handle, cusparseOperation_t opA, const void* alpha,
                                         cusparseConstSpMatDescr_t matA, cusparseConstDnVecDescr_t vecX,
                                         cusparseDnVecDescr_t vecY, cudaDataType computeType, cusparseSpSVAlg_t alg,
                                         cusparseSpSVDescr_t spsvDescr, size_t* bufferSize);
cusparseStatus_t cusparseSpSV_analysis(cusparseHandle_t handle, cusparseOperation_t opA, const void* alpha,
                                       cusparseConstSpMatDescr_t matA, cusparseConstDnVecDescr_t vecX,
                                       cusparseDnVecDescr_t vecY, cudaDataType computeType, cusparseSpSVAlg_t alg,
                                       cusparseSpSVDescr_t spsvDescr, void* externalBuffer);
cusparseStatus_t cusparseSpSV_solve(cusparseHandle_t handle, cusparseOperation_t opA, const void* alpha,
                                    cusparseConstSpMatDescr_t matA, cusparseConstDnVecDescr_t vecX,
                                    cusparseDnVecDescr_t vecY, cudaDataType computeType, cusparseSpSVAlg_t alg,
                                    cusparseSpSVDescr_t spsvDescr);
cusparseStatus_t cusparseSpSM_createDescr(cusparseSpSMDescr_t* descr);
cusparseStatus_t cusparseSpSM_destroyDescr(cusparseSpSMDescr_t descr);
cusparseStatus_t cusparseSpSM_bufferSize(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB,
                                         const void* alpha, cusparseConstSpMatDescr_t matA,
                                         cusparseConstDnMatDescr_t matB, cusparseDnMatDescr_t matC,
                                         cudaDataType computeType, cusparseSpSMAlg_t alg,
                                         cusparseSpSMDescr_t spsmDescr, size_t* bufferSize);
cusparseStatus_t cusparseSpSM_analysis(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB,
                                       const void* alpha, cusparseConstSpMatDescr_t matA,
                                       cusparseConstDnMatDescr_t matB, cusparseDnMatDescr_t matC,
                                       cudaDataType computeType, cusparseSpSMAlg_t alg, cusparseSpSMDescr_t spsmDescr,
                                       void* externalBuffer);
cusparseStatus_t cusparseSpSM_solve(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB,
                                    const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstDnMatDescr_t matB,
                                    cusparseDnMatDescr_t matC, cudaDataType computeType, cusparseSpSMAlg_t alg,
                                    cusparseSpSMDescr_t spsmDescr);

// C = alpha A + beta B, CSR with 32-bit indices (sorted columns)
cusparseStatus_t cusparseScsrgeam2_bufferSizeExt(cusparseHandle_t handle, int m, int n, const float* alpha,
                                                 const cusparseMatDescr_t descrA, int nnzA, const float* csrSortedValA,
                                                 const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                                 const float* beta, const cusparseMatDescr_t descrB, int nnzB,
                                                 const float* csrSortedValB, const int* csrSortedRowPtrB,
                                                 const int* csrSortedColIndB, const cusparseMatDescr_t descrC,
                                                 const float* csrSortedValC, const int* csrSortedRowPtrC,
                                                 const int* csrSortedColIndC, size_t* pBufferSizeInBytes);
cusparseStatus_t cusparseDcsrgeam2_bufferSizeExt(cusparseHandle_t handle, int m, int n, const double* alpha,
                                                 const cusparseMatDescr_t descrA, int nnzA,
                                                 const double* csrSortedValA, const int* csrSortedRowPtrA,
                                                 const int* csrSortedColIndA, const double* beta,
                                                 const cusparseMatDescr_t descrB, int nnzB,
                                                 const double* csrSortedValB, const int* csrSortedRowPtrB,
                                                 const int* csrSortedColIndB, const cusparseMatDescr_t descrC,
                                                 const double* csrSortedValC, const int* csrSortedRowPtrC,
                                                 const int* csrSortedColIndC, size_t* pBufferSizeInBytes);
cusparseStatus_t cusparseXcsrgeam2Nnz(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, int nnzA,
                                      const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                      const cusparseMatDescr_t descrB, int nnzB, const int* csrSortedRowPtrB,
                                      const int* csrSortedColIndB, const cusparseMatDescr_t descrC,
                                      int* csrSortedRowPtrC, int* nnzTotalDevHostPtr, void* workspace);
cusparseStatus_t cusparseScsrgeam2(cusparseHandle_t handle, int m, int n, const float* alpha,
                                   const cusparseMatDescr_t descrA, int nnzA, const float* csrSortedValA,
                                   const int* csrSortedRowPtrA, const int* csrSortedColIndA, const float* beta,
                                   const cusparseMatDescr_t descrB, int nnzB, const float* csrSortedValB,
                                   const int* csrSortedRowPtrB, const int* csrSortedColIndB,
                                   const cusparseMatDescr_t descrC, float* csrSortedValC, int* csrSortedRowPtrC,
                                   int* csrSortedColIndC, void* pBuffer);
cusparseStatus_t cusparseDcsrgeam2(cusparseHandle_t handle, int m, int n, const double* alpha,
                                   const cusparseMatDescr_t descrA, int nnzA, const double* csrSortedValA,
                                   const int* csrSortedRowPtrA, const int* csrSortedColIndA, const double* beta,
                                   const cusparseMatDescr_t descrB, int nnzB, const double* csrSortedValB,
                                   const int* csrSortedRowPtrB, const int* csrSortedColIndB,
                                   const cusparseMatDescr_t descrC, double* csrSortedValC, int* csrSortedRowPtrC,
                                   int* csrSortedColIndC, void* pBuffer);

// Format conversion and sorting (32-bit indices)
cusparseStatus_t cusparseXcoo2csr(cusparseHandle_t handle, const int* cooRowInd, int nnz, int m, int* csrSortedRowPtr,
                                  cusparseIndexBase_t idxBase);
cusparseStatus_t cusparseCreateIdentityPermutation(cusparseHandle_t handle, int n, int* p);
cusparseStatus_t cusparseXcoosort_bufferSizeExt(cusparseHandle_t handle, int m, int n, int nnz, const int* cooRowsA,
                                                const int* cooColsA, size_t* pBufferSizeInBytes);
cusparseStatus_t cusparseXcoosortByRow(cusparseHandle_t handle, int m, int n, int nnz, int* cooRowsA, int* cooColsA,
                                       int* P, void* pBuffer);
cusparseStatus_t cusparseXcsrsort_bufferSizeExt(cusparseHandle_t handle, int m, int n, int nnz,
                                                const int* csrRowPtrA, const int* csrColIndA,
                                                size_t* pBufferSizeInBytes);
cusparseStatus_t cusparseXcsrsort(cusparseHandle_t handle, int m, int n, int nnz, const cusparseMatDescr_t descrA,
                                  const int* csrRowPtrA, int* csrColIndA, int* P, void* pBuffer);

#ifdef __cplusplus
}
#endif
