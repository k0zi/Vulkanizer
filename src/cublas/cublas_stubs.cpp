// cuBLAS entry points PyTorch links against that Vulkanizer does not implement yet (f64 and complex
// BLAS, BLAS 1/2 routines, batched factorizations). Each one reports CUBLAS_STATUS_NOT_SUPPORTED,
// with a warning the first time. They ignore their arguments, which the C calling convention
// allows, so one parameterless definition serves every signature.

#include <cublas_v2.h>

#include "common/log.h"

#define VULKANIZER_CUBLAS_UNSUPPORTED(X) \
    X(cublasCaxpy_v2) \
    X(cublasCcopy_v2) \
    X(cublasCdotc_v2) \
    X(cublasCdotu_v2) \
    X(cublasCgelsBatched) \
    X(cublasCgemmBatched) \
    X(cublasCgemmStridedBatched) \
    X(cublasCgemm_v2) \
    X(cublasCgemv_v2) \
    X(cublasCgeqrfBatched) \
    X(cublasCgerc_v2) \
    X(cublasCgeru_v2) \
    X(cublasCgetrfBatched) \
    X(cublasCgetrsBatched) \
    X(cublasChemm_v2) \
    X(cublasChemv_v2) \
    X(cublasCher2_v2) \
    X(cublasCher2k_v2) \
    X(cublasCher_v2) \
    X(cublasCherk_v2) \
    X(cublasCrot_v2) \
    X(cublasCrotg_v2) \
    X(cublasCscal_v2) \
    X(cublasCsrot_v2) \
    X(cublasCsscal_v2) \
    X(cublasCswap_v2) \
    X(cublasCsymm_v2) \
    X(cublasCsymv_v2) \
    X(cublasCsyr2_v2) \
    X(cublasCsyr2k_v2) \
    X(cublasCsyr_v2) \
    X(cublasCsyrk_v2) \
    X(cublasCtrmm_v2) \
    X(cublasCtrmv_v2) \
    X(cublasCtrsmBatched) \
    X(cublasCtrsm_v2) \
    X(cublasCtrsv_v2) \
    X(cublasDasum_v2) \
    X(cublasDaxpy_v2) \
    X(cublasDcopy_v2) \
    X(cublasDdot_v2) \
    X(cublasDgelsBatched) \
    X(cublasDgemmBatched) \
    X(cublasDgemmStridedBatched) \
    X(cublasDgemm_v2) \
    X(cublasDgemv_v2) \
    X(cublasDgeqrfBatched) \
    X(cublasDger_v2) \
    X(cublasDgetrfBatched) \
    X(cublasDgetrsBatched) \
    X(cublasDnrm2_v2) \
    X(cublasDotEx) \
    X(cublasDrot_v2) \
    X(cublasDrotg_v2) \
    X(cublasDrotm_v2) \
    X(cublasDrotmg_v2) \
    X(cublasDscal_v2) \
    X(cublasDswap_v2) \
    X(cublasDsymm_v2) \
    X(cublasDsymv_v2) \
    X(cublasDsyr2_v2) \
    X(cublasDsyr2k_v2) \
    X(cublasDsyr_v2) \
    X(cublasDsyrk_v2) \
    X(cublasDtrmm_v2) \
    X(cublasDtrmv_v2) \
    X(cublasDtrsmBatched) \
    X(cublasDtrsm_v2) \
    X(cublasDtrsv_v2) \
    X(cublasDzasum_v2) \
    X(cublasDznrm2_v2) \
    X(cublasGetMatrixAsync) \
    X(cublasGetVectorAsync) \
    X(cublasIcamax_v2) \
    X(cublasIcamin_v2) \
    X(cublasIdamax_v2) \
    X(cublasIdamin_v2) \
    X(cublasIsamax_v2) \
    X(cublasIsamin_v2) \
    X(cublasIzamax_v2) \
    X(cublasIzamin_v2) \
    X(cublasSasum_v2) \
    X(cublasSaxpy_v2) \
    X(cublasScasum_v2) \
    X(cublasScnrm2_v2) \
    X(cublasScopy_v2) \
    X(cublasSdot_v2) \
    X(cublasSetMatrixAsync) \
    X(cublasSetVectorAsync) \
    X(cublasSgelsBatched) \
    X(cublasSgeqrfBatched) \
    X(cublasSger_v2) \
    X(cublasSgetrfBatched) \
    X(cublasSgetrsBatched) \
    X(cublasSnrm2_v2) \
    X(cublasSrot_v2) \
    X(cublasSrotg_v2) \
    X(cublasSrotm_v2) \
    X(cublasSrotmg_v2) \
    X(cublasSscal_v2) \
    X(cublasSswap_v2) \
    X(cublasSsymm_v2) \
    X(cublasSsymv_v2) \
    X(cublasSsyr2_v2) \
    X(cublasSsyr2k_v2) \
    X(cublasSsyr_v2) \
    X(cublasSsyrk_v2) \
    X(cublasStrmm_v2) \
    X(cublasStrmv_v2) \
    X(cublasStrsm_v2) \
    X(cublasStrsv_v2) \
    X(cublasZaxpy_v2) \
    X(cublasZcopy_v2) \
    X(cublasZdotc_v2) \
    X(cublasZdotu_v2) \
    X(cublasZdrot_v2) \
    X(cublasZdscal_v2) \
    X(cublasZgelsBatched) \
    X(cublasZgemmBatched) \
    X(cublasZgemmStridedBatched) \
    X(cublasZgemm_v2) \
    X(cublasZgemv_v2) \
    X(cublasZgeqrfBatched) \
    X(cublasZgerc_v2) \
    X(cublasZgeru_v2) \
    X(cublasZgetrfBatched) \
    X(cublasZgetrsBatched) \
    X(cublasZhemm_v2) \
    X(cublasZhemv_v2) \
    X(cublasZher2_v2) \
    X(cublasZher2k_v2) \
    X(cublasZher_v2) \
    X(cublasZherk_v2) \
    X(cublasZrot_v2) \
    X(cublasZrotg_v2) \
    X(cublasZscal_v2) \
    X(cublasZswap_v2) \
    X(cublasZsymm_v2) \
    X(cublasZsymv_v2) \
    X(cublasZsyr2_v2) \
    X(cublasZsyr2k_v2) \
    X(cublasZsyr_v2) \
    X(cublasZsyrk_v2) \
    X(cublasZtrmm_v2) \
    X(cublasZtrmv_v2) \
    X(cublasZtrsmBatched) \
    X(cublasZtrsm_v2) \
    X(cublasZtrsv_v2) \

#define VULKANIZER_DEFINE_STUB(name)                                              \
    extern "C" cublasStatus_t name() {                                           \
        VULKANIZER_WARN_ONCE("cublas: %s is not implemented", #name);            \
        return CUBLAS_STATUS_NOT_SUPPORTED;                                       \
    }

VULKANIZER_CUBLAS_UNSUPPORTED(VULKANIZER_DEFINE_STUB)
