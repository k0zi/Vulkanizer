// cuSOLVER entry points PyTorch's torch.linalg links against that Vulkanizer does not implement yet
// (complex types, batched gesvda, LDL^T, non-symmetric eigenproblems). Each one reports
// CUSOLVER_STATUS_NOT_SUPPORTED, with a warning the first time. They ignore their arguments, which
// the C calling convention allows, so one parameterless definition serves every signature.

#include <cusolver/cusolverDn.h>

#include "common/log.h"

#define VULKANIZER_CUSOLVER_UNSUPPORTED(X) \
    X(cusolverDnCgeqrf) \
    X(cusolverDnCgeqrf_bufferSize) \
    X(cusolverDnCgesvd) \
    X(cusolverDnCgesvd_bufferSize) \
    X(cusolverDnCgesvdaStridedBatched) \
    X(cusolverDnCgesvdaStridedBatched_bufferSize) \
    X(cusolverDnCgesvdj) \
    X(cusolverDnCgesvdjBatched) \
    X(cusolverDnCgesvdjBatched_bufferSize) \
    X(cusolverDnCgesvdj_bufferSize) \
    X(cusolverDnCgetrf) \
    X(cusolverDnCgetrf_bufferSize) \
    X(cusolverDnCgetrs) \
    X(cusolverDnCheevd) \
    X(cusolverDnCheevd_bufferSize) \
    X(cusolverDnCheevjBatched) \
    X(cusolverDnCheevjBatched_bufferSize) \
    X(cusolverDnCpotrf) \
    X(cusolverDnCpotrfBatched) \
    X(cusolverDnCpotrf_bufferSize) \
    X(cusolverDnCpotrs) \
    X(cusolverDnCpotrsBatched) \
    X(cusolverDnCsytrf) \
    X(cusolverDnCsytrf_bufferSize) \
    X(cusolverDnCungqr) \
    X(cusolverDnCungqr_bufferSize) \
    X(cusolverDnCunmqr) \
    X(cusolverDnCunmqr_bufferSize) \
    X(cusolverDnDgesvdaStridedBatched) \
    X(cusolverDnDgesvdaStridedBatched_bufferSize) \
    X(cusolverDnDsytrf) \
    X(cusolverDnDsytrf_bufferSize) \
    X(cusolverDnSgesvdaStridedBatched) \
    X(cusolverDnSgesvdaStridedBatched_bufferSize) \
    X(cusolverDnSsytrf) \
    X(cusolverDnSsytrf_bufferSize) \
    X(cusolverDnXgeev) \
    X(cusolverDnXgeev_bufferSize) \
    X(cusolverDnXsytrs) \
    X(cusolverDnXsytrs_bufferSize) \
    X(cusolverDnZgeqrf) \
    X(cusolverDnZgeqrf_bufferSize) \
    X(cusolverDnZgesvd) \
    X(cusolverDnZgesvd_bufferSize) \
    X(cusolverDnZgesvdaStridedBatched) \
    X(cusolverDnZgesvdaStridedBatched_bufferSize) \
    X(cusolverDnZgesvdj) \
    X(cusolverDnZgesvdjBatched) \
    X(cusolverDnZgesvdjBatched_bufferSize) \
    X(cusolverDnZgesvdj_bufferSize) \
    X(cusolverDnZgetrf) \
    X(cusolverDnZgetrf_bufferSize) \
    X(cusolverDnZgetrs) \
    X(cusolverDnZheevd) \
    X(cusolverDnZheevd_bufferSize) \
    X(cusolverDnZheevjBatched) \
    X(cusolverDnZheevjBatched_bufferSize) \
    X(cusolverDnZpotrf) \
    X(cusolverDnZpotrfBatched) \
    X(cusolverDnZpotrf_bufferSize) \
    X(cusolverDnZpotrs) \
    X(cusolverDnZpotrsBatched) \
    X(cusolverDnZsytrf) \
    X(cusolverDnZsytrf_bufferSize) \
    X(cusolverDnZungqr) \
    X(cusolverDnZungqr_bufferSize) \
    X(cusolverDnZunmqr) \
    X(cusolverDnZunmqr_bufferSize) \


#define VULKANIZER_DEFINE_STUB(name)                                              \
    extern "C" cusolverStatus_t name() {                                         \
        VULKANIZER_WARN_ONCE("cusolver: %s is not implemented", #name);          \
        return CUSOLVER_STATUS_NOT_SUPPORTED;                                     \
    }

VULKANIZER_CUSOLVER_UNSUPPORTED(VULKANIZER_DEFINE_STUB)
