// cuSPARSE entry points PyTorch links against that Vulkanizer does not implement (block sparse BSR
// products and solves, complex values). Each one reports CUSPARSE_STATUS_NOT_SUPPORTED, with a
// warning the first time. They ignore their arguments, which the C calling convention allows, so
// one parameterless definition serves every signature.

#include <cusparse/cusparse.h>

#include "common/log.h"

#define VULKANIZER_CUSPARSE_UNSUPPORTED(X) \
    X(cusparseCbsrmm) \
    X(cusparseCbsrmv) \
    X(cusparseCbsrsm2_analysis) \
    X(cusparseCbsrsm2_bufferSize) \
    X(cusparseCbsrsm2_solve) \
    X(cusparseCbsrsv2_analysis) \
    X(cusparseCbsrsv2_bufferSize) \
    X(cusparseCbsrsv2_solve) \
    X(cusparseCcsrgeam2) \
    X(cusparseCcsrgeam2_bufferSizeExt) \
    X(cusparseCreateBsrsm2Info) \
    X(cusparseCreateBsrsv2Info) \
    X(cusparseDbsrmm) \
    X(cusparseDbsrmv) \
    X(cusparseDbsrsm2_analysis) \
    X(cusparseDbsrsm2_bufferSize) \
    X(cusparseDbsrsm2_solve) \
    X(cusparseDbsrsv2_analysis) \
    X(cusparseDbsrsv2_bufferSize) \
    X(cusparseDbsrsv2_solve) \
    X(cusparseDestroyBsrsm2Info) \
    X(cusparseDestroyBsrsv2Info) \
    X(cusparseSbsrmm) \
    X(cusparseSbsrmv) \
    X(cusparseSbsrsm2_analysis) \
    X(cusparseSbsrsm2_bufferSize) \
    X(cusparseSbsrsm2_solve) \
    X(cusparseSbsrsv2_analysis) \
    X(cusparseSbsrsv2_bufferSize) \
    X(cusparseSbsrsv2_solve) \
    X(cusparseXbsrsm2_zeroPivot) \
    X(cusparseXbsrsv2_zeroPivot) \
    X(cusparseZbsrmm) \
    X(cusparseZbsrmv) \
    X(cusparseZbsrsm2_analysis) \
    X(cusparseZbsrsm2_bufferSize) \
    X(cusparseZbsrsm2_solve) \
    X(cusparseZbsrsv2_analysis) \
    X(cusparseZbsrsv2_bufferSize) \
    X(cusparseZbsrsv2_solve) \
    X(cusparseZcsrgeam2) \
    X(cusparseZcsrgeam2_bufferSizeExt) \


#define VULKANIZER_DEFINE_STUB(name)                                              \
    extern "C" cusparseStatus_t name() {                                         \
        VULKANIZER_WARN_ONCE("cusparse: %s is not implemented", #name);          \
        return CUSPARSE_STATUS_NOT_SUPPORTED;                                     \
    }

VULKANIZER_CUSPARSE_UNSUPPORTED(VULKANIZER_DEFINE_STUB)
