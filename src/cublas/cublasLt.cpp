// Vulkanizer's libcublasLt: cublasLtMatmul on top of the cuBLAS GEMMs (public API) plus an epilogue
// kernel (shaders/epilogue.comp) for bias, ReLU and GELU. A cuBLAS handle is the cuBLASLt handle.

#include <cublasLt.h>
#include <cuda.h>
#include <vulkanizer/vulkanizer_ext.h>

#include "common/log.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <mutex>

struct cublasLtMatmulDescOpaque_st {
    cublasComputeType_t compute = CUBLAS_COMPUTE_32F;
    cudaDataType scaleType = CUDA_R_32F;
    int32_t pointerMode = CUBLASLT_POINTER_MODE_HOST;
    int32_t transa = CUBLAS_OP_N, transb = CUBLAS_OP_N, transc = CUBLAS_OP_N;
    uint32_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;
    const void* bias = nullptr;
    int64_t biasBatchStride = 0;
    int32_t biasType = -1;  // -1: the type of D
};

struct cublasLtMatrixLayoutOpaque_st {
    cudaDataType type = CUDA_R_32F;
    int32_t order = CUBLASLT_ORDER_COL;
    uint64_t rows = 0, cols = 0;
    int64_t ld = 0;
    int32_t batchCount = 1;
    int64_t batchStride = 0;
};

struct cublasLtMatmulPreferenceOpaque_st {
    uint64_t maxWorkspace = 0;
};

namespace {

const uint32_t kEpilogueSpirv[] = {
#include "epilogue.comp.inc"
};

// Push constants of epilogue.comp (std430).
struct EpilogueParams {
    uint64_t d, bias;
    int64_t strideD, strideBias;
    int32_t rows, cols, ldd;
    uint32_t flags;
    int32_t batchOffset;
};
static_assert(sizeof(EpilogueParams) == 56, "must match epilogue.comp");
constexpr uint32_t kRelu = 0x10, kGelu = 0x20, kBias = 0x40, kMaxGridZ = 65535;

bool typeCode(cudaDataType t, uint32_t& code) {
    switch (t) {
    case CUDA_R_32F: code = 0; return true;
    case CUDA_R_16F: code = 1; return true;
    case CUDA_R_16BF: code = 2; return true;
    default: return false;
    }
}

size_t elemSize(cudaDataType t) { return t == CUDA_R_32F ? 4 : 2; }

// Epilogue bits for the supported epilogues; false for the others (aux outputs, gradients).
bool epilogueFlags(uint32_t epilogue, uint32_t& flags) {
    switch (epilogue) {
    case CUBLASLT_EPILOGUE_DEFAULT: flags = 0; return true;
    case CUBLASLT_EPILOGUE_RELU: flags = kRelu; return true;
    case CUBLASLT_EPILOGUE_BIAS: flags = kBias; return true;
    case CUBLASLT_EPILOGUE_RELU_BIAS: flags = kRelu | kBias; return true;
    case CUBLASLT_EPILOGUE_GELU: flags = kGelu; return true;
    case CUBLASLT_EPILOGUE_GELU_BIAS: flags = kGelu | kBias; return true;
    default: return false;
    }
}

bool supported(const cublasLtMatmulDescOpaque_st& d, const cublasLtMatrixLayoutOpaque_st* const (&layouts)[4]) {
    uint32_t code, flags;
    for (const auto* l : layouts)
        if (!l || l->order != CUBLASLT_ORDER_COL || !typeCode(l->type, code)) return false;
    if (d.transc != CUBLAS_OP_N || !epilogueFlags(d.epilogue, flags)) return false;
    if (d.biasType >= 0 && !typeCode(cudaDataType(d.biasType), code)) return false;
    return d.pointerMode == CUBLASLT_POINTER_MODE_HOST || d.pointerMode == CUBLASLT_POINTER_MODE_DEVICE;
}

template <class T>
cublasStatus_t setField(T& field, const void* buf, size_t size) {
    if (!buf || size != sizeof(T)) return CUBLAS_STATUS_INVALID_VALUE;
    std::memcpy(&field, buf, sizeof(T));
    return CUBLAS_STATUS_SUCCESS;
}

template <class T>
cublasStatus_t getField(const T& field, void* buf, size_t size, size_t* written) {
    if (written) *written = sizeof(T);
    if (!buf && size == 0) return written ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_INVALID_VALUE;
    if (!buf || size < sizeof(T)) return CUBLAS_STATUS_INVALID_VALUE;
    std::memcpy(buf, &field, sizeof(T));
    return CUBLAS_STATUS_SUCCESS;
}

// The epilogue kernel, created once per context.
CUfunction epilogueKernel() {
    static std::mutex mutex;
    static std::map<CUcontext, CUfunction> kernels;
    CUcontext ctx = nullptr;
    if (cuCtxGetCurrent(&ctx) != CUDA_SUCCESS) return nullptr;
    std::lock_guard<std::mutex> lock(mutex);
    CUfunction& f = kernels[ctx];
    if (!f && vulkanizerCreateSpirvFunction(&f, kEpilogueSpirv, sizeof kEpilogueSpirv / 4, "main",
                                           sizeof(EpilogueParams)) != CUDA_SUCCESS)
        f = nullptr;
    return f;
}

// beta == 0 (read with the scalar's type and pointer mode): C is not read at all.
bool betaIsZero(const void* beta, cudaDataType scaleType, bool device) {
    unsigned char buf[8] = {};
    const size_t size = scaleType == CUDA_R_64F ? 8 : scaleType == CUDA_R_16F || scaleType == CUDA_R_16BF ? 2 : 4;
    if (device) {
        if (cuMemcpyDtoH(buf, reinterpret_cast<CUdeviceptr>(beta), size) != CUDA_SUCCESS) return false;
    } else {
        std::memcpy(buf, beta, size);
    }
    for (size_t i = 0; i < size; ++i)  // +0.0 and -0.0
        if ((buf[i] & (i + 1 == size ? 0x7F : 0xFF)) != 0) return false;
    return true;
}

}  // namespace

extern "C" {

cublasStatus_t cublasLtCreate(cublasLtHandle_t* lightHandle) { return cublasCreate_v2(lightHandle); }
cublasStatus_t cublasLtDestroy(cublasLtHandle_t lightHandle) { return cublasDestroy_v2(lightHandle); }
size_t cublasLtGetVersion(void) { return CUBLAS_VERSION; }
size_t cublasLtGetCudartVersion(void) { return 12090; }

cublasStatus_t cublasLtMatmulDescCreate(cublasLtMatmulDesc_t* matmulDesc, cublasComputeType_t computeType,
                                        cudaDataType scaleType) {
    VULKANIZER_TRACE_API();
    if (!matmulDesc) return CUBLAS_STATUS_INVALID_VALUE;
    *matmulDesc = new cublasLtMatmulDescOpaque_st;
    (*matmulDesc)->compute = computeType;
    (*matmulDesc)->scaleType = scaleType;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t matmulDesc) {
    delete matmulDesc;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulDescSetAttribute(cublasLtMatmulDesc_t d, cublasLtMatmulDescAttributes_t attr,
                                              const void* buf, size_t size) {
    VULKANIZER_TRACE_API();
    if (!d) return CUBLAS_STATUS_INVALID_VALUE;
    switch (attr) {
    case CUBLASLT_MATMUL_DESC_COMPUTE_TYPE: return setField(d->compute, buf, size);
    case CUBLASLT_MATMUL_DESC_SCALE_TYPE: return setField(d->scaleType, buf, size);
    case CUBLASLT_MATMUL_DESC_POINTER_MODE: return setField(d->pointerMode, buf, size);
    case CUBLASLT_MATMUL_DESC_TRANSA: return setField(d->transa, buf, size);
    case CUBLASLT_MATMUL_DESC_TRANSB: return setField(d->transb, buf, size);
    case CUBLASLT_MATMUL_DESC_TRANSC: return setField(d->transc, buf, size);
    case CUBLASLT_MATMUL_DESC_EPILOGUE: return setField(d->epilogue, buf, size);
    case CUBLASLT_MATMUL_DESC_BIAS_POINTER: return setField(d->bias, buf, size);
    case CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE: return setField(d->biasBatchStride, buf, size);
    case CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE: return setField(d->biasType, buf, size);
    // hints without an effect here
    case CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET:
    case CUBLASLT_MATMUL_DESC_FAST_ACCUM:
    case CUBLASLT_MATMUL_DESC_FILL_MODE: return buf ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_INVALID_VALUE;
    default:
        VULKANIZER_WARN_ONCE("cublasLt: matmul descriptor attribute %d is not supported", int(attr));
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
}

cublasStatus_t cublasLtMatmulDescGetAttribute(cublasLtMatmulDesc_t d, cublasLtMatmulDescAttributes_t attr, void* buf,
                                              size_t size, size_t* written) {
    VULKANIZER_TRACE_API();
    if (!d) return CUBLAS_STATUS_INVALID_VALUE;
    switch (attr) {
    case CUBLASLT_MATMUL_DESC_COMPUTE_TYPE: return getField(d->compute, buf, size, written);
    case CUBLASLT_MATMUL_DESC_SCALE_TYPE: return getField(d->scaleType, buf, size, written);
    case CUBLASLT_MATMUL_DESC_POINTER_MODE: return getField(d->pointerMode, buf, size, written);
    case CUBLASLT_MATMUL_DESC_TRANSA: return getField(d->transa, buf, size, written);
    case CUBLASLT_MATMUL_DESC_TRANSB: return getField(d->transb, buf, size, written);
    case CUBLASLT_MATMUL_DESC_TRANSC: return getField(d->transc, buf, size, written);
    case CUBLASLT_MATMUL_DESC_EPILOGUE: return getField(d->epilogue, buf, size, written);
    case CUBLASLT_MATMUL_DESC_BIAS_POINTER: return getField(d->bias, buf, size, written);
    case CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE: return getField(d->biasBatchStride, buf, size, written);
    case CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE: return getField(d->biasType, buf, size, written);
    default: return CUBLAS_STATUS_NOT_SUPPORTED;
    }
}

cublasStatus_t cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t* matLayout, cudaDataType type, uint64_t rows,
                                          uint64_t cols, int64_t ld) {
    VULKANIZER_TRACE_API();
    if (!matLayout) return CUBLAS_STATUS_INVALID_VALUE;
    *matLayout = new cublasLtMatrixLayoutOpaque_st;
    (*matLayout)->type = type;
    (*matLayout)->rows = rows;
    (*matLayout)->cols = cols;
    (*matLayout)->ld = ld;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t matLayout) {
    delete matLayout;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatrixLayoutSetAttribute(cublasLtMatrixLayout_t l, cublasLtMatrixLayoutAttribute_t attr,
                                                const void* buf, size_t size) {
    VULKANIZER_TRACE_API();
    if (!l) return CUBLAS_STATUS_INVALID_VALUE;
    switch (attr) {
    case CUBLASLT_MATRIX_LAYOUT_TYPE: return setField(l->type, buf, size);
    case CUBLASLT_MATRIX_LAYOUT_ORDER: return setField(l->order, buf, size);
    case CUBLASLT_MATRIX_LAYOUT_ROWS: return setField(l->rows, buf, size);
    case CUBLASLT_MATRIX_LAYOUT_COLS: return setField(l->cols, buf, size);
    case CUBLASLT_MATRIX_LAYOUT_LD: return setField(l->ld, buf, size);
    case CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT: return setField(l->batchCount, buf, size);
    case CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET: return setField(l->batchStride, buf, size);
    default:
        VULKANIZER_WARN_ONCE("cublasLt: matrix layout attribute %d is not supported", int(attr));
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
}

cublasStatus_t cublasLtMatrixLayoutGetAttribute(cublasLtMatrixLayout_t l, cublasLtMatrixLayoutAttribute_t attr,
                                                void* buf, size_t size, size_t* written) {
    VULKANIZER_TRACE_API();
    if (!l) return CUBLAS_STATUS_INVALID_VALUE;
    switch (attr) {
    case CUBLASLT_MATRIX_LAYOUT_TYPE: return getField(l->type, buf, size, written);
    case CUBLASLT_MATRIX_LAYOUT_ORDER: return getField(l->order, buf, size, written);
    case CUBLASLT_MATRIX_LAYOUT_ROWS: return getField(l->rows, buf, size, written);
    case CUBLASLT_MATRIX_LAYOUT_COLS: return getField(l->cols, buf, size, written);
    case CUBLASLT_MATRIX_LAYOUT_LD: return getField(l->ld, buf, size, written);
    case CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT: return getField(l->batchCount, buf, size, written);
    case CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET: return getField(l->batchStride, buf, size, written);
    default: return CUBLAS_STATUS_NOT_SUPPORTED;
    }
}

cublasStatus_t cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t* pref) {
    if (!pref) return CUBLAS_STATUS_INVALID_VALUE;
    *pref = new cublasLtMatmulPreferenceOpaque_st;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulPreferenceDestroy(cublasLtMatmulPreference_t pref) {
    delete pref;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulPreferenceSetAttribute(cublasLtMatmulPreference_t pref,
                                                    cublasLtMatmulPreferenceAttributes_t attr, const void* buf,
                                                    size_t size) {
    if (!pref || !buf) return CUBLAS_STATUS_INVALID_VALUE;
    if (attr == CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES) return setField(pref->maxWorkspace, buf, size);
    return CUBLAS_STATUS_SUCCESS;  // search mode, alignments, waves: every request is served the same way
}

cublasStatus_t cublasLtMatmulPreferenceGetAttribute(cublasLtMatmulPreference_t pref,
                                                    cublasLtMatmulPreferenceAttributes_t attr, void* buf,
                                                    size_t size, size_t* written) {
    if (!pref) return CUBLAS_STATUS_INVALID_VALUE;
    if (attr == CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES) return getField(pref->maxWorkspace, buf, size, written);
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

// One "algorithm" (needing no workspace) for every supported problem.
cublasStatus_t cublasLtMatmulAlgoGetHeuristic(cublasLtHandle_t h, cublasLtMatmulDesc_t desc,
                                              cublasLtMatrixLayout_t Adesc, cublasLtMatrixLayout_t Bdesc,
                                              cublasLtMatrixLayout_t Cdesc, cublasLtMatrixLayout_t Ddesc,
                                              cublasLtMatmulPreference_t, int requested,
                                              cublasLtMatmulHeuristicResult_t results[], int* returned) {
    VULKANIZER_TRACE_API();
    if (!h || !desc || requested < 1 || !results || !returned) return CUBLAS_STATUS_INVALID_VALUE;
    *returned = 0;
    const cublasLtMatrixLayoutOpaque_st* const layouts[4] = {Adesc, Bdesc, Cdesc, Ddesc};
    if (!supported(*desc, layouts)) return CUBLAS_STATUS_NOT_SUPPORTED;
    results[0] = {};
    results[0].workspaceSize = 0;
    results[0].state = CUBLAS_STATUS_SUCCESS;
    results[0].wavesCount = 1.0f;
    *returned = 1;
    return CUBLAS_STATUS_SUCCESS;
}

// Every algorithm is the same one: initializing, checking and configuring it is trivial.
cublasStatus_t cublasLtMatmulAlgoInit(cublasLtHandle_t h, cublasComputeType_t, cudaDataType, cudaDataType,
                                      cudaDataType, cudaDataType, cudaDataType, int, cublasLtMatmulAlgo_t* algo) {
    if (!h || !algo) return CUBLAS_STATUS_INVALID_VALUE;
    *algo = {};
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulAlgoCheck(cublasLtHandle_t h, cublasLtMatmulDesc_t desc, cublasLtMatrixLayout_t Adesc,
                                       cublasLtMatrixLayout_t Bdesc, cublasLtMatrixLayout_t Cdesc,
                                       cublasLtMatrixLayout_t Ddesc, const cublasLtMatmulAlgo_t* algo,
                                       cublasLtMatmulHeuristicResult_t* result) {
    if (!h || !desc || !algo || !result) return CUBLAS_STATUS_INVALID_VALUE;
    const cublasLtMatrixLayoutOpaque_st* const layouts[4] = {Adesc, Bdesc, Cdesc, Ddesc};
    if (!supported(*desc, layouts)) return CUBLAS_STATUS_NOT_SUPPORTED;
    *result = {};
    result->algo = *algo;
    result->state = CUBLAS_STATUS_SUCCESS;
    result->wavesCount = 1.0f;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulAlgoConfigSetAttribute(cublasLtMatmulAlgo_t* algo, int, const void* buf, size_t) {
    return algo && buf ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_INVALID_VALUE;
}

cublasStatus_t cublasLtMatmulAlgoConfigGetAttribute(const cublasLtMatmulAlgo_t* algo, int, void* buf, size_t size,
                                                    size_t* written) {
    if (!algo) return CUBLAS_STATUS_INVALID_VALUE;
    // ids, tiles, stages, splits: all 0 (32-bit fields; a few are 16-bit or 64-bit)
    if (buf) std::memset(buf, 0, size);
    if (written) *written = size;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmul(cublasLtHandle_t h, cublasLtMatmulDesc_t desc, const void* alpha, const void* A,
                              cublasLtMatrixLayout_t Adesc, const void* B, cublasLtMatrixLayout_t Bdesc,
                              const void* beta, const void* C, cublasLtMatrixLayout_t Cdesc, void* D,
                              cublasLtMatrixLayout_t Ddesc, const cublasLtMatmulAlgo_t*, void*, size_t,
                              cudaStream_t stream) {
    VULKANIZER_TRACE_API();
    if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (!desc || !Adesc || !Bdesc || !Cdesc || !Ddesc || !alpha || !beta) return CUBLAS_STATUS_INVALID_VALUE;
    const cublasLtMatrixLayoutOpaque_st* const layouts[4] = {Adesc, Bdesc, Cdesc, Ddesc};
    if (!supported(*desc, layouts)) {
        VULKANIZER_WARN_ONCE("cublasLtMatmul: unsupported configuration (epilogue %u, order or types)",
                             desc->epilogue);
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    const auto ta = cublasOperation_t(desc->transa), tb = cublasOperation_t(desc->transb);
    const int m = int(Ddesc->rows), n = int(Ddesc->cols);
    const int k = int(ta == CUBLAS_OP_N ? Adesc->cols : Adesc->rows);
    const int batch = std::max(1, Ddesc->batchCount);
    if (Cdesc->rows != Ddesc->rows || Cdesc->cols != Ddesc->cols) return CUBLAS_STATUS_INVALID_VALUE;
    const bool devicePointers = desc->pointerMode == CUBLASLT_POINTER_MODE_DEVICE;

    cudaStream_t oldStream = nullptr;
    cublasPointerMode_t oldMode = CUBLAS_POINTER_MODE_HOST;
    cublasGetStream_v2(h, &oldStream);
    cublasGetPointerMode_v2(h, &oldMode);
    cublasSetStream_v2(h, stream);
    cublasSetPointerMode_v2(h, devicePointers ? CUBLAS_POINTER_MODE_DEVICE : CUBLAS_POINTER_MODE_HOST);
    cublasStatus_t st = CUBLAS_STATUS_SUCCESS;

    // The GEMMs accumulate in place: D starts as C when C is read and is another matrix.
    if (C != D && !betaIsZero(beta, desc->scaleType, devicePointers)) {
        const size_t es = elemSize(Ddesc->type);
        for (int b = 0; b < batch && st == CUBLAS_STATUS_SUCCESS; ++b)
            for (int j = 0; j < n && st == CUBLAS_STATUS_SUCCESS; ++j) {
                const auto src = reinterpret_cast<CUdeviceptr>(C) + (b * Cdesc->batchStride + j * Cdesc->ld) * es;
                const auto dst = reinterpret_cast<CUdeviceptr>(D) + (b * Ddesc->batchStride + j * Ddesc->ld) * es;
                if (cuMemcpyDtoDAsync(dst, src, size_t(m) * es, stream) != CUDA_SUCCESS)
                    st = CUBLAS_STATUS_EXECUTION_FAILED;
                if (Cdesc->ld == Ddesc->ld && j == 0 && n > 1) {  // same layout: the rest in one copy
                    if (cuMemcpyDtoDAsync(dst + Ddesc->ld * es, src + Cdesc->ld * es,
                                          (size_t(n - 2) * Ddesc->ld + m) * es, stream) != CUDA_SUCCESS)
                        st = CUBLAS_STATUS_EXECUTION_FAILED;
                    break;
                }
            }
    }
    if (st == CUBLAS_STATUS_SUCCESS)
        st = cublasGemmStridedBatchedEx(h, ta, tb, m, n, k, alpha, A, Adesc->type, int(Adesc->ld),
                                        Adesc->batchCount > 1 ? Adesc->batchStride : 0, B, Bdesc->type,
                                        int(Bdesc->ld), Bdesc->batchCount > 1 ? Bdesc->batchStride : 0, beta, D,
                                        Ddesc->type, int(Ddesc->ld), Ddesc->batchStride, batch, desc->compute,
                                        CUBLAS_GEMM_DEFAULT);

    uint32_t flags = 0;
    epilogueFlags(desc->epilogue, flags);
    if (st == CUBLAS_STATUS_SUCCESS && flags) {
        if ((flags & kBias) && !desc->bias) st = CUBLAS_STATUS_INVALID_VALUE;
        CUfunction f = st == CUBLAS_STATUS_SUCCESS ? epilogueKernel() : nullptr;
        if (!f && st == CUBLAS_STATUS_SUCCESS) st = CUBLAS_STATUS_INTERNAL_ERROR;
        uint32_t typeD = 0, typeBias = 0;
        typeCode(Ddesc->type, typeD);
        typeCode(desc->biasType >= 0 ? cudaDataType(desc->biasType) : Ddesc->type, typeBias);
        EpilogueParams p{};
        p.d = reinterpret_cast<uint64_t>(D);
        p.bias = reinterpret_cast<uint64_t>(desc->bias);
        p.strideD = Ddesc->batchStride;
        p.strideBias = desc->biasBatchStride;
        p.rows = m, p.cols = n, p.ldd = int(Ddesc->ld);
        p.flags = flags | typeD | typeBias << 2;
        for (int first = 0; first < batch && st == CUBLAS_STATUS_SUCCESS; first += int(kMaxGridZ)) {
            p.batchOffset = first;
            size_t size = sizeof p;
            void* extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, &p, CU_LAUNCH_PARAM_BUFFER_SIZE, &size,
                             CU_LAUNCH_PARAM_END};
            const unsigned gz = unsigned(std::min<int>(batch - first, int(kMaxGridZ)));
            if (cuLaunchKernel(f, (unsigned(m) + 63) / 64, (unsigned(n) + 3) / 4, gz, 64, 4, 1, 0, stream, nullptr,
                               extra) != CUDA_SUCCESS)
                st = CUBLAS_STATUS_EXECUTION_FAILED;
        }
    }
    cublasSetPointerMode_v2(h, oldMode);
    cublasSetStream_v2(h, oldStream);
    return st;
}

}  // extern "C"
