// Vulkanizer's libcusparse: the sparse operations PyTorch uses (torch.sparse.mm and addmm, sparse
// CSR/COO products, sampled_addmm, sparse triangular solves, CSR addition, COO to CSR), f32 and
// f64 values with 32- or 64-bit indices, on Vulkanizer's own kernels (shaders/*.comp), queued on
// the handle's stream.
//
// Products work on CSR: a COO matrix (rows sorted, as PyTorch's coalesced tensors are) gets row
// offsets in the workspace, and op(A) = A^T is done on the CSR of the transpose, also built in the
// workspace. SpGEMM and the sorting routines run on the host (they synchronize the stream); block
// sparse (BSR) and complex values report CUSPARSE_STATUS_NOT_SUPPORTED (cusparse_stubs.cpp).

#include <cusparse/cusparse.h>
#include <cuda.h>
#include <vulkanizer/vulkanizer_ext.h>

#include "common/log.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <map>
#include <mutex>
#include <numeric>
#include <unordered_map>
#include <vector>

struct cusparseContext {
    CUstream stream = nullptr;
    cusparsePointerMode_t pointerMode = CUSPARSE_POINTER_MODE_HOST;
};

struct cusparseMatDescr {
    cusparseMatrixType_t type = CUSPARSE_MATRIX_TYPE_GENERAL;
    cusparseIndexBase_t base = CUSPARSE_INDEX_BASE_ZERO;
    cusparseFillMode_t fill = CUSPARSE_FILL_MODE_LOWER;
    cusparseDiagType_t diag = CUSPARSE_DIAG_TYPE_NON_UNIT;
};

struct cusparseSpMatDescr {
    cusparseFormat_t format = CUSPARSE_FORMAT_CSR;
    int64_t rows = 0, cols = 0, nnz = 0;
    CUdeviceptr offsets = 0;  // CSR row offsets, or COO row indices
    CUdeviceptr columns = 0, values = 0;
    cusparseIndexType_t offsetsType = CUSPARSE_INDEX_32I, columnsType = CUSPARSE_INDEX_32I;
    cusparseIndexBase_t base = CUSPARSE_INDEX_BASE_ZERO;
    cudaDataType valueType = CUDA_R_32F;
    int batch = 1;
    int64_t offsetsStride = 0, entriesStride = 0;
    cusparseFillMode_t fill = CUSPARSE_FILL_MODE_LOWER;
    cusparseDiagType_t diag = CUSPARSE_DIAG_TYPE_NON_UNIT;
};

struct cusparseDnMatDescr {
    int64_t rows = 0, cols = 0, ld = 0;
    CUdeviceptr values = 0;
    cudaDataType valueType = CUDA_R_32F;
    cusparseOrder_t order = CUSPARSE_ORDER_COL;
    int batch = 1;
    int64_t stride = 0;
};

struct cusparseDnVecDescr {
    int64_t size = 0;
    CUdeviceptr values = 0;
    cudaDataType valueType = CUDA_R_32F;
};

// SpGEMM's product, computed on the host by SpGEMM_compute and stored by SpGEMM_copy.
struct cusparseSpGEMMDescr {
    std::vector<int64_t> offsets, columns;
    std::vector<double> values;
};

// The solves keep the analysis' workspace (a COO matrix's row offsets live there).
struct cusparseSpSVDescr {
    CUdeviceptr workspace = 0;
};
struct cusparseSpSMDescr {
    CUdeviceptr workspace = 0;
};

namespace {

const uint32_t kSpmm32[] = {
#include "spmm.comp.inc"
};
const uint32_t kSpmm64[] = {
#include "spmm_f64.comp.inc"
};
const uint32_t kSddmm32[] = {
#include "sddmm.comp.inc"
};
const uint32_t kSddmm64[] = {
#include "sddmm_f64.comp.inc"
};
const uint32_t kSpsv32[] = {
#include "spsv.comp.inc"
};
const uint32_t kSpsv64[] = {
#include "spsv_f64.comp.inc"
};
const uint32_t kCsrUtil32[] = {
#include "csr_util.comp.inc"
};
const uint32_t kCsrUtil64[] = {
#include "csr_util_f64.comp.inc"
};
const uint32_t kGeam32[] = {
#include "geam.comp.inc"
};
const uint32_t kGeam64[] = {
#include "geam_f64.comp.inc"
};

// Push constants of every kernel (shaders/common.glsl, std430).
struct Params {
    uint64_t ptr[8];
    uint64_t alphaBits, betaBits;
    int32_t stride[4];
    int32_t items;
    int32_t m, n, k;
    int32_t ld0, ld1;
    int32_t extra;
    uint32_t flags;
};
static_assert(sizeof(Params) == 128, "must match shaders/common.glsl");

constexpr uint32_t kIndex64A = 1, kIndex64B = 2, kRowMajor0 = 4, kRowMajor1 = 8, kTrans0 = 16, kTrans1 = 32,
                   kBetaZero = 64, kBaseShift = 8, kModeShift = 16, kUpper = 1u << 20, kUnit = 1u << 21;
enum UtilMode : uint32_t { Iota, CooToCsr, Scan, Count, Transpose, Nnz };
enum GeamMode : uint32_t { GeamCount, GeamColumns, GeamValues };

enum Kernel { Spmm, Sddmm, Spsv, CsrUtil, Geam, KernelCount };

CUresult function(Kernel kernel, bool f64, CUfunction& out) {
    static std::mutex mutex;
    static std::map<CUcontext, std::array<std::array<CUfunction, 2>, KernelCount>> cache;
    CUcontext ctx = nullptr;
    if (CUresult r = cuCtxGetCurrent(&ctx)) return r;
    if (!ctx) return CUDA_ERROR_INVALID_CONTEXT;
    std::lock_guard<std::mutex> lock(mutex);
    CUfunction& f = cache[ctx][kernel][f64];
    if (!f) {
        struct Code {
            const uint32_t* words;
            size_t count;
        };
        static const Code sources[KernelCount][2] = {
            {{kSpmm32, std::size(kSpmm32)}, {kSpmm64, std::size(kSpmm64)}},
            {{kSddmm32, std::size(kSddmm32)}, {kSddmm64, std::size(kSddmm64)}},
            {{kSpsv32, std::size(kSpsv32)}, {kSpsv64, std::size(kSpsv64)}},
            {{kCsrUtil32, std::size(kCsrUtil32)}, {kCsrUtil64, std::size(kCsrUtil64)}},
            {{kGeam32, std::size(kGeam32)}, {kGeam64, std::size(kGeam64)}},
        };
        const Code& s = sources[kernel][f64];
        if (CUresult r = vulkanizerCreateSpirvFunction(&f, s.words, s.count, "main", sizeof(Params))) return r;
    }
    out = f;
    return CUDA_SUCCESS;
}

// `groups` workgroups of 256 threads on a 2D grid (by default enough for p.items work items).
CUresult launch(CUstream stream, Kernel kernel, bool f64, Params& p, int64_t groups = -1) {
    if (groups < 0) groups = (int64_t(p.items) + 255) / 256;
    if (groups <= 0) return CUDA_SUCCESS;
    CUfunction f;
    if (CUresult r = function(kernel, f64, f)) return r;
    const unsigned gx = unsigned(std::min<int64_t>(groups, 65535)), gy = unsigned((groups + gx - 1) / gx);
    size_t size = sizeof p;
    void* extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, &p, CU_LAUNCH_PARAM_BUFFER_SIZE, &size, CU_LAUNCH_PARAM_END};
    return cuLaunchKernel(f, gx, gy, 1, 256, 1, 1, 0, stream, nullptr, extra);
}

cusparseStatus_t status(CUresult r) {
    if (r == CUDA_SUCCESS) return CUSPARSE_STATUS_SUCCESS;
    if (r == CUDA_ERROR_NOT_SUPPORTED) return CUSPARSE_STATUS_NOT_SUPPORTED;
    VULKANIZER_WARN("cuSPARSE: kernel launch failed (%d)", int(r));
    return CUSPARSE_STATUS_EXECUTION_FAILED;
}

// Value types: f32 or f64 (f64 where the device has it).
bool realType(cudaDataType t, bool& f64) {
    if (t == CUDA_R_32F) return f64 = false, true;
    if (t == CUDA_R_64F) {
        int v = 0;
        f64 = true;
        return vulkanizerGetFeature(VULKANIZER_FEATURE_FLOAT64, &v) == CUDA_SUCCESS && v;
    }
    return false;
}

bool fits(int64_t v) { return v >= 0 && v < INT32_MAX; }

// A scalar of the value type, from host or device memory as the pointer mode says, as its bits.
bool scalarBits(cusparseHandle_t h, const void* p, bool f64, uint64_t& bits, double* value = nullptr) {
    if (!p) return false;
    unsigned char buf[8] = {};
    const size_t size = f64 ? 8 : 4;
    if (h->pointerMode == CUSPARSE_POINTER_MODE_DEVICE) {
        if (cuStreamSynchronize(h->stream) != CUDA_SUCCESS) return false;
        if (cuMemcpyDtoH(buf, CUdeviceptr(p), size) != CUDA_SUCCESS) return false;
    } else {
        std::memcpy(buf, p, size);
    }
    bits = 0;
    std::memcpy(&bits, buf, size);
    if (value) {
        if (f64) std::memcpy(value, buf, 8);
        else {
            float f;
            std::memcpy(&f, buf, 4);
            *value = f;
        }
    }
    return true;
}

uint32_t baseFlag(cusparseIndexBase_t base, int which) { return base == CUSPARSE_INDEX_BASE_ONE ? 1u << (kBaseShift + which) : 0; }

// A CSR matrix the kernels can use directly.
struct Csr {
    CUdeviceptr offsets = 0, columns = 0, values = 0;
    bool wideOffsets = false, wideColumns = false;
    cusparseIndexBase_t base = CUSPARSE_INDEX_BASE_ZERO;
    int rows = 0, cols = 0, nnz = 0;
    int batch = 1;
    int offsetsStride = 0, entriesStride = 0;
};

size_t align(size_t v) { return (v + 15) & ~size_t(15); }

// Workspace for csrOf: a COO matrix's row offsets; the transpose's offsets, cursors, rows and values.
size_t csrWorkspace(const cusparseSpMatDescr& a, bool transpose, bool f64) {
    size_t bytes = 0;
    if (a.format == CUSPARSE_FORMAT_COO) bytes += align(size_t(a.rows + 1) * 4);
    if (transpose)
        bytes += align(size_t(a.nnz) * (f64 ? 8 : 4)) + 2 * align(size_t(a.cols + 1) * 4) + align(size_t(a.nnz) * 4);
    return bytes;
}

// A (or A^T) as CSR, built in the workspace where needed; queued on the stream.
cusparseStatus_t csrOf(cusparseHandle_t h, const cusparseSpMatDescr& a, bool transpose, bool f64,
                       CUdeviceptr workspace, Csr& out) {
    if (a.format != CUSPARSE_FORMAT_CSR && a.format != CUSPARSE_FORMAT_COO) return CUSPARSE_STATUS_NOT_SUPPORTED;
    if (!fits(a.rows) || !fits(a.cols) || !fits(a.nnz) || !fits(a.offsetsStride) || !fits(a.entriesStride))
        return CUSPARSE_STATUS_NOT_SUPPORTED;
    if (csrWorkspace(a, transpose, f64) && !workspace) return CUSPARSE_STATUS_INVALID_VALUE;
    out = Csr{};
    out.offsets = a.offsets;
    out.columns = a.columns;
    out.values = a.values;
    out.wideOffsets = a.offsetsType == CUSPARSE_INDEX_64I;
    out.wideColumns = a.columnsType == CUSPARSE_INDEX_64I;
    out.base = a.base;
    out.rows = int(a.rows);
    out.cols = int(a.cols);
    out.nnz = int(a.nnz);
    out.batch = a.batch;
    out.offsetsStride = int(a.offsetsStride);
    out.entriesStride = int(a.entriesStride);
    CUdeviceptr next = workspace;
    if (a.format == CUSPARSE_FORMAT_COO) {
        if (a.batch > 1) return CUSPARSE_STATUS_NOT_SUPPORTED;
        Params p{};
        p.ptr[0] = a.offsets;  // the row indices
        p.ptr[1] = next;
        p.items = out.rows + 1;
        p.k = out.nnz;
        p.extra = a.base == CUSPARSE_INDEX_BASE_ONE;  // the offsets in the same base as the columns
        p.flags = CooToCsr << kModeShift | (out.wideOffsets ? kIndex64A : 0) | baseFlag(a.base, 0);
        if (CUresult r = launch(h->stream, CsrUtil, f64, p)) return status(r);
        out.offsets = next;
        out.wideOffsets = false;
        out.offsetsStride = 0;
        next += align(size_t(a.rows + 1) * 4);
    }
    if (!transpose) return CUSPARSE_STATUS_SUCCESS;
    if (a.batch > 1) return CUSPARSE_STATUS_NOT_SUPPORTED;
    // counts of each column (offset by one), their prefix sums, cursors, the scatter
    const CUdeviceptr values = next;
    next += align(size_t(a.nnz) * (f64 ? 8 : 4));
    const CUdeviceptr offsets = next;
    next += align(size_t(a.cols + 1) * 4);
    const CUdeviceptr cursors = next;
    next += align(size_t(a.cols + 1) * 4);
    const CUdeviceptr rows = next;
    if (CUresult r = cuMemsetD32Async(offsets, 0, size_t(a.cols + 1), h->stream)) return status(r);
    Params p{};
    p.ptr[0] = out.columns;
    p.ptr[1] = offsets;
    p.items = out.nnz;
    p.flags = Count << kModeShift | (out.wideColumns ? kIndex64B : 0) | baseFlag(out.base, 0);
    if (CUresult r = launch(h->stream, CsrUtil, f64, p)) return status(r);
    p = Params{};
    p.ptr[0] = offsets;
    p.items = out.cols + 1;
    p.flags = Scan << kModeShift;
    if (CUresult r = launch(h->stream, CsrUtil, f64, p, 1)) return status(r);
    if (CUresult r = cuMemcpyDtoDAsync(cursors, offsets, size_t(a.cols) * 4, h->stream)) return status(r);
    p = Params{};
    p.ptr[0] = out.offsets;
    p.ptr[1] = out.columns;
    p.ptr[2] = out.values;
    p.ptr[3] = cursors;
    p.ptr[4] = rows;
    p.ptr[5] = values;
    p.items = out.rows;
    p.flags = Transpose << kModeShift | (out.wideOffsets ? kIndex64A : 0) | (out.wideColumns ? kIndex64B : 0) |
              baseFlag(out.base, 0);
    if (CUresult r = launch(h->stream, CsrUtil, f64, p)) return status(r);
    out.offsets = offsets;
    out.columns = rows;
    out.values = values;
    out.wideOffsets = out.wideColumns = false;
    out.base = CUSPARSE_INDEX_BASE_ZERO;
    std::swap(out.rows, out.cols);
    return CUSPARSE_STATUS_SUCCESS;
}

// A dense operand: rows x cols, its leading dimension, order and batch stride.
struct Dense {
    CUdeviceptr values = 0;
    int rows = 0, cols = 0, ld = 0;
    bool rowMajor = false;
    int batch = 1, stride = 0;
};

bool denseOf(const cusparseDnMatDescr& d, Dense& out) {
    if (!fits(d.rows) || !fits(d.cols) || !fits(d.ld) || !fits(d.stride)) return false;
    out = Dense{d.values, int(d.rows), int(d.cols), int(d.ld), d.order == CUSPARSE_ORDER_ROW, d.batch, int(d.stride)};
    return true;
}

Dense denseOf(const cusparseDnVecDescr& v) { return Dense{v.values, int(v.size), 1, int(std::max<int64_t>(v.size, 1)), false, 1, 0}; }

// C = alpha op(A) op(B) + beta C; SpMV is the case of one column.
cusparseStatus_t spmm(cusparseHandle_t h, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha,
                      const cusparseSpMatDescr* matA, const Dense& b, const void* beta, const Dense& c,
                      cudaDataType computeType, void* externalBuffer, size_t* bufferSize, cudaDataType bType,
                      cudaDataType cType) {
    if (!h) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (!matA) return CUSPARSE_STATUS_INVALID_VALUE;
    bool f64;
    if (!realType(computeType, f64) || matA->valueType != computeType || bType != computeType || cType != computeType)
        return CUSPARSE_STATUS_NOT_SUPPORTED;
    const bool transA = opA != CUSPARSE_OPERATION_NON_TRANSPOSE, transB = opB != CUSPARSE_OPERATION_NON_TRANSPOSE;
    if (bufferSize) {
        *bufferSize = csrWorkspace(*matA, transA, f64);
        return CUSPARSE_STATUS_SUCCESS;
    }
    const int64_t m = transA ? matA->cols : matA->rows, k = transA ? matA->rows : matA->cols;
    if (c.rows != m || (transB ? b.cols : b.rows) != k || (transB ? b.rows : b.cols) != c.cols)
        return CUSPARSE_STATUS_INVALID_VALUE;
    Params p{};
    if (!scalarBits(h, alpha, f64, p.alphaBits)) return CUSPARSE_STATUS_INVALID_VALUE;
    double betaValue = 0;
    if (!scalarBits(h, beta, f64, p.betaBits, &betaValue)) return CUSPARSE_STATUS_INVALID_VALUE;
    if (c.rows == 0 || c.cols == 0) return CUSPARSE_STATUS_SUCCESS;
    Csr a;
    if (cusparseStatus_t s = csrOf(h, *matA, transA, f64, CUdeviceptr(externalBuffer), a)) return s;
    const int batch = std::max({c.batch, b.batch, a.batch});
    if ((c.batch != batch) || (b.batch != 1 && b.batch != batch) || (a.batch != 1 && a.batch != batch))
        return CUSPARSE_STATUS_INVALID_VALUE;
    if (int64_t(batch) * c.rows * c.cols >= INT32_MAX) return CUSPARSE_STATUS_NOT_SUPPORTED;
    p.ptr[0] = a.offsets;
    p.ptr[1] = a.columns;
    p.ptr[2] = a.values;
    p.ptr[3] = b.values;
    p.ptr[4] = c.values;
    p.stride[0] = a.batch > 1 ? a.offsetsStride : 0;
    p.stride[1] = a.batch > 1 ? a.entriesStride : 0;
    p.stride[2] = b.batch > 1 ? b.stride : 0;
    p.stride[3] = c.stride;
    p.items = batch * c.rows * c.cols;
    p.m = c.rows;
    p.n = c.cols;
    p.k = int(k);
    p.ld0 = b.ld;
    p.ld1 = c.ld;
    p.flags = (a.wideOffsets ? kIndex64A : 0) | (a.wideColumns ? kIndex64B : 0) | (b.rowMajor ? kRowMajor0 : 0) |
              (c.rowMajor ? kRowMajor1 : 0) | (transB ? kTrans1 : 0) | (betaValue == 0 ? kBetaZero : 0) |
              baseFlag(a.base, 0);
    return status(launch(h->stream, Spmm, f64, p));
}

// op(A) Y = alpha X (SpSV, SpSM).
cusparseStatus_t spsv(cusparseHandle_t h, cusparseOperation_t opA, const void* alpha, const cusparseSpMatDescr* matA,
                      const Dense& x, const Dense& y, cudaDataType computeType, cudaDataType xType,
                      cudaDataType yType, CUdeviceptr workspace) {
    if (!h) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (!matA) return CUSPARSE_STATUS_INVALID_VALUE;
    bool f64;
    if (!realType(computeType, f64) || matA->valueType != computeType || xType != computeType || yType != computeType)
        return CUSPARSE_STATUS_NOT_SUPPORTED;
    if (matA->rows != matA->cols || x.rows != matA->rows || y.rows != matA->rows || x.cols != y.cols)
        return CUSPARSE_STATUS_INVALID_VALUE;
    Params p{};
    if (!scalarBits(h, alpha, f64, p.alphaBits)) return CUSPARSE_STATUS_INVALID_VALUE;
    Csr a;
    if (cusparseStatus_t s = csrOf(h, *matA, false, f64, workspace, a)) return s;
    if (a.batch > 1) return CUSPARSE_STATUS_NOT_SUPPORTED;
    p.ptr[0] = a.offsets;
    p.ptr[1] = a.columns;
    p.ptr[2] = a.values;
    p.ptr[3] = x.values;
    p.ptr[4] = y.values;
    p.n = a.rows;
    p.k = y.cols;
    p.ld0 = x.ld;
    p.ld1 = y.ld;
    p.flags = (a.wideOffsets ? kIndex64A : 0) | (a.wideColumns ? kIndex64B : 0) | (x.rowMajor ? kRowMajor0 : 0) |
              (y.rowMajor ? kRowMajor1 : 0) | (opA != CUSPARSE_OPERATION_NON_TRANSPOSE ? kTrans0 : 0) |
              (matA->fill == CUSPARSE_FILL_MODE_UPPER ? kUpper : 0) |
              (matA->diag == CUSPARSE_DIAG_TYPE_UNIT ? kUnit : 0) | baseFlag(a.base, 0);
    return status(launch(h->stream, Spsv, f64, p, y.cols));
}

// Host copies of index arrays (32 or 64 bits) and values (f32 or f64), widened.
std::vector<int64_t> downloadIndices(CUdeviceptr p, int64_t count, bool wide) {
    std::vector<int64_t> v(size_t(std::max<int64_t>(count, 0)));
    if (count <= 0) return v;
    if (wide) {
        cuMemcpyDtoH(v.data(), p, size_t(count) * 8);
    } else {
        std::vector<int32_t> n(static_cast<size_t>(count));
        cuMemcpyDtoH(n.data(), p, size_t(count) * 4);
        std::copy(n.begin(), n.end(), v.begin());
    }
    return v;
}
std::vector<double> downloadValues(CUdeviceptr p, int64_t count, bool f64) {
    std::vector<double> v(size_t(std::max<int64_t>(count, 0)));
    if (count <= 0) return v;
    if (f64) {
        cuMemcpyDtoH(v.data(), p, size_t(count) * 8);
    } else {
        std::vector<float> f(static_cast<size_t>(count));
        cuMemcpyDtoH(f.data(), p, size_t(count) * 4);
        std::copy(f.begin(), f.end(), v.begin());
    }
    return v;
}
CUresult uploadIndices(CUdeviceptr p, const std::vector<int64_t>& v, bool wide, int64_t add = 0) {
    if (v.empty()) return CUDA_SUCCESS;
    if (wide) {
        std::vector<int64_t> w(v);
        for (int64_t& x : w) x += add;
        return cuMemcpyHtoD(p, w.data(), w.size() * 8);
    }
    std::vector<int32_t> n(v.size());
    for (size_t i = 0; i < v.size(); ++i) n[i] = int32_t(v[i] + add);
    return cuMemcpyHtoD(p, n.data(), n.size() * 4);
}
CUresult uploadValues(CUdeviceptr p, const std::vector<double>& v, bool f64) {
    if (v.empty()) return CUDA_SUCCESS;
    if (f64) return cuMemcpyHtoD(p, v.data(), v.size() * 8);
    std::vector<float> f(v.begin(), v.end());
    return cuMemcpyHtoD(p, f.data(), f.size() * 4);
}

const char* const kStatusNames[] = {
    "CUSPARSE_STATUS_SUCCESS",           "CUSPARSE_STATUS_NOT_INITIALIZED", "CUSPARSE_STATUS_ALLOC_FAILED",
    "CUSPARSE_STATUS_INVALID_VALUE",     "CUSPARSE_STATUS_ARCH_MISMATCH",   "CUSPARSE_STATUS_MAPPING_ERROR",
    "CUSPARSE_STATUS_EXECUTION_FAILED",  "CUSPARSE_STATUS_INTERNAL_ERROR",  "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED",
    "CUSPARSE_STATUS_ZERO_PIVOT",        "CUSPARSE_STATUS_NOT_SUPPORTED",   "CUSPARSE_STATUS_INSUFFICIENT_RESOURCES",
};
const char* const kStatusStrings[] = {
    "success", "initialization error", "out of memory", "invalid value", "architecture mismatch", "mapping error",
    "execution failed", "internal error", "matrix type not supported", "zero pivot", "operation not supported",
    "insufficient resources",
};

}  // namespace

extern "C" {

// ---- handle -----------------------------------------------------------------------------------

cusparseStatus_t cusparseCreate(cusparseHandle_t* handle) {
    VULKANIZER_TRACE_API();
    if (!handle) return CUSPARSE_STATUS_INVALID_VALUE;
    if (cuInit(0) != CUDA_SUCCESS) return CUSPARSE_STATUS_NOT_INITIALIZED;
    *handle = new cusparseContext();
    return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseDestroy(cusparseHandle_t handle) {
    VULKANIZER_TRACE_API();
    if (!handle) return CUSPARSE_STATUS_NOT_INITIALIZED;
    delete handle;
    return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseSetStream(cusparseHandle_t handle, cudaStream_t streamId) {
    if (!handle) return CUSPARSE_STATUS_NOT_INITIALIZED;
    handle->stream = reinterpret_cast<CUstream>(streamId);
    return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseGetStream(cusparseHandle_t handle, cudaStream_t* streamId) {
    if (!handle) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (!streamId) return CUSPARSE_STATUS_INVALID_VALUE;
    *streamId = reinterpret_cast<cudaStream_t>(handle->stream);
    return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseSetPointerMode(cusparseHandle_t handle, cusparsePointerMode_t mode) {
    if (!handle) return CUSPARSE_STATUS_NOT_INITIALIZED;
    handle->pointerMode = mode;
    return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseGetPointerMode(cusparseHandle_t handle, cusparsePointerMode_t* mode) {
    if (!handle) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (!mode) return CUSPARSE_STATUS_INVALID_VALUE;
    *mode = handle->pointerMode;
    return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseGetVersion(cusparseHandle_t handle, int* version) {
    if (!handle) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (!version) return CUSPARSE_STATUS_INVALID_VALUE;
    *version = VULKANIZER_CUSPARSE_VERSION;
    return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseGetProperty(libraryPropertyType type, int* value) {
    if (!value) return CUSPARSE_STATUS_INVALID_VALUE;
    const int v = VULKANIZER_CUSPARSE_VERSION;
    switch (type) {
    case MAJOR_VERSION: *value = v / 1000; return CUSPARSE_STATUS_SUCCESS;
    case MINOR_VERSION: *value = v % 1000 / 100; return CUSPARSE_STATUS_SUCCESS;
    case PATCH_LEVEL: *value = v % 100; return CUSPARSE_STATUS_SUCCESS;
    default: return CUSPARSE_STATUS_INVALID_VALUE;
    }
}
const char* cusparseGetErrorName(cusparseStatus_t status) {
    const unsigned i = unsigned(status);
    return i < std::size(kStatusNames) ? kStatusNames[i] : "CUSPARSE_STATUS_UNKNOWN";
}
const char* cusparseGetErrorString(cusparseStatus_t status) {
    const unsigned i = unsigned(status);
    return i < std::size(kStatusStrings) ? kStatusStrings[i] : "unknown error";
}

// ---- legacy matrix descriptors ----------------------------------------------------------------

cusparseStatus_t cusparseCreateMatDescr(cusparseMatDescr_t* descrA) {
    if (!descrA) return CUSPARSE_STATUS_INVALID_VALUE;
    *descrA = new cusparseMatDescr();
    return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseDestroyMatDescr(cusparseMatDescr_t descrA) {
    delete descrA;
    return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseSetMatType(cusparseMatDescr_t descrA, cusparseMatrixType_t type) {
    if (!descrA) return CUSPARSE_STATUS_INVALID_VALUE;
    descrA->type = type;
    return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseSetMatIndexBase(cusparseMatDescr_t descrA, cusparseIndexBase_t base) {
    if (!descrA || (base != CUSPARSE_INDEX_BASE_ZERO && base != CUSPARSE_INDEX_BASE_ONE))
        return CUSPARSE_STATUS_INVALID_VALUE;
    descrA->base = base;
    return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseSetMatFillMode(cusparseMatDescr_t descrA, cusparseFillMode_t fillMode) {
    if (!descrA) return CUSPARSE_STATUS_INVALID_VALUE;
    descrA->fill = fillMode;
    return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseSetMatDiagType(cusparseMatDescr_t descrA, cusparseDiagType_t diagType) {
    if (!descrA) return CUSPARSE_STATUS_INVALID_VALUE;
    descrA->diag = diagType;
    return CUSPARSE_STATUS_SUCCESS;
}
cusparseMatrixType_t cusparseGetMatType(const cusparseMatDescr_t descrA) {
    return descrA ? descrA->type : CUSPARSE_MATRIX_TYPE_GENERAL;
}
cusparseIndexBase_t cusparseGetMatIndexBase(const cusparseMatDescr_t descrA) {
    return descrA ? descrA->base : CUSPARSE_INDEX_BASE_ZERO;
}

// ---- generic API descriptors ------------------------------------------------------------------

cusparseStatus_t cusparseCreateCsr(cusparseSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz,
                                   void* csrRowOffsets, void* csrColInd, void* csrValues,
                                   cusparseIndexType_t csrRowOffsetsType, cusparseIndexType_t csrColIndType,
                                   cusparseIndexBase_t idxBase, cudaDataType valueType) {
    if (!spMatDescr || rows < 0 || cols < 0 || nnz < 0) return CUSPARSE_STATUS_INVALID_VALUE;
    if (csrRowOffsetsType == CUSPARSE_INDEX_16U || csrColIndType == CUSPARSE_INDEX_16U)
        return CUSPARSE_STATUS_NOT_SUPPORTED;
    auto* d = new cusparseSpMatDescr();
    d->format = CUSPARSE_FORMAT_CSR;
    d->rows = rows, d->cols = cols, d->nnz = nnz;
    d->offsets = CUdeviceptr(csrRowOffsets), d->columns = CUdeviceptr(csrColInd), d->values = CUdeviceptr(csrValues);
    d->offsetsType = csrRowOffsetsType, d->columnsType = csrColIndType;
    d->base = idxBase;
    d->valueType = valueType;
    *spMatDescr = d;
    return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseCreateCoo(cusparseSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz,
                                   void* cooRowInd, void* cooColInd, void* cooValues, cusparseIndexType_t cooIdxType,
                                   cusparseIndexBase_t idxBase, cudaDataType valueType) {
    if (cusparseStatus_t s = cusparseCreateCsr(spMatDescr, rows, cols, nnz, cooRowInd, cooColInd, cooValues,
                                               cooIdxType, cooIdxType, idxBase, valueType))
        return s;
    (*spMatDescr)->format = CUSPARSE_FORMAT_COO;
    return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseDestroySpMat(cusparseConstSpMatDescr_t spMatDescr) {
    delete spMatDescr;
    return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseCsrSetPointers(cusparseSpMatDescr_t spMatDescr, void* csrRowOffsets, void* csrColInd,
                                        void* csrValues) {
    if (!spMatDescr) return CUSPARSE_STATUS_INVALID_VALUE;
    spMatDescr->offsets = CUdeviceptr(csrRowOffsets);
    spMatDescr->columns = CUdeviceptr(csrColInd);
    spMatDescr->values = CUdeviceptr(csrValues);
    return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseCsrSetStridedBatch(cusparseSpMatDescr_t spMatDescr, int batchCount,
                                            int64_t offsetsBatchStride, int64_t columnsValuesBatchStride) {
    if (!spMatDescr || batchCount < 1) return CUSPARSE_STATUS_INVALID_VALUE;
    spMatDescr->batch = batchCount;
    spMatDescr->offsetsStride = offsetsBatchStride;
    spMatDescr->entriesStride = columnsValuesBatchStride;
    return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseCooSetStridedBatch(cusparseSpMatDescr_t spMatDescr, int batchCount, int64_t batchStride) {
    return cusparseCsrSetStridedBatch(spMatDescr, batchCount, batchStride, batchStride);
}
cusparseStatus_t cusparseSpMatGetSize(cusparseConstSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols,
                                      int64_t* nnz) {
    if (!spMatDescr) return CUSPARSE_STATUS_INVALID_VALUE;
    if (rows) *rows = spMatDescr->rows;
    if (cols) *cols = spMatDescr->cols;
    if (nnz) *nnz = spMatDescr->nnz;
    return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseSpMatSetAttribute(cusparseSpMatDescr_t spMatDescr, cusparseSpMatAttribute_t attribute,
                                           void* data, size_t dataSize) {
    if (!spMatDescr || !data || dataSize < sizeof(int)) return CUSPARSE_STATUS_INVALID_VALUE;
    int v;
    std::memcpy(&v, data, sizeof v);
    if (attribute == CUSPARSE_SPMAT_FILL_MODE) spMatDescr->fill = cusparseFillMode_t(v);
    else if (attribute == CUSPARSE_SPMAT_DIAG_TYPE) spMatDescr->diag = cusparseDiagType_t(v);
    else return CUSPARSE_STATUS_INVALID_VALUE;
    return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseCreateDnMat(cusparseDnMatDescr_t* dnMatDescr, int64_t rows, int64_t cols, int64_t ld,
                                     void* values, cudaDataType valueType, cusparseOrder_t order) {
    if (!dnMatDescr || rows < 0 || cols < 0) return CUSPARSE_STATUS_INVALID_VALUE;
    if (ld < (order == CUSPARSE_ORDER_ROW ? cols : rows)) return CUSPARSE_STATUS_INVALID_VALUE;
    auto* d = new cusparseDnMatDescr();
    d->rows = rows, d->cols = cols, d->ld = ld;
    d->values = CUdeviceptr(values);
    d->valueType = valueType;
    d->order = order;
    *dnMatDescr = d;
    return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseDestroyDnMat(cusparseConstDnMatDescr_t dnMatDescr) {
    delete dnMatDescr;
    return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseDnMatSetStridedBatch(cusparseDnMatDescr_t dnMatDescr, int batchCount, int64_t batchStride) {
    if (!dnMatDescr || batchCount < 1) return CUSPARSE_STATUS_INVALID_VALUE;
    dnMatDescr->batch = batchCount;
    dnMatDescr->stride = batchStride;
    return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseCreateDnVec(cusparseDnVecDescr_t* dnVecDescr, int64_t size, void* values,
                                     cudaDataType valueType) {
    if (!dnVecDescr || size < 0) return CUSPARSE_STATUS_INVALID_VALUE;
    *dnVecDescr = new cusparseDnVecDescr{size, CUdeviceptr(values), valueType};
    return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseDestroyDnVec(cusparseConstDnVecDescr_t dnVecDescr) {
    delete dnVecDescr;
    return CUSPARSE_STATUS_SUCCESS;
}

// ---- SpMV, SpMM -------------------------------------------------------------------------------

cusparseStatus_t cusparseSpMV_bufferSize(cusparseHandle_t handle, cusparseOperation_t opA, const void* alpha,
                                         cusparseConstSpMatDescr_t matA, cusparseConstDnVecDescr_t vecX,
                                         const void* beta, cusparseDnVecDescr_t vecY, cudaDataType computeType,
                                         cusparseSpMVAlg_t, size_t* bufferSize) {
    if (!vecX || !vecY || !bufferSize) return CUSPARSE_STATUS_INVALID_VALUE;
    return spmm(handle, opA, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha, matA, denseOf(*vecX), beta, denseOf(*vecY),
                computeType, nullptr, bufferSize, vecX->valueType, vecY->valueType);
}
cusparseStatus_t cusparseSpMV(cusparseHandle_t handle, cusparseOperation_t opA, const void* alpha,
                              cusparseConstSpMatDescr_t matA, cusparseConstDnVecDescr_t vecX, const void* beta,
                              cusparseDnVecDescr_t vecY, cudaDataType computeType, cusparseSpMVAlg_t,
                              void* externalBuffer) {
    VULKANIZER_TRACE_API();
    if (!vecX || !vecY) return CUSPARSE_STATUS_INVALID_VALUE;
    return spmm(handle, opA, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha, matA, denseOf(*vecX), beta, denseOf(*vecY),
                computeType, externalBuffer, nullptr, vecX->valueType, vecY->valueType);
}
cusparseStatus_t cusparseSpMM_bufferSize(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB,
                                         const void* alpha, cusparseConstSpMatDescr_t matA,
                                         cusparseConstDnMatDescr_t matB, const void* beta, cusparseDnMatDescr_t matC,
                                         cudaDataType computeType, cusparseSpMMAlg_t, size_t* bufferSize) {
    Dense b, c;
    if (!matB || !matC || !bufferSize || !denseOf(*matB, b) || !denseOf(*matC, c)) return CUSPARSE_STATUS_INVALID_VALUE;
    return spmm(handle, opA, opB, alpha, matA, b, beta, c, computeType, nullptr, bufferSize, matB->valueType,
                matC->valueType);
}
cusparseStatus_t cusparseSpMM(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB,
                              const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstDnMatDescr_t matB,
                              const void* beta, cusparseDnMatDescr_t matC, cudaDataType computeType,
                              cusparseSpMMAlg_t, void* externalBuffer) {
    VULKANIZER_TRACE_API();
    Dense b, c;
    if (!matB || !matC || !denseOf(*matB, b) || !denseOf(*matC, c)) return CUSPARSE_STATUS_INVALID_VALUE;
    return spmm(handle, opA, opB, alpha, matA, b, beta, c, computeType, externalBuffer, nullptr, matB->valueType,
                matC->valueType);
}

// ---- SDDMM ------------------------------------------------------------------------------------

cusparseStatus_t cusparseSDDMM_bufferSize(cusparseHandle_t handle, cusparseOperation_t, cusparseOperation_t,
                                          const void*, cusparseConstDnMatDescr_t, cusparseConstDnMatDescr_t,
                                          const void*, cusparseSpMatDescr_t, cudaDataType, cusparseSDDMMAlg_t,
                                          size_t* bufferSize) {
    if (!handle) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (!bufferSize) return CUSPARSE_STATUS_INVALID_VALUE;
    *bufferSize = 0;
    return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseSDDMM_preprocess(cusparseHandle_t handle, cusparseOperation_t, cusparseOperation_t,
                                          const void*, cusparseConstDnMatDescr_t, cusparseConstDnMatDescr_t,
                                          const void*, cusparseSpMatDescr_t, cudaDataType, cusparseSDDMMAlg_t, void*) {
    return handle ? CUSPARSE_STATUS_SUCCESS : CUSPARSE_STATUS_NOT_INITIALIZED;
}
cusparseStatus_t cusparseSDDMM(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB,
                               const void* alpha, cusparseConstDnMatDescr_t matA, cusparseConstDnMatDescr_t matB,
                               const void* beta, cusparseSpMatDescr_t matC, cudaDataType computeType,
                               cusparseSDDMMAlg_t, void*) {
    VULKANIZER_TRACE_API();
    if (!handle) return CUSPARSE_STATUS_NOT_INITIALIZED;
    Dense a, b;
    if (!matA || !matB || !matC || !denseOf(*matA, a) || !denseOf(*matB, b)) return CUSPARSE_STATUS_INVALID_VALUE;
    bool f64;
    if (!realType(computeType, f64) || matA->valueType != computeType || matB->valueType != computeType ||
        matC->valueType != computeType || matC->format != CUSPARSE_FORMAT_CSR)
        return CUSPARSE_STATUS_NOT_SUPPORTED;
    const bool transA = opA != CUSPARSE_OPERATION_NON_TRANSPOSE, transB = opB != CUSPARSE_OPERATION_NON_TRANSPOSE;
    const int m = transA ? a.cols : a.rows, k = transA ? a.rows : a.cols;
    if (m != matC->rows || (transB ? b.cols : b.rows) != k || (transB ? b.rows : b.cols) != matC->cols)
        return CUSPARSE_STATUS_INVALID_VALUE;
    Params p{};
    double betaValue = 0;
    if (!scalarBits(handle, alpha, f64, p.alphaBits) || !scalarBits(handle, beta, f64, p.betaBits, &betaValue))
        return CUSPARSE_STATUS_INVALID_VALUE;
    Csr c;
    if (cusparseStatus_t s = csrOf(handle, *matC, false, f64, 0, c)) return s;
    const int batch = c.batch;
    if ((a.batch != 1 && a.batch != batch) || (b.batch != 1 && b.batch != batch)) return CUSPARSE_STATUS_INVALID_VALUE;
    if (int64_t(batch) * c.nnz >= INT32_MAX) return CUSPARSE_STATUS_NOT_SUPPORTED;
    if (c.nnz == 0) return CUSPARSE_STATUS_SUCCESS;
    p.ptr[0] = c.offsets;
    p.ptr[1] = c.columns;
    p.ptr[2] = c.values;
    p.ptr[3] = a.values;
    p.ptr[4] = b.values;
    p.stride[0] = batch > 1 ? c.offsetsStride : 0;
    p.stride[1] = batch > 1 ? c.entriesStride : 0;
    p.stride[2] = a.batch > 1 ? a.stride : 0;
    p.stride[3] = b.batch > 1 ? b.stride : 0;
    p.items = batch * c.nnz;
    p.m = c.rows;
    p.k = k;
    p.ld0 = a.ld;
    p.ld1 = b.ld;
    p.extra = c.nnz;
    p.flags = (c.wideOffsets ? kIndex64A : 0) | (c.wideColumns ? kIndex64B : 0) | (a.rowMajor ? kRowMajor0 : 0) |
              (b.rowMajor ? kRowMajor1 : 0) | (transA ? kTrans0 : 0) | (transB ? kTrans1 : 0) |
              (betaValue == 0 ? kBetaZero : 0) | baseFlag(c.base, 0);
    return status(launch(handle->stream, Sddmm, f64, p));
}

// ---- SpGEMM (on the host) ---------------------------------------------------------------------

cusparseStatus_t cusparseSpGEMM_createDescr(cusparseSpGEMMDescr_t* descr) {
    if (!descr) return CUSPARSE_STATUS_INVALID_VALUE;
    *descr = new cusparseSpGEMMDescr();
    return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseSpGEMM_destroyDescr(cusparseSpGEMMDescr_t descr) {
    delete descr;
    return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseSpGEMM_workEstimation(cusparseHandle_t handle, cusparseOperation_t, cusparseOperation_t,
                                               const void*, cusparseConstSpMatDescr_t, cusparseConstSpMatDescr_t,
                                               const void*, cusparseSpMatDescr_t, cudaDataType, cusparseSpGEMMAlg_t,
                                               cusparseSpGEMMDescr_t, size_t* bufferSize1, void*) {
    if (!handle) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (!bufferSize1) return CUSPARSE_STATUS_INVALID_VALUE;
    *bufferSize1 = 16;  // not used; nonzero, so that callers pass a buffer
    return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseSpGEMM_compute(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB,
                                        const void* alpha, cusparseConstSpMatDescr_t matA,
                                        cusparseConstSpMatDescr_t matB, const void*, cusparseSpMatDescr_t matC,
                                        cudaDataType computeType, cusparseSpGEMMAlg_t,
                                        cusparseSpGEMMDescr_t spgemmDescr, size_t* bufferSize2,
                                        void* externalBuffer2) {
    VULKANIZER_TRACE_API();
    if (!handle) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (!matA || !matB || !matC || !spgemmDescr || !bufferSize2) return CUSPARSE_STATUS_INVALID_VALUE;
    if (!externalBuffer2) {
        *bufferSize2 = 16;
        return CUSPARSE_STATUS_SUCCESS;
    }
    bool f64;
    if (!realType(computeType, f64) || matA->valueType != computeType || matB->valueType != computeType)
        return CUSPARSE_STATUS_NOT_SUPPORTED;
    if (opA != CUSPARSE_OPERATION_NON_TRANSPOSE || opB != CUSPARSE_OPERATION_NON_TRANSPOSE ||
        matA->format != CUSPARSE_FORMAT_CSR || matB->format != CUSPARSE_FORMAT_CSR || matA->batch > 1 ||
        matB->batch > 1)
        return CUSPARSE_STATUS_NOT_SUPPORTED;
    if (matA->cols != matB->rows) return CUSPARSE_STATUS_INVALID_VALUE;
    uint64_t bits;
    double alphaValue;
    if (!scalarBits(handle, alpha, f64, bits, &alphaValue)) return CUSPARSE_STATUS_INVALID_VALUE;
    if (cuStreamSynchronize(handle->stream) != CUDA_SUCCESS) return CUSPARSE_STATUS_EXECUTION_FAILED;
    const auto load = [&](const cusparseSpMatDescr* m, std::vector<int64_t>& off, std::vector<int64_t>& col,
                          std::vector<double>& val) {
        off = downloadIndices(m->offsets, m->rows + 1, m->offsetsType == CUSPARSE_INDEX_64I);
        col = downloadIndices(m->columns, m->nnz, m->columnsType == CUSPARSE_INDEX_64I);
        val = downloadValues(m->values, m->nnz, f64);
        const int64_t base = m->base == CUSPARSE_INDEX_BASE_ONE;
        for (int64_t& x : off) x -= base;
        for (int64_t& x : col) x -= base;
    };
    std::vector<int64_t> ao, ac, bo, bc;
    std::vector<double> av, bv;
    load(matA, ao, ac, av);
    load(matB, bo, bc, bv);
    // row by row, a dense accumulator over the touched columns
    auto& d = *spgemmDescr;
    d.offsets.assign(size_t(matA->rows + 1), 0);
    d.columns.clear();
    d.values.clear();
    std::vector<double> acc(size_t(matB->cols), 0);
    std::vector<char> used(size_t(matB->cols), 0);
    std::vector<int64_t> touched;
    for (int64_t i = 0; i < matA->rows; ++i) {
        touched.clear();
        for (int64_t t = ao[size_t(i)]; t < ao[size_t(i + 1)]; ++t) {
            const int64_t l = ac[size_t(t)];
            for (int64_t u = bo[size_t(l)]; u < bo[size_t(l + 1)]; ++u) {
                const int64_t j = bc[size_t(u)];
                if (!used[size_t(j)]) used[size_t(j)] = 1, touched.push_back(j);
                acc[size_t(j)] += av[size_t(t)] * bv[size_t(u)];
            }
        }
        std::sort(touched.begin(), touched.end());
        for (int64_t j : touched) {
            d.columns.push_back(j);
            d.values.push_back(alphaValue * acc[size_t(j)]);
            acc[size_t(j)] = 0;
            used[size_t(j)] = 0;
        }
        d.offsets[size_t(i + 1)] = int64_t(d.columns.size());
    }
    matC->rows = matA->rows;
    matC->cols = matB->cols;
    matC->nnz = int64_t(d.columns.size());
    return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseSpGEMM_copy(cusparseHandle_t handle, cusparseOperation_t, cusparseOperation_t, const void*,
                                     cusparseConstSpMatDescr_t, cusparseConstSpMatDescr_t, const void*,
                                     cusparseSpMatDescr_t matC, cudaDataType computeType, cusparseSpGEMMAlg_t,
                                     cusparseSpGEMMDescr_t spgemmDescr) {
    VULKANIZER_TRACE_API();
    if (!handle) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (!matC || !spgemmDescr) return CUSPARSE_STATUS_INVALID_VALUE;
    bool f64;
    if (!realType(computeType, f64)) return CUSPARSE_STATUS_NOT_SUPPORTED;
    const auto& d = *spgemmDescr;
    const int64_t base = matC->base == CUSPARSE_INDEX_BASE_ONE;
    if (cuStreamSynchronize(handle->stream) != CUDA_SUCCESS ||
        uploadIndices(matC->offsets, d.offsets, matC->offsetsType == CUSPARSE_INDEX_64I, base) ||
        uploadIndices(matC->columns, d.columns, matC->columnsType == CUSPARSE_INDEX_64I, base) ||
        uploadValues(matC->values, d.values, f64))
        return CUSPARSE_STATUS_EXECUTION_FAILED;
    return CUSPARSE_STATUS_SUCCESS;
}

// ---- SpSV, SpSM -------------------------------------------------------------------------------

cusparseStatus_t cusparseSpSV_createDescr(cusparseSpSVDescr_t* descr) {
    if (!descr) return CUSPARSE_STATUS_INVALID_VALUE;
    *descr = new cusparseSpSVDescr();
    return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseSpSV_destroyDescr(cusparseSpSVDescr_t descr) {
    delete descr;
    return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseSpSV_bufferSize(cusparseHandle_t handle, cusparseOperation_t, const void*,
                                         cusparseConstSpMatDescr_t matA, cusparseConstDnVecDescr_t,
                                         cusparseDnVecDescr_t, cudaDataType computeType, cusparseSpSVAlg_t,
                                         cusparseSpSVDescr_t, size_t* bufferSize) {
    if (!handle) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (!matA || !bufferSize) return CUSPARSE_STATUS_INVALID_VALUE;
    *bufferSize = csrWorkspace(*matA, false, computeType == CUDA_R_64F);
    return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseSpSV_analysis(cusparseHandle_t handle, cusparseOperation_t, const void*,
                                       cusparseConstSpMatDescr_t, cusparseConstDnVecDescr_t, cusparseDnVecDescr_t,
                                       cudaDataType, cusparseSpSVAlg_t, cusparseSpSVDescr_t spsvDescr,
                                       void* externalBuffer) {
    if (!handle) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (!spsvDescr) return CUSPARSE_STATUS_INVALID_VALUE;
    spsvDescr->workspace = CUdeviceptr(externalBuffer);
    return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseSpSV_solve(cusparseHandle_t handle, cusparseOperation_t opA, const void* alpha,
                                    cusparseConstSpMatDescr_t matA, cusparseConstDnVecDescr_t vecX,
                                    cusparseDnVecDescr_t vecY, cudaDataType computeType, cusparseSpSVAlg_t,
                                    cusparseSpSVDescr_t spsvDescr) {
    VULKANIZER_TRACE_API();
    if (!vecX || !vecY || !spsvDescr) return CUSPARSE_STATUS_INVALID_VALUE;
    return spsv(handle, opA, alpha, matA, denseOf(*vecX), denseOf(*vecY), computeType, vecX->valueType,
                vecY->valueType, spsvDescr->workspace);
}
cusparseStatus_t cusparseSpSM_createDescr(cusparseSpSMDescr_t* descr) {
    if (!descr) return CUSPARSE_STATUS_INVALID_VALUE;
    *descr = new cusparseSpSMDescr();
    return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseSpSM_destroyDescr(cusparseSpSMDescr_t descr) {
    delete descr;
    return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseSpSM_bufferSize(cusparseHandle_t handle, cusparseOperation_t, cusparseOperation_t,
                                         const void*, cusparseConstSpMatDescr_t matA, cusparseConstDnMatDescr_t,
                                         cusparseDnMatDescr_t, cudaDataType computeType, cusparseSpSMAlg_t,
                                         cusparseSpSMDescr_t, size_t* bufferSize) {
    if (!handle) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (!matA || !bufferSize) return CUSPARSE_STATUS_INVALID_VALUE;
    *bufferSize = csrWorkspace(*matA, false, computeType == CUDA_R_64F);
    return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseSpSM_analysis(cusparseHandle_t handle, cusparseOperation_t, cusparseOperation_t,
                                       const void*, cusparseConstSpMatDescr_t, cusparseConstDnMatDescr_t,
                                       cusparseDnMatDescr_t, cudaDataType, cusparseSpSMAlg_t,
                                       cusparseSpSMDescr_t spsmDescr, void* externalBuffer) {
    if (!handle) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (!spsmDescr) return CUSPARSE_STATUS_INVALID_VALUE;
    spsmDescr->workspace = CUdeviceptr(externalBuffer);
    return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseSpSM_solve(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB,
                                    const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstDnMatDescr_t matB,
                                    cusparseDnMatDescr_t matC, cudaDataType computeType, cusparseSpSMAlg_t,
                                    cusparseSpSMDescr_t spsmDescr) {
    VULKANIZER_TRACE_API();
    Dense b, c;
    if (!matB || !matC || !spsmDescr || !denseOf(*matB, b) || !denseOf(*matC, c)) return CUSPARSE_STATUS_INVALID_VALUE;
    if (opB != CUSPARSE_OPERATION_NON_TRANSPOSE || b.batch > 1 || c.batch > 1) return CUSPARSE_STATUS_NOT_SUPPORTED;
    return spsv(handle, opA, alpha, matA, b, c, computeType, matB->valueType, matC->valueType, spsmDescr->workspace);
}

}  // extern "C"

// ---- csrgeam2 ---------------------------------------------------------------------------------

namespace {

cusparseIndexBase_t baseOf(const cusparseMatDescr_t d) { return d ? d->base : CUSPARSE_INDEX_BASE_ZERO; }

cusparseStatus_t geamNnz(cusparseHandle_t h, int m, const cusparseMatDescr_t descrA, const int* rowA,
                         const int* colA, const cusparseMatDescr_t descrB, const int* rowB, const int* colB,
                         const cusparseMatDescr_t descrC, int* rowC, int* nnzTotal) {
    if (!h) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (m < 0 || !rowC) return CUSPARSE_STATUS_INVALID_VALUE;
    const uint32_t bases = baseFlag(baseOf(descrA), 0) | baseFlag(baseOf(descrB), 1) | baseFlag(baseOf(descrC), 2);
    const int baseC = baseOf(descrC) == CUSPARSE_INDEX_BASE_ONE;
    Params p{};
    p.ptr[0] = CUdeviceptr(rowA);
    p.ptr[1] = CUdeviceptr(colA);
    p.ptr[2] = CUdeviceptr(rowB);
    p.ptr[3] = CUdeviceptr(colB);
    p.ptr[4] = CUdeviceptr(rowC);
    p.items = m;
    p.flags = GeamCount << kModeShift | bases;
    if (m == 0) {
        if (CUresult r = cuMemsetD32Async(CUdeviceptr(rowC), unsigned(baseC), 1, h->stream)) return status(r);
    } else if (CUresult r = launch(h->stream, Geam, false, p)) {
        return status(r);
    }
    p = Params{};
    p.ptr[0] = CUdeviceptr(rowC);
    p.items = m + 1;
    p.extra = baseC;
    p.flags = Scan << kModeShift;
    if (m > 0) {
        if (CUresult r = launch(h->stream, CsrUtil, false, p, 1)) return status(r);
    }
    if (!nnzTotal) return CUSPARSE_STATUS_SUCCESS;
    if (h->pointerMode == CUSPARSE_POINTER_MODE_DEVICE) {
        p = Params{};
        p.ptr[0] = CUdeviceptr(nnzTotal);
        p.ptr[1] = CUdeviceptr(rowC);
        p.items = 1;
        p.m = m;
        p.extra = baseC;
        p.flags = Nnz << kModeShift;
        return status(launch(h->stream, CsrUtil, false, p));
    }
    int last = 0;
    if (cuStreamSynchronize(h->stream) != CUDA_SUCCESS ||
        cuMemcpyDtoH(&last, CUdeviceptr(rowC) + CUdeviceptr(m) * 4, 4) != CUDA_SUCCESS)
        return CUSPARSE_STATUS_EXECUTION_FAILED;
    *nnzTotal = last - baseC;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t geam(cusparseHandle_t h, bool f64, int m, const void* alpha, const cusparseMatDescr_t descrA,
                      int nnzA, const void* valA, const int* rowA, const int* colA, const void* beta,
                      const cusparseMatDescr_t descrB, int nnzB, const void* valB, const int* rowB, const int* colB,
                      const cusparseMatDescr_t descrC, void* valC, int* rowC, int* colC, void* buffer) {
    if (!h) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (m < 0 || nnzA < 0 || nnzB < 0) return CUSPARSE_STATUS_INVALID_VALUE;
    if (f64) {
        int v = 0;
        if (vulkanizerGetFeature(VULKANIZER_FEATURE_FLOAT64, &v) != CUDA_SUCCESS || !v) return CUSPARSE_STATUS_NOT_SUPPORTED;
    }
    if (m == 0 || nnzA + nnzB == 0) return CUSPARSE_STATUS_SUCCESS;
    if (!buffer) return CUSPARSE_STATUS_INVALID_VALUE;
    const uint32_t bases = baseFlag(baseOf(descrA), 0) | baseFlag(baseOf(descrB), 1) | baseFlag(baseOf(descrC), 2);
    const int bound = nnzA + nnzB;
    Params p{};
    if (!scalarBits(h, alpha, f64, p.alphaBits) || !scalarBits(h, beta, f64, p.betaBits))
        return CUSPARSE_STATUS_INVALID_VALUE;
    p.ptr[0] = CUdeviceptr(rowA);
    p.ptr[1] = CUdeviceptr(colA);
    p.ptr[2] = CUdeviceptr(rowB);
    p.ptr[3] = CUdeviceptr(colB);
    p.ptr[4] = CUdeviceptr(rowC);
    p.ptr[5] = CUdeviceptr(colC);
    p.ptr[6] = CUdeviceptr(buffer);
    p.items = m;
    p.extra = bound;
    p.flags = GeamColumns << kModeShift | bases;
    if (CUresult r = launch(h->stream, Geam, f64, p)) return status(r);
    // the values: one item per possible entry, those beyond C's nnz (rowC[m]) do nothing
    p.ptr[0] = CUdeviceptr(valA);
    p.ptr[1] = CUdeviceptr(valB);
    p.ptr[2] = CUdeviceptr(valC);
    p.ptr[3] = CUdeviceptr(rowC);
    p.items = bound;
    p.m = m;
    p.flags = GeamValues << kModeShift | bases;
    return status(launch(h->stream, Geam, f64, p));
}

}  // namespace

extern "C" {

cusparseStatus_t cusparseScsrgeam2_bufferSizeExt(cusparseHandle_t handle, int, int, const float*,
                                                 const cusparseMatDescr_t, int nnzA, const float*, const int*,
                                                 const int*, const float*, const cusparseMatDescr_t, int nnzB,
                                                 const float*, const int*, const int*, const cusparseMatDescr_t,
                                                 const float*, const int*, const int*, size_t* pBufferSizeInBytes) {
    if (!handle) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (!pBufferSizeInBytes || nnzA < 0 || nnzB < 0) return CUSPARSE_STATUS_INVALID_VALUE;
    *pBufferSizeInBytes = size_t(nnzA + nnzB) * 8 + 16;  // where each entry of C comes from (two ints)
    return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseDcsrgeam2_bufferSizeExt(cusparseHandle_t handle, int m, int n, const double*,
                                                 const cusparseMatDescr_t descrA, int nnzA, const double*,
                                                 const int*, const int*, const double*,
                                                 const cusparseMatDescr_t descrB, int nnzB, const double*,
                                                 const int*, const int*, const cusparseMatDescr_t descrC,
                                                 const double*, const int*, const int*, size_t* pBufferSizeInBytes) {
    return cusparseScsrgeam2_bufferSizeExt(handle, m, n, nullptr, descrA, nnzA, nullptr, nullptr, nullptr, nullptr,
                                           descrB, nnzB, nullptr, nullptr, nullptr, descrC, nullptr, nullptr, nullptr,
                                           pBufferSizeInBytes);
}
cusparseStatus_t cusparseXcsrgeam2Nnz(cusparseHandle_t handle, int m, int, const cusparseMatDescr_t descrA, int,
                                      const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                      const cusparseMatDescr_t descrB, int, const int* csrSortedRowPtrB,
                                      const int* csrSortedColIndB, const cusparseMatDescr_t descrC,
                                      int* csrSortedRowPtrC, int* nnzTotalDevHostPtr, void*) {
    VULKANIZER_TRACE_API();
    return geamNnz(handle, m, descrA, csrSortedRowPtrA, csrSortedColIndA, descrB, csrSortedRowPtrB, csrSortedColIndB,
                   descrC, csrSortedRowPtrC, nnzTotalDevHostPtr);
}
cusparseStatus_t cusparseScsrgeam2(cusparseHandle_t handle, int m, int, const float* alpha,
                                   const cusparseMatDescr_t descrA, int nnzA, const float* csrSortedValA,
                                   const int* csrSortedRowPtrA, const int* csrSortedColIndA, const float* beta,
                                   const cusparseMatDescr_t descrB, int nnzB, const float* csrSortedValB,
                                   const int* csrSortedRowPtrB, const int* csrSortedColIndB,
                                   const cusparseMatDescr_t descrC, float* csrSortedValC, int* csrSortedRowPtrC,
                                   int* csrSortedColIndC, void* pBuffer) {
    VULKANIZER_TRACE_API();
    return geam(handle, false, m, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, descrB,
                nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC,
                csrSortedColIndC, pBuffer);
}
cusparseStatus_t cusparseDcsrgeam2(cusparseHandle_t handle, int m, int, const double* alpha,
                                   const cusparseMatDescr_t descrA, int nnzA, const double* csrSortedValA,
                                   const int* csrSortedRowPtrA, const int* csrSortedColIndA, const double* beta,
                                   const cusparseMatDescr_t descrB, int nnzB, const double* csrSortedValB,
                                   const int* csrSortedRowPtrB, const int* csrSortedColIndB,
                                   const cusparseMatDescr_t descrC, double* csrSortedValC, int* csrSortedRowPtrC,
                                   int* csrSortedColIndC, void* pBuffer) {
    VULKANIZER_TRACE_API();
    return geam(handle, true, m, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, descrB,
                nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC,
                csrSortedColIndC, pBuffer);
}

// ---- conversion and sorting -------------------------------------------------------------------

cusparseStatus_t cusparseXcoo2csr(cusparseHandle_t handle, const int* cooRowInd, int nnz, int m, int* csrSortedRowPtr,
                                  cusparseIndexBase_t idxBase) {
    VULKANIZER_TRACE_API();
    if (!handle) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (nnz < 0 || m < 0 || !csrSortedRowPtr) return CUSPARSE_STATUS_INVALID_VALUE;
    Params p{};
    p.ptr[0] = CUdeviceptr(cooRowInd);
    p.ptr[1] = CUdeviceptr(csrSortedRowPtr);
    p.items = m + 1;
    p.k = nnz;
    p.extra = idxBase == CUSPARSE_INDEX_BASE_ONE;
    p.flags = CooToCsr << kModeShift | baseFlag(idxBase, 0);
    return status(launch(handle->stream, CsrUtil, false, p));
}
cusparseStatus_t cusparseCreateIdentityPermutation(cusparseHandle_t handle, int n, int* p) {
    if (!handle) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (n < 0 || (n && !p)) return CUSPARSE_STATUS_INVALID_VALUE;
    Params q{};
    q.ptr[0] = CUdeviceptr(p);
    q.items = n;
    q.flags = Iota << kModeShift;
    return status(launch(handle->stream, CsrUtil, false, q));
}
cusparseStatus_t cusparseXcoosort_bufferSizeExt(cusparseHandle_t handle, int, int, int, const int*, const int*,
                                                size_t* pBufferSizeInBytes) {
    if (!handle) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (!pBufferSizeInBytes) return CUSPARSE_STATUS_INVALID_VALUE;
    *pBufferSizeInBytes = 16;  // sorted on the host
    return CUSPARSE_STATUS_SUCCESS;
}
// By row, then column; P permuted alike. On the host.
cusparseStatus_t cusparseXcoosortByRow(cusparseHandle_t handle, int, int, int nnz, int* cooRowsA, int* cooColsA, int* P,
                                       void*) {
    VULKANIZER_TRACE_API();
    if (!handle) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (nnz < 0) return CUSPARSE_STATUS_INVALID_VALUE;
    if (nnz == 0) return CUSPARSE_STATUS_SUCCESS;
    if (cuStreamSynchronize(handle->stream) != CUDA_SUCCESS) return CUSPARSE_STATUS_EXECUTION_FAILED;
    std::vector<int> rows(static_cast<size_t>(nnz)), cols(static_cast<size_t>(nnz)), perm(static_cast<size_t>(nnz)), order(static_cast<size_t>(nnz));
    cuMemcpyDtoH(rows.data(), CUdeviceptr(cooRowsA), size_t(nnz) * 4);
    cuMemcpyDtoH(cols.data(), CUdeviceptr(cooColsA), size_t(nnz) * 4);
    if (P) cuMemcpyDtoH(perm.data(), CUdeviceptr(P), size_t(nnz) * 4);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return rows[size_t(a)] != rows[size_t(b)] ? rows[size_t(a)] < rows[size_t(b)] : cols[size_t(a)] < cols[size_t(b)];
    });
    std::vector<int> r2(static_cast<size_t>(nnz)), c2(static_cast<size_t>(nnz)), p2(static_cast<size_t>(nnz));
    for (size_t i = 0; i < order.size(); ++i) {
        r2[i] = rows[size_t(order[i])];
        c2[i] = cols[size_t(order[i])];
        p2[i] = perm[size_t(order[i])];
    }
    if (cuMemcpyHtoD(CUdeviceptr(cooRowsA), r2.data(), size_t(nnz) * 4) ||
        cuMemcpyHtoD(CUdeviceptr(cooColsA), c2.data(), size_t(nnz) * 4) ||
        (P && cuMemcpyHtoD(CUdeviceptr(P), p2.data(), size_t(nnz) * 4)))
        return CUSPARSE_STATUS_EXECUTION_FAILED;
    return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseXcsrsort_bufferSizeExt(cusparseHandle_t handle, int, int, int, const int*, const int*,
                                                size_t* pBufferSizeInBytes) {
    return cusparseXcoosort_bufferSizeExt(handle, 0, 0, 0, nullptr, nullptr, pBufferSizeInBytes);
}
// The columns of each row ascending; P permuted alike. On the host.
cusparseStatus_t cusparseXcsrsort(cusparseHandle_t handle, int m, int, int nnz, const cusparseMatDescr_t descrA,
                                  const int* csrRowPtrA, int* csrColIndA, int* P, void*) {
    VULKANIZER_TRACE_API();
    if (!handle) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (nnz < 0 || m < 0) return CUSPARSE_STATUS_INVALID_VALUE;
    if (nnz == 0) return CUSPARSE_STATUS_SUCCESS;
    if (cuStreamSynchronize(handle->stream) != CUDA_SUCCESS) return CUSPARSE_STATUS_EXECUTION_FAILED;
    const int base = baseOf(descrA) == CUSPARSE_INDEX_BASE_ONE;
    std::vector<int> offsets(static_cast<size_t>(m + 1)), cols(static_cast<size_t>(nnz)), perm(static_cast<size_t>(nnz));
    cuMemcpyDtoH(offsets.data(), CUdeviceptr(csrRowPtrA), size_t(m + 1) * 4);
    cuMemcpyDtoH(cols.data(), CUdeviceptr(csrColIndA), size_t(nnz) * 4);
    if (P) cuMemcpyDtoH(perm.data(), CUdeviceptr(P), size_t(nnz) * 4);
    std::vector<int> order(static_cast<size_t>(nnz)), c2(static_cast<size_t>(nnz)), p2(static_cast<size_t>(nnz));
    for (int i = 0; i < m; ++i) {
        const int first = offsets[size_t(i)] - base, last = offsets[size_t(i + 1)] - base;
        std::iota(order.begin() + first, order.begin() + last, first);
        std::stable_sort(order.begin() + first, order.begin() + last,
                         [&](int a, int b) { return cols[size_t(a)] < cols[size_t(b)]; });
    }
    for (size_t t = 0; t < order.size(); ++t) {
        c2[t] = cols[size_t(order[t])];
        p2[t] = perm[size_t(order[t])];
    }
    if (cuMemcpyHtoD(CUdeviceptr(csrColIndA), c2.data(), size_t(nnz) * 4) ||
        (P && cuMemcpyHtoD(CUdeviceptr(P), p2.data(), size_t(nnz) * 4)))
        return CUSPARSE_STATUS_EXECUTION_FAILED;
    return CUSPARSE_STATUS_SUCCESS;
}

}  // extern "C"
