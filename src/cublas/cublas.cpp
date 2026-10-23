// Vulkanizer's libcublas: the cuBLAS GEMM family on hand-written Vulkan kernels (shaders/gemm.comp),
// launched through the Vulkanizer driver (vulkanizerCreateSpirvFunction + cuLaunchKernel), so they run
// on the caller's stream like any CUDA kernel.

#include <cublas_v2.h>
#include <cuda.h>
#include <vulkanizer/vulkanizer_ext.h>

#include "common/log.h"
#include "linalg/linalg.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <cstring>
#include <map>
#include <mutex>
#include <vector>

namespace {

const uint32_t kGemmSpirv[] = {
#include "gemm.comp.inc"
};
const uint32_t kGemmTiledSpirv[] = {
#include "gemm_tiled.comp.inc"
};
const uint32_t kGemmF16TnSpirv[] = {
#include "gemm_f16_tn.comp.inc"
};
const uint32_t kGemmF16TnHSpirv[] = {
#include "gemm_f16_tn_h.comp.inc"
};
const uint32_t kGemmF16TnMSpirv[] = {
#include "gemm_f16_tn_m.comp.inc"
};
const uint32_t kGemmF16TnMHSpirv[] = {
#include "gemm_f16_tn_mh.comp.inc"
};
// gemm_coopmat.comp: S = 16 and 8, each with f32 and f16 C
const uint32_t kGemmCoop16Spirv[] = {
#include "gemm_coopmat16.comp.inc"
};
const uint32_t kGemmCoop16HSpirv[] = {
#include "gemm_coopmat16_h.comp.inc"
};
const uint32_t kGemmCoop8Spirv[] = {
#include "gemm_coopmat8.comp.inc"
};
const uint32_t kGemmCoop8HSpirv[] = {
#include "gemm_coopmat8_h.comp.inc"
};
const uint32_t kGemvSpirv[] = {
#include "gemv.comp.inc"
};
const uint32_t kTrsmSpirv[] = {
#include "trsm.comp.inc"
};

// Push constants of trsm.comp (std430).
struct TrsmParams {
    uint64_t a, b;
    int32_t m, n;
    int32_t lda, ldb;
    float alpha;
    uint32_t flags;
    int32_t batchOffset;
};
static_assert(sizeof(TrsmParams) == 48, "must match trsm.comp");

// Push constants of gemm.comp (std430).
struct GemmParams {
    uint64_t a, b, c;
    int64_t strideA, strideB, strideC;
    int32_t m, n, k;
    int32_t lda, ldb, ldc;
    float alpha, beta;
    uint32_t flags;
    int32_t batchOffset;
};
static_assert(sizeof(GemmParams) == 88, "must match gemm.comp");

// Push constants of gemv.comp (std430).
struct GemvParams {
    uint64_t m, v, y;
    int64_t strideM, strideV, strideY;
    int64_t rowStride, colStride;
    int32_t incv, incy;
    int32_t rows, len;
    float alpha, beta;
    uint32_t flags;
    int32_t batchOffset;
};
static_assert(sizeof(GemvParams) == 96, "must match gemv.comp");
constexpr uint32_t kGemvRowMajor = 0x100;

constexpr uint32_t kTransA = 1, kTransB = 2, kPointerArrays = 0x10000;
constexpr uint32_t kTile = 64, kFastTile = 128, kMaxGridZ = 65535;

float halfToFloat(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000) << 16, exp = (h >> 10) & 0x1F, mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0x1F) bits = sign | 0x7F800000 | (mant << 13);
    else if (exp) bits = sign | ((exp + 112) << 23) | (mant << 13);
    else if (!mant) bits = sign;
    else {  // subnormal
        int e = -1;
        uint32_t m = mant;
        do { ++e; m <<= 1; } while (!(m & 0x400));
        bits = sign | ((112 - e) << 23) | ((m & 0x3FF) << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

bool typeCode(cudaDataType t, uint32_t& code) {
    switch (t) {
    case CUDA_R_32F: code = 0; return true;
    case CUDA_R_16F: code = 1; return true;
    case CUDA_R_16BF: code = 2; return true;
    default: return false;
    }
}

}  // namespace

struct cublasContext {
    CUcontext ctx = nullptr;
    CUstream stream = nullptr;
    cublasMath_t math = CUBLAS_DEFAULT_MATH;
    cublasPointerMode_t pointerMode = CUBLAS_POINTER_MODE_HOST;
    cublasAtomicsMode_t atomics = CUBLAS_ATOMICS_NOT_ALLOWED;
    CUfunction gemm = nullptr;
    std::map<uint32_t, CUfunction> gemmTiled;  // gemm_tiled.comp per operand layout (its kConfig)
    CUfunction gemv = nullptr;
    // gemm_f16_tn.comp variants, indexed by bit 0 = f16 C, bit 1 = f16 math
    CUfunction gemmF16Tn[4] = {};
    int float16 = -1;  // device has f16 arithmetic (-1: not queried yet)
    // gemm_coopmat.comp, indexed by f16 C; its matrix size S (0: no cooperative matrices, -1: not
    // queried yet) and the subgroup size (a workgroup is 4 subgroups)
    CUfunction gemmCoop[2] = {};
    int coopSize = -1, subgroupSize = 0;
    CUfunction trsm = nullptr;
};

namespace {

// The SPIR-V with specialization constant `id`'s default changed to `value` (for constants the
// pipeline creation does not set: they then fold like literals).
std::vector<uint32_t> withSpecConstant(const uint32_t* code, size_t words, uint32_t id, uint32_t value) {
    std::vector<uint32_t> out(code, code + words);
    uint32_t target = 0;
    for (size_t i = 5; i < out.size(); i += out[i] >> 16)  // OpDecorate <target> SpecId <id>
        if ((out[i] & 0xFFFF) == 71 && (out[i] >> 16) == 4 && out[i + 2] == 1 && out[i + 3] == id) target = out[i + 1];
    for (size_t i = 5; target && i < out.size(); i += out[i] >> 16)  // OpSpecConstant <type> <target> <value>
        if ((out[i] & 0xFFFF) == 50 && out[i + 2] == target) out[i + 3] = value;
    return out;
}

// alpha/beta as float: host or device pointer, of the compute type's scalar type.
bool scalar(cublasHandle_t h, const void* p, cublasComputeType_t compute, float& out) {
    unsigned char buf[8] = {};
    const size_t size = compute == CUBLAS_COMPUTE_16F || compute == CUBLAS_COMPUTE_16F_PEDANTIC ? 2
                        : compute == CUBLAS_COMPUTE_64F || compute == CUBLAS_COMPUTE_64F_PEDANTIC ? 8
                                                                                                  : 4;
    if (!p) return false;
    if (h->pointerMode == CUBLAS_POINTER_MODE_DEVICE) {
        if (cuMemcpyDtoH(buf, reinterpret_cast<CUdeviceptr>(p), size) != CUDA_SUCCESS) return false;
    } else {
        std::memcpy(buf, p, size);
    }
    if (size == 2) {
        uint16_t v;
        std::memcpy(&v, buf, 2);
        out = halfToFloat(v);
    } else if (size == 8) {
        double d;
        std::memcpy(&d, buf, 8);
        out = float(d);
    } else {
        std::memcpy(&out, buf, 4);
    }
    return true;
}

struct Gemm {
    cublasOperation_t transa, transb;
    int m, n, k;
    const void* alpha;
    const void* a;
    cudaDataType ta;
    int lda;
    long long strideA;
    const void* b;
    cudaDataType tb;
    int ldb;
    long long strideB;
    const void* beta;
    void* c;
    cudaDataType tc;
    int ldc;
    long long strideC;
    int batch;
    bool pointerArrays;
    cublasComputeType_t compute;
};

// The device array of output pointers, passed like the matrices (only its address is used).
template <class T>
void* pointerArray(T* const* array) {
    return const_cast<void*>(static_cast<const void*>(array));
}

// alpha of the handle's pointer mode, as a double.
template <class T>
bool readAlpha(cublasHandle_t h, const T* alpha, double& out) {
    if (!alpha) return false;
    T v;
    if (h->pointerMode == CUBLAS_POINTER_MODE_DEVICE) {
        if (cuMemcpyDtoH(&v, CUdeviceptr(alpha), sizeof v) != CUDA_SUCCESS) return false;
    } else {
        v = *alpha;
    }
    out = double(v);
    return true;
}

// f64 GEMMs (all three matrices f64) on the shared f64 kernel (src/linalg), f64 accumulation.
cublasStatus_t runF64(cublasHandle_t h, const Gemm& g) {
    if (g.ta != CUDA_R_64F || g.tb != CUDA_R_64F || g.tc != CUDA_R_64F) {
        VULKANIZER_WARN("cublas: unsupported GEMM data types (%d, %d, %d)", int(g.ta), int(g.tb), int(g.tc));
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    if (!vulkanizer::linalg::hasFloat64()) return CUBLAS_STATUS_NOT_SUPPORTED;
    double alpha, beta;
    if (!readAlpha(h, static_cast<const double*>(g.alpha), alpha) || !readAlpha(h, static_cast<const double*>(g.beta), beta))
        return CUBLAS_STATUS_INVALID_VALUE;
    const auto matrices = [&](const void* base, long long stride, int ld) {
        return vulkanizer::linalg::Matrices{CUdeviceptr(base), g.pointerArrays, g.pointerArrays ? 0 : stride, ld};
    };
    return vulkanizer::linalg::gemm(h->stream, true, g.transa != CUBLAS_OP_N, g.transb != CUBLAS_OP_N, g.m, g.n, g.k, alpha,
                        matrices(g.a, g.strideA, g.lda), matrices(g.b, g.strideB, g.ldb), beta,
                        matrices(g.c, g.strideC, g.ldc), g.batch) == CUDA_SUCCESS
               ? CUBLAS_STATUS_SUCCESS
               : CUBLAS_STATUS_EXECUTION_FAILED;
}

cublasStatus_t run(cublasHandle_t h, const Gemm& g) {
    if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (g.m < 0 || g.n < 0 || g.k < 0 || g.batch < 0) return CUBLAS_STATUS_INVALID_VALUE;
    const int rowsA = g.transa == CUBLAS_OP_N ? g.m : g.k, rowsB = g.transb == CUBLAS_OP_N ? g.k : g.n;
    if (g.lda < std::max(1, rowsA) || g.ldb < std::max(1, rowsB) || g.ldc < std::max(1, g.m))
        return CUBLAS_STATUS_INVALID_VALUE;
    if (g.transa > CUBLAS_OP_C || g.transb > CUBLAS_OP_C) return CUBLAS_STATUS_INVALID_VALUE;
    if (g.m == 0 || g.n == 0 || g.batch == 0) return CUBLAS_STATUS_SUCCESS;
    uint32_t ta, tb, tc;
    if (!typeCode(g.ta, ta) || !typeCode(g.tb, tb) || !typeCode(g.tc, tc)) return runF64(h, g);
    GemmParams p{};
    if (!scalar(h, g.alpha, g.compute, p.alpha) || !scalar(h, g.beta, g.compute, p.beta))
        return CUBLAS_STATUS_INVALID_VALUE;
    p.a = reinterpret_cast<uint64_t>(g.a);
    p.b = reinterpret_cast<uint64_t>(g.b);
    p.c = reinterpret_cast<uint64_t>(g.c);
    p.strideA = g.strideA;
    p.strideB = g.strideB;
    p.strideC = g.strideC;
    p.m = g.m, p.n = g.n, p.k = g.k;
    p.lda = g.lda, p.ldb = g.ldb, p.ldc = g.ldc;
    p.flags = (g.transa != CUBLAS_OP_N ? kTransA : 0) | (g.transb != CUBLAS_OP_N ? kTransB : 0) | ta << 4 | tb << 8 |
              tc << 12 | (g.pointerArrays ? kPointerArrays : 0);

    // n = 1 (or m = 1, as C^T = op(B)^T op(A)^T): a matrix-vector product, which the tiled
    // kernels would compute at 1/64 of their tile.
    if ((g.n == 1 || g.m == 1) && g.k > 0) {
        const bool transA = g.transa != CUBLAS_OP_N, transB = g.transb != CUBLAS_OP_N;
        GemvParams q{};
        q.alpha = p.alpha, q.beta = p.beta, q.len = g.k;
        uint32_t tm, tv;
        if (g.n == 1) {  // y = op(A) * op(B)[:, 0]
            q.m = p.a, q.v = p.b, q.y = p.c;
            q.strideM = g.strideA, q.strideV = g.strideB, q.strideY = g.strideC;
            q.rows = g.m;
            q.rowStride = transA ? g.lda : 1, q.colStride = transA ? 1 : g.lda;
            q.incv = transB ? g.ldb : 1, q.incy = 1;
            tm = ta, tv = tb;
        } else {  // y^T = op(A)[0, :] * op(B)
            q.m = p.b, q.v = p.a, q.y = p.c;
            q.strideM = g.strideB, q.strideV = g.strideA, q.strideY = g.strideC;
            q.rows = g.n;
            q.rowStride = transB ? 1 : g.ldb, q.colStride = transB ? g.ldb : 1;
            q.incv = transA ? 1 : g.lda, q.incy = g.ldc;
            tm = tb, tv = ta;
        }
        const bool rowMajor = q.colStride == 1;
        // threads per row: about 16 elements each, 4..64
        uint32_t lanesLog2 = 2;
        while (lanesLog2 < 6 && (16u << lanesLog2) < uint32_t(g.k)) ++lanesLog2;
        q.flags = tm | tv << 2 | tc << 4 | (rowMajor ? kGemvRowMajor : 0) | lanesLog2 << 12 |
                  (g.pointerArrays ? kPointerArrays : 0);
        VULKANIZER_DEBUG("gemm %c%c m=%d n=%d k=%d batch %d%s -> gemv", transA ? 'T' : 'N', transB ? 'T' : 'N', g.m, g.n,
                    g.k, g.batch, g.pointerArrays ? " (arrays)" : "");
        if (!h->gemv &&
            vulkanizerCreateSpirvFunction(&h->gemv, kGemvSpirv, sizeof kGemvSpirv / 4, "main", sizeof(GemvParams)) != CUDA_SUCCESS)
            return CUBLAS_STATUS_INTERNAL_ERROR;
        const unsigned perGroup = rowMajor ? 256u >> lanesLog2 : 64;
        const unsigned gx = (unsigned(q.rows) + perGroup - 1) / perGroup;
        for (int first = 0; first < g.batch; first += int(kMaxGridZ)) {
            q.batchOffset = first;
            size_t size = sizeof q;
            void* extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, &q, CU_LAUNCH_PARAM_BUFFER_SIZE, &size, CU_LAUNCH_PARAM_END};
            const unsigned gz = unsigned(std::min<int>(g.batch - first, int(kMaxGridZ)));
            if (cuLaunchKernel(h->gemv, gx, 1, gz, 64, 4, 1, 0, h->stream, nullptr, extra) != CUDA_SUCCESS)
                return CUBLAS_STATUS_EXECUTION_FAILED;
        }
        return CUBLAS_STATUS_SUCCESS;
    }

    // f16 A^T * B with both operands contiguous along k and 16-byte aligned rows: the
    // vectorized kernel (transformer projections). Everything else takes the generic one.
    const auto aligned = [](const void* ptr, long long stride, int ld) {
        return (reinterpret_cast<uintptr_t>(ptr) & 15) == 0 && stride % 8 == 0 && ld % 8 == 0;
    };
    const bool fast = g.transa != CUBLAS_OP_N && g.transb == CUBLAS_OP_N && ta == 1 && tb == 1 && tc != 2 &&
                      !g.pointerArrays && aligned(g.a, g.strideA, g.lda) && aligned(g.b, g.strideB, g.ldb);
    // CUBLAS_COMPUTE_16F permits f16 arithmetic: packed f16 products where the device has it.
    if (fast && h->float16 < 0 && vulkanizerGetFeature(VULKANIZER_FEATURE_FLOAT16_ARITHMETIC, &h->float16) != CUDA_SUCCESS)
        h->float16 = 0;
    // On cooperative matrices where the device has them: f16 products, f32 sums.
    if (fast && h->coopSize < 0) {
        if (vulkanizerGetFeature(VULKANIZER_FEATURE_COOPERATIVE_MATRIX_F16, &h->coopSize) != CUDA_SUCCESS ||
            vulkanizerGetFeature(VULKANIZER_FEATURE_SUBGROUP_SIZE, &h->subgroupSize) != CUDA_SUCCESS ||
            4 * h->subgroupSize > 1024)
            h->coopSize = 0;
    }
    const bool coop = fast && h->coopSize > 0;
    const bool f16Math = fast && !coop && h->float16 > 0 &&
                         (g.compute == CUBLAS_COMPUTE_16F || g.compute == CUBLAS_COMPUTE_32F_FAST_16F) &&
                         !std::getenv("VULKANIZER_GEMM_F32");
    VULKANIZER_DEBUG("gemm %c%c m=%d n=%d k=%d types %u/%u/%u compute %d batch %d%s -> %s", g.transa ? 'T' : 'N',
                g.transb ? 'T' : 'N', g.m, g.n, g.k, ta, tb, tc, int(g.compute), g.batch, g.pointerArrays ? " (arrays)" : "",
                !fast ? (g.m >= int(kFastTile) && g.n >= int(kFastTile) ? "tiled" : "generic")
                      : coop ? "f16 tn, cooperative matrices" : f16Math ? "f16 tn, f16 math" : "f16 tn");
    const int variant = (tc == 1 ? 1 : 0) | (f16Math ? 2 : 0);
    // The rest: 128 x 128 tiles once both dimensions fill one, compiled for the operands' layout:
    // contiguous along k or not, the types, and 16-byte loads where every row start is 16-byte
    // aligned (not knowable for pointer arrays) and the contiguous dimension a multiple of 8.
    const bool tiled = !fast && g.m >= int(kFastTile) && g.n >= int(kFastTile);
    uint32_t config = 0;
    if (tiled) {
        const bool alongKA = g.transa != CUBLAS_OP_N, alongKB = g.transb == CUBLAS_OP_N;
        const auto vec = [&](const void* ptr, long long stride, int ld, uint32_t type, int contiguous) {
            const long long size = type == 0 ? 4 : 2;
            return !g.pointerArrays && (reinterpret_cast<uintptr_t>(ptr) & 15) == 0 && (stride * size) % 16 == 0 &&
                   (ld * size) % 16 == 0 && contiguous % 8 == 0;
        };
        config = (alongKA ? 1u : 0u) | (alongKB ? 2u : 0u) | ta << 2 | tb << 4 |
                 (vec(g.a, g.strideA, g.lda, ta, alongKA ? g.k : g.m) ? 64u : 0u) |
                 (vec(g.b, g.strideB, g.ldb, tb, alongKB ? g.k : g.n) ? 128u : 0u);
    }
    if (coop) {
        CUfunction& fn = h->gemmCoop[tc == 1 ? 1 : 0];
        if (!fn) {
            const bool s16 = h->coopSize == 16, halfC = tc == 1;
            const uint32_t* code = s16 ? (halfC ? kGemmCoop16HSpirv : kGemmCoop16Spirv) : (halfC ? kGemmCoop8HSpirv : kGemmCoop8Spirv);
            const size_t words = (s16 ? (halfC ? sizeof kGemmCoop16HSpirv : sizeof kGemmCoop16Spirv)
                                      : (halfC ? sizeof kGemmCoop8HSpirv : sizeof kGemmCoop8Spirv)) / 4;
            if (vulkanizerCreateSpirvFunction(&fn, code, words, "main", sizeof(GemmParams)) != CUDA_SUCCESS)
                return CUBLAS_STATUS_INTERNAL_ERROR;
        }
        const unsigned tile = 4 * unsigned(h->coopSize);  // 2 x 2 subgroups of 2 x 2 matrices
        const unsigned gx = (unsigned(g.m) + tile - 1) / tile, gy = (unsigned(g.n) + tile - 1) / tile;
        for (int first = 0; first < g.batch; first += int(kMaxGridZ)) {
            p.batchOffset = first;
            size_t size = sizeof p;
            void* extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, &p, CU_LAUNCH_PARAM_BUFFER_SIZE, &size, CU_LAUNCH_PARAM_END};
            const unsigned gz = unsigned(std::min<int>(g.batch - first, int(kMaxGridZ)));
            if (cuLaunchKernel(fn, gx, gy, gz, 4 * unsigned(h->subgroupSize), 1, 1, 0, h->stream, nullptr, extra) != CUDA_SUCCESS)
                return CUBLAS_STATUS_EXECUTION_FAILED;
        }
        return CUBLAS_STATUS_SUCCESS;
    }
    CUfunction* fn = fast ? &h->gemmF16Tn[variant] : tiled ? &h->gemmTiled[config] : &h->gemm;
    if (!*fn) {
        static const uint32_t* const kCode[] = {kGemmF16TnSpirv, kGemmF16TnHSpirv, kGemmF16TnMSpirv, kGemmF16TnMHSpirv};
        static const size_t kWords[] = {sizeof kGemmF16TnSpirv / 4, sizeof kGemmF16TnHSpirv / 4,
                                        sizeof kGemmF16TnMSpirv / 4, sizeof kGemmF16TnMHSpirv / 4};
        std::vector<uint32_t> specialized;
        if (tiled) specialized = withSpecConstant(kGemmTiledSpirv, sizeof kGemmTiledSpirv / 4, 10, config);
        const uint32_t* code = fast ? kCode[variant] : tiled ? specialized.data() : kGemmSpirv;
        const size_t words = fast ? kWords[variant] : tiled ? specialized.size() : sizeof kGemmSpirv / 4;
        if (vulkanizerCreateSpirvFunction(fn, code, words, "main", sizeof(GemmParams)) != CUDA_SUCCESS)
            return CUBLAS_STATUS_INTERNAL_ERROR;
    }
    const unsigned tile = fast || tiled ? kFastTile : kTile;
    const unsigned gx = (unsigned(g.m) + tile - 1) / tile, gy = (unsigned(g.n) + tile - 1) / tile;
    for (int first = 0; first < g.batch; first += int(kMaxGridZ)) {
        p.batchOffset = first;
        size_t size = sizeof p;
        void* extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, &p, CU_LAUNCH_PARAM_BUFFER_SIZE, &size, CU_LAUNCH_PARAM_END};
        const unsigned gz = unsigned(std::min<int>(g.batch - first, int(kMaxGridZ)));
        if (cuLaunchKernel(*fn, gx, gy, gz, 16, 16, 1, 0, h->stream, nullptr, extra) != CUDA_SUCCESS)
            return CUBLAS_STATUS_EXECUTION_FAILED;
    }
    return CUBLAS_STATUS_SUCCESS;
}

}  // namespace

extern "C" {

cublasStatus_t cublasCreate_v2(cublasHandle_t* handle) {
    VULKANIZER_TRACE_API();
    if (!handle) return CUBLAS_STATUS_INVALID_VALUE;
    CUcontext ctx = nullptr;
    if (cuCtxGetCurrent(&ctx) != CUDA_SUCCESS || !ctx) {
        // No current context (driver API user without one): use device 0's primary context.
        if (cuInit(0) != CUDA_SUCCESS || cuDevicePrimaryCtxRetain(&ctx, 0) != CUDA_SUCCESS ||
            cuCtxSetCurrent(ctx) != CUDA_SUCCESS)
            return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    auto* h = new cublasContext;
    h->ctx = ctx;
    *handle = h;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDestroy_v2(cublasHandle_t handle) {
    VULKANIZER_TRACE_API();
    if (!handle) return CUBLAS_STATUS_NOT_INITIALIZED;
    delete handle;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetVersion_v2(cublasHandle_t, int* version) {
    if (!version) return CUBLAS_STATUS_INVALID_VALUE;
    *version = CUBLAS_VERSION;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetProperty(libraryPropertyType type, int* value) {
    if (!value) return CUBLAS_STATUS_INVALID_VALUE;
    *value = type == MAJOR_VERSION ? CUBLAS_VER_MAJOR : type == MINOR_VERSION ? CUBLAS_VER_MINOR : CUBLAS_VER_PATCH;
    return CUBLAS_STATUS_SUCCESS;
}

size_t cublasGetCudartVersion(void) { return 12090; }

cublasStatus_t cublasSetStream_v2(cublasHandle_t h, cudaStream_t stream) {
    if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
    h->stream = stream;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetStream_v2(cublasHandle_t h, cudaStream_t* stream) {
    if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (!stream) return CUBLAS_STATUS_INVALID_VALUE;
    *stream = h->stream;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetWorkspace_v2(cublasHandle_t h, void*, size_t) {
    return h ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_NOT_INITIALIZED;  // not needed by these kernels
}

cublasStatus_t cublasSetPointerMode_v2(cublasHandle_t h, cublasPointerMode_t mode) {
    if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
    h->pointerMode = mode;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetPointerMode_v2(cublasHandle_t h, cublasPointerMode_t* mode) {
    if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (!mode) return CUBLAS_STATUS_INVALID_VALUE;
    *mode = h->pointerMode;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetSmCountTarget(cublasHandle_t h, int) {
    return h ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_NOT_INITIALIZED;  // a hint without effect here
}

cublasStatus_t cublasSetMathMode(cublasHandle_t h, cublasMath_t mode) {
    if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
    h->math = mode;  // accumulation is always f32
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetMathMode(cublasHandle_t h, cublasMath_t* mode) {
    if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (!mode) return CUBLAS_STATUS_INVALID_VALUE;
    *mode = h->math;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetAtomicsMode(cublasHandle_t h, cublasAtomicsMode_t mode) {
    if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
    h->atomics = mode;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetAtomicsMode(cublasHandle_t h, cublasAtomicsMode_t* mode) {
    if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (!mode) return CUBLAS_STATUS_INVALID_VALUE;
    *mode = h->atomics;
    return CUBLAS_STATUS_SUCCESS;
}

const char* cublasGetStatusName(cublasStatus_t status) {
    switch (status) {
    case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return "CUBLAS_STATUS_UNKNOWN";
}

const char* cublasGetStatusString(cublasStatus_t status) {
    switch (status) {
    case CUBLAS_STATUS_SUCCESS: return "success";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "the library was not initialized";
    case CUBLAS_STATUS_ALLOC_FAILED: return "the resource allocation failed";
    case CUBLAS_STATUS_INVALID_VALUE: return "an invalid numeric value was used as an argument";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "an absent device architectural feature is required";
    case CUBLAS_STATUS_MAPPING_ERROR: return "an access to GPU memory space failed";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "the GPU program failed to execute";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "an internal operation failed";
    case CUBLAS_STATUS_NOT_SUPPORTED: return "the functionality requested is not supported";
    case CUBLAS_STATUS_LICENSE_ERROR: return "license error";
    }
    return "unknown error";
}

cublasStatus_t cublasLoggerConfigure(int, int, int, const char*) { return CUBLAS_STATUS_SUCCESS; }

cublasStatus_t cublasSgemm_v2(cublasHandle_t h, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
                              int k, const float* alpha, const float* A, int lda, const float* B, int ldb,
                              const float* beta, float* C, int ldc) {
    VULKANIZER_TRACE_API();
    return run(h, {transa, transb, m, n, k, alpha, A, CUDA_R_32F, lda, 0, B, CUDA_R_32F, ldb, 0, beta, C, CUDA_R_32F,
                   ldc, 0, 1, false, CUBLAS_COMPUTE_32F});
}

cublasStatus_t cublasSgemmBatched(cublasHandle_t h, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
                                  int k, const float* alpha, const float* const A[], int lda, const float* const B[],
                                  int ldb, const float* beta, float* const C[], int ldc, int batchCount) {
    VULKANIZER_TRACE_API();
    return run(h, {transa, transb, m, n, k, alpha, A, CUDA_R_32F, lda, 0, B, CUDA_R_32F, ldb, 0, beta,
                   pointerArray(C), CUDA_R_32F, ldc, 0, batchCount, true, CUBLAS_COMPUTE_32F});
}

cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t h, cublasOperation_t transa, cublasOperation_t transb, int m,
                                         int n, int k, const float* alpha, const float* A, int lda, long long strideA,
                                         const float* B, int ldb, long long strideB, const float* beta, float* C,
                                         int ldc, long long strideC, int batchCount) {
    VULKANIZER_TRACE_API();
    return run(h, {transa, transb, m, n, k, alpha, A, CUDA_R_32F, lda, strideA, B, CUDA_R_32F, ldb, strideB, beta, C,
                   CUDA_R_32F, ldc, strideC, batchCount, false, CUBLAS_COMPUTE_32F});
}

cublasStatus_t cublasDgemm_v2(cublasHandle_t h, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
                              int k, const double* alpha, const double* A, int lda, const double* B, int ldb,
                              const double* beta, double* C, int ldc) {
    VULKANIZER_TRACE_API();
    return run(h, {transa, transb, m, n, k, alpha, A, CUDA_R_64F, lda, 0, B, CUDA_R_64F, ldb, 0, beta, C, CUDA_R_64F,
                   ldc, 0, 1, false, CUBLAS_COMPUTE_64F});
}

cublasStatus_t cublasDgemmBatched(cublasHandle_t h, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
                                  int k, const double* alpha, const double* const A[], int lda,
                                  const double* const B[], int ldb, const double* beta, double* const C[], int ldc,
                                  int batchCount) {
    VULKANIZER_TRACE_API();
    return run(h, {transa, transb, m, n, k, alpha, A, CUDA_R_64F, lda, 0, B, CUDA_R_64F, ldb, 0, beta,
                   pointerArray(C), CUDA_R_64F, ldc, 0, batchCount, true, CUBLAS_COMPUTE_64F});
}

cublasStatus_t cublasDgemmStridedBatched(cublasHandle_t h, cublasOperation_t transa, cublasOperation_t transb, int m,
                                         int n, int k, const double* alpha, const double* A, int lda,
                                         long long strideA, const double* B, int ldb, long long strideB,
                                         const double* beta, double* C, int ldc, long long strideC, int batchCount) {
    VULKANIZER_TRACE_API();
    return run(h, {transa, transb, m, n, k, alpha, A, CUDA_R_64F, lda, strideA, B, CUDA_R_64F, ldb, strideB, beta, C,
                   CUDA_R_64F, ldc, strideC, batchCount, false, CUBLAS_COMPUTE_64F});
}

cublasStatus_t cublasGemmEx(cublasHandle_t h, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
                            const void* alpha, const void* A, cudaDataType Atype, int lda, const void* B,
                            cudaDataType Btype, int ldb, const void* beta, void* C, cudaDataType Ctype, int ldc,
                            cublasComputeType_t computeType, cublasGemmAlgo_t) {
    VULKANIZER_TRACE_API();
    return run(h, {transa, transb, m, n, k, alpha, A, Atype, lda, 0, B, Btype, ldb, 0, beta, C, Ctype, ldc, 0, 1,
                   false, computeType});
}

// Half precision GEMM (MAGMA links it): GemmEx with half matrices and half scalars.
cublasStatus_t cublasHgemm(cublasHandle_t h, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
                           const void* alpha, const void* A, int lda, const void* B, int ldb, const void* beta, void* C,
                           int ldc) {
    VULKANIZER_TRACE_API();
    return run(h, {transa, transb, m, n, k, alpha, A, CUDA_R_16F, lda, 0, B, CUDA_R_16F, ldb, 0, beta, C, CUDA_R_16F, ldc,
                   0, 1, false, CUBLAS_COMPUTE_16F});
}

cublasStatus_t cublasGemmBatchedEx(cublasHandle_t h, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
                                   int k, const void* alpha, const void* const Aarray[], cudaDataType Atype, int lda,
                                   const void* const Barray[], cudaDataType Btype, int ldb, const void* beta,
                                   void* const Carray[], cudaDataType Ctype, int ldc, int batchCount,
                                   cublasComputeType_t computeType, cublasGemmAlgo_t) {
    VULKANIZER_TRACE_API();
    return run(h, {transa, transb, m, n, k, alpha, Aarray, Atype, lda, 0, Barray, Btype, ldb, 0, beta,
                   pointerArray(Carray), Ctype, ldc, 0, batchCount, true, computeType});
}

// y = alpha op(A) x + beta y, as the GEMM y^T = x^T op(A)^T (m = 1): the vector strides become
// leading dimensions, and the matrix-vector kernel does the work.
cublasStatus_t cublasSgemv_v2(cublasHandle_t h, cublasOperation_t trans, int m, int n, const float* alpha,
                              const float* A, int lda, const float* x, int incx, const float* beta, float* y,
                              int incy) {
    VULKANIZER_TRACE_API();
    if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (m < 0 || n < 0 || lda < std::max(1, m) || incx == 0 || incy == 0) return CUBLAS_STATUS_INVALID_VALUE;
    if (incx < 0 || incy < 0) {
        VULKANIZER_WARN_ONCE("cublasSgemv: negative vector increments are not supported");
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    const bool t = trans != CUBLAS_OP_N;
    const int rows = t ? n : m, cols = t ? m : n;  // op(A) is rows x cols
    if (rows == 0) return CUBLAS_STATUS_SUCCESS;
    // op(A)^T as the GEMM's B (k = cols, n = rows): A^T stored as A, or A itself
    return run(h, {CUBLAS_OP_N, t ? CUBLAS_OP_N : CUBLAS_OP_T, 1, rows, cols, alpha, x, CUDA_R_32F, incx, 0, A,
                   CUDA_R_32F, lda, 0, beta, y, CUDA_R_32F, incy, 0, 1, false, CUBLAS_COMPUTE_32F});
}

// The 64-bit forms: the kernels index with 32 bits, so larger sizes are not supported.
static bool fitsInt(std::initializer_list<int64_t> values) {
    for (int64_t v : values)
        if (v < INT32_MIN || v > INT32_MAX) return false;
    return true;
}

cublasStatus_t cublasSgemm_v2_64(cublasHandle_t h, cublasOperation_t transa, cublasOperation_t transb, int64_t m,
                                 int64_t n, int64_t k, const float* alpha, const float* A, int64_t lda,
                                 const float* B, int64_t ldb, const float* beta, float* C, int64_t ldc) {
    if (!fitsInt({m, n, k, lda, ldb, ldc})) return CUBLAS_STATUS_NOT_SUPPORTED;
    return cublasSgemm_v2(h, transa, transb, int(m), int(n), int(k), alpha, A, int(lda), B, int(ldb), beta, C,
                          int(ldc));
}

cublasStatus_t cublasSgemmStridedBatched_64(cublasHandle_t h, cublasOperation_t transa, cublasOperation_t transb,
                                            int64_t m, int64_t n, int64_t k, const float* alpha, const float* A,
                                            int64_t lda, long long strideA, const float* B, int64_t ldb,
                                            long long strideB, const float* beta, float* C, int64_t ldc,
                                            long long strideC, int64_t batchCount) {
    if (!fitsInt({m, n, k, lda, ldb, ldc, batchCount})) return CUBLAS_STATUS_NOT_SUPPORTED;
    return cublasSgemmStridedBatched(h, transa, transb, int(m), int(n), int(k), alpha, A, int(lda), strideA, B,
                                     int(ldb), strideB, beta, C, int(ldc), strideC, int(batchCount));
}

cublasStatus_t cublasDgemm_v2_64(cublasHandle_t h, cublasOperation_t transa, cublasOperation_t transb, int64_t m,
                                 int64_t n, int64_t k, const double* alpha, const double* A, int64_t lda,
                                 const double* B, int64_t ldb, const double* beta, double* C, int64_t ldc) {
    if (!fitsInt({m, n, k, lda, ldb, ldc})) return CUBLAS_STATUS_NOT_SUPPORTED;
    return cublasDgemm_v2(h, transa, transb, int(m), int(n), int(k), alpha, A, int(lda), B, int(ldb), beta, C,
                          int(ldc));
}

cublasStatus_t cublasDgemmStridedBatched_64(cublasHandle_t h, cublasOperation_t transa, cublasOperation_t transb,
                                            int64_t m, int64_t n, int64_t k, const double* alpha, const double* A,
                                            int64_t lda, long long strideA, const double* B, int64_t ldb,
                                            long long strideB, const double* beta, double* C, int64_t ldc,
                                            long long strideC, int64_t batchCount) {
    if (!fitsInt({m, n, k, lda, ldb, ldc, batchCount})) return CUBLAS_STATUS_NOT_SUPPORTED;
    return cublasDgemmStridedBatched(h, transa, transb, int(m), int(n), int(k), alpha, A, int(lda), strideA, B,
                                     int(ldb), strideB, beta, C, int(ldc), strideC, int(batchCount));
}

cublasStatus_t cublasSgemv_v2_64(cublasHandle_t h, cublasOperation_t trans, int64_t m, int64_t n, const float* alpha,
                                 const float* A, int64_t lda, const float* x, int64_t incx, const float* beta,
                                 float* y, int64_t incy) {
    if (!fitsInt({m, n, lda, incx, incy})) return CUBLAS_STATUS_NOT_SUPPORTED;
    return cublasSgemv_v2(h, trans, int(m), int(n), alpha, A, int(lda), x, int(incx), beta, y, int(incy));
}

cublasStatus_t cublasSgemmEx(cublasHandle_t h, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
                             const float* alpha, const void* A, cudaDataType Atype, int lda, const void* B,
                             cudaDataType Btype, int ldb, const float* beta, void* C, cudaDataType Ctype, int ldc) {
    VULKANIZER_TRACE_API();
    return run(h, {transa, transb, m, n, k, alpha, A, Atype, lda, 0, B, Btype, ldb, 0, beta, C, Ctype, ldc, 0, 1,
                   false, CUBLAS_COMPUTE_32F});
}

// Groups of pointer-array batches, each group with its own shape, scalars and leading dimensions
// (host arrays); the matrix pointer arrays list every problem, group after group.
cublasStatus_t cublasGemmGroupedBatchedEx(cublasHandle_t h, const cublasOperation_t transa_array[],
                                          const cublasOperation_t transb_array[], const int m_array[],
                                          const int n_array[], const int k_array[], const void* alpha_array,
                                          const void* const Aarray[], cudaDataType Atype, const int lda_array[],
                                          const void* const Barray[], cudaDataType Btype, const int ldb_array[],
                                          const void* beta_array, void* const Carray[], cudaDataType Ctype,
                                          const int ldc_array[], int group_count, const int group_size[],
                                          cublasComputeType_t computeType) {
    VULKANIZER_TRACE_API();
    if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (group_count < 0) return CUBLAS_STATUS_INVALID_VALUE;
    if (group_count && (!transa_array || !transb_array || !m_array || !n_array || !k_array || !alpha_array ||
                        !lda_array || !ldb_array || !beta_array || !ldc_array || !group_size))
        return CUBLAS_STATUS_INVALID_VALUE;
    const size_t scalarBytes = computeType == CUBLAS_COMPUTE_16F || computeType == CUBLAS_COMPUTE_16F_PEDANTIC ? 2
                               : computeType == CUBLAS_COMPUTE_64F || computeType == CUBLAS_COMPUTE_64F_PEDANTIC ? 8
                                                                                                         : 4;
    size_t first = 0;  // index of the group's first problem in the pointer arrays
    for (int g = 0; g < group_count; ++g) {
        if (group_size[g] < 0) return CUBLAS_STATUS_INVALID_VALUE;
        const void* alpha = static_cast<const char*>(alpha_array) + g * scalarBytes;
        const void* beta = static_cast<const char*>(beta_array) + g * scalarBytes;
        if (cublasStatus_t st = run(h, {transa_array[g], transb_array[g], m_array[g], n_array[g], k_array[g], alpha,
                                        Aarray + first, Atype, lda_array[g], 0, Barray + first, Btype, ldb_array[g], 0,
                                        beta, pointerArray(Carray + first), Ctype, ldc_array[g], 0, group_size[g],
                                        true, computeType}))
            return st;
        first += size_t(group_size[g]);
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t h, cublasOperation_t transa, cublasOperation_t transb, int m,
                                          int n, int k, const void* alpha, const void* A, cudaDataType Atype, int lda,
                                          long long strideA, const void* B, cudaDataType Btype, int ldb,
                                          long long strideB, const void* beta, void* C, cudaDataType Ctype, int ldc,
                                          long long strideC, int batchCount, cublasComputeType_t computeType,
                                          cublasGemmAlgo_t) {
    VULKANIZER_TRACE_API();
    return run(h, {transa, transb, m, n, k, alpha, A, Atype, lda, strideA, B, Btype, ldb, strideB, beta, C, Ctype, ldc,
                   strideC, batchCount, false, computeType});
}

cublasStatus_t cublasStrsmBatched(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo,
                                  cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float* alpha,
                                  const float* const A[], int lda, float* const B[], int ldb, int batchCount) {
    VULKANIZER_TRACE_API();
    if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
    const int order = side == CUBLAS_SIDE_LEFT ? m : n;
    if (m < 0 || n < 0 || batchCount < 0 || lda < std::max(1, order) || ldb < std::max(1, m) ||
        (uplo != CUBLAS_FILL_MODE_LOWER && uplo != CUBLAS_FILL_MODE_UPPER) || trans > CUBLAS_OP_C)
        return CUBLAS_STATUS_INVALID_VALUE;
    if (m == 0 || n == 0 || batchCount == 0) return CUBLAS_STATUS_SUCCESS;
    TrsmParams p{};
    if (!scalar(h, alpha, CUBLAS_COMPUTE_32F, p.alpha)) return CUBLAS_STATUS_INVALID_VALUE;
    p.a = reinterpret_cast<uint64_t>(A);
    p.b = reinterpret_cast<uint64_t>(B);
    p.m = m, p.n = n, p.lda = lda, p.ldb = ldb;
    p.flags = (side == CUBLAS_SIDE_LEFT ? 1u : 0u) | (uplo == CUBLAS_FILL_MODE_LOWER ? 2u : 0u) |
              (trans != CUBLAS_OP_N ? 4u : 0u) | (diag == CUBLAS_DIAG_UNIT ? 8u : 0u);
    if (!h->trsm &&
        vulkanizerCreateSpirvFunction(&h->trsm, kTrsmSpirv, sizeof kTrsmSpirv / 4, "main", sizeof(TrsmParams)) != CUDA_SUCCESS)
        return CUBLAS_STATUS_INTERNAL_ERROR;
    const unsigned rhs = unsigned(side == CUBLAS_SIDE_LEFT ? n : m), block = 64;
    for (int first = 0; first < batchCount; first += int(kMaxGridZ)) {
        p.batchOffset = first;
        size_t size = sizeof p;
        void* extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, &p, CU_LAUNCH_PARAM_BUFFER_SIZE, &size, CU_LAUNCH_PARAM_END};
        const unsigned gz = unsigned(std::min<int>(batchCount - first, int(kMaxGridZ)));
        if (cuLaunchKernel(h->trsm, (rhs + block - 1) / block, 1, gz, block, 1, 1, 0, h->stream, nullptr, extra) !=
            CUDA_SUCCESS)
            return CUBLAS_STATUS_EXECUTION_FAILED;
    }
    return CUBLAS_STATUS_SUCCESS;
}

}  // extern "C"

// ---- batched LU (src/linalg), as PyTorch's torch.linalg uses for batches of small matrices ----

namespace {

cublasStatus_t getrfBatched(cublasHandle_t h, bool f64, int n, const void* A, int lda, int* P, int* info, int batch) {
    if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (n < 0 || lda < (n > 1 ? n : 1) || batch < 0) return CUBLAS_STATUS_INVALID_VALUE;
    if (f64 && !vulkanizer::linalg::hasFloat64()) return CUBLAS_STATUS_NOT_SUPPORTED;
    const vulkanizer::linalg::Matrices a{CUdeviceptr(A), true, 0, lda};
    return vulkanizer::linalg::getrf(h->stream, f64, n, n, a, CUdeviceptr(P), n, CUdeviceptr(info), batch)
               ? CUBLAS_STATUS_EXECUTION_FAILED
               : CUBLAS_STATUS_SUCCESS;
}

// `info` is a host pointer here: it reports invalid arguments only.
cublasStatus_t getrsBatched(cublasHandle_t h, bool f64, cublasOperation_t trans, int n, int nrhs, const void* A,
                            int lda, const int* P, const void* B, int ldb, int* info, int batch) {
    if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
    int bad = 0;
    if (n < 0) bad = -2;
    else if (nrhs < 0) bad = -3;
    else if (lda < (n > 1 ? n : 1)) bad = -5;
    else if (ldb < (n > 1 ? n : 1)) bad = -8;
    else if (batch < 0) bad = -10;
    if (info) *info = bad;
    if (bad) return CUBLAS_STATUS_INVALID_VALUE;
    if (f64 && !vulkanizer::linalg::hasFloat64()) return CUBLAS_STATUS_NOT_SUPPORTED;
    const vulkanizer::linalg::Matrices a{CUdeviceptr(A), true, 0, lda}, b{CUdeviceptr(B), true, 0, ldb};
    return vulkanizer::linalg::getrs(h->stream, f64, trans != CUBLAS_OP_N, n, nrhs, a, CUdeviceptr(P), n, b, batch)
               ? CUBLAS_STATUS_EXECUTION_FAILED
               : CUBLAS_STATUS_SUCCESS;
}

}  // namespace

extern "C" {

cublasStatus_t cublasSgetrfBatched(cublasHandle_t h, int n, float* const A[], int lda, int* P, int* info, int batchSize) {
    VULKANIZER_TRACE_API();
    return getrfBatched(h, false, n, A, lda, P, info, batchSize);
}
cublasStatus_t cublasDgetrfBatched(cublasHandle_t h, int n, double* const A[], int lda, int* P, int* info, int batchSize) {
    VULKANIZER_TRACE_API();
    return getrfBatched(h, true, n, A, lda, P, info, batchSize);
}
cublasStatus_t cublasSgetrsBatched(cublasHandle_t h, cublasOperation_t trans, int n, int nrhs, const float* const A[],
                                   int lda, const int* devIpiv, float* const B[], int ldb, int* info, int batchSize) {
    VULKANIZER_TRACE_API();
    return getrsBatched(h, false, trans, n, nrhs, A, lda, devIpiv, B, ldb, info, batchSize);
}
cublasStatus_t cublasDgetrsBatched(cublasHandle_t h, cublasOperation_t trans, int n, int nrhs, const double* const A[],
                                   int lda, const int* devIpiv, double* const B[], int ldb, int* info, int batchSize) {
    VULKANIZER_TRACE_API();
    return getrsBatched(h, true, trans, n, nrhs, A, lda, devIpiv, B, ldb, info, batchSize);
}

}  // extern "C"

// ---- triangular solves and batched QR on the shared kernels (src/linalg) ----------------------

namespace {

template <class T>
cublasStatus_t trsm(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans,
                    cublasDiagType_t diag, int m, int n, const T* alpha, const void* A, int lda, const void* B, int ldb,
                    int batch, bool arrays) {
    if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
    const int order = side == CUBLAS_SIDE_LEFT ? m : n;
    if (m < 0 || n < 0 || batch < 0 || lda < (order > 1 ? order : 1) || ldb < (m > 1 ? m : 1))
        return CUBLAS_STATUS_INVALID_VALUE;
    constexpr bool f64 = sizeof(T) == 8;
    if (f64 && !vulkanizer::linalg::hasFloat64()) return CUBLAS_STATUS_NOT_SUPPORTED;
    double a;
    if (!readAlpha(h, alpha, a)) return CUBLAS_STATUS_INVALID_VALUE;
    const vulkanizer::linalg::Matrices am{CUdeviceptr(A), arrays, 0, lda}, bm{CUdeviceptr(B), arrays, 0, ldb};
    return vulkanizer::linalg::trsm(h->stream, f64, side == CUBLAS_SIDE_RIGHT, uplo == CUBLAS_FILL_MODE_UPPER,
                                    trans != CUBLAS_OP_N, diag == CUBLAS_DIAG_UNIT, m, n, a, am, bm, batch)
               ? CUBLAS_STATUS_EXECUTION_FAILED
               : CUBLAS_STATUS_SUCCESS;
}

// `info` is a host pointer: it reports invalid arguments only.
cublasStatus_t geqrfBatched(cublasHandle_t h, bool f64, int m, int n, const void* A, int lda, const void* tau,
                            int* info, int batch) {
    if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
    int bad = 0;
    if (m < 0) bad = -1;
    else if (n < 0) bad = -2;
    else if (lda < (m > 1 ? m : 1)) bad = -4;
    else if (batch < 0) bad = -7;
    if (info) *info = bad;
    if (bad) return CUBLAS_STATUS_INVALID_VALUE;
    if (f64 && !vulkanizer::linalg::hasFloat64()) return CUBLAS_STATUS_NOT_SUPPORTED;
    const vulkanizer::linalg::Matrices a{CUdeviceptr(A), true, 0, lda};
    return vulkanizer::linalg::geqrf(h->stream, f64, m, n, a, CUdeviceptr(tau), 0, batch, true)
               ? CUBLAS_STATUS_EXECUTION_FAILED
               : CUBLAS_STATUS_SUCCESS;
}

}  // namespace

extern "C" {

cublasStatus_t cublasStrsm_v2(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans,
                              cublasDiagType_t diag, int m, int n, const float* alpha, const float* A, int lda,
                              float* B, int ldb) {
    VULKANIZER_TRACE_API();
    return trsm(h, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, 1, false);
}
cublasStatus_t cublasDtrsm_v2(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans,
                              cublasDiagType_t diag, int m, int n, const double* alpha, const double* A, int lda,
                              double* B, int ldb) {
    VULKANIZER_TRACE_API();
    return trsm(h, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, 1, false);
}
cublasStatus_t cublasDtrsmBatched(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo,
                                  cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double* alpha,
                                  const double* const A[], int lda, double* const B[], int ldb, int batchCount) {
    VULKANIZER_TRACE_API();
    return trsm(h, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount, true);
}
cublasStatus_t cublasSgeqrfBatched(cublasHandle_t h, int m, int n, float* const A[], int lda, float* const tau[],
                                   int* info, int batchSize) {
    VULKANIZER_TRACE_API();
    return geqrfBatched(h, false, m, n, A, lda, tau, info, batchSize);
}
cublasStatus_t cublasDgeqrfBatched(cublasHandle_t h, int m, int n, double* const A[], int lda, double* const tau[],
                                   int* info, int batchSize) {
    VULKANIZER_TRACE_API();
    return geqrfBatched(h, true, m, n, A, lda, tau, info, batchSize);
}

}  // extern "C"
