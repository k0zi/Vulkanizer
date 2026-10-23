#include "linalg/linalg.h"

#include <vulkanizer/vulkanizer_ext.h>

#include "common/log.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <map>
#include <mutex>

namespace vulkanizer::linalg {

namespace {

// SPIR-V of every kernel, f32 and f64 (-DDOUBLE).
const uint32_t kLuPanel32[] = {
#include "lu_panel.comp.inc"
};
const uint32_t kLuPanel64[] = {
#include "lu_panel_f64.comp.inc"
};
const uint32_t kLuUpdate32[] = {
#include "lu_update.comp.inc"
};
const uint32_t kLuUpdate64[] = {
#include "lu_update_f64.comp.inc"
};
const uint32_t kTriSolve32[] = {
#include "tri_solve.comp.inc"
};
const uint32_t kTriSolve64[] = {
#include "tri_solve_f64.comp.inc"
};
const uint32_t kCholPanel32[] = {
#include "chol_panel.comp.inc"
};
const uint32_t kCholPanel64[] = {
#include "chol_panel_f64.comp.inc"
};
const uint32_t kCholUpdate32[] = {
#include "chol_update.comp.inc"
};
const uint32_t kCholUpdate64[] = {
#include "chol_update_f64.comp.inc"
};
const uint32_t kQrPanel32[] = {
#include "qr_panel.comp.inc"
};
const uint32_t kQrPanel64[] = {
#include "qr_panel_f64.comp.inc"
};
const uint32_t kHouseholderApply32[] = {
#include "householder_apply.comp.inc"
};
const uint32_t kHouseholderApply64[] = {
#include "householder_apply_f64.comp.inc"
};
const uint32_t kOrgqrColumn32[] = {
#include "orgqr_column.comp.inc"
};
const uint32_t kOrgqrColumn64[] = {
#include "orgqr_column_f64.comp.inc"
};
const uint32_t kEigSetup32[] = {
#include "eig_setup.comp.inc"
};
const uint32_t kEigSetup64[] = {
#include "eig_setup_f64.comp.inc"
};
const uint32_t kJacobiRound32[] = {
#include "jacobi_round.comp.inc"
};
const uint32_t kJacobiRound64[] = {
#include "jacobi_round_f64.comp.inc"
};
const uint32_t kEigFinish32[] = {
#include "eig_finish.comp.inc"
};
const uint32_t kEigFinish64[] = {
#include "eig_finish_f64.comp.inc"
};
const uint32_t kSvdSetup32[] = {
#include "svd_setup.comp.inc"
};
const uint32_t kSvdSetup64[] = {
#include "svd_setup_f64.comp.inc"
};
const uint32_t kSvdRound32[] = {
#include "svd_round.comp.inc"
};
const uint32_t kSvdRound64[] = {
#include "svd_round_f64.comp.inc"
};
const uint32_t kSvdFinish32[] = {
#include "svd_finish.comp.inc"
};
const uint32_t kSvdFinish64[] = {
#include "svd_finish_f64.comp.inc"
};
const uint32_t kSvdSign32[] = {
#include "svd_sign.comp.inc"
};
const uint32_t kSvdSign64[] = {
#include "svd_sign_f64.comp.inc"
};
const uint32_t kGemm32[] = {
#include "gemm.comp.inc"
};
const uint32_t kGemm64[] = {
#include "gemm_f64.comp.inc"
};
const uint32_t kTrd32[] = {
#include "trd.comp.inc"
};
const uint32_t kTrd64[] = {
#include "trd_f64.comp.inc"
};

// Push constants of every kernel (shaders/common.glsl, std430).
struct Params {
    uint64_t a, b;
    uint64_t ipiv, info;
    uint64_t tau;
    int64_t strideA, strideB;
    int64_t strideP;
    int32_t m, n, nrhs, k;
    int32_t lda, ldb;
    uint32_t flags;
    int32_t batchOffset;
    uint64_t alphaBits;
    uint64_t c;
    int64_t strideC;
    int32_t ldc, pad;
};
static_assert(sizeof(Params) == 128, "must match shaders/common.glsl");

constexpr uint32_t kPtrA = 1, kPtrB = 2, kUpper = 4, kTrans = 8, kUnit = 16, kNoPivot = 32, kPivotsBefore = 64,
                   kPivotsAfter = 128, kRight = 256, kPtrTau = 512, kScale = 1024, kNoVectors = 2048, kVt = 4096,
                   kNoLong = 8192, kNoShort = 16384, kPtrC = 32768, kTransB = 65536;
constexpr unsigned kMaxGridZ = 65535;

enum Kernel { LuPanel, LuUpdate, TriSolve, CholPanel, CholUpdate, QrPanel, HouseholderApply, OrgqrColumn, EigSetup, JacobiRound, EigFinish,
             SvdSetup, SvdRound, SvdFinish, SvdSign, Gemm, Trd, KernelCount };

struct Kernels {
    CUfunction f[KernelCount][2] = {};
};

CUresult function(Kernel kernel, bool f64, CUfunction& out) {
    static std::mutex mutex;
    static std::map<CUcontext, Kernels> cache;
    CUcontext ctx = nullptr;
    if (CUresult r = cuCtxGetCurrent(&ctx)) return r;
    if (!ctx) return CUDA_ERROR_INVALID_CONTEXT;
    std::lock_guard<std::mutex> lock(mutex);
    CUfunction& f = cache[ctx].f[kernel][f64];
    if (!f) {
        struct Code {
            const uint32_t* words;
            size_t count;
        };
        static const Code sources[KernelCount][2] = {
            {{kLuPanel32, std::size(kLuPanel32)}, {kLuPanel64, std::size(kLuPanel64)}},
            {{kLuUpdate32, std::size(kLuUpdate32)}, {kLuUpdate64, std::size(kLuUpdate64)}},
            {{kTriSolve32, std::size(kTriSolve32)}, {kTriSolve64, std::size(kTriSolve64)}},
            {{kCholPanel32, std::size(kCholPanel32)}, {kCholPanel64, std::size(kCholPanel64)}},
            {{kCholUpdate32, std::size(kCholUpdate32)}, {kCholUpdate64, std::size(kCholUpdate64)}},
            {{kQrPanel32, std::size(kQrPanel32)}, {kQrPanel64, std::size(kQrPanel64)}},
            {{kHouseholderApply32, std::size(kHouseholderApply32)}, {kHouseholderApply64, std::size(kHouseholderApply64)}},
            {{kOrgqrColumn32, std::size(kOrgqrColumn32)}, {kOrgqrColumn64, std::size(kOrgqrColumn64)}},
            {{kEigSetup32, std::size(kEigSetup32)}, {kEigSetup64, std::size(kEigSetup64)}},
            {{kJacobiRound32, std::size(kJacobiRound32)}, {kJacobiRound64, std::size(kJacobiRound64)}},
            {{kEigFinish32, std::size(kEigFinish32)}, {kEigFinish64, std::size(kEigFinish64)}},
            {{kSvdSetup32, std::size(kSvdSetup32)}, {kSvdSetup64, std::size(kSvdSetup64)}},
            {{kSvdRound32, std::size(kSvdRound32)}, {kSvdRound64, std::size(kSvdRound64)}},
            {{kSvdFinish32, std::size(kSvdFinish32)}, {kSvdFinish64, std::size(kSvdFinish64)}},
            {{kSvdSign32, std::size(kSvdSign32)}, {kSvdSign64, std::size(kSvdSign64)}},
            {{kGemm32, std::size(kGemm32)}, {kGemm64, std::size(kGemm64)}},
            {{kTrd32, std::size(kTrd32)}, {kTrd64, std::size(kTrd64)}},
        };
        const Code& s = sources[kernel][f64];
        if (CUresult r = vulkanizerCreateSpirvFunction(&f, s.words, s.count, "main",
                                                       sizeof(Params)))
            return r;
    }
    out = f;
    return CUDA_SUCCESS;
}

// One kernel over a grid of (x, y) workgroups per batch entry; batches beyond the grid limit go
// in several launches.
CUresult launch(CUstream stream, Kernel kernel, bool f64, unsigned gx, unsigned gy, unsigned bx, unsigned by,
                Params params, int batch) {
    if (!gx || !gy || batch <= 0) return CUDA_SUCCESS;
    CUfunction f;
    if (CUresult r = function(kernel, f64, f)) return r;
    for (int first = 0; first < batch; first += int(kMaxGridZ)) {
        params.batchOffset = first;
        size_t size = sizeof params;
        void* extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, &params, CU_LAUNCH_PARAM_BUFFER_SIZE, &size,
                         CU_LAUNCH_PARAM_END};
        const unsigned gz = unsigned(std::min<int>(batch - first, int(kMaxGridZ)));
        if (CUresult r = cuLaunchKernel(f, gx, gy, gz, bx, by, 1, 0, stream, nullptr, extra)) return r;
    }
    return CUDA_SUCCESS;
}

Params base(const Matrices& a, int m, int n) {
    Params p{};
    p.a = a.base;
    p.strideA = a.stride;
    p.lda = a.ld;
    p.m = m;
    p.n = n;
    p.flags = a.pointerArray ? kPtrA : 0;
    return p;
}

void setB(Params& p, const Matrices& b, int nrhs) {
    p.b = b.base;
    p.strideB = b.stride;
    p.ldb = b.ld;
    p.nrhs = nrhs;
    if (b.pointerArray) p.flags |= kPtrB;
}

// A scalar parameter's bits: f64, or f32 in the low word.
uint64_t scalarBits(bool f64, double v) {
    uint64_t bits = 0;
    if (f64) {
        std::memcpy(&bits, &v, sizeof v);
    } else {
        const float v32 = float(v);
        uint32_t b32;
        std::memcpy(&b32, &v32, sizeof b32);
        bits = b32;
    }
    return bits;
}

unsigned groups(int count, unsigned size) { return count > 0 ? unsigned((count + int(size) - 1) / int(size)) : 0; }

}  // namespace

bool hasFloat64() {
    int v = 0;
    return vulkanizerGetFeature(VULKANIZER_FEATURE_FLOAT64, &v) == CUDA_SUCCESS && v;
}

CUresult getrf(CUstream stream, bool f64, int m, int n, const Matrices& a, CUdeviceptr pivots, int64_t pivotStride,
               CUdeviceptr info, int batch) {
    if (info) {
        if (CUresult r = cuMemsetD32Async(info, 0, size_t(batch), stream)) return r;
    }
    Params p = base(a, m, n);
    p.ipiv = pivots;
    p.strideP = pivotStride;
    p.info = info;
    if (!pivots) p.flags |= kNoPivot;
    for (int k = 0; k < std::min(m, n); ++k) {
        p.k = k;
        if (CUresult r = launch(stream, LuPanel, f64, 1, 1, 256, 1, p, batch)) return r;
        if (CUresult r = launch(stream, LuUpdate, f64, groups(m - k - 1, 16), groups(n - k - 1, 16), 16, 16, p, batch))
            return r;
    }
    return CUDA_SUCCESS;
}

CUresult getrs(CUstream stream, bool f64, bool trans, int n, int nrhs, const Matrices& a, CUdeviceptr pivots,
               int64_t pivotStride, const Matrices& b, int batch) {
    Params p = base(a, n, n);
    setB(p, b, nrhs);
    p.ipiv = pivots;
    p.strideP = pivotStride;
    const uint32_t common = p.flags;
    // A = P L U. A X = B: L Y = P^T B, then U X = Y. A^T X = B: U^T Y = B, L^T Z = Y, X = P Z.
    p.flags = common | (trans ? kTrans | kUpper : kUnit | (pivots ? kPivotsBefore : 0));
    if (CUresult r = launch(stream, TriSolve, f64, unsigned(nrhs), 1, 256, 1, p, batch)) return r;
    p.flags = common | (trans ? kTrans | kUnit | (pivots ? kPivotsAfter : 0) : kUpper);
    return launch(stream, TriSolve, f64, unsigned(nrhs), 1, 256, 1, p, batch);
}

CUresult potrf(CUstream stream, bool f64, bool upper, int n, const Matrices& a, CUdeviceptr info, int batch) {
    if (CUresult r = cuMemsetD32Async(info, 0, size_t(batch), stream)) return r;
    Params p = base(a, n, n);
    p.info = info;
    if (upper) p.flags |= kUpper;
    for (int k = 0; k < n; ++k) {
        p.k = k;
        if (CUresult r = launch(stream, CholPanel, f64, 1, 1, 256, 1, p, batch)) return r;
        if (CUresult r = launch(stream, CholUpdate, f64, groups(n - k - 1, 16), groups(n - k - 1, 16), 16, 16, p, batch))
            return r;
    }
    return CUDA_SUCCESS;
}

CUresult potrs(CUstream stream, bool f64, bool upper, int n, int nrhs, const Matrices& a, const Matrices& b, int batch) {
    Params p = base(a, n, n);
    setB(p, b, nrhs);
    const uint32_t common = p.flags | (upper ? kUpper : 0);
    // lower: L Y = B, L^T X = Y; upper: U^T Y = B, U X = Y
    p.flags = common | (upper ? kTrans : 0);
    if (CUresult r = launch(stream, TriSolve, f64, unsigned(nrhs), 1, 256, 1, p, batch)) return r;
    p.flags = common | (upper ? 0 : kTrans);
    return launch(stream, TriSolve, f64, unsigned(nrhs), 1, 256, 1, p, batch);
}

CUresult geqrf(CUstream stream, bool f64, int m, int n, const Matrices& a, CUdeviceptr tau, int64_t tauStride, int batch,
               bool tauArray) {
    Params p = base(a, m, n);
    p.tau = tau;
    p.strideP = tauStride;
    if (tauArray) p.flags |= kPtrTau;
    setB(p, a, 0);  // the reflectors act on A's own trailing columns
    for (int k = 0; k < std::min(m, n); ++k) {
        p.k = k;
        p.nrhs = k + 1;
        if (CUresult r = launch(stream, QrPanel, f64, 1, 1, 256, 1, p, batch)) return r;
        if (CUresult r = launch(stream, HouseholderApply, f64, groups(n - k - 1, 1), 1, 256, 1, p, batch)) return r;
    }
    return CUDA_SUCCESS;
}

CUresult orgqr(CUstream stream, bool f64, int m, int n, int k, const Matrices& a, CUdeviceptr tau, int64_t tauStride,
               int batch) {
    Params p = base(a, m, n);
    p.tau = tau;
    p.strideP = tauStride;
    setB(p, a, 0);
    const uint32_t common = p.flags;
    // as LAPACK's dorg2r: columns k.. start as unit vectors, then H(i) for i = k-1 .. 0
    p.k = k;
    p.flags = common | kTrans;
    if (CUresult r = launch(stream, OrgqrColumn, f64, groups(n - k, 1), 1, 256, 1, p, batch)) return r;
    p.flags = common;
    for (int i = k - 1; i >= 0; --i) {
        p.k = i;
        p.nrhs = i + 1;
        if (CUresult r = launch(stream, HouseholderApply, f64, groups(n - i - 1, 1), 1, 256, 1, p, batch)) return r;
        if (CUresult r = launch(stream, OrgqrColumn, f64, 1, 1, 256, 1, p, batch)) return r;
    }
    return CUDA_SUCCESS;
}

CUresult ormqr(CUstream stream, bool f64, bool right, bool trans, int m, int n, int k, const Matrices& a,
               CUdeviceptr tau, int64_t tauStride, const Matrices& c, int batch) {
    Params p = base(a, m, n);
    p.tau = tau;
    p.strideP = tauStride;
    setB(p, c, 0);
    if (right) p.flags |= kRight;
    // Q = H(0) ... H(k-1): Q^T C and C Q apply H(0) first, Q C and C Q^T H(k-1) first.
    const bool forward = right != trans;
    for (int s = 0; s < k; ++s) {
        p.k = forward ? s : k - 1 - s;
        if (CUresult r = launch(stream, HouseholderApply, f64, unsigned(right ? m : n), 1, 256, 1, p, batch)) return r;
    }
    return CUDA_SUCCESS;
}

CUresult trsm(CUstream stream, bool f64, bool right, bool upper, bool trans, bool unit, int m, int n, double alpha,
              const Matrices& a, const Matrices& b, int batch) {
    // X op(A) = B as op(A)^T X^T = B^T: the rows of B are the right-hand sides
    const int order = right ? n : m, rhs = right ? m : n;
    Params p = base(a, order, order);
    setB(p, b, rhs);
    p.flags |= (upper ? kUpper : 0) | ((trans != right) ? kTrans : 0) | (unit ? kUnit : 0) | (right ? kRight : 0) |
               (alpha != 1 ? kScale : 0);
    p.alphaBits = scalarBits(f64, alpha);
    return launch(stream, TriSolve, f64, unsigned(rhs), 1, 256, 1, p, batch);
}

// Jacobi: S (n x n), then the n ranks and a pad; the tridiagonal path: S, then d, e, pv, wv and tau.
int64_t syevjWorkspace(int n) { return int64_t(n) * n + std::max<int64_t>(n + (n & 1), 5 * int64_t(n)); }

namespace {

// Eigenvalues of the symmetric tridiagonal matrix with diagonal d and subdiagonal e (e[i] between
// rows i and i + 1; e[n-1] unused) by the implicit QL method with Wilkinson shifts (as in EISPACK's
// tql2), in place of d, unsorted. With `rotations`, every plane rotation the eigenvectors need is
// appended in order: column i of (i, i + 1) to idx, (c, s) to cs; applied to the identity from the
// right (Z(:, i+1) = s Z(:, i) + c Z(:, i+1), Z(:, i) = c Z(:, i) - s Z(:, i+1)) they give the
// eigenvectors. False if an eigenvalue did not converge.
bool tridiagonalQl(std::vector<double>& d, std::vector<double>& e, bool rotations, std::vector<int32_t>& idx,
                   std::vector<double>& cs) {
    const int n = int(d.size());
    if (n == 0) return true;
    e[n - 1] = 0;
    constexpr double eps = 0x1p-52;
    double f = 0, tst1 = 0;
    for (int l = 0; l < n; ++l) {
        tst1 = std::max(tst1, std::fabs(d[l]) + std::fabs(e[l]));
        int m = l;
        while (m < n - 1 && std::fabs(e[m]) > eps * tst1) ++m;
        if (m > l) {
            for (int iter = 0;; ++iter) {
                if (iter == 30 * n) return false;
                double g = d[l], p = (d[l + 1] - g) / (2 * e[l]), r = std::hypot(p, 1.0);
                if (p < 0) r = -r;
                d[l] = e[l] / (p + r);
                d[l + 1] = e[l] * (p + r);
                const double dl1 = d[l + 1];
                double h = g - d[l];
                for (int i = l + 2; i < n; ++i) d[i] -= h;
                f += h;
                p = d[m];
                double c = 1, c2 = 1, c3 = 1, s = 0, s2 = 0;
                const double el1 = e[l + 1];
                for (int i = m - 1; i >= l; --i) {
                    c3 = c2;
                    c2 = c;
                    s2 = s;
                    g = c * e[i];
                    h = c * p;
                    r = std::hypot(p, e[i]);
                    e[i + 1] = s * r;
                    s = e[i] / r;
                    c = p / r;
                    p = c * d[i] - s * g;
                    d[i + 1] = h + s * (c * g + s * d[i]);
                    if (rotations) {
                        idx.push_back(i);
                        cs.push_back(c);
                        cs.push_back(s);
                    }
                }
                p = -s * s2 * c3 * el1 * e[l] / dl1;
                e[l] = s * p;
                d[l] = c * p;
                if (std::fabs(e[l]) <= eps * tst1) break;
            }
        }
        d[l] += f;
        e[l] = 0;
    }
    return true;
}

// One n x n matrix by Householder tridiagonalization on the device (trd.comp), the tridiagonal
// problem on the host (tridiagonalQl) and its rotations and Q back on the device: O(n) launches
// instead of the Jacobi sweeps' O(n) per sweep.
CUresult syevTridiagonal(CUstream stream, bool f64, bool upper, bool vectors, int n, const Matrices& a, CUdeviceptr w,
                         CUdeviceptr work) {
    const size_t rb = f64 ? 8 : 4;
    Params p = base(a, n, n);
    setB(p, Matrices{work, false, 0, n}, 0);
    const uint32_t common = p.flags | (upper ? kUpper : 0) | (vectors ? 0 : kNoVectors);
    p.flags = common;
    if (CUresult r = launch(stream, EigSetup, f64, unsigned(n), 1, 256, 1, p, 1)) return r;  // S
    p.flags = common | kTrans;
    if (CUresult r = launch(stream, EigSetup, f64, unsigned(n), 1, 256, 1, p, 1)) return r;  // A = I
    p.flags = common;
    auto step = [&](int mode, int k, unsigned groupsX) {
        p.nrhs = mode;
        p.k = k;
        return launch(stream, Trd, f64, groupsX, 1, 256, 1, p, 1);
    };
    for (int k = 0; k + 1 < n; ++k) {
        if (CUresult r = step(0, k, 1)) return r;
        if (k + 2 == n) break;  // nothing below S(k+1, k) to annihilate
        if (CUresult r = step(1, k, unsigned(n - k - 1))) return r;
        if (CUresult r = step(2, k, 1)) return r;
        if (CUresult r = step(3, k, unsigned(n - k - 1))) return r;
    }
    if (CUresult r = step(4, 0, groups(n, 256))) return r;

    // d and e (n each, after S) to the host
    const CUdeviceptr dAddr = work + CUdeviceptr(size_t(n) * n * rb);
    std::vector<uint8_t> raw(2 * size_t(n) * rb);
    if (CUresult r = cuMemcpyDtoHAsync(raw.data(), dAddr, raw.size(), stream)) return r;
    if (CUresult r = cuStreamSynchronize(stream)) return r;
    std::vector<double> d(n), e(n);
    for (int i = 0; i < n; ++i) {
        if (f64) {
            std::memcpy(&d[i], &raw[i * 8], 8);
            std::memcpy(&e[i], &raw[(n + i) * 8], 8);
        } else {
            float x, y;
            std::memcpy(&x, &raw[i * 4], 4);
            std::memcpy(&y, &raw[(n + i) * 4], 4);
            d[i] = x, e[i] = y;
        }
    }
    std::vector<int32_t> idx;
    std::vector<double> cs;
    if (!tridiagonalQl(d, e, vectors, idx, cs)) VULKANIZER_WARN("syevd: the QL iteration did not converge (n = %d)", n);

    if (vectors && !idx.empty()) {
        // Rotation chunks of at most ~2^26 element updates per launch (the GPU timeout).
        const size_t chunk = std::max<size_t>(1, (size_t(1) << 26) / size_t(n));
        const size_t count = std::min(chunk, idx.size());
        CUdeviceptr dIdx = 0, dCs = 0;
        if (CUresult r = cuMemAlloc(&dIdx, count * 4)) return r;
        if (CUresult r = cuMemAlloc(&dCs, 2 * count * rb)) {
            cuMemFree(dIdx);
            return r;
        }
        std::vector<uint8_t> csRaw(2 * count * rb);
        CUresult res = CUDA_SUCCESS;
        Params q = p;
        q.nrhs = 6;
        q.ipiv = dIdx;
        q.c = dCs;
        q.k = 0;
        for (size_t first = 0; first < idx.size() && res == CUDA_SUCCESS; first += count) {
            const size_t m = std::min(count, idx.size() - first);
            for (size_t t = 0; t < 2 * m; ++t) {
                if (f64) std::memcpy(&csRaw[t * 8], &cs[2 * first + t], 8);
                else {
                    const float v = float(cs[2 * first + t]);
                    std::memcpy(&csRaw[t * 4], &v, 4);
                }
            }
            // in queue order: the previous chunk's launch has read the buffers before they change
            if ((res = cuMemcpyHtoD(dIdx, idx.data() + first, m * 4)) != CUDA_SUCCESS) break;
            if ((res = cuMemcpyHtoD(dCs, csRaw.data(), 2 * m * rb)) != CUDA_SUCCESS) break;
            q.ldc = int(m);
            res = launch(stream, Trd, f64, groups(n, 256), 1, 256, 1, q, 1);
        }
        cuMemFree(dIdx);
        cuMemFree(dCs);
        if (res) return res;
        // A = Q Z: the reflectors are geqrf's one row down (rows 1.., of S and of A)
        if (n > 2) {
            if (CUresult r = ormqr(stream, f64, false, false, n - 1, n, n - 2, Matrices{work + rb, false, 0, n},
                                   work + CUdeviceptr(size_t(n) * n * rb + 4 * size_t(n) * rb), 0,
                                   Matrices{a.base + rb, false, 0, a.ld}, 1))
                return r;
        }
    }
    // the eigenvalues on S's diagonal, for eig_finish to sort (with the vectors)
    std::vector<uint8_t> dRaw(size_t(n) * rb);
    for (int i = 0; i < n; ++i) {
        if (f64) std::memcpy(&dRaw[i * 8], &d[i], 8);
        else {
            const float v = float(d[i]);
            std::memcpy(&dRaw[i * 4], &v, 4);
        }
    }
    if (CUresult r = cuMemcpyHtoD(dAddr, dRaw.data(), dRaw.size())) return r;
    if (CUresult r = step(5, 0, groups(n, 256))) return r;
    p.nrhs = 0;
    p.k = 0;
    p.tau = w;
    p.strideP = n;
    p.flags = common;
    return launch(stream, EigFinish, f64, 1, 1, 256, 1, p, 1);
}

}  // namespace

CUresult syevj(CUstream stream, bool f64, bool upper, bool vectors, int n, const Matrices& a, CUdeviceptr w,
               int64_t wStride, CUdeviceptr work, int64_t workStride, int batch) {
    if (n <= 0) return CUDA_SUCCESS;
    // One larger matrix: tridiagonalization (VULKANIZER_EIG_JACOBI=1 keeps the Jacobi iteration).
    static const bool jacobiOnly = [] {
        const char* env = std::getenv("VULKANIZER_EIG_JACOBI");
        return env && *env && std::strcmp(env, "0") != 0;
    }();
    if (batch == 1 && n >= 32 && !a.pointerArray && !jacobiOnly)
        return syevTridiagonal(stream, f64, upper, vectors, n, a, w, work);
    Params p = base(a, n, n);
    setB(p, Matrices{work, false, workStride, n}, 0);
    p.tau = w;
    p.strideP = wStride;
    const uint32_t common = p.flags | (upper ? kUpper : 0) | (vectors ? 0 : kNoVectors);
    p.flags = common;
    if (CUresult r = launch(stream, EigSetup, f64, unsigned(n), 1, 256, 1, p, batch)) return r;
    p.flags = common | kTrans;
    if (CUresult r = launch(stream, EigSetup, f64, unsigned(n), 1, 256, 1, p, batch)) return r;
    // Cyclic Jacobi converges quadratically: a fixed number of sweeps, rotations that would no
    // longer change anything are skipped by the kernel.
    const int count = n + (n & 1), sweeps = f64 ? 16 : 12;
    for (int sweep = 0; sweep < sweeps; ++sweep) {
        for (int round = 0; round < count - 1; ++round) {
            p.k = round;
            p.flags = common;
            if (CUresult r = launch(stream, JacobiRound, f64, unsigned(count / 2), 1, 256, 1, p, batch)) return r;
            p.flags = common | kRight;
            if (CUresult r = launch(stream, JacobiRound, f64, unsigned(count / 2), 1, 256, 1, p, batch)) return r;
        }
    }
    p.flags = common;
    return launch(stream, EigFinish, f64, 1, 1, 256, 1, p, batch);
}

int64_t gesvdjWorkspace(int m, int n) {
    const int64_t r = std::max(m, n), c = std::min(m, n);
    return r * c + c * c + 3 * c + 1;
}

CUresult gesvdj(CUstream stream, bool f64, int m, int n, const Matrices& a, CUdeviceptr s, int64_t sStride,
                bool wantU, const Matrices& u, bool wantV, const Matrices& v, bool vt, bool full, CUdeviceptr work,
                int64_t workStride, int batch) {
    if (m <= 0 || n <= 0) return CUDA_SUCCESS;
    // A = U S V^T is found as G = A (tall) or A^T (wide) = L S R^T, L r x c, R c x c
    const bool tall = m >= n;
    const int r = std::max(m, n), c = std::min(m, n);
    const bool wantLong = tall ? wantU : wantV, wantShort = tall ? wantV : wantU;
    if (vt && wantV && !tall) return CUDA_ERROR_NOT_SUPPORTED;
    Params p = base(a, r, c);
    setB(p, Matrices{work, false, workStride, r}, 0);
    p.tau = s;
    p.strideP = sStride;
    const uint32_t common = wantLong || wantShort ? 0 : kNoVectors;
    p.flags = (p.flags & kPtrA) | common | (tall ? 0 : kTrans);
    if (CUresult res = launch(stream, SvdSetup, f64, unsigned(c), 1, 256, 1, p, batch)) return res;
    p.flags = common;
    const int count = c + (c & 1), sweeps = f64 ? 16 : 12;
    for (int sweep = 0; sweep < sweeps; ++sweep) {
        for (int round = 0; round < count - 1; ++round) {
            p.k = round;
            if (CUresult res = launch(stream, SvdRound, f64, unsigned(count / 2), 1, 256, 1, p, batch)) return res;
        }
    }
    if (CUresult res = launch(stream, SvdFinish, f64, 1, 1, 256, 1, p, batch)) return res;
    if (!wantLong && !wantShort) return CUDA_SUCCESS;
    const Matrices& longSide = tall ? u : v;
    const Matrices& shortSide = tall ? v : u;
    p.a = longSide.base;
    p.strideA = longSide.stride;
    p.lda = longSide.ld;
    p.c = shortSide.base;
    p.strideC = shortSide.stride;
    p.ldc = shortSide.ld;
    p.flags = kTrans | (wantLong ? 0 : kNoLong) | (wantShort ? 0 : kNoShort) | (vt && tall ? kVt : 0);
    if (CUresult res = launch(stream, SvdFinish, f64, unsigned(c), 1, 256, 1, p, batch)) return res;
    if (!wantLong) return CUDA_SUCCESS;
    // The long side's vectors, zero for negligible singular values, completed to an orthonormal
    // basis (all r of them if full): L = Q R, Q's first columns are L's up to sign(R(j, j)).
    const CUdeviceptr tau = work + CUdeviceptr(int64_t(r) * c + int64_t(c) * c + c + 1) * (f64 ? 8 : 4);
    if (CUresult res = geqrf(stream, f64, r, c, longSide, tau, workStride, batch)) return res;
    p.flags = 0;
    if (CUresult res = launch(stream, SvdSign, f64, 1, 1, 256, 1, p, batch)) return res;
    if (CUresult res = orgqr(stream, f64, r, full ? r : c, c, longSide, tau, workStride, batch)) return res;
    p.flags = kTrans;
    return launch(stream, SvdSign, f64, unsigned(c), 1, 256, 1, p, batch);
}

CUresult gemm(CUstream stream, bool f64, bool transA, bool transB, int m, int n, int k, double alpha,
              const Matrices& a, const Matrices& b, double beta, const Matrices& c, int batch) {
    Params p = base(a, m, n);
    setB(p, b, 0);
    p.k = k;
    p.c = c.base;
    p.strideC = c.stride;
    p.ldc = c.ld;
    p.flags |= (transA ? kTrans : 0) | (transB ? kTransB : 0) | (c.pointerArray ? kPtrC : 0);
    p.alphaBits = scalarBits(f64, alpha);
    p.info = scalarBits(f64, beta);
    return launch(stream, Gemm, f64, groups(m, 16), groups(n, 16), 16, 16, p, batch);
}

}  // namespace vulkanizer::linalg
