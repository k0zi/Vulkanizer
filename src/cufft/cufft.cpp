// Vulkanizer's libcufft: complex and real FFTs of rank 1 to 3, batched, with cuFFT's advanced data
// layouts, in f32, f64 (where the device has it) and f16 (computed in f32), on Vulkanizer's own
// kernels (shaders/*.comp), queued on the plan's stream.
//
// A transform gathers the input into the full complex array in the workspace (shaders/common.glsl),
// transforms it along each axis with Stockham passes of radix 4, 2, 3, 5, 7, 11 and 13 (lengths with
// a larger prime factor go through Bluestein's algorithm, with power-of-two FFTs), and scatters the
// result to the output. Lengths up to 2048 (f64: 1024) take all their passes in one dispatch, in
// shared memory, several lines per workgroup; longer ones one dispatch per pass. Real transforms are done as complex ones of the full length. The tables
// (roots of unity, Bluestein's chirp and filter) are computed on the host in f64 when the plan is made.

#include <cufft/cufft.h>
#include <cuda.h>
#include <vulkanizer/vulkanizer_ext.h>

#include "common/log.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

namespace {

const uint32_t kGather32[] = {
#include "fft_gather.comp.inc"
};
const uint32_t kGather64[] = {
#include "fft_gather_f64.comp.inc"
};
const uint32_t kPass32[] = {
#include "fft_pass.comp.inc"
};
const uint32_t kPass64[] = {
#include "fft_pass_f64.comp.inc"
};
const uint32_t kLocal32[] = {
#include "fft_local.comp.inc"
};
const uint32_t kLocal64[] = {
#include "fft_local_f64.comp.inc"
};
const uint32_t kBluestein32[] = {
#include "fft_bluestein.comp.inc"
};
const uint32_t kBluestein64[] = {
#include "fft_bluestein_f64.comp.inc"
};
const uint32_t kScatter32[] = {
#include "fft_scatter.comp.inc"
};
const uint32_t kScatter64[] = {
#include "fft_scatter_f64.comp.inc"
};

// Push constants of every kernel (shaders/common.glsl, std430).
struct Params {
    uint64_t src, dst;
    uint64_t table, aux;
    int64_t dist;
    int64_t stride[3];
    int64_t items;
    int32_t n[3];
    int32_t len;
    int32_t radix, span;
    int32_t inner;
    uint32_t flags;
};
static_assert(sizeof(Params) == 104, "must match shaders/common.glsl");

constexpr uint32_t kF32 = 0, kF64 = 1, kF16 = 2;
constexpr uint32_t kModeShift = 2, kC2C = 0, kR2C = 1, kC2R = 2;
constexpr uint32_t kInverse = 16;
constexpr uint32_t kStepShift = 5, kPre = 0, kMultiply = 1, kPost = 2;
constexpr int kRadices[] = {4, 2, 3, 5, 7, 11, 13};  // in the order passes take them

enum Kernel { Gather, Pass, Local, Bluestein, Scatter, KernelCount };

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
            {{kGather32, std::size(kGather32)}, {kGather64, std::size(kGather64)}},
            {{kPass32, std::size(kPass32)}, {kPass64, std::size(kPass64)}},
            {{kLocal32, std::size(kLocal32)}, {kLocal64, std::size(kLocal64)}},
            {{kBluestein32, std::size(kBluestein32)}, {kBluestein64, std::size(kBluestein64)}},
            {{kScatter32, std::size(kScatter32)}, {kScatter64, std::size(kScatter64)}},
        };
        const Code& s = sources[kernel][f64];
        if (CUresult r = vulkanizerCreateSpirvFunction(&f, s.words, s.count, "main", sizeof(Params))) return r;
    }
    out = f;
    return CUDA_SUCCESS;
}

// p.items work items in 256-thread workgroups (Local: a workgroup per item), on a 2D grid.
CUresult launch(CUstream stream, Kernel kernel, bool f64, Params& p) {
    if (p.items <= 0) return CUDA_SUCCESS;
    CUfunction f;
    if (CUresult r = function(kernel, f64, f)) return r;
    const int64_t groups = kernel == Local ? p.items : (p.items + 255) / 256;
    const unsigned gx = unsigned(std::min<int64_t>(groups, 65535)), gy = unsigned((groups + gx - 1) / gx);
    size_t size = sizeof p;
    void* extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, &p, CU_LAUNCH_PARAM_BUFFER_SIZE, &size, CU_LAUNCH_PARAM_END};
    return cuLaunchKernel(f, gx, gy, 1, 256, 1, 1, 0, stream, nullptr, extra);
}

using Complex = std::complex<double>;

// The radices of a Stockham FFT of length n, or none if n has a prime factor above 13.
bool factor(int64_t n, std::vector<int>& radices) {
    radices.clear();
    for (int r : kRadices)
        while (n % r == 0) radices.push_back(r), n /= r;
    return n == 1;
}

// In-place radix-2 FFT of a power-of-two length (forward: exp(-2 pi i nk / N)), for Bluestein's filter.
void hostFft(std::vector<Complex>& a) {
    const size_t n = a.size();
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        for (size_t i = 0; i < n; i += len)
            for (size_t k = 0; k < len / 2; ++k) {
                const Complex w = std::polar(1.0, -2 * M_PI * double(k) / double(len));
                const Complex u = a[i + k], v = a[i + k + len / 2] * w;
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
            }
    }
}

std::vector<Complex> roots(int64_t n) {
    std::vector<Complex> t(static_cast<size_t>(n));
    for (int64_t m = 0; m < n; ++m) t[size_t(m)] = std::polar(1.0, -2 * M_PI * double(m) / double(n));
    return t;
}

// One axis of the transform.
struct Axis {
    int n = 1;
    std::vector<int> radices;  // of n, or (Bluestein) of m
    bool bluestein = false;
    int m = 0;                     // Bluestein: the power-of-two length
    size_t rootsOffset = 0;        // into the plan's table buffer, in complex elements: roots of n (or m)
    size_t chirpOffset = 0, filterOffset = 0;  // Bluestein: chirp (n), filter (2m: forward, inverse)
};

struct Layout {
    int64_t dist = 0;
    int64_t stride[3] = {};  // of the three padded dimensions, in elements
};

struct Plan {
    CUstream stream = nullptr;
    bool autoAllocate = true;
    bool made = false;
    CUdeviceptr work = 0, ownWork = 0;
    size_t workSize = 0;
    CUdeviceptr tables = 0;

    uint32_t mode = kC2C, inType = kF32, outType = kF32;
    bool f64 = false;
    int n[3] = {1, 1, 1};
    int64_t batch = 1;
    Axis axes[3];
    Layout in, out;
    bool basicLayout = false;  // in-place real transforms then use the padded real layout
    int64_t total = 0;          // of the full array

    ~Plan() {
        if (ownWork) cuMemFree(ownWork);
        if (tables) cuMemFree(tables);
    }
};

std::mutex g_mutex;
std::map<cufftHandle, std::unique_ptr<Plan>> g_plans;
cufftHandle g_next = 1;

Plan* find(cufftHandle h) {
    std::lock_guard<std::mutex> lock(g_mutex);
    const auto it = g_plans.find(h);
    return it == g_plans.end() ? nullptr : it->second.get();
}

// The element strides of a rank-r layout (embed: the storage dimensions, embed[0] unused) for the
// padded three dimensions.
Layout layout(int rank, const long long* embed, int64_t stride, int64_t dist) {
    Layout l;
    l.dist = dist;
    int64_t s = stride;
    for (int k = rank - 1, d = 2; k >= 0; --k, --d) {
        l.stride[d] = s;
        s *= embed[k];
    }
    return l;
}

cufftResult makePlan(Plan& plan, int rank, const long long* n, const long long* inembed, long long istride,
                     long long idist, const long long* onembed, long long ostride, long long odist, uint32_t mode,
                     uint32_t inType, uint32_t outType, bool f64, long long batch, size_t* workSize) {
    if (rank < 1 || rank > 3 || !n || batch < 1) return CUFFT_INVALID_VALUE;
    for (int k = 0; k < rank; ++k)
        if (n[k] < 1 || n[k] > INT32_MAX) return CUFFT_INVALID_SIZE;
    if (f64) {
        int v = 0;
        if (vulkanizerGetFeature(VULKANIZER_FEATURE_FLOAT64, &v) != CUDA_SUCCESS || !v) return CUFFT_NOT_SUPPORTED;
    }
    if (plan.tables) cuMemFree(plan.tables), plan.tables = 0;
    if (plan.ownWork) cuMemFree(plan.ownWork), plan.ownWork = 0;
    plan.mode = mode;
    plan.inType = inType;
    plan.outType = outType;
    plan.f64 = f64;
    plan.batch = batch;
    for (int d = 0; d < 3; ++d) plan.n[d] = 1;
    for (int k = 0; k < rank; ++k) plan.n[3 - rank + k] = int(n[k]);
    plan.total = batch * int64_t(plan.n[0]) * plan.n[1] * plan.n[2];
    if (plan.total >= INT32_MAX / 4) return CUFFT_NOT_SUPPORTED;  // the kernels index with 32 bits

    // layouts: the basic one for null embeds (strides 1, batch entries contiguous)
    const int64_t last = n[rank - 1], half = last / 2 + 1;
    std::vector<long long> realDims(n, n + rank), complexDims(n, n + rank);
    if (mode != kC2C) complexDims[rank - 1] = half;
    const auto& inDims = mode == kC2R ? complexDims : mode == kR2C ? realDims : complexDims;
    const auto& outDims = mode == kR2C ? complexDims : mode == kC2R ? realDims : complexDims;
    const auto product = [&](const std::vector<long long>& d) {
        int64_t p = 1;
        for (long long x : d) p *= x;
        return p;
    };
    plan.basicLayout = !inembed || !onembed;
    if (plan.basicLayout) {
        plan.in = layout(rank, inDims.data(), 1, product(inDims));
        plan.out = layout(rank, outDims.data(), 1, product(outDims));
    } else {
        plan.in = layout(rank, inembed, istride, idist);
        plan.out = layout(rank, onembed, ostride, odist);
    }

    // tables: roots for every axis length, chirp and filter for Bluestein's
    std::vector<Complex> table;
    size_t blueWork = 0;
    for (int d = 0; d < 3; ++d) {
        Axis& a = plan.axes[d];
        a = Axis{};
        a.n = plan.n[d];
        if (a.n == 1) continue;
        a.bluestein = !factor(a.n, a.radices);
        if (!a.bluestein) {
            a.rootsOffset = table.size();
            const auto r = roots(a.n);
            table.insert(table.end(), r.begin(), r.end());
            continue;
        }
        a.m = 1;
        while (a.m < 2 * a.n - 1) a.m <<= 1;
        factor(a.m, a.radices);
        a.rootsOffset = table.size();
        const auto r = roots(a.m);
        table.insert(table.end(), r.begin(), r.end());
        a.chirpOffset = table.size();
        std::vector<Complex> chirp(size_t(a.n));
        for (int64_t k = 0; k < a.n; ++k)  // exp(-pi i k^2 / n), k^2 reduced mod 2n for accuracy
            chirp[size_t(k)] = std::polar(1.0, -M_PI * double((k * k) % (2 * int64_t(a.n))) / double(a.n));
        table.insert(table.end(), chirp.begin(), chirp.end());
        a.filterOffset = table.size();
        for (int inverse = 0; inverse < 2; ++inverse) {
            std::vector<Complex> b(size_t(a.m));
            for (int k = 0; k < a.n; ++k) {
                const Complex w = inverse ? chirp[size_t(k)] : std::conj(chirp[size_t(k)]);  // conj of that direction's chirp
                b[size_t(k)] = w;
                if (k) b[size_t(a.m - k)] = w;
            }
            hostFft(b);
            for (Complex& x : b) x /= double(a.m);
            table.insert(table.end(), b.begin(), b.end());
        }
        blueWork = std::max(blueWork, size_t(plan.total / a.n) * size_t(a.m));
        if (blueWork >= INT32_MAX / 4) return CUFFT_NOT_SUPPORTED;
    }
    const size_t complexBytes = f64 ? 16 : 8;
    if (!table.empty()) {
        std::vector<unsigned char> bytes(table.size() * complexBytes);
        for (size_t i = 0; i < table.size(); ++i) {
            if (f64) {
                const double v[2] = {table[i].real(), table[i].imag()};
                std::memcpy(&bytes[i * 16], v, 16);
            } else {
                const float v[2] = {float(table[i].real()), float(table[i].imag())};
                std::memcpy(&bytes[i * 8], v, 8);
            }
        }
        if (cuMemAlloc(&plan.tables, bytes.size()) != CUDA_SUCCESS) return CUFFT_ALLOC_FAILED;
        if (cuMemcpyHtoD(plan.tables, bytes.data(), bytes.size()) != CUDA_SUCCESS) return CUFFT_INTERNAL_ERROR;
    }
    // workspace: two full arrays (ping-pong), and two Bluestein arrays
    plan.workSize = (2 * size_t(plan.total) + 2 * blueWork) * complexBytes;
    if (plan.autoAllocate && plan.workSize) {
        if (cuMemAlloc(&plan.ownWork, plan.workSize) != CUDA_SUCCESS) return CUFFT_ALLOC_FAILED;
        plan.work = plan.ownWork;
    }
    plan.made = true;
    if (workSize) *workSize = plan.workSize;
    return CUFFT_SUCCESS;
}

// Longest transform fft_local.comp does in shared memory (two buffers of 16 KiB).
int localLimit(bool f64) { return f64 ? 1024 : 2048; }

// Stockham passes of one axis: all in shared memory if the length allows (in place on *cur), else
// one dispatch each, ping-ponging between *cur and *other (swapped as they go).
CUresult passes(const Plan& plan, Params p, const std::vector<int>& radices, int len, int inner, int64_t elements,
                CUdeviceptr& cur, CUdeviceptr& other, bool inverse) {
    p.table = plan.tables + CUdeviceptr(p.table);
    p.len = len;
    p.inner = inner;
    p.flags = inverse ? kInverse : 0;
    if (len <= localLimit(plan.f64) && radices.size() <= 16) {
        uint64_t packed = 0;
        for (size_t i = 0; i < radices.size(); ++i) packed |= uint64_t(radices[i]) << (4 * i);
        // lines per workgroup: as many as fit (up to 64), for a strided axis adjacent ones (L | inner)
        const int64_t lines = elements / len;
        int group = 1;
        while (group < 64 && 2 * group * len <= localLimit(plan.f64) && (inner == 1 || inner % (2 * group) == 0))
            group *= 2;
        p.dist = int64_t(packed);
        p.src = p.dst = cur;
        p.radix = group;
        p.span = int(std::min<int64_t>(lines, INT32_MAX));
        p.items = (lines + group - 1) / group;
        if (lines > INT32_MAX) return CUDA_ERROR_NOT_SUPPORTED;
        return launch(plan.stream, Local, plan.f64, p);
    }
    int span = 1;
    for (int r : radices) {
        p.src = cur;
        p.dst = other;
        p.radix = r;
        p.span = span;
        p.items = elements / r;
        if (CUresult res = launch(plan.stream, Pass, plan.f64, p)) return res;
        std::swap(cur, other);
        span *= r;
    }
    return CUDA_SUCCESS;
}

cufftResult exec(Plan* plan, const void* input, void* output, int direction) {
    if (!plan || !plan->made) return CUFFT_INVALID_PLAN;
    if (!input || !output) return CUFFT_INVALID_VALUE;
    if (plan->workSize && !plan->work) return CUFFT_NO_WORKSPACE;
    const bool inverse = plan->mode == kC2R || (plan->mode == kC2C && direction == CUFFT_INVERSE);
    const size_t complexBytes = plan->f64 ? 16 : 8;
    const CUstream stream = plan->stream;
    CUdeviceptr full0 = plan->work, full1 = plan->work + plan->total * complexBytes;
    const CUdeviceptr blue0 = full1 + plan->total * complexBytes;

    Layout in = plan->in, out = plan->out;
    if (plan->basicLayout && input == output && plan->mode != kC2C) {
        // in place with the basic layout: the real array's rows are padded to 2 (n / 2 + 1)
        Layout& real = plan->mode == kR2C ? in : out;
        const int64_t padded = 2 * (plan->n[2] / 2 + 1);
        real.stride[2] = 1;
        real.stride[1] = padded;
        real.stride[0] = padded * plan->n[1];
        real.dist = padded * plan->n[1] * plan->n[0];
    }

    Params p{};
    std::copy(plan->n, plan->n + 3, p.n);
    p.src = CUdeviceptr(input);
    p.dst = full0;
    p.dist = in.dist;
    std::copy(in.stride, in.stride + 3, p.stride);
    p.items = plan->total;
    p.flags = plan->inType | plan->mode << kModeShift;
    if (launch(stream, Gather, plan->f64, p)) return CUFFT_EXEC_FAILED;

    CUdeviceptr cur = full0, other = full1;
    for (int d = 2; d >= 0; --d) {
        const Axis& a = plan->axes[d];
        if (a.n == 1) continue;
        int inner = 1;
        for (int k = d + 1; k < 3; ++k) inner *= plan->n[k];
        Params q{};
        q.table = a.rootsOffset * complexBytes;
        if (!a.bluestein) {
            if (passes(*plan, q, a.radices, a.n, inner, plan->total, cur, other, inverse)) return CUFFT_EXEC_FAILED;
            continue;
        }
        const int64_t lines = plan->total / a.n, elements = lines * a.m;
        Params b{};
        b.table = plan->tables + a.chirpOffset * complexBytes;
        b.aux = plan->tables + a.filterOffset * complexBytes;
        b.len = a.m;
        b.span = a.n;
        b.inner = inner;
        const uint32_t dir = inverse ? kInverse : 0;
        b.src = cur;
        b.dst = blue0;
        b.items = elements;
        b.flags = dir | kPre << kStepShift;
        if (launch(stream, Bluestein, plan->f64, b)) return CUFFT_EXEC_FAILED;
        CUdeviceptr bcur = blue0, bother = blue0 + elements * complexBytes;
        if (passes(*plan, q, a.radices, a.m, 1, elements, bcur, bother, false)) return CUFFT_EXEC_FAILED;
        b.src = b.dst = bcur;
        b.flags = dir | kMultiply << kStepShift;
        if (launch(stream, Bluestein, plan->f64, b)) return CUFFT_EXEC_FAILED;
        if (passes(*plan, q, a.radices, a.m, 1, elements, bcur, bother, true)) return CUFFT_EXEC_FAILED;
        b.src = bcur;
        b.dst = cur;
        b.items = plan->total;
        b.flags = dir | kPost << kStepShift;
        if (launch(stream, Bluestein, plan->f64, b)) return CUFFT_EXEC_FAILED;
    }

    p.src = cur;
    p.dst = CUdeviceptr(output);
    p.dist = out.dist;
    std::copy(out.stride, out.stride + 3, p.stride);
    p.items = plan->total / plan->n[2] * (plan->mode == kR2C ? plan->n[2] / 2 + 1 : plan->n[2]);
    p.flags = plan->outType | plan->mode << kModeShift;
    if (launch(stream, Scatter, plan->f64, p)) return CUFFT_EXEC_FAILED;
    return CUFFT_SUCCESS;
}

// cufftType: the transform and its precision.
bool typeOf(cufftType type, uint32_t& mode, bool& f64) {
    switch (type) {
    case CUFFT_C2C: mode = kC2C, f64 = false; return true;
    case CUFFT_R2C: mode = kR2C, f64 = false; return true;
    case CUFFT_C2R: mode = kC2R, f64 = false; return true;
    case CUFFT_Z2Z: mode = kC2C, f64 = true; return true;
    case CUFFT_D2Z: mode = kR2C, f64 = true; return true;
    case CUFFT_Z2D: mode = kC2R, f64 = true; return true;
    default: return false;
    }
}

cufftResult makePlanMany(cufftHandle handle, int rank, const long long* n, const long long* inembed,
                         long long istride, long long idist, const long long* onembed, long long ostride,
                         long long odist, cufftType type, long long batch, size_t* workSize) {
    Plan* plan = find(handle);
    if (!plan) return CUFFT_INVALID_PLAN;
    uint32_t mode;
    bool f64;
    if (!typeOf(type, mode, f64)) return CUFFT_INVALID_TYPE;
    const uint32_t t = f64 ? kF64 : kF32;
    return makePlan(*plan, rank, n, inembed, istride, idist, onembed, ostride, odist, mode, t, t, f64, batch,
                    workSize);
}

std::vector<long long> widen(const int* v, int count) {
    return v ? std::vector<long long>(v, v + count) : std::vector<long long>();
}

cufftResult createAndMake(cufftHandle* handle, int rank, const long long* n, cufftType type, long long batch) {
    if (!handle) return CUFFT_INVALID_VALUE;
    if (cufftResult r = cufftCreate(handle)) return r;
    const cufftResult r = makePlanMany(*handle, rank, n, nullptr, 1, 0, nullptr, 1, 0, type, batch, nullptr);
    if (r != CUFFT_SUCCESS) cufftDestroy(*handle);
    return r;
}

}  // namespace

extern "C" {

cufftResult cufftCreate(cufftHandle* plan) {
    VULKANIZER_TRACE_API();
    if (!plan) return CUFFT_INVALID_VALUE;
    if (cuInit(0) != CUDA_SUCCESS) return CUFFT_SETUP_FAILED;
    std::lock_guard<std::mutex> lock(g_mutex);
    *plan = g_next++;
    g_plans[*plan] = std::make_unique<Plan>();
    return CUFFT_SUCCESS;
}

cufftResult cufftDestroy(cufftHandle plan) {
    VULKANIZER_TRACE_API();
    std::unique_ptr<Plan> p;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        const auto it = g_plans.find(plan);
        if (it == g_plans.end()) return CUFFT_INVALID_PLAN;
        p = std::move(it->second);
        g_plans.erase(it);
    }
    if (p->stream) cuStreamSynchronize(p->stream);  // its tables and workspace may still be in use
    else cuCtxSynchronize();
    return CUFFT_SUCCESS;
}

cufftResult cufftSetStream(cufftHandle plan, cudaStream_t stream) {
    Plan* p = find(plan);
    if (!p) return CUFFT_INVALID_PLAN;
    p->stream = reinterpret_cast<CUstream>(stream);
    return CUFFT_SUCCESS;
}

cufftResult cufftSetAutoAllocation(cufftHandle plan, int autoAllocate) {
    Plan* p = find(plan);
    if (!p) return CUFFT_INVALID_PLAN;
    p->autoAllocate = autoAllocate != 0;
    return CUFFT_SUCCESS;
}

cufftResult cufftSetWorkArea(cufftHandle plan, void* workArea) {
    Plan* p = find(plan);
    if (!p) return CUFFT_INVALID_PLAN;
    p->work = CUdeviceptr(workArea);
    return CUFFT_SUCCESS;
}

cufftResult cufftGetSize(cufftHandle plan, size_t* workSize) {
    Plan* p = find(plan);
    if (!p) return CUFFT_INVALID_PLAN;
    if (!workSize) return CUFFT_INVALID_VALUE;
    *workSize = p->workSize;
    return CUFFT_SUCCESS;
}

cufftResult cufftGetVersion(int* version) {
    if (!version) return CUFFT_INVALID_VALUE;
    *version = VULKANIZER_CUFFT_VERSION;
    return CUFFT_SUCCESS;
}

cufftResult cufftGetProperty(libraryPropertyType type, int* value) {
    if (!value) return CUFFT_INVALID_VALUE;
    const int v = VULKANIZER_CUFFT_VERSION;
    switch (type) {
    case MAJOR_VERSION: *value = v / 1000; return CUFFT_SUCCESS;
    case MINOR_VERSION: *value = v % 1000 / 100; return CUFFT_SUCCESS;
    case PATCH_LEVEL: *value = v % 100; return CUFFT_SUCCESS;
    default: return CUFFT_INVALID_TYPE;
    }
}

cufftResult cufftMakePlanMany64(cufftHandle plan, int rank, long long int* n, long long int* inembed,
                                long long int istride, long long int idist, long long int* onembed,
                                long long int ostride, long long int odist, cufftType type, long long int batch,
                                size_t* workSize) {
    VULKANIZER_TRACE_API();
    return makePlanMany(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workSize);
}

cufftResult cufftMakePlanMany(cufftHandle plan, int rank, int* n, int* inembed, int istride, int idist, int* onembed,
                              int ostride, int odist, cufftType type, int batch, size_t* workSize) {
    VULKANIZER_TRACE_API();
    if (rank < 1 || rank > 3 || !n) return CUFFT_INVALID_VALUE;
    const auto n64 = widen(n, rank), in64 = widen(inembed, rank), on64 = widen(onembed, rank);
    return makePlanMany(plan, rank, n64.data(), inembed ? in64.data() : nullptr, istride, idist,
                        onembed ? on64.data() : nullptr, ostride, odist, type, batch, workSize);
}

cufftResult cufftMakePlan1d(cufftHandle plan, int nx, cufftType type, int batch, size_t* workSize) {
    const long long n[] = {nx};
    return makePlanMany(plan, 1, n, nullptr, 1, 0, nullptr, 1, 0, type, batch, workSize);
}
cufftResult cufftMakePlan2d(cufftHandle plan, int nx, int ny, cufftType type, size_t* workSize) {
    const long long n[] = {nx, ny};
    return makePlanMany(plan, 2, n, nullptr, 1, 0, nullptr, 1, 0, type, 1, workSize);
}
cufftResult cufftMakePlan3d(cufftHandle plan, int nx, int ny, int nz, cufftType type, size_t* workSize) {
    const long long n[] = {nx, ny, nz};
    return makePlanMany(plan, 3, n, nullptr, 1, 0, nullptr, 1, 0, type, 1, workSize);
}

cufftResult cufftPlan1d(cufftHandle* plan, int nx, cufftType type, int batch) {
    const long long n[] = {nx};
    return createAndMake(plan, 1, n, type, batch);
}
cufftResult cufftPlan2d(cufftHandle* plan, int nx, int ny, cufftType type) {
    const long long n[] = {nx, ny};
    return createAndMake(plan, 2, n, type, 1);
}
cufftResult cufftPlan3d(cufftHandle* plan, int nx, int ny, int nz, cufftType type) {
    const long long n[] = {nx, ny, nz};
    return createAndMake(plan, 3, n, type, 1);
}
cufftResult cufftPlanMany(cufftHandle* plan, int rank, int* n, int* inembed, int istride, int idist, int* onembed,
                          int ostride, int odist, cufftType type, int batch) {
    if (!plan) return CUFFT_INVALID_VALUE;
    if (cufftResult r = cufftCreate(plan)) return r;
    const cufftResult r =
        cufftMakePlanMany(*plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, nullptr);
    if (r != CUFFT_SUCCESS) cufftDestroy(*plan);
    return r;
}

cufftResult cufftXtMakePlanMany(cufftHandle plan, int rank, long long int* n, long long int* inembed,
                                long long int istride, long long int idist, cudaDataType inputtype,
                                long long int* onembed, long long int ostride, long long int odist,
                                cudaDataType outputtype, long long int batch, size_t* workSize,
                                cudaDataType executiontype) {
    VULKANIZER_TRACE_API();
    Plan* p = find(plan);
    if (!p) return CUFFT_INVALID_PLAN;
    // the element type of each side: real or complex, and its precision (f16 is computed in f32)
    const auto decode = [](cudaDataType t, bool& complex, uint32_t& code) {
        switch (t) {
        case CUDA_R_32F: complex = false, code = kF32; return true;
        case CUDA_C_32F: complex = true, code = kF32; return true;
        case CUDA_R_64F: complex = false, code = kF64; return true;
        case CUDA_C_64F: complex = true, code = kF64; return true;
        case CUDA_R_16F: complex = false, code = kF16; return true;
        case CUDA_C_16F: complex = true, code = kF16; return true;
        default: return false;
        }
    };
    bool inComplex, outComplex, execComplex;
    uint32_t inType, outType, execType;
    if (!decode(inputtype, inComplex, inType) || !decode(outputtype, outComplex, outType) ||
        !decode(executiontype, execComplex, execType))
        return CUFFT_INVALID_TYPE;
    if (!execComplex || inType != execType || outType != execType || (!inComplex && !outComplex))
        return CUFFT_INVALID_TYPE;
    const uint32_t mode = inComplex && outComplex ? kC2C : inComplex ? kC2R : kR2C;
    return makePlan(*p, rank, n, inembed, istride, idist, onembed, ostride, odist, mode, inType, outType,
                    execType == kF64, batch, workSize);
}

cufftResult cufftXtExec(cufftHandle plan, void* input, void* output, int direction) {
    VULKANIZER_TRACE_API();
    return exec(find(plan), input, output, direction);
}

cufftResult cufftExecC2C(cufftHandle plan, cufftComplex* idata, cufftComplex* odata, int direction) {
    VULKANIZER_TRACE_API();
    Plan* p = find(plan);
    if (p && (p->mode != kC2C || p->f64)) return CUFFT_INVALID_TYPE;
    return exec(p, idata, odata, direction);
}
cufftResult cufftExecR2C(cufftHandle plan, cufftReal* idata, cufftComplex* odata) {
    VULKANIZER_TRACE_API();
    Plan* p = find(plan);
    if (p && (p->mode != kR2C || p->f64)) return CUFFT_INVALID_TYPE;
    return exec(p, idata, odata, CUFFT_FORWARD);
}
cufftResult cufftExecC2R(cufftHandle plan, cufftComplex* idata, cufftReal* odata) {
    VULKANIZER_TRACE_API();
    Plan* p = find(plan);
    if (p && (p->mode != kC2R || p->f64)) return CUFFT_INVALID_TYPE;
    return exec(p, idata, odata, CUFFT_INVERSE);
}
cufftResult cufftExecZ2Z(cufftHandle plan, cufftDoubleComplex* idata, cufftDoubleComplex* odata, int direction) {
    VULKANIZER_TRACE_API();
    Plan* p = find(plan);
    if (p && (p->mode != kC2C || !p->f64)) return CUFFT_INVALID_TYPE;
    return exec(p, idata, odata, direction);
}
cufftResult cufftExecD2Z(cufftHandle plan, cufftDoubleReal* idata, cufftDoubleComplex* odata) {
    VULKANIZER_TRACE_API();
    Plan* p = find(plan);
    if (p && (p->mode != kR2C || !p->f64)) return CUFFT_INVALID_TYPE;
    return exec(p, idata, odata, CUFFT_FORWARD);
}
cufftResult cufftExecZ2D(cufftHandle plan, cufftDoubleComplex* idata, cufftDoubleReal* odata) {
    VULKANIZER_TRACE_API();
    Plan* p = find(plan);
    if (p && (p->mode != kC2R || !p->f64)) return CUFFT_INVALID_TYPE;
    return exec(p, idata, odata, CUFFT_INVERSE);
}

}  // extern "C"
