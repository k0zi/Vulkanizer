// Vulkanizer's libcurand: the cuRAND host API on Vulkanizer's own kernels (shaders/*.comp), queued on
// the generator's stream.
//
// XORWOW (the default), MRG32k3a, Philox4x32-10 and the Sobol sequences give cuRAND's results:
// the same raw bits and uniforms, and normals equal to rounding (the transcendental functions are
// Vulkanizer's own). The orderings were established by comparing against NVIDIA's libcurand host
// generator (which runs on the CPU), black box: each generator has threads on subsequences of the
// seed (Philox 65536, XORWOW 4096, MRG32k3a 81920), hands out groups of outputs to them round robin,
// and every thread continues its own stream from call to call. MTGP32 and MT19937 are accepted but draw from another
// Philox stream (their parallel layouts are not reproduced), scrambled Sobol uses Vulkanizer's own
// scrambling, and Poisson values come from inversion of one uniform each, so those sequences differ
// from NVIDIA's.
//
// The host generator (curandCreateGeneratorHost) runs the same kernels into device memory and copies
// the result back.

#include <curand/curand.h>
#include <cuda.h>
#include <vulkanizer/vulkanizer_ext.h>

#include "common/log.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

struct curandGenerator_st;
struct curandDiscreteDistribution_st;

namespace {

const uint32_t kPhilox32[] = {
#include "philox.comp.inc"
};
const uint32_t kPhilox64[] = {
#include "philox_f64.comp.inc"
};
const uint32_t kXorwow32[] = {
#include "xorwow.comp.inc"
};
const uint32_t kXorwow64[] = {
#include "xorwow_f64.comp.inc"
};
const uint32_t kMrg32[] = {
#include "mrg32k3a.comp.inc"
};
const uint32_t kMrg64[] = {
#include "mrg32k3a_f64.comp.inc"
};
const uint32_t kSobol32[] = {
#include "sobol.comp.inc"
};
const uint32_t kSobol64[] = {
#include "sobol_f64.comp.inc"
};

// Joe and Kuo's direction numbers of dimensions 2 to 20000 (tools/gen_sobol_table.sh).
const uint32_t kJoeKuo[] = {
#include "sobol_joe_kuo.inc"
};

// Push constants of every kernel (shaders/common.glsl, std430).
struct Params {
    uint64_t out, state, aux, table;
    uint64_t offset, count, seed;
    uint64_t mean, stddev;
    uint32_t kind, flags, threads, dims;
    uint32_t tableSize, tableBase;
};
static_assert(sizeof(Params) == 96, "must match shaders/common.glsl");

enum Kind : uint32_t {
    U32 = 0, U64 = 1, Uniform = 2, UniformDouble = 3, Normal = 4, NormalDouble = 5, LogNormal = 6,
    LogNormalDouble = 7, Poisson = 8
};
constexpr uint32_t kFlagInit = 1, kFlagSobol64 = 2, kFlagMtgp = 4, kFlagAdvance = 8, kFlagParity = 16, kFlagMt19937 = 32;

constexpr uint32_t kXorwowThreads = 4096, kMrgThreads = 81920, kPhiloxThreads = 65536;
constexpr unsigned kMaxDimensions = 20000;

enum KernelId { Philox, Xorwow, Mrg, Sobol, KernelCount };

CUresult function(KernelId kernel, bool f64, CUfunction& out) {
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
            {{kPhilox32, std::size(kPhilox32)}, {kPhilox64, std::size(kPhilox64)}},
            {{kXorwow32, std::size(kXorwow32)}, {kXorwow64, std::size(kXorwow64)}},
            {{kMrg32, std::size(kMrg32)}, {kMrg64, std::size(kMrg64)}},
            {{kSobol32, std::size(kSobol32)}, {kSobol64, std::size(kSobol64)}},
        };
        const Code& s = sources[kernel][f64];
        if (CUresult r = vulkanizerCreateSpirvFunction(&f, s.words, s.count, "main", sizeof(Params))) return r;
    }
    out = f;
    return CUDA_SUCCESS;
}

bool hasFloat64() {
    int v = 0;
    return vulkanizerGetFeature(VULKANIZER_FEATURE_FLOAT64, &v) == CUDA_SUCCESS && v;
}

// items invocations in 256-thread workgroups, on a 2D grid.
CUresult launch(CUstream stream, KernelId kernel, bool f64, Params& p, uint64_t items) {
    if (items == 0) return CUDA_SUCCESS;
    CUfunction f;
    if (CUresult r = function(kernel, f64, f)) return r;
    const uint64_t groups = (items + 255) / 256;
    const unsigned gx = unsigned(std::min<uint64_t>(groups, 65535)), gy = unsigned((groups + gx - 1) / gx);
    size_t size = sizeof p;
    void* extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, &p, CU_LAUNCH_PARAM_BUFFER_SIZE, &size, CU_LAUNCH_PARAM_END};
    return cuLaunchKernel(f, gx, gy, 1, 256, 1, 1, 0, stream, nullptr, extra);
}

// ---- Jump matrices --------------------------------------------------------------------------------

// XORWOW's xorshift part as a 160 x 160 matrix over GF(2): column j (5 words) is the image of bit j.
using Gf2 = std::array<std::array<uint32_t, 5>, 160>;

void xorshift(std::array<uint32_t, 5>& v) {
    uint32_t t = v[0] ^ (v[0] >> 2);
    v = {v[1], v[2], v[3], v[4], (v[4] ^ (v[4] << 4)) ^ (t ^ (t << 1))};
}

std::array<uint32_t, 5> apply(const Gf2& m, const std::array<uint32_t, 5>& v) {
    std::array<uint32_t, 5> r{};
    for (int j = 0; j < 160; ++j)
        if ((v[j / 32] >> (j % 32)) & 1)
            for (int k = 0; k < 5; ++k) r[k] ^= m[j][k];
    return r;
}

Gf2 square(const Gf2& m) {
    Gf2 r;
    for (int j = 0; j < 160; ++j) r[j] = apply(m, m[j]);
    return r;
}

// 12 matrices of 2^(67 + k) steps (subsequences), then 64 of 2^k steps (xorwow.comp).
const std::vector<uint32_t>& xorwowJumps() {
    static const std::vector<uint32_t> words = [] {
        Gf2 m;
        for (int j = 0; j < 160; ++j) {
            std::array<uint32_t, 5> v{};
            v[j / 32] = 1u << (j % 32);
            xorshift(v);
            m[j] = v;
        }
        std::vector<Gf2> step, seq;
        for (int k = 0; k < 64; ++k) step.push_back(m), m = square(m);
        for (int k = 64; k < 67; ++k) m = square(m);
        for (int k = 0; k < 12; ++k) seq.push_back(m), m = square(m);
        std::vector<uint32_t> out;
        for (const auto* list : {&seq, &step})
            for (const Gf2& g : *list)
                for (const auto& column : g) out.insert(out.end(), column.begin(), column.end());
        return out;
    }();
    return words;
}

constexpr uint64_t kM1 = 4294967087ull, kM2 = 4294944443ull;
using Mat3 = std::array<uint64_t, 9>;  // row-major

Mat3 multiply(const Mat3& a, const Mat3& b, uint64_t m) {
    Mat3 r{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            unsigned __int128 s = 0;
            for (int k = 0; k < 3; ++k) s += (unsigned __int128)a[i * 3 + k] * b[k * 3 + j];
            r[i * 3 + j] = uint64_t(s % m);
        }
    return r;
}

// 17 levels of 2^(76 + k) steps, then 64 of 2^k steps; each the m1 matrix, then the m2 one
// (mrg32k3a.comp).
const std::vector<uint32_t>& mrgJumps() {
    static const std::vector<uint32_t> words = [] {
        Mat3 a1 = {0, 1, 0, 0, 0, 1, kM1 - 810728, 1403580, 0};
        Mat3 a2 = {0, 1, 0, 0, 0, 1, kM2 - 1370589, 0, 527612};
        std::vector<std::pair<Mat3, Mat3>> step, seq;
        for (int k = 0; k < 64; ++k) {
            step.push_back({a1, a2});
            a1 = multiply(a1, a1, kM1), a2 = multiply(a2, a2, kM2);
        }
        for (int k = 64; k < 76; ++k) a1 = multiply(a1, a1, kM1), a2 = multiply(a2, a2, kM2);
        for (int k = 0; k < 17; ++k) {
            seq.push_back({a1, a2});
            a1 = multiply(a1, a1, kM1), a2 = multiply(a2, a2, kM2);
        }
        std::vector<uint32_t> out;
        for (const auto* list : {&seq, &step})
            for (const auto& [x, y] : *list) {
                for (uint64_t v : x) out.push_back(uint32_t(v));
                for (uint64_t v : y) out.push_back(uint32_t(v));
            }
        return out;
    }();
    return words;
}

// The jump matrices of a kernel in the current context's memory (kept for the process).
CUresult jumpMatrices(KernelId kernel, CUdeviceptr& out) {
    static std::mutex mutex;
    static std::map<std::pair<CUcontext, int>, CUdeviceptr> cache;
    CUcontext ctx = nullptr;
    if (CUresult r = cuCtxGetCurrent(&ctx)) return r;
    std::lock_guard<std::mutex> lock(mutex);
    CUdeviceptr& d = cache[{ctx, kernel}];
    if (!d) {
        const std::vector<uint32_t>& words = kernel == Xorwow ? xorwowJumps() : mrgJumps();
        if (CUresult r = cuMemAlloc(&d, words.size() * 4)) return d = 0, r;
        if (CUresult r = cuMemcpyHtoD(d, words.data(), words.size() * 4)) return r;
    }
    out = d;
    return CUDA_SUCCESS;
}

// ---- Sobol direction vectors ----------------------------------------------------------------------

uint64_t splitmix(uint64_t& s) {
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// Joe and Kuo's direction vectors of all dimensions, of bits bits (dimension 0 is van der Corput's),
// optionally scrambled: each dimension's vectors multiplied by a random unit lower triangular matrix
// (Owen's linear matrix scrambling, Vulkanizer's own random matrices).
template <typename Word>
std::vector<Word> directionVectors(bool scrambled) {
    constexpr int bits = sizeof(Word) * 8;
    std::vector<Word> v(size_t(kMaxDimensions) * bits);
    for (int k = 0; k < bits; ++k) v[k] = Word(1) << (bits - 1 - k);
    size_t at = 0;
    for (unsigned d = 1; d < kMaxDimensions; ++d) {
        Word* w = &v[size_t(d) * bits];
        const int s = int(kJoeKuo[at] & 31);
        const uint32_t a = kJoeKuo[at] >> 5;
        const uint32_t* m = &kJoeKuo[at + 1];
        at += 1 + size_t(s);
        for (int k = 0; k < std::min(s, bits); ++k) w[k] = Word(m[k]) << (bits - 1 - k);
        for (int k = s; k < bits; ++k) {
            w[k] = w[k - s] ^ (w[k - s] >> s);
            for (int i = 1; i < s; ++i)
                if ((a >> (s - 1 - i)) & 1) w[k] ^= w[k - i];
        }
    }
    if (scrambled) {
        for (unsigned d = 0; d < kMaxDimensions; ++d) {
            uint64_t state = 0x5C0B01ull * (d + 1) + bits;
            std::vector<Word> rows(bits);  // row r (output bit r from the top): its bits below the diagonal
            for (int r = 0; r < bits; ++r) {
                Word mask = r ? Word(~Word(0)) << (bits - r) : 0;  // the r bits above bit r (from the top)
                rows[r] = (Word(splitmix(state)) & mask) | (Word(1) << (bits - 1 - r));
            }
            for (int k = 0; k < bits; ++k) {
                Word x = v[size_t(d) * bits + k], y = 0;
                for (int r = 0; r < bits; ++r)
                    if (__builtin_parityll(uint64_t(x & rows[r]))) y |= Word(1) << (bits - 1 - r);
                v[size_t(d) * bits + k] = y;
            }
        }
    }
    return v;
}

const std::vector<uint32_t>& vectors32(bool scrambled) {
    static std::once_flag once[2];
    static std::vector<uint32_t> v[2];
    std::call_once(once[scrambled], [&] { v[scrambled] = directionVectors<uint32_t>(scrambled); });
    return v[scrambled];
}

const std::vector<unsigned long long>& vectors64(bool scrambled) {
    static std::once_flag once[2];
    static std::vector<unsigned long long> v[2];
    std::call_once(once[scrambled], [&] { v[scrambled] = directionVectors<unsigned long long>(scrambled); });
    return v[scrambled];
}

template <typename Word>
const std::vector<Word>& scrambleConstants() {
    static const std::vector<Word> c = [] {
        std::vector<Word> out(kMaxDimensions);
        uint64_t state = 0x5C0B02ull + sizeof(Word);
        for (Word& w : out) w = Word(splitmix(state));
        return out;
    }();
    return c;
}

// ---- Poisson --------------------------------------------------------------------------------------

// The CDF table of shaders/common.glsl's poisson(): floor(CDF(base + i) * 2^33), the last entry
// above every word. Empty (normal approximation) for lambda too large for a table.
struct PoissonTable {
    double lambda = 0;
    CUdeviceptr table = 0;
    uint32_t size = 0, base = 0;
    ~PoissonTable() {
        if (table) cuMemFree(table);
    }
};

constexpr size_t kMaxPoissonTable = size_t(1) << 22;

curandStatus_t makePoissonTable(double lambda, PoissonTable& t) {
    t.lambda = lambda;
    const double spread = 12 * std::sqrt(lambda) + 20;
    const double mode = std::floor(lambda);
    const double lo = std::max(0.0, mode - spread), hi = mode + spread;
    if (hi - lo + 1 > double(kMaxPoissonTable) || hi > 4e9) return CURAND_STATUS_SUCCESS;
    const uint32_t base = uint32_t(lo), size = uint32_t(hi - lo) + 1;
    std::vector<long double> pmf(size);
    long double sum = 0;
    for (uint32_t i = 0; i < size; ++i) {
        const long double k = (long double)base + i;
        pmf[i] = std::exp(-(long double)lambda + k * std::log((long double)lambda) - std::lgamma(k + 1));
        sum += pmf[i];
    }
    std::vector<uint64_t> cdf(size);
    long double acc = 0;
    for (uint32_t i = 0; i < size; ++i) {
        acc += pmf[i];
        cdf[i] = uint64_t(std::floor(acc / sum * 8589934592.0L));
    }
    cdf[size - 1] = uint64_t(1) << 34;
    if (cuMemAlloc(&t.table, size_t(size) * 8) != CUDA_SUCCESS) return CURAND_STATUS_ALLOCATION_FAILED;
    if (cuMemcpyHtoD(t.table, cdf.data(), size_t(size) * 8) != CUDA_SUCCESS) return CURAND_STATUS_LAUNCH_FAILURE;
    t.size = size;
    t.base = base;
    return CURAND_STATUS_SUCCESS;
}

bool pseudo(curandRngType_t t) { return t >= CURAND_RNG_PSEUDO_DEFAULT && t < CURAND_RNG_QUASI_DEFAULT; }
bool sobol64(curandRngType_t t) {
    return t == CURAND_RNG_QUASI_SOBOL64 || t == CURAND_RNG_QUASI_SCRAMBLED_SOBOL64;
}
bool scrambled(curandRngType_t t) {
    return t == CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 || t == CURAND_RNG_QUASI_SCRAMBLED_SOBOL64;
}

}  // namespace

struct curandDiscreteDistribution_st {
    PoissonTable poisson;
};

struct curandGenerator_st {
    curandRngType_t type;
    bool host = false;
    CUcontext ctx = nullptr;
    CUstream stream = nullptr;
    unsigned long long seed = 0, offset = 0;
    unsigned dims = 1;

    // The thread states (Philox: and the block of the group a call ended inside, in one of two slots),
    // recomputed from seed and offset when stale
    CUdeviceptr states = 0;
    bool stale = true;
    bool parity = false;
    // Sobol: the direction vectors (and scramble constants) of dims dimensions
    CUdeviceptr vectors = 0, constants = 0;
    unsigned vectorDims = 0;
    // Poisson tables of recent lambdas
    std::vector<std::unique_ptr<PoissonTable>> poisson;

    ~curandGenerator_st() {
        if (states) cuMemFree(states);
        if (vectors) cuMemFree(vectors);
        if (constants) cuMemFree(constants);
    }

    KernelId kernel() const {
        switch (type) {
        case CURAND_RNG_PSEUDO_XORWOW: return Xorwow;
        case CURAND_RNG_PSEUDO_MRG32K3A: return Mrg;
        case CURAND_RNG_PSEUDO_PHILOX4_32_10:
        case CURAND_RNG_PSEUDO_MTGP32:
        case CURAND_RNG_PSEUDO_MT19937: return Philox;
        default: return Sobol;
        }
    }
    uint32_t threads() const {
        return kernel() == Xorwow ? kXorwowThreads : kernel() == Mrg ? kMrgThreads : kPhiloxThreads;
    }
    size_t stateBytes() const { return kernel() == Philox ? (size_t(kPhiloxThreads) + 2) * 8 : size_t(threads()) * 24; }
};

namespace {

curandStatus_t status(CUresult r) {
    switch (r) {
    case CUDA_SUCCESS: return CURAND_STATUS_SUCCESS;
    case CUDA_ERROR_OUT_OF_MEMORY: return CURAND_STATUS_ALLOCATION_FAILED;
    case CUDA_ERROR_INVALID_CONTEXT:
    case CUDA_ERROR_NOT_INITIALIZED: return CURAND_STATUS_INITIALIZATION_FAILED;
    default: return CURAND_STATUS_LAUNCH_FAILURE;
    }
}

curandStatus_t create(curandGenerator_t* generator, curandRngType_t type, bool host) {
    if (!generator) return CURAND_STATUS_NOT_INITIALIZED;
    if (type == CURAND_RNG_PSEUDO_DEFAULT) type = CURAND_RNG_PSEUDO_XORWOW;
    if (type == CURAND_RNG_QUASI_DEFAULT) type = CURAND_RNG_QUASI_SOBOL32;
    switch (type) {
    case CURAND_RNG_PSEUDO_XORWOW:
    case CURAND_RNG_PSEUDO_MRG32K3A:
    case CURAND_RNG_PSEUDO_MTGP32:
    case CURAND_RNG_PSEUDO_MT19937:
    case CURAND_RNG_PSEUDO_PHILOX4_32_10:
    case CURAND_RNG_QUASI_SOBOL32:
    case CURAND_RNG_QUASI_SCRAMBLED_SOBOL32:
    case CURAND_RNG_QUASI_SOBOL64:
    case CURAND_RNG_QUASI_SCRAMBLED_SOBOL64: break;
    default: return CURAND_STATUS_TYPE_ERROR;
    }
    CUcontext ctx = nullptr;
    if (cuCtxGetCurrent(&ctx) != CUDA_SUCCESS || !ctx) {
        // No current context (a host generator user may never touch CUDA): device 0's primary context.
        if (cuInit(0) != CUDA_SUCCESS || cuDevicePrimaryCtxRetain(&ctx, 0) != CUDA_SUCCESS ||
            cuCtxSetCurrent(ctx) != CUDA_SUCCESS)
            return CURAND_STATUS_INITIALIZATION_FAILED;
    }
    auto* g = new curandGenerator_st;
    g->type = type;
    g->host = host;
    g->ctx = ctx;
    *generator = g;
    return CURAND_STATUS_SUCCESS;
}

// Makes the thread states of a pseudo-random generator current.
curandStatus_t prepareStates(curandGenerator_t g, bool f64) {
    if (!g->stale) return CURAND_STATUS_SUCCESS;
    const KernelId kernel = g->kernel();
    if (!g->states && cuMemAlloc(&g->states, g->stateBytes()) != CUDA_SUCCESS) return CURAND_STATUS_ALLOCATION_FAILED;
    CUdeviceptr jumps = 0;
    if (kernel != Philox)
        if (CUresult r = jumpMatrices(kernel, jumps)) return status(r);
    Params p{};
    p.aux = jumps;
    p.state = g->states;
    p.offset = g->offset;
    p.seed = g->seed;
    p.threads = g->threads();
    p.flags = kFlagInit;
    if (CUresult r = launch(g->stream, kernel, f64, p, p.threads)) return status(r);
    g->stale = false;
    g->parity = false;
    return CURAND_STATUS_SUCCESS;
}

// Makes the Sobol direction vectors of g->dims dimensions current.
curandStatus_t prepareVectors(curandGenerator_t g) {
    if (g->vectorDims == g->dims) return CURAND_STATUS_SUCCESS;
    const bool wide = sobol64(g->type), scramble = scrambled(g->type);
    const size_t word = wide ? 8 : 4, bits = word * 8;
    if (g->vectors) cuMemFree(g->vectors), g->vectors = 0;
    if (g->constants) cuMemFree(g->constants), g->constants = 0;
    g->vectorDims = 0;
    const void* v = wide ? (const void*)vectors64(scramble).data() : (const void*)vectors32(scramble).data();
    if (cuMemAlloc(&g->vectors, g->dims * bits * word) != CUDA_SUCCESS) return CURAND_STATUS_ALLOCATION_FAILED;
    if (cuMemcpyHtoD(g->vectors, v, g->dims * bits * word) != CUDA_SUCCESS) return CURAND_STATUS_LAUNCH_FAILURE;
    if (scramble) {
        const void* c = wide ? (const void*)scrambleConstants<unsigned long long>().data()
                             : (const void*)scrambleConstants<uint32_t>().data();
        if (cuMemAlloc(&g->constants, g->dims * word) != CUDA_SUCCESS) return CURAND_STATUS_ALLOCATION_FAILED;
        if (cuMemcpyHtoD(g->constants, c, g->dims * word) != CUDA_SUCCESS) return CURAND_STATUS_LAUNCH_FAILURE;
    }
    g->vectorDims = g->dims;
    return CURAND_STATUS_SUCCESS;
}

// The table for lambda: a distribution's, or one of the generator's recent ones.
curandStatus_t poissonTable(curandGenerator_t g, double lambda, const PoissonTable*& out) {
    for (auto& t : g->poisson)
        if (t->lambda == lambda) return out = t.get(), CURAND_STATUS_SUCCESS;
    auto t = std::make_unique<PoissonTable>();
    if (curandStatus_t s = makePoissonTable(lambda, *t)) return s;
    if (g->poisson.size() >= 8) g->poisson.erase(g->poisson.begin());
    g->poisson.push_back(std::move(t));
    out = g->poisson.back().get();
    return CURAND_STATUS_SUCCESS;
}

// Generates n outputs of kind into out (device memory, or host memory for a host generator).
// mean/stddev are the normal's parameters (f32 bits or f64 bits), lambda Poisson's.
curandStatus_t generate(curandGenerator_t g, void* out, size_t n, Kind kind, uint64_t mean = 0, uint64_t stddev = 0,
                        double lambda = 0) {
    if (!g) return CURAND_STATUS_NOT_INITIALIZED;
    const bool quasi = !pseudo(g->type);
    if (kind == U64 && !sobol64(g->type)) return CURAND_STATUS_TYPE_ERROR;
    if (kind == U32 && sobol64(g->type)) return CURAND_STATUS_TYPE_ERROR;
    if (quasi && n % g->dims) return CURAND_STATUS_LENGTH_NOT_MULTIPLE;
    const bool pairs = kind == Normal || kind == NormalDouble || kind == LogNormal || kind == LogNormalDouble;
    // XORWOW's and MRG32k3a's normals come in pairs (Philox's in groups of four, partial ones allowed)
    if (pairs && n % 2 && (g->kernel() == Xorwow || g->kernel() == Mrg)) return CURAND_STATUS_LENGTH_NOT_MULTIPLE;
    if (n == 0) return CURAND_STATUS_SUCCESS;
    if (!out) return CURAND_STATUS_OUT_OF_RANGE;
    const bool f64 = hasFloat64();
    const bool wideOutput = kind == U64 || kind == UniformDouble || kind == NormalDouble || kind == LogNormalDouble;
    if (wideOutput && kind != U64 && !f64) return CURAND_STATUS_DOUBLE_PRECISION_REQUIRED;

    CUcontext previous = nullptr;
    cuCtxGetCurrent(&previous);
    if (previous != g->ctx) cuCtxSetCurrent(g->ctx);
    struct Restore {
        CUcontext ctx, previous;
        ~Restore() {
            if (previous != ctx) cuCtxSetCurrent(previous);
        }
    } restore{g->ctx, previous};

    const size_t bytes = n * (wideOutput ? 8 : 4);
    CUdeviceptr target = reinterpret_cast<CUdeviceptr>(out), staging = 0;
    if (g->host) {
        if (cuMemAlloc(&staging, bytes) != CUDA_SUCCESS) return CURAND_STATUS_ALLOCATION_FAILED;
        target = staging;
    }
    struct Free {
        CUdeviceptr p;
        ~Free() {
            if (p) cuMemFree(p);
        }
    } freeStaging{staging};

    Params p{};
    p.out = target;
    p.kind = kind;
    p.seed = g->seed;
    p.mean = mean;
    p.stddev = stddev;
    p.dims = g->dims;
    if (kind == Poisson) {
        const PoissonTable* t = nullptr;
        if (curandStatus_t s = poissonTable(g, lambda, t)) return s;
        p.table = t->table;
        p.tableSize = t->size;
        p.tableBase = t->base;
        float l = float(lambda);
        std::memcpy(&p.mean, &l, 4);
    }
    const KernelId kernel = g->kernel();
    uint64_t items = 0, advance = 0;
    if (kernel == Philox) {
        if (curandStatus_t s = prepareStates(g, f64)) return s;
        p.state = g->states;
        p.offset = g->offset;
        p.count = n;
        if (g->type != CURAND_RNG_PSEUDO_PHILOX4_32_10) p.flags |= kFlagMtgp;
        if (g->type == CURAND_RNG_PSEUDO_MT19937) p.flags |= kFlagMt19937;
        if (g->parity) p.flags |= kFlagParity;
        items = (g->offset + n + 3) / 4 - g->offset / 4;
        advance = n;
    } else if (kernel == Sobol) {
        if (curandStatus_t s = prepareVectors(g)) return s;
        p.state = g->vectors;
        p.aux = g->constants;
        p.offset = g->offset;
        p.count = n / g->dims;
        if (sobol64(g->type)) p.flags |= kFlagSobol64;
        items = n;
        advance = n / g->dims;
    } else {
        if (curandStatus_t s = prepareStates(g, f64)) return s;
        p.state = g->states;
        p.offset = g->offset;
        p.count = n;
        p.threads = g->threads();
        items = p.threads;
        advance = pairs ? n / 2 : n;
    }
    if (CUresult r = launch(g->stream, kernel, f64, p, items)) return status(r);
    if (kernel == Philox) {
        p.flags |= kFlagAdvance;
        if (CUresult r = launch(g->stream, kernel, f64, p, kPhiloxThreads)) return status(r);
        if ((g->offset + n) % 4) g->parity = !g->parity;
    }
    g->offset += advance;
    if (g->host) {
        if (CUresult r = cuStreamSynchronize(g->stream)) return status(r);
        if (CUresult r = cuMemcpyDtoH(out, staging, bytes)) return status(r);
    }
    return CURAND_STATUS_SUCCESS;
}

template <typename T>
uint64_t bits(T v) {
    uint64_t b = 0;
    std::memcpy(&b, &v, sizeof v);
    return b;
}

}  // namespace

extern "C" {

curandStatus_t curandCreateGenerator(curandGenerator_t* generator, curandRngType_t rng_type) {
    VULKANIZER_TRACE_API();
    return create(generator, rng_type, false);
}

curandStatus_t curandCreateGeneratorHost(curandGenerator_t* generator, curandRngType_t rng_type) {
    VULKANIZER_TRACE_API();
    return create(generator, rng_type, true);
}

curandStatus_t curandDestroyGenerator(curandGenerator_t generator) {
    VULKANIZER_TRACE_API();
    if (!generator) return CURAND_STATUS_NOT_INITIALIZED;
    // Pending work may still use the generator's buffers: Vulkanizer frees them after it.
    delete generator;
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGetVersion(int* version) {
    if (!version) return CURAND_STATUS_OUT_OF_RANGE;
    *version = VULKANIZER_CURAND_VERSION;
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGetProperty(libraryPropertyType type, int* value) {
    if (!value) return CURAND_STATUS_OUT_OF_RANGE;
    const int v = VULKANIZER_CURAND_VERSION;
    switch (type) {
    case MAJOR_VERSION: *value = v / 1000; return CURAND_STATUS_SUCCESS;
    case MINOR_VERSION: *value = v % 1000 / 100; return CURAND_STATUS_SUCCESS;
    case PATCH_LEVEL: *value = v % 100; return CURAND_STATUS_SUCCESS;
    default: return CURAND_STATUS_OUT_OF_RANGE;
    }
}

curandStatus_t curandSetStream(curandGenerator_t generator, cudaStream_t stream) {
    if (!generator) return CURAND_STATUS_NOT_INITIALIZED;
    generator->stream = reinterpret_cast<CUstream>(stream);
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandSetPseudoRandomGeneratorSeed(curandGenerator_t generator, unsigned long long seed) {
    if (!generator) return CURAND_STATUS_NOT_INITIALIZED;
    if (!pseudo(generator->type)) return CURAND_STATUS_TYPE_ERROR;
    generator->seed = seed;  // the offset stays
    generator->stale = true;
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandSetGeneratorOffset(curandGenerator_t generator, unsigned long long offset) {
    if (!generator) return CURAND_STATUS_NOT_INITIALIZED;
    generator->offset = offset;
    generator->stale = true;
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandSetGeneratorOrdering(curandGenerator_t generator, curandOrdering_t order) {
    if (!generator) return CURAND_STATUS_NOT_INITIALIZED;
    // Every ordering gives the default one's results.
    const bool valid = pseudo(generator->type) ? order >= CURAND_ORDERING_PSEUDO_BEST && order <= CURAND_ORDERING_PSEUDO_DYNAMIC
                                               : order == CURAND_ORDERING_QUASI_DEFAULT;
    return valid ? CURAND_STATUS_SUCCESS : CURAND_STATUS_OUT_OF_RANGE;
}

curandStatus_t curandSetQuasiRandomGeneratorDimensions(curandGenerator_t generator, unsigned int num_dimensions) {
    if (!generator) return CURAND_STATUS_NOT_INITIALIZED;
    if (pseudo(generator->type)) return CURAND_STATUS_TYPE_ERROR;
    if (num_dimensions < 1 || num_dimensions > kMaxDimensions) return CURAND_STATUS_OUT_OF_RANGE;
    generator->dims = num_dimensions;
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGenerateSeeds(curandGenerator_t generator) {
    VULKANIZER_TRACE_API();
    if (!generator) return CURAND_STATUS_NOT_INITIALIZED;
    if (!pseudo(generator->type)) return CURAND_STATUS_SUCCESS;
    CUcontext previous = nullptr;
    cuCtxGetCurrent(&previous);
    if (previous != generator->ctx) cuCtxSetCurrent(generator->ctx);
    curandStatus_t s = prepareStates(generator, hasFloat64());
    if (previous != generator->ctx) cuCtxSetCurrent(previous);
    return s;
}

curandStatus_t curandGenerate(curandGenerator_t generator, unsigned int* outputPtr, size_t num) {
    VULKANIZER_TRACE_API();
    return generate(generator, outputPtr, num, U32);
}

curandStatus_t curandGenerateLongLong(curandGenerator_t generator, unsigned long long* outputPtr, size_t num) {
    VULKANIZER_TRACE_API();
    return generate(generator, outputPtr, num, U64);
}

curandStatus_t curandGenerateUniform(curandGenerator_t generator, float* outputPtr, size_t num) {
    VULKANIZER_TRACE_API();
    return generate(generator, outputPtr, num, Uniform);
}

curandStatus_t curandGenerateUniformDouble(curandGenerator_t generator, double* outputPtr, size_t num) {
    VULKANIZER_TRACE_API();
    return generate(generator, outputPtr, num, UniformDouble);
}

curandStatus_t curandGenerateNormal(curandGenerator_t generator, float* outputPtr, size_t n, float mean, float stddev) {
    VULKANIZER_TRACE_API();
    return generate(generator, outputPtr, n, Normal, bits(mean), bits(stddev));
}

curandStatus_t curandGenerateNormalDouble(curandGenerator_t generator, double* outputPtr, size_t n, double mean,
                                          double stddev) {
    VULKANIZER_TRACE_API();
    return generate(generator, outputPtr, n, NormalDouble, bits(mean), bits(stddev));
}

curandStatus_t curandGenerateLogNormal(curandGenerator_t generator, float* outputPtr, size_t n, float mean,
                                       float stddev) {
    VULKANIZER_TRACE_API();
    return generate(generator, outputPtr, n, LogNormal, bits(mean), bits(stddev));
}

curandStatus_t curandGenerateLogNormalDouble(curandGenerator_t generator, double* outputPtr, size_t n, double mean,
                                             double stddev) {
    VULKANIZER_TRACE_API();
    return generate(generator, outputPtr, n, LogNormalDouble, bits(mean), bits(stddev));
}

curandStatus_t curandCreatePoissonDistribution(double lambda, curandDiscreteDistribution_t* discrete_distribution) {
    VULKANIZER_TRACE_API();
    if (!discrete_distribution) return CURAND_STATUS_NOT_INITIALIZED;
    if (!(lambda > 0)) return CURAND_STATUS_OUT_OF_RANGE;
    auto d = std::make_unique<curandDiscreteDistribution_st>();
    if (curandStatus_t s = makePoissonTable(lambda, d->poisson)) return s;
    *discrete_distribution = d.release();
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandDestroyDistribution(curandDiscreteDistribution_t discrete_distribution) {
    if (!discrete_distribution) return CURAND_STATUS_OUT_OF_RANGE;
    delete discrete_distribution;
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGeneratePoisson(curandGenerator_t generator, unsigned int* outputPtr, size_t n, double lambda) {
    VULKANIZER_TRACE_API();
    if (!(lambda > 0)) return CURAND_STATUS_OUT_OF_RANGE;
    return generate(generator, outputPtr, n, Poisson, 0, 0, lambda);
}

curandStatus_t curandGeneratePoissonMethod(curandGenerator_t generator, unsigned int* outputPtr, size_t n,
                                           double lambda, curandMethod_t) {
    return curandGeneratePoisson(generator, outputPtr, n, lambda);
}

curandStatus_t curandGetDirectionVectors32(curandDirectionVectors32_t* vectors[], curandDirectionVectorSet_t set) {
    if (!vectors) return CURAND_STATUS_OUT_OF_RANGE;
    if (set != CURAND_DIRECTION_VECTORS_32_JOEKUO6 && set != CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6)
        return CURAND_STATUS_OUT_OF_RANGE;
    const auto& v = vectors32(set == CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6);
    *vectors = reinterpret_cast<curandDirectionVectors32_t*>(const_cast<uint32_t*>(v.data()));
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGetDirectionVectors64(curandDirectionVectors64_t* vectors[], curandDirectionVectorSet_t set) {
    if (!vectors) return CURAND_STATUS_OUT_OF_RANGE;
    if (set != CURAND_DIRECTION_VECTORS_64_JOEKUO6 && set != CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6)
        return CURAND_STATUS_OUT_OF_RANGE;
    const auto& v = vectors64(set == CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6);
    *vectors = reinterpret_cast<curandDirectionVectors64_t*>(const_cast<unsigned long long*>(v.data()));
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGetScrambleConstants32(unsigned int** constants) {
    if (!constants) return CURAND_STATUS_OUT_OF_RANGE;
    *constants = const_cast<unsigned int*>(scrambleConstants<uint32_t>().data());
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGetScrambleConstants64(unsigned long long** constants) {
    if (!constants) return CURAND_STATUS_OUT_OF_RANGE;
    *constants = const_cast<unsigned long long*>(scrambleConstants<unsigned long long>().data());
    return CURAND_STATUS_SUCCESS;
}

}  // extern "C"
