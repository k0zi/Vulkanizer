// Vulkanizer's libcurand: reference values of NVIDIA's libcurand (raw bits and uniforms, also after
// offsets and mixed calls), device and host generators agreeing, the moments of every distribution,
// Sobol's dimension layout and the error statuses.
//
// With VULKANIZER_CURAND_REFERENCE=/path/to/NVIDIA/libcurand.so.10 it also compares long mixed call
// sequences against that library's host generator (which runs on the CPU, so no NVIDIA GPU is needed):
// raw bits and uniforms bit for bit, normals to rounding.

#include "test_util.h"

#include <curand/curand.h>

#include <dlfcn.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#define CHECK_RAND(call)                                                                           \
    do {                                                                                           \
        curandStatus_t s_ = (call);                                                                \
        if (s_ != CURAND_STATUS_SUCCESS) {                                                         \
            std::fprintf(stderr, "%s:%d: %s failed: %d\n", __FILE__, __LINE__, #call, int(s_));    \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

namespace {

int g_failures = 0;
bool g_f64 = true;

void expect(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "OK  " : "FAIL", what.c_str());
    if (!ok) ++g_failures;
}

// One generator call: kind (u32, u64, uf, ud, nf, nd, lf, ld, p), count, and the distribution's parameters.
struct Call {
    const char* kind;
    size_t n;
    double a = 0, b = 1;
};

size_t elementBytes(const std::string& k) { return k == "u64" || k == "ud" || k == "nd" || k == "ld" ? 8 : 4; }

// The entry points of a libcurand (Vulkanizer's, or NVIDIA's loaded with dlopen).
struct Api {
    decltype(&curandCreateGeneratorHost) createHost = curandCreateGeneratorHost;
    decltype(&curandDestroyGenerator) destroy = curandDestroyGenerator;
    decltype(&curandSetPseudoRandomGeneratorSeed) seed = curandSetPseudoRandomGeneratorSeed;
    decltype(&curandSetGeneratorOffset) offset = curandSetGeneratorOffset;
    decltype(&curandSetQuasiRandomGeneratorDimensions) dims = curandSetQuasiRandomGeneratorDimensions;
    decltype(&curandGenerate) u32 = curandGenerate;
    decltype(&curandGenerateLongLong) u64 = curandGenerateLongLong;
    decltype(&curandGenerateUniform) uf = curandGenerateUniform;
    decltype(&curandGenerateUniformDouble) ud = curandGenerateUniformDouble;
    decltype(&curandGenerateNormal) nf = curandGenerateNormal;
    decltype(&curandGenerateNormalDouble) nd = curandGenerateNormalDouble;
    decltype(&curandGenerateLogNormal) lf = curandGenerateLogNormal;
    decltype(&curandGenerateLogNormalDouble) ld = curandGenerateLogNormalDouble;
    decltype(&curandGeneratePoisson) poisson = curandGeneratePoisson;
};

curandStatus_t issue(const Api& api, curandGenerator_t g, const Call& c, void* out) {
    const std::string k = c.kind;
    if (k == "u32") return api.u32(g, static_cast<unsigned*>(out), c.n);
    if (k == "u64") return api.u64(g, static_cast<unsigned long long*>(out), c.n);
    if (k == "uf") return api.uf(g, static_cast<float*>(out), c.n);
    if (k == "ud") return api.ud(g, static_cast<double*>(out), c.n);
    if (k == "nf") return api.nf(g, static_cast<float*>(out), c.n, float(c.a), float(c.b));
    if (k == "nd") return api.nd(g, static_cast<double*>(out), c.n, c.a, c.b);
    if (k == "lf") return api.lf(g, static_cast<float*>(out), c.n, float(c.a), float(c.b));
    if (k == "ld") return api.ld(g, static_cast<double*>(out), c.n, c.a, c.b);
    return api.poisson(g, static_cast<unsigned*>(out), c.n, c.a);
}

// Runs the calls on a host generator of api, concatenating the results.
std::vector<uint8_t> runHost(const Api& api, curandRngType_t type, unsigned long long seedOrDims,
                             unsigned long long offset, const std::vector<Call>& calls) {
    curandGenerator_t g;
    CHECK_RAND(api.createHost(&g, type));
    if (type < CURAND_RNG_QUASI_DEFAULT) CHECK_RAND(api.seed(g, seedOrDims));
    else CHECK_RAND(api.dims(g, unsigned(seedOrDims)));
    if (offset) CHECK_RAND(api.offset(g, offset));
    std::vector<uint8_t> all;
    for (const Call& c : calls) {
        std::vector<uint8_t> out(c.n * elementBytes(c.kind));
        CHECK_RAND(issue(api, g, c, out.data()));
        all.insert(all.end(), out.begin(), out.end());
    }
    CHECK_RAND(api.destroy(g));
    return all;
}

// The same on a device generator, copying each result back.
std::vector<uint8_t> runDevice(curandRngType_t type, unsigned long long seedOrDims, unsigned long long offset,
                               const std::vector<Call>& calls) {
    curandGenerator_t g;
    CHECK_RAND(curandCreateGenerator(&g, type));
    if (type < CURAND_RNG_QUASI_DEFAULT) CHECK_RAND(curandSetPseudoRandomGeneratorSeed(g, seedOrDims));
    else CHECK_RAND(curandSetQuasiRandomGeneratorDimensions(g, unsigned(seedOrDims)));
    if (offset) CHECK_RAND(curandSetGeneratorOffset(g, offset));
    std::vector<uint8_t> all;
    for (const Call& c : calls) {
        const size_t bytes = c.n * elementBytes(c.kind);
        CUdeviceptr d;
        CHECK_CU(cuMemAlloc(&d, bytes));
        CHECK_RAND(issue(Api{}, g, c, reinterpret_cast<void*>(d)));
        std::vector<uint8_t> out(bytes);
        CHECK_CU(cuMemcpyDtoH(out.data(), d, bytes));
        CHECK_CU(cuMemFree(d));
        all.insert(all.end(), out.begin(), out.end());
    }
    CHECK_RAND(curandDestroyGenerator(g));
    return all;
}

bool usesDouble(const std::vector<Call>& calls) {
    for (const Call& c : calls)
        if (elementBytes(c.kind) == 8 && std::string(c.kind) != "u64") return true;
    return false;
}

// Compares two concatenated results call by call: bits exactly, normals to rounding.
bool same(const std::vector<uint8_t>& x, const std::vector<uint8_t>& y, const std::vector<Call>& calls,
          std::string& detail) {
    if (x.size() != y.size()) return detail = "sizes differ", false;
    size_t at = 0;
    for (const Call& c : calls) {
        const std::string k = c.kind;
        const size_t e = elementBytes(k);
        for (size_t i = 0; i < c.n; ++i, at += e) {
            const bool normal = k[0] == 'n' || k[0] == 'l';
            if (!normal) {
                if (std::memcmp(&x[at], &y[at], e)) return detail = k + " #" + std::to_string(i) + " differs", false;
                continue;
            }
            double a, b;
            if (e == 4) {
                float fa, fb;
                std::memcpy(&fa, &x[at], 4), std::memcpy(&fb, &y[at], 4);
                a = fa, b = fb;
            } else {
                std::memcpy(&a, &x[at], 8), std::memcpy(&b, &y[at], 8);
            }
            const double tol = (e == 4 ? 4e-6 : 1e-11) * std::max(1.0, std::fabs(a));
            if (!(std::fabs(a - b) <= tol))
                return detail = k + " #" + std::to_string(i) + ": " + std::to_string(a) + " vs " + std::to_string(b), false;
        }
    }
    return true;
}

template <typename T>
T word(const std::vector<uint8_t>& bytes, size_t i) {
    T v;
    std::memcpy(&v, &bytes[i * sizeof(T)], sizeof(T));
    return v;
}

struct Golden {
    const char* name;
    curandRngType_t type;
    unsigned long long seedOrDims, offset;
    std::vector<Call> calls;
    std::vector<uint64_t> tail;  // the last results (of the last call, which is u32/u64/uf/ud)
};

void goldens() {
    using T = curandRngType_t;
    const Golden cases[] = {
        {"XORWOW", T::CURAND_RNG_PSEUDO_DEFAULT, 1234, 0, {{"u32", 4}}, {0x253d5e15, 0xd1f7605a, 0x8ce6f6f8, 0x4b79ff67}},
        {"XORWOW mixed calls", T::CURAND_RNG_PSEUDO_XORWOW, 1234, 0, {{"u32", 7}, {"nf", 4}, {"ud", 5000}, {"u32", 4}},
         {0xb87386dc, 0x1225fa96, 0x10c14591, 0x4e2cc822}},
        {"XORWOW double", T::CURAND_RNG_PSEUDO_XORWOW, 1234, 0, {{"ud", 2}}, {0x3fdbd561cb7abc2bull, 0x3feda53518d7605aull}},
        {"MRG32k3a", T::CURAND_RNG_PSEUDO_MRG32K3A, 1234, 0, {{"u32", 4}}, {0xc96891f0, 0x26f0d612, 0xef269080, 0xfc8fa50e}},
        {"MRG32k3a mixed calls", T::CURAND_RNG_PSEUDO_MRG32K3A, 1234, 0, {{"nd", 81922}, {"u32", 3}, {"ud", 90000}, {"u32", 4}},
         {0x4baed93e, 0x42472839, 0x79d13586, 0xeb4ea574}},
        {"MRG32k3a double", T::CURAND_RNG_PSEUDO_MRG32K3A, 1234, 0, {{"ud", 2}}, {0x3fe92d123e0dcbe5ull, 0x3fc3786b09654f63ull}},
        {"Philox", T::CURAND_RNG_PSEUDO_PHILOX4_32_10, 1234, 0, {{"u32", 4}}, {0x2090b348, 0xda7cf0ab, 0x4401906f, 0xcbca470e}},
        {"Philox uniform", T::CURAND_RNG_PSEUDO_PHILOX4_32_10, 1234, 0, {{"uf", 4}}, {0x3e0242cd, 0x3f5a7cf1, 0x3e880321, 0x3f4bca47}},
        {"Philox offset, doubles across rounds", T::CURAND_RNG_PSEUDO_PHILOX4_32_10, 1234, 5,
         {{"ud", 300000}, {"ud", 300001}, {"u32", 4}}, {0x3acc6a47, 0x83b08455, 0x943e386a, 0xa663654a}},
        {"Sobol32 3 dims, offset 10", T::CURAND_RNG_QUASI_SOBOL32, 3, 10, {{"u32", 12}},
         {0xb0000000, 0x30000000, 0x50000000, 0xd0000000}},
        {"Sobol64 2 dims", T::CURAND_RNG_QUASI_SOBOL64, 2, 0, {{"u64", 6}}, {0x8000000000000000ull, 0x4000000000000000ull}},
    };
    for (const Golden& c : cases) {
        if (!g_f64 && usesDouble(c.calls)) {
            std::printf("skip %s (no f64)\n", c.name);
            continue;
        }
        for (bool device : {false, true}) {
            const auto out = device ? runDevice(c.type, c.seedOrDims, c.offset, c.calls)
                                    : runHost(Api{}, c.type, c.seedOrDims, c.offset, c.calls);
            const std::string last = c.calls.back().kind;
            const size_t e = elementBytes(last), count = out.size() / e;
            bool ok = true;
            for (size_t i = 0; i < c.tail.size(); ++i) {
                const size_t at = count - c.tail.size() + i;
                const uint64_t v = e == 8 ? word<uint64_t>(out, at) : word<uint32_t>(out, at);
                if (v != c.tail[i]) ok = false;
            }
            expect(ok, std::string(c.name) + (device ? " (device)" : " (host)"));
        }
    }
}

// Mean and variance of a distribution's sample against the expected ones.
void moments(const char* name, curandRngType_t type, const Call& c, double mean, double var) {
    if (!g_f64 && elementBytes(c.kind) == 8) return;
    const auto out = runDevice(type, type < CURAND_RNG_QUASI_DEFAULT ? 42 : 1, 0, {c});
    const std::string k = c.kind;
    double s = 0, s2 = 0;
    for (size_t i = 0; i < c.n; ++i) {
        double v = k == "u32" || k == "p" ? double(word<uint32_t>(out, i))
                 : elementBytes(k) == 8 ? word<double>(out, i) : double(word<float>(out, i));
        s += v, s2 += v * v;
    }
    const double m = s / c.n, v = s2 / c.n - m * m;
    const double se = std::sqrt(var / c.n);  // the mean's standard error
    const bool ok = std::fabs(m - mean) < 6 * se + 1e-12 * std::fabs(mean) && std::fabs(v - var) < 0.03 * var;
    char line[256];
    std::snprintf(line, sizeof line, "%s %s n=%zu: mean %.6g (%.6g), variance %.6g (%.6g)", name, c.kind, c.n, m, mean,
                  v, var);
    expect(ok, line);
}

void distributions() {
    using T = curandRngType_t;
    const struct {
        const char* name;
        T type;
    } gens[] = {{"XORWOW", T::CURAND_RNG_PSEUDO_XORWOW},     {"MRG32k3a", T::CURAND_RNG_PSEUDO_MRG32K3A},
                {"Philox", T::CURAND_RNG_PSEUDO_PHILOX4_32_10}, {"MTGP32", T::CURAND_RNG_PSEUDO_MTGP32},
                {"MT19937", T::CURAND_RNG_PSEUDO_MT19937},      {"Sobol32", T::CURAND_RNG_QUASI_SOBOL32},
                {"scrambled Sobol64", T::CURAND_RNG_QUASI_SCRAMBLED_SOBOL64}};
    const double ln = std::exp(0.5 + 0.25 * 0.25 / 2), lnVar = (std::exp(0.25 * 0.25) - 1) * std::exp(1 + 0.25 * 0.25);
    for (const auto& g : gens) {
        const size_t n = 1 << 20;
        moments(g.name, g.type, {"uf", n}, 0.5, 1.0 / 12);
        moments(g.name, g.type, {"ud", n}, 0.5, 1.0 / 12);
        moments(g.name, g.type, {"nf", n, 1.5, 2}, 1.5, 4);
        moments(g.name, g.type, {"nd", n, -1, 0.5}, -1, 0.25);
        moments(g.name, g.type, {"lf", n, 0.5, 0.25}, ln, lnVar);
        moments(g.name, g.type, {"ld", n, 0.5, 0.25}, ln, lnVar);
        moments(g.name, g.type, {"p", n, 3.5}, 3.5, 3.5);
    }
    for (double lambda : {0.25, 17.0, 1000.0, 2e6, 1e9})
        moments("Philox Poisson", T::CURAND_RNG_PSEUDO_PHILOX4_32_10, {"p", 1 << 20, lambda}, lambda, lambda);
}

// The pseudo-random generators: the device generator against the host generator for odd lengths
// and offsets (the kernels are shared, the paths around them are not).
void hostDevice() {
    using T = curandRngType_t;
    const std::vector<Call> calls = g_f64 ? std::vector<Call>{{"u32", 1001}, {"uf", 70001}, {"ud", 33}, {"nf", 8}, {"nd", 300002},
                                                              {"lf", 10}, {"p", 999, 7.25}, {"u32", 5}}
                                          : std::vector<Call>{{"u32", 1001}, {"uf", 70001}, {"nf", 8}, {"lf", 10}, {"p", 999, 7.25}, {"u32", 5}};
    for (T type : {T::CURAND_RNG_PSEUDO_XORWOW, T::CURAND_RNG_PSEUDO_MRG32K3A, T::CURAND_RNG_PSEUDO_PHILOX4_32_10,
                   T::CURAND_RNG_PSEUDO_MTGP32}) {
        std::string detail;
        expect(same(runHost(Api{}, type, 77, 12345, calls), runDevice(type, 77, 12345, calls), calls, detail),
               "host and device generator agree, type " + std::to_string(int(type)) + " " + detail);
    }
}

void sobolLayout() {
    // Dimension-major: n / dims points of each dimension; dimension 0 is van der Corput's sequence.
    const auto out = runDevice(CURAND_RNG_QUASI_SOBOL32, 4, 0, {{"u32", 4 * 8}});
    const uint32_t vdc[8] = {0, 0x80000000u, 0xc0000000u, 0x40000000u, 0x60000000u, 0xe0000000u, 0xa0000000u, 0x20000000u};
    bool ok = true;
    for (int i = 0; i < 8; ++i) ok &= word<uint32_t>(out, i) == vdc[i];
    for (int d = 1; d < 4; ++d) ok &= word<uint32_t>(out, d * 8) == 0 && word<uint32_t>(out, d * 8 + 1) == 0x80000000u;
    expect(ok, "Sobol32 dimension-major layout");

    curandDirectionVectors32_t* v32;
    curandDirectionVectors64_t* v64;
    CHECK_RAND(curandGetDirectionVectors32(&v32, CURAND_DIRECTION_VECTORS_32_JOEKUO6));
    CHECK_RAND(curandGetDirectionVectors64(&v64, CURAND_DIRECTION_VECTORS_64_JOEKUO6));
    // dimension 1 (x + 1, m = 1): v_k = 2^(31 - k) ^ v_(k - 1) >> 1 ...: 0x80000000, 0xc0000000, 0xa0000000
    expect(v32[0][0] == 0x80000000u && v32[0][31] == 1u && v32[1][0] == 0x80000000u && v32[1][1] == 0xc0000000u &&
               v32[1][2] == 0xa0000000u && v64[1][1] == 0xc000000000000000ull && v32[19999][0] == 0x80000000u,
           "direction vectors");
    unsigned* c32;
    CHECK_RAND(curandGetScrambleConstants32(&c32));
    expect(c32[0] != c32[1], "scramble constants");
}

void statuses() {
    curandGenerator_t g, q, q64;
    CHECK_RAND(curandCreateGenerator(&g, CURAND_RNG_PSEUDO_XORWOW));
    CHECK_RAND(curandCreateGenerator(&q, CURAND_RNG_QUASI_SOBOL32));
    CHECK_RAND(curandCreateGenerator(&q64, CURAND_RNG_QUASI_SOBOL64));
    CUdeviceptr d;
    CHECK_CU(cuMemAlloc(&d, 64));
    curandGenerator_t bad;
    curandDiscreteDistribution_t dist;
    const bool ok =
        curandCreateGenerator(&bad, curandRngType_t(7)) == CURAND_STATUS_TYPE_ERROR &&
        curandSetPseudoRandomGeneratorSeed(q, 1) == CURAND_STATUS_TYPE_ERROR &&
        curandSetQuasiRandomGeneratorDimensions(g, 2) == CURAND_STATUS_TYPE_ERROR &&
        curandSetQuasiRandomGeneratorDimensions(q, 0) == CURAND_STATUS_OUT_OF_RANGE &&
        curandSetQuasiRandomGeneratorDimensions(q, 20001) == CURAND_STATUS_OUT_OF_RANGE &&
        curandGenerateLongLong(g, reinterpret_cast<unsigned long long*>(d), 2) == CURAND_STATUS_TYPE_ERROR &&
        curandGenerate(q64, reinterpret_cast<unsigned*>(d), 2) == CURAND_STATUS_TYPE_ERROR &&
        curandGenerateNormal(g, reinterpret_cast<float*>(d), 3, 0, 1) == CURAND_STATUS_LENGTH_NOT_MULTIPLE &&
        curandSetQuasiRandomGeneratorDimensions(q, 3) == CURAND_STATUS_SUCCESS &&
        curandGenerateUniform(q, reinterpret_cast<float*>(d), 7) == CURAND_STATUS_LENGTH_NOT_MULTIPLE &&
        curandGeneratePoisson(g, reinterpret_cast<unsigned*>(d), 2, 0.0) == CURAND_STATUS_OUT_OF_RANGE &&
        curandCreatePoissonDistribution(-1, &dist) == CURAND_STATUS_OUT_OF_RANGE &&
        curandSetGeneratorOrdering(g, CURAND_ORDERING_QUASI_DEFAULT) == CURAND_STATUS_OUT_OF_RANGE &&
        curandSetGeneratorOrdering(q, CURAND_ORDERING_QUASI_DEFAULT) == CURAND_STATUS_SUCCESS;
    expect(ok, "error statuses");
    CHECK_RAND(curandCreatePoissonDistribution(4.5, &dist));
    CHECK_RAND(curandDestroyDistribution(dist));
    int version = 0, major = 0;
    CHECK_RAND(curandGetVersion(&version));
    CHECK_RAND(curandGetProperty(MAJOR_VERSION, &major));
    expect(major == 10 && version / 1000 == 10, "version " + std::to_string(version));
    CHECK_CU(cuMemFree(d));
    CHECK_RAND(curandDestroyGenerator(g));
    CHECK_RAND(curandDestroyGenerator(q));
    CHECK_RAND(curandDestroyGenerator(q64));
}

// Against NVIDIA's libcurand host generator, if given.
void reference(const char* path) {
    void* h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        expect(false, std::string("load ") + path + ": " + dlerror());
        return;
    }
    Api nv;
#define SYM(field, name) nv.field = reinterpret_cast<decltype(nv.field)>(dlsym(h, name))
    SYM(createHost, "curandCreateGeneratorHost");
    SYM(destroy, "curandDestroyGenerator");
    SYM(seed, "curandSetPseudoRandomGeneratorSeed");
    SYM(offset, "curandSetGeneratorOffset");
    SYM(dims, "curandSetQuasiRandomGeneratorDimensions");
    SYM(u32, "curandGenerate");
    SYM(u64, "curandGenerateLongLong");
    SYM(uf, "curandGenerateUniform");
    SYM(ud, "curandGenerateUniformDouble");
    SYM(nf, "curandGenerateNormal");
    SYM(nd, "curandGenerateNormalDouble");
    SYM(lf, "curandGenerateLogNormal");
    SYM(ld, "curandGenerateLogNormalDouble");
#undef SYM
    using T = curandRngType_t;
    const std::vector<Call> pseudo = {{"u32", 7},       {"ud", 300000},     {"u32", 5},    {"nf", 100000, 0.5, 2},
                                      {"nd", 81924, 1, 3}, {"uf", 262147}, {"lf", 4096, 0, 0.5}, {"ld", 2, 0, 0.5},
                                      {"ud", 4097},     {"u32", 1000003}, {"nf", 2},     {"uf", 3}};
    for (T type : {T::CURAND_RNG_PSEUDO_XORWOW, T::CURAND_RNG_PSEUDO_MRG32K3A, T::CURAND_RNG_PSEUDO_PHILOX4_32_10})
        for (unsigned long long offset : {0ull, 1ull, 300001ull}) {
            if (!g_f64) break;
            std::string detail;
            expect(same(runHost(nv, type, 0x123456789abcull, offset, pseudo),
                        runHost(Api{}, type, 0x123456789abcull, offset, pseudo), pseudo, detail),
                   "NVIDIA reference, type " + std::to_string(int(type)) + " offset " + std::to_string(offset) + " " + detail);
        }
    for (T type : {T::CURAND_RNG_QUASI_SOBOL32, T::CURAND_RNG_QUASI_SOBOL64})
        for (unsigned dims : {1u, 7u, 20000u}) {
            const bool wide = type == T::CURAND_RNG_QUASI_SOBOL64;
            std::vector<Call> calls = {{wide ? "u64" : "u32", dims * 300}, {"uf", dims * 5}, {"nf", dims * 64}};
            if (g_f64) calls.push_back({"nd", dims * 3}), calls.push_back({"ud", dims * 9});
            std::string detail;
            expect(same(runHost(nv, type, dims, 17, calls), runHost(Api{}, type, dims, 17, calls), calls, detail),
                   "NVIDIA reference, type " + std::to_string(int(type)) + " dims " + std::to_string(dims) + " " + detail);
        }
}

}  // namespace

int main() {
    CHECK_CU(cuInit(0));
    CUdevice dev;
    CHECK_CU(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CHECK_CU(cuDevicePrimaryCtxRetain(&ctx, dev));
    CHECK_CU(cuCtxSetCurrent(ctx));
    {
        // Doubles need shaderFloat64: probe with one value.
        curandGenerator_t g;
        CHECK_RAND(curandCreateGeneratorHost(&g, CURAND_RNG_PSEUDO_PHILOX4_32_10));
        double d;
        g_f64 = curandGenerateUniformDouble(g, &d, 1) == CURAND_STATUS_SUCCESS;
        CHECK_RAND(curandDestroyGenerator(g));
        if (!g_f64) std::printf("the device has no f64: double outputs skipped\n");
    }
    goldens();
    hostDevice();
    sobolLayout();
    statuses();
    distributions();
    if (const char* path = std::getenv("VULKANIZER_CURAND_REFERENCE")) reference(path);
    std::printf(g_failures ? "%d FAILED\n" : "ALL OK\n", g_failures);
    return g_failures ? 1 : 0;
}
