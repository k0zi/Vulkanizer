// Phase 3 kernel features, each checked against a CPU reference:
// shared memory (static, dynamic, sub-word), __syncthreads, atomics (global/shared, 32/64-bit,
// float), warp shuffles and votes, local arrays, __constant__/__device__ variables, generic
// pointers, math functions, bit manipulation and half conversions.
//
//   rt_kernels            all tests
//   rt_kernels warp math  only the named ones
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <vector>

namespace cg = cooperative_groups;

#define CHECK(call)                                                                            \
    do {                                                                                       \
        cudaError_t e = (call);                                                                \
        if (e != cudaSuccess) {                                                                \
            std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(e)); \
            std::exit(1);                                                                      \
        }                                                                                      \
    } while (0)

static int g_mismatches = 0;

template <class T>
static bool same(T a, T b) {
    return a == b;
}
static bool same(float a, float b) {
    if (std::isnan(a) || std::isnan(b)) return std::isnan(a) && std::isnan(b);
    return std::fabs(a - b) <= 2e-6f * std::fmax(1.0f, std::fmax(std::fabs(a), std::fabs(b)));
}

template <class T>
static bool expectEqual(const char* what, const std::vector<T>& got, const std::vector<T>& want) {
    for (size_t i = 0; i < want.size(); ++i) {
        if (!same(got[i], want[i])) {
            std::fprintf(stderr, "  %s[%zu]: got %.9g, want %.9g\n", what, i, double(got[i]), double(want[i]));
            ++g_mismatches;
            return false;
        }
    }
    return true;
}

template <class T>
struct DeviceArray {
    T* ptr = nullptr;
    size_t count;
    explicit DeviceArray(size_t n, int fill = 0) : count(n) {
        CHECK(cudaMalloc(&ptr, n * sizeof(T)));
        CHECK(cudaMemset(ptr, fill, n * sizeof(T)));
    }
    explicit DeviceArray(const std::vector<T>& v) : count(v.size()) {
        CHECK(cudaMalloc(&ptr, v.size() * sizeof(T)));
        CHECK(cudaMemcpy(ptr, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    ~DeviceArray() { cudaFree(ptr); }
    std::vector<T> get() const {
        std::vector<T> v(count);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(v.data(), ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
        return v;
    }
};

static uint32_t lcg(uint32_t& s) { return s = s * 1664525u + 1013904223u; }

// ---- shared memory and barriers ---------------------------------------------------------------

__global__ void reduceSum(const float* in, float* out, int n) {
    __shared__ float s[256];
    int tid = threadIdx.x, i = blockIdx.x * blockDim.x + tid;
    s[tid] = i < n ? in[i] : 0.0f;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (tid < k) s[tid] += s[tid + k];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = s[0];
}

static bool testReduce() {
    const int n = 100000, block = 256, blocks = (n + block - 1) / block;
    std::vector<float> in(n);
    for (int i = 0; i < n; ++i) in[i] = float(i % 97) * 0.25f;
    DeviceArray<float> din(in), dout(blocks);
    reduceSum<<<blocks, block>>>(din.ptr, dout.ptr, n);
    std::vector<float> want(blocks, 0.0f);
    for (int b = 0; b < blocks; ++b) {
        // same pairwise order as the kernel
        float s[256];
        for (int t = 0; t < block; ++t) s[t] = b * block + t < n ? in[b * block + t] : 0.0f;
        for (int k = block / 2; k > 0; k >>= 1)
            for (int t = 0; t < k; ++t) s[t] += s[t + k];
        want[b] = s[0];
    }
    return expectEqual("block sums", dout.get(), want);
}

__global__ void syncReductions(const int* in, int* out) {
    const int v = in[blockIdx.x * blockDim.x + threadIdx.x];
    const int count = __syncthreads_count(v > 0);
    const int any = __syncthreads_or(v == 7);
    const int all = __syncthreads_and(v > -100);
    if (threadIdx.x == 0) {
        out[blockIdx.x * 3 + 0] = count;
        out[blockIdx.x * 3 + 1] = any != 0;
        out[blockIdx.x * 3 + 2] = all != 0;
    }
}

static bool testSyncReductions() {
    const int blocks = 6, block = 192;
    std::vector<int> in(blocks * block), want(blocks * 3);
    uint32_t s = 5;
    for (int& v : in) v = int(lcg(s) >> 27) - 16;
    in[3 * block + 10] = 7;
    in[4 * block] = -500;
    for (int b = 0; b < blocks; ++b) {
        int count = 0, any = 0, all = 1;
        for (int t = 0; t < block; ++t) {
            const int v = in[b * block + t];
            count += v > 0, any |= v == 7, all &= v > -100;
        }
        want[b * 3] = count, want[b * 3 + 1] = any, want[b * 3 + 2] = all;
    }
    DeviceArray<int> din(in), dout(blocks * 3);
    syncReductions<<<blocks, block>>>(din.ptr, dout.ptr);
    return expectEqual("__syncthreads_count/or/and", dout.get(), want);
}

// Odd warps write shared memory and exit; the others pass a barrier and read what they wrote.
// CUDA counts exited threads as arrived, so their writes must stay visible.
__global__ void exitBeforeBarrier(float* out) {
    __shared__ float s[256];
    const int t = threadIdx.x, warp = t / 32;
    if (warp % 2 == 1) {
        s[t] = float(t) * 2.0f;
        return;
    }
    __syncthreads();
    out[t] = s[t + 32];
}

// flash_attn_tile's combine step: odd warps leave partial results in shared memory, even warps
// synchronize, add them up and store under further conditions, so that both arms end in one
// shared return (no arm returns on its own).
__global__ void combineAfterSplit(const float* in, float* out, int limit) {
    __shared__ float part[256];
    __shared__ float sums[8];
    const int t = threadIdx.x, warp = t / 32, lane = t % 32;
    float v = in[t] * 3.0f;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    if (warp % 2 != 0) {
        part[t] = in[t] + 1.0f;
        if (lane == 0) sums[warp] = v;
    } else {
        __syncthreads();
        const float total = in[t] + part[t + 32] + sums[warp + 1];
        if (blockIdx.x * 256 + t < limit) {
            out[blockIdx.x * 256 + t] = total;
            if (lane == 0 && limit > 1) out[512 + blockIdx.x * 8 + warp] = sums[warp + 1];
        }
    }
}

static bool testExitBeforeBarrier() {
    std::vector<float> in(512);
    for (int i = 0; i < 512; ++i) in[i] = float(i % 97) * 0.5f;
    DeviceArray<float> din(in), cout(528);
    combineAfterSplit<<<2, 256>>>(din.ptr, cout.ptr, 500);
    std::vector<float> gotC = cout.get(), wantC(528, 0.0f);
    for (int b = 0; b < 2; ++b)
        for (int t = 0; t < 256; ++t) {
            const int w = t / 32;
            if (w % 2 != 0 || b * 256 + t >= 500) continue;
            float s = 0.0f;  // the odd warp's reduced sum
            for (int l = 0; l < 32; ++l) s += in[(w + 1) * 32 + l] * 3.0f;
            wantC[b * 256 + t] = in[t] + (in[t + 32] + 1.0f) + s;
            if (t % 32 == 0) wantC[512 + b * 8 + w] = s;
        }
    // second block reads in[t] of the first 256 entries only, the same values in both blocks
    bool ok = expectEqual("combine after split", gotC, wantC);

    DeviceArray<float> out(256);
    exitBeforeBarrier<<<1, 256>>>(out.ptr);
    std::vector<float> got = out.get(), want(256, 0.0f);
    for (int t = 0; t < 256; ++t) {
        if ((t / 32) % 2 == 0) want[t] = float(t + 32) * 2.0f;
        else got[t] = 0.0f;  // not written by the kernel
    }
    return expectEqual("exit before barrier", got, want) && ok;
}

template <int Tile>
__global__ void matmul(const float* a, const float* b, float* c, int n) {
    __shared__ float ta[Tile][Tile];
    __shared__ float tb[Tile][Tile + 1];  // padded
    int row = blockIdx.y * Tile + threadIdx.y, col = blockIdx.x * Tile + threadIdx.x;
    float acc = 0.0f;
    for (int t = 0; t < n; t += Tile) {
        ta[threadIdx.y][threadIdx.x] = row < n && t + threadIdx.x < n ? a[row * n + t + threadIdx.x] : 0.0f;
        tb[threadIdx.y][threadIdx.x] = col < n && t + threadIdx.y < n ? b[(t + threadIdx.y) * n + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < Tile; ++k) acc += ta[threadIdx.y][k] * tb[k][threadIdx.x];
        __syncthreads();
    }
    if (row < n && col < n) c[row * n + col] = acc;
}

static bool testMatmul() {
    const int n = 70;
    std::vector<float> a(n * n), b(n * n), want(n * n);
    for (int i = 0; i < n * n; ++i) a[i] = float(i % 13) - 6.0f, b[i] = float(i % 7) * 0.5f;
    for (int r = 0; r < n; ++r)
        for (int c = 0; c < n; ++c) {
            float acc = 0;
            for (int k = 0; k < n; ++k) acc += a[r * n + k] * b[k * n + c];
            want[r * n + c] = acc;
        }
    DeviceArray<float> da(a), db(b), dc(n * n);
    matmul<16><<<dim3((n + 15) / 16, (n + 15) / 16), dim3(16, 16)>>>(da.ptr, db.ptr, dc.ptr, n);
    return expectEqual("c", dc.get(), want);
}

// dynamic shared memory, shared atomics, byte loads, sub-word shared stores
__global__ void histogram(const unsigned char* in, unsigned* bins, int n) {
    extern __shared__ unsigned local[];
    for (int b = threadIdx.x; b < 256; b += blockDim.x) local[b] = 0;
    __syncthreads();
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) atomicAdd(&local[in[i]], 1u);
    __syncthreads();
    for (int b = threadIdx.x; b < 256; b += blockDim.x) atomicAdd(&bins[b], local[b]);
}

__global__ void byteShuffle(const unsigned char* in, unsigned char* out, int n) {
    __shared__ unsigned char bytes[512];
    __shared__ short halves[256];
    int t = threadIdx.x;
    bytes[t] = in[blockIdx.x * blockDim.x + t];  // neighbouring threads write the same word
    halves[t / 2] = 0;
    __syncthreads();
    halves[t / 2] = short(t);  // two writers per half: either value is fine, bytes must survive
    __syncthreads();
    out[blockIdx.x * blockDim.x + t] = bytes[blockDim.x - 1 - t] + (halves[t / 2] >= 0 ? 0 : 1);
}

static bool testHistogram() {
    const int n = 200000;
    std::vector<unsigned char> in(n);
    uint32_t s = 7;
    for (int i = 0; i < n; ++i) in[i] = (unsigned char)(lcg(s) >> 24);
    std::vector<unsigned> want(256, 0);
    for (unsigned char v : in) ++want[v];
    DeviceArray<unsigned char> din(in);
    DeviceArray<unsigned> bins(256);
    histogram<<<40, 128, 256 * sizeof(unsigned)>>>(din.ptr, bins.ptr, n);
    bool ok = expectEqual("bins", bins.get(), want);

    const int m = 256 * 8;
    DeviceArray<unsigned char> dout(m);
    byteShuffle<<<8, 256>>>(din.ptr, dout.ptr, m);
    std::vector<unsigned char> wantBytes(m);
    for (int b = 0; b < 8; ++b)
        for (int t = 0; t < 256; ++t) wantBytes[b * 256 + t] = in[b * 256 + 255 - t];
    return expectEqual("bytes", dout.get(), wantBytes) && ok;
}

// ---- atomics ----------------------------------------------------------------------------------

__global__ void atomics(int* ints, unsigned* uints, unsigned long long* u64, float* f, int n) {
    __shared__ int smax;
    if (threadIdx.x == 0) smax = -1;
    __syncthreads();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int v = (i * 7919) % 1000 - 500;
        atomicMax(&ints[0], v);
        atomicMin(&ints[1], v);
        atomicAdd(&ints[2], v);
        atomicSub(&ints[3], 1);
        atomicAnd(&uints[0], ~(1u << (i % 32)));
        atomicOr(&uints[1], 1u << (i % 32));
        atomicXor(&uints[2], unsigned(i));
        atomicInc(&uints[3], 9u);  // wraps to 0 after 9
        atomicAdd(&u64[0], 1ull << 33);
        atomicMax(&u64[1], (unsigned long long)i << 20);
        atomicAdd(&f[0], 0.5f);
        atomicMax(&smax, v);
        int old = atomicCAS(&ints[4], i - 1, i);  // a chain: only succeeds in order... or not
        (void)old;
        atomicExch(&ints[5], 42);
    }
    __syncthreads();
    if (threadIdx.x == 0) atomicMax(&ints[6], smax);
}

static bool testAtomics() {
    const int n = 3000;
    DeviceArray<int> ints(std::vector<int>{-1000000, 1000000, 0, 0, -1, 0, -1000000});
    DeviceArray<unsigned> uints(std::vector<unsigned>{0xFFFFFFFFu, 0, 0, 0});
    DeviceArray<unsigned long long> u64(2);
    DeviceArray<float> f(1);
    atomics<<<(n + 127) / 128, 128>>>(ints.ptr, uints.ptr, u64.ptr, f.ptr, n);
    int mx = -1000000, mn = 1000000, sum = 0;
    unsigned x = 0;
    for (int i = 0; i < n; ++i) {
        int v = (i * 7919) % 1000 - 500;
        mx = std::max(mx, v), mn = std::min(mn, v), sum += v;
        x ^= unsigned(i);
    }
    const std::vector<int> gi = ints.get();
    const std::vector<unsigned> gu = uints.get();
    const std::vector<unsigned long long> g64 = u64.get();
    bool ok = expectEqual("ints", std::vector<int>(gi.begin(), gi.begin() + 4), std::vector<int>{mx, mn, sum, -n});
    ok &= expectEqual("ints[5..6]", std::vector<int>{gi[5], gi[6]}, std::vector<int>{42, mx});
    ok &= gi[4] >= 0 && gi[4] < n;  // the CAS chain advanced and stayed in range
    ok &= expectEqual("uints", gu, std::vector<unsigned>{0, 0xFFFFFFFFu, x, unsigned(n % 10)});
    ok &= expectEqual("u64", g64, std::vector<unsigned long long>{(unsigned long long)n << 33,
                                                                   (unsigned long long)(n - 1) << 20});
    ok &= expectEqual("float", f.get(), std::vector<float>{0.5f * n});
    if (!ok) ++g_mismatches;
    return ok;
}

// ---- warps ------------------------------------------------------------------------------------

__global__ void warpOps(const int* in, int* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x & 31;
    int v = in[i];
    int sum = v;
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
    int all = v;
    for (int o = 1; o < 32; o <<= 1) all += __shfl_xor_sync(0xffffffff, all, o);
    int scan = v;
    for (int o = 1; o < 32; o <<= 1) {
        int up = __shfl_up_sync(0xffffffff, scan, o);
        if (lane >= o) scan += up;
    }
    const int bcast = __shfl_sync(0xffffffff, v, 5);
    const int seg = __shfl_down_sync(0xffffffff, v, 1, 8);  // width-8 segments
    const unsigned ballot = __ballot_sync(0xffffffff, v > 0);
    const int any = __any_sync(0xffffffff, v == 7);
    const int every = __all_sync(0xffffffff, v > -1000);
    unsigned laneId;
    asm("mov.u32 %0, %%laneid;" : "=r"(laneId));
    int* o = out + i * 9;
    o[0] = lane == 0 ? sum : 0;
    o[1] = all;
    o[2] = scan;
    o[3] = bcast;
    o[4] = seg;
    o[5] = int(ballot);
    o[6] = any;
    o[7] = every;
    o[8] = int(laneId) - lane;  // must be 0: lanes follow threadIdx order
}

static bool testWarps() {
    const int n = 256;
    std::vector<int> in(n);
    uint32_t s = 3;
    for (int i = 0; i < n; ++i) in[i] = int(lcg(s) >> 26) - 32;
    in[37] = 7;
    std::vector<int> want(n * 9);
    for (int w = 0; w < n / 32; ++w) {
        const int* v = &in[w * 32];
        int total = 0;
        unsigned ballot = 0;
        bool any = false, every = true;
        for (int l = 0; l < 32; ++l) {
            total += v[l];
            ballot |= unsigned(v[l] > 0) << l;
            any |= v[l] == 7;
            every &= v[l] > -1000;
        }
        int scan = 0;
        for (int l = 0; l < 32; ++l) {
            int* o = &want[(w * 32 + l) * 9];
            scan += v[l];
            o[0] = l == 0 ? total : 0;
            o[1] = total;
            o[2] = scan;
            o[3] = v[5];
            o[4] = (l % 8) < 7 ? v[l + 1] : v[l];
            o[5] = int(ballot);
            o[6] = any;
            o[7] = every;
            o[8] = 0;
        }
    }
    DeviceArray<int> din(in), dout(n * 9);
    warpOps<<<n / 64, 64>>>(din.ptr, dout.ptr);
    return expectEqual("warp", dout.get(), want);
}

// ---- local arrays, module variables, generic pointers -----------------------------------------

__constant__ float coeffs[4];
__device__ int callCount;
__device__ int table[8] = {3, 1, 4, 1, 5, 9, 2, 6};

__global__ void variables(const float* data, float* out, const float* other, int n, int useShared) {
    __shared__ float stage[128];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    float local[8];
    for (int k = 0; k < 8; ++k) local[k] = data[(i + k) % n];
    float acc = 0;
    for (int k = 0; k < 8; ++k) acc += local[(k * 3 + i) & 7] * coeffs[k & 3] + float(table[(i + k) & 7]);
    stage[threadIdx.x] = acc;
    __syncthreads();
    const float* p = useShared ? stage : other;  // a generic pointer into shared or global memory
    const float picked = p[(threadIdx.x + 1) % blockDim.x];
    atomicAdd(&callCount, 1);
    __syncthreads();
    if (i < n) out[i] = acc + picked;
}

static bool testVariables() {
    const int n = 128 * 4;
    const float c[4] = {1.0f, -2.0f, 0.5f, 3.0f};
    CHECK(cudaMemcpyToSymbol(coeffs, c, sizeof c));
    const int zero = 0;
    CHECK(cudaMemcpyToSymbol(callCount, &zero, sizeof zero));
    int t[8];
    CHECK(cudaMemcpyFromSymbol(t, table, sizeof t));
    bool ok = expectEqual("table", std::vector<int>(t, t + 8), std::vector<int>{3, 1, 4, 1, 5, 9, 2, 6});

    std::vector<float> data(n), other(n);
    for (int i = 0; i < n; ++i) data[i] = float(i % 11), other[i] = float(i) * 0.01f;
    for (int useShared = 0; useShared < 2; ++useShared) {
        DeviceArray<float> dd(data), dout(n), dother(other);
        variables<<<n / 128, 128>>>(dd.ptr, dout.ptr, dother.ptr, n, useShared);
        std::vector<float> acc(n), want(n);
        for (int i = 0; i < n; ++i) {
            float local[8];
            for (int k = 0; k < 8; ++k) local[k] = data[(i + k) % n];
            float a = 0;
            for (int k = 0; k < 8; ++k) a += local[(k * 3 + i) & 7] * c[k & 3] + float(t[(i + k) & 7]);
            acc[i] = a;
        }
        for (int i = 0; i < n; ++i) {
            const int b = i / 128, tid = i % 128, src = (tid + 1) % 128;
            want[i] = acc[i] + (useShared ? acc[b * 128 + src] : other[src]);
        }
        ok &= expectEqual(useShared ? "data (shared)" : "data (global)", dout.get(), want);
    }
    int calls = 0;
    CHECK(cudaMemcpyFromSymbol(&calls, callCount, sizeof calls));
    ok &= expectEqual("callCount", std::vector<int>{calls}, std::vector<int>{2 * n});
    void* addr = nullptr;
    size_t size = 0;
    CHECK(cudaGetSymbolAddress(&addr, table));
    CHECK(cudaGetSymbolSize(&size, table));
    ok &= addr != nullptr && size == sizeof t;
    return ok;
}

// ---- math, bits, halves -----------------------------------------------------------------------

__global__ void mathFns(const float* in, float* out, double* dout, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = in[i];
    float* o = out + i * 10;
    o[0] = sqrtf(fabsf(x));
    o[1] = rsqrtf(fabsf(x) + 1.0f);
    o[2] = sinf(x);
    o[3] = cosf(x);
    o[4] = expf(x * 0.1f);
    o[5] = logf(fabsf(x) + 1.0f);
    o[6] = powf(fabsf(x), 0.75f);
    o[7] = tanhf(x * 0.1f);
    o[8] = fminf(x, 1.0f) / 3.0f;
    o[9] = __half2float(__float2half(x));
    dout[i] = sqrt(double(x) * double(x) + 1.0) * 0.5;
}

static bool testMath() {
    std::vector<float> in = {0.0f, 1.0f, -1.0f, 0.5f, 3.14159265f, -2.5f, 10.0f, 100.0f, 1e-3f, 12345.678f};
    for (int i = 0; i < 300; ++i) in.push_back(float(i) * 0.37f - 50.0f);
    const int n = int(in.size());
    DeviceArray<float> din(in), dout(n * 10);
    DeviceArray<double> dd(n);
    mathFns<<<(n + 63) / 64, 64>>>(din.ptr, dout.ptr, dd.ptr, n);
    const std::vector<float> got = dout.get();
    const std::vector<double> gotD = dd.get();
    // Vulkan's precision guarantees are looser than CUDA's: compare with a relative tolerance.
    const char* names[10] = {"sqrt", "rsqrt", "sin", "cos", "exp", "log", "pow", "tanh", "div", "half"};
    bool ok = true;
    for (int i = 0; i < n && ok; ++i) {
        const float x = in[i];
        const float want[10] = {std::sqrt(std::fabs(x)), 1.0f / std::sqrt(std::fabs(x) + 1.0f), std::sin(x),
                                std::cos(x), std::exp(x * 0.1f), std::log(std::fabs(x) + 1.0f),
                                std::pow(std::fabs(x), 0.75f), std::tanh(x * 0.1f), std::fmin(x, 1.0f) / 3.0f,
                                __half2float(__float2half(x))};
        for (int k = 0; k < 10; ++k) {
            const float g = got[i * 10 + k];
            const float tol = 1e-5f * std::fmax(1.0f, std::fabs(want[k])) + (k == 9 ? 0.0f : 2e-6f);
            if (g != want[k] && !(std::fabs(g - want[k]) <= tol)) {
                std::fprintf(stderr, "  %s(%g): got %.9g, want %.9g\n", names[k], x, g, want[k]);
                ok = false;
            }
        }
        const double wd = std::sqrt(double(x) * double(x) + 1.0) * 0.5;
        if (std::fabs(gotD[i] - wd) > 1e-12 * wd) std::fprintf(stderr, "  double(%g) wrong\n", x), ok = false;
    }
    if (!ok) ++g_mismatches;
    return ok;
}

struct Packed {
    unsigned a : 5;
    int b : 7;
    unsigned c : 20;
};

__global__ void bits(const unsigned* in, unsigned* out, unsigned long long* out64, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned x = in[i], y = in[(i + 1) % n];
    unsigned* o = out + i * 10;
    o[0] = __popc(x);
    o[1] = __clz(x);
    o[2] = __brev(x);
    o[3] = __ffs(int(x));
    o[4] = __byte_perm(x, y, 0x5140);
    o[5] = __funnelshift_l(x, y, i % 40);
    o[6] = __funnelshift_r(x, y, i % 40);
    Packed p;
    std::memcpy(&p, &x, sizeof p);
    o[7] = p.a + unsigned(p.b) + p.c;
    o[8] = __usad(x & 0xFFFF, y & 0xFFFF, 7u);
    o[9] = (x ^ y) & ~(x | (y >> 3));
    const unsigned long long w = (unsigned long long)x << 32 | y;
    out64[i * 3] = __popcll(w) + ((unsigned long long)__clzll(w >> (i % 64)) << 32);
    out64[i * 3 + 1] = __umul64hi(w, w + 12345);
    out64[i * 3 + 2] = __brevll(w);
}

static unsigned brev32(unsigned v) {
    unsigned r = 0;
    for (int i = 0; i < 32; ++i) r |= ((v >> i) & 1u) << (31 - i);
    return r;
}

static bool testBits() {
    const int n = 500;
    std::vector<unsigned> in(n);
    uint32_t s = 11;
    for (int i = 0; i < n; ++i) in[i] = lcg(s);
    in[0] = 0, in[1] = 1, in[2] = 0x80000000u, in[3] = 0xFFFFFFFFu;
    std::vector<unsigned> want(n * 10);
    std::vector<unsigned long long> want64(n * 3);
    for (int i = 0; i < n; ++i) {
        const unsigned x = in[i], y = in[(i + 1) % n];
        unsigned* o = &want[i * 10];
        o[0] = __builtin_popcount(x);
        o[1] = x ? __builtin_clz(x) : 32;
        o[2] = brev32(x);
        o[3] = x ? __builtin_ctz(x) + 1 : 0;
        const uint64_t bytes = uint64_t(y) << 32 | x;
        unsigned perm = 0;
        for (int k = 0; k < 4; ++k) perm |= unsigned((bytes >> (8 * ((0x5140 >> (4 * k)) & 7))) & 0xFF) << (8 * k);
        o[4] = perm;
        const unsigned sh = unsigned(i % 40) & 31;  // __funnelshift_l/r wrap the shift amount
        const uint64_t yx = uint64_t(y) << 32 | x;
        o[5] = unsigned((yx << sh) >> 32);
        o[6] = unsigned(yx >> sh);
        Packed p;
        std::memcpy(&p, &x, sizeof p);
        o[7] = p.a + unsigned(p.b) + p.c;
        const unsigned a = x & 0xFFFF, b = y & 0xFFFF;
        o[8] = (a > b ? a - b : b - a) + 7u;
        o[9] = (x ^ y) & ~(x | (y >> 3));
        const unsigned long long w = (unsigned long long)x << 32 | y;
        const unsigned long long shifted = w >> (i % 64);
        want64[i * 3] = __builtin_popcountll(w) + ((unsigned long long)(shifted ? __builtin_clzll(shifted) : 64) << 32);
        want64[i * 3 + 1] = (unsigned long long)(((unsigned __int128)w * (w + 12345)) >> 64);
        want64[i * 3 + 2] = (unsigned long long)brev32(unsigned(w)) << 32 | brev32(unsigned(w >> 32));
    }
    DeviceArray<unsigned> din(in), dout(n * 10);
    DeviceArray<unsigned long long> d64(n * 3);
    bits<<<(n + 127) / 128, 128>>>(din.ptr, dout.ptr, d64.ptr, n);
    bool ok = expectEqual("bits", dout.get(), want);
    return expectEqual("bits64", d64.get(), want64) && ok;
}

// ---- half precision ---------------------------------------------------------------------------

// __float2bfloat16 below sm_80 is integer code whose control flow exits two levels at once.
__global__ void toBf16(const float* in, unsigned short* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const __nv_bfloat16 b = __float2bfloat16(in[i]);
        out[i] = *reinterpret_cast<const unsigned short*>(&b);
    }
}

__global__ void halfOps(const __half* a, const __half2* b, __half* out, __half2* out2, unsigned* flags, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const __half x = a[i], y = a[(i + 1) % n];
    out[i * 4 + 0] = __hadd(x, y);
    out[i * 4 + 1] = __hmul(x, y);
    out[i * 4 + 2] = __hfma(x, y, __hneg(x));
    out[i * 4 + 3] = __hsub(__habs(x), y);
    const __half2 u = b[i], v = b[(i + 1) % n];
    out2[i * 2 + 0] = __hmul2(u, v);
    out2[i * 2 + 1] = __hfma2(u, v, u);
    flags[i] = unsigned(__hlt(x, y)) | (unsigned(__hbgt2(u, v)) << 1);
}

static bool testHalf() {
    const int n = 300;
    std::vector<__half> a(n);
    std::vector<__half2> b(n);
    for (int i = 0; i < n; ++i) {
        a[i] = __float2half(std::sin(float(i)) * float(i % 37));
        b[i] = __halves2half2(__float2half(float(i % 13) - 6.5f), __float2half(std::cos(float(i)) * 3.0f));
    }
    auto h2f = [](__half h) { return __half2float(h); };
    std::vector<float> want(n * 4), want2(n * 4);
    std::vector<unsigned> wantFlags(n);
    for (int i = 0; i < n; ++i) {
        const float x = h2f(a[i]), y = h2f(a[(i + 1) % n]);
        const float r[4] = {x + y, x * y, std::fma(x, y, -x), std::fabs(x) - y};
        for (int k = 0; k < 4; ++k) want[i * 4 + k] = h2f(__float2half(r[k]));
        const float ul = h2f(__low2half(b[i])), uh = h2f(__high2half(b[i]));
        const float vl = h2f(__low2half(b[(i + 1) % n])), vh = h2f(__high2half(b[(i + 1) % n]));
        const float r2[4] = {ul * vl, uh * vh, std::fma(ul, vl, ul), std::fma(uh, vh, uh)};
        for (int k = 0; k < 4; ++k) want2[i * 4 + k] = h2f(__float2half(r2[k]));
        wantFlags[i] = unsigned(x < y) | (unsigned(ul > vl && uh > vh) << 1);
    }
    DeviceArray<__half> da(a), dout(n * 4);
    DeviceArray<__half2> db(b), dout2(n * 2);
    DeviceArray<unsigned> dflags(n);
    halfOps<<<(n + 63) / 64, 64>>>(da.ptr, db.ptr, dout.ptr, dout2.ptr, dflags.ptr, n);
    std::vector<float> got, got2;
    for (__half h : dout.get()) got.push_back(h2f(h));
    for (__half2 h : dout2.get()) got2.push_back(h2f(__low2half(h))), got2.push_back(h2f(__high2half(h)));
    bool ok = expectEqual("half", got, want);
    ok &= expectEqual("half2", got2, want2);
    ok &= expectEqual("half compare", dflags.get(), wantFlags);

    const int m = 4096;
    std::vector<float> f(m);
    std::vector<unsigned short> wantBf(m);
    uint32_t seed = 9;
    for (int i = 0; i < m; ++i) {
        uint32_t bits = lcg(seed);
        std::memcpy(&f[i], &bits, 4);
        if (std::isnan(f[i])) f[i] = 1.0f;
        const __nv_bfloat16 b = __float2bfloat16(f[i]);
        std::memcpy(&wantBf[i], &b, 2);
    }
    DeviceArray<float> df(f);
    DeviceArray<unsigned short> dbf(m);
    toBf16<<<m / 256, 256>>>(df.ptr, dbf.ptr, m);
    return expectEqual("bf16 rounding", dbf.get(), wantBf) && ok;
}

// ---- half-precision atomics (compare-and-swap loops on the enclosing word) --------------------

__global__ void halfAtomics(__half2* pair, __half* halves, unsigned* bfPair) {
    const int t = threadIdx.x;
    atomicAdd(pair, __halves2half2(__float2half(1.0f), __float2half(0.5f)));
    atomicAdd(&halves[t % 4], __float2half(1.0f));  // two words, both of their halves
    const unsigned one = 0x3F803F80u;              // bf16x2 (1.0, 1.0)
    unsigned old;
    asm volatile("atom.global.add.noftz.bf16x2 %0, [%1], %2;" : "=r"(old) : "l"(bfPair), "r"(one) : "memory");
}

static float fromBits(unsigned u) {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

static bool testHalfAtomics() {
    DeviceArray<__half2> pair(1);
    DeviceArray<__half> halves(4);
    DeviceArray<unsigned> bfPair(1);
    halfAtomics<<<1, 256>>>(pair.ptr, halves.ptr, bfPair.ptr);
    const __half2 p = pair.get()[0];
    const std::vector<__half> h = halves.get();
    const unsigned bf = bfPair.get()[0];
    std::vector<float> got = {__half2float(__low2half(p)), __half2float(__high2half(p))};
    for (const __half x : h) got.push_back(__half2float(x));
    got.push_back(fromBits(bf << 16)), got.push_back(fromBits(bf & 0xFFFF0000u));
    return expectEqual("half atomics", got, std::vector<float>{256.0f, 128.0f, 64.0f, 64.0f, 64.0f, 64.0f, 256.0f, 256.0f});
}

// ---- block packing: small barrier-free blocks sharing a workgroup ------------------------------

__global__ void packedIds(unsigned* out, unsigned* perBlock) {
    const unsigned tid = threadIdx.y * blockDim.x + threadIdx.x;
    const unsigned block = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned threads = blockDim.x * blockDim.y;
    unsigned sum = tid;
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, o);  // over this warp
    out[block * threads + tid] = block * 100000u + tid * 100u + (threadIdx.x & 31) + (sum << 20) + gridDim.x * 7u;
    atomicAdd(&perBlock[block], 1u);
}

static bool testPacking() {
    bool ok = true;
    const unsigned shapes[][3] = {{32, 1, 37}, {64, 1, 5}, {96, 1, 3}, {32, 2, 11}, {32, 1, 1}, {128, 1, 9}};
    for (const auto& sh : shapes) {
        const unsigned bx = sh[0], by = sh[1], gx = sh[2], gy = 2, threads = bx * by, blocks = gx * gy;
        DeviceArray<unsigned> out(blocks * threads), perBlock(std::vector<unsigned>(blocks, 0));
        packedIds<<<dim3(gx, gy), dim3(bx, by)>>>(out.ptr, perBlock.ptr);
        std::vector<unsigned> want(blocks * threads);
        for (unsigned b = 0; b < blocks; ++b)
            for (unsigned t = 0; t < threads; ++t) {
                const unsigned warp0 = t & ~31u;
                unsigned sum = 0;
                for (unsigned l = 0; l < 32; ++l) sum += warp0 + l;
                want[b * threads + t] = b * 100000u + t * 100u + ((t % bx) & 31) + (sum << 20) + gx * 7u;
            }
        char name[64];
        std::snprintf(name, sizeof name, "packing %ux%u grid %ux%u", bx, by, gx, gy);
        ok &= expectEqual(name, out.get(), want);
        ok &= expectEqual(name, perBlock.get(), std::vector<unsigned>(blocks, threads));
    }
    return ok;
}

// ---- explicitly built CUDA graphs, stream-ordered allocation -----------------------------------

__global__ void graphStep(float* data, float scale, float add, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = data[i] * scale + add;
}

static bool testGraphs() {
    const int n = 1000;
    float* data = nullptr;
    CHECK(cudaMallocAsync(reinterpret_cast<void**>(&data), n * sizeof(float), 0));
    CHECK(cudaMemset(data, 0, n * sizeof(float)));
    cudaGraph_t graph;
    CHECK(cudaGraphCreate(&graph, 0));
    float scales[3] = {2.0f, 1.0f, 3.0f}, adds[3] = {1.0f, 5.0f, -1.0f};
    int count = n;
    cudaGraphNode_t nodes[3];
    for (int s = 0; s < 3; ++s) {
        void* args[] = {&data, &scales[s], &adds[s], &count};
        cudaKernelNodeParams p = {};
        p.func = reinterpret_cast<void*>(graphStep);
        p.gridDim = dim3((n + 127) / 128);
        p.blockDim = dim3(128);
        p.kernelParams = args;
        CHECK(cudaGraphAddKernelNode(&nodes[s], graph, s ? &nodes[s - 1] : nullptr, s ? 1 : 0, &p));
    }
    scales[0] = 100.0f;  // parameters were copied when the node was added
    cudaGraphExec_t exec;
    CHECK(cudaGraphInstantiate(&exec, graph, 0));
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    CHECK(cudaGraphLaunch(exec, stream));
    CHECK(cudaGraphLaunch(exec, stream));
    // x -> ((2x + 1) + 5) * 3 - 1: 0 -> 17 -> 119
    float add = 10.0f, scale = 1.0f;
    void* args[] = {&data, &scale, &add, &count};
    cudaKernelNodeParams p = {};
    p.func = reinterpret_cast<void*>(graphStep);
    p.gridDim = dim3((n + 127) / 128);
    p.blockDim = dim3(128);
    p.kernelParams = args;
    CHECK(cudaGraphExecKernelNodeSetParams(exec, nodes[1], &p));
    CHECK(cudaGraphLaunch(exec, stream));  // the middle node now adds 10: 119 -> (239 + 10) * 3 - 1 = 746
    CHECK(cudaStreamSynchronize(stream));
    std::vector<float> got(n);
    CHECK(cudaMemcpy(got.data(), data, n * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaGraphExecDestroy(exec));
    CHECK(cudaGraphDestroy(graph));
    bool ok = expectEqual("graph replay", got, std::vector<float>(n, 746.0f));

    // Stream capture, as llama.cpp uses it: <<<>>> launches and a memset are recorded, not run;
    // a re-capture with other parameters updates the executable graph.
    auto capture = [&](float s, float a) {
        cudaGraph_t g;
        CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));
        CHECK(cudaMemsetAsync(data, 0, n * sizeof(float), stream));
        graphStep<<<(n + 127) / 128, 128, 0, stream>>>(data, s, a, n);
        graphStep<<<(n + 127) / 128, 128, 0, stream>>>(data, s, a, n);
        CHECK(cudaStreamEndCapture(stream, &g));
        return g;
    };
    cudaGraph_t captured = capture(2.0f, 3.0f);  // 0 -> 3 -> 9
    CHECK(cudaStreamSynchronize(stream));
    CHECK(cudaMemcpy(got.data(), data, n * sizeof(float), cudaMemcpyDeviceToHost));
    ok &= expectEqual("nothing runs while capturing", got, std::vector<float>(n, 746.0f));
    CHECK(cudaGraphInstantiate(&exec, captured, 0));
    CHECK(cudaGraphLaunch(exec, stream));
    CHECK(cudaStreamSynchronize(stream));
    CHECK(cudaMemcpy(got.data(), data, n * sizeof(float), cudaMemcpyDeviceToHost));
    ok &= expectEqual("captured graph", got, std::vector<float>(n, 9.0f));
    cudaGraph_t recaptured = capture(1.0f, 4.0f);  // 0 -> 4 -> 8
    cudaGraphExecUpdateResultInfo info;
    CHECK(cudaGraphExecUpdate(exec, recaptured, &info));
    CHECK(cudaGraphLaunch(exec, stream));
    CHECK(cudaStreamSynchronize(stream));
    CHECK(cudaMemcpy(got.data(), data, n * sizeof(float), cudaMemcpyDeviceToHost));
    ok &= expectEqual("updated graph", got, std::vector<float>(n, 8.0f));
    CHECK(cudaGraphExecDestroy(exec));
    CHECK(cudaGraphDestroy(captured));
    CHECK(cudaGraphDestroy(recaptured));
    CHECK(cudaFreeAsync(data, stream));
    CHECK(cudaStreamDestroy(stream));
    return ok;
}

// ---- managed and mapped host memory -----------------------------------------------------------

__global__ void squarePlusOne(float* data, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = data[i] * data[i] + 1.0f;
}

static bool testManaged() {
    int supported = 0;  // host memory at a device address (llvmpipe has none)
    CHECK(cudaDeviceGetAttribute(&supported, cudaDevAttrManagedMemory, 0));
    if (!supported) return true;
    const int n = 50000;
    float* m = nullptr;
    CHECK(cudaMallocManaged(reinterpret_cast<void**>(&m), n * sizeof(float), cudaMemAttachGlobal));
    for (int i = 0; i < n; ++i) m[i] = float(i % 100);  // the CPU writes through the same pointer
    squarePlusOne<<<(n + 255) / 256, 256>>>(m, n);
    CHECK(cudaDeviceSynchronize());
    bool ok = true;
    for (int i = 0; i < n && ok; ++i) ok = m[i] == float((i % 100) * (i % 100) + 1);
    if (!ok) std::fprintf(stderr, "  managed: wrong result\n");
    cudaPointerAttributes a;
    CHECK(cudaPointerGetAttributes(&a, m));
    if (a.type != cudaMemoryTypeManaged || a.devicePointer != m || a.hostPointer != m) {
        std::fprintf(stderr, "  managed: pointer attributes type %d\n", int(a.type));
        ok = false;
    }
    CHECK(cudaFree(m));

    float* pinned = nullptr;  // zero-copy: the kernel uses the host allocation directly
    CHECK(cudaHostAlloc(reinterpret_cast<void**>(&pinned), n * sizeof(float), cudaHostAllocMapped));
    for (int i = 0; i < n; ++i) pinned[i] = 2.0f;
    float* device = nullptr;
    CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&device), pinned, 0));
    squarePlusOne<<<(n + 255) / 256, 256>>>(device, n);
    CHECK(cudaDeviceSynchronize());
    for (int i = 0; i < n && ok; ++i) ok = pinned[i] == 5.0f;
    if (!ok) std::fprintf(stderr, "  mapped pinned memory: wrong result\n");
    CHECK(cudaPointerGetAttributes(&a, pinned));
    if (a.type != cudaMemoryTypeHost || a.devicePointer != pinned) {
        std::fprintf(stderr, "  pinned: pointer attributes type %d\n", int(a.type));
        ok = false;
    }
    CHECK(cudaFreeHost(pinned));
    int dummy = 0;
    CHECK(cudaPointerGetAttributes(&a, &dummy));
    if (a.type != cudaMemoryTypeUnregistered) ok = false;
    if (!ok) ++g_mismatches;
    return ok;
}

// ---- cooperative groups: grid-wide synchronization ----------------------------------------------

// Each round every thread reads the element another block wrote in the previous round: wrong
// without a working grid.sync().
__global__ void gridRotate(int* a, int* b, int rounds, int* valid) {
    cg::grid_group grid = cg::this_grid();
    const int n = int(grid.size());
    const int i = int(grid.thread_rank());
    if (i == 0) *valid = grid.is_valid();
    for (int r = 0; r < rounds; ++r) {
        b[i] = a[(i + blockDim.x + 1) % n] + 1;
        grid.sync();
        int* t = a;
        a = b;
        b = t;
    }
}

__global__ void gridValid(int* valid) {
    if (threadIdx.x == 0 && blockIdx.x == 0) *valid = cg::this_grid().is_valid();
}

static bool testCooperative() {
    int coop = 0, sms = 0, perSm = 0;
    CHECK(cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, 0));
    CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    const int block = 128, rounds = 50;
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSm, gridRotate, block, 0));
    const int blocks = sms * std::min(perSm, 2), n = blocks * block;
    std::vector<int> host(n);
    for (int i = 0; i < n; ++i) host[i] = (i * 7919) % 1000;
    int *a, *b, *valid;
    CHECK(cudaMalloc(&a, n * sizeof(int)));
    CHECK(cudaMalloc(&b, n * sizeof(int)));
    CHECK(cudaMalloc(&valid, sizeof(int)));
    CHECK(cudaMemcpy(a, host.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    int roundsArg = rounds;
    void* args[] = {&a, &b, &roundsArg, &valid};
    CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(gridRotate), blocks, block, args, 0, nullptr));
    std::vector<int> got(n);
    int isValid = 0;
    CHECK(cudaMemcpy(got.data(), rounds % 2 ? b : a, n * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(&isValid, valid, sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<int> want = host, next(n);
    for (int r = 0; r < rounds; ++r) {
        for (int i = 0; i < n; ++i) next[i] = want[(i + block + 1) % n] + 1;
        want.swap(next);
    }
    bool ok = coop == 1 && isValid == 1 && expectEqual("grid.sync", got, want);

    // the same through cudaLaunchKernelEx's cooperative attribute
    CHECK(cudaMemcpy(a, host.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    cudaLaunchAttribute attr;
    attr.id = cudaLaunchAttributeCooperative;
    attr.val.cooperative = 1;
    cudaLaunchConfig_t config = {};
    config.gridDim = blocks;
    config.blockDim = block;
    config.attrs = &attr;
    config.numAttrs = 1;
    CHECK(cudaLaunchKernelExC(&config, reinterpret_cast<void*>(gridRotate), args));
    CHECK(cudaMemcpy(got.data(), rounds % 2 ? b : a, n * sizeof(int), cudaMemcpyDeviceToHost));
    ok = ok && expectEqual("cudaLaunchKernelEx cooperative", got, want);

    gridValid<<<1, 32>>>(valid);  // an ordinary launch has no grid group
    CHECK(cudaMemcpy(&isValid, valid, sizeof(int), cudaMemcpyDeviceToHost));
    ok = ok && isValid == 0;

    // more blocks than can be resident at once
    const cudaError_t e = cudaLaunchCooperativeKernel(reinterpret_cast<void*>(gridRotate), sms * perSm + 1, block,
                                                      args, 0, nullptr);
    ok = ok && e == cudaErrorCooperativeLaunchTooLarge;
    cudaGetLastError();
    if (!ok) std::fprintf(stderr, "  cooperative: attribute %d, is_valid %d, too large -> %d\n", coop, isValid, int(e));
    CHECK(cudaFree(a));
    CHECK(cudaFree(b));
    CHECK(cudaFree(valid));
    if (!ok) ++g_mismatches;
    return ok;
}

// ---- several devices (VULKANIZER_ALLOW_CPU=1 adds llvmpipe next to the GPU) ----------------------

static bool testMultiGpu() {
    int count = 0;
    CHECK(cudaGetDeviceCount(&count));
    if (count < 2) return true;
    const int n = 4096;
    std::vector<float> host(n);
    for (int i = 0; i < n; ++i) host[i] = float(i % 7);
    float *d0, *d1;
    CHECK(cudaSetDevice(0));
    CHECK(cudaMalloc(&d0, n * sizeof(float)));
    CHECK(cudaMemcpy(d0, host.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    squarePlusOne<<<(n + 255) / 256, 256>>>(d0, n);  // x*x+1 on device 0
    cudaEvent_t done0;
    CHECK(cudaEventCreateWithFlags(&done0, cudaEventDisableTiming));
    CHECK(cudaEventRecord(done0, nullptr));

    CHECK(cudaSetDevice(1));
    cudaStream_t s1;
    CHECK(cudaStreamCreate(&s1));
    CHECK(cudaMalloc(&d1, n * sizeof(float)));
    CHECK(cudaStreamWaitEvent(s1, done0, 0));  // another device's event
    CHECK(cudaMemcpyPeerAsync(d1, 1, d0, 0, n * sizeof(float), s1));
    squarePlusOne<<<(n + 255) / 256, 256, 0, s1>>>(d1, n);  // and again on device 1
    std::vector<float> got(n);
    CHECK(cudaMemcpyAsync(got.data(), d1, n * sizeof(float), cudaMemcpyDeviceToHost, s1));
    CHECK(cudaStreamSynchronize(s1));
    std::vector<float> want(n);
    for (int i = 0; i < n; ++i) {
        const float once = host[i] * host[i] + 1.0f;
        want[i] = once * once + 1.0f;
    }
    bool ok = expectEqual("peer copy between devices", got, want);

    CHECK(cudaMemcpyPeer(d0, 0, d1, 1, n * sizeof(float)));  // back to device 0
    CHECK(cudaSetDevice(0));
    CHECK(cudaMemcpy(got.data(), d0, n * sizeof(float), cudaMemcpyDeviceToHost));
    ok = expectEqual("peer copy back", got, want) && ok;

    int access = 1;
    CHECK(cudaDeviceCanAccessPeer(&access, 0, 1));
    ok = ok && access == 0 && cudaDeviceEnablePeerAccess(1, 0) == cudaErrorPeerAccessUnsupported;
    cudaGetLastError();
    CHECK(cudaFree(d0));
    CHECK(cudaSetDevice(1));
    CHECK(cudaFree(d1));
    CHECK(cudaStreamDestroy(s1));
    CHECK(cudaSetDevice(0));
    CHECK(cudaEventDestroy(done0));
    if (!ok) ++g_mismatches;
    return ok;
}

// ---- fma accuracy (emulated where the driver's Fma is not fused) ---------------------------------

__global__ void fmaKernel(const float* a, const float* b, const float* c, float* out, float* err, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = fmaf(a[i], b[i], c[i]);
    const float p = a[i] * b[i];
    err[i] = fmaf(a[i], b[i], -p);  // error-free transformation: the product's exact error
}

static bool testFma() {
    const int n = 1 << 14;
    std::vector<float> a(n), b(n), c(n);
    uint32_t seed = 77;
    auto uniform = [&] { return float(lcg(seed) >> 8) / float(1 << 24); };
    for (int i = 0; i < n; ++i) {
        a[i] = std::ldexp(1.0f + uniform(), int(lcg(seed) % 40) - 20) * (lcg(seed) & 1 ? -1.0f : 1.0f);
        b[i] = std::ldexp(1.0f + uniform(), int(lcg(seed) % 40) - 20);
        // c near -a*b (cancellation), or of unrelated magnitude
        c[i] = i % 2 ? -a[i] * b[i] * (1.0f + std::ldexp(uniform(), -10))
                     : std::ldexp(uniform() - 0.5f, int(lcg(seed) % 60) - 30);
    }
    DeviceArray<float> da(a), db(b), dc(c), dout(n), derr(n);
    fmaKernel<<<n / 256, 256>>>(da.ptr, db.ptr, dc.ptr, dout.ptr, derr.ptr, n);
    const std::vector<float> got = dout.get(), gotErr = derr.get();
    int bad = 0;
    for (int i = 0; i < n && bad < 5; ++i) {
        const float want = std::fma(a[i], b[i], c[i]), wantErr = std::fma(a[i], b[i], -(a[i] * b[i]));
        const float ulp = std::fabs(std::nextafter(want, INFINITY) - want);
        if (gotErr[i] != wantErr || !(std::fabs(got[i] - want) <= ulp)) {
            std::fprintf(stderr, "  fma(%a, %a, %a) = %a (want %a), error %a (want %a)\n", a[i], b[i], c[i], got[i],
                         want, gotErr[i], wantErr);
            ++bad;
        }
    }
    return bad == 0;
}

// ---- device function calls (inlined by Vulkanizer) -------------------------------------------------

struct Params {
    float scale;
    int offset;
    double bias;
};

__device__ __noinline__ float applyParams(float x, Params p) { return x * p.scale + float(p.offset) + float(p.bias); }
__device__ __noinline__ void accumulate(float* values, int n) {
    for (int i = 0; i < n; ++i) values[i] += float(i);
}
__device__ __noinline__ int twice(int v) { return v * 2; }
__device__ __noinline__ int quad(int v) { return twice(twice(v)) + 1; }
__device__ __noinline__ void put(float* where, int i, float v) { where[i] = v; }

__global__ void calls(const float* in, float* out, int* ints) {
    __shared__ float stage[64];
    float local[4] = {1, 2, 3, 4};
    accumulate(local, 4);  // pointer into this thread's local memory
    const int t = threadIdx.x;
    put(stage, t, in[t] * 2.0f);  // pointer into shared memory
    __syncthreads();
    out[t] = applyParams(stage[63 - t], Params{0.5f, 3, 0.25}) + local[t & 3];
    ints[t] = quad(t);
}

static bool testCalls() {
    std::vector<float> in(64), want(64);
    std::vector<int> wantInts(64);
    for (int t = 0; t < 64; ++t) in[t] = float(t) - 10.0f;
    for (int t = 0; t < 64; ++t) {
        const float local[4] = {1, 3, 5, 7};
        want[t] = in[63 - t] * 2.0f * 0.5f + 3.0f + 0.25f + local[t & 3];
        wantInts[t] = t * 4 + 1;
    }
    DeviceArray<float> din(in), dout(64);
    DeviceArray<int> dints(64);
    calls<<<1, 64>>>(din.ptr, dout.ptr, dints.ptr);
    bool ok = expectEqual("calls", dout.get(), want);
    return expectEqual("nested calls", dints.get(), wantInts) && ok;
}

// ---- launch limits: huge grids, parameter blocks larger than the push constants ---------------

__global__ void gridIndex(unsigned* out, unsigned n) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = i ^ gridDim.x;
}

struct Big {
    float w[100];
    int count;
};

template <class P>
__global__ void sumParams(P b, float* out) {
    float s = 0;
    for (int k = 0; k < b.count; ++k) s += b.w[k] * float(threadIdx.x + 1);  // dynamic index
    out[threadIdx.x] = s;
}

struct Medium {  // fits in the 256-byte push constants
    float w[40];
    int count;
};

static bool testLimits() {
    const unsigned blocks = 200000, block = 32, n = blocks * block - 5;  // more blocks than llvmpipe dispatches
    DeviceArray<unsigned> out(n);
    gridIndex<<<blocks, block>>>(out.ptr, n);
    std::vector<unsigned> want(n);
    for (unsigned i = 0; i < n; ++i) want[i] = i ^ blocks;
    bool ok = expectEqual("grid", out.get(), want);

    auto check = [&](auto params, int count, const char* what) {
        for (int k = 0; k < count; ++k) params.w[k] = float(k) * 0.5f;
        params.count = count;
        DeviceArray<float> sums(64);
        sumParams<<<1, 64>>>(params, sums.ptr);
        std::vector<float> want(64);
        for (int t = 0; t < 64; ++t) {
            float s = 0;
            for (int k = 0; k < count; ++k) s += params.w[k] * float(t + 1);
            want[t] = s;
        }
        return expectEqual(what, sums.get(), want);
    };
    ok &= check(Medium{}, 40, "medium params");
    return check(Big{}, 100, "big params") && ok;
}

// -----------------------------------------------------------------------------------------------

struct Test {
    const char* name;
    bool (*fn)();
};

int main(int argc, char** argv) {
    const Test tests[] = {{"reduce", testReduce},   {"syncreduce", testSyncReductions}, {"exitbarrier", testExitBeforeBarrier},   {"matmul", testMatmul}, {"histogram", testHistogram},
                          {"atomics", testAtomics}, {"warp", testWarps},    {"variables", testVariables},
                          {"math", testMath},       {"bits", testBits},     {"calls", testCalls},
                          {"limits", testLimits}, {"half", testHalf},
                          {"fma", testFma}, {"packing", testPacking},
                          {"graphs", testGraphs}, {"managed", testManaged},
                          {"halfatomics", testHalfAtomics}, {"cooperative", testCooperative}, {"multigpu", testMultiGpu}};
    int failed = 0;
    for (const Test& t : tests) {
        bool selected = argc < 2;
        for (int i = 1; i < argc; ++i) selected |= std::strcmp(argv[i], t.name) == 0;
        if (!selected) continue;
        const bool ok = t.fn();
        const cudaError_t e = cudaGetLastError();
        std::printf("%-10s %s%s%s\n", t.name, ok && e == cudaSuccess ? "ok" : "FAILED", e ? ": " : "",
                    e ? cudaGetErrorString(e) : "");
        failed += !(ok && e == cudaSuccess);
    }
    return failed ? 1 : 0;
}
