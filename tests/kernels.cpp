// Runs every kernel of data/kernels.ptx and compares against a CPU reference.
//   kernels            all kernels
//   kernels saxpy      only the named ones

#include "test_util.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <vector>

namespace {

CUmodule g_module;
int g_failures = 0;

CUfunction kernel(const char* name) {
    CUfunction f;
    CHECK_CU(cuModuleGetFunction(&f, g_module, name));
    return f;
}

template <class T>
CUdeviceptr upload(const std::vector<T>& v) {
    CUdeviceptr p;
    CHECK_CU(cuMemAlloc(&p, v.size() * sizeof(T)));
    CHECK_CU(cuMemcpyHtoD(p, v.data(), v.size() * sizeof(T)));
    return p;
}

template <class T>
CUdeviceptr alloc(size_t count) {
    CUdeviceptr p;
    CHECK_CU(cuMemAlloc(&p, count * sizeof(T)));
    CHECK_CU(cuMemsetD8(p, 0xCD, count * sizeof(T)));
    return p;
}

template <class T>
std::vector<T> download(CUdeviceptr p, size_t count) {
    std::vector<T> v(count);
    CHECK_CU(cuMemcpyDtoH(v.data(), p, count * sizeof(T)));
    CHECK_CU(cuMemFree(p));
    return v;
}

void launch(CUfunction f, unsigned gx, unsigned gy, unsigned bx, unsigned by, void** args) {
    CHECK_CU(cuLaunchKernel(f, gx, gy, 1, bx, by, 1, 0, nullptr, args, nullptr));
    CHECK_CU(cuCtxSynchronize());
}

unsigned blocks(int n, unsigned block) { return (unsigned(n) + block - 1) / block; }

bool sameFloat(float a, float b) {
    if (std::isnan(a) || std::isnan(b)) return std::isnan(a) && std::isnan(b);
    return a == b || std::fabs(a - b) <= 1e-6f * std::max(std::fabs(a), std::fabs(b));
}

template <class T, class Eq>
bool compare(const char* what, const std::vector<T>& got, const std::vector<T>& want, Eq eq) {
    for (size_t i = 0; i < want.size(); ++i) {
        if (!eq(got[i], want[i])) {
            std::fprintf(stderr, "  %s: mismatch at %zu: got %s, want %s\n", what, i, std::to_string(got[i]).c_str(),
                         std::to_string(want[i]).c_str());
            return false;
        }
    }
    return true;
}
template <class T>
bool compare(const char* what, const std::vector<T>& got, const std::vector<T>& want) {
    return compare(what, got, want, [](const T& a, const T& b) { return a == b; });
}

std::vector<int> testInts(int n) {
    std::vector<int> v(n);
    const int special[] = {0, 1, -1, 7, -7, 8, -8, 100, 101, -100, std::numeric_limits<int>::max(),
                           std::numeric_limits<int>::min() + 1, 0x12345678, -0x12345678};
    uint32_t s = 12345;
    for (int i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        v[i] = i < int(std::size(special)) ? special[i] : int(s);
    }
    return v;
}

// ---------------------------------------------------------------------------------------------

bool saxpy() {
    int n = 10007;
    float a = 2.5f;
    std::vector<float> x(n), y(n), want(n);
    for (int i = 0; i < n; ++i) {
        x[i] = float(i) * 0.25f;
        y[i] = float(n - i);
        want[i] = std::fma(x[i], a, y[i]);
    }
    CUdeviceptr dx = upload(x), dy = upload(y);
    void* args[] = {&a, &dx, &dy, &n};
    launch(kernel("saxpy"), blocks(n, 128), 1, 128, 1, args);
    CHECK_CU(cuMemFree(dx));
    return compare("y", download<float>(dy, n), want, sameFloat);
}

bool scaleGridStride() {
    int n = 100000;
    float s = -0.5f;
    std::vector<float> d(n), want(n);
    for (int i = 0; i < n; ++i) {
        d[i] = float(i);
        want[i] = d[i] * s;
    }
    CUdeviceptr dd = upload(d);
    void* args[] = {&dd, &n, &s};
    launch(kernel("scaleGridStride"), 7, 1, 64, 1, args);  // far fewer threads than elements
    return compare("data", download<float>(dd, n), want, sameFloat);
}

bool index2d() {
    int w = 300, h = 77;
    CUdeviceptr out = alloc<int>(size_t(w) * h);
    void* args[] = {&out, &w, &h};
    launch(kernel("index2d"), blocks(w, 16), blocks(h, 8), 16, 8, args);
    std::vector<int> want(size_t(w) * h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) want[size_t(y) * w + x] = (y << 16) | x;
    return compare("out", download<int>(out, want.size()), want);
}

bool intOps() {
    int n = 1000;
    short bias = -1234;
    std::vector<int> in = testInts(n);
    std::vector<long long> want(size_t(n) * 8);
    for (int i = 0; i < n; ++i) {
        const int v = in[i];
        long long* o = &want[size_t(i) * 8];
        o[0] = (long long)v * 3000000000LL;
        o[1] = v / 7;
        o[2] = v % 7;
        o[3] = std::min(v, 100);
        o[4] = v >> 3;
        o[5] = v < 0 ? 1 : -1;
        o[6] = (long long)(((int64_t)v * 0x55555556LL) >> 32);
        o[7] = (unsigned)(((unsigned)v << 4) | ((unsigned)v >> 28)) + unsigned(int(bias));
        o[7] = (long long)(uint32_t)o[7];
    }
    CUdeviceptr din = upload(in), dout = alloc<long long>(want.size());
    void* args[] = {&dout, &din, &n, &bias};
    launch(kernel("intOps"), blocks(n, 64), 1, 64, 1, args);
    CHECK_CU(cuMemFree(din));
    return compare("out", download<long long>(dout, want.size()), want);
}

bool collatz() {
    int n = 5000;
    std::vector<int> in(n), want(n);
    for (int i = 0; i < n; ++i) {
        in[i] = i + 1;
        int x = in[i], c = 0;
        while (x != 1 && c < 1000) {
            x = (x & 1) ? 3 * x + 1 : x >> 1;
            ++c;
        }
        want[i] = c;
    }
    CUdeviceptr din = upload(in), dout = alloc<int>(n);
    void* args[] = {&dout, &din, &n};
    launch(kernel("collatz"), blocks(n, 256), 1, 256, 1, args);
    CHECK_CU(cuMemFree(din));
    return compare("steps", download<int>(dout, n), want);
}

int nestedRef(int i) {
    int sum = 0;
    for (int a = 0; a < i % 5 + 1; ++a) {
        for (int b = 0; b < 10; ++b) {
            if (b == a + 3) break;
            if (b == 1) continue;
            if (b == i % 11) return sum;
            sum += b;
        }
        sum = sum * 3 + a;
    }
    return sum;
}

bool nested() {
    int n = 2000;
    std::vector<int> want(n);
    for (int i = 0; i < n; ++i) want[i] = nestedRef(i);
    CUdeviceptr dout = alloc<int>(n);
    void* args[] = {&dout, &n};
    launch(kernel("nested"), blocks(n, 96), 1, 96, 1, args);
    return compare("out", download<int>(dout, n), want);
}

float truncSat(float x) {
    if (std::isnan(x)) return 0.0f;
    if (x >= 2147483648.0f) return float(std::numeric_limits<int>::max());
    if (x <= -2147483648.0f) return float(std::numeric_limits<int>::min());
    return float(int(x));
}

bool floatOps() {
    std::vector<float> in = {0.0f, -0.0f, 1.5f, -1.5f, 2.75f, -2.75f, 0.25f, -0.25f, 1e10f, -1e10f, 123456.789f,
                             std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::quiet_NaN(), 3.0f, 1e-30f};
    for (int i = 0; i < 500; ++i) in.push_back(std::sin(float(i)) * float(i));
    int n = int(in.size());
    std::vector<float> want(size_t(n) * 8);
    for (int i = 0; i < n; ++i) {
        const float x = in[i];
        float* o = &want[size_t(i) * 8];
        o[0] = truncSat(x);
        o[1] = std::floor(x);
        o[2] = std::fabs(x);
        o[3] = -x;
        o[4] = std::fmin(x, 0.5f);
        o[5] = std::fmax(x, -0.5f);
        o[6] = x / 3.0f;
        o[7] = float(double(x) * 2.0);
    }
    CUdeviceptr din = upload(in), dout = alloc<float>(want.size());
    void* args[] = {&dout, &din, &n};
    launch(kernel("floatOps"), blocks(n, 64), 1, 64, 1, args);
    CHECK_CU(cuMemFree(din));
    return compare("out", download<float>(dout, want.size()), want, sameFloat);
}

bool vec4() {
    int n = 333;
    std::vector<float> in(size_t(n) * 4), want(size_t(n) * 4);
    std::vector<unsigned long long> wantPacked(n);
    std::vector<uint32_t> wantSwapped(size_t(n) * 2);
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < 4; ++k) in[size_t(i) * 4 + k] = float(i * 4 + k) * 0.5f;
        for (int k = 0; k < 4; ++k) want[size_t(i) * 4 + k] = in[size_t(i) * 4 + (3 - k)] + 1.0f;
        uint32_t x, y;
        std::memcpy(&x, &in[size_t(i) * 4], 4);
        std::memcpy(&y, &in[size_t(i) * 4 + 1], 4);
        wantPacked[i] = x | (unsigned long long)y << 32;
        wantSwapped[size_t(i) * 2] = y;
        wantSwapped[size_t(i) * 2 + 1] = x;
    }
    CUdeviceptr din = upload(in), dout = alloc<float>(want.size()), dpacked = alloc<unsigned long long>(n),
                dswapped = alloc<uint32_t>(wantSwapped.size());
    void* args[] = {&dout, &dpacked, &dswapped, &din, &n};
    launch(kernel("vec4"), blocks(n, 32), 1, 32, 1, args);
    CHECK_CU(cuMemFree(din));
    bool ok = compare("out", download<float>(dout, want.size()), want, sameFloat);
    ok &= compare("packed", download<unsigned long long>(dpacked, n), wantPacked);
    ok &= compare("swapped", download<uint32_t>(dswapped, wantSwapped.size()), wantSwapped);
    return ok;
}

struct Test {
    const char* name;
    bool (*fn)();
};

const Test kTests[] = {
    {"saxpy", saxpy},     {"scaleGridStride", scaleGridStride},
    {"index2d", index2d}, {"intOps", intOps},
    {"collatz", collatz}, {"nested", nested},
    {"floatOps", floatOps}, {"vec4", vec4},
};

}  // namespace

int main(int argc, char** argv) {
    CHECK_CU(cuInit(0));
    CUdevice dev;
    CHECK_CU(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CHECK_CU(cuCtxCreate(&ctx, 0, dev));
    CHECK_CU(cuModuleLoad(&g_module, VULKANIZER_TEST_DATA_DIR "/kernels.ptx"));

    for (const Test& t : kTests) {
        bool selected = argc < 2;
        for (int i = 1; i < argc; ++i) selected |= std::strcmp(argv[i], t.name) == 0;
        if (!selected) continue;
        const bool ok = t.fn();
        std::printf("%-16s %s\n", t.name, ok ? "ok" : "FAILED");
        if (!ok) ++g_failures;
    }

    CHECK_CU(cuModuleUnload(g_module));
    CHECK_CU(cuCtxDestroy(ctx));
    return g_failures ? 1 : 0;
}
