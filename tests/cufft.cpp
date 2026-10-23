// Vulkanizer's libcufft against a naive f64 DFT on the CPU: complex and real transforms of rank 1
// to 3, batched, both directions, power-of-two, mixed-radix and prime (Bluestein) lengths, in
// place and out of place, with advanced layouts; f32, f64 (where the device has it) and f16
// through the Xt API.

#include "test_util.h"

#include <cufft/cufft.h>

#include <cmath>
#include <complex>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CHECK_FFT(call)                                                                            \
    do {                                                                                           \
        cufftResult s_ = (call);                                                                   \
        if (s_ != CUFFT_SUCCESS) {                                                                 \
            std::fprintf(stderr, "%s:%d: %s failed: %d\n", __FILE__, __LINE__, #call, int(s_));    \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

namespace {

using Complex = std::complex<double>;
enum class Kind { C2C, R2C, C2R };
enum class Precision { F32, F64, F16 };

std::mt19937 rng(11);
int g_failures = 0;

// The transform of `x` (batch x n[0] x n[1] x n[2], n[2] fastest) along every axis, unnormalized.
std::vector<Complex> dft(const std::vector<Complex>& x, const int n[3], int64_t batch, int sign) {
    std::vector<Complex> a = x, b(x.size());
    for (int d = 0; d < 3; ++d) {
        if (n[d] == 1) continue;
        int64_t inner = 1;
        for (int k = d + 1; k < 3; ++k) inner *= n[k];
        const int64_t lines = int64_t(a.size()) / n[d];
        for (int64_t l = 0; l < lines; ++l) {
            const int64_t base = (l / inner) * n[d] * inner + l % inner;
            for (int k = 0; k < n[d]; ++k) {
                Complex s = 0;
                for (int j = 0; j < n[d]; ++j)
                    s += a[base + j * inner] * std::polar(1.0, sign * 2 * M_PI * double((int64_t(j) * k) % n[d]) / n[d]);
                b[base + k * inner] = s;
            }
        }
        std::swap(a, b);
    }
    (void)batch;
    return a;
}

size_t elementBytes(Precision p, bool complex) {
    const size_t real = p == Precision::F64 ? 8 : p == Precision::F32 ? 4 : 2;
    return complex ? 2 * real : real;
}

void storeValue(std::vector<unsigned char>& buf, int64_t index, Precision p, double v) {
    if (p == Precision::F64) std::memcpy(&buf[index * 8], &v, 8);
    else if (p == Precision::F32) {
        const float f = float(v);
        std::memcpy(&buf[index * 4], &f, 4);
    } else {
        const _Float16 h = _Float16(v);
        std::memcpy(&buf[index * 2], &h, 2);
    }
}

double loadValue(const std::vector<unsigned char>& buf, int64_t index, Precision p) {
    if (p == Precision::F64) {
        double v;
        std::memcpy(&v, &buf[index * 8], 8);
        return v;
    }
    if (p == Precision::F32) {
        float f;
        std::memcpy(&f, &buf[index * 4], 4);
        return f;
    }
    _Float16 h;
    std::memcpy(&h, &buf[index * 2], 2);
    return double(h);
}

struct Case {
    std::string name;
    Kind kind;
    Precision precision;
    std::vector<int> dims;  // rank 1..3, slowest first
    int batch = 1;
    int direction = CUFFT_FORWARD;
    bool inPlace = false;
    int istride = 0;  // > 0: an advanced layout with these strides (embeds = the logical sizes, padded by 1
    int ostride = 0;  //  per batch entry)
    double tolerance = 1e-4;
};

void run(const Case& c) {
    const int rank = int(c.dims.size());
    int n[3] = {1, 1, 1};
    for (int k = 0; k < rank; ++k) n[3 - rank + k] = c.dims[k];
    const int half = n[2] / 2 + 1;
    const int64_t full = int64_t(n[0]) * n[1] * n[2], halfCount = int64_t(n[0]) * n[1] * half;
    const bool inComplex = c.kind != Kind::R2C, outComplex = c.kind != Kind::C2R;
    // logical shapes of input and output (the half dimension for real transforms)
    const int inLast = c.kind == Kind::C2R ? half : n[2], outLast = c.kind == Kind::R2C ? half : n[2];

    // the reference: random input, expected output, both over the full array
    std::uniform_real_distribution<double> u(-1, 1);
    std::vector<Complex> x(size_t(full * c.batch));
    for (Complex& v : x) v = c.kind == Kind::C2C ? Complex(u(rng), u(rng)) : Complex(u(rng), 0);
    if (c.precision == Precision::F16)  // round the input to what f16 holds
        for (Complex& v : x) v = Complex(double(_Float16(v.real())), double(_Float16(v.imag())));
    std::vector<Complex> input = x, expected;
    if (c.kind == Kind::C2R) {
        input = dft(x, n, c.batch, -1);  // Hermitian; the output is full * x
        expected.resize(x.size());
        for (size_t i = 0; i < x.size(); ++i) expected[i] = x[i] * double(full);
    } else {
        expected = dft(x, n, c.batch, c.kind == Kind::C2C && c.direction == CUFFT_INVERSE ? 1 : -1);
    }

    // the caller's layouts: element strides of the padded dimensions and the batch distance
    struct Lay {
        int64_t stride[3], dist, size;
    };
    const auto makeLayout = [&](int last, int stride, bool realInPlace) {
        Lay l;
        const int64_t lastStorage = realInPlace ? 2 * half : last;
        const int64_t s = stride > 0 ? stride : 1;
        l.stride[2] = s;
        l.stride[1] = s * lastStorage;
        l.stride[0] = l.stride[1] * n[1];
        l.dist = stride > 0 ? l.stride[0] * n[0] + 1 : l.stride[0] * n[0];
        l.size = l.dist * c.batch;
        return l;
    };
    const Lay inL = makeLayout(inLast, c.istride, c.inPlace && c.kind == Kind::R2C);
    const Lay outL = makeLayout(outLast, c.ostride, c.inPlace && c.kind == Kind::C2R);
    const size_t inBytes = size_t(inL.size) * elementBytes(c.precision, inComplex);
    const size_t outBytes = size_t(outL.size) * elementBytes(c.precision, outComplex);
    std::vector<unsigned char> inBuf(inBytes);
    for (int b = 0; b < c.batch; ++b)
        for (int i0 = 0; i0 < n[0]; ++i0)
            for (int i1 = 0; i1 < n[1]; ++i1)
                for (int i2 = 0; i2 < inLast; ++i2) {
                    const Complex v = input[((b * int64_t(n[0]) + i0) * n[1] + i1) * n[2] + i2];
                    const int64_t o = b * inL.dist + i0 * inL.stride[0] + i1 * inL.stride[1] + i2 * inL.stride[2];
                    if (inComplex) {
                        storeValue(inBuf, 2 * o, c.precision, v.real());
                        storeValue(inBuf, 2 * o + 1, c.precision, v.imag());
                    } else {
                        storeValue(inBuf, o, c.precision, v.real());
                    }
                }
    CUdeviceptr din, dout;
    CHECK_CU(cuMemAlloc(&din, std::max(inBytes, outBytes)));
    CHECK_CU(cuMemcpyHtoD(din, inBuf.data(), inBytes));
    if (c.inPlace) dout = din;
    else CHECK_CU(cuMemAlloc(&dout, outBytes));

    cufftHandle plan;
    CHECK_FFT(cufftCreate(&plan));
    size_t work = 0;
    long long dims[3], inembed[3], onembed[3];
    for (int k = 0; k < rank; ++k) dims[k] = c.dims[k];
    for (int k = 0; k < rank; ++k) inembed[k] = onembed[k] = c.dims[k];
    inembed[rank - 1] = c.inPlace && c.kind == Kind::R2C ? 2 * half : inLast;
    onembed[rank - 1] = c.inPlace && c.kind == Kind::C2R ? 2 * half : outLast;
    const bool advanced = c.istride > 0;
    cufftResult made;
    if (c.precision == Precision::F16) {
        const cudaDataType real = CUDA_R_16F, cplx = CUDA_C_16F;
        made = cufftXtMakePlanMany(plan, rank, dims, advanced ? inembed : nullptr, inL.stride[2], inL.dist,
                                   inComplex ? cplx : real, advanced ? onembed : nullptr, outL.stride[2], outL.dist,
                                   outComplex ? cplx : real, c.batch, &work, cplx);
    } else {
        const bool f64 = c.precision == Precision::F64;
        const cufftType type = c.kind == Kind::C2C ? (f64 ? CUFFT_Z2Z : CUFFT_C2C)
                               : c.kind == Kind::R2C ? (f64 ? CUFFT_D2Z : CUFFT_R2C)
                                                     : (f64 ? CUFFT_Z2D : CUFFT_C2R);
        made = cufftMakePlanMany64(plan, rank, dims, advanced ? inembed : nullptr, inL.stride[2], inL.dist,
                                   advanced ? onembed : nullptr, outL.stride[2], outL.dist, type, c.batch, &work);
    }
    if (made == CUFFT_NOT_SUPPORTED && c.precision == Precision::F64) {
        std::printf("  %-36s skipped (no f64)\n", c.name.c_str());
        CHECK_FFT(cufftDestroy(plan));
        cuMemFree(din);
        if (!c.inPlace) cuMemFree(dout);
        return;
    }
    CHECK_FFT(made);
    if (c.precision == Precision::F16 || c.kind == Kind::C2C) {
        CHECK_FFT(cufftXtExec(plan, (void*)din, (void*)dout, c.direction));
    } else if (c.precision == Precision::F32) {
        if (c.kind == Kind::R2C) CHECK_FFT(cufftExecR2C(plan, (cufftReal*)din, (cufftComplex*)dout));
        else CHECK_FFT(cufftExecC2R(plan, (cufftComplex*)din, (cufftReal*)dout));
    } else {
        if (c.kind == Kind::R2C) CHECK_FFT(cufftExecD2Z(plan, (cufftDoubleReal*)din, (cufftDoubleComplex*)dout));
        else CHECK_FFT(cufftExecZ2D(plan, (cufftDoubleComplex*)din, (cufftDoubleReal*)dout));
    }
    std::vector<unsigned char> outBuf(outBytes);
    CHECK_CU(cuMemcpyDtoH(outBuf.data(), dout, outBytes));
    CHECK_FFT(cufftDestroy(plan));
    cuMemFree(din);
    if (!c.inPlace) cuMemFree(dout);

    double worst = 0, scale = 0;
    for (int b = 0; b < c.batch; ++b)
        for (int i0 = 0; i0 < n[0]; ++i0)
            for (int i1 = 0; i1 < n[1]; ++i1)
                for (int i2 = 0; i2 < outLast; ++i2) {
                    const Complex want = expected[((b * int64_t(n[0]) + i0) * n[1] + i1) * n[2] + i2];
                    const int64_t o = b * outL.dist + i0 * outL.stride[0] + i1 * outL.stride[1] + i2 * outL.stride[2];
                    const Complex got = outComplex ? Complex(loadValue(outBuf, 2 * o, c.precision),
                                                             loadValue(outBuf, 2 * o + 1, c.precision))
                                                   : Complex(loadValue(outBuf, o, c.precision), 0);
                    worst = std::max(worst, std::abs(got - (outComplex ? want : Complex(want.real(), 0))));
                    scale = std::max(scale, std::abs(want));
                }
    const double err = worst / scale;
    const bool ok = err < c.tolerance;
    std::printf("  %-36s workspace %8zu B, relative error %.2e%s\n", c.name.c_str(), work, err, ok ? "" : "  FAILED");
    if (!ok) ++g_failures;
    (void)halfCount;
}

}  // namespace

int main() {
    CHECK_CU(cuInit(0));
    CUdevice dev;
    CHECK_CU(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CHECK_CU(cuDevicePrimaryCtxRetain(&ctx, dev));
    CHECK_CU(cuCtxSetCurrent(ctx));
    int version = 0;
    CHECK_FFT(cufftGetVersion(&version));
    std::printf("cuFFT version %d\n", version);

    using K = Kind;
    using P = Precision;
    const Case cases[] = {
        {"C2C 1", K::C2C, P::F32, {1}},
        {"C2C 2", K::C2C, P::F32, {2}},
        {"C2C 8 x 5", K::C2C, P::F32, {8}, 5},
        {"C2C 64 inverse", K::C2C, P::F32, {64}, 1, CUFFT_INVERSE},
        {"C2C 1000 (2^3 5^3)", K::C2C, P::F32, {1000}, 2},
        {"C2C 4096 x 3", K::C2C, P::F32, {4096}, 3},
        {"C2C 2 3 5 7 11 13 = 30030", K::C2C, P::F32, {30030}, 1, CUFFT_FORWARD, false, 0, 0, 3e-4},
        {"C2C 17 (Bluestein)", K::C2C, P::F32, {17}, 4},
        {"C2C 97 inverse (Bluestein)", K::C2C, P::F32, {97}, 2, CUFFT_INVERSE},
        {"C2C 2 x 1031 (Bluestein)", K::C2C, P::F32, {1031 * 2}},
        // four-step (too long for one shared-memory dispatch): along a strided axis, inverse, real,
        // Bluestein's long power of two, f64 (limit 1024)
        {"C2C 4096 x 3 axis 0 (four-step)", K::C2C, P::F32, {4096, 3}},
        {"C2C 6000 inverse (four-step)", K::C2C, P::F32, {6000}, 2, CUFFT_INVERSE},
        {"R2C 5000 (four-step)", K::R2C, P::F32, {5000}},
        {"C2C 5003 (Bluestein, four-step)", K::C2C, P::F32, {5003}, 1, CUFFT_FORWARD, false, 0, 0, 3e-4},
        {"C2C 12 x 20 in place", K::C2C, P::F32, {12, 20}, 2, CUFFT_FORWARD, true},
        {"C2C 6 x 17 x 8 inverse", K::C2C, P::F32, {6, 17, 8}, 1, CUFFT_INVERSE},
        {"C2C 40 strided", K::C2C, P::F32, {40}, 3, CUFFT_FORWARD, false, 2, 3},
        {"R2C 12", K::R2C, P::F32, {12}, 3},
        {"R2C 31", K::R2C, P::F32, {31}},
        {"R2C 256 in place", K::R2C, P::F32, {256}, 2, CUFFT_FORWARD, true},
        {"R2C 6 x 5 x 8", K::R2C, P::F32, {6, 5, 8}, 2},
        {"R2C 30 strided", K::R2C, P::F32, {30}, 2, CUFFT_FORWARD, false, 3, 2},
        {"C2R 16", K::C2R, P::F32, {16}, 2},
        {"C2R 15", K::C2R, P::F32, {15}},
        {"C2R 9 x 14", K::C2R, P::F32, {9, 14}, 2},
        {"C2R 64 in place", K::C2R, P::F32, {64}, 1, CUFFT_INVERSE, true},
        {"C2R 4 x 6 x 10", K::C2R, P::F32, {4, 6, 10}},
        {"Z2Z 360", K::C2C, P::F64, {360}, 2, CUFFT_FORWARD, false, 0, 0, 1e-12},
        {"Z2Z 101 inverse (Bluestein)", K::C2C, P::F64, {101}, 1, CUFFT_INVERSE, false, 0, 0, 1e-12},
        {"Z2Z 3072 (four-step)", K::C2C, P::F64, {3072}, 2, CUFFT_FORWARD, false, 0, 0, 1e-12},
        {"D2Z 10 x 12", K::R2C, P::F64, {10, 12}, 2, CUFFT_FORWARD, false, 0, 0, 1e-12},
        {"Z2D 7 x 9", K::C2R, P::F64, {7, 9}, 1, CUFFT_INVERSE, false, 0, 0, 1e-12},
        {"f16 C2C 64 (Xt)", K::C2C, P::F16, {64}, 2, CUFFT_FORWARD, false, 0, 0, 1e-2},
        {"f16 R2C 32 x 8 (Xt)", K::R2C, P::F16, {32, 8}, 1, CUFFT_FORWARD, false, 0, 0, 1e-2},
        {"f16 C2R 128 (Xt)", K::C2R, P::F16, {128}, 1, CUFFT_INVERSE, false, 0, 0, 1e-2},
    };
    for (const Case& c : cases) run(c);
    CHECK_CU(cuDevicePrimaryCtxRelease(dev));
    std::printf(g_failures ? "FAILED\n" : "OK\n");
    return g_failures ? 1 : 0;
}
