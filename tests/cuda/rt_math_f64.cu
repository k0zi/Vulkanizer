// Double precision math from CUDA's device library against the host's libm. The f64
// trigonometric functions reduce large arguments with multiword integer arithmetic
// (mad.lo.cc / madc.hi.cc), which is what PyTorch's cos/sin kernels on f64 tensors run.
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

#define CHECK(call)                                                                            \
    do {                                                                                       \
        cudaError_t e = (call);                                                                \
        if (e != cudaSuccess) {                                                                \
            std::fprintf(stderr, "%s:%d: %s: %s (%s)\n", __FILE__, __LINE__, #call,             \
                         cudaGetErrorString(e), cudaGetErrorName(e));                           \
            return 1;                                                                          \
        }                                                                                      \
    } while (0)

constexpr int kFuncs = 7;

__global__ void mathF64(const double* x, double* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double v = x[i];
    out[i * kFuncs + 0] = sin(v);
    out[i * kFuncs + 1] = cos(v);
    out[i * kFuncs + 2] = tan(v);
    out[i * kFuncs + 3] = exp(v / 1000.0);
    out[i * kFuncs + 4] = log(fabs(v) + 1.0);
    out[i * kFuncs + 5] = pow(fabs(v) + 0.5, 0.3);
    out[i * kFuncs + 6] = sqrt(fabs(v));
}

static double reference(int f, double v) {
    switch (f) {
        case 0: return std::sin(v);
        case 1: return std::cos(v);
        case 2: return std::tan(v);
        case 3: return std::exp(v / 1000.0);
        case 4: return std::log(std::fabs(v) + 1.0);
        case 5: return std::pow(std::fabs(v) + 0.5, 0.3);
        default: return std::sqrt(std::fabs(v));
    }
}

int main() {
    // Small, medium and huge arguments: the huge ones take the Payne-Hanek reduction path.
    const double inputs[] = {0.0, 0.5, -1.25, 3.14159, 10.0, -100.5, 1234.5678, 1.0e5, -3.0e6,
                             1.0e9, 1.0e15, -2.5e22, 1.0e300, 7.0e-10, 0.785398163397448};
    const int n = sizeof(inputs) / sizeof(inputs[0]);
    double *dx, *dout;
    CHECK(cudaMalloc(&dx, n * sizeof(double)));
    CHECK(cudaMalloc(&dout, n * kFuncs * sizeof(double)));
    CHECK(cudaMemcpy(dx, inputs, n * sizeof(double), cudaMemcpyHostToDevice));
    mathF64<<<1, 32>>>(dx, dout, n);
    CHECK(cudaGetLastError());
    double out[n * kFuncs];
    CHECK(cudaMemcpy(out, dout, sizeof(out), cudaMemcpyDeviceToHost));

    int failures = 0;
    for (int i = 0; i < n; ++i) {
        for (int f = 0; f < kFuncs; ++f) {
            const double got = out[i * kFuncs + f], want = reference(f, inputs[i]);
            // CUDA's f64 functions are within 2 ulp; tan near its poles is ill-conditioned.
            const double tol = 4 * std::fabs(std::nextafter(want, INFINITY) - want) + 1e-300;
            if (got != want && !(std::fabs(got - want) <= tol)) {
                std::fprintf(stderr, "f%d(%.17g) = %.17g, expected %.17g\n", f, inputs[i], got, want);
                ++failures;
            }
        }
    }
    CHECK(cudaFree(dx));
    CHECK(cudaFree(dout));
    if (failures) return 1;
    std::printf("f64 math ok\n");
    return 0;
}
