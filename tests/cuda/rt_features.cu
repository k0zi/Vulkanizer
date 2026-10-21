// CUDA Runtime API coverage beyond vectorAdd: several kernels in one translation unit,
// struct and scalar parameters, 2D launches, streams, events, memset, cudaMemcpyDefault,
// device properties and error reporting.
#include <cstdio>
#include <cstring>
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

#define EXPECT(cond)                                                                           \
    do {                                                                                       \
        if (!(cond)) {                                                                         \
            std::fprintf(stderr, "%s:%d: expectation failed: %s\n", __FILE__, __LINE__, #cond); \
            return 1;                                                                          \
        }                                                                                      \
    } while (0)

struct Affine {
    float scale;
    float offset;
    int stride;
};

__global__ void affine(float* data, Affine a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i * a.stride] = data[i * a.stride] * a.scale + a.offset;
}

__global__ void fill2d(unsigned* out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) out[y * width + x] = (unsigned(y) << 16) | unsigned(x);
}

__global__ void sumRows(const unsigned* in, unsigned long long* rowSums, int width, int height) {
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= height) return;
    unsigned long long s = 0;
    for (int x = 0; x < width; ++x) s += in[y * width + x];
    rowSums[y] = s;
}

int main() {
    int runtimeVersion = 0, driverVersion = 0, device = -1;
    CHECK(cudaRuntimeGetVersion(&runtimeVersion));
    CHECK(cudaDriverGetVersion(&driverVersion));
    CHECK(cudaSetDevice(0));
    CHECK(cudaGetDevice(&device));
    EXPECT(device == 0 && runtimeVersion >= 12000 && driverVersion >= 12000);

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    EXPECT(prop.warpSize == 32 && prop.maxThreadsPerBlock >= 256 && prop.totalGlobalMem > 0);
    EXPECT(std::strlen(prop.name) > 0 && prop.major >= 3);
    int smCount = 0;
    CHECK(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, 0));
    EXPECT(smCount == prop.multiProcessorCount);

    // struct parameter, strided access, async copies on a stream, timed with events
    const int n = 4096, stride = 2;
    float* d;
    CHECK(cudaMalloc(&d, n * stride * sizeof(float)));
    float* h;
    CHECK(cudaMallocHost(&h, n * stride * sizeof(float)));
    for (int i = 0; i < n * stride; ++i) h[i] = float(i);
    cudaStream_t s;
    CHECK(cudaStreamCreate(&s));
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start, s));
    CHECK(cudaMemcpyAsync(d, h, n * stride * sizeof(float), cudaMemcpyHostToDevice, s));
    affine<<<(n + 127) / 128, 128, 0, s>>>(d, Affine{2.0f, -1.0f, stride}, n);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpyAsync(h, d, n * stride * sizeof(float), cudaMemcpyDeviceToHost, s));
    CHECK(cudaEventRecord(stop, s));
    CHECK(cudaStreamSynchronize(s));
    float ms = -1;
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    EXPECT(ms >= 0);
    for (int i = 0; i < n * stride; ++i) EXPECT(h[i] == (i % stride ? float(i) : float(i) * 2.0f - 1.0f));

    // 2D launch, a second kernel consuming its output, memset, cudaMemcpyDefault
    const int w = 100, hgt = 37;
    unsigned* img;
    unsigned long long* sums;
    CHECK(cudaMalloc(&img, w * hgt * sizeof(unsigned)));
    CHECK(cudaMalloc(&sums, hgt * sizeof(unsigned long long)));
    CHECK(cudaMemset(sums, 0xFF, hgt * sizeof(unsigned long long)));
    fill2d<<<dim3((w + 15) / 16, (hgt + 7) / 8), dim3(16, 8)>>>(img, w, hgt);
    sumRows<<<1, 64>>>(img, sums, w, hgt);
    CHECK(cudaDeviceSynchronize());
    unsigned long long hostSums[hgt];
    CHECK(cudaMemcpy(hostSums, sums, sizeof hostSums, cudaMemcpyDefault));
    for (int y = 0; y < hgt; ++y) {
        unsigned long long want = 0;
        for (int x = 0; x < w; ++x) want += (unsigned(y) << 16) | unsigned(x);
        EXPECT(hostSums[y] == want);
    }

    // errors: invalid launch configuration is reported and cleared by cudaGetLastError
    affine<<<1, 4096>>>(d, Affine{1, 0, 1}, 1);
    const cudaError_t launchError = cudaGetLastError();
    EXPECT(launchError == cudaErrorInvalidConfiguration);
    EXPECT(cudaGetLastError() == cudaSuccess);
    EXPECT(cudaSetDevice(42) == cudaErrorInvalidDevice);
    EXPECT(cudaGetLastError() == cudaErrorInvalidDevice);

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaStreamDestroy(s));
    CHECK(cudaFreeHost(h));
    CHECK(cudaFree(d));
    CHECK(cudaFree(img));
    CHECK(cudaFree(sums));
    CHECK(cudaDeviceReset());
    std::printf("OK\n");
    return 0;
}
