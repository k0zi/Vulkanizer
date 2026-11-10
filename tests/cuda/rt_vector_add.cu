// CUDA Runtime API program compiled with a real nvcc (tests/cuda/build.sh).
// The first milestone of phase 2: it must run unmodified on Vulkanizer.
#include <cstdio>
#include <cuda_runtime.h>

__global__ void vecAdd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

#define CHECK(call)                                                                   \
    do {                                                                              \
        cudaError_t e = (call);                                                       \
        if (e != cudaSuccess) {                                                       \
            std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #call,         \
                         cudaGetErrorString(e));                                      \
            return 1;                                                                 \
        }                                                                             \
    } while (0)

int main() {
    int count = 0;
    CHECK(cudaGetDeviceCount(&count));
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("%d device(s), device 0: %s (cc %d.%d)\n", count, prop.name, prop.major, prop.minor);

    const int n = 1 << 20;
    float *a, *b, *c;
    CHECK(cudaMalloc(&a, n * sizeof(float)));
    CHECK(cudaMalloc(&b, n * sizeof(float)));
    CHECK(cudaMalloc(&c, n * sizeof(float)));
    float* h = new float[n];
    for (int i = 0; i < n; ++i) h[i] = float(i);
    CHECK(cudaMemcpy(a, h, n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(b, h, n * sizeof(float), cudaMemcpyHostToDevice));

    vecAdd<<<(n + 255) / 256, 256>>>(a, b, c, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h, c, n * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; ++i) {
        if (h[i] != 2.0f * float(i)) {
            std::fprintf(stderr, "mismatch at %d: %f\n", i, h[i]);
            return 1;
        }
    }
    CHECK(cudaFree(a));
    CHECK(cudaFree(b));
    CHECK(cudaFree(c));
    delete[] h;
    std::printf("OK\n");
    return 0;
}
