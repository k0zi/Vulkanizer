// Indirect calls (virtual functions, function pointers in device memory and from the host) and
// device-side malloc/free.
#include <cstdio>
#include <cuda_runtime.h>

struct Shape {
    __device__ virtual float area(float x) const = 0;
};
struct Square : Shape {
    __device__ float area(float x) const override { return x * x; }
};
struct Circle : Shape {
    __device__ float area(float x) const override { return 3.0f * x * x; }
};

__device__ float twice(float x) { return 2 * x; }
__device__ float half(float x) { return x / 2; }
typedef float (*Op)(float);
__device__ Op g_ops[2] = {twice, half};
__device__ Op g_half = half;

__global__ void virt(float* out, int kind) {
    Square s;
    Circle c;
    const Shape* p = (threadIdx.x + kind) & 1 ? static_cast<const Shape*>(&c) : static_cast<const Shape*>(&s);
    out[threadIdx.x] = p->area(float(threadIdx.x));
}

__global__ void table(float* out) { out[threadIdx.x] = g_ops[threadIdx.x & 1](float(threadIdx.x)); }

__global__ void given(float* out, Op op) { out[threadIdx.x] = op(float(threadIdx.x)); }

// Every thread allocates, fills, sums and frees `n` floats, `rounds` times (blocks are reused).
__global__ void heap(float* out, int n, int rounds) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    float total = 0;
    for (int r = 0; r < rounds; ++r) {
        float* p = static_cast<float*>(malloc(n * sizeof(float)));
        if (!p || (reinterpret_cast<size_t>(p) & 15)) {
            out[t] = -1;
            return;
        }
        for (int i = 0; i < n; ++i) p[i] = float(i + t + r);
        for (int i = 0; i < n; ++i) total += p[i];
        free(p);
    }
    out[t] = total;
}

__global__ void tooLarge(int* out) { out[0] = malloc(size_t(64) << 20) == nullptr; }

int check(const char* what, const float* got, const float* want, int n) {
    for (int i = 0; i < n; ++i)
        if (got[i] != want[i]) {
            std::printf("%s: element %d is %g, want %g\n", what, i, got[i], want[i]);
            return 1;
        }
    return 0;
}

int main() {
    const int threads = 64, blocks = 8, n = 37, rounds = 4;
    if (cudaDeviceSetLimit(cudaLimitMallocHeapSize, size_t(16) << 20) != cudaSuccess) return 1;
    size_t limit = 0;
    cudaDeviceGetLimit(&limit, cudaLimitMallocHeapSize);
    float* d;
    cudaMalloc(&d, blocks * threads * sizeof(float));
    float got[blocks * threads], want[blocks * threads];
    int failures = limit == (size_t(16) << 20) ? 0 : 1;

    for (int kind = 0; kind < 2; ++kind) {
        virt<<<1, threads>>>(d, kind);
        cudaMemcpy(got, d, threads * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < threads; ++i) want[i] = ((i + kind) & 1 ? 3.0f : 1.0f) * float(i) * float(i);
        failures += check("virtual", got, want, threads);
    }

    table<<<1, threads>>>(d);
    cudaMemcpy(got, d, threads * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < threads; ++i) want[i] = i & 1 ? float(i) / 2 : 2.0f * float(i);
    failures += check("table", got, want, threads);

    Op op;
    cudaMemcpyFromSymbol(&op, g_half, sizeof op);
    given<<<1, threads>>>(d, op);
    cudaMemcpy(got, d, threads * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < threads; ++i) want[i] = float(i) / 2;
    failures += check("pointer from the host", got, want, threads);

    heap<<<blocks, threads>>>(d, n, rounds);
    cudaMemcpy(got, d, blocks * threads * sizeof(float), cudaMemcpyDeviceToHost);
    for (int t = 0; t < blocks * threads; ++t) {
        float total = 0;
        for (int r = 0; r < rounds; ++r)
            for (int i = 0; i < n; ++i) total += float(i + t + r);
        want[t] = total;
    }
    failures += check("malloc", got, want, blocks * threads);

    int* flag;
    cudaMalloc(&flag, sizeof(int));
    tooLarge<<<1, 1>>>(flag);
    int isNull = 0;
    cudaMemcpy(&isNull, flag, sizeof isNull, cudaMemcpyDeviceToHost);
    if (!isNull) {
        std::printf("malloc beyond the heap did not return NULL\n");
        ++failures;
    }

    const cudaError_t e = cudaDeviceSynchronize();
    std::printf("%s (%s)\n", failures ? "FAILED" : "OK", cudaGetErrorString(e));
    return failures || e != cudaSuccess;
}
