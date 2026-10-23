// Device-side assert(): the failing launch's synchronization reports cudaErrorAssert with CUDA's
// message, the error is sticky, and cudaDeviceReset clears it. CTest compares the output
// (tests/CMakeLists.txt).
#include <cassert>
#include <cstdio>
#include <cuda_runtime.h>

__global__ void check(const int* data, int n, int* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (data[i] < 0) return;  // only the thread with a marked element asserts
    assert(i < n);
    out[0] = i;
}

__global__ void add(int* out, int v) { out[0] += v; }

int main() {
    int* data;
    int* out;
    cudaMalloc(&data, 64 * sizeof(int));
    cudaMalloc(&out, sizeof(int));
    int host[64];
    for (int& h : host) h = -1;
    cudaMemcpy(data, host, sizeof host, cudaMemcpyHostToDevice);

    check<<<4, 16>>>(data, 64, out);  // nothing asserts
    std::printf("passing: %s\n", cudaGetErrorString(cudaDeviceSynchronize()));

    host[2 * 16 + 5] = 1;  // block 2, thread 5 checks i = 37 < 30
    cudaMemcpy(data, host, sizeof host, cudaMemcpyHostToDevice);
    check<<<4, 16>>>(data, 30, out);
    const cudaError_t failed = cudaDeviceSynchronize();
    std::printf("failing: %s\n", cudaGetErrorString(failed));
    int v = 0;
    const cudaError_t sticky = cudaMemcpy(&v, out, sizeof v, cudaMemcpyDeviceToHost);
    std::printf("sticky: %s\n", cudaGetErrorString(sticky));

    cudaDeviceReset();
    cudaMalloc(&out, sizeof(int));
    cudaMemset(out, 0, sizeof(int));
    add<<<1, 1>>>(out, 3);
    const cudaError_t after = cudaMemcpy(&v, out, sizeof v, cudaMemcpyDeviceToHost);
    std::printf("after reset: %s, %d\n", cudaGetErrorString(after), v);
    return failed == cudaErrorAssert && sticky == cudaErrorAssert && after == cudaSuccess && v == 3 ? 0 : 1;
}
