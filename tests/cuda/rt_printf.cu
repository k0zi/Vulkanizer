// Device-side printf. CTest compares the output (tests/CMakeLists.txt).
#include <cstdio>
#include <cuda_runtime.h>

__global__ void hello(int n, float f, double d, const char* s) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("hello from the GPU\n");
        printf("int %d, unsigned %u, hex %#x, char %c\n", n, 4000000000u, 255, 'Z');
        printf("long %lld, float %.3f, double %g, padded [%5d]\n", -1234567890123ll, f, d, 42);
        printf("string '%s', percent 100%%\n", s);
    }
    __syncthreads();  // keeps the output order deterministic
    if (threadIdx.x == 3) printf("thread %d of %d\n", int(threadIdx.x), int(blockDim.x));
}

int main() {
    char* s;
    cudaMalloc(&s, 16);
    cudaMemcpy(s, "device", 7, cudaMemcpyHostToDevice);
    hello<<<1, 32>>>(-7, 3.14159f, 2.5e-3, s);
    cudaError_t e = cudaDeviceSynchronize();
    std::printf("done: %s\n", cudaGetErrorString(e));
    cudaFree(s);
    return e == cudaSuccess ? 0 : 1;
}
