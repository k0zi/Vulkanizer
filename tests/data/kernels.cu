// Source of kernels.ptx. The committed PTX is hand-written in nvcc's shape (no CUDA SDK
// on the development machine), with the control flow spelled out explicitly where the
// test depends on it. tests/kernels.cpp holds the matching CPU references.
#include "vulkanizer_cuda_min.h"

extern "C" __global__ void saxpy(float a, const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

extern "C" __global__ void scaleGridStride(float* data, int n, float s) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) data[i] *= s;
}

extern "C" __global__ void index2d(int* out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) out[y * width + x] = (y << 16) | x;
}

extern "C" __global__ void intOps(long long* out, const int* in, int n, short bias) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int v = in[i];
    long long* o = out + i * 8;
    o[0] = (long long)v * 3000000000LL;
    o[1] = v / 7;
    o[2] = v % 7;
    o[3] = min(v, 100);
    o[4] = v >> 3;
    o[5] = v < 0 ? 1 : -1;
    o[6] = __mulhi(v, 0x55555556);
    o[7] = (unsigned)((v << 4) | ((unsigned)v >> 28)) + bias;  // plus "v << 40" == 0 in PTX
}

extern "C" __global__ void collatz(int* steps, const int* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int x = in[i], count = 0;
    while (x != 1 && count < 1000) {
        x = (x & 1) ? 3 * x + 1 : x >> 1;  // predicated instructions in the PTX
        ++count;
    }
    steps[i] = count;
}

// Nested loops with break, continue and a two-level break (see kernels.cpp: nestedRef).
extern "C" __global__ void nested(int* out, int n);

extern "C" __global__ void floatOps(float* out, const float* in, int n);  // see kernels.cpp: floatOpsRef

extern "C" __global__ void vec4(float4* out, unsigned long long* packed, uint2* swapped, const float4* in, int n);
