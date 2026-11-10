// Source of vector_add.ptx (the committed PTX is hand-written in nvcc's shape).
// Regenerate with ./gen_ptx.sh vector_add.cu (needs clang with the NVPTX target).
#include "vulkanizer_cuda_min.h"

extern "C" __global__ void vecAdd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
