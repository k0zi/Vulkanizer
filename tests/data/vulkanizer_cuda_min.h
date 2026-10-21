// Minimal CUDA device-side declarations for compiling the test kernels with clang
// without a CUDA SDK (clang -nocudainc -nocudalib). Not used by nvcc.
#pragma once

#ifndef __NVCC__
#include <__clang_cuda_builtin_vars.h>  // threadIdx, blockIdx, blockDim, gridDim (ships with clang)

#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))
#endif
