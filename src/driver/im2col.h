#pragma once

// PyTorch's im2col_kernel (the columns of a convolution computed as a GEMM, when cuDNN is off) as
// Vulkanizer's own shader (shaders/im2col.comp): the translated kernel loads and stores one element at a
// time, one after the other. Recognized by name for elements of 2, 4 and 8 bytes; VULKANIZER_BUILTIN_KERNELS=0
// turns it off.

#include "driver/driver.h"

#include <string>
#include <vector>

namespace vulkanizer::driver {

// PyTorch's mangled kernel name -> the element size (0 if it is not one of the kernels).
uint32_t matchIm2colKernel(const std::string& name);

// Runs the launch of an im2col kernel (f->builtin == BuiltinKernel::Im2col): `params` are its arguments.
CUresult launchIm2col(ApiScope& s, CUfunction f, const std::vector<uint8_t>& params);

}  // namespace vulkanizer::driver
