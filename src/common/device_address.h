#pragma once

#include <cstdint>

namespace vulkanizer::common {

// The device addresses Vulkanizer hands out (CUdeviceptr, Buffer::address) are the buffer device
// addresses without their top bits: RADV places buffers in the upper half of the 48-bit address
// space (0xffff8...), which as a signed 64-bit integer is negative and breaks programs that keep
// pointers in an int64 (PyTorch's CUDA graph trees, ctypes). Kernels set bits 63..47 again before
// every access (the PTX compiler, and canon() in the GLSL shaders): an OR with kHostAddressBits.
constexpr uint64_t kAddressMask = 0x0000FFFFFFFFFFFFull;
// The same OR maps a host pointer h (below 2^47) to h | kHostAddressBits: host memory imported
// at that device address (Device::importHostMemory) has one pointer for the CPU and the GPU,
// as CUDA's pinned, registered and managed memory do.
constexpr uint64_t kHostAddressBits = 0xFFFF800000000000ull;

}  // namespace vulkanizer::common
