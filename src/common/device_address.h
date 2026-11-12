#pragma once

#include <cstdint>

namespace vulkanizer::common {

// The device addresses Vulkanizer hands out (CUdeviceptr, Buffer::address) are the buffer device
// addresses without their top bits: RADV places buffers in the upper half of the 48-bit address
// space (0xffff8...), which as a signed 64-bit integer is negative and breaks programs that keep
// pointers in an int64 (PyTorch's CUDA graph trees, ctypes). Kernels set the device's bits 63..47
// again before every access (the PTX compiler, and canon() in the GLSL shaders): an OR of the high
// word with specialization constant 5, vk::Device::hostAddressBits() (0 where buffers live in the
// lower half, as llvmpipe's CPU pointers do).
// The same OR maps a host pointer h (below 2^47) to h | hostAddressBits(): host memory placed at
// that device address (Device::importHostMemory, allocateHostMemory) has one pointer for the CPU
// and the GPU, as CUDA's pinned, registered and managed memory do.
constexpr uint64_t kAddressMask = 0x0000FFFFFFFFFFFFull;

// Several devices (multi-GPU): every Vulkan device has an address space of its own, so two GPUs
// hand out the same buffer addresses. For unified addressing (cudaMemcpyDefault,
// cudaPointerGetAttributes) device n's addresses carry n in bits 55..48 (vk::Device::addressTag()),
// where its bits 63..48 are all set anyway: the kernels' OR sets those bits again, so the tag never
// reaches the GPU. Device 0 has no tag; devices whose buffers live in the lower half (llvmpipe: CPU
// pointers, unique in the process already) neither. Bits 63..56 stay free for the PTX compiler's
// generic shared/local tags; host memory used in place keeps its host address on every device.
constexpr unsigned kAddressTagShift = 48;
constexpr uint64_t kAddressTagMask = 0x00FF000000000000ull;

}  // namespace vulkanizer::common
