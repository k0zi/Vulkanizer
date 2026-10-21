#pragma once

#include "ptx/parser.h"

#include <cstdint>
#include <string>
#include <vector>

namespace vulkanizer::compiler {

// Properties of the Vulkan device a kernel is compiled for.
struct Target {
    uint32_t subgroupSize = 32;
    bool subgroupBallot = true;   // subgroup ballot and shuffle operations in compute shaders
    bool subgroupShuffle = true;
    uint32_t maxPushConstantsSize = 128;
    uint32_t maxSharedMemorySize = 48 * 1024;
    bool float16 = false;
    bool float64 = false;
    bool int8 = false;
    bool int16 = false;
    bool storage8 = false;   // 8-bit loads/stores in physical storage buffers
    bool storage16 = false;  // 16-bit loads/stores in physical storage buffers
    bool int64Atomics = false;
    bool integerDotProduct = false;  // OpSDot/OpUDot/OpSUDot on packed 4x8-bit vectors
    bool fusedFma = false;           // the driver's Fma rounds once; otherwise f32 fma is emulated:
    bool fmaInF64 = false;           // in f64 (else with f32 TwoProduct/TwoSum)
};

// Kernel ABI (see plan.md):
//  - the CUDA parameter buffer is the push constant block, byte for byte
//  - pointers are PhysicalStorageBuffer64 addresses (buffer device address)
//  - blockDim.x/y/z come from specialization constants 0/1/2 (the WorkgroupSize builtin)
//  - shared memory: static size in the binary; the dynamic size in words is
//    specialization constant 3
//  - .global/.const module variables: the module buffer's address goes into the push
//    constants at `globalsOffset`; gridDim (3 x u32) at `gridOffset`
//  - parameter blocks that do not fit: the push constants start with the address of a copy
//    of the parameter buffer (`paramsInBuffer`)
//  - device printf: the address of the launch's printf buffer at `printfOffset`
//  - packable kernels: specialization constant 4 = K blocks per workgroup along x; the
//    workgroup is (K * blockDim.x, blockDim.y, blockDim.z) and the grid ceil(gridDim.x / K)
struct KernelBinary {
    static constexpr uint32_t kNone = ~0u;
    static constexpr uint32_t kPrintfArgBytes = 64;
    std::vector<uint32_t> spirv;
    std::string entryPoint;
    uint32_t pushConstantSize = 0;  // multiple of 4
    uint32_t globalsOffset = kNone;
    uint32_t gridOffset = kNone;
    bool paramsInBuffer = false;
    uint32_t printfOffset = kNone;
    uint32_t staticSharedBytes = 0;
    uint32_t localBytes = 0;
    // No block-level barrier and no shared memory: several CUDA blocks may share one workgroup.
    // Specialization constant 4 is then the number of blocks packed along x (see plan.md).
    bool packable = false;
};

bool compileKernel(const ptx::Module& module, const ptx::Entry& entry, const Target& target, KernelBinary& out,
                   std::string& error);

}  // namespace vulkanizer::compiler
