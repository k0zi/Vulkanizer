#pragma once

#include "compiler/ptx_to_spirv.h"

#include <string>

// On-disk cache of translated kernels, keyed by everything the translation depends on: this
// libcuda build, the target device's properties, the module's PTX and the kernel name.
// Location: $VULKANIZER_CACHE_DIR, else $XDG_CACHE_HOME/vulkanizer, else ~/.cache/vulkanizer;
// VULKANIZER_CACHE_DIR=off disables it.
namespace vulkanizer::driver {

std::string kernelCacheKey(const std::string& moduleHash, const std::string& entry, const compiler::Target& target);
bool loadCachedKernel(const std::string& key, compiler::KernelBinary& out);
void storeCachedKernel(const std::string& key, const compiler::KernelBinary& binary);

}  // namespace vulkanizer::driver
