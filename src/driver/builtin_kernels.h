#pragma once

// Library kernels that run as Vulkanizer's own shaders instead of their translated PTX: recognized by
// their mangled names when a function is first compiled, launched from their parameters. Each is
// faster than (or, for PyTorch's attention below sm_80, unlike) the translation. VULKANIZER_BUILTIN_KERNELS=0
// turns them all off.

#include "driver/attention.h"
#include "driver/im2col.h"

namespace vulkanizer::driver {

inline BuiltinKernel matchBuiltinKernel(const std::string& name, uint32_t& param) {
    if (BuiltinKernel k = matchAttentionKernel(name, param); k != BuiltinKernel::None) return k;
    if ((param = matchIm2colKernel(name))) return BuiltinKernel::Im2col;
    return BuiltinKernel::None;
}

inline CUresult launchBuiltinKernel(ApiScope& s, CUfunction f, const unsigned grid[3],
                                    const std::vector<uint8_t>& params) {
    if (f->builtin == BuiltinKernel::Im2col) return launchIm2col(s, f, params);
    return launchAttention(s, f, grid, params);
}

}  // namespace vulkanizer::driver
