#pragma once

// Fused attention in place of PyTorch's memory-efficient attention forward kernels (plan.md 6.3).
// The fmha_cutlassF_* kernels of PyTorch's fatbins are built around tensor core instructions, and
// below sm_80 their PTX is only a "kernel is for sm80+" stub. They are recognized by name and run as
// Vulkanizer's own shaders (shaders/attention_*.comp) from their AttentionKernel::Params, never
// translated: cooperative matrices for f16 / bf16 with head dimensions up to 128, a scalar kernel for
// the rest. VULKANIZER_FUSED_ATTENTION=0 turns this off (the PTX is translated as any other), as does
// VULKANIZER_BUILTIN_KERNELS=0.

#include "driver/driver.h"

#include <string>
#include <vector>

namespace vulkanizer::driver {

// PyTorch's mangled kernel name -> element type and queries per CUDA block (BuiltinKernel::None if it
// is not one of the kernels).
BuiltinKernel matchAttentionKernel(const std::string& name, uint32_t& queriesPerBlock);

// Runs the launch of an attention kernel (f->builtin one of them): `params` is its AttentionKernel::Params.
CUresult launchAttention(ApiScope& s, CUfunction f, const unsigned grid[3], const std::vector<uint8_t>& params);

}  // namespace vulkanizer::driver
