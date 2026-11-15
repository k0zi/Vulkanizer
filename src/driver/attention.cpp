// Fused attention in place of PyTorch's memory-efficient attention forward kernels (see attention.h).

#include "driver/attention.h"

#include "common/log.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <regex>

using namespace vulkanizer;
using namespace vulkanizer::driver;

namespace {

const uint32_t kCoopF16Spirv[] = {
#include "attention_coopmat.comp.inc"
};
const uint32_t kCoopBF16Spirv[] = {
#include "attention_coopmat_b.comp.inc"
};
const uint32_t kScalarF32Spirv[] = {
#include "attention_scalar_f32.comp.inc"
};
const uint32_t kScalarF16Spirv[] = {
#include "attention_scalar_f16.comp.inc"
};
const uint32_t kScalarBF16Spirv[] = {
#include "attention_scalar_bf16.comp.inc"
};

// The arguments of shaders/attention_*.comp (Args in attention_common.glsl).
struct Args {
    uint64_t q, k, v, bias, o, lse;
    int64_t qStrideB, kStrideB, vStrideB, biasStrideB, biasStrideH;
    uint64_t seqstartQ, seqstartK, seqlenK;
    int32_t qStrideM, kStrideM, vStrideM, biasStrideM, oStrideM, qStrideH, kStrideH, vStrideH;
    int32_t headDim, headDimV, numQueries, numKeys, numHeads, qHeadsPerKv, maskType, causalOffset;
    int32_t windowSize;
    float scale;
    int32_t queriesPerBlock, gridX;
    int32_t slices;
    uint32_t flags;
};
static_assert(sizeof(Args) == 200);  // std430 in the shaders: no padding
constexpr uint32_t kVectorLoads = 1;

// PyTorch's AttentionKernel<...>::Params (aten/src/ATen/native/transformers/cuda/mem_eff_attention/
// kernel_forward.h), the same from 2.9 to 2.14 but for one field: 2.14 added int32 q_heads_per_kv
// (grouped-query attention) at 200, where 2.13 has bool use_dropout. dropout_prob is at 216 in both.
constexpr size_t kParamsBytes = 272;
template <typename T>
T field(const std::vector<uint8_t>& p, size_t at) {
    T v;
    std::memcpy(&v, p.data() + at, sizeof v);
    return v;
}

bool decodeParams(const std::vector<uint8_t>& p, Args& a, std::string& why) {
    if (p.size() < kParamsBytes) {
        why = "unexpected parameter size " + std::to_string(p.size());
        return false;
    }
    if (field<float>(p, 216) != 0.0f) {
        why = "dropout";
        return false;
    }
    a = {};
    a.q = field<uint64_t>(p, 0);
    a.k = field<uint64_t>(p, 8);
    a.v = field<uint64_t>(p, 16);
    a.bias = field<uint64_t>(p, 24);
    a.seqstartQ = field<uint64_t>(p, 32);
    a.seqstartK = field<uint64_t>(p, 40);
    a.seqlenK = field<uint64_t>(p, 48);
    a.causalOffset = field<int32_t>(p, 56);
    a.o = field<uint64_t>(p, 64);
    a.lse = field<uint64_t>(p, 80);
    a.windowSize = field<int32_t>(p, 88);
    a.scale = field<float>(p, 92);
    a.headDim = field<int32_t>(p, 96);
    a.headDimV = field<int32_t>(p, 100);
    a.numQueries = field<int32_t>(p, 104);
    a.numKeys = field<int32_t>(p, 108);
    a.maskType = field<uint8_t>(p, 116);
    a.qStrideM = field<int32_t>(p, 120);
    a.kStrideM = field<int32_t>(p, 124);
    a.vStrideM = field<int32_t>(p, 128);
    a.biasStrideM = field<int32_t>(p, 132);
    a.oStrideM = field<int32_t>(p, 136);
    a.qStrideH = field<int32_t>(p, 140);
    a.kStrideH = field<int32_t>(p, 144);
    a.vStrideH = field<int32_t>(p, 148);
    a.biasStrideH = field<int64_t>(p, 152);
    a.qStrideB = field<int64_t>(p, 160);
    a.kStrideB = field<int64_t>(p, 168);
    a.vStrideB = field<int64_t>(p, 176);
    a.biasStrideB = field<int64_t>(p, 184);
    a.numHeads = field<int32_t>(p, 196);
    // Without dropout, 2.13's use_dropout byte is 0; 2.14's q_heads_per_kv is at least 1.
    a.qHeadsPerKv = field<uint8_t>(p, 200) ? field<int32_t>(p, 200) : 1;
    if (a.qHeadsPerKv < 1 || a.maskType > 2 || a.headDim <= 0 || a.headDimV <= 0 || a.numHeads <= 0) {
        why = "unexpected parameters";
        return false;
    }
    return true;
}

bool envOff(const char* name) {
    const char* v = std::getenv(name);
    return v && std::strcmp(v, "0") == 0;
}

}  // namespace

namespace vulkanizer::driver {

BuiltinKernel matchAttentionKernel(const std::string& name, uint32_t& queriesPerBlock) {
    static const bool off = envOff("VULKANIZER_FUSED_ATTENTION") || envOff("VULKANIZER_BUILTIN_KERNELS");
    if (off || name.find("PyTorchMemEffAttention15AttentionKernel") == std::string::npos) return BuiltinKernel::None;
    static const std::regex pattern(R"(^_Z\d+fmha_cutlassF_(f16|bf16|f32)_(not)?aligned_(\d+)x\d+_(rf|gmem)_sm\d+N)");
    std::smatch m;
    if (!std::regex_search(name, m, pattern)) return BuiltinKernel::None;
    queriesPerBlock = uint32_t(std::stoul(m[3].str()));
    if (!queriesPerBlock) return BuiltinKernel::None;
    return m[1] == "f16" ? BuiltinKernel::AttentionF16 : m[1] == "bf16" ? BuiltinKernel::AttentionBF16
                                                                        : BuiltinKernel::AttentionF32;
}

CUresult launchAttention(ApiScope& s, CUfunction f, const unsigned grid[3], const std::vector<uint8_t>& params) {
    Args a;
    std::string why;
    if (!decodeParams(params, a, why)) {
        VULKANIZER_ERROR("'%s': %s is not supported by the fused attention", f->entry->name.c_str(), why.c_str());
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    a.queriesPerBlock = int32_t(f->builtinParam);
    a.gridX = int32_t(grid[0]);
    const BuiltinKernel type = f->builtin;
    const vk::Features& feat = s.dev->features();
    const uint32_t subgroup = s.slot->info->subgroupSize;
    static const bool scalarOnly = [] {
        const char* v = std::getenv("VULKANIZER_FUSED_ATTENTION");
        return v && std::strcmp(v, "scalar") == 0;
    }();
    const bool coop = !scalarOnly && a.headDim <= 128 && a.headDimV <= 128 && (subgroup == 32 || subgroup == 64) &&
                      feat.coopMatrixSize == 16 &&
                      (type == BuiltinKernel::AttentionF16 || (type == BuiltinKernel::AttentionBF16 && feat.coopMatrixBf16));
    if (!coop && type != BuiltinKernel::AttentionF32 && !feat.storageBuffer16Bit) {
        VULKANIZER_ERROR("'%s': the fused attention needs 16-bit storage buffers", f->entry->name.c_str());
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    const uint64_t elem = type == BuiltinKernel::AttentionF32 ? 4 : 2;
    const bool aligned = ((a.q | a.k | a.v) % 16 == 0) && a.headDim % 8 == 0 && a.headDimV % 8 == 0 &&
                         a.qStrideM % 8 == 0 && a.kStrideM % 8 == 0 && a.vStrideM % 8 == 0 && a.qStrideH % 8 == 0 &&
                         a.kStrideH % 8 == 0 && a.vStrideH % 8 == 0 && a.qStrideB % 8 == 0 && a.kStrideB % 8 == 0 &&
                         a.vStrideB % 8 == 0;
    a.flags = elem == 2 && aligned ? kVectorLoads : 0;

    // attention_coopmat.comp: 256 invocations, 16 * tiles query rows per subgroup
    const uint32_t subgroups = 256 / subgroup;
    static const uint32_t tilesEnv = [] {
        const char* v = std::getenv("VULKANIZER_ATTENTION_TILES");
        return v ? uint32_t(std::atoi(v)) : 0u;
    }();
    // Two tiles (twice the queries per K / V block): 9060 XT 30 x 4128^2 d128 8.2 -> 6.6 ms, 60 heads d64
    // 9.4 -> 7.5; 890M d128 67 -> 42, d64 58 -> 50.
    const uint32_t tiles = tilesEnv ? tilesEnv : 2;
    const uint32_t rowsPerGroup = coop ? 16 * subgroups * tiles : 32;
    a.slices = coop ? 1 : (a.headDimV + 63) / 64;
    const uint64_t queries = uint64_t(grid[0]) * f->builtinParam;
    const uint32_t groupsX = uint32_t((queries + rowsPerGroup - 1) / rowsPerGroup);
    const uint32_t groupsY = grid[1] * uint32_t(a.slices);

    const uint32_t* code;
    size_t words;
    if (coop) {
        code = type == BuiltinKernel::AttentionF16 ? kCoopF16Spirv : kCoopBF16Spirv;
        words = type == BuiltinKernel::AttentionF16 ? std::size(kCoopF16Spirv) : std::size(kCoopBF16Spirv);
    } else if (type == BuiltinKernel::AttentionF32) {
        code = kScalarF32Spirv, words = std::size(kScalarF32Spirv);
    } else if (type == BuiltinKernel::AttentionF16) {
        code = kScalarF16Spirv, words = std::size(kScalarF16Spirv);
    } else {
        code = kScalarBF16Spirv, words = std::size(kScalarBF16Spirv);
    }
    // Specialization: workgroup size, head_dim and head_dim_value in 16-column tiles, subgroups per
    // workgroup (0: the scalar kernel, which keeps the two apart in the function's pipeline table).
    const std::array<uint32_t, 5> spec{coop ? 256u : 128u, uint32_t(coop ? (a.headDim + 15) / 16 : 0),
                                       uint32_t(coop ? (a.headDimV + 15) / 16 : 0), coop ? subgroups : 0, coop ? tiles : 1};
    auto it = f->pipelines.find(spec);
    if (it == f->pipelines.end()) {
        vk::ComputePipeline pipeline;
        if (VkResult vr = s.dev->createComputePipeline(code, words, "main", 8, spec.data(), pipeline)) {
            VULKANIZER_ERROR("'%s': fused attention pipeline creation failed: %d", f->entry->name.c_str(), vr);
            return toCuResult(vr);
        }
        VULKANIZER_DEBUG("'%s': fused attention (%s, head_dim %d / %d)", f->entry->name.c_str(),
                         coop ? "cooperative matrices" : "scalar", a.headDim, a.headDimV);
        it = f->pipelines.emplace(spec, pipeline).first;
    }
    const uint64_t push = 0;  // the arguments' address, filled in by dispatch
    return toCuResult(s.dev->dispatch(it->second, groupsX, groupsY, grid[2], &push, sizeof push,
                                      coop ? "fused_attention" : "fused_attention_scalar", &a, sizeof a));
}

}  // namespace vulkanizer::driver
