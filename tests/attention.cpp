// The fused attention in place of PyTorch's memory-efficient attention kernels (driver/attention.cpp):
// a stub kernel with the name of PyTorch's fmha_cutlassF_* kernels, launched the way PyTorch launches it
// (AttentionKernel::Params by value, PyTorch 2.14 layout), against a reference on the CPU. f16 and bf16
// (cooperative matrices where the device has them), f32 (scalar), causal masks, a sliding window, bias,
// grouped-query heads, logsumexp, variable sequence lengths, odd head dimensions and strides.

#include "test_util.h"

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

namespace {

enum class Type { F16, BF16, F32 };

uint16_t toHalf(float x) {
    _Float16 h = static_cast<_Float16>(x);
    uint16_t b;
    std::memcpy(&b, &h, 2);
    return b;
}
float fromHalf(uint16_t b) {
    _Float16 h;
    std::memcpy(&h, &b, 2);
    return float(h);
}
uint16_t toBf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return uint16_t(u >> 16);
}
float fromBf16(uint16_t b) {
    uint32_t u = uint32_t(b) << 16;
    float x;
    std::memcpy(&x, &u, 4);
    return x;
}

struct Tensor {
    Type type;
    std::vector<float> values;  // as stored (rounded to the type)
    CUdeviceptr ptr = 0;

    Tensor(Type t, size_t n, std::mt19937& rng, float range) : type(t), values(n) {
        std::uniform_real_distribution<float> dist(-range, range);
        for (float& v : values) v = round(dist(rng));
        upload();
    }
    Tensor(Type t, size_t n) : type(t), values(n, 0.0f) { upload(); }
    size_t elem() const { return type == Type::F32 ? 4 : 2; }
    float round(float x) const {
        return type == Type::F16 ? fromHalf(toHalf(x)) : type == Type::BF16 ? fromBf16(toBf16(x)) : x;
    }
    void upload() {
        CHECK_CU(cuMemAlloc(&ptr, values.size() * elem() + 16));
        std::vector<uint8_t> raw(values.size() * elem());
        for (size_t i = 0; i < values.size(); ++i) {
            if (type == Type::F32) std::memcpy(raw.data() + 4 * i, &values[i], 4);
            else {
                const uint16_t b = type == Type::F16 ? toHalf(values[i]) : toBf16(values[i]);
                std::memcpy(raw.data() + 2 * i, &b, 2);
            }
        }
        CHECK_CU(cuMemcpyHtoD(ptr, raw.data(), raw.size()));
    }
    std::vector<float> download() const {
        std::vector<uint8_t> raw(values.size() * elem());
        CHECK_CU(cuMemcpyDtoH(raw.data(), ptr, raw.size()));
        std::vector<float> out(values.size());
        for (size_t i = 0; i < out.size(); ++i) {
            if (type == Type::F32) std::memcpy(&out[i], raw.data() + 4 * i, 4);
            else {
                uint16_t b;
                std::memcpy(&b, raw.data() + 2 * i, 2);
                out[i] = type == Type::F16 ? fromHalf(b) : fromBf16(b);
            }
        }
        return out;
    }
    ~Tensor() { cuMemFree(ptr); }
};

struct Case {
    const char* name;
    Type type;
    int queriesPerBlock;
    int batches, heads, kvHeads, numQ, numK, headDim, headDimV;
    int mask = 0;  // 0 none, 1 causal from the top left, 2 from the bottom right
    int window = 0;
    bool bias = false, lse = false, varlen = false;
    int pad = 0;   // extra elements per row (strides that are not multiples of 8)
};

const char* kernelName(Type t, int queriesPerBlock) {
    // The kernels PyTorch picks by element type and head dimension (sm_75 ones: stubs in the PTX)
    if (t == Type::F16)
        return queriesPerBlock == 32
            ? "_Z40fmha_cutlassF_f16_aligned_32x128_rf_sm75N22PyTorchMemEffAttention15AttentionKernelIN7cutlass6half_tENS1_4arch4Sm75ELb1ELi32ELi128ELi128ELb1ELb1EE6ParamsE"
            : "_Z39fmha_cutlassF_f16_aligned_64x64_rf_sm75N22PyTorchMemEffAttention15AttentionKernelIN7cutlass6half_tENS1_4arch4Sm75ELb1ELi64ELi64ELi64ELb1ELb1EE6ParamsE";
    if (t == Type::BF16)
        return "_Z41fmha_cutlassF_bf16_aligned_64x128_rf_sm80N22PyTorchMemEffAttention15AttentionKernelIN7cutlass10bfloat16_tENS1_4arch4Sm80ELb1ELi64ELi128ELi128ELb1ELb1EE6ParamsE";
    return "_Z45fmha_cutlassF_f32_notaligned_32x128_gmem_sm75N22PyTorchMemEffAttention15AttentionKernelIfN7cutlass4arch4Sm75ELb0ELi32ELi128ELi65536ELb1ELb1EE6ParamsE";
}

template <typename T>
void put(uint8_t* p, size_t at, T v) {
    std::memcpy(p + at, &v, sizeof v);
}

bool run(const Case& c, CUmodule mod, std::mt19937& rng) {
    CUfunction fn;
    CHECK_CU(cuModuleGetFunction(&fn, mod, kernelName(c.type, c.queriesPerBlock)));
    // Layout as PyTorch's: [batch, seq, heads, dim] (variable lengths: one batch of all sequences)
    std::vector<int> seqQ{0}, seqK{0};
    int maxQ = c.numQ, maxK = c.numK;
    if (c.varlen) {
        maxQ = maxK = 0;
        for (int b = 0; b < c.batches; ++b) {
            const int nq = c.numQ - 23 * b, nk = c.numK - 31 * b;
            seqQ.push_back(seqQ.back() + nq);
            seqK.push_back(seqK.back() + nk);
            maxQ = std::max(maxQ, nq);
            maxK = std::max(maxK, nk);
        }
    } else {
        for (int b = 0; b < c.batches; ++b) {
            seqQ.push_back(seqQ.back() + c.numQ);
            seqK.push_back(seqK.back() + c.numK);
        }
    }
    const int totalQ = seqQ.back(), totalK = seqK.back();
    const int qRow = c.heads * c.headDim + c.pad, kRow = c.kvHeads * c.headDim + c.pad;
    const int vRow = c.kvHeads * c.headDimV + c.pad, oRow = c.heads * c.headDimV;
    Tensor q(c.type, size_t(totalQ) * qRow, rng, 1.0f), k(c.type, size_t(totalK) * kRow, rng, 1.0f);
    Tensor v(c.type, size_t(totalK) * vRow, rng, 1.0f), o(c.type, size_t(totalQ) * oRow);
    const size_t biasCount = c.bias ? size_t(c.batches) * c.heads * maxQ * maxK : 1;
    Tensor bias(c.type, biasCount, rng, 2.0f);
    const int lseDim = (maxQ + 31) / 32 * 32;
    Tensor lse(Type::F32, size_t(c.batches) * c.heads * lseDim);
    CUdeviceptr dSeqQ = 0, dSeqK = 0;
    if (c.varlen) {
        CHECK_CU(cuMemAlloc(&dSeqQ, seqQ.size() * 4));
        CHECK_CU(cuMemAlloc(&dSeqK, seqK.size() * 4));
        CHECK_CU(cuMemcpyHtoD(dSeqQ, seqQ.data(), seqQ.size() * 4));
        CHECK_CU(cuMemcpyHtoD(dSeqK, seqK.data(), seqK.size() * 4));
    }
    const float scale = 1.0f / std::sqrt(float(c.headDim));

    uint8_t params[272] = {};
    put<uint64_t>(params, 0, q.ptr);
    put<uint64_t>(params, 8, k.ptr);
    put<uint64_t>(params, 16, v.ptr);
    put<uint64_t>(params, 24, c.bias ? bias.ptr : 0);
    put<uint64_t>(params, 32, dSeqQ);
    put<uint64_t>(params, 40, dSeqK);
    put<uint64_t>(params, 64, o.ptr);
    put<uint64_t>(params, 80, c.lse ? lse.ptr : 0);
    put<int32_t>(params, 88, c.window);
    put<float>(params, 92, scale);
    put<int32_t>(params, 96, c.headDim);
    put<int32_t>(params, 100, c.headDimV);
    put<int32_t>(params, 104, maxQ);
    put<int32_t>(params, 108, maxK);
    put<uint8_t>(params, 116, uint8_t(c.mask));
    put<int32_t>(params, 120, qRow);
    put<int32_t>(params, 124, kRow);
    put<int32_t>(params, 128, vRow);
    put<int32_t>(params, 132, maxK);
    put<int32_t>(params, 136, oRow);
    put<int32_t>(params, 140, c.headDim);
    put<int32_t>(params, 144, c.headDim);
    put<int32_t>(params, 148, c.headDimV);
    put<int64_t>(params, 152, int64_t(maxQ) * maxK);
    put<int64_t>(params, 160, c.varlen ? 0 : int64_t(c.numQ) * qRow);
    put<int64_t>(params, 168, c.varlen ? 0 : int64_t(c.numK) * kRow);
    put<int64_t>(params, 176, c.varlen ? 0 : int64_t(c.numK) * vRow);
    put<int64_t>(params, 184, int64_t(c.heads) * maxQ * maxK);
    put<int32_t>(params, 192, c.batches);
    put<int32_t>(params, 196, c.heads);
    put<int32_t>(params, 200, c.heads / c.kvHeads);
    void* args[] = {params};
    const unsigned threads = c.queriesPerBlock == 32 ? 128 : 256;
    CHECK_CU(cuLaunchKernel(fn, (maxQ + c.queriesPerBlock - 1) / c.queriesPerBlock, c.heads, c.batches, threads, 1, 1,
                            0, nullptr, args, nullptr));
    const std::vector<float> out = o.download(), lseOut = lse.download();

    double worst = 0, worstLse = 0;
    const double tol = c.type == Type::F32 ? 1e-4 : c.type == Type::F16 ? 6e-3 : 4e-2;
    for (int b = 0; b < c.batches; ++b) {
        const int nq = seqQ[b + 1] - seqQ[b], nk = seqK[b + 1] - seqK[b];
        const int diag = c.mask == 2 ? nk - nq : 0;
        for (int h = 0; h < c.heads; ++h) {
            const int kh = h / (c.heads / c.kvHeads);
            for (int i = 0; i < nq; ++i) {
                std::vector<double> s(nk);
                double mx = -INFINITY;
                for (int j = 0; j < nk; ++j) {
                    double dot = 0;
                    for (int d = 0; d < c.headDim; ++d)
                        dot += double(q.values[size_t(seqQ[b] + i) * qRow + h * c.headDim + d]) *
                               k.values[size_t(seqK[b] + j) * kRow + kh * c.headDim + d];
                    s[j] = dot * scale;
                    if (c.bias) s[j] += bias.values[((size_t(b) * c.heads + h) * maxQ + i) * maxK + j];
                    if ((c.mask && j > i + diag) || (c.window && j <= i + diag - c.window)) s[j] = -INFINITY;
                    mx = std::max(mx, s[j]);
                }
                double sum = 0;
                for (int j = 0; j < nk; ++j) sum += s[j] = std::isinf(s[j]) ? 0.0 : std::exp(s[j] - mx);
                for (int d = 0; d < c.headDimV; ++d) {
                    double acc = 0;
                    for (int j = 0; j < nk; ++j) acc += s[j] * v.values[size_t(seqK[b] + j) * vRow + kh * c.headDimV + d];
                    const double ref = sum ? acc / sum : 0.0;
                    const double got = out[size_t(seqQ[b] + i) * oRow + h * c.headDimV + d];
                    worst = std::max(worst, std::abs(got - ref) / (1.0 + std::abs(ref)));
                }
                if (c.lse) {
                    const double ref = sum ? mx + std::log(sum) : 0.0;
                    worstLse = std::max(worstLse, std::abs(lseOut[(size_t(b) * c.heads + h) * lseDim + i] - ref));
                }
            }
            if (c.lse && !c.varlen)  // padding of the last CUDA block: +inf
                for (int i = nq; i < lseDim; ++i)
                    if (!std::isinf(lseOut[(size_t(b) * c.heads + h) * lseDim + i])) worstLse = 1e9;
        }
    }
    if (dSeqQ) cuMemFree(dSeqQ);
    if (dSeqK) cuMemFree(dSeqK);
    const bool ok = worst <= tol && worstLse <= 1e-3;
    std::printf("%-28s max error %.2e (tolerance %.0e), logsumexp %.2e %s\n", c.name, worst, tol, worstLse,
                ok ? "" : "FAILED");
    return ok;
}

}  // namespace

int main() {
    CHECK_CU(cuInit(0));
    CUdevice dev;
    CHECK_CU(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CHECK_CU(cuDevicePrimaryCtxRetain(&ctx, dev));
    CHECK_CU(cuCtxSetCurrent(ctx));

    std::string ptx = ".version 8.0\n.target sm_75\n.address_size 64\n";
    for (Type t : {Type::F16, Type::BF16, Type::F32})
        for (int qpb : {32, 64}) {
            const std::string name = kernelName(t, qpb);
            if (ptx.find(name) == std::string::npos)
                ptx += ".visible .entry " + name + "(.param .align 8 .b8 " + name + "_param_0[272]) { ret; }\n";
        }
    CUmodule mod;
    CHECK_CU(cuModuleLoadData(&mod, ptx.c_str()));

    const Case cases[] = {
        {"f16 d64", Type::F16, 64, 2, 3, 3, 100, 77, 64, 64},
        {"f16 d128 causal lse", Type::F16, 32, 1, 2, 2, 130, 200, 128, 128, 2, 0, false, true},
        {"f16 d128 long", Type::F16, 32, 1, 2, 2, 300, 1000, 128, 128},
        {"f16 d80 odd strides", Type::F16, 64, 2, 2, 1, 45, 70, 80, 72, 1, 0, false, false, false, 3},
        {"f16 varlen", Type::F16, 32, 3, 2, 2, 90, 120, 64, 64, 0, 0, false, true, true},
        {"bf16 bias gqa", Type::BF16, 64, 2, 4, 2, 70, 90, 128, 128, 0, 0, true, true},
        {"f32 window", Type::F32, 32, 1, 2, 2, 60, 60, 40, 40, 1, 16},
        {"f32 d200 bias", Type::F32, 32, 1, 1, 1, 33, 50, 200, 136, 0, 0, true, true},
    };
    std::mt19937 rng(7);
    bool ok = true;
    const char* only = std::getenv("ATTENTION_CASE");  // debugging: one case by index
    for (size_t i = 0; i < std::size(cases); ++i)
        if (!only || std::atoi(only) == int(i)) ok &= run(cases[i], mod, rng);
    CHECK_CU(cuModuleUnload(mod));
    CHECK_CU(cuDevicePrimaryCtxRelease(dev));
    if (!ok) return 1;
    std::printf("OK\n");
    return 0;
}
