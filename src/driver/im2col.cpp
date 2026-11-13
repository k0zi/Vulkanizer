// PyTorch's im2col_kernel as Vulkanizer's own shader (see im2col.h).

#include "driver/im2col.h"

#include "common/log.h"

#include <cstring>
#include <regex>

using namespace vulkanizer;
using namespace vulkanizer::driver;

namespace {

const uint32_t kIm2col16Spirv[] = {
#include "im2col_16.comp.inc"
};
const uint32_t kIm2col32Spirv[] = {
#include "im2col_32.comp.inc"
};
const uint32_t kIm2col64Spirv[] = {
#include "im2col_64.comp.inc"
};

// The push constants of shaders/im2col.comp.
struct Push {
    uint64_t im, col;
    int32_t channels, height, width, kh, kw, ph, pw, sh, sw, dh, dw, hcol, wcol;
    uint32_t vectorStores;
};

}  // namespace

namespace vulkanizer::driver {

uint32_t matchIm2colKernel(const std::string& name) {
    static const bool off = [] {
        const char* v = std::getenv("VULKANIZER_BUILTIN_KERNELS");
        return v && std::strcmp(v, "0") == 0;
    }();
    // template <typename dt> __global__ void im2col_kernel(int64_t n, const dt* data_im, int64_t height,
    //     int64_t width, int64_t kernel_height, int64_t kernel_width, int64_t pad_height, int64_t pad_width,
    //     int64_t stride_height, int64_t stride_width, int64_t dilation_height, int64_t dilation_width,
    //     int64_t height_col, int64_t width_col, dt* data_col)
    static const std::regex pattern(R"(^_ZN2at6native13im2col_kernelI(.+)EEvlPKT_llllllllllllPS\d*_$)");
    std::smatch m;
    if (off || !std::regex_match(name, m, pattern)) return 0;
    const std::string t = m[1];
    if (t == "N3c104HalfE" || t == "N3c108BFloat16E") return 2;
    if (t == "f") return 4;
    if (t == "d" || t == "N3c107complexIfEE") return 8;
    return 0;
}

CUresult launchIm2col(ApiScope& s, CUfunction f, const std::vector<uint8_t>& params) {
    const ptx::Entry& e = *f->entry;
    if (e.params.size() != 15 || params.size() < 120) {
        VULKANIZER_ERROR("'%s': unexpected parameters", e.name.c_str());
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    int64_t v[15];
    for (int i = 0; i < 15; ++i) std::memcpy(&v[i], params.data() + e.params[i].offset, 8);
    const int64_t n = v[0], hcol = v[12], wcol = v[13];
    if (n <= 0 || hcol <= 0 || wcol <= 0) return CUDA_SUCCESS;
    for (int i = 2; i < 14; ++i)
        if (v[i] < 0 || v[i] > INT32_MAX) {
            VULKANIZER_ERROR("'%s': sizes beyond 32 bits are not supported", e.name.c_str());
            return CUDA_ERROR_NOT_SUPPORTED;
        }
    const uint32_t elem = f->builtinParam, vw = 16 / elem;
    Push p{};
    p.im = uint64_t(v[1]);
    p.col = uint64_t(v[14]);
    p.channels = int32_t(n / (hcol * wcol));
    p.height = int32_t(v[2]), p.width = int32_t(v[3]), p.kh = int32_t(v[4]), p.kw = int32_t(v[5]);
    p.ph = int32_t(v[6]), p.pw = int32_t(v[7]), p.sh = int32_t(v[8]), p.sw = int32_t(v[9]);
    p.dh = int32_t(v[10]), p.dw = int32_t(v[11]), p.hcol = int32_t(hcol), p.wcol = int32_t(wcol);
    p.vectorStores = p.col % 16 == 0 && wcol % vw == 0;

    const uint32_t* code = elem == 2 ? kIm2col16Spirv : elem == 4 ? kIm2col32Spirv : kIm2col64Spirv;
    const size_t words = elem == 2 ? std::size(kIm2col16Spirv) : elem == 4 ? std::size(kIm2col32Spirv)
                                                                           : std::size(kIm2col64Spirv);
    const std::array<uint32_t, 5> spec{64, 1, 1, 0, 1};
    auto it = f->pipelines.find(spec);
    if (it == f->pipelines.end()) {
        vk::ComputePipeline pipeline;
        if (VkResult vr = s.dev->createComputePipeline(code, words, "main", sizeof(Push), spec.data(), pipeline)) {
            VULKANIZER_ERROR("'%s': im2col pipeline creation failed: %d", e.name.c_str(), vr);
            return toCuResult(vr);
        }
        it = f->pipelines.emplace(spec, pipeline).first;
    }
    const uint64_t invocations = uint64_t(p.channels) * uint64_t(hcol) * uint64_t((wcol + vw - 1) / vw);
    const uint64_t groups = (invocations + 63) / 64;
    if (groups > UINT32_MAX) return CUDA_ERROR_INVALID_VALUE;
    return toCuResult(s.dev->dispatch(it->second, uint32_t(groups), 1, 1, &p, sizeof p, "im2col"));
}

}  // namespace vulkanizer::driver
