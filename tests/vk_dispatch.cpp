// Validates the Vulkan half of the kernel ABI with a hand-written GLSL kernel:
// BDA pointers + CUDA-layout parameters in push constants + blockDim spec constants.

#include "vk/device.h"
#include "vk/instance.h"

#include <cstdio>
#include <fstream>
#include <vector>

using namespace vulkanizer::vk;

#define CHECK_VK(call)                                                                     \
    do {                                                                                   \
        VkResult r_ = (call);                                                              \
        if (r_ != VK_SUCCESS) {                                                            \
            std::fprintf(stderr, "%s:%d: %s failed: %d\n", __FILE__, __LINE__, #call, r_); \
            return 1;                                                                      \
        }                                                                                  \
    } while (0)

int main() {
    Instance* inst = Instance::get();
    if (!inst || inst->devices().empty()) {
        std::fprintf(stderr, "no usable Vulkan device\n");
        return 1;
    }
    std::unique_ptr<Device> dev;
    CHECK_VK(Device::create(inst->devices()[0], dev));
    std::printf("device: %s (subgroup %u)\n", dev->info().name(), dev->info().subgroupSize);

    std::ifstream in(VULKANIZER_SHADER_DIR "/vec_add.comp.spv", std::ios::binary | std::ios::ate);
    if (!in) {
        std::fprintf(stderr, "cannot open shader\n");
        return 1;
    }
    std::vector<uint32_t> spirv(static_cast<size_t>(in.tellg()) / 4);
    in.seekg(0);
    in.read(reinterpret_cast<char*>(spirv.data()), static_cast<std::streamsize>(spirv.size() * 4));

    constexpr int n = 1'000'003;
    std::vector<float> a(n), b(n), c(n, -1.0f);
    for (int i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = 2.0f;
    }

    Buffer da, db, dc;
    CHECK_VK(dev->allocateBuffer(n * sizeof(float), da));
    CHECK_VK(dev->allocateBuffer(n * sizeof(float), db));
    CHECK_VK(dev->allocateBuffer(n * sizeof(float), dc));
    CHECK_VK(dev->write(da, 0, a.data(), n * sizeof(float)));
    CHECK_VK(dev->write(db, 0, b.data(), n * sizeof(float)));
    CHECK_VK(dev->fill(dc, 0, n * sizeof(float), 0));

    // Byte-identical to the CUDA parameter buffer of vecAdd(float*, float*, float*, int).
    struct Params {
        uint64_t a, b, c;
        int32_t n;
    } params{da.address, db.address, dc.address, n};
    constexpr uint32_t kParamBytes = 28;

    const uint32_t spec[4] = {256, 1, 1, 0};  // blockDim, dynamic shared words
    ComputePipeline pipeline;
    CHECK_VK(dev->createComputePipeline(spirv.data(), spirv.size(), "main", kParamBytes, spec, pipeline));
    CHECK_VK(dev->dispatch(pipeline, (n + 255) / 256, 1, 1, &params, kParamBytes));
    CHECK_VK(dev->read(dc, 0, c.data(), n * sizeof(float)));

    for (int i = 0; i < n; ++i) {
        if (c[i] != a[i] + 2.0f) {
            std::fprintf(stderr, "mismatch at %d: %f\n", i, c[i]);
            return 1;
        }
    }

    dev->destroyComputePipeline(pipeline);
    dev->freeBuffer(da);
    dev->freeBuffer(db);
    dev->freeBuffer(dc);
    std::printf("OK\n");
    return 0;
}
