// Lists the Vulkan devices Vulkanizer can use, including CPU implementations.
//   vulkanizer-info          human-readable summary
//   vulkanizer-info --names  one device name per line (for scripts)

#include "vk/instance.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

using vulkanizer::vk::Instance;
using vulkanizer::vk::PhysicalDeviceInfo;

int main(int argc, char** argv) {
    const bool namesOnly = argc > 1 && std::strcmp(argv[1], "--names") == 0;
    setenv("VULKANIZER_ALLOW_CPU", "1", 0);

    Instance* inst = Instance::get();
    if (!inst) {
        std::fprintf(stderr, "Vulkan is not available\n");
        return 1;
    }
    if (inst->devices().empty()) {
        std::fprintf(stderr, "no usable Vulkan device\n");
        return 1;
    }
    for (size_t i = 0; i < inst->devices().size(); ++i) {
        const PhysicalDeviceInfo& d = inst->devices()[i];
        if (namesOnly) {
            std::printf("%s\n", d.name());
            continue;
        }
        const auto& lim = d.props.limits;
        std::printf("[%zu] %s\n", i, d.name());
        std::printf("    type            %s, Vulkan %u.%u.%u\n", d.typeName(), VK_API_VERSION_MAJOR(d.props.apiVersion),
                    VK_API_VERSION_MINOR(d.props.apiVersion), VK_API_VERSION_PATCH(d.props.apiVersion));
        std::printf("    subgroup        %u (min %u, max %u)\n", d.subgroupSize, d.minSubgroupSize, d.maxSubgroupSize);
        std::printf("    workgroup       %u invocations, %u x %u x %u\n", lim.maxComputeWorkGroupInvocations,
                    lim.maxComputeWorkGroupSize[0], lim.maxComputeWorkGroupSize[1], lim.maxComputeWorkGroupSize[2]);
        std::printf("    dispatch        %u x %u x %u groups\n", lim.maxComputeWorkGroupCount[0],
                    lim.maxComputeWorkGroupCount[1], lim.maxComputeWorkGroupCount[2]);
        std::printf("    shared memory   %u KiB\n", lim.maxComputeSharedMemorySize / 1024);
        std::printf("    push constants  %u bytes\n", lim.maxPushConstantsSize);
        std::printf("    device-local    %.1f GiB (max allocation %.1f GiB)\n",
                    double(d.deviceLocalBytes) / (1ull << 30), double(d.maxMemoryAllocationSize) / (1ull << 30));
    }
    return 0;
}
