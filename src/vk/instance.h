#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>
#include <vector>

namespace vulkanizer::vk {

// Everything Vulkanizer needs to know about a physical device, gathered once at startup.
struct PhysicalDeviceInfo {
    VkPhysicalDevice handle = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties props{};
    VkPhysicalDeviceMemoryProperties memProps{};
    uint32_t apiVersion = 0;  // min(instance, device) API version
    uint32_t subgroupSize = 0;
    uint32_t minSubgroupSize = 0;
    uint32_t maxSubgroupSize = 0;
    VkSubgroupFeatureFlags subgroupOps = 0;  // supported in compute shaders
    bool dot4x8Accelerated = false;          // 4x8-bit packed integer dot products run in hardware
    VkDeviceSize maxMemoryAllocationSize = 0;
    VkDeviceSize maxBufferSize = 0;  // maintenance4 (Vulkan 1.3); else maxMemoryAllocationSize
    VkDeviceSize deviceLocalBytes = 0;
    uint8_t uuid[VK_UUID_SIZE]{};
    uint32_t computeQueueFamily = 0;
    bool computeQueueSparse = false;  // the compute queue family supports sparse binding
    uint32_t computeUnits = 0;        // active compute units (VK_AMD_shader_core_properties2); 0: unknown
    bool hasPciInfo = false;          // VK_EXT_pci_bus_info
    uint32_t pciDomain = 0, pciBus = 0, pciDevice = 0;

    bool isCpu() const { return props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU; }
    const char* name() const { return props.deviceName; }
    const char* typeName() const;
};

// Process-wide Vulkan instance plus the list of devices usable by Vulkanizer, ordered
// discrete > integrated > virtual > other > cpu. Filtered by VULKANIZER_DEVICE (name
// substring) and VULKANIZER_ALLOW_CPU. Intentionally never destroyed: driver state must
// stay valid through the application's own atexit handlers.
class Instance {
public:
    // Returns nullptr when Vulkan is unavailable.
    static Instance* get();

    VkInstance handle() const { return instance_; }
    uint32_t apiVersion() const { return apiVersion_; }
    const std::vector<PhysicalDeviceInfo>& devices() const { return devices_; }

private:
    Instance() = default;
    ~Instance();
    bool init();
    bool queryDevice(VkPhysicalDevice pd, PhysicalDeviceInfo& out, std::string& whyNot) const;

    VkInstance instance_ = VK_NULL_HANDLE;
    uint32_t apiVersion_ = 0;
    std::vector<PhysicalDeviceInfo> devices_;
};

}  // namespace vulkanizer::vk
