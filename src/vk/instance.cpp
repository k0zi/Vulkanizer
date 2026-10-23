#include "vk/instance.h"

#include "common/log.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>

namespace vulkanizer::vk {

static bool envFlag(const char* name) {
    const char* v = std::getenv(name);
    return v && *v && std::strcmp(v, "0") != 0;
}

static std::string toLower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

static int typeRank(VkPhysicalDeviceType t) {
    switch (t) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return 0;
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 1;
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return 2;
        case VK_PHYSICAL_DEVICE_TYPE_CPU: return 4;
        default: return 3;
    }
}

const char* PhysicalDeviceInfo::typeName() const {
    switch (props.deviceType) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return "discrete";
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "integrated";
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return "virtual";
        case VK_PHYSICAL_DEVICE_TYPE_CPU: return "cpu";
        default: return "other";
    }
}

Instance* Instance::get() {
    static Instance* instance = [] {
        auto* inst = new Instance();
        if (!inst->init()) {
            delete inst;
            return static_cast<Instance*>(nullptr);
        }
        return inst;
    }();
    return instance;
}

Instance::~Instance() {
    if (instance_) vkDestroyInstance(instance_, nullptr);
}

bool Instance::init() {
    uint32_t loaderVersion = VK_API_VERSION_1_0;
    auto enumerateVersion = reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
        vkGetInstanceProcAddr(nullptr, "vkEnumerateInstanceVersion"));
    if (enumerateVersion) enumerateVersion(&loaderVersion);
    if (loaderVersion < VK_API_VERSION_1_2) {
        VULKANIZER_ERROR("Vulkan loader version %u.%u is too old (need 1.2)", VK_API_VERSION_MAJOR(loaderVersion),
                    VK_API_VERSION_MINOR(loaderVersion));
        return false;
    }
    apiVersion_ = std::min<uint32_t>(loaderVersion, VK_API_VERSION_1_3);

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "vulkanizer";
    app.pEngineName = "vulkanizer";
    app.apiVersion = apiVersion_;

    std::vector<const char*> layers;
    if (envFlag("VULKANIZER_VALIDATION")) {
        uint32_t count = 0;
        vkEnumerateInstanceLayerProperties(&count, nullptr);
        std::vector<VkLayerProperties> available(count);
        vkEnumerateInstanceLayerProperties(&count, available.data());
        bool found = std::any_of(available.begin(), available.end(), [](const VkLayerProperties& l) {
            return std::strcmp(l.layerName, "VK_LAYER_KHRONOS_validation") == 0;
        });
        if (found)
            layers.push_back("VK_LAYER_KHRONOS_validation");
        else
            VULKANIZER_WARN("VULKANIZER_VALIDATION set but VK_LAYER_KHRONOS_validation is not installed");
    }

    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &app;
    ci.enabledLayerCount = static_cast<uint32_t>(layers.size());
    ci.ppEnabledLayerNames = layers.data();
    VkResult r = vkCreateInstance(&ci, nullptr, &instance_);
    if (r != VK_SUCCESS) {
        VULKANIZER_ERROR("vkCreateInstance failed: %d", r);
        return false;
    }

    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance_, &count, nullptr);
    std::vector<VkPhysicalDevice> pds(count);
    vkEnumeratePhysicalDevices(instance_, &count, pds.data());

    const char* filterEnv = std::getenv("VULKANIZER_DEVICE");
    std::string filter = filterEnv ? toLower(filterEnv) : std::string();
    bool allowCpu = envFlag("VULKANIZER_ALLOW_CPU") || !filter.empty();

    for (VkPhysicalDevice pd : pds) {
        PhysicalDeviceInfo info;
        std::string whyNot;
        if (!queryDevice(pd, info, whyNot)) {
            VULKANIZER_INFO("skipping Vulkan device '%s': %s", info.name(), whyNot.c_str());
            continue;
        }
        if (!filter.empty() && toLower(info.name()).find(filter) == std::string::npos) continue;
        if (info.isCpu() && !allowCpu) {
            VULKANIZER_INFO("skipping CPU Vulkan device '%s' (set VULKANIZER_ALLOW_CPU=1)", info.name());
            continue;
        }
        devices_.push_back(info);
    }
    std::stable_sort(devices_.begin(), devices_.end(), [](const PhysicalDeviceInfo& a, const PhysicalDeviceInfo& b) {
        return typeRank(a.props.deviceType) < typeRank(b.props.deviceType);
    });
    for (size_t i = 0; i < devices_.size(); ++i)
        VULKANIZER_INFO("device %zu: %s (%s, subgroup %u)", i, devices_[i].name(), devices_[i].typeName(),
                   devices_[i].subgroupSize);
    return true;
}

bool Instance::queryDevice(VkPhysicalDevice pd, PhysicalDeviceInfo& out, std::string& whyNot) const {
    out.handle = pd;
    vkGetPhysicalDeviceProperties(pd, &out.props);
    vkGetPhysicalDeviceMemoryProperties(pd, &out.memProps);
    out.apiVersion = std::min(apiVersion_, out.props.apiVersion);
    if (out.apiVersion < VK_API_VERSION_1_2) {
        whyNot = "Vulkan 1.2 required";
        return false;
    }

    VkPhysicalDeviceVulkan13Properties p13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES};
    VkPhysicalDeviceVulkan11Properties p11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES};
    VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    p2.pNext = &p11;
    VkPhysicalDeviceShaderIntegerDotProductProperties pdot{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES};
    if (out.apiVersion >= VK_API_VERSION_1_3) p11.pNext = &p13, p13.pNext = &pdot;
    vkGetPhysicalDeviceProperties2(pd, &p2);
    out.dot4x8Accelerated = pdot.integerDotProduct4x8BitPackedSignedAccelerated &&
                            pdot.integerDotProduct4x8BitPackedUnsignedAccelerated &&
                            pdot.integerDotProduct4x8BitPackedMixedSignednessAccelerated;
    out.subgroupSize = p11.subgroupSize;
    if (p11.subgroupSupportedStages & VK_SHADER_STAGE_COMPUTE_BIT) out.subgroupOps = p11.subgroupSupportedOperations;
    out.maxMemoryAllocationSize = p11.maxMemoryAllocationSize;
    std::memcpy(out.uuid, p11.deviceUUID, VK_UUID_SIZE);
    if (out.apiVersion >= VK_API_VERSION_1_3) {
        out.minSubgroupSize = p13.minSubgroupSize;
        out.maxSubgroupSize = p13.maxSubgroupSize;
        out.maxBufferSize = p13.maxBufferSize;
    } else {
        out.minSubgroupSize = out.maxSubgroupSize = out.subgroupSize;
    }
    if (!out.maxBufferSize) out.maxBufferSize = out.maxMemoryAllocationSize;

    // The number of compute units (CUDA's multiprocessor count) and the PCI address, where the
    // driver tells them.
    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> exts(extCount);
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &extCount, exts.data());
    for (const VkExtensionProperties& e : exts)
        if (!std::strcmp(e.extensionName, VK_AMD_SHADER_CORE_PROPERTIES_2_EXTENSION_NAME)) {
            VkPhysicalDeviceShaderCoreProperties2AMD core2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_2_AMD};
            VkPhysicalDeviceProperties2 q{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
            q.pNext = &core2;
            vkGetPhysicalDeviceProperties2(pd, &q);
            out.computeUnits = core2.activeComputeUnitCount;
        } else if (!std::strcmp(e.extensionName, VK_EXT_PCI_BUS_INFO_EXTENSION_NAME)) {
            VkPhysicalDevicePCIBusInfoPropertiesEXT pci{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT};
            VkPhysicalDeviceProperties2 q{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
            q.pNext = &pci;
            vkGetPhysicalDeviceProperties2(pd, &q);
            out.hasPciInfo = true;
            out.pciDomain = pci.pciDomain;
            out.pciBus = pci.pciBus;
            out.pciDevice = pci.pciDevice;
        }

    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.pNext = &f12;
    vkGetPhysicalDeviceFeatures2(pd, &f2);
    if (!f12.bufferDeviceAddress) {
        whyNot = "bufferDeviceAddress not supported";
        return false;
    }
    if (!f2.features.shaderInt64) {
        whyNot = "shaderInt64 not supported";
        return false;
    }

    uint32_t qfCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qfCount, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(qfCount);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qfCount, qfs.data());
    int computeOnly = -1, anyCompute = -1;
    for (uint32_t i = 0; i < qfCount; ++i) {
        if (!(qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) continue;
        if (anyCompute < 0) anyCompute = static_cast<int>(i);
        if (!(qfs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && computeOnly < 0) computeOnly = static_cast<int>(i);
    }
    if (anyCompute < 0) {
        whyNot = "no compute queue";
        return false;
    }
    out.computeQueueFamily = static_cast<uint32_t>(computeOnly >= 0 ? computeOnly : anyCompute);
    out.computeQueueSparse = qfs[out.computeQueueFamily].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT;

    for (uint32_t i = 0; i < out.memProps.memoryHeapCount; ++i)
        if (out.memProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            out.deviceLocalBytes += out.memProps.memoryHeaps[i].size;
    return true;
}

}  // namespace vulkanizer::vk
