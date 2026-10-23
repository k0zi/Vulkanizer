#include "vk/device.h"

#include "common/log.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cxxabi.h>
#include <map>
#include <string>
#include <sys/mman.h>

namespace vulkanizer::vk {

// AMD device-coherent/uncached types need VK_AMD_device_coherent_memory; protected
// and lazily allocated memory is never what we want for compute buffers.
static constexpr VkMemoryPropertyFlags kAvoidedMemoryFlags =
    VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD | VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD |
    VK_MEMORY_PROPERTY_PROTECTED_BIT | VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT;

static constexpr VkBufferUsageFlags kDeviceBufferUsage =
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

static constexpr VkBufferUsageFlags kStagingUsage =
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

const uint32_t kFmaProbeSpirv[] = {
#include "fma_probe.comp.inc"
};

const uint32_t kFmaProbeF64Spirv[] = {
#include "fma_probe_f64.comp.inc"
};

// VULKANIZER_PROFILE=1: GPU time per kernel, summed over all devices and printed at exit.
namespace {

struct ProfileEntry {
    uint64_t count = 0;
    double ns = 0;
};

std::mutex& profileMutex() {
    static auto* m = new std::mutex;
    return *m;
}
// Per batch: from the first command's start to the last one's end, and the gaps in between.
struct ProfileSpan {
    double busyNs = 0, gapNs = 0;
    uint64_t batches = 0;
};
ProfileSpan& profileSpan() {
    static ProfileSpan span;
    return span;
}

std::map<std::string, ProfileEntry>& profile() {
    static auto* p = new std::map<std::string, ProfileEntry>;
    return *p;
}

void printProfile() {
    std::lock_guard<std::mutex> lock(profileMutex());
    std::vector<std::pair<std::string, ProfileEntry>> rows(profile().begin(), profile().end());
    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.second.ns > b.second.ns; });
    double total = 0;
    uint64_t launches = 0;
    for (const auto& r : rows) total += r.second.ns, launches += r.second.count;
    std::fprintf(stderr, "\n[vulkanizer] GPU time per kernel (%llu launches, %.3f ms in total):\n",
                 (unsigned long long)launches, total * 1e-6);
    const ProfileSpan& span = profileSpan();
    std::fprintf(stderr, "[vulkanizer] %llu batches: %.3f ms from first start to last end, %.3f ms of it between commands\n",
                 (unsigned long long)span.batches, span.busyNs * 1e-6, span.gapNs * 1e-6);
    std::fprintf(stderr, "  %10s %8s %10s %6s  %s\n", "total ms", "count", "avg us", "%", "kernel");
    for (size_t i = 0; i < rows.size() && i < 40; ++i) {
        const auto& [name, e] = rows[i];
        std::fprintf(stderr, "  %10.3f %8llu %10.2f %5.1f%%  %s\n", e.ns * 1e-6, (unsigned long long)e.count,
                     e.ns * 1e-3 / double(e.count), 100.0 * e.ns / total, name.c_str());
    }
}

std::string displayName(const char* label) {
    if (!label) return "(unnamed)";
    int status = 0;
    char* d = abi::__cxa_demangle(label, nullptr, nullptr, &status);
    std::string s = status == 0 && d ? d : label;
    std::free(d);
    if (s.size() > 110) s = s.substr(0, 107) + "...";
    return s;
}

}  // namespace

VkResult Device::create(const PhysicalDeviceInfo& info, std::unique_ptr<Device>& out) {
    std::unique_ptr<Device> dev(new Device(info));
    VkResult r = dev->init();
    if (r != VK_SUCCESS) return r;
    out = std::move(dev);
    return VK_SUCCESS;
}

VkResult Device::init() {
    const bool has13 = info_.apiVersion >= VK_API_VERSION_1_3;

    // Query what the device supports...
    VkPhysicalDeviceVulkan13Features s13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceVulkan12Features s12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan11Features s11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceFeatures2 sup{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    sup.pNext = &s11;
    s11.pNext = &s12;
    if (has13) s12.pNext = &s13;
    vkGetPhysicalDeviceFeatures2(info_.handle, &sup);

    // ...and enable the subset that matters for compute.
    VkPhysicalDeviceVulkan13Features e13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceVulkan12Features e12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan11Features e11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceFeatures2 en{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    en.pNext = &e11;
    e11.pNext = &e12;
    if (has13) e12.pNext = &e13;

    en.features.shaderInt64 = VK_TRUE;
    en.features.shaderInt16 = sup.features.shaderInt16;
    en.features.shaderFloat64 = sup.features.shaderFloat64;
    e11.storageBuffer16BitAccess = s11.storageBuffer16BitAccess;
    e11.variablePointersStorageBuffer = s11.variablePointersStorageBuffer;
    e11.variablePointers = s11.variablePointers;
    e12.bufferDeviceAddress = VK_TRUE;
    e12.timelineSemaphore = s12.timelineSemaphore;
    e12.hostQueryReset = s12.hostQueryReset;
    e12.shaderInt8 = s12.shaderInt8;
    e12.shaderFloat16 = s12.shaderFloat16;
    e12.storageBuffer8BitAccess = s12.storageBuffer8BitAccess;
    e12.scalarBlockLayout = s12.scalarBlockLayout;
    e12.vulkanMemoryModel = s12.vulkanMemoryModel;
    e12.vulkanMemoryModelDeviceScope = s12.vulkanMemoryModelDeviceScope;
    e12.shaderBufferInt64Atomics = s12.shaderBufferInt64Atomics;
    e12.shaderSharedInt64Atomics = s12.shaderSharedInt64Atomics;
    // CUDA virtual memory: sparse buffers, several of them side by side for reservations larger
    // than maxBufferSize (VULKANIZER_NO_VMM=1 turns it off).
    const char* noVmm = std::getenv("VULKANIZER_NO_VMM");
    if (info_.computeQueueSparse && sup.features.sparseBinding && sup.features.sparseResidencyBuffer &&
        !(noVmm && *noVmm && std::strcmp(noVmm, "0") != 0)) {
        en.features.sparseBinding = VK_TRUE;
        en.features.sparseResidencyBuffer = VK_TRUE;
        e12.bufferDeviceAddressCaptureReplay = s12.bufferDeviceAddressCaptureReplay;
    }
    // Host memory the device can use at a chosen address (CUDA pinned, registered and managed
    // memory): imported host pages (registered memory) or memory of our own mapped at a chosen
    // address (VK_EXT_map_memory_placed), bound to a sparse buffer placed with capture/replay.
    std::vector<const char*> extensions;
    bool hostImport = false;
    VkPhysicalDeviceMapMemoryPlacedFeaturesEXT placed{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAP_MEMORY_PLACED_FEATURES_EXT};
    if (en.features.sparseBinding && e12.bufferDeviceAddressCaptureReplay) {
        uint32_t count = 0;
        vkEnumerateDeviceExtensionProperties(info_.handle, nullptr, &count, nullptr);
        std::vector<VkExtensionProperties> props(count);
        vkEnumerateDeviceExtensionProperties(info_.handle, nullptr, &count, props.data());
        auto has = [&](const char* name) {
            for (const VkExtensionProperties& p : props)
                if (!std::strcmp(p.extensionName, name)) return true;
            return false;
        };
        if (has(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME)) {
            extensions.push_back(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
            hostImport = true;
        }
        if (has(VK_EXT_MAP_MEMORY_PLACED_EXTENSION_NAME) && has(VK_KHR_MAP_MEMORY_2_EXTENSION_NAME)) {
            VkPhysicalDeviceMapMemoryPlacedFeaturesEXT supported{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAP_MEMORY_PLACED_FEATURES_EXT};
            VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
            f2.pNext = &supported;
            vkGetPhysicalDeviceFeatures2(info_.handle, &f2);
            if (supported.memoryMapPlaced && supported.memoryUnmapReserve) {
                extensions.push_back(VK_KHR_MAP_MEMORY_2_EXTENSION_NAME);
                extensions.push_back(VK_EXT_MAP_MEMORY_PLACED_EXTENSION_NAME);
                placed.memoryMapPlaced = VK_TRUE;
                placed.memoryUnmapReserve = VK_TRUE;
                placed.pNext = en.pNext;
                en.pNext = &placed;
            }
        }
    }
    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR pipelineStats{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR};
    if (const char* env = std::getenv("VULKANIZER_PIPELINE_STATS"); env && *env && std::strcmp(env, "0") != 0) {
        uint32_t count = 0;
        vkEnumerateDeviceExtensionProperties(info_.handle, nullptr, &count, nullptr);
        std::vector<VkExtensionProperties> props(count);
        vkEnumerateDeviceExtensionProperties(info_.handle, nullptr, &count, props.data());
        for (const VkExtensionProperties& p : props)
            if (!std::strcmp(p.extensionName, VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME)) {
                extensions.push_back(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);
                pipelineStats.pipelineExecutableInfo = VK_TRUE;
                pipelineStats.pNext = en.pNext;
                en.pNext = &pipelineStats;
            }
    }
    if (has13) {
        e13.subgroupSizeControl = s13.subgroupSizeControl;
        e13.computeFullSubgroups = s13.computeFullSubgroups;
        e13.maintenance4 = s13.maintenance4;
        e13.shaderIntegerDotProduct = s13.shaderIntegerDotProduct;
    }

    features_.shaderInt8 = e12.shaderInt8;
    features_.shaderInt16 = en.features.shaderInt16;
    features_.shaderFloat16 = e12.shaderFloat16;
    features_.shaderFloat64 = en.features.shaderFloat64;
    features_.storageBuffer8Bit = e12.storageBuffer8BitAccess;
    features_.storageBuffer16Bit = e11.storageBuffer16BitAccess;
    features_.bufferInt64Atomics = e12.shaderBufferInt64Atomics;
    features_.sharedInt64Atomics = e12.shaderSharedInt64Atomics;
    features_.vulkanMemoryModel = e12.vulkanMemoryModel;
    features_.subgroupSizeControl = e13.subgroupSizeControl;
    features_.computeFullSubgroups = e13.computeFullSubgroups;
    features_.timelineSemaphore = e12.timelineSemaphore;
    features_.sparseBuffers = en.features.sparseResidencyBuffer;
    features_.addressCaptureReplay = e12.bufferDeviceAddressCaptureReplay;
    // Without hardware support drivers lower it into byte extracts and full 32-bit multiplies;
    // the compiler's own emulation does better.
    features_.integerDotProduct = e13.shaderIntegerDotProduct && info_.dot4x8Accelerated;

    float priority = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = info_.computeQueueFamily;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;

    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &en;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = uint32_t(extensions.size());
    dci.ppEnabledExtensionNames = extensions.data();
    VkResult r = vkCreateDevice(info_.handle, &dci, nullptr, &device_);
    if (r != VK_SUCCESS) {
        VULKANIZER_ERROR("vkCreateDevice failed for '%s': %d", info_.name(), r);
        return r;
    }
    vkGetDeviceQueue(device_, info_.computeQueueFamily, 0, &queue_);

    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = info_.computeQueueFamily;
    if ((r = vkCreateCommandPool(device_, &pci, nullptr, &pool_)) != VK_SUCCESS) return r;

    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = pool_;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    if ((r = vkAllocateCommandBuffers(device_, &cai, &cmd_)) != VK_SUCCESS) return r;

    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if ((r = vkCreateFence(device_, &fci, nullptr, &fence_)) != VK_SUCCESS) return r;
    for (Batch& b : batches_) {
        if ((r = vkAllocateCommandBuffers(device_, &cai, &b.cmd)) != VK_SUCCESS) return r;
        if ((r = vkCreateFence(device_, &fci, nullptr, &b.fence)) != VK_SUCCESS) return r;
    }

    // GPU timestamps for CUDA events, if the queue supports them.
    uint32_t familyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(info_.handle, &familyCount, nullptr);
    std::vector<VkQueueFamilyProperties> families(familyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(info_.handle, &familyCount, families.data());
    if (e12.hostQueryReset && families[info_.computeQueueFamily].timestampValidBits &&
        info_.props.limits.timestampPeriod > 0) {
        VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qpi.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpi.queryCount = kTimestamps;
        if (vkCreateQueryPool(device_, &qpi, nullptr, &timestamps_) == VK_SUCCESS) {
            timestampPeriodNs_ = info_.props.limits.timestampPeriod;
            timestampSerial_.assign(kTimestamps, 0);
            for (int32_t i = kTimestamps - 1; i >= 0; --i) freeTimestamps_.push_back(i);
        }
    }

    // Every dispatch is timed: the measurements size the batches and feed VULKANIZER_PROFILE.
    if (timestamps_) {
        VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qpi.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpi.queryCount = 2 * kMaxCommandsPerBatch;
        timing_ = true;
        for (Batch& b : batches_)
            if (vkCreateQueryPool(device_, &qpi, nullptr, &b.timing) != VK_SUCCESS) timing_ = false;
    }
    if (const char* env = std::getenv("VULKANIZER_PROFILE"); env && *env && std::strcmp(env, "0") != 0 && timing_) {
        profiling_ = true;
        static std::once_flag once;
        std::call_once(once, [] { std::atexit(printProfile); });
    }
    double budgetMs = 50;
    if (const char* env = std::getenv("VULKANIZER_BATCH_MS"); env && *env) budgetMs = std::atof(env);
    if (timing_ && budgetMs > 0) batchBudgetNs_ = budgetMs * 1e6;
    // As with NVIDIA's driver: every launch waits for completion, so errors (and GPU hangs) show
    // at the launch that caused them.
    if (const char* env = std::getenv("CUDA_LAUNCH_BLOCKING"); env && std::strcmp(env, "1") == 0) launchBlocking_ = true;
    if (const char* env = std::getenv("VULKANIZER_NO_SUBALLOC"); env && *env && std::strcmp(env, "0") != 0)
        subAlloc_ = false;

    // The top address bits, from a buffer's actual address: the pipelines need them from now on.
    {
        Buffer probe;
        if (VkResult r = createBuffer(16, kDeviceBufferUsage, {0}, probe)) return r;
        VkBufferDeviceAddressInfo bai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        bai.buffer = probe.buffer;
        addressHigh_ = uint32_t(vkGetBufferDeviceAddress(device_, &bai) >> 32) & 0xFFFF8000u;
        destroyBuffer(probe);
    }
    features_.fusedFma = probeFusedFma();
    features_.fusedFma64 = features_.shaderFloat64 && probeFusedFma64();
    if (features_.sparseBuffers) probeSparse();
    if (pipelineStats.pipelineExecutableInfo)
        getPipelineStats_ = reinterpret_cast<PFN_vkGetPipelineExecutableStatisticsKHR>(
            vkGetDeviceProcAddr(device_, "vkGetPipelineExecutableStatisticsKHR"));
    if (placed.memoryMapPlaced && features_.sparseBuffers) {
        mapMemory2_ = reinterpret_cast<PFN_vkMapMemory2KHR>(vkGetDeviceProcAddr(device_, "vkMapMemory2KHR"));
        unmapMemory2_ = reinterpret_cast<PFN_vkUnmapMemory2KHR>(vkGetDeviceProcAddr(device_, "vkUnmapMemory2KHR"));
        // h | hostAddressBits() must be outside the host's half of the address space
        features_.hostMemoryAlloc = mapMemory2_ && unmapMemory2_ && (addressHigh_ & 0x8000u);
    }
    if (hostImport && features_.sparseBuffers) {
        VkPhysicalDeviceExternalMemoryHostPropertiesEXT hp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT};
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        p2.pNext = &hp;
        vkGetPhysicalDeviceProperties2(info_.handle, &p2);
        hostImportAlignment_ = std::max<VkDeviceSize>(hp.minImportedHostPointerAlignment, sparsePageSize_);
        getHostPointerProperties_ = reinterpret_cast<PFN_vkGetMemoryHostPointerPropertiesEXT>(
            vkGetDeviceProcAddr(device_, "vkGetMemoryHostPointerPropertiesEXT"));
        features_.hostMemoryImport = getHostPointerProperties_ != nullptr && (addressHigh_ & 0x8000u);
    }

    VULKANIZER_INFO("created Vulkan device '%s' (queue family %u, %s f32 fma, %s f64 fma)", info_.name(),
               info_.computeQueueFamily, features_.fusedFma ? "fused" : "unfused",
               features_.fusedFma64 ? "fused" : "unfused");
    return VK_SUCCESS;
}

Device::~Device() {
    if (!device_) return;
    vkDeviceWaitIdle(device_);
    for (auto& [serial, buf] : deferredFrees_) destroyBuffer(buf);
    destroyBuffer(readStaging_);
    for (auto& b : blocks_) destroyBuffer(b->buf);
    for (Batch& b : batches_) {
        if (b.fence) vkDestroyFence(device_, b.fence, nullptr);
        if (b.arena.buffer) {
            vkUnmapMemory(device_, b.arena.memory);
            vkDestroyBuffer(device_, b.arena.buffer, nullptr);
            vkFreeMemory(device_, b.arena.memory, nullptr);
        }
    }
    if (timestamps_) vkDestroyQueryPool(device_, timestamps_, nullptr);
    for (Batch& b : batches_)
        if (b.timing) vkDestroyQueryPool(device_, b.timing, nullptr);
    if (fence_) vkDestroyFence(device_, fence_, nullptr);
    if (pool_) vkDestroyCommandPool(device_, pool_, nullptr);
    vkDestroyDevice(device_, nullptr);
}

int32_t Device::findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags required) const {
    for (uint32_t i = 0; i < info_.memProps.memoryTypeCount; ++i) {
        VkMemoryPropertyFlags flags = info_.memProps.memoryTypes[i].propertyFlags;
        if ((typeBits & (1u << i)) && (flags & required) == required && !(flags & kAvoidedMemoryFlags))
            return static_cast<int32_t>(i);
    }
    return -1;
}

VkResult Device::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                              std::initializer_list<VkMemoryPropertyFlags> preferences, Buffer& out) {
    if (info_.maxMemoryAllocationSize && size > info_.maxMemoryAllocationSize) {
        VULKANIZER_WARN("allocation of %llu bytes exceeds maxMemoryAllocationSize (%llu)",
                   static_cast<unsigned long long>(size),
                   static_cast<unsigned long long>(info_.maxMemoryAllocationSize));
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }

    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = size;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkResult r = vkCreateBuffer(device_, &bci, nullptr, &buffer);
    if (r != VK_SUCCESS) return r;

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device_, buffer, &req);

    const bool wantAddress = usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    VkMemoryDedicatedAllocateInfo dedicated{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
    dedicated.buffer = buffer;
    VkMemoryAllocateFlagsInfo flagsInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    flagsInfo.pNext = &dedicated;

    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkMemoryPropertyFlags chosen = 0;
    r = VK_ERROR_OUT_OF_DEVICE_MEMORY;
    // Try each preference in order; a heap can be full even if the type exists (e.g. small BAR).
    for (VkMemoryPropertyFlags pref : preferences) {
        int32_t type = findMemoryType(req.memoryTypeBits, pref);
        if (type < 0) continue;
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.pNext = wantAddress ? static_cast<void*>(&flagsInfo) : static_cast<void*>(&dedicated);
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = static_cast<uint32_t>(type);
        r = vkAllocateMemory(device_, &mai, nullptr, &memory);
        if (r == VK_SUCCESS) {
            chosen = info_.memProps.memoryTypes[type].propertyFlags;
            break;
        }
    }
    if (r != VK_SUCCESS) {
        vkDestroyBuffer(device_, buffer, nullptr);
        return r;
    }

    if ((r = vkBindBufferMemory(device_, buffer, memory, 0)) != VK_SUCCESS) {
        vkFreeMemory(device_, memory, nullptr);
        vkDestroyBuffer(device_, buffer, nullptr);
        return r;
    }

    out = Buffer{};
    out.buffer = buffer;
    out.memory = memory;
    out.size = size;
    constexpr VkMemoryPropertyFlags kMappable = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if ((chosen & kMappable) == kMappable) {
        if (vkMapMemory(device_, memory, 0, VK_WHOLE_SIZE, 0, &out.mapped) != VK_SUCCESS) out.mapped = nullptr;
        out.hostCached = (chosen & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) != 0;
    }
    if (wantAddress) {
        VkBufferDeviceAddressInfo bai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        bai.buffer = buffer;
        out.address = vkGetBufferDeviceAddress(device_, &bai) & kAddressMask;
    }
    return VK_SUCCESS;
}

VkResult Device::allocateBuffer(VkDeviceSize size, Buffer& out) {
    if (subAlloc_ && size <= kSubAllocLimit && subAllocate(size, out) == VK_SUCCESS) return VK_SUCCESS;
    return allocateDedicated(size, out);
}

VkResult Device::allocateDedicated(VkDeviceSize size, Buffer& out) {
    // Prefer memory that is both device-local and host-mappable (UMA / ReBAR): copies become memcpy.
    auto attempt = [&] {
        return createBuffer(size, kDeviceBufferUsage,
                            {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0},
                            out);
    };
    reapDeferredFrees();
    VkResult r = attempt();
    if (r == VK_ERROR_OUT_OF_DEVICE_MEMORY || r == VK_ERROR_OUT_OF_HOST_MEMORY) {
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        if (!deferredFrees_.empty() && finish() == VK_SUCCESS) {  // memory freed asynchronously comes back
            reapDeferredFrees();
            r = attempt();
        }
    }
    return r;
}

void Device::destroyBuffer(Buffer& buf) {
    if (buf.placedMapping) {
        // Back to a reserved range first: nothing else may be mapped there before the munmap.
        VkMemoryUnmapInfoKHR ui{VK_STRUCTURE_TYPE_MEMORY_UNMAP_INFO_KHR};
        ui.flags = VK_MEMORY_UNMAP_RESERVE_BIT_EXT;
        ui.memory = buf.memory;
        unmapMemory2_(device_, &ui);
        munmap(buf.mapped, size_t(buf.size));
    } else if (buf.mapped && !buf.hostAddressed) {
        vkUnmapMemory(device_, buf.memory);
    }
    if (buf.buffer) vkDestroyBuffer(device_, buf.buffer, nullptr);
    if (buf.memory) vkFreeMemory(device_, buf.memory, nullptr);
    buf = Buffer{};
}

void Device::freeBuffer(Buffer& buf) {
    if (buf.subAllocated) {
        releaseSubAllocation(buf);
        return;
    }
    {
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        if (hasRecordedWork()) finish();  // recorded kernels may still use it
    }
    destroyBuffer(buf);
}

void Device::freeBufferAfter(Buffer& buf, uint64_t serial) {
    if (buf.subAllocated) {  // queue order protects it, see freeBuffer
        releaseSubAllocation(buf);
        return;
    }
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    deferredFrees_.emplace_back(serial, buf);
    buf = Buffer{};
    reapDeferredFrees();
}

VkResult Device::subAllocate(VkDeviceSize size, Buffer& out) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    const VkDeviceSize need = (std::max<VkDeviceSize>(size, 1) + kSubAllocAlign - 1) / kSubAllocAlign * kSubAllocAlign;
    Block* block = nullptr;
    VkDeviceSize at = 0;
    for (auto& b : blocks_) {  // first fit
        for (auto& [offset, n] : b->free)
            if (n >= need) {
                block = b.get();
                at = offset;
                break;
            }
        if (block) break;
    }
    if (!block) {
        auto b = std::make_unique<Block>();
        if (VkResult r = allocateDedicated(kBlockSize, b->buf)) return r;
        b->free[0] = kBlockSize;
        block = b.get();
        blocks_.push_back(std::move(b));
    }
    auto it = block->free.find(at);
    const VkDeviceSize rest = it->second - need;
    block->free.erase(it);
    if (rest) block->free[at + need] = rest;
    block->used += need;
    out = block->buf;
    out.size = size;
    out.offset = at;
    out.address += at;
    if (out.mapped) out.mapped = static_cast<char*>(out.mapped) + at;
    out.subAllocated = true;
    return VK_SUCCESS;
}

void Device::releaseSubAllocation(Buffer& buf) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto b = std::find_if(blocks_.begin(), blocks_.end(), [&](auto& blk) { return blk->buf.buffer == buf.buffer; });
    if (b == blocks_.end()) return;
    Block& block = **b;
    VkDeviceSize at = buf.offset;
    VkDeviceSize n = (std::max<VkDeviceSize>(buf.size, 1) + kSubAllocAlign - 1) / kSubAllocAlign * kSubAllocAlign;
    block.used -= n;
    auto next = block.free.lower_bound(at);
    if (next != block.free.end() && next->first == at + n) {  // merge with the following range
        n += next->second;
        next = block.free.erase(next);
    }
    if (next != block.free.begin()) {
        auto prev = std::prev(next);
        if (prev->first + prev->second == at) {  // and the preceding one
            at = prev->first;
            n += prev->second;
            block.free.erase(prev);
        }
    }
    block.free[at] = n;
    buf = Buffer{};
    // Keep one empty block for the next allocations; release the others once unused.
    if (block.used == 0 && std::count_if(blocks_.begin(), blocks_.end(), [](auto& blk) { return blk->used == 0; }) > 1) {
        freeBufferAfter(block.buf, currentSerial());
        blocks_.erase(b);
    }
}

void Device::reapDeferredFrees() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    size_t kept = 0;
    for (auto& entry : deferredFrees_) {
        if (isComplete(entry.first)) destroyBuffer(entry.second);
        else deferredFrees_[kept++] = entry;
    }
    deferredFrees_.resize(kept);
}

// ---- virtual memory --------------------------------------------------------------------------

// Page size and memory types of sparse buffers, from a small one.
void Device::probeSparse() {
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.flags = VK_BUFFER_CREATE_SPARSE_BINDING_BIT | VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT;
    bci.size = 1 << 20;
    bci.usage = kDeviceBufferUsage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer buffer = VK_NULL_HANDLE;
    if (vkCreateBuffer(device_, &bci, nullptr, &buffer) != VK_SUCCESS) {
        features_.sparseBuffers = false;
        return;
    }
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device_, buffer, &req);
    vkDestroyBuffer(device_, buffer, nullptr);
    sparsePageSize_ = req.alignment;
    sparseMemoryTypes_ = req.memoryTypeBits;
    // Whole pages, and below 4 GB where drivers' 32-bit size fields would wrap.
    maxSparseBufferSize_ = std::min<VkDeviceSize>(info_.maxBufferSize, VkDeviceSize(4) << 30) / req.alignment * req.alignment;
}

VkResult Device::createSparseBuffer(VkDeviceSize size, uint64_t opaqueAddress, Buffer& out, uint64_t& opaqueOut) {
    if (!features_.sparseBuffers || size > maxSparseBufferSize_) return VK_ERROR_FEATURE_NOT_PRESENT;
    if (opaqueAddress && !features_.addressCaptureReplay) return VK_ERROR_FEATURE_NOT_PRESENT;
    VkBufferOpaqueCaptureAddressCreateInfo oci{VK_STRUCTURE_TYPE_BUFFER_OPAQUE_CAPTURE_ADDRESS_CREATE_INFO};
    oci.opaqueCaptureAddress = opaqueAddress;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.pNext = opaqueAddress ? &oci : nullptr;
    bci.flags = VK_BUFFER_CREATE_SPARSE_BINDING_BIT | VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT;
    if (features_.addressCaptureReplay) bci.flags |= VK_BUFFER_CREATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT;
    bci.size = size;
    bci.usage = kDeviceBufferUsage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    out = Buffer{};
    if (VkResult r = vkCreateBuffer(device_, &bci, nullptr, &out.buffer)) return r;
    out.size = size;
    VkBufferDeviceAddressInfo bai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    bai.buffer = out.buffer;
    out.address = vkGetBufferDeviceAddress(device_, &bai) & kAddressMask;
    opaqueOut = features_.addressCaptureReplay ? vkGetBufferOpaqueCaptureAddress(device_, &bai) : 0;
    return VK_SUCCESS;
}

VkResult Device::createAtHostAddress(uint64_t h, VkDeviceSize size, bool imported, Buffer& out) {
    out = Buffer{};
    const uint64_t wanted = h | hostAddressBits();
    VkExternalMemoryBufferCreateInfo ext{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    ext.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    VkBufferOpaqueCaptureAddressCreateInfo oci{VK_STRUCTURE_TYPE_BUFFER_OPAQUE_CAPTURE_ADDRESS_CREATE_INFO};
    oci.opaqueCaptureAddress = wanted;
    oci.pNext = imported ? &ext : nullptr;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.pNext = &oci;
    bci.flags = VK_BUFFER_CREATE_SPARSE_BINDING_BIT | VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT |
                VK_BUFFER_CREATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT;
    bci.size = size;
    bci.usage = kDeviceBufferUsage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (VkResult r = vkCreateBuffer(device_, &bci, nullptr, &out.buffer)) return r;
    VkBufferDeviceAddressInfo bai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    bai.buffer = out.buffer;
    if (vkGetBufferDeviceAddress(device_, &bai) != wanted) {
        destroyBuffer(out);
        return VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS;  // the range is taken
    }
    out.size = size;
    return VK_SUCCESS;
}

VkResult Device::importHostMemory(void* host, VkDeviceSize size, Buffer& out) {
    out = Buffer{};
    if (!features_.hostMemoryImport) return VK_ERROR_FEATURE_NOT_PRESENT;
    const uint64_t h = reinterpret_cast<uint64_t>(host);
    if (h % hostImportAlignment_ || size % hostImportAlignment_ || !size || h >= (uint64_t(1) << 47) ||
        size > maxSparseBufferSize_)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (VkResult r = createAtHostAddress(h, size, true, out)) return r;
    VkMemoryHostPointerPropertiesEXT hpp{VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device_, out.buffer, &req);
    VkResult r = getHostPointerProperties_(device_, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, host, &hpp);
    const int32_t type = r == VK_SUCCESS ? findMemoryType(hpp.memoryTypeBits & req.memoryTypeBits, 0) : -1;
    if (r == VK_SUCCESS && type < 0) r = VK_ERROR_FORMAT_NOT_SUPPORTED;
    if (r == VK_SUCCESS) {
        VkImportMemoryHostPointerInfoEXT imp{VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT};
        imp.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
        imp.pHostPointer = host;
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.pNext = &imp;
        mai.allocationSize = size;
        mai.memoryTypeIndex = uint32_t(type);
        r = vkAllocateMemory(device_, &mai, nullptr, &out.memory);
    }
    if (r == VK_SUCCESS) r = bindSparse({{&out, 0, size, out.memory, 0}});
    if (r != VK_SUCCESS) {
        destroyBuffer(out);
        return r;
    }
    out.address = h;
    out.mapped = host;
    out.hostAddressed = true;
    out.hostCached = true;
    return VK_SUCCESS;
}

VkResult Device::allocateHostMemory(VkDeviceSize size, Buffer& out) {
    out = Buffer{};
    if (!features_.hostMemoryAlloc) return VK_ERROR_FEATURE_NOT_PRESENT;
    // RADV places memory of 2 MiB or more only at 2 MiB aligned addresses.
    constexpr VkDeviceSize align = 2 << 20;
    const VkDeviceSize page = std::max<VkDeviceSize>(sparsePageSize_, 64 << 10);
    size = (size + page - 1) / page * page;
    if (!size || size > maxSparseBufferSize_) return VK_ERROR_OUT_OF_HOST_MEMORY;
    // An aligned range of the address space, reserved until the memory is mapped over it.
    char* reserved = static_cast<char*>(
        mmap(nullptr, size + align, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0));
    if (reserved == MAP_FAILED) return VK_ERROR_OUT_OF_HOST_MEMORY;
    char* host = reinterpret_cast<char*>((uintptr_t(reserved) + align - 1) / align * align);
    if (host > reserved) munmap(reserved, size_t(host - reserved));
    if (host + size < reserved + size + align) munmap(host + size, size_t(reserved + size + align - (host + size)));

    // Preferably a plain buffer on memory placed at the address (the memory's capture/replay
    // address), otherwise a sparse buffer placed there with the memory bound to it: on RADV,
    // system memory bound to a sparse buffer makes every submission slower (~4 ms per 256 MB).
    const uint64_t h = reinterpret_cast<uint64_t>(host);
    VkResult r = VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS;
    for (bool sparse : {false, true}) {
        r = placeHostMemory(h, size, sparse, out);
        VULKANIZER_DEBUG("host memory at 0x%llx (%llu bytes), %s buffer: %d", (unsigned long long)h, (unsigned long long)size, sparse ? "sparse" : "plain", r);
        if (r != VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS) break;
    }
    if (r == VK_SUCCESS) {
        VkMemoryMapPlacedInfoEXT placed{VK_STRUCTURE_TYPE_MEMORY_MAP_PLACED_INFO_EXT};
        placed.pPlacedAddress = host;
        VkMemoryMapInfoKHR mi{VK_STRUCTURE_TYPE_MEMORY_MAP_INFO_KHR};
        mi.pNext = &placed;
        mi.flags = VK_MEMORY_MAP_PLACED_BIT_EXT;
        mi.memory = out.memory;
        mi.size = VK_WHOLE_SIZE;
        void* mapped = nullptr;
        r = mapMemory2_(device_, &mi, &mapped);
        if (r == VK_SUCCESS && mapped != host) r = VK_ERROR_MEMORY_MAP_FAILED;  // not expected
        if (r == VK_SUCCESS) {
            out.mapped = host;
            out.placedMapping = true;
        } else {
            destroyBuffer(out);
        }
    }
    if (r != VK_SUCCESS) {
        munmap(host, size);
        return r;
    }
    out.address = h;
    out.hostAddressed = true;
    return VK_SUCCESS;
}

VkResult Device::placeHostMemory(uint64_t h, VkDeviceSize size, bool sparse, Buffer& out) {
    const uint64_t wanted = h | hostAddressBits();
    VkResult r = VK_SUCCESS;
    if (sparse) {
        r = createAtHostAddress(h, size, false, out);
    } else {
        out = Buffer{};
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.flags = VK_BUFFER_CREATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT;
        bci.size = size;
        bci.usage = kDeviceBufferUsage;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        r = vkCreateBuffer(device_, &bci, nullptr, &out.buffer);
        out.size = size;
    }
    if (r != VK_SUCCESS) return r;
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device_, out.buffer, &req);
    constexpr VkMemoryPropertyFlags kHost = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    int32_t type = findMemoryType(req.memoryTypeBits, kHost | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
    if (type < 0) type = findMemoryType(req.memoryTypeBits, kHost);
    VkMemoryOpaqueCaptureAddressAllocateInfo cai{VK_STRUCTURE_TYPE_MEMORY_OPAQUE_CAPTURE_ADDRESS_ALLOCATE_INFO};
    cai.opaqueCaptureAddress = wanted;
    VkMemoryAllocateFlagsInfo flagsInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    if (!sparse) {
        flagsInfo.flags |= VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT;
        flagsInfo.pNext = &cai;
    }
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.pNext = &flagsInfo;
    mai.allocationSize = size;
    mai.memoryTypeIndex = uint32_t(type);
    r = type < 0 ? VK_ERROR_OUT_OF_HOST_MEMORY : vkAllocateMemory(device_, &mai, nullptr, &out.memory);
    if (r == VK_SUCCESS) {
        out.hostCached = info_.memProps.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        if (sparse) {
            r = bindSparse({{&out, 0, size, out.memory, 0}});
        } else if ((r = vkBindBufferMemory(device_, out.buffer, out.memory, 0)) == VK_SUCCESS) {
            VkBufferDeviceAddressInfo bai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
            bai.buffer = out.buffer;
            if (vkGetBufferDeviceAddress(device_, &bai) != wanted) r = VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS;
        }
    }
    if (r != VK_SUCCESS) destroyBuffer(out);
    return r;
}

VkResult Device::allocateMemory(VkDeviceSize size, VkDeviceMemory& out) {
    VkMemoryAllocateFlagsInfo flagsInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkResult r = VK_ERROR_OUT_OF_DEVICE_MEMORY;
    for (VkMemoryPropertyFlags pref : {VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT), VkMemoryPropertyFlags(0)}) {
        const int32_t type = findMemoryType(sparseMemoryTypes_, pref);
        if (type < 0) continue;
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.pNext = &flagsInfo;
        mai.allocationSize = size;
        mai.memoryTypeIndex = uint32_t(type);
        if ((r = vkAllocateMemory(device_, &mai, nullptr, &out)) == VK_SUCCESS) return r;
    }
    return r;
}

void Device::freeMemory(VkDeviceMemory memory) {
    if (memory) vkFreeMemory(device_, memory, nullptr);
}

VkResult Device::bindSparse(const std::vector<SparseBind>& binds) {
    if (binds.empty()) return VK_SUCCESS;
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    // Pages being unbound may still be used by recorded work. Newly bound pages are not
    // (CUDA forbids touching unmapped memory), and the work recorded later is submitted after
    // the binding completed below.
    for (const SparseBind& b : binds)
        if (!b.memory) {
            if (VkResult r = finish()) return r;
            break;
        }
    std::vector<VkSparseMemoryBind> pages(binds.size());
    std::vector<VkSparseBufferMemoryBindInfo> infos(binds.size());
    for (size_t i = 0; i < binds.size(); ++i) {
        pages[i] = {binds[i].offset, binds[i].size, binds[i].memory, binds[i].memoryOffset, 0};
        infos[i] = {binds[i].buffer->buffer, 1, &pages[i]};
    }
    VkBindSparseInfo bsi{VK_STRUCTURE_TYPE_BIND_SPARSE_INFO};
    bsi.bufferBindCount = uint32_t(infos.size());
    bsi.pBufferBinds = infos.data();
    if (VkResult r = vkResetFences(device_, 1, &fence_)) return r;
    if (VkResult r = vkQueueBindSparse(queue_, 1, &bsi, fence_)) return r;
    return vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
}

bool Device::probeFusedFma() {
    Buffer out;
    if (allocateBuffer(4, out) != VK_SUCCESS) return false;
    ComputePipeline pipeline;
    const uint32_t spec[5] = {1, 1, 1, 0, 1};
    bool fused = false;
    struct {
        VkDeviceAddress out;
        float a, c;
    } params{out.address, 1.0f + 1.0f / 4096.0f, -(1.0f + 1.0f / 2048.0f)};
    if (createComputePipeline(kFmaProbeSpirv, sizeof kFmaProbeSpirv / 4, "main", sizeof params, spec, pipeline) ==
        VK_SUCCESS) {
        float r = 0;
        if (dispatch(pipeline, 1, 1, 1, &params, sizeof params) == VK_SUCCESS && read(out, 0, &r, 4) == VK_SUCCESS)
            fused = r == 1.0f / 16777216.0f;
        destroyComputePipeline(pipeline);
    }
    freeBuffer(out);
    return fused;
}

bool Device::probeFusedFma64() {
    Buffer out;
    if (allocateBuffer(8, out) != VK_SUCCESS) return false;
    ComputePipeline pipeline;
    const uint32_t spec[5] = {1, 1, 1, 0, 1};
    bool fused = false;
    struct {
        VkDeviceAddress out;
        double a, c;
    } params{out.address, 1.0 + 1.0 / 134217728.0, -(1.0 + 1.0 / 67108864.0)};
    if (createComputePipeline(kFmaProbeF64Spirv, sizeof kFmaProbeF64Spirv / 4, "main", sizeof params, spec,
                              pipeline) == VK_SUCCESS) {
        double r = 0;
        if (dispatch(pipeline, 1, 1, 1, &params, sizeof params) == VK_SUCCESS && read(out, 0, &r, 8) == VK_SUCCESS)
            fused = r == 1.0 / 18014398509481984.0;  // 2^-54
        destroyComputePipeline(pipeline);
    }
    freeBuffer(out);
    return fused;
}

bool Device::hasRecordedWork() const {
    if (open_ >= 0) return true;
    for (const Batch& b : batches_)
        if (b.pending) return true;
    return false;
}

static void fullBarrier(VkCommandBuffer cb) {
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0,
                         nullptr, 0, nullptr);
}

static void hostBarrier(VkCommandBuffer cb) {
    // Make every device write visible to the host once the fence signals.
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT | VK_ACCESS_HOST_WRITE_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, nullptr,
                         0, nullptr);
}

VkResult Device::beginBatch() {
    Batch& b = batches_[nextBatch_];
    if (b.pending) {
        if (VkResult r = waitBatch(b)) return r;
    }
    if (VkResult r = vkResetCommandBuffer(b.cmd, 0)) return r;
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (VkResult r = vkBeginCommandBuffer(b.cmd, &bi)) return r;
    b.serial = nextSerial_++;
    b.arenaUsed = 0;
    b.kernels.clear();
    b.estimateNs = 0;
    if (timing_) vkCmdResetQueryPool(b.cmd, b.timing, 0, 2 * kMaxCommandsPerBatch);
    open_ = int32_t(nextBatch_);
    nextBatch_ = (nextBatch_ + 1) % kBatches;
    commands_ = 0;
    return VK_SUCCESS;
}

VkResult Device::flushLocked() {
    if (open_ < 0) return VK_SUCCESS;
    Batch& b = batches_[open_];
    open_ = -1;
    hostBarrier(b.cmd);
    if (VkResult r = vkEndCommandBuffer(b.cmd)) return r;
    if (VkResult r = vkResetFences(device_, 1, &b.fence)) return r;
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &b.cmd;
    if (VkResult r = vkQueueSubmit(queue_, 1, &si, b.fence)) return r;
    b.pending = true;
    return VK_SUCCESS;
}

void Device::collectTimings(Batch& b) {
    if (b.timed.empty()) return;
    std::vector<uint64_t> ticks(2 * b.timed.size());
    if (vkGetQueryPoolResults(device_, b.timing, 0, uint32_t(ticks.size()), ticks.size() * 8, ticks.data(), 8,
                              VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
        for (size_t i = 0; i < b.timed.size(); ++i) {
            const Timed& t = b.timed[i];
            const double ns = double(int64_t(ticks[2 * i + 1] - ticks[2 * i])) * timestampPeriodNs_;
            double& rate = nsPerGroup_[t.pipeline];
            rate = std::max(rate, ns / double(std::max(t.groups, kRateGroups)));
        }
        if (profiling_) {
            std::lock_guard<std::mutex> lock(profileMutex());
            for (size_t i = 0; i < b.timed.size(); ++i) {
                ProfileEntry& e = profile()[displayName(b.timed[i].label)];
                ++e.count;
                e.ns += double(ticks[2 * i + 1] - ticks[2 * i]) * timestampPeriodNs_;
                if (i) profileSpan().gapNs += double(int64_t(ticks[2 * i] - ticks[2 * i - 1])) * timestampPeriodNs_;
            }
            ++profileSpan().batches;
            profileSpan().busyNs += double(ticks.back() - ticks.front()) * timestampPeriodNs_;
        }
    }
    b.timed.clear();
}

VkResult Device::waitBatch(Batch& b) {
    const VkResult r = vkWaitForFences(device_, 1, &b.fence, VK_TRUE, UINT64_MAX);
    if (r == VK_ERROR_DEVICE_LOST && !deviceLost_) {
        // Usually a kernel that hung (the kernel driver resets the GPU after its timeout). The
        // culprit is the oldest unfinished batch, not necessarily the one waited for.
        deviceLost_ = true;
        const Batch* lost = &b;
        for (const Batch& o : batches_)
            if (o.pending && o.serial < lost->serial && vkGetFenceStatus(device_, o.fence) != VK_SUCCESS) lost = &o;
        std::string names;
        if (!lost->timed.empty()) {  // every dispatch, with its size
            for (const Timed& t : lost->timed) {
                char size[96];
                std::snprintf(size, sizeof size, "\n  %8llu groups, estimated %8.3f ms: ", (unsigned long long)t.groups,
                              t.estimateNs * 1e-6);
                names += size + (t.label ? displayName(t.label) : std::string("(internal)"));
            }
        } else {
            for (const char* k : lost->kernels) {
                const std::string n = k ? displayName(k) : std::string("(internal)");
                if (names.find(n) == std::string::npos) names += "\n  " + n;
            }
        }
        VULKANIZER_ERROR("Vulkan device lost; the oldest unfinished batch ran these kernels:%s\n"
                         "(if the GPU was reset for a timeout, a lower VULKANIZER_BATCH_MS may help)", names.c_str());
    }
    if (r != VK_SUCCESS) return r;
    b.pending = false;
    collectTimings(b);
    completedSerial_ = std::max(completedSerial_, b.serial);
    return VK_SUCCESS;
}

VkResult Device::record(const std::function<void(VkCommandBuffer)>& commands) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (open_ < 0) {
        if (VkResult r = beginBatch()) return r;
    }
    VkCommandBuffer cb = batches_[open_].cmd;
    fullBarrier(cb);  // CUDA orders all work; finer-grained dependencies can come later
    commands(cb);
    if (++commands_ >= kMaxCommandsPerBatch) return flushLocked();
    return VK_SUCCESS;
}

uint64_t Device::currentSerial() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    return open_ >= 0 ? batches_[open_].serial : nextSerial_ - 1;
}

VkResult Device::flush() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    return flushLocked();
}

VkResult Device::wait(uint64_t serial) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (serial <= completedSerial_) return VK_SUCCESS;
    if (open_ >= 0 && batches_[open_].serial <= serial) {
        if (VkResult r = flushLocked()) return r;
    }
    for (Batch& b : batches_)
        if (b.pending && b.serial <= serial) {
            if (VkResult r = waitBatch(b)) return r;
        }
    return VK_SUCCESS;
}

VkResult Device::finish() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (VkResult r = flushLocked()) return r;
    for (Batch& b : batches_)
        if (b.pending) {
            if (VkResult r = waitBatch(b)) return r;
        }
    return VK_SUCCESS;
}

bool Device::idleLocked() {
    if (open_ >= 0) return false;
    for (Batch& b : batches_) {
        if (!b.pending) continue;
        if (vkGetFenceStatus(device_, b.fence) != VK_SUCCESS) return false;
        b.pending = false;
        collectTimings(b);
        completedSerial_ = std::max(completedSerial_, b.serial);
    }
    return true;
}

bool Device::isComplete(uint64_t serial) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (serial <= completedSerial_) return true;
    if (open_ >= 0 && batches_[open_].serial <= serial) return false;
    for (Batch& b : batches_) {
        if (!b.pending || b.serial > serial) continue;
        if (vkGetFenceStatus(device_, b.fence) != VK_SUCCESS) return false;
        b.pending = false;
        collectTimings(b);
        completedSerial_ = std::max(completedSerial_, b.serial);
    }
    return true;
}

VkResult Device::upload(const void* data, VkDeviceSize size, VkDeviceAddress& address) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (open_ < 0) {
        if (VkResult r = beginBatch()) return r;
    }
    VkDeviceSize at = (batches_[open_].arenaUsed + 15) & ~VkDeviceSize(15);
    if (!batches_[open_].arena.buffer || at + size > batches_[open_].arena.size) {
        if (commands_ > 0) {  // the arena is in use: continue in a new batch
            if (VkResult r = flushLocked()) return r;
            if (VkResult r = beginBatch()) return r;
        }
        Batch& b = batches_[open_];
        if (!b.arena.buffer || size > b.arena.size) {
            if (b.arena.buffer) {
                vkUnmapMemory(device_, b.arena.memory);
                vkDestroyBuffer(device_, b.arena.buffer, nullptr);
                vkFreeMemory(device_, b.arena.memory, nullptr);
                b.arena = Buffer{};
            }
            VkDeviceSize cap = 64 * 1024;
            while (cap < size) cap *= 2;
            constexpr VkMemoryPropertyFlags kHost = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            if (VkResult r = createBuffer(cap, kDeviceBufferUsage, {kHost | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, kHost}, b.arena))
                return r;
            if (!b.arena.mapped) return VK_ERROR_MEMORY_MAP_FAILED;
        }
        at = 0;
    }
    Batch& b = batches_[open_];
    std::memcpy(static_cast<char*>(b.arena.mapped) + at, data, size);
    b.arenaUsed = at + size;
    address = b.arena.address + at;
    return VK_SUCCESS;
}

int32_t Device::allocateTimestamp() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (freeTimestamps_.empty()) return -1;
    const int32_t slot = freeTimestamps_.back();
    freeTimestamps_.pop_back();
    timestampSerial_[slot] = 0;
    return slot;
}

void Device::freeTimestamp(int32_t slot) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (slot < 0) return;
    if (timestampSerial_[slot]) wait(timestampSerial_[slot]);
    freeTimestamps_.push_back(slot);
}

VkResult Device::recordTimestamp(int32_t slot, uint64_t& serial) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (timestampSerial_[slot] && !isComplete(timestampSerial_[slot])) {
        if (VkResult r = wait(timestampSerial_[slot])) return r;
    }
    vkResetQueryPool(device_, timestamps_, uint32_t(slot), 1);
    VkResult r = record([&](VkCommandBuffer cb) {
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestamps_, uint32_t(slot));
    });
    serial = timestampSerial_[slot] = currentSerial();
    return r;
}

bool Device::readTimestampNs(int32_t slot, double& ns) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!timestampSerial_[slot] || !isComplete(timestampSerial_[slot])) return false;
    uint64_t ticks = 0;
    if (vkGetQueryPoolResults(device_, timestamps_, uint32_t(slot), 1, sizeof ticks, &ticks, sizeof ticks,
                              VK_QUERY_RESULT_64_BIT) != VK_SUCCESS)
        return false;
    ns = double(ticks) * timestampPeriodNs_;
    return true;
}

VkResult Device::submit(const std::function<void(VkCommandBuffer)>& record) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (VkResult r = finish()) return r;
    VkResult r = vkResetCommandBuffer(cmd_, 0);
    if (r != VK_SUCCESS) return r;

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if ((r = vkBeginCommandBuffer(cmd_, &bi)) != VK_SUCCESS) return r;
    record(cmd_);
    hostBarrier(cmd_);
    if ((r = vkEndCommandBuffer(cmd_)) != VK_SUCCESS) return r;

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd_;
    if ((r = vkResetFences(device_, 1, &fence_)) != VK_SUCCESS) return r;
    if ((r = vkQueueSubmit(queue_, 1, &si, fence_)) != VK_SUCCESS) return r;
    return vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
}

VkResult Device::write(const Buffer& dst, VkDeviceSize offset, const void* src, VkDeviceSize size) {
    if (size == 0) return VK_SUCCESS;
    if (dst.mapped) {
        // Nothing queued or running: a direct host write keeps the order and costs no
        // GPU round trip (inputs uploaded right before a synchronize, e.g. per LLM token).
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        if (idleLocked()) {
            std::memcpy(static_cast<char*>(dst.mapped) + offset, src, size);
            return VK_SUCCESS;
        }
    }
    if (size <= kAsyncWriteLimit) {
        // Staged in the batch's upload arena (the caller's memory is free again on return) and
        // copied in queue order, without waiting for the GPU.
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        VkDeviceAddress staged;
        if (VkResult r = upload(src, size, staged)) return r;
        const Batch& b = batches_[open_];
        const VkDeviceSize from = staged - b.arena.address;
        return record([&](VkCommandBuffer cb) {
            VkBufferCopy region{from, dst.offset + offset, size};
            vkCmdCopyBuffer(cb, b.arena.buffer, dst.buffer, 1, &region);
        });
    }
    if (VkResult r = finish()) return r;  // host access: recorded work comes first
    if (dst.mapped) {
        std::memcpy(static_cast<char*>(dst.mapped) + offset, src, size);
        return VK_SUCCESS;
    }
    Buffer staging;
    VkResult r = createBuffer(size, kStagingUsage,
                              {VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT}, staging);
    if (r != VK_SUCCESS) return r;
    std::memcpy(staging.mapped, src, size);
    r = submit([&](VkCommandBuffer cb) {
        VkBufferCopy region{0, dst.offset + offset, size};
        vkCmdCopyBuffer(cb, staging.buffer, dst.buffer, 1, &region);
    });
    freeBuffer(staging);
    return r;
}

VkResult Device::read(const Buffer& src, VkDeviceSize offset, void* dst, VkDeviceSize size) {
    if (size == 0) return VK_SUCCESS;
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (VkResult r = finish()) return r;  // host access: recorded work comes first
    if (src.mapped && (src.hostCached || size <= kDirectReadLimit)) {
        std::memcpy(dst, static_cast<const char*>(src.mapped) + offset, size);
        return VK_SUCCESS;
    }
    if (!readStaging_.buffer) {
        constexpr VkMemoryPropertyFlags kHost = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        if (VkResult r = createBuffer(2 * kReadChunk, kStagingUsage, {kHost | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, kHost},
                                      readStaging_))
            return r;
        if (!readStaging_.mapped) {
            destroyBuffer(readStaging_);
            return VK_ERROR_MEMORY_MAP_FAILED;
        }
    }
    // The GPU copies chunk i into one half of the staging buffer while the host copies chunk i - 1
    // out of the other.
    uint64_t prevSerial = 0;
    VkDeviceSize prevAt = 0, prevSize = 0;
    for (VkDeviceSize at = 0, i = 0;; at += prevSize, ++i) {
        const VkDeviceSize n = std::min(kReadChunk, size - at);
        if (n) {
            const VkDeviceSize half = (i % 2) * kReadChunk;
            if (VkResult r = record([&](VkCommandBuffer cb) {
                    VkBufferCopy region{src.offset + offset + at, half, n};
                    vkCmdCopyBuffer(cb, src.buffer, readStaging_.buffer, 1, &region);
                }))
                return r;
            if (VkResult r = flushLocked()) return r;
        }
        if (prevSize) {
            if (VkResult r = wait(prevSerial)) return r;
            std::memcpy(static_cast<char*>(dst) + prevAt,
                        static_cast<const char*>(readStaging_.mapped) + ((i - 1) % 2) * kReadChunk, prevSize);
        }
        if (!n) return VK_SUCCESS;
        prevSerial = nextSerial_ - 1;
        prevAt = at;
        prevSize = n;
    }
}

VkResult Device::copy(const Buffer& src, VkDeviceSize srcOffset, const Buffer& dst, VkDeviceSize dstOffset,
                      VkDeviceSize size) {
    if (size == 0) return VK_SUCCESS;
    return record([&](VkCommandBuffer cb) {  // device to device: no host synchronization
        VkBufferCopy region{src.offset + srcOffset, dst.offset + dstOffset, size};
        vkCmdCopyBuffer(cb, src.buffer, dst.buffer, 1, &region);
    });
}

VkResult Device::fill(const Buffer& dst, VkDeviceSize offset, VkDeviceSize size, uint32_t pattern) {
    if (size == 0) return VK_SUCCESS;
    return record([&](VkCommandBuffer cb) { vkCmdFillBuffer(cb, dst.buffer, dst.offset + offset, size, pattern); });
}

VkResult Device::createComputePipeline(const uint32_t* code, size_t wordCount, const char* entryPoint,
                                       uint32_t pushConstantSize, const uint32_t spec[5],
                                       ComputePipeline& out) {
    out = ComputePipeline{};
    pushConstantSize = (pushConstantSize + 3u) & ~3u;
    if (pushConstantSize > info_.props.limits.maxPushConstantsSize) return VK_ERROR_FEATURE_NOT_PRESENT;

    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = wordCount * sizeof(uint32_t);
    smi.pCode = code;
    VkResult r = vkCreateShaderModule(device_, &smi, nullptr, &out.module);
    if (r != VK_SUCCESS) return r;

    VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, pushConstantSize};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.pushConstantRangeCount = pushConstantSize ? 1 : 0;
    pli.pPushConstantRanges = &range;
    if ((r = vkCreatePipelineLayout(device_, &pli, nullptr, &out.layout)) != VK_SUCCESS) {
        destroyComputePipeline(out);
        return r;
    }

    // 0..4 from the caller, 5: the address bits kernels set (hostAddressBits())
    const VkSpecializationMapEntry entries[6] = {{0, 0, 4}, {1, 4, 4}, {2, 8, 4}, {3, 12, 4}, {4, 16, 4}, {5, 20, 4}};
    const uint32_t values[6] = {spec[0], spec[1], spec[2], spec[3], spec[4], addressHigh_};
    VkSpecializationInfo specInfo{6, entries, sizeof values, values};

    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = out.module;
    cpi.stage.pName = entryPoint;
    cpi.stage.pSpecializationInfo = &specInfo;
    cpi.layout = out.layout;
    cpi.flags = VK_PIPELINE_CREATE_DISPATCH_BASE_BIT;  // grids beyond the dispatch limit are split
    if (getPipelineStats_) cpi.flags |= VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
    if ((r = vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &cpi, nullptr, &out.pipeline)) != VK_SUCCESS) {
        destroyComputePipeline(out);
        return r;
    }
    if (getPipelineStats_) printPipelineStats(out.pipeline, entryPoint, spec);
    out.pushConstantSize = pushConstantSize;
    return VK_SUCCESS;
}

void Device::printPipelineStats(VkPipeline pipeline, const char* entryPoint, const uint32_t spec[5]) {
    VkPipelineExecutableInfoKHR exe{VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR};
    exe.pipeline = pipeline;
    uint32_t count = 0;
    if (getPipelineStats_(device_, &exe, &count, nullptr) != VK_SUCCESS) return;
    std::vector<VkPipelineExecutableStatisticKHR> stats(count, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR});
    if (getPipelineStats_(device_, &exe, &count, stats.data()) != VK_SUCCESS) return;
    std::string line;
    for (const VkPipelineExecutableStatisticKHR& s : stats) {
        char v[64];
        switch (s.format) {
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR: std::snprintf(v, sizeof v, "%u", s.value.b32); break;
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR: std::snprintf(v, sizeof v, "%lld", (long long)s.value.i64); break;
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR: std::snprintf(v, sizeof v, "%llu", (unsigned long long)s.value.u64); break;
            default: std::snprintf(v, sizeof v, "%g", s.value.f64); break;
        }
        line += std::string(line.empty() ? "" : ", ") + s.name + " " + v;
    }
    std::fprintf(stderr, "[vulkanizer] pipeline %s (block %ux%ux%u, pack %u): %s\n", displayName(entryPoint).c_str(),
                 spec[0], spec[1], spec[2], spec[4], line.c_str());
}

void Device::destroyComputePipeline(ComputePipeline& p) {
    finish();  // recorded dispatches may still use it
    {
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        nsPerGroup_.erase(p.pipeline);  // the handle may be reused
    }
    if (p.pipeline) vkDestroyPipeline(device_, p.pipeline, nullptr);
    if (p.layout) vkDestroyPipelineLayout(device_, p.layout, nullptr);
    if (p.module) vkDestroyShaderModule(device_, p.module, nullptr);
    p = ComputePipeline{};
}

VkResult Device::dispatch(const ComputePipeline& p, uint32_t groupsX, uint32_t groupsY, uint32_t groupsZ,
                          const void* pushData, uint32_t pushSize, const char* label, const void* params,
                          size_t paramsSize, bool whole) {
    if (pushSize > p.pushConstantSize || (params && pushSize < sizeof(VkDeviceAddress))) return VK_ERROR_FEATURE_NOT_PRESENT;
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    std::vector<uint8_t> push(static_cast<const uint8_t*>(pushData), static_cast<const uint8_t*>(pushData) + pushSize);
    uint64_t uploaded = 0;
    const uint32_t grid[3] = {groupsX, groupsY, groupsZ};
    if (!batchBudgetNs_ || whole) {
        if (params) {
            if (VkResult r = uploadParams(params, paramsSize, push, uploaded)) return r;
        }
        if (VkResult r = recordDispatch(p, std::array<uint32_t, 3>{}.data(), grid, push.data(), pushSize, label, 0))
            return r;
        return launchBlocking_ ? finish() : VK_SUCCESS;
    }

    // Workgroups per chunk: as many as fit in the budget, or a probe on its own for a new pipeline.
    auto known = nsPerGroup_.find(p.pipeline);
    const bool probe = known == nsPerGroup_.end();
    uint64_t chunk = kRateGroups;
    if (probe) {
        if (VkResult r = flushLocked()) return r;
    } else if (known->second > 0) {
        chunk = std::max<uint64_t>(1, uint64_t(batchBudgetNs_ / known->second));
    } else {
        chunk = UINT64_MAX;
    }
    // Chunks are boxes of whole planes, whole rows or part of a row, in blockIdx order.
    const uint64_t row = groupsX, plane = row * groupsY;
    uint32_t step[3] = {groupsX, groupsY, groupsZ};
    if (chunk < plane) {
        step[2] = 1;
        if (chunk < row) step[1] = 1, step[0] = uint32_t(chunk);
        else step[1] = uint32_t(chunk / row);
    } else {
        step[2] = uint32_t(std::min<uint64_t>(groupsZ, chunk / plane));
    }
    bool first = true;
    for (uint32_t z = 0; z < groupsZ; z += step[2])
        for (uint32_t y = 0; y < groupsY; y += step[1])
            for (uint32_t x = 0; x < groupsX; x += step[0]) {
                const uint32_t base[3] = {x, y, z};
                const uint32_t count[3] = {std::min(step[0], groupsX - x), std::min(step[1], groupsY - y),
                                           std::min(step[2], groupsZ - z)};
                double estimate = 0;
                if (!probe || !first) {
                    const uint64_t groups = uint64_t(count[0]) * count[1] * count[2];
                    estimate = nsPerGroup_[p.pipeline] * double(std::max(groups, kRateGroups));
                    if (open_ >= 0 && commands_ && batches_[open_].estimateNs + estimate > batchBudgetNs_) {
                        if (VkResult r = flushLocked()) return r;
                    }
                }
                if (params) {  // each batch keeps its own copy (its arena is reused once it has run)
                    if (VkResult r = uploadParams(params, paramsSize, push, uploaded)) return r;
                }
                if (VkResult r = recordDispatch(p, base, count, push.data(), pushSize, label, estimate)) return r;
                if (probe && first) {
                    if (VkResult r = finish()) return r;  // measures it
                }
                first = false;
            }
    return launchBlocking_ ? finish() : VK_SUCCESS;
}

VkResult Device::uploadParams(const void* params, size_t size, std::vector<uint8_t>& push, uint64_t& uploadedSerial) {
    if (open_ >= 0 && batches_[open_].serial == uploadedSerial) return VK_SUCCESS;
    VkDeviceAddress addr;
    if (VkResult r = upload(params, size, addr)) return r;
    std::memcpy(push.data(), &addr, sizeof addr);
    uploadedSerial = batches_[open_].serial;
    return VK_SUCCESS;
}

VkResult Device::recordDispatch(const ComputePipeline& p, const uint32_t base[3], const uint32_t count[3],
                                const void* pushData, uint32_t pushSize, const char* label, double estimateNs) {
    return record([&](VkCommandBuffer cb) {
        Batch& b = batches_[open_];
        const uint32_t query = uint32_t(2 * b.timed.size());
        if (timing_) vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, b.timing, query);
        b.kernels.push_back(label);
        b.estimateNs += estimateNs;
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline);
        if (pushSize) vkCmdPushConstants(cb, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushSize, pushData);
        // CUDA allows 2^31-1 blocks in x; split along x when the device allows fewer. The base
        // offset keeps WorkgroupId (blockIdx) global.
        const uint32_t limit = info_.props.limits.maxComputeWorkGroupCount[0];
        for (uint32_t x = 0; x < count[0]; x += limit)
            vkCmdDispatchBase(cb, base[0] + x, base[1], base[2], std::min(limit, count[0] - x), count[1], count[2]);
        if (timing_) {
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, b.timing, query + 1);
            b.timed.push_back({label, p.pipeline, uint64_t(count[0]) * count[1] * count[2], estimateNs});
        }
    });
}



}  // namespace vulkanizer::vk
