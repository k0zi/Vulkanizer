#include "driver/driver.h"

#include "common/log.h"

#include <algorithm>

namespace vulkanizer::driver {

Driver& Driver::get() {
    static Driver* driver = new Driver();
    return *driver;
}

CUresult Driver::init() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (initialized()) return CUDA_SUCCESS;
    if (initAttempted_) return initResult_;
    initAttempted_ = true;

    vk::Instance* instance = vk::Instance::get();
    if (!instance) {
        VULKANIZER_ERROR("Vulkan is not available");
        return initResult_ = CUDA_ERROR_NO_DEVICE;
    }
    if (instance->devices().empty()) {
        VULKANIZER_ERROR("no usable Vulkan device (needs Vulkan 1.2, bufferDeviceAddress, shaderInt64; "
                    "CPU devices need VULKANIZER_ALLOW_CPU=1)");
        return initResult_ = CUDA_ERROR_NO_DEVICE;
    }
    for (const vk::PhysicalDeviceInfo& info : instance->devices()) {
        auto slot = std::make_unique<DeviceSlot>();
        slot->ordinal = static_cast<CUdevice>(slots_.size());
        slot->info = &info;
        slot->primary.device = slot->ordinal;
        slot->primary.primary = true;
        liveHandles_.insert(&slot->primary);
        slot->defaultPool.device = slot->ordinal;
        slot->defaultPool.props.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
        slot->defaultPool.props.location = {CU_MEM_LOCATION_TYPE_DEVICE, slot->ordinal};
        liveHandles_.insert(&slot->defaultPool);
        slots_.push_back(std::move(slot));
    }
    initialized_.store(true, std::memory_order_release);
    VULKANIZER_INFO("initialized with %d device(s)", deviceCount());
    return initResult_ = CUDA_SUCCESS;
}

CUresult Driver::ensureDevice(DeviceSlot& slot) {
    if (slot.vk) return CUDA_SUCCESS;
    VkResult r = vk::Device::create(*slot.info, slot.vk);
    if (r != VK_SUCCESS) {
        VULKANIZER_ERROR("failed to create Vulkan device for '%s': %d", slot.info->name(), r);
        return toCuResult(r);
    }
    return CUDA_SUCCESS;
}

void Driver::registerHandle(const void* h) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    liveHandles_.insert(h);
}

void Driver::unregisterHandle(const void* h) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    liveHandles_.erase(h);
}

bool Driver::isLive(const void* h) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    return liveHandles_.count(h) != 0;
}

ContextStack& contextStack() {
    static thread_local ContextStack stack;
    return stack;
}

CUcontext currentContext() {
    auto& stack = contextStack();
    return stack.empty() ? nullptr : stack.back();
}

CUresult enterContext(ApiScope& scope) {
    Driver& d = Driver::get();
    if (!d.initialized()) return CUDA_ERROR_NOT_INITIALIZED;
    CUcontext ctx = currentContext();
    if (!ctx) return CUDA_ERROR_INVALID_CONTEXT;
    scope.lock = std::unique_lock<std::recursive_mutex>(d.mutex());
    if (!validHandle(ctx)) return CUDA_ERROR_CONTEXT_IS_DESTROYED;
    scope.ctx = ctx;
    scope.slot = d.slot(ctx->device);
    if (!scope.slot) return CUDA_ERROR_INVALID_CONTEXT;
    if (CUresult r = d.ensureDevice(*scope.slot); r != CUDA_SUCCESS) return r;
    scope.dev = scope.slot->vk.get();
    return CUDA_SUCCESS;
}

bool validStream(CUstream stream) {
    return stream == nullptr || stream == CU_STREAM_LEGACY || stream == CU_STREAM_PER_THREAD || validHandle(stream);
}

VmReservation* findReservation(DeviceSlot& slot, CUdeviceptr ptr) {
    auto it = slot.reservations.upper_bound(ptr);
    if (it == slot.reservations.begin()) return nullptr;
    --it;
    return ptr - it->first < it->second.size ? &it->second : nullptr;
}

bool findAllocation(DeviceSlot& slot, CUdeviceptr ptr, size_t bytes, AllocRef& out) {
    auto it = slot.allocations.upper_bound(ptr);
    if (it != slot.allocations.begin()) {
        --it;
        const VkDeviceSize offset = ptr - it->first;
        if (offset < it->second.size && bytes <= it->second.size - offset) {
            out = {&it->second, it->first, offset};
            return true;
        }
    }
    if (VmReservation* res = findReservation(slot, ptr)) {
        const size_t i = (ptr - res->base) / res->chunkSize;
        vk::Buffer& chunk = res->chunks[i];
        const VkDeviceSize offset = ptr - chunk.address;
        if (bytes > chunk.size - offset) return false;
        out = {&chunk, chunk.address, offset};
        return true;
    }
    return false;
}

bool findPieces(DeviceSlot& slot, CUdeviceptr ptr, size_t bytes, std::vector<AllocPiece>& out) {
    out.clear();
    AllocRef ref;
    if (findAllocation(slot, ptr, bytes, ref)) {
        out.push_back({ref, 0, bytes});
        return true;
    }
    VmReservation* res = findReservation(slot, ptr);
    if (!res || bytes > res->base + res->size - ptr) return false;
    for (size_t at = 0; at < bytes;) {
        const CUdeviceptr p = ptr + at;
        vk::Buffer& chunk = res->chunks[(p - res->base) / res->chunkSize];
        const VkDeviceSize offset = p - chunk.address;
        const size_t n = std::min<size_t>(bytes - at, chunk.size - offset);
        out.push_back({{&chunk, chunk.address, offset}, at, n});
        at += n;
    }
    return true;
}

CUresult toCuResult(VkResult r) {
    switch (r) {
        case VK_SUCCESS: return CUDA_SUCCESS;
        case VK_ERROR_OUT_OF_HOST_MEMORY:
        case VK_ERROR_OUT_OF_DEVICE_MEMORY: return CUDA_ERROR_OUT_OF_MEMORY;
        case VK_ERROR_DEVICE_LOST: return CUDA_ERROR_LAUNCH_FAILED;
        case VK_ERROR_FEATURE_NOT_PRESENT:
        case VK_ERROR_EXTENSION_NOT_PRESENT: return CUDA_ERROR_NOT_SUPPORTED;
        case VK_ERROR_INCOMPATIBLE_DRIVER: return CUDA_ERROR_NO_DEVICE;
        default: return CUDA_ERROR_UNKNOWN;
    }
}

}  // namespace vulkanizer::driver
