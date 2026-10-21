#pragma once

#include <cuda.h>

#include "compiler/ptx_to_spirv.h"
#include "ptx/parser.h"
#include "vk/device.h"
#include "vk/instance.h"

#include <array>
#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

// Handle objects behind the opaque CUDA handle types. Each carries a magic value
// and is registered in Driver's live-handle set so stale handles are rejected.

struct CUctx_st {
    static constexpr uint32_t kMagic = 0x58544353;  // 'SCTX'
    uint32_t magic = kMagic;
    CUdevice device = 0;
    unsigned int flags = 0;
    bool primary = false;
};

struct CUfunc_st {
    static constexpr uint32_t kMagic = 0x434e5553;  // 'SUNC'
    uint32_t magic = kMagic;
    CUmod_st* module = nullptr;
    const vulkanizer::ptx::Entry* entry = nullptr;
    bool compileAttempted = false;
    bool compiled = false;
    std::string compileError;
    vulkanizer::compiler::KernelBinary binary;
    // keyed by the specialization: blockDim.x/y/z, dynamic shared memory words
    std::map<std::array<uint32_t, 5>, vulkanizer::vk::ComputePipeline> pipelines;  // by spec constants
};

struct CUmod_st {
    static constexpr uint32_t kMagic = 0x444f4d53;  // 'SMOD'
    uint32_t magic = kMagic;
    CUcontext ctx = nullptr;
    std::string source;       // PTX text; `ptx` holds views into it, so it must never be moved
    vulkanizer::ptx::Module ptx;
    CUdeviceptr globals = 0;  // buffer of the module's .global/.const variables (0 if none)
    std::vector<uint8_t> globalsImage;  // their initial contents (printf format strings)
    std::string sourceHash;             // SHA-256 of `source`, for the kernel cache (computed lazily)
    std::vector<std::unique_ptr<CUfunc_st>> functions;
};

struct CUstream_st {
    static constexpr uint32_t kMagic = 0x52545353;  // 'SSTR'
    uint32_t magic = kMagic;
    CUcontext ctx = nullptr;
    unsigned int flags = 0;
};

struct CUevent_st {
    static constexpr uint32_t kMagic = 0x54564553;  // 'SEVT'
    uint32_t magic = kMagic;
    CUcontext ctx = nullptr;
    unsigned int flags = 0;
    bool recorded = false;
    uint64_t serial = 0;      // device work the event completes with
    int32_t timestamp = -1;   // GPU timestamp slot, -1 without timing
};

// A memory pool (stream-ordered allocation). Allocations come straight from the device; the pool
// keeps the statistics and attributes applications read back.
struct CUmemPoolHandle_st {
    static constexpr uint32_t kMagic = 0x4c4f5053;  // 'SPOL'
    uint32_t magic = kMagic;
    CUdevice device = 0;
    CUmemPoolProps props{};
    uint64_t releaseThreshold = 0;
    int reuseFlags[3] = {1, 1, 1};  // CU_MEMPOOL_ATTR_REUSE_*
    uint64_t usedBytes = 0, usedHigh = 0;
};

// Physical memory of the virtual memory API (cuMemCreate); the CUmemGenericAllocationHandle
// points to it. Freed when it is released and no longer mapped.
struct CUmemAllocation_st {
    static constexpr uint32_t kMagic = 0x4d454d53;  // 'SMEM'
    uint32_t magic = kMagic;
    CUdevice device = 0;
    CUmemAllocationProp prop{};
    VkDeviceMemory memory = VK_NULL_HANDLE;
    size_t size = 0;
    int refs = 1;  // the handle itself plus one per mapping
};

namespace vulkanizer::driver {

// An address range of cuMemAddressReserve: sparse buffers of `chunkSize` bytes side by side
// (one buffer is limited to maxBufferSize, 4 GB on RADV).
struct VmReservation {
    CUdeviceptr base = 0;
    size_t size = 0;
    VkDeviceSize chunkSize = 0;
    std::vector<vk::Buffer> chunks;
};

struct VmMapping {
    size_t size = 0;
    CUmemAllocation_st* allocation = nullptr;
    size_t offset = 0;  // into the allocation
};

struct DeviceSlot {
    CUdevice ordinal = 0;
    const vk::PhysicalDeviceInfo* info = nullptr;
    std::unique_ptr<vk::Device> vk;  // created with the first context that needs it
    CUctx_st primary;
    int primaryRefs = 0;
    std::map<CUdeviceptr, vk::Buffer> allocations;  // keyed by base address
    size_t allocatedBytes = 0;
    std::map<CUdeviceptr, CUmemPoolHandle_st*> poolAllocations;  // stream-ordered ones, by pool
    CUmemPoolHandle_st defaultPool;
    CUmemPoolHandle_st* currentPool = &defaultPool;
    std::map<CUdeviceptr, VmReservation> reservations;  // keyed by base address
    std::map<CUdeviceptr, VmMapping> mappings;          // keyed by mapped address
    vk::Buffer printfBuffer;  // device printf records of the current launch
};

class Driver {
public:
    // Never destroyed (see vk::Instance).
    static Driver& get();

    CUresult init();
    bool initialized() const { return initialized_.load(std::memory_order_acquire); }

    int deviceCount() const { return static_cast<int>(slots_.size()); }
    DeviceSlot* slot(CUdevice dev) {
        return dev >= 0 && dev < deviceCount() ? slots_[static_cast<size_t>(dev)].get() : nullptr;
    }
    // Lazily creates the Vulkan logical device. Caller holds mutex().
    CUresult ensureDevice(DeviceSlot& slot);

    std::recursive_mutex& mutex() { return mutex_; }

    void registerHandle(const void* h);
    void unregisterHandle(const void* h);
    bool isLive(const void* h);

private:
    Driver() = default;

    std::recursive_mutex mutex_;
    std::atomic<bool> initialized_{false};
    CUresult initResult_ = CUDA_SUCCESS;
    bool initAttempted_ = false;
    std::vector<std::unique_ptr<DeviceSlot>> slots_;
    std::unordered_set<const void*> liveHandles_;
};

template <class T>
bool validHandle(T* h) {
    return h && Driver::get().isLive(h) && h->magic == T::kMagic;
}

// Per-thread context stack. Fixed capacity and trivially destructible, so it stays valid
// for atexit handlers (e.g. libcudart unregistering its modules) that run after the
// thread_local destructors of the main thread.
class ContextStack {
public:
    static constexpr size_t kCapacity = 64;
    bool empty() const { return size_ == 0; }
    CUcontext& back() { return items_[size_ - 1]; }
    bool push(CUcontext c) {
        if (size_ == kCapacity) return false;
        items_[size_++] = c;
        return true;
    }
    void pop() { --size_; }
    void remove(CUcontext c) {
        size_t n = 0;
        for (size_t i = 0; i < size_; ++i)
            if (items_[i] != c) items_[n++] = items_[i];
        size_ = n;
    }

private:
    CUcontext items_[kCapacity];
    size_t size_;
};

CUcontext currentContext();
ContextStack& contextStack();

// Common prologue for calls that need a current context: checks initialization,
// takes the driver lock and makes sure the Vulkan device exists.
struct ApiScope {
    std::unique_lock<std::recursive_mutex> lock;
    CUcontext ctx = nullptr;
    DeviceSlot* slot = nullptr;
    vk::Device* dev = nullptr;
};
CUresult enterContext(ApiScope& scope);

// The default (0), legacy and per-thread streams, or a live stream of the context.
bool validStream(CUstream stream);

struct AllocRef {
    vk::Buffer* buffer = nullptr;
    CUdeviceptr base = 0;
    VkDeviceSize offset = 0;
};
// Finds the allocation containing [ptr, ptr + bytes). Caller holds the driver lock.
bool findAllocation(DeviceSlot& slot, CUdeviceptr ptr, size_t bytes, AllocRef& out);

// [ptr, ptr + bytes) as consecutive buffer ranges: a virtual memory reservation consists of
// several buffers. `at` is the piece's offset within the range.
struct AllocPiece {
    AllocRef ref;
    size_t at = 0, size = 0;
};
bool findPieces(DeviceSlot& slot, CUdeviceptr ptr, size_t bytes, std::vector<AllocPiece>& out);

// The reservation containing ptr, or nullptr.
VmReservation* findReservation(DeviceSlot& slot, CUdeviceptr ptr);

CUresult toCuResult(VkResult r);

// Compute capability reported to applications (default 7.5; VULKANIZER_CC=86 overrides).
void computeCapability(int& major, int& minor);

}  // namespace vulkanizer::driver
