// Virtual memory management (cuMemAddressReserve / cuMemCreate / cuMemMap) on sparse buffers,
// and host memory registration.
//
// A reservation is a row of sparse residency buffers on a contiguous device address range: one
// buffer is limited to maxBufferSize (4 GB on RADV), while applications reserve far more (the
// ComfyUI DynamicVRAM allocator reserves model-sized ranges). Buffers after the first are placed
// with bufferDeviceAddressCaptureReplay. cuMemCreate allocates a VkDeviceMemory, cuMemMap binds
// its pages with vkQueueBindSparse. Kernels see one flat range through the device address; the
// host-side copies split at buffer boundaries (findPieces).

#include "driver/driver.h"

#include "common/log.h"

#include <algorithm>
#include <map>
#include <mutex>
#include <vector>

using namespace vulkanizer;
using namespace vulkanizer::driver;

namespace {

constexpr size_t kMinGranularity = size_t(2) << 20;

// The page size of mappings: the sparse page size, at least 2 MB like CUDA's.
size_t granularity(vk::Device& dev) { return std::max<size_t>(kMinGranularity, dev.sparsePageSize()); }

CUmemAllocation_st* allocationOf(CUmemGenericAllocationHandle handle) {
    auto* a = reinterpret_cast<CUmemAllocation_st*>(static_cast<uintptr_t>(handle));
    return validHandle(a) ? a : nullptr;
}

void releaseRef(vk::Device& dev, DeviceSlot& slot, CUmemAllocation_st* a) {
    if (--a->refs > 0) return;
    dev.freeMemory(a->memory);
    slot.allocatedBytes -= a->size;
    Driver::get().unregisterHandle(a);
    delete a;
}

// Sparse page ranges of [ptr, ptr + size) in a reservation, bound to memory at memoryOffset
// (or unbound).
std::vector<vk::SparseBind> sparseBinds(VmReservation& res, CUdeviceptr ptr, size_t size, VkDeviceMemory memory,
                                        size_t memoryOffset) {
    std::vector<vk::SparseBind> binds;
    for (size_t at = 0; at < size;) {
        const CUdeviceptr p = ptr + at;
        const vk::Buffer& chunk = res.chunks[(p - res.base) / res.chunkSize];
        const VkDeviceSize offset = p - chunk.address;
        const size_t n = std::min<size_t>(size - at, chunk.size - offset);
        binds.push_back({&chunk, offset, n, memory, memory ? memoryOffset + at : 0});
        at += n;
    }
    return binds;
}

// Creates the buffers of a reservation of `size` bytes on one contiguous address range. All
// buffers but a lone one are chunkSize large (address space only: nothing is allocated).
VkResult createChunks(vk::Device& dev, size_t size, size_t alignment, VmReservation& res) {
    const VkDeviceSize chunk = dev.maxSparseBufferSize() / granularity(dev) * granularity(dev);
    if (!chunk) return VK_ERROR_FEATURE_NOT_PRESENT;
    const uint64_t n = (size + chunk - 1) / chunk;
    auto done = [&](std::vector<vk::Buffer> chunks) {
        res.chunks = std::move(chunks);
        res.chunkSize = chunk;
        res.base = res.chunks[0].address;
        return VK_SUCCESS;
    };

    // The first buffer goes wherever the driver puts it; that also gives the relation between
    // opaque capture addresses and device addresses (equal on RADV and most drivers).
    vk::Buffer first;
    uint64_t firstOpaque = 0;
    if (VkResult r = dev.createSparseBuffer(n == 1 ? size : chunk, 0, first, firstOpaque)) return r;
    if (n == 1 && (!alignment || first.address % alignment == 0)) return done({first});
    if (!dev.features().addressCaptureReplay) {
        dev.freeBuffer(first);
        VULKANIZER_WARN_ONCE("the device cannot place buffers at chosen addresses (bufferDeviceAddressCaptureReplay): "
                             "virtual address reservations are limited to %llu bytes",
                             static_cast<unsigned long long>(chunk));
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    const uint64_t delta = first.address - firstOpaque, span = n * chunk;
    VULKANIZER_DEBUG("cuMemAddressReserve: %llu buffers of %llu bytes, the first at 0x%llx (opaque 0x%llx)",
                     static_cast<unsigned long long>(n), static_cast<unsigned long long>(chunk),
                     static_cast<unsigned long long>(first.address), static_cast<unsigned long long>(firstOpaque));

    // Candidate starts: ending at the first buffer (RADV hands out addresses top-down, so the
    // space below it is free), starting at it, then further below and above. A candidate that
    // covers the first buffer's place keeps it there. Ranges that wrap around are never asked
    // for: libdrm's address manager does not catch that and is corrupted by it.
    std::vector<uint64_t> starts;
    auto candidate = [&](uint64_t from, uint64_t below, uint64_t above) {
        if (below > from || UINT64_MAX - (from - below) < above) return;
        uint64_t s = from - below + above;
        if (alignment) s = (s + delta) / alignment * alignment - delta;
        if (UINT64_MAX - s >= span) starts.push_back(s);
    };
    candidate(firstOpaque, (n - 1) * chunk, 0);
    candidate(firstOpaque, 0, 0);
    for (uint64_t k = 1; k <= 4; ++k) candidate(firstOpaque, (n - 1) * chunk + k * span, 0);
    for (uint64_t k = 1; k <= 4; ++k) candidate(firstOpaque, 0, k * span);

    bool firstKept = true;
    for (uint64_t start : starts) {
        uint64_t keep = n;  // index of the first buffer in this candidate, n if none
        if (firstKept && firstOpaque >= start && (firstOpaque - start) % chunk == 0 && (firstOpaque - start) / chunk < n)
            keep = (firstOpaque - start) / chunk;
        else if (firstKept && firstOpaque < start + span && firstOpaque + chunk > start) {
            dev.freeBuffer(first);  // in the way
            firstKept = false;
        }
        std::vector<vk::Buffer> chunks;
        bool ok = true;
        for (uint64_t i = 0; i < n && ok; ++i) {
            if (i == keep) {
                chunks.push_back(first);
                continue;
            }
            const uint64_t opaque = start + i * chunk;
            vk::Buffer b;
            uint64_t got = 0;
            if (VkResult r = dev.createSparseBuffer(chunk, opaque, b, got)) {
                VULKANIZER_DEBUG("cuMemAddressReserve: buffer %llu at 0x%llx failed: %d",
                                 static_cast<unsigned long long>(i), static_cast<unsigned long long>(opaque), r);
                ok = false;
                break;
            }
            chunks.push_back(b);
            ok = b.address == opaque + delta;
        }
        if (ok) return done(std::move(chunks));
        for (vk::Buffer& b : chunks)
            if (b.buffer != first.buffer) dev.freeBuffer(b);
    }
    if (firstKept) dev.freeBuffer(first);
    return VK_ERROR_OUT_OF_DEVICE_MEMORY;
}

}  // namespace

extern "C" {

CUresult CUDAAPI cuMemGetAllocationGranularity(size_t* granularityOut, const CUmemAllocationProp* prop,
                                               CUmemAllocationGranularity_flags option) {
    VULKANIZER_TRACE_API();
    if (!granularityOut || !prop) return CUDA_ERROR_INVALID_VALUE;
    if (prop->type != CU_MEM_ALLOCATION_TYPE_PINNED || prop->location.type != CU_MEM_LOCATION_TYPE_DEVICE)
        return CUDA_ERROR_INVALID_VALUE;
    Driver& d = Driver::get();
    if (!d.initialized()) return CUDA_ERROR_NOT_INITIALIZED;
    std::lock_guard<std::recursive_mutex> lock(d.mutex());
    DeviceSlot* slot = d.slot(prop->location.id);
    if (!slot) return CUDA_ERROR_INVALID_DEVICE;
    if (CUresult r = d.ensureDevice(*slot)) return r;
    if (!slot->vk->features().sparseBuffers) return CUDA_ERROR_NOT_SUPPORTED;
    *granularityOut = granularity(*slot->vk);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemAddressReserve(CUdeviceptr* ptr, size_t size, size_t alignment, CUdeviceptr addr,
                                     unsigned long long flags) {
    VULKANIZER_TRACE_API();
    if (!ptr || size == 0 || flags != 0) return CUDA_ERROR_INVALID_VALUE;
    if (alignment & (alignment - 1)) return CUDA_ERROR_INVALID_VALUE;
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!s.dev->features().sparseBuffers) return CUDA_ERROR_NOT_SUPPORTED;
    if (size % granularity(*s.dev)) return CUDA_ERROR_INVALID_VALUE;
    (void)addr;  // a hint only
    VmReservation res;
    res.size = size;
    if (VkResult vr = createChunks(*s.dev, size, alignment, res)) {
        VULKANIZER_WARN("cuMemAddressReserve(%zu bytes) failed: %d", size, vr);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    VULKANIZER_DEBUG("cuMemAddressReserve(%zu) -> 0x%llx (%zu buffer(s))", size,
                     static_cast<unsigned long long>(res.base), res.chunks.size());
    *ptr = res.base;
    s.slot->reservations[res.base] = std::move(res);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemAddressFree(CUdeviceptr ptr, size_t size) {
    VULKANIZER_TRACE_API();
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    auto it = s.slot->reservations.find(ptr);
    if (it == s.slot->reservations.end() || it->second.size != size) return CUDA_ERROR_INVALID_VALUE;
    // Whatever is still mapped (an application error) is unmapped first.
    for (auto m = s.slot->mappings.lower_bound(ptr); m != s.slot->mappings.end() && m->first < ptr + size;) {
        if (VkResult vr = s.dev->bindSparse(sparseBinds(it->second, m->first, m->second.size, VK_NULL_HANDLE, 0)))
            return toCuResult(vr);
        releaseRef(*s.dev, *s.slot, m->second.allocation);
        m = s.slot->mappings.erase(m);
    }
    for (vk::Buffer& b : it->second.chunks) s.dev->freeBuffer(b);
    s.slot->reservations.erase(it);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemCreate(CUmemGenericAllocationHandle* handle, size_t size, const CUmemAllocationProp* prop,
                             unsigned long long flags) {
    VULKANIZER_TRACE_API();
    if (!handle || !prop || size == 0 || flags != 0) return CUDA_ERROR_INVALID_VALUE;
    if (prop->type != CU_MEM_ALLOCATION_TYPE_PINNED || prop->location.type != CU_MEM_LOCATION_TYPE_DEVICE)
        return CUDA_ERROR_INVALID_VALUE;
    Driver& d = Driver::get();
    if (!d.initialized()) return CUDA_ERROR_NOT_INITIALIZED;
    std::lock_guard<std::recursive_mutex> lock(d.mutex());
    DeviceSlot* slot = d.slot(prop->location.id);
    if (!slot) return CUDA_ERROR_INVALID_DEVICE;
    if (CUresult r = d.ensureDevice(*slot)) return r;
    vk::Device& dev = *slot->vk;
    if (!dev.features().sparseBuffers) return CUDA_ERROR_NOT_SUPPORTED;
    if (size % granularity(dev)) return CUDA_ERROR_INVALID_VALUE;
    // Shareable handles (requestedHandleTypes) are accepted, but cannot be exported.
    VkDeviceMemory memory = VK_NULL_HANDLE;
    if (VkResult vr = dev.allocateMemory(size, memory)) {
        VULKANIZER_DEBUG("cuMemCreate(%zu) failed: %d", size, vr);
        return toCuResult(vr);
    }
    auto* a = new CUmemAllocation_st();
    a->device = prop->location.id;
    a->prop = *prop;
    a->memory = memory;
    a->size = size;
    slot->allocatedBytes += size;
    d.registerHandle(a);
    *handle = reinterpret_cast<uintptr_t>(a);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemRelease(CUmemGenericAllocationHandle handle) {
    VULKANIZER_TRACE_API();
    Driver& d = Driver::get();
    std::lock_guard<std::recursive_mutex> lock(d.mutex());
    CUmemAllocation_st* a = allocationOf(handle);
    if (!a) return CUDA_ERROR_INVALID_VALUE;
    DeviceSlot* slot = d.slot(a->device);
    releaseRef(*slot->vk, *slot, a);  // the memory stays while it is mapped
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemMap(CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle,
                          unsigned long long flags) {
    VULKANIZER_TRACE_API();
    if (size == 0 || flags != 0) return CUDA_ERROR_INVALID_VALUE;
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    CUmemAllocation_st* a = allocationOf(handle);
    if (!a || a->device != s.slot->ordinal) return CUDA_ERROR_INVALID_HANDLE;
    const size_t g = granularity(*s.dev);
    if (ptr % g || size % g || offset % g || offset > a->size || size > a->size - offset)
        return CUDA_ERROR_INVALID_VALUE;
    VmReservation* res = findReservation(*s.slot, ptr);
    if (!res || size > res->base + res->size - ptr) return CUDA_ERROR_INVALID_VALUE;
    auto next = s.slot->mappings.lower_bound(ptr);
    if (next != s.slot->mappings.end() && next->first < ptr + size) return CUDA_ERROR_ALREADY_MAPPED;
    if (next != s.slot->mappings.begin() && std::prev(next)->first + std::prev(next)->second.size > ptr)
        return CUDA_ERROR_ALREADY_MAPPED;
    if (VkResult vr = s.dev->bindSparse(sparseBinds(*res, ptr, size, a->memory, offset))) return toCuResult(vr);
    s.slot->mappings[ptr] = {size, a, offset};
    ++a->refs;
    return CUDA_SUCCESS;
}

// Unmaps whole mappings (one or several adjacent ones), as CUDA requires.
CUresult CUDAAPI cuMemUnmap(CUdeviceptr ptr, size_t size) {
    VULKANIZER_TRACE_API();
    if (size == 0) return CUDA_ERROR_INVALID_VALUE;
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    VmReservation* res = findReservation(*s.slot, ptr);
    if (!res || size > res->base + res->size - ptr) return CUDA_ERROR_INVALID_VALUE;
    auto first = s.slot->mappings.lower_bound(ptr);
    auto last = first;
    for (; last != s.slot->mappings.end() && last->first < ptr + size; ++last)
        if (last->first + last->second.size > ptr + size) return CUDA_ERROR_INVALID_VALUE;
    if (first == last || first->first != ptr) return CUDA_ERROR_INVALID_VALUE;
    std::vector<vk::SparseBind> binds;
    for (auto m = first; m != last; ++m) {
        auto part = sparseBinds(*res, m->first, m->second.size, VK_NULL_HANDLE, 0);
        binds.insert(binds.end(), part.begin(), part.end());
    }
    if (VkResult vr = s.dev->bindSparse(binds)) return toCuResult(vr);  // waits for the recorded work
    for (auto m = first; m != last;) {
        releaseRef(*s.dev, *s.slot, m->second.allocation);
        m = s.slot->mappings.erase(m);
    }
    return CUDA_SUCCESS;
}

// Mapped memory is always readable and writable by its own device; peers have no access.
CUresult CUDAAPI cuMemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc* desc, size_t count) {
    VULKANIZER_TRACE_API();
    if (!desc || count == 0 || size == 0) return CUDA_ERROR_INVALID_VALUE;
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!findReservation(*s.slot, ptr)) return CUDA_ERROR_INVALID_VALUE;
    for (size_t i = 0; i < count; ++i) {
        if (desc[i].location.type != CU_MEM_LOCATION_TYPE_DEVICE) return CUDA_ERROR_INVALID_VALUE;
        if (desc[i].location.id != s.slot->ordinal && desc[i].flags != CU_MEM_ACCESS_FLAGS_PROT_NONE)
            return CUDA_ERROR_NOT_SUPPORTED;
    }
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemGetAccess(unsigned long long* flags, const CUmemLocation* location, CUdeviceptr ptr) {
    VULKANIZER_TRACE_API();
    if (!flags || !location) return CUDA_ERROR_INVALID_VALUE;
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    auto m = s.slot->mappings.upper_bound(ptr);
    if (m == s.slot->mappings.begin() || ptr - std::prev(m)->first >= std::prev(m)->second.size)
        return CUDA_ERROR_INVALID_VALUE;
    *flags = location->type == CU_MEM_LOCATION_TYPE_DEVICE && location->id == s.slot->ordinal
                 ? CU_MEM_ACCESS_FLAGS_PROT_READWRITE
                 : CU_MEM_ACCESS_FLAGS_PROT_NONE;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemGetAllocationPropertiesFromHandle(CUmemAllocationProp* prop, CUmemGenericAllocationHandle handle) {
    VULKANIZER_TRACE_API();
    if (!prop) return CUDA_ERROR_INVALID_VALUE;
    std::lock_guard<std::recursive_mutex> lock(Driver::get().mutex());
    CUmemAllocation_st* a = allocationOf(handle);
    if (!a) return CUDA_ERROR_INVALID_VALUE;
    *prop = a->prop;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemRetainAllocationHandle(CUmemGenericAllocationHandle* handle, void* addr) {
    VULKANIZER_TRACE_API();
    if (!handle) return CUDA_ERROR_INVALID_VALUE;
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    const CUdeviceptr ptr = reinterpret_cast<CUdeviceptr>(addr);
    auto m = s.slot->mappings.upper_bound(ptr);
    if (m == s.slot->mappings.begin() || ptr - std::prev(m)->first >= std::prev(m)->second.size)
        return CUDA_ERROR_INVALID_VALUE;
    CUmemAllocation_st* a = std::prev(m)->second.allocation;
    ++a->refs;
    *handle = reinterpret_cast<uintptr_t>(a);
    return CUDA_SUCCESS;
}

}  // extern "C"
