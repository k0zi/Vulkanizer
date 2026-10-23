// Stream-ordered allocation and memory pools. All streams share one queue, so an allocation
// is usable by any stream right away; a free is deferred until the work recorded before it has
// executed, without waiting for it. Pools do not cache memory: they hold the attributes and
// the usage statistics applications read back.

#include "driver/driver.h"

#include "common/log.h"

#include <algorithm>
#include <cstring>

using namespace vulkanizer;
using namespace vulkanizer::driver;

// On a capturing stream these become memory nodes (graph memory, api_graph.cpp). Graph memory freed
// outside its graph stays the graph's.
static CUresult allocFromPool(CUdeviceptr* dptr, size_t bytesize, CUmemPoolHandle_st* pool, CUstream hStream) {
    if (!dptr || bytesize == 0) return CUDA_ERROR_INVALID_VALUE;
    if (!validStream(hStream)) return CUDA_ERROR_INVALID_HANDLE;
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (CUstream cs = capturingStream(hStream)) return captureMemAlloc(s, cs, bytesize, *dptr);
    if (!pool) pool = s.slot->currentPool;
    if (pool->device != s.slot->ordinal) return CUDA_ERROR_INVALID_VALUE;
    if (pool->props.maxSize && pool->usedBytes + bytesize > pool->props.maxSize) return CUDA_ERROR_OUT_OF_MEMORY;
    vk::Buffer buf;
    if (VkResult vr = s.dev->allocateBuffer(bytesize, buf)) {
        VULKANIZER_DEBUG("cuMemAllocAsync(%zu) failed: %d", bytesize, vr);
        return toCuResult(vr);
    }
    s.slot->allocations[buf.address] = buf;
    s.slot->allocatedBytes += buf.size;
    s.slot->poolAllocations[buf.address] = pool;
    pool->usedBytes += buf.size;
    pool->usedHigh = std::max(pool->usedHigh, pool->usedBytes);
    *dptr = buf.address;
    return CUDA_SUCCESS;
}

static CUresult freeAsync(CUdeviceptr dptr, CUstream hStream) {
    if (!validStream(hStream)) return CUDA_ERROR_INVALID_HANDLE;
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (dptr && s.slot->graphAllocations.count(dptr)) {
        if (CUstream cs = capturingStream(hStream)) return captureMemFree(cs, dptr);
        if (releaseGraphMemory(*s.slot, dptr)) return CUDA_SUCCESS;
    }
    if (CUresult r = refuseIfCapturing(hStream)) return r;
    if (dptr == 0) return CUDA_SUCCESS;
    auto it = s.slot->allocations.find(dptr);
    if (it == s.slot->allocations.end()) return CUDA_ERROR_INVALID_VALUE;
    s.slot->allocatedBytes -= it->second.size;
    if (auto pool = s.slot->poolAllocations.find(dptr); pool != s.slot->poolAllocations.end()) {
        if (pool->second) pool->second->usedBytes -= it->second.size;
        s.slot->poolAllocations.erase(pool);
    }
    s.dev->freeBufferAfter(it->second, s.dev->currentSerial());
    s.slot->allocations.erase(it);
    return CUDA_SUCCESS;
}

static bool validPool(CUmemoryPool pool) { return validHandle(pool); }

extern "C" {

CUresult CUDAAPI cuMemAllocAsync(CUdeviceptr* dptr, size_t bytesize, CUstream hStream) {
    VULKANIZER_TRACE_API();
    return allocFromPool(dptr, bytesize, nullptr, hStream);
}

CUresult CUDAAPI cuMemAllocAsync_ptsz(CUdeviceptr* dptr, size_t bytesize, CUstream hStream) {
    return cuMemAllocAsync(dptr, bytesize, hStream);
}

CUresult CUDAAPI cuMemAllocFromPoolAsync(CUdeviceptr* dptr, size_t bytesize, CUmemoryPool pool, CUstream hStream) {
    VULKANIZER_TRACE_API();
    if (!validPool(pool)) return CUDA_ERROR_INVALID_VALUE;
    return allocFromPool(dptr, bytesize, pool, hStream);
}

CUresult CUDAAPI cuMemAllocFromPoolAsync_ptsz(CUdeviceptr* dptr, size_t bytesize, CUmemoryPool pool,
                                              CUstream hStream) {
    return cuMemAllocFromPoolAsync(dptr, bytesize, pool, hStream);
}

CUresult CUDAAPI cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream) {
    VULKANIZER_TRACE_API();
    return freeAsync(dptr, hStream);
}

CUresult CUDAAPI cuMemFreeAsync_ptsz(CUdeviceptr dptr, CUstream hStream) { return cuMemFreeAsync(dptr, hStream); }

CUresult CUDAAPI cuMemPoolCreate(CUmemoryPool* pool, const CUmemPoolProps* poolProps) {
    VULKANIZER_TRACE_API();
    if (!pool || !poolProps) return CUDA_ERROR_INVALID_VALUE;
    if (poolProps->allocType != CU_MEM_ALLOCATION_TYPE_PINNED ||
        poolProps->location.type != CU_MEM_LOCATION_TYPE_DEVICE)
        return CUDA_ERROR_NOT_SUPPORTED;
    if (poolProps->handleTypes != CU_MEM_HANDLE_TYPE_NONE) return CUDA_ERROR_NOT_SUPPORTED;  // no IPC
    Driver& d = Driver::get();
    if (!d.initialized()) return CUDA_ERROR_NOT_INITIALIZED;
    if (!d.slot(poolProps->location.id)) return CUDA_ERROR_INVALID_DEVICE;
    auto* p = new CUmemPoolHandle_st();
    p->device = poolProps->location.id;
    p->props = *poolProps;
    d.registerHandle(p);
    *pool = p;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemPoolDestroy(CUmemoryPool pool) {
    VULKANIZER_TRACE_API();
    Driver& d = Driver::get();
    std::lock_guard<std::recursive_mutex> lock(d.mutex());
    if (!validPool(pool)) return CUDA_ERROR_INVALID_VALUE;
    DeviceSlot* slot = d.slot(pool->device);
    if (pool == &slot->defaultPool) return CUDA_ERROR_INVALID_VALUE;
    if (slot->currentPool == pool) slot->currentPool = &slot->defaultPool;
    // Outstanding allocations stay valid until they are freed.
    for (auto& [ptr, owner] : slot->poolAllocations)
        if (owner == pool) owner = nullptr;
    d.unregisterHandle(pool);
    delete pool;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemPoolTrimTo(CUmemoryPool pool, size_t minBytesToKeep) {
    VULKANIZER_TRACE_API();
    return validPool(pool) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;  // nothing cached
}

CUresult CUDAAPI cuMemPoolSetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void* value) {
    VULKANIZER_TRACE_API();
    if (!validPool(pool) || !value) return CUDA_ERROR_INVALID_VALUE;
    std::lock_guard<std::recursive_mutex> lock(Driver::get().mutex());
    switch (attr) {
        case CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES:
        case CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC:
        case CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES:
            pool->reuseFlags[attr - CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES] = *static_cast<int*>(value);
            return CUDA_SUCCESS;
        case CU_MEMPOOL_ATTR_RELEASE_THRESHOLD: pool->releaseThreshold = *static_cast<cuuint64_t*>(value); return CUDA_SUCCESS;
        case CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH:  // only resettable to 0
        case CU_MEMPOOL_ATTR_USED_MEM_HIGH:
            if (*static_cast<cuuint64_t*>(value) != 0) return CUDA_ERROR_INVALID_VALUE;
            pool->usedHigh = pool->usedBytes;
            return CUDA_SUCCESS;
        default: return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI cuMemPoolGetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void* value) {
    VULKANIZER_TRACE_API();
    if (!validPool(pool) || !value) return CUDA_ERROR_INVALID_VALUE;
    std::lock_guard<std::recursive_mutex> lock(Driver::get().mutex());
    auto set64 = [&](uint64_t v) { *static_cast<cuuint64_t*>(value) = v; };
    switch (attr) {
        case CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES:
        case CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC:
        case CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES:
            *static_cast<int*>(value) = pool->reuseFlags[attr - CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES];
            return CUDA_SUCCESS;
        case CU_MEMPOOL_ATTR_RELEASE_THRESHOLD: set64(pool->releaseThreshold); return CUDA_SUCCESS;
        // Nothing is cached: the reserved memory is the used memory.
        case CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT:
        case CU_MEMPOOL_ATTR_USED_MEM_CURRENT: set64(pool->usedBytes); return CUDA_SUCCESS;
        case CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH:
        case CU_MEMPOOL_ATTR_USED_MEM_HIGH: set64(pool->usedHigh); return CUDA_SUCCESS;
        default: return CUDA_ERROR_INVALID_VALUE;
    }
}

// Every device can access its own pools; there is no peer access.
CUresult CUDAAPI cuMemPoolSetAccess(CUmemoryPool pool, const CUmemAccessDesc* map, size_t count) {
    VULKANIZER_TRACE_API();
    if (!validPool(pool) || (count && !map)) return CUDA_ERROR_INVALID_VALUE;
    for (size_t i = 0; i < count; ++i) {
        if (map[i].location.type != CU_MEM_LOCATION_TYPE_DEVICE) return CUDA_ERROR_INVALID_VALUE;
        if (map[i].location.id != pool->device && map[i].flags != CU_MEM_ACCESS_FLAGS_PROT_NONE)
            return CUDA_ERROR_NOT_SUPPORTED;
    }
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemPoolGetAccess(CUmemAccess_flags* flags, CUmemoryPool memPool, CUmemLocation* location) {
    VULKANIZER_TRACE_API();
    if (!flags || !location || !validPool(memPool)) return CUDA_ERROR_INVALID_VALUE;
    *flags = location->type == CU_MEM_LOCATION_TYPE_DEVICE && location->id == memPool->device
                 ? CU_MEM_ACCESS_FLAGS_PROT_READWRITE
                 : CU_MEM_ACCESS_FLAGS_PROT_NONE;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetDefaultMemPool(CUmemoryPool* pool_out, CUdevice dev) {
    VULKANIZER_TRACE_API();
    if (!pool_out) return CUDA_ERROR_INVALID_VALUE;
    Driver& d = Driver::get();
    if (!d.initialized()) return CUDA_ERROR_NOT_INITIALIZED;
    DeviceSlot* slot = d.slot(dev);
    if (!slot) return CUDA_ERROR_INVALID_DEVICE;
    *pool_out = &slot->defaultPool;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetMemPool(CUmemoryPool* pool, CUdevice dev) {
    VULKANIZER_TRACE_API();
    if (!pool) return CUDA_ERROR_INVALID_VALUE;
    Driver& d = Driver::get();
    if (!d.initialized()) return CUDA_ERROR_NOT_INITIALIZED;
    std::lock_guard<std::recursive_mutex> lock(d.mutex());
    DeviceSlot* slot = d.slot(dev);
    if (!slot) return CUDA_ERROR_INVALID_DEVICE;
    *pool = slot->currentPool;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceSetMemPool(CUdevice dev, CUmemoryPool pool) {
    VULKANIZER_TRACE_API();
    Driver& d = Driver::get();
    if (!d.initialized()) return CUDA_ERROR_NOT_INITIALIZED;
    std::lock_guard<std::recursive_mutex> lock(d.mutex());
    DeviceSlot* slot = d.slot(dev);
    if (!slot) return CUDA_ERROR_INVALID_DEVICE;
    if (!validPool(pool) || pool->device != dev) return CUDA_ERROR_INVALID_VALUE;
    slot->currentPool = pool;
    return CUDA_SUCCESS;
}

}  // extern "C"
