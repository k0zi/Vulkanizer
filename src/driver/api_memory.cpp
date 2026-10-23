// Device memory: allocation, copies and memsets. Everything runs in order on the single device
// queue; the *Async variants differ only in not waiting for copies into host memory the device
// accesses (pinned, registered, managed).

#include "driver/driver.h"

#include "common/log.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <map>
#include <vector>

using namespace vulkanizer;
using namespace vulkanizer::driver;

// Copies and memsets into pinned or managed memory are synchronous in CUDA: the host may read
// the result right away. Waits for the work when `dst` is such memory.
static CUresult finishIfHostTarget(vk::Device& dev, const std::vector<AllocPiece>& dst) {
    for (const AllocPiece& p : dst)
        if (p.ref.buffer->hostAddressed) return toCuResult(dev.finish());
    return CUDA_SUCCESS;
}

static CUresult memset32(CUdeviceptr dst, uint32_t pattern, size_t bytes, size_t elemSize) {
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (bytes == 0) return CUDA_SUCCESS;
    std::vector<AllocPiece> pieces;
    if (!findPieces(*s.slot, dst, bytes, pieces)) return CUDA_ERROR_INVALID_VALUE;
    if (dst % elemSize != 0) return CUDA_ERROR_INVALID_VALUE;
    // Piece boundaries are page aligned, so the pattern stays in phase. The 4-byte aligned middle
    // is filled on the device; an unaligned head or tail is written from the host.
    auto patternBytes = [&](VkDeviceSize offset, size_t n) {
        std::vector<uint8_t> v(n);
        for (size_t i = 0; i < n; ++i) v[i] = static_cast<uint8_t>(pattern >> (8 * ((offset + i) % 4)));
        return v;
    };
    for (const AllocPiece& piece : pieces) {
        const AllocRef& ref = piece.ref;
        const size_t head = std::min<size_t>(piece.size, (4 - ref.offset % 4) % 4);
        const size_t middle = (piece.size - head) / 4 * 4, tail = piece.size - head - middle;
        if (head) {
            if (VkResult vr = s.dev->write(*ref.buffer, ref.offset, patternBytes(ref.offset, head).data(), head))
                return toCuResult(vr);
        }
        if (middle) {
            if (VkResult vr = s.dev->fill(*ref.buffer, ref.offset + head, middle, pattern)) return toCuResult(vr);
        }
        if (tail) {
            const VkDeviceSize at = ref.offset + head + middle;
            if (VkResult vr = s.dev->write(*ref.buffer, at, patternBytes(at, tail).data(), tail)) return toCuResult(vr);
        }
    }
    return finishIfHostTarget(*s.dev, pieces);
}

static CUresult writePieces(vk::Device& dev, const std::vector<AllocPiece>& dst, const void* src) {
    for (const AllocPiece& p : dst)
        if (VkResult vr = dev.write(*p.ref.buffer, p.ref.offset, static_cast<const char*>(src) + p.at, p.size))
            return toCuResult(vr);
    return CUDA_SUCCESS;
}

static CUresult readPieces(vk::Device& dev, const std::vector<AllocPiece>& src, void* dst) {
    for (const AllocPiece& p : src)
        if (VkResult vr = dev.read(*p.ref.buffer, p.ref.offset, static_cast<char*>(dst) + p.at, p.size))
            return toCuResult(vr);
    return CUDA_SUCCESS;
}

// Copies between two piece lists of the same total size, one overlap at a time. Either side may be
// host memory the device accesses (pinned, registered, managed): a synchronous copy then waits
// for the GPU, as in CUDA, an asynchronous one stays in queue order.
static CUresult copyPieces(vk::Device& dev, const std::vector<AllocPiece>& src, const std::vector<AllocPiece>& dst,
                           bool sync = true) {
    size_t i = 0, j = 0, at = 0;
    while (i < src.size() && j < dst.size()) {
        const AllocPiece &a = src[i], &b = dst[j];
        const size_t n = std::min(a.at + a.size, b.at + b.size) - at;
        if (VkResult vr = dev.copy(*a.ref.buffer, a.ref.offset + (at - a.at), *b.ref.buffer, b.ref.offset + (at - b.at), n))
            return toCuResult(vr);
        at += n;
        if (at == a.at + a.size) ++i;
        if (at == b.at + b.size) ++j;
    }
    if (!sync) return CUDA_SUCCESS;
    if (CUresult r = finishIfHostTarget(dev, src)) return r;
    return finishIfHostTarget(dev, dst);
}

// Unified addressing: each side is device memory if it falls inside an allocation (device memory,
// or host memory the device accesses), other host memory otherwise. `dstDevice` and `srcDevice`
// require that side to be device memory (cuMemcpyHtoD and the like).
static CUresult copyUnified(ApiScope& s, CUdeviceptr dst, CUdeviceptr src, size_t bytes, bool sync,
                            bool dstDevice = false, bool srcDevice = false) {
    if (bytes == 0) return CUDA_SUCCESS;
    if (!dst || !src) return CUDA_ERROR_INVALID_VALUE;
    std::vector<AllocPiece> d, sr;
    const bool dstDev = findPieces(*s.slot, dst, bytes, d);
    const bool srcDev = findPieces(*s.slot, src, bytes, sr);
    if ((dstDevice && !dstDev) || (srcDevice && !srcDev)) return CUDA_ERROR_INVALID_VALUE;
    if (dstDev && srcDev) return copyPieces(*s.dev, sr, d, sync);
    if (dstDev) return writePieces(*s.dev, d, reinterpret_cast<const void*>(src));
    if (srcDev) return readPieces(*s.dev, sr, reinterpret_cast<void*>(dst));
    std::memmove(reinterpret_cast<void*>(dst), reinterpret_cast<const void*>(src), bytes);
    return CUDA_SUCCESS;
}

extern "C" {

CUresult CUDAAPI cuMemGetInfo_v2(size_t* free, size_t* total) {
    VULKANIZER_TRACE_API();
    if (!free || !total) return CUDA_ERROR_INVALID_VALUE;
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    // TODO: VK_EXT_memory_budget for real numbers.
    const size_t capacity = static_cast<size_t>(s.slot->info->deviceLocalBytes);
    *total = capacity;
    *free = capacity > s.slot->allocatedBytes ? capacity - s.slot->allocatedBytes : 0;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemAlloc_v2(CUdeviceptr* dptr, size_t bytesize) {
    VULKANIZER_TRACE_API();
    if (!dptr || bytesize == 0) return CUDA_ERROR_INVALID_VALUE;
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    vk::Buffer buf;
    VkResult vr = s.dev->allocateBuffer(bytesize, buf);
    if (vr != VK_SUCCESS) {
        VULKANIZER_DEBUG("cuMemAlloc(%zu) failed: %d", bytesize, vr);
        return toCuResult(vr);
    }
    if (buf.address % 256 != 0) VULKANIZER_WARN_ONCE("device address is not 256-byte aligned as CUDA guarantees");
    s.slot->allocations[buf.address] = buf;
    s.slot->allocatedBytes += bytesize;
    *dptr = buf.address;
    VULKANIZER_DEBUG("cuMemAlloc(%zu) -> 0x%llx%s", bytesize, static_cast<unsigned long long>(buf.address),
                buf.mapped ? " (host-mapped)" : "");
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemFree_v2(CUdeviceptr dptr) {
    VULKANIZER_TRACE_API();
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (dptr == 0) return CUDA_SUCCESS;
    if (CUresult r = freeManaged(s, dptr); r != CUDA_ERROR_NOT_FOUND) return r;  // cuMemAllocManaged memory
    auto it = s.slot->allocations.find(dptr);
    if (it == s.slot->allocations.end()) return CUDA_ERROR_INVALID_VALUE;
    s.slot->allocatedBytes -= it->second.size;
    if (auto pool = s.slot->poolAllocations.find(dptr); pool != s.slot->poolAllocations.end()) {
        if (pool->second) pool->second->usedBytes -= it->second.size;
        s.slot->poolAllocations.erase(pool);
    }
    s.dev->freeBuffer(it->second);
    s.slot->allocations.erase(it);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemGetAddressRange_v2(CUdeviceptr* pbase, size_t* psize, CUdeviceptr dptr) {
    VULKANIZER_TRACE_API();
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    auto it = s.slot->allocations.upper_bound(dptr);
    if (it != s.slot->allocations.begin() && dptr - std::prev(it)->first < std::prev(it)->second.size) {
        if (pbase) *pbase = std::prev(it)->first;
        if (psize) *psize = static_cast<size_t>(std::prev(it)->second.size);
        return CUDA_SUCCESS;
    }
    auto m = s.slot->mappings.upper_bound(dptr);  // virtual memory: the mapping containing it
    if (m != s.slot->mappings.begin() && dptr - std::prev(m)->first < std::prev(m)->second.size) {
        if (pbase) *pbase = std::prev(m)->first;
        if (psize) *psize = std::prev(m)->second.size;
        return CUDA_SUCCESS;
    }
    return CUDA_ERROR_NOT_FOUND;
}

// The synchronous copies return once the copy is done; the asynchronous ones only order it on the
// device queue when both sides are device-accessible (otherwise the host copies right away).
static CUresult memcpyEntry(CUdeviceptr dst, CUdeviceptr src, size_t bytes, bool sync, bool dstDevice, bool srcDevice) {
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    return copyUnified(s, dst, src, bytes, sync, dstDevice, srcDevice);
}

CUresult CUDAAPI cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount) {
    VULKANIZER_TRACE_API();
    return memcpyEntry(dstDevice, reinterpret_cast<CUdeviceptr>(srcHost), ByteCount, true, true, false);
}

CUresult CUDAAPI cuMemcpyDtoH_v2(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
    VULKANIZER_TRACE_API();
    return memcpyEntry(reinterpret_cast<CUdeviceptr>(dstHost), srcDevice, ByteCount, true, false, true);
}

CUresult CUDAAPI cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) {
    VULKANIZER_TRACE_API();
    return memcpyEntry(dstDevice, srcDevice, ByteCount, true, true, true);
}

CUresult CUDAAPI cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount) {
    VULKANIZER_TRACE_API();
    return memcpyEntry(dst, src, ByteCount, true, false, false);
}

// A captured copy becomes a memcpy node, which repeats cuMemcpy on replay.
static CUresult captureCopy(CUstream hStream, CUdeviceptr dst, CUdeviceptr src, size_t bytes) {
    auto node = std::make_unique<CUgraphNode_st>();
    node->type = CU_GRAPH_NODE_TYPE_MEMCPY;
    node->dst = dst;
    node->src = src;
    node->bytes = bytes;
    return captureNode(hStream, std::move(node));
}

static CUresult memcpyAsyncEntry(CUdeviceptr dst, CUdeviceptr src, size_t bytes, CUstream hStream, bool dstDevice,
                                 bool srcDevice) {
    if (!validStream(hStream)) return CUDA_ERROR_INVALID_HANDLE;
    if (CUstream cs = capturingStream(hStream)) return captureCopy(cs, dst, src, bytes);
    return memcpyEntry(dst, src, bytes, false, dstDevice, srcDevice);
}

CUresult CUDAAPI cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream) {
    VULKANIZER_TRACE_API();
    return memcpyAsyncEntry(dst, src, ByteCount, hStream, false, false);
}

CUresult CUDAAPI cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream hStream) {
    VULKANIZER_TRACE_API();
    return memcpyAsyncEntry(dstDevice, reinterpret_cast<CUdeviceptr>(srcHost), ByteCount, hStream, true, false);
}

CUresult CUDAAPI cuMemcpyDtoHAsync_v2(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) {
    VULKANIZER_TRACE_API();
    return memcpyAsyncEntry(reinterpret_cast<CUdeviceptr>(dstHost), srcDevice, ByteCount, hStream, false, true);
}

CUresult CUDAAPI cuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount,
                                      CUstream hStream) {
    VULKANIZER_TRACE_API();
    return memcpyAsyncEntry(dstDevice, srcDevice, ByteCount, hStream, true, true);
}

// Between devices (multi-GPU): through host memory, as separate Vulkan devices share no memory.
// The read waits for the source device's work so far; the write is queued on the destination
// device. Within one device it is an ordinary device copy.
CUresult CUDAAPI cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext,
                              size_t ByteCount) {
    VULKANIZER_TRACE_API();
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!validHandle(dstContext) || !validHandle(srcContext)) return CUDA_ERROR_INVALID_CONTEXT;
    if (ByteCount == 0) return CUDA_SUCCESS;
    DeviceSlot* dstSlot = Driver::get().slot(dstContext->device);
    DeviceSlot* srcSlot = Driver::get().slot(srcContext->device);
    std::vector<AllocPiece> dst, src;
    if (!findPieces(*dstSlot, dstDevice, ByteCount, dst) || !findPieces(*srcSlot, srcDevice, ByteCount, src))
        return CUDA_ERROR_INVALID_VALUE;
    if (dstSlot == srcSlot) return copyPieces(*dstSlot->vk, src, dst);
    std::vector<char> staging(ByteCount);
    if (CUresult r = readPieces(*srcSlot->vk, src, staging.data())) return r;
    return writePieces(*dstSlot->vk, dst, staging.data());
}

CUresult CUDAAPI cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice,
                                   CUcontext srcContext, size_t ByteCount, CUstream hStream) {
    if (!validStream(hStream)) return CUDA_ERROR_INVALID_HANDLE;
    if (capturingStream(hStream)) return CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED;
    return cuMemcpyPeer(dstDevice, dstContext, srcDevice, srcContext, ByteCount);
}

CUresult CUDAAPI cuMemcpyPeerAsync_ptsz(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice,
                                        CUcontext srcContext, size_t ByteCount, CUstream hStream) {
    return cuMemcpyPeerAsync(dstDevice, dstContext, srcDevice, srcContext, ByteCount, hStream);
}

CUresult CUDAAPI cuDeviceCanAccessPeer(int* canAccessPeer, CUdevice dev, CUdevice peerDev) {
    VULKANIZER_TRACE_API();
    if (!canAccessPeer) return CUDA_ERROR_INVALID_VALUE;
    if (!Driver::get().slot(dev) || !Driver::get().slot(peerDev)) return CUDA_ERROR_INVALID_DEVICE;
    *canAccessPeer = 0;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxEnablePeerAccess(CUcontext peerContext, unsigned int Flags) {
    VULKANIZER_TRACE_API();
    if (!validHandle(peerContext) || Flags) return validHandle(peerContext) ? CUDA_ERROR_INVALID_VALUE : CUDA_ERROR_INVALID_CONTEXT;
    return CUDA_ERROR_PEER_ACCESS_UNSUPPORTED;
}

CUresult CUDAAPI cuCtxDisablePeerAccess(CUcontext peerContext) {
    VULKANIZER_TRACE_API();
    return validHandle(peerContext) ? CUDA_ERROR_PEER_ACCESS_NOT_ENABLED : CUDA_ERROR_INVALID_CONTEXT;
}

CUresult CUDAAPI cuMemsetD8_v2(CUdeviceptr dstDevice, unsigned char uc, size_t N) {
    VULKANIZER_TRACE_API();
    return memset32(dstDevice, uc * 0x01010101u, N, 1);
}

CUresult CUDAAPI cuMemsetD16_v2(CUdeviceptr dstDevice, unsigned short us, size_t N) {
    VULKANIZER_TRACE_API();
    return memset32(dstDevice, us * 0x00010001u, N * 2, 2);
}

CUresult CUDAAPI cuMemsetD32_v2(CUdeviceptr dstDevice, unsigned int ui, size_t N) {
    VULKANIZER_TRACE_API();
    return memset32(dstDevice, ui, N * 4, 4);
}

// A captured memset becomes a memset node (one row of N elements).
static CUresult captureMemset(CUstream hStream, CUdeviceptr dst, unsigned value, unsigned elementSize, size_t n) {
    auto node = std::make_unique<CUgraphNode_st>();
    node->type = CU_GRAPH_NODE_TYPE_MEMSET;
    node->memset = {dst, n * elementSize, value, elementSize, n, 1};
    return captureNode(hStream, std::move(node));
}

CUresult CUDAAPI cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream) {
    if (!validStream(hStream)) return CUDA_ERROR_INVALID_HANDLE;
    if (CUstream cs = capturingStream(hStream)) return captureMemset(cs, dstDevice, uc, 1, N);
    return cuMemsetD8_v2(dstDevice, uc, N);
}

CUresult CUDAAPI cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream) {
    if (!validStream(hStream)) return CUDA_ERROR_INVALID_HANDLE;
    if (CUstream cs = capturingStream(hStream)) return captureMemset(cs, dstDevice, us, 2, N);
    return cuMemsetD16_v2(dstDevice, us, N);
}

CUresult CUDAAPI cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream) {
    if (!validStream(hStream)) return CUDA_ERROR_INVALID_HANDLE;
    if (CUstream cs = capturingStream(hStream)) return captureMemset(cs, dstDevice, ui, 4, N);
    return cuMemsetD32_v2(dstDevice, ui, N);
}

// Pointer attributes. Device memory: plain and stream-ordered allocations, and mapped virtual
// memory; host memory: cuMemAllocHost allocations (not device-accessible, so no device pointer).
namespace {

struct PointerInfo {
    bool device = false, host = false, managed = false;
    CUdeviceptr base = 0;  // of the allocation or mapping
    size_t size = 0;
    CUmemoryPool pool = nullptr;
};

PointerInfo pointerInfo(DeviceSlot& slot, CUdeviceptr ptr) {
    PointerInfo info;
    HostMemoryInfo host;
    if (findHostMemory(ptr, host)) {  // pinned, registered or managed: one pointer for both sides
        info.host = true;
        info.device = host.deviceAccessible;
        info.managed = host.managed;
        info.base = host.base;
        info.size = host.size;
        return info;
    }
    auto it = slot.allocations.upper_bound(ptr);
    if (it != slot.allocations.begin() && ptr - std::prev(it)->first < std::prev(it)->second.size) {
        info.device = true;
        info.base = std::prev(it)->first;
        info.size = std::prev(it)->second.size;
        if (auto p = slot.poolAllocations.find(info.base); p != slot.poolAllocations.end()) info.pool = p->second;
        return info;
    }
    auto m = slot.mappings.upper_bound(ptr);
    if (m != slot.mappings.begin() && ptr - std::prev(m)->first < std::prev(m)->second.size) {
        info.device = true;
        info.base = std::prev(m)->first;
        info.size = std::prev(m)->second.size;
        return info;
    }
    return info;
}

// Writes one attribute; false if the pointer does not have it.
bool pointerAttribute(const ApiScope& s, const PointerInfo& info, CUdeviceptr ptr, CUpointer_attribute attribute,
                      void* data) {
    auto put = [&](auto v) { std::memcpy(data, &v, sizeof v); };
    switch (attribute) {
        case CU_POINTER_ATTRIBUTE_CONTEXT: put(s.ctx); return true;
        case CU_POINTER_ATTRIBUTE_MEMORY_TYPE:
            // managed memory reports the device, as on CUDA (IS_MANAGED tells it apart)
            put(unsigned(info.host && !info.managed ? CU_MEMORYTYPE_HOST : CU_MEMORYTYPE_DEVICE));
            return true;
        case CU_POINTER_ATTRIBUTE_DEVICE_POINTER:
            if (!info.device) return false;
            put(ptr);
            return true;
        case CU_POINTER_ATTRIBUTE_HOST_POINTER:
            if (!info.host) return false;
            put(reinterpret_cast<void*>(ptr));
            return true;
        case CU_POINTER_ATTRIBUTE_SYNC_MEMOPS: put(true); return true;
        case CU_POINTER_ATTRIBUTE_BUFFER_ID:
        case CU_POINTER_ATTRIBUTE_MEMORY_BLOCK_ID: put(static_cast<unsigned long long>(info.base)); return true;
        case CU_POINTER_ATTRIBUTE_IS_MANAGED: put(info.managed); return true;
        case CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE:
        case CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE: put(false); return true;
        case CU_POINTER_ATTRIBUTE_MAPPED: put(true); return true;
        case CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL: put(int(s.slot->ordinal)); return true;
        case CU_POINTER_ATTRIBUTE_RANGE_START_ADDR:
        case CU_POINTER_ATTRIBUTE_MAPPING_BASE_ADDR: put(info.base); return true;
        case CU_POINTER_ATTRIBUTE_RANGE_SIZE:
        case CU_POINTER_ATTRIBUTE_MAPPING_SIZE: put(info.size); return true;
        case CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES: put(0ull); return true;
        case CU_POINTER_ATTRIBUTE_ACCESS_FLAGS: put(3u); return true;  // read-write
        case CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE: put(info.pool); return true;
        default: return false;
    }
}

}  // namespace

CUresult CUDAAPI cuPointerGetAttribute(void* data, CUpointer_attribute attribute, CUdeviceptr ptr) {
    VULKANIZER_TRACE_API();
    if (!data) return CUDA_ERROR_INVALID_VALUE;
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    const PointerInfo info = pointerInfo(*s.slot, ptr);
    if (!info.device && !info.host) return CUDA_ERROR_INVALID_VALUE;  // unregistered host memory
    return pointerAttribute(s, info, ptr, attribute, data) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

// Unknown pointers are not an error here: their memory type reads 0.
CUresult CUDAAPI cuPointerGetAttributes(unsigned int numAttributes, CUpointer_attribute* attributes, void** data,
                                        CUdeviceptr ptr) {
    VULKANIZER_TRACE_API();
    if (!attributes || !data) return CUDA_ERROR_INVALID_VALUE;
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    const PointerInfo info = pointerInfo(*s.slot, ptr);
    for (unsigned int i = 0; i < numAttributes; ++i) {
        if (!data[i]) continue;
        if (!info.device && !info.host) {
            if (attributes[i] == CU_POINTER_ATTRIBUTE_MEMORY_TYPE || attributes[i] == CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL)
                std::memset(data[i], attributes[i] == CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL ? 0xff : 0, sizeof(int));
            else if (attributes[i] == CU_POINTER_ATTRIBUTE_CONTEXT || attributes[i] == CU_POINTER_ATTRIBUTE_DEVICE_POINTER ||
                     attributes[i] == CU_POINTER_ATTRIBUTE_HOST_POINTER)
                std::memset(data[i], 0, sizeof(void*));
            continue;
        }
        if (!pointerAttribute(s, info, ptr, attributes[i], data[i])) {
            if (attributes[i] == CU_POINTER_ATTRIBUTE_DEVICE_POINTER || attributes[i] == CU_POINTER_ATTRIBUTE_HOST_POINTER)
                std::memset(data[i], 0, sizeof(void*));
        }
    }
    return CUDA_SUCCESS;
}

}  // extern "C"
