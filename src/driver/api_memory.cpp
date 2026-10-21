// Device memory: allocation, copies and memsets. All operations are synchronous
// for now, so the *Async variants simply run in order on the calling thread.

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

static std::map<uintptr_t, size_t>& hostAllocations() {  // cuMemAllocHost: address -> size
    static auto* map = new std::map<uintptr_t, size_t>();
    return *map;
}

// The cuMemAllocHost allocation containing p (caller holds the driver lock).
static bool findHostAllocation(uintptr_t p, uintptr_t& base, size_t& size) {
    auto it = hostAllocations().upper_bound(p);
    if (it == hostAllocations().begin() || p - std::prev(it)->first >= std::prev(it)->second) return false;
    base = std::prev(it)->first;
    size = std::prev(it)->second;
    return true;
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
    return CUDA_SUCCESS;
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

// Copies between two piece lists of the same total size, one overlap at a time.
static CUresult copyPieces(vk::Device& dev, const std::vector<AllocPiece>& src, const std::vector<AllocPiece>& dst) {
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

// TODO: real pinned memory that the device can access (VK_EXT_external_memory_host).
CUresult CUDAAPI cuMemAllocHost_v2(void** pp, size_t bytesize) {
    VULKANIZER_TRACE_API();
    if (!pp || bytesize == 0) return CUDA_ERROR_INVALID_VALUE;
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    void* p = std::aligned_alloc(4096, (bytesize + 4095) / 4096 * 4096);
    if (!p) return CUDA_ERROR_OUT_OF_MEMORY;
    hostAllocations()[reinterpret_cast<uintptr_t>(p)] = bytesize;
    *pp = p;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemFreeHost(void* p) {
    VULKANIZER_TRACE_API();
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!hostAllocations().erase(reinterpret_cast<uintptr_t>(p))) return CUDA_ERROR_INVALID_VALUE;
    std::free(p);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount) {
    VULKANIZER_TRACE_API();
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (ByteCount == 0) return CUDA_SUCCESS;
    if (!srcHost) return CUDA_ERROR_INVALID_VALUE;
    std::vector<AllocPiece> dst;
    if (!findPieces(*s.slot, dstDevice, ByteCount, dst)) return CUDA_ERROR_INVALID_VALUE;
    return writePieces(*s.dev, dst, srcHost);
}

CUresult CUDAAPI cuMemcpyDtoH_v2(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
    VULKANIZER_TRACE_API();
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (ByteCount == 0) return CUDA_SUCCESS;
    if (!dstHost) return CUDA_ERROR_INVALID_VALUE;
    std::vector<AllocPiece> src;
    if (!findPieces(*s.slot, srcDevice, ByteCount, src)) return CUDA_ERROR_INVALID_VALUE;
    return readPieces(*s.dev, src, dstHost);
}

CUresult CUDAAPI cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) {
    VULKANIZER_TRACE_API();
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (ByteCount == 0) return CUDA_SUCCESS;
    std::vector<AllocPiece> src, dst;
    if (!findPieces(*s.slot, srcDevice, ByteCount, src) || !findPieces(*s.slot, dstDevice, ByteCount, dst))
        return CUDA_ERROR_INVALID_VALUE;
    return copyPieces(*s.dev, src, dst);
}

// Unified addressing: each side is a device pointer if it falls inside an allocation, a host pointer otherwise.
CUresult CUDAAPI cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount) {
    VULKANIZER_TRACE_API();
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (ByteCount == 0) return CUDA_SUCCESS;
    std::vector<AllocPiece> d, sr;
    const bool dstDev = findPieces(*s.slot, dst, ByteCount, d);
    const bool srcDev = findPieces(*s.slot, src, ByteCount, sr);
    if (dstDev && srcDev) return copyPieces(*s.dev, sr, d);
    if (dstDev) return writePieces(*s.dev, d, reinterpret_cast<const void*>(src));
    if (srcDev) return readPieces(*s.dev, sr, reinterpret_cast<void*>(dst));
    std::memmove(reinterpret_cast<void*>(dst), reinterpret_cast<const void*>(src), ByteCount);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream) {
    if (!validStream(hStream)) return CUDA_ERROR_INVALID_HANDLE;
    return cuMemcpy(dst, src, ByteCount);
}

CUresult CUDAAPI cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream hStream) {
    if (!validStream(hStream)) return CUDA_ERROR_INVALID_HANDLE;
    return cuMemcpyHtoD_v2(dstDevice, srcHost, ByteCount);
}

CUresult CUDAAPI cuMemcpyDtoHAsync_v2(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) {
    if (!validStream(hStream)) return CUDA_ERROR_INVALID_HANDLE;
    return cuMemcpyDtoH_v2(dstHost, srcDevice, ByteCount);
}

CUresult CUDAAPI cuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount,
                                      CUstream hStream) {
    if (!validStream(hStream)) return CUDA_ERROR_INVALID_HANDLE;
    return cuMemcpyDtoD_v2(dstDevice, srcDevice, ByteCount);
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

CUresult CUDAAPI cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream) {
    if (!validStream(hStream)) return CUDA_ERROR_INVALID_HANDLE;
    return cuMemsetD8_v2(dstDevice, uc, N);
}

CUresult CUDAAPI cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream) {
    if (!validStream(hStream)) return CUDA_ERROR_INVALID_HANDLE;
    return cuMemsetD16_v2(dstDevice, us, N);
}

CUresult CUDAAPI cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream) {
    if (!validStream(hStream)) return CUDA_ERROR_INVALID_HANDLE;
    return cuMemsetD32_v2(dstDevice, ui, N);
}

// Pointer attributes. Device memory: plain and stream-ordered allocations, and mapped virtual
// memory; host memory: cuMemAllocHost allocations (not device-accessible, so no device pointer).
namespace {

struct PointerInfo {
    bool device = false, host = false;
    CUdeviceptr base = 0;  // of the allocation or mapping
    size_t size = 0;
    CUmemoryPool pool = nullptr;
};

PointerInfo pointerInfo(DeviceSlot& slot, CUdeviceptr ptr) {
    PointerInfo info;
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
    uintptr_t base;
    if (findHostAllocation(uintptr_t(ptr), base, info.size)) {
        info.host = true;
        info.base = base;
    }
    return info;
}

// Writes one attribute; false if the pointer does not have it.
bool pointerAttribute(const ApiScope& s, const PointerInfo& info, CUdeviceptr ptr, CUpointer_attribute attribute,
                      void* data) {
    auto put = [&](auto v) { std::memcpy(data, &v, sizeof v); };
    switch (attribute) {
        case CU_POINTER_ATTRIBUTE_CONTEXT: put(s.ctx); return true;
        case CU_POINTER_ATTRIBUTE_MEMORY_TYPE: put(unsigned(info.device ? CU_MEMORYTYPE_DEVICE : CU_MEMORYTYPE_HOST)); return true;
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
        case CU_POINTER_ATTRIBUTE_IS_MANAGED:
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
