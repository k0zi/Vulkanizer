// Device memory: allocation, copies and memsets. All operations are synchronous
// for now, so the *Async variants simply run in order on the calling thread.

#include "driver/driver.h"

#include "common/log.h"

#include <cstdlib>
#include <cstring>
#include <unordered_set>

using namespace vulkanizer;
using namespace vulkanizer::driver;

static std::unordered_set<void*>& hostAllocations() {
    static auto* set = new std::unordered_set<void*>();
    return *set;
}

static CUresult memset32(CUdeviceptr dst, uint32_t pattern, size_t bytes, size_t elemSize) {
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (bytes == 0) return CUDA_SUCCESS;
    AllocRef ref;
    if (!findAllocation(*s.slot, dst, bytes, ref)) return CUDA_ERROR_INVALID_VALUE;
    if (dst % elemSize != 0) return CUDA_ERROR_INVALID_VALUE;
    if (ref.offset % 4 == 0 && bytes % 4 == 0) return toCuResult(s.dev->fill(*ref.buffer, ref.offset, bytes, pattern));
    if (ref.buffer->mapped) {
        // Unaligned head/tail on host-visible memory: write the pattern bytewise.
        if (VkResult vr = s.dev->finish()) return toCuResult(vr);
        auto* p = static_cast<uint8_t*>(ref.buffer->mapped) + ref.offset;
        for (size_t i = 0; i < bytes; ++i) p[i] = static_cast<uint8_t>(pattern >> (8 * ((ref.offset + i) % 4)));
        return CUDA_SUCCESS;
    }
    VULKANIZER_WARN_ONCE("unaligned memset on non-mappable memory is not implemented yet");
    return CUDA_ERROR_NOT_SUPPORTED;
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
    s.dev->freeBuffer(it->second);
    s.slot->allocations.erase(it);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemGetAddressRange_v2(CUdeviceptr* pbase, size_t* psize, CUdeviceptr dptr) {
    VULKANIZER_TRACE_API();
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    AllocRef ref;
    if (!findAllocation(*s.slot, dptr, 1, ref)) return CUDA_ERROR_NOT_FOUND;
    if (pbase) *pbase = ref.base;
    if (psize) *psize = static_cast<size_t>(ref.buffer->size);
    return CUDA_SUCCESS;
}

// TODO: real pinned memory that the device can access (VK_EXT_external_memory_host).
CUresult CUDAAPI cuMemAllocHost_v2(void** pp, size_t bytesize) {
    VULKANIZER_TRACE_API();
    if (!pp || bytesize == 0) return CUDA_ERROR_INVALID_VALUE;
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    void* p = std::aligned_alloc(4096, (bytesize + 4095) / 4096 * 4096);
    if (!p) return CUDA_ERROR_OUT_OF_MEMORY;
    hostAllocations().insert(p);
    *pp = p;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemFreeHost(void* p) {
    VULKANIZER_TRACE_API();
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!hostAllocations().erase(p)) return CUDA_ERROR_INVALID_VALUE;
    std::free(p);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount) {
    VULKANIZER_TRACE_API();
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (ByteCount == 0) return CUDA_SUCCESS;
    if (!srcHost) return CUDA_ERROR_INVALID_VALUE;
    AllocRef dst;
    if (!findAllocation(*s.slot, dstDevice, ByteCount, dst)) return CUDA_ERROR_INVALID_VALUE;
    return toCuResult(s.dev->write(*dst.buffer, dst.offset, srcHost, ByteCount));
}

CUresult CUDAAPI cuMemcpyDtoH_v2(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
    VULKANIZER_TRACE_API();
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (ByteCount == 0) return CUDA_SUCCESS;
    if (!dstHost) return CUDA_ERROR_INVALID_VALUE;
    AllocRef src;
    if (!findAllocation(*s.slot, srcDevice, ByteCount, src)) return CUDA_ERROR_INVALID_VALUE;
    return toCuResult(s.dev->read(*src.buffer, src.offset, dstHost, ByteCount));
}

CUresult CUDAAPI cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) {
    VULKANIZER_TRACE_API();
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (ByteCount == 0) return CUDA_SUCCESS;
    AllocRef src, dst;
    if (!findAllocation(*s.slot, srcDevice, ByteCount, src) || !findAllocation(*s.slot, dstDevice, ByteCount, dst))
        return CUDA_ERROR_INVALID_VALUE;
    return toCuResult(s.dev->copy(*src.buffer, src.offset, *dst.buffer, dst.offset, ByteCount));
}

// Unified addressing: each side is a device pointer if it falls inside an allocation, a host pointer otherwise.
CUresult CUDAAPI cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount) {
    VULKANIZER_TRACE_API();
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (ByteCount == 0) return CUDA_SUCCESS;
    AllocRef d, sr;
    const bool dstDev = findAllocation(*s.slot, dst, ByteCount, d);
    const bool srcDev = findAllocation(*s.slot, src, ByteCount, sr);
    if (dstDev && srcDev) return toCuResult(s.dev->copy(*sr.buffer, sr.offset, *d.buffer, d.offset, ByteCount));
    if (dstDev) return toCuResult(s.dev->write(*d.buffer, d.offset, reinterpret_cast<const void*>(src), ByteCount));
    if (srcDev) return toCuResult(s.dev->read(*sr.buffer, sr.offset, reinterpret_cast<void*>(dst), ByteCount));
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

}  // extern "C"
