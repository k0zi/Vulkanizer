// Host memory the device can use: pinned (cuMemAllocHost/cuMemHostAlloc), registered
// (cuMemHostRegister) and managed (cuMemAllocManaged) memory. All three are host pages imported
// into the device at device address h | kHostAddressBits (vk::Device::importHostMemory), and
// kernels set those bits on every address, so the host pointer h is also the device pointer, as
// with CUDA's unified addressing. The imported range is an ordinary allocation of the slot
// (keyed by h), so copies, memsets and pointer queries treat it like device memory.
//
// Managed memory never migrates: it stays host memory the device reads and writes in place
// (on an integrated GPU that is the same memory anyway). Advice and prefetches are accepted and
// ignored. Registration of pages that are already imported (two registrations sharing a page) and
// devices without the import support fall back to plain host memory the device cannot access.

#include "driver/driver.h"

#include "common/log.h"

#include <sys/mman.h>

#include <map>

using namespace vulkanizer;
using namespace vulkanizer::driver;

namespace {

enum class HostKind { Pinned, Registered, Managed };

struct HostRange {
    HostKind kind = HostKind::Pinned;
    size_t size = 0;            // as the application asked for
    unsigned int flags = 0;
    CUdeviceptr pages = 0;      // the imported, aligned range (its allocation key); 0: not imported
    size_t pagesSize = 0;
    void* mapping = nullptr;    // our own mmap (pinned and managed memory)
    size_t mappingSize = 0;
};

// By host address. The driver lock protects it.
std::map<uintptr_t, HostRange>& hostRanges() {
    static auto* ranges = new std::map<uintptr_t, HostRange>();
    return *ranges;
}

const std::pair<const uintptr_t, HostRange>* findRange(uintptr_t p) {
    auto& ranges = hostRanges();
    auto it = ranges.upper_bound(p);
    if (it == ranges.begin()) return nullptr;
    --it;
    return p - it->first < it->second.size ? &*it : nullptr;
}

size_t importAlignment(const ApiScope& s) {
    const size_t a = s.dev->hostImportAlignment();
    return a ? a : 4096;
}

// Imports the pages around [p, p + size) and records them as an allocation of the slot.
bool importPages(ApiScope& s, uintptr_t p, size_t size, HostRange& r) {
    if (!s.dev->features().hostMemoryImport) return false;
    const size_t align = importAlignment(s);
    const uintptr_t begin = p / align * align, end = (p + size + align - 1) / align * align;
    vk::Buffer buf;
    if (VkResult vr = s.dev->importHostMemory(reinterpret_cast<void*>(begin), end - begin, buf)) {
        VULKANIZER_DEBUG("host memory 0x%llx (%zu bytes) cannot be imported: %d", static_cast<unsigned long long>(begin),
                         size_t(end - begin), vr);
        return false;
    }
    s.slot->allocations[buf.address] = buf;
    r.pages = buf.address;
    r.pagesSize = end - begin;
    return true;
}

void releasePages(ApiScope& s, HostRange& r) {
    if (!r.pages) return;
    auto it = s.slot->allocations.find(r.pages);
    if (it != s.slot->allocations.end()) {
        s.dev->freeBuffer(it->second);  // waits for the work that may still use it
        s.slot->allocations.erase(it);
    }
    r.pages = 0;
}

// Fresh, page-aligned host memory of our own, imported if possible.
CUresult allocateHost(ApiScope& s, size_t bytesize, HostKind kind, unsigned int flags, bool needDevice, void** out) {
    const size_t align = importAlignment(s), mapped = (bytesize + align - 1) / align * align;
    void* p = mmap(nullptr, mapped, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return CUDA_ERROR_OUT_OF_MEMORY;
    HostRange r;
    r.kind = kind;
    r.size = bytesize;
    r.flags = flags;
    r.mapping = p;
    r.mappingSize = mapped;
    if (!importPages(s, uintptr_t(p), bytesize, r) && needDevice) {
        munmap(p, mapped);
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    hostRanges()[uintptr_t(p)] = r;
    *out = p;
    return CUDA_SUCCESS;
}

CUresult freeHost(ApiScope& s, uintptr_t p, HostKind kind) {
    auto& ranges = hostRanges();
    auto it = ranges.find(p);
    if (it == ranges.end() || it->second.kind != kind) return CUDA_ERROR_INVALID_VALUE;
    releasePages(s, it->second);
    if (it->second.mapping) munmap(it->second.mapping, it->second.mappingSize);
    ranges.erase(it);
    return CUDA_SUCCESS;
}

}  // namespace

namespace vulkanizer::driver {

bool findHostMemory(CUdeviceptr ptr, HostMemoryInfo& info) {
    const auto* r = findRange(uintptr_t(ptr));
    if (!r) return false;
    info.base = r->first;
    info.size = r->second.size;
    info.managed = r->second.kind == HostKind::Managed;
    info.deviceAccessible = r->second.pages != 0;
    return true;
}

CUresult freeManaged(ApiScope& s, CUdeviceptr ptr) {
    const auto* r = findRange(uintptr_t(ptr));
    if (!r || r->first != ptr || r->second.kind != HostKind::Managed) return CUDA_ERROR_NOT_FOUND;
    return freeHost(s, uintptr_t(ptr), HostKind::Managed);
}

}  // namespace vulkanizer::driver

extern "C" {

// ---- pinned memory ----------------------------------------------------------------------------

CUresult CUDAAPI cuMemHostAlloc(void** pp, size_t bytesize, unsigned int Flags) {
    VULKANIZER_TRACE_API();
    if (!pp || bytesize == 0) return CUDA_ERROR_INVALID_VALUE;
    if (Flags & ~unsigned(CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP | CU_MEMHOSTALLOC_WRITECOMBINED))
        return CUDA_ERROR_INVALID_VALUE;
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    return allocateHost(s, bytesize, HostKind::Pinned, Flags, Flags & CU_MEMHOSTALLOC_DEVICEMAP, pp);
}

CUresult CUDAAPI cuMemAllocHost_v2(void** pp, size_t bytesize) { return cuMemHostAlloc(pp, bytesize, 0); }

CUresult CUDAAPI cuMemFreeHost(void* p) {
    VULKANIZER_TRACE_API();
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    return freeHost(s, uintptr_t(p), HostKind::Pinned);
}

CUresult CUDAAPI cuMemHostGetFlags(unsigned int* pFlags, void* p) {
    if (!pFlags) return CUDA_ERROR_INVALID_VALUE;
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    const auto* r = findRange(uintptr_t(p));
    if (!r || r->second.kind != HostKind::Pinned) return CUDA_ERROR_INVALID_VALUE;
    *pFlags = r->second.flags | (r->second.pages ? CU_MEMHOSTALLOC_DEVICEMAP : 0u);
    return CUDA_SUCCESS;
}

// ---- registered memory ------------------------------------------------------------------------

CUresult CUDAAPI cuMemHostRegister_v2(void* p, size_t bytesize, unsigned int Flags) {
    VULKANIZER_TRACE_API();
    if (!p || bytesize == 0) return CUDA_ERROR_INVALID_VALUE;
    if (Flags & ~unsigned(CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP | CU_MEMHOSTREGISTER_IOMEMORY |
                          CU_MEMHOSTREGISTER_READ_ONLY))
        return CUDA_ERROR_INVALID_VALUE;
    if (Flags & CU_MEMHOSTREGISTER_IOMEMORY) return CUDA_ERROR_NOT_SUPPORTED;
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    const uintptr_t begin = uintptr_t(p);
    auto& ranges = hostRanges();
    auto next = ranges.lower_bound(begin);
    if (next != ranges.end() && next->first < begin + bytesize) return CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED;
    if (next != ranges.begin() && std::prev(next)->first + std::prev(next)->second.size > begin)
        return CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED;
    HostRange r;
    r.kind = HostKind::Registered;
    r.size = bytesize;
    r.flags = Flags;
    // With unified addressing CUDA maps every registration; only DEVICEMAP insists on it.
    if (!importPages(s, begin, bytesize, r) && (Flags & CU_MEMHOSTREGISTER_DEVICEMAP)) return CUDA_ERROR_NOT_SUPPORTED;
    ranges[begin] = r;
    return CUDA_SUCCESS;
}

// The pre-CUDA 11 entry point (the header maps the name to _v2).
#undef cuMemHostRegister
CUresult CUDAAPI cuMemHostRegister(void* p, size_t bytesize, unsigned int Flags);
CUresult CUDAAPI cuMemHostRegister(void* p, size_t bytesize, unsigned int Flags) {
    return cuMemHostRegister_v2(p, bytesize, Flags);
}

CUresult CUDAAPI cuMemHostUnregister(void* p) {
    VULKANIZER_TRACE_API();
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    return freeHost(s, uintptr_t(p), HostKind::Registered) == CUDA_SUCCESS ? CUDA_SUCCESS
                                                                           : CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED;
}

// Unified addressing: the device pointer of mapped host memory is the host pointer.
CUresult CUDAAPI cuMemHostGetDevicePointer_v2(CUdeviceptr* pdptr, void* p, unsigned int Flags) {
    VULKANIZER_TRACE_API();
    if (!pdptr || !p || Flags != 0) return CUDA_ERROR_INVALID_VALUE;
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    const auto* r = findRange(uintptr_t(p));
    if (!r || !r->second.pages) return CUDA_ERROR_INVALID_VALUE;
    *pdptr = CUdeviceptr(p);
    return CUDA_SUCCESS;
}

// ---- managed memory ---------------------------------------------------------------------------

CUresult CUDAAPI cuMemAllocManaged(CUdeviceptr* dptr, size_t bytesize, unsigned int flags) {
    VULKANIZER_TRACE_API();
    if (!dptr || bytesize == 0 || (flags != CU_MEM_ATTACH_GLOBAL && flags != CU_MEM_ATTACH_HOST))
        return CUDA_ERROR_INVALID_VALUE;
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    void* p = nullptr;
    if (CUresult r = allocateHost(s, bytesize, HostKind::Managed, flags, true, &p)) return r;
    *dptr = CUdeviceptr(p);
    return CUDA_SUCCESS;
}

// Hints and migrations: managed memory stays where it is.
CUresult CUDAAPI cuMemAdvise(CUdeviceptr devPtr, size_t count, CUmem_advise advice, CUdevice device) {
    if (advice < CU_MEM_ADVISE_SET_READ_MOSTLY || advice > CU_MEM_ADVISE_UNSET_ACCESSED_BY) return CUDA_ERROR_INVALID_VALUE;
    return devPtr || !count ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}
CUresult CUDAAPI cuMemAdvise_v2(CUdeviceptr devPtr, size_t count, CUmem_advise advice, CUmemLocation) {
    return cuMemAdvise(devPtr, count, advice, 0);
}
CUresult CUDAAPI cuMemPrefetchAsync(CUdeviceptr devPtr, size_t count, CUdevice, CUstream hStream) {
    if (!validStream(hStream)) return CUDA_ERROR_INVALID_HANDLE;
    return devPtr || !count ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}
CUresult CUDAAPI cuMemPrefetchAsync_ptsz(CUdeviceptr devPtr, size_t count, CUdevice dstDevice, CUstream hStream) {
    return cuMemPrefetchAsync(devPtr, count, dstDevice, hStream);
}
CUresult CUDAAPI cuMemPrefetchAsync_v2(CUdeviceptr devPtr, size_t count, CUmemLocation, unsigned int, CUstream hStream) {
    return cuMemPrefetchAsync(devPtr, count, 0, hStream);
}
CUresult CUDAAPI cuMemPrefetchAsync_v2_ptsz(CUdeviceptr devPtr, size_t count, CUmemLocation location, unsigned int flags,
                                           CUstream hStream) {
    return cuMemPrefetchAsync_v2(devPtr, count, location, flags, hStream);
}
CUresult CUDAAPI cuStreamAttachMemAsync(CUstream hStream, CUdeviceptr, size_t, unsigned int flags) {
    if (!validStream(hStream)) return CUDA_ERROR_INVALID_HANDLE;
    return flags & ~unsigned(CU_MEM_ATTACH_GLOBAL | CU_MEM_ATTACH_HOST | CU_MEM_ATTACH_SINGLE) ? CUDA_ERROR_INVALID_VALUE
                                                                                                : CUDA_SUCCESS;
}
CUresult CUDAAPI cuStreamAttachMemAsync_ptsz(CUstream hStream, CUdeviceptr dptr, size_t length, unsigned int flags) {
    return cuStreamAttachMemAsync(hStream, dptr, length, flags);
}

}  // extern "C"
