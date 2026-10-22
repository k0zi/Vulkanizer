// Initialization, versioning, error strings, entry point lookup and device queries.

#include "driver/driver.h"

#include "common/log.h"

#include <dlfcn.h>

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

using namespace vulkanizer;
using namespace vulkanizer::driver;

// What cuDriverGetVersion reports: new enough for the CUDA 13.4 runtime (libcudart.so.13).
static constexpr int kDriverVersion = 13040;

void vulkanizer::driver::computeCapability(int& major, int& minor) {
    static const int cc = [] {
        const char* env = std::getenv("VULKANIZER_CC");
        int v = env ? std::atoi(env) : 0;
        return v >= 30 && v <= 120 ? v : 75;
    }();
    major = cc / 10;
    minor = cc % 10;
}

extern "C" {

CUresult CUDAAPI cuInit(unsigned int Flags) {
    VULKANIZER_TRACE_API();
    if (Flags != 0) return CUDA_ERROR_INVALID_VALUE;
    return Driver::get().init();
}

CUresult CUDAAPI cuDriverGetVersion(int* driverVersion) {
    VULKANIZER_TRACE_API();
    if (!driverVersion) return CUDA_ERROR_INVALID_VALUE;
    *driverVersion = kDriverVersion;
    return CUDA_SUCCESS;
}

}  // extern "C"

struct ErrorInfo {
    CUresult code;
    const char* name;
    const char* description;
};

static const ErrorInfo kErrors[] = {
    {CUDA_SUCCESS, "CUDA_SUCCESS", "no error"},
    {CUDA_ERROR_INVALID_VALUE, "CUDA_ERROR_INVALID_VALUE", "invalid argument"},
    {CUDA_ERROR_OUT_OF_MEMORY, "CUDA_ERROR_OUT_OF_MEMORY", "out of memory"},
    {CUDA_ERROR_NOT_INITIALIZED, "CUDA_ERROR_NOT_INITIALIZED", "initialization error"},
    {CUDA_ERROR_DEINITIALIZED, "CUDA_ERROR_DEINITIALIZED", "driver shutting down"},
    {CUDA_ERROR_STUB_LIBRARY, "CUDA_ERROR_STUB_LIBRARY", "CUDA driver is a stub library"},
    {CUDA_ERROR_DEVICE_UNAVAILABLE, "CUDA_ERROR_DEVICE_UNAVAILABLE", "CUDA-capable device(s) is/are busy or unavailable"},
    {CUDA_ERROR_NO_DEVICE, "CUDA_ERROR_NO_DEVICE", "no CUDA-capable device is detected"},
    {CUDA_ERROR_INVALID_DEVICE, "CUDA_ERROR_INVALID_DEVICE", "invalid device ordinal"},
    {CUDA_ERROR_INVALID_IMAGE, "CUDA_ERROR_INVALID_IMAGE", "device kernel image is invalid"},
    {CUDA_ERROR_INVALID_CONTEXT, "CUDA_ERROR_INVALID_CONTEXT", "invalid device context"},
    {CUDA_ERROR_CONTEXT_ALREADY_CURRENT, "CUDA_ERROR_CONTEXT_ALREADY_CURRENT", "context already current"},
    {CUDA_ERROR_NO_BINARY_FOR_GPU, "CUDA_ERROR_NO_BINARY_FOR_GPU", "no kernel image is available for execution on the device"},
    {CUDA_ERROR_INVALID_PTX, "CUDA_ERROR_INVALID_PTX", "a PTX JIT compilation failed"},
    {CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY, "CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY",
     "the provided execution affinity is not supported by the device"},
    {CUDA_ERROR_JIT_COMPILER_NOT_FOUND, "CUDA_ERROR_JIT_COMPILER_NOT_FOUND", "PTX JIT compiler library not found"},
    {CUDA_ERROR_UNSUPPORTED_PTX_VERSION, "CUDA_ERROR_UNSUPPORTED_PTX_VERSION", "the provided PTX was compiled with an unsupported toolchain"},
    {CUDA_ERROR_INVALID_SOURCE, "CUDA_ERROR_INVALID_SOURCE", "device kernel image is invalid"},
    {CUDA_ERROR_FILE_NOT_FOUND, "CUDA_ERROR_FILE_NOT_FOUND", "file not found"},
    {CUDA_ERROR_INVALID_HANDLE, "CUDA_ERROR_INVALID_HANDLE", "invalid resource handle"},
    {CUDA_ERROR_ILLEGAL_STATE, "CUDA_ERROR_ILLEGAL_STATE", "the operation cannot be performed in the present state"},
    {CUDA_ERROR_NOT_FOUND, "CUDA_ERROR_NOT_FOUND", "named symbol not found"},
    {CUDA_ERROR_NOT_READY, "CUDA_ERROR_NOT_READY", "device not ready"},
    {CUDA_ERROR_ILLEGAL_ADDRESS, "CUDA_ERROR_ILLEGAL_ADDRESS", "an illegal memory access was encountered"},
    {CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES, "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES", "too many resources requested for launch"},
    {CUDA_ERROR_LAUNCH_TIMEOUT, "CUDA_ERROR_LAUNCH_TIMEOUT", "the launch timed out and was terminated"},
    {CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE, "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE", "primary context is active"},
    {CUDA_ERROR_CONTEXT_IS_DESTROYED, "CUDA_ERROR_CONTEXT_IS_DESTROYED", "context is destroyed"},
    {CUDA_ERROR_ALREADY_MAPPED, "CUDA_ERROR_ALREADY_MAPPED", "resource already mapped"},
    {CUDA_ERROR_NOT_MAPPED, "CUDA_ERROR_NOT_MAPPED", "resource not mapped"},
    {CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED, "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED",
     "part or all of the requested memory range is already mapped"},
    {CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED, "CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED",
     "pointer does not correspond to a registered memory region"},
    {CUDA_ERROR_LAUNCH_FAILED, "CUDA_ERROR_LAUNCH_FAILED", "unspecified launch failure"},
    {CUDA_ERROR_NOT_PERMITTED, "CUDA_ERROR_NOT_PERMITTED", "operation not permitted"},
    {CUDA_ERROR_NOT_SUPPORTED, "CUDA_ERROR_NOT_SUPPORTED", "operation not supported"},
    {CUDA_ERROR_UNKNOWN, "CUDA_ERROR_UNKNOWN", "unknown error"},
};

static const ErrorInfo* findError(CUresult error) {
    for (const ErrorInfo& e : kErrors)
        if (e.code == error) return &e;
    return nullptr;
}

// Entry points whose ABI changed after CUDA 3.2: the variant a caller of that CUDA version
// expects (from NVIDIA's cudaTypedefs.h PFN_<name>_v<version> typedefs), newest first. A
// variant Vulkanizer lacks is "not found", never an older one with a different signature.
struct VersionedEntry {
    const char* symbol;
    int since;
    const char* variant;
};
constexpr VersionedEntry kVersionedEntries[] = {
    {"cuCtxCreate", 12050, "cuCtxCreate_v4"},
    {"cuCtxCreate", 11040, "cuCtxCreate_v3"},
    {"cuCtxCreate", 3020, "cuCtxCreate_v2"},
    {"cuCtxGetDevice", 13000, "cuCtxGetDevice_v2"},
    {"cuCtxSynchronize", 13000, "cuCtxSynchronize_v2"},
    {"cuDeviceGetUuid", 11040, "cuDeviceGetUuid_v2"},
    {"cuEventElapsedTime", 12080, "cuEventElapsedTime_v2"},
    {"cuGetProcAddress", 12000, "cuGetProcAddress_v2"},
    {"cuMemAdvise", 12020, "cuMemAdvise_v2"},
    {"cuMemPrefetchAsync", 12020, "cuMemPrefetchAsync_v2"},
    {"cuStreamGetCaptureInfo", 12030, "cuStreamGetCaptureInfo_v3"},
    {"cuStreamGetCaptureInfo", 11030, "cuStreamGetCaptureInfo_v2"},
    {"cuGraphAddNode", 12030, "cuGraphAddNode_v2"},
    {"cuGraphGetEdges", 12030, "cuGraphGetEdges_v2"},
    {"cuGraphNodeGetDependencies", 12030, "cuGraphNodeGetDependencies_v2"},
    {"cuGraphNodeGetDependentNodes", 12030, "cuGraphNodeGetDependentNodes_v2"},
    {"cuGraphAddDependencies", 12030, "cuGraphAddDependencies_v2"},
    {"cuGraphRemoveDependencies", 12030, "cuGraphRemoveDependencies_v2"},
    {"cuGraphInstantiate", 12000, "cuGraphInstantiateWithFlags"},
    {"cuGraphAddKernelNode", 12000, "cuGraphAddKernelNode_v2"},
    {"cuGraphKernelNodeGetParams", 12000, "cuGraphKernelNodeGetParams_v2"},
    {"cuGraphKernelNodeSetParams", 12000, "cuGraphKernelNodeSetParams_v2"},
    {"cuGraphExecKernelNodeSetParams", 12000, "cuGraphExecKernelNodeSetParams_v2"},
    {"cuGraphExecUpdate", 12000, "cuGraphExecUpdate_v2"},
    {"cuStreamBeginCapture", 10010, "cuStreamBeginCapture_v2"},
    {"cuStreamUpdateCaptureDependencies", 12030, "cuStreamUpdateCaptureDependencies_v2"},
    {"cuLaunchHostFunc", 13010, "cuLaunchHostFunc_v2"},
};

// Resolves a driver entry point by name. Symbols in kVersionedEntries get the variant of the
// caller's CUDA version; for the others, from 3.2 on the "_v2" ABI variant wins when it exists
// (the cuda.h #define mapping). With the per-thread default stream flag, the "_ptsz"/"_ptds"
// variant does (where one exists).
static void* lookupEntryPoint(const char* symbol, int cudaVersion, cuuint64_t flags) {
    static void* self = [] {
        Dl_info info{};
        if (!dladdr(reinterpret_cast<void*>(&cuInit), &info) || !info.dli_fname) return static_cast<void*>(nullptr);
        return dlopen(info.dli_fname, RTLD_NOW | RTLD_NOLOAD);
    }();
    if (!self || std::strncmp(symbol, "cu", 2) != 0) return nullptr;
    std::string name = symbol;
    bool versioned = false;
    for (const VersionedEntry& e : kVersionedEntries) {
        if (std::strcmp(e.symbol, symbol) != 0) continue;
        versioned = true;
        if (cudaVersion >= e.since) {
            name = e.variant;
            break;
        }
    }
    if (!versioned && cudaVersion >= 3020 && dlsym(self, (name + "_v2").c_str())) name += "_v2";
    // the per-thread variant of the chosen ABI (cuStreamBeginCapture_v2_ptsz), where one exists
    if (flags & CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM) {
        for (const char* suffix : {"_ptsz", "_ptds"})
            if (void* p = dlsym(self, (name + suffix).c_str())) return p;
    }
    return dlsym(self, name.c_str());
}

extern "C" {

CUresult CUDAAPI cuGetErrorName(CUresult error, const char** pStr) {
    if (!pStr) return CUDA_ERROR_INVALID_VALUE;
    const ErrorInfo* e = findError(error);
    *pStr = e ? e->name : nullptr;
    return e ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

CUresult CUDAAPI cuGetErrorString(CUresult error, const char** pStr) {
    if (!pStr) return CUDA_ERROR_INVALID_VALUE;
    const ErrorInfo* e = findError(error);
    *pStr = e ? e->description : nullptr;
    return e ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

CUresult CUDAAPI cuGetProcAddress_v2(const char* symbol, void** pfn, int cudaVersion, cuuint64_t flags,
                                     CUdriverProcAddressQueryResult* symbolStatus) {
    VULKANIZER_TRACE_API();
    if (!symbol || !pfn) return CUDA_ERROR_INVALID_VALUE;
    *pfn = lookupEntryPoint(symbol, cudaVersion, flags);
    if (symbolStatus) *symbolStatus = *pfn ? CU_GET_PROC_ADDRESS_SUCCESS : CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
    if (!*pfn) {
        // Very useful for discovering which entry points real applications need.
        VULKANIZER_INFO("cuGetProcAddress: '%s' (version %d) not implemented", symbol, cudaVersion);
        return CUDA_ERROR_NOT_FOUND;
    }
    return CUDA_SUCCESS;
}

// Private interfaces the CUDA runtime requests by UUID. None are provided yet; the request is
// logged so the needed tables can be identified (plan.md, phase 2.1).
CUresult CUDAAPI cuGetExportTable(const void** ppExportTable, const CUuuid* pExportTableId) {
    VULKANIZER_TRACE_API();
    if (!ppExportTable || !pExportTableId) return CUDA_ERROR_INVALID_VALUE;
    char hex[33];
    for (int i = 0; i < 16; ++i)
        std::snprintf(hex + 2 * i, 3, "%02x", static_cast<unsigned char>(pExportTableId->bytes[i]));
    VULKANIZER_WARN("cuGetExportTable: unknown interface %s", hex);
    *ppExportTable = nullptr;
    return CUDA_ERROR_NOT_SUPPORTED;
}

// CUDA 11.x signature (no symbolStatus).
#undef cuGetProcAddress
CUresult CUDAAPI cuGetProcAddress(const char* symbol, void** pfn, int cudaVersion, cuuint64_t flags);
CUresult CUDAAPI cuGetProcAddress(const char* symbol, void** pfn, int cudaVersion, cuuint64_t flags) {
    return cuGetProcAddress_v2(symbol, pfn, cudaVersion, flags, nullptr);
}

// ---------------------------------------------------------------------------
// Devices
// ---------------------------------------------------------------------------

CUresult CUDAAPI cuDeviceGetCount(int* count) {
    VULKANIZER_TRACE_API();
    if (!count) return CUDA_ERROR_INVALID_VALUE;
    if (!Driver::get().initialized()) return CUDA_ERROR_NOT_INITIALIZED;
    *count = Driver::get().deviceCount();
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGet(CUdevice* device, int ordinal) {
    VULKANIZER_TRACE_API();
    if (!device) return CUDA_ERROR_INVALID_VALUE;
    if (!Driver::get().initialized()) return CUDA_ERROR_NOT_INITIALIZED;
    if (!Driver::get().slot(ordinal)) return CUDA_ERROR_INVALID_DEVICE;
    *device = ordinal;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetName(char* name, int len, CUdevice dev) {
    VULKANIZER_TRACE_API();
    if (!name || len <= 0) return CUDA_ERROR_INVALID_VALUE;
    if (!Driver::get().initialized()) return CUDA_ERROR_NOT_INITIALIZED;
    DeviceSlot* s = Driver::get().slot(dev);
    if (!s) return CUDA_ERROR_INVALID_DEVICE;
    std::strncpy(name, s->info->name(), static_cast<size_t>(len) - 1);
    name[len - 1] = '\0';
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetUuid_v2(CUuuid* uuid, CUdevice dev) {
    VULKANIZER_TRACE_API();
    if (!uuid) return CUDA_ERROR_INVALID_VALUE;
    if (!Driver::get().initialized()) return CUDA_ERROR_NOT_INITIALIZED;
    DeviceSlot* s = Driver::get().slot(dev);
    if (!s) return CUDA_ERROR_INVALID_DEVICE;
    static_assert(sizeof(uuid->bytes) == VK_UUID_SIZE);
    std::memcpy(uuid->bytes, s->info->uuid, VK_UUID_SIZE);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetUuid(CUuuid* uuid, CUdevice dev) { return cuDeviceGetUuid_v2(uuid, dev); }

CUresult CUDAAPI cuDeviceTotalMem_v2(size_t* bytes, CUdevice dev) {
    VULKANIZER_TRACE_API();
    if (!bytes) return CUDA_ERROR_INVALID_VALUE;
    if (!Driver::get().initialized()) return CUDA_ERROR_NOT_INITIALIZED;
    DeviceSlot* s = Driver::get().slot(dev);
    if (!s) return CUDA_ERROR_INVALID_DEVICE;
    *bytes = static_cast<size_t>(s->info->deviceLocalBytes);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceComputeCapability(int* major, int* minor, CUdevice dev) {
    VULKANIZER_TRACE_API();
    if (!major || !minor) return CUDA_ERROR_INVALID_VALUE;
    if (!Driver::get().initialized()) return CUDA_ERROR_NOT_INITIALIZED;
    if (!Driver::get().slot(dev)) return CUDA_ERROR_INVALID_DEVICE;
    computeCapability(*major, *minor);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev) {
    VULKANIZER_TRACE_API();
    if (!pi) return CUDA_ERROR_INVALID_VALUE;
    if (!Driver::get().initialized()) return CUDA_ERROR_NOT_INITIALIZED;
    DeviceSlot* s = Driver::get().slot(dev);
    if (!s) return CUDA_ERROR_INVALID_DEVICE;

    const vk::PhysicalDeviceInfo& info = *s->info;
    const VkPhysicalDeviceLimits& lim = info.props.limits;
    auto clampInt = [](uint64_t v) { return static_cast<int>(std::min<uint64_t>(v, INT_MAX)); };
    int major, minor;
    computeCapability(major, minor);

    switch (attrib) {
        case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK:
            *pi = clampInt(std::min<uint32_t>(1024, lim.maxComputeWorkGroupInvocations)); break;
        case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X: *pi = clampInt(std::min<uint32_t>(1024, lim.maxComputeWorkGroupSize[0])); break;
        case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y: *pi = clampInt(std::min<uint32_t>(1024, lim.maxComputeWorkGroupSize[1])); break;
        case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z: *pi = clampInt(std::min<uint32_t>(64, lim.maxComputeWorkGroupSize[2])); break;
        case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X: *pi = 0x7FFFFFFF; break;  // split into several dispatches
        case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y: *pi = clampInt(lim.maxComputeWorkGroupCount[1]); break;
        case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z: *pi = clampInt(lim.maxComputeWorkGroupCount[2]); break;
        case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK:
        case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR:
        case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN: *pi = clampInt(lim.maxComputeSharedMemorySize); break;
        case CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY: *pi = 65536; break;
        case CU_DEVICE_ATTRIBUTE_WARP_SIZE: *pi = 32; break;  // CUDA semantics; emulated on other subgroup sizes
        case CU_DEVICE_ATTRIBUTE_MAX_PITCH: *pi = INT_MAX; break;
        case CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK:
        case CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR: *pi = 65536; break;
        case CU_DEVICE_ATTRIBUTE_CLOCK_RATE: *pi = 1000000; break;         // kHz, placeholder
        case CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE: *pi = 1000000; break;  // kHz, placeholder
        case CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH: *pi = 128; break;
        case CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE: *pi = 4 * 1024 * 1024; break;
        case CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT: *pi = 512; break;
        case CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT:
            // TODO: VK_AMD_shader_core_properties / VK_ARM_shader_core_builtins give real numbers.
            if (const char* env = std::getenv("VULKANIZER_SM_COUNT"); env && std::atoi(env) > 0) *pi = std::atoi(env);
            else *pi = info.isCpu() ? static_cast<int>(std::max(1u, std::thread::hardware_concurrency())) : 8;
            break;
        case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR: *pi = 2048; break;
        case CU_DEVICE_ATTRIBUTE_INTEGRATED:
            *pi = info.props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU || info.isCpu();
            break;
        case CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING: *pi = 1; break;
        case CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED: {  // sparse buffers (api_vmm.cpp)
            std::lock_guard<std::recursive_mutex> lock(Driver::get().mutex());
            *pi = Driver::get().ensureDevice(*s) == CUDA_SUCCESS && s->vk->features().sparseBuffers;
            break;
        }
        case CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED:
        case CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED: *pi = 1; break;
        case CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM:
        case CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED:
        case CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED:
        case CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED:
        case CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES: *pi = 0; break;
        case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: *pi = major; break;
        case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: *pi = minor; break;
        case CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED:
        case CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED: *pi = 1; break;
        case CU_DEVICE_ATTRIBUTE_GPU_OVERLAP:
        case CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT:
        case CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT:
        case CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY:
        case CU_DEVICE_ATTRIBUTE_COMPUTE_MODE:
        case CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS:
        case CU_DEVICE_ATTRIBUTE_ECC_ENABLED:
        case CU_DEVICE_ATTRIBUTE_PCI_BUS_ID:
        case CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID:
        case CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID:
        case CU_DEVICE_ATTRIBUTE_TCC_DRIVER:
        case CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED:
        case CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY:
        case CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD:
        case CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH: *pi = 0; break;
        default:
            if (attrib <= 0 || attrib > 150) return CUDA_ERROR_INVALID_VALUE;
            VULKANIZER_INFO("cuDeviceGetAttribute: attribute %d not modeled, reporting 0", static_cast<int>(attrib));
            *pi = 0;
            break;
    }
    return CUDA_SUCCESS;
}

}  // extern "C"
