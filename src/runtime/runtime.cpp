// Vulkanizer's libcudart: the CUDA Runtime API implemented on top of the Driver API (libcuda.so),
// so programs built with nvcc -cudart shared run unmodified. Covers the registration ABI
// that nvcc's host stubs call, kernel launch, devices, memory, streams and events.

#include "runtime/cudart_types.h"

#include "common/log.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr int kRuntimeVersion = VULKANIZER_CUDART_MAJOR >= 13 ? 13040 : 12090;

struct FatBinary;

// What cudaKernel_t points to: one __global__ function registered by the host stubs.
struct Kernel {
    FatBinary* fatbin = nullptr;
    std::string name;
    const void* hostFun = nullptr;
    std::map<int, CUfunction> functions;  // per device
};

struct FatBinary;

// A __device__ / __constant__ variable registered by the host stubs.
struct Variable {
    FatBinary* fatbin = nullptr;
    std::string name;
    size_t size = 0;
};

struct FatBinary {
    const void* image = nullptr;            // __fatBinC_Wrapper_t
    std::map<int, CUmodule> modules;        // loaded lazily per device
    std::vector<std::unique_ptr<Kernel>> kernels;
};

struct LaunchConfig {
    dim3 grid, block;
    size_t sharedMem;
    cudaStream_t stream;
};

struct Runtime {
    std::recursive_mutex mutex;
    std::once_flag initOnce;
    cudaError_t initError = cudaSuccess;
    int deviceCount = 0;
    std::vector<CUcontext> primary;  // retained primary contexts, per device
    std::vector<std::unique_ptr<FatBinary>> fatbins;
    std::unordered_map<const void*, Kernel*> kernelsByHostFun;
    std::unordered_map<const void*, Variable> variablesByHostVar;

    static Runtime& get() {
        static Runtime* r = new Runtime;  // never destroyed: atexit handlers still unregister
        return *r;
    }
};

thread_local int t_device = 0;
thread_local cudaError_t t_lastError = cudaSuccess;
thread_local std::vector<LaunchConfig> t_configs;

cudaError_t toCudaError(CUresult r) {
    // The runtime's error codes are numerically aligned with the driver's for every value
    // the Vulkanizer driver returns.
    return static_cast<cudaError_t>(r);
}

// Records non-success results for cudaGetLastError / cudaPeekAtLastError.
cudaError_t record(cudaError_t e) {
    if (e != cudaSuccess) t_lastError = e;
    return e;
}
cudaError_t record(CUresult r) { return record(toCudaError(r)); }

// The device's primary context, retained on first use (after enter() initialized the driver).
cudaError_t primaryContext(int device, CUcontext& ctx) {
    Runtime& rt = Runtime::get();
    if (device < 0 || device >= rt.deviceCount) return cudaErrorInvalidDevice;
    std::lock_guard<std::recursive_mutex> lock(rt.mutex);
    if (!rt.primary[device]) {
        if (CUresult r = cuDevicePrimaryCtxRetain(&rt.primary[device], device)) return toCudaError(r);
    }
    ctx = rt.primary[device];
    return cudaSuccess;
}

// Every API call starts here: initialize the driver once and make the current device's
// primary context current on this thread.
cudaError_t enter() {
    Runtime& rt = Runtime::get();
    std::call_once(rt.initOnce, [&] {
        CUresult r = cuInit(0);
        if (r == CUDA_SUCCESS) r = cuDeviceGetCount(&rt.deviceCount);
        if (r != CUDA_SUCCESS) rt.initError = r == CUDA_ERROR_NO_DEVICE ? cudaErrorNoDevice : toCudaError(r);
        else if (rt.deviceCount == 0) rt.initError = cudaErrorNoDevice;
        rt.primary.assign(size_t(rt.deviceCount), nullptr);
    });
    if (rt.initError != cudaSuccess) return rt.initError;
    if (t_device < 0 || t_device >= rt.deviceCount) return cudaErrorInvalidDevice;

    CUcontext ctx;
    if (cudaError_t e = primaryContext(t_device, ctx)) return e;
    CUcontext current = nullptr;
    cuCtxGetCurrent(&current);
    if (current != ctx) {
        if (CUresult r = cuCtxSetCurrent(ctx)) return toCudaError(r);
    }
    return cudaSuccess;
}

#define ENTER()                                     \
    do {                                            \
        VULKANIZER_TRACE_API();                          \
        if (cudaError_t e_ = enter()) return record(e_); \
    } while (0)

// The module of a fat binary on the current device, loaded on first use. Caller holds the lock.
cudaError_t moduleFor(FatBinary& fb, CUmodule& out) {
    CUmodule& mod = fb.modules[t_device];
    if (!mod) {
        if (CUresult r = cuModuleLoadData(&mod, fb.image)) {
            mod = nullptr;
            VULKANIZER_ERROR("cannot load device code for device %d (%d)", t_device, int(r));
            return r == CUDA_ERROR_NO_BINARY_FOR_GPU ? cudaErrorNoKernelImageForDevice : toCudaError(r);
        }
    }
    out = mod;
    return cudaSuccess;
}

// Device address of a registered __device__/__constant__ variable on the current device.
cudaError_t symbolAddress(const void* symbol, CUdeviceptr& ptr, size_t& size) {
    Runtime& rt = Runtime::get();
    std::lock_guard<std::recursive_mutex> lock(rt.mutex);
    auto it = rt.variablesByHostVar.find(symbol);
    if (it == rt.variablesByHostVar.end()) return cudaErrorInvalidSymbol;
    CUmodule mod;
    if (cudaError_t e = moduleFor(*it->second.fatbin, mod)) return e;
    if (CUresult r = cuModuleGetGlobal(&ptr, &size, mod, it->second.name.c_str()))
        return r == CUDA_ERROR_NOT_FOUND ? cudaErrorInvalidSymbol : toCudaError(r);
    return cudaSuccess;
}

// Resolves a kernel to a driver function on the current device, loading its module on first use.
cudaError_t resolve(Kernel* k, CUfunction& out) {
    Runtime& rt = Runtime::get();
    std::lock_guard<std::recursive_mutex> lock(rt.mutex);
    auto it = k->functions.find(t_device);
    if (it != k->functions.end()) {
        out = it->second;
        return cudaSuccess;
    }
    CUmodule mod;
    if (cudaError_t e = moduleFor(*k->fatbin, mod)) return e;
    CUfunction f;
    if (CUresult r = cuModuleGetFunction(&f, mod, k->name.c_str())) {
        VULKANIZER_ERROR("kernel '%s' not found in its module", k->name.c_str());
        return r == CUDA_ERROR_NOT_FOUND ? cudaErrorInvalidDeviceFunction : toCudaError(r);
    }
    k->functions[t_device] = f;
    out = f;
    return cudaSuccess;
}

Kernel* findKernel(const void* hostFun) {
    Runtime& rt = Runtime::get();
    std::lock_guard<std::recursive_mutex> lock(rt.mutex);
    auto it = rt.kernelsByHostFun.find(hostFun);
    if (it == rt.kernelsByHostFun.end()) {
        VULKANIZER_WARN("no kernel registered for host function %p", hostFun);
        return nullptr;
    }
    return it->second;
}

// `attrs`: launch attributes (cudaLaunchKernelEx), handed to cuLaunchKernelEx.
cudaError_t launch(Kernel* k, dim3 grid, dim3 block, void** args, size_t sharedMem, cudaStream_t stream,
                   const cudaLaunchAttribute* attrs = nullptr, unsigned numAttrs = 0) {
    if (!k) return record(cudaErrorInvalidDeviceFunction);
    CUfunction f;
    if (cudaError_t e = resolve(k, f)) return record(e);
    CUresult r;
    if (numAttrs) {
        // cudaLaunchAttribute is laid out as CUlaunchAttribute, with the same attribute ids.
        CUlaunchConfig config{grid.x,  grid.y, grid.z, block.x, block.y, block.z, unsigned(sharedMem), stream,
                              reinterpret_cast<CUlaunchAttribute*>(const_cast<cudaLaunchAttribute*>(attrs)), numAttrs};
        r = cuLaunchKernelEx(&config, f, args, nullptr);
    } else {
        r = cuLaunchKernel(f, grid.x, grid.y, grid.z, block.x, block.y, block.z, unsigned(sharedMem), stream, args,
                           nullptr);
    }
    // The driver reports bad grid/block dimensions as an invalid value.
    return record(r == CUDA_ERROR_INVALID_VALUE ? cudaErrorInvalidConfiguration : toCudaError(r));
}

int attr(int attribute, int device) {
    int v = 0;
    cuDeviceGetAttribute(&v, static_cast<CUdevice_attribute>(attribute), device);
    return v;
}

struct ErrorText {
    cudaError_t code;
    const char* name;
    const char* text;
};

const ErrorText kErrors[] = {
    {cudaSuccess, "cudaSuccess", "no error"},
    {cudaErrorInvalidValue, "cudaErrorInvalidValue", "invalid argument"},
    {cudaErrorMemoryAllocation, "cudaErrorMemoryAllocation", "out of memory"},
    {cudaErrorInitializationError, "cudaErrorInitializationError", "initialization error"},
    {cudaErrorCudartUnloading, "cudaErrorCudartUnloading", "driver shutting down"},
    {cudaErrorInvalidConfiguration, "cudaErrorInvalidConfiguration", "invalid configuration argument"},
    {cudaErrorInvalidSymbol, "cudaErrorInvalidSymbol", "invalid device symbol"},
    {cudaErrorInvalidDevicePointer, "cudaErrorInvalidDevicePointer", "invalid device pointer"},
    {cudaErrorInvalidMemcpyDirection, "cudaErrorInvalidMemcpyDirection", "invalid copy direction for memcpy"},
    {cudaErrorInsufficientDriver, "cudaErrorInsufficientDriver",
     "CUDA driver version is insufficient for CUDA runtime version"},
    {cudaErrorMissingConfiguration, "cudaErrorMissingConfiguration", "__global__ function call is not configured"},
    {cudaErrorInvalidDeviceFunction, "cudaErrorInvalidDeviceFunction", "invalid device function"},
    {cudaErrorNoDevice, "cudaErrorNoDevice", "no CUDA-capable device is detected"},
    {cudaErrorInvalidDevice, "cudaErrorInvalidDevice", "invalid device ordinal"},
    {cudaErrorStartupFailure, "cudaErrorStartupFailure", "device kernel image is invalid"},
    {cudaErrorInvalidKernelImage, "cudaErrorInvalidKernelImage", "device kernel image is invalid"},
    {cudaErrorDeviceUninitialized, "cudaErrorDeviceUninitialized", "invalid device context"},
    {cudaErrorNoKernelImageForDevice, "cudaErrorNoKernelImageForDevice",
     "no kernel image is available for execution on the device"},
    {cudaErrorInvalidPtx, "cudaErrorInvalidPtx", "a PTX JIT compilation failed"},
    {cudaErrorFileNotFound, "cudaErrorFileNotFound", "file not found"},
    {cudaErrorInvalidResourceHandle, "cudaErrorInvalidResourceHandle", "invalid resource handle"},
    {cudaErrorSymbolNotFound, "cudaErrorSymbolNotFound", "named symbol not found"},
    {cudaErrorNotReady, "cudaErrorNotReady", "device not ready"},
    {cudaErrorIllegalAddress, "cudaErrorIllegalAddress", "an illegal memory access was encountered"},
    {cudaErrorLaunchOutOfResources, "cudaErrorLaunchOutOfResources", "too many resources requested for launch"},
    {cudaErrorLaunchFailure, "cudaErrorLaunchFailure", "unspecified launch failure"},
    {cudaErrorCooperativeLaunchTooLarge, "cudaErrorCooperativeLaunchTooLarge", "too many blocks in cooperative launch"},
    {cudaErrorNotPermitted, "cudaErrorNotPermitted", "operation not permitted"},
    {cudaErrorNotSupported, "cudaErrorNotSupported", "operation not supported"},
    {cudaErrorUnknown, "cudaErrorUnknown", "unknown error"},
};

const ErrorText* findError(cudaError_t e) {
    for (const ErrorText& t : kErrors)
        if (t.code == e) return &t;
    return nullptr;
}

}  // namespace

struct CUkern_st : Kernel {};

extern "C" {

// ---- registration (called from nvcc-generated host code before main) ------------------------

void** CUDARTAPI __cudaRegisterFatBinary(void* fatCubin) {
    VULKANIZER_TRACE_API();
    Runtime& rt = Runtime::get();
    std::lock_guard<std::recursive_mutex> lock(rt.mutex);
    auto fb = std::make_unique<FatBinary>();
    fb->image = fatCubin;
    rt.fatbins.push_back(std::move(fb));
    return reinterpret_cast<void**>(rt.fatbins.back().get());
}

void CUDARTAPI __cudaRegisterFatBinaryEnd(void** fatCubinHandle) { VULKANIZER_TRACE_API(); }

void CUDARTAPI __cudaUnregisterFatBinary(void** fatCubinHandle) {
    VULKANIZER_TRACE_API();
    Runtime& rt = Runtime::get();
    std::lock_guard<std::recursive_mutex> lock(rt.mutex);
    auto* fb = reinterpret_cast<FatBinary*>(fatCubinHandle);
    for (auto& [device, mod] : fb->modules)
        if (mod) cuModuleUnload(mod);
    fb->modules.clear();
    for (auto& k : fb->kernels) {
        k->functions.clear();
        rt.kernelsByHostFun.erase(k->hostFun);
    }
    std::erase_if(rt.variablesByHostVar, [&](const auto& v) { return v.second.fatbin == fb; });
}

void CUDARTAPI __cudaRegisterFunction(void** fatCubinHandle, const char* hostFun, char* deviceFun,
                                      const char* deviceName, int threadLimit, uint3* tid, uint3* bid, dim3* bDim,
                                      dim3* gDim, int* wSize) {
    VULKANIZER_TRACE_API();
    Runtime& rt = Runtime::get();
    std::lock_guard<std::recursive_mutex> lock(rt.mutex);
    auto* fb = reinterpret_cast<FatBinary*>(fatCubinHandle);
    auto k = std::make_unique<CUkern_st>();
    k->fatbin = fb;
    k->name = deviceName;
    k->hostFun = hostFun;
    VULKANIZER_DEBUG("registered kernel %s at %p", deviceName, static_cast<const void*>(hostFun));
    rt.kernelsByHostFun[hostFun] = k.get();
    fb->kernels.push_back(std::move(k));
}

void CUDARTAPI __cudaRegisterVar(void** fatCubinHandle, char* hostVar, char* deviceAddress, const char* deviceName,
                                 int ext, size_t size, int constant, int global) {
    VULKANIZER_TRACE_API();
    Runtime& rt = Runtime::get();
    std::lock_guard<std::recursive_mutex> lock(rt.mutex);
    rt.variablesByHostVar[hostVar] = {reinterpret_cast<FatBinary*>(fatCubinHandle), deviceName, size};
}

void CUDARTAPI __cudaRegisterManagedVar(void** fatCubinHandle, void** hostVarPtrAddress, char* deviceAddress,
                                        const char* deviceName, int ext, size_t size, int constant, int global) {
    VULKANIZER_TRACE_API();
    VULKANIZER_WARN_ONCE("__managed__ variables are not supported yet ('%s')", deviceName);
}

char CUDARTAPI __cudaInitModule(void** fatCubinHandle) {
    VULKANIZER_TRACE_API();
    return 1;
}

void CUDARTAPI __cudaRegisterTexture(void** fatCubinHandle, const void* hostVar, const void** deviceAddress,
                                     const char* deviceName, int dim, int norm, int ext) {
    VULKANIZER_WARN_ONCE("texture references are not supported ('%s')", deviceName);
}

void CUDARTAPI __cudaRegisterSurface(void** fatCubinHandle, const void* hostVar, const void** deviceAddress,
                                     const char* deviceName, int dim, int ext) {
    VULKANIZER_WARN_ONCE("surface references are not supported ('%s')", deviceName);
}

// ---- launch -----------------------------------------------------------------------------------

unsigned CUDARTAPI __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem,
                                               struct CUstream_st* stream) {
    VULKANIZER_TRACE_API();
    t_configs.push_back({gridDim, blockDim, sharedMem, stream});
    return 0;
}

cudaError_t CUDARTAPI __cudaPopCallConfiguration(dim3* gridDim, dim3* blockDim, size_t* sharedMem, void* stream) {
    VULKANIZER_TRACE_API();
    if (t_configs.empty()) return record(cudaErrorMissingConfiguration);
    const LaunchConfig c = t_configs.back();
    t_configs.pop_back();
    *gridDim = c.grid;
    *blockDim = c.block;
    *sharedMem = c.sharedMem;
    *static_cast<cudaStream_t*>(stream) = c.stream;
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem,
                                       cudaStream_t stream) {
    ENTER();
    return launch(findKernel(func), gridDim, blockDim, args, sharedMem, stream);
}

cudaError_t CUDARTAPI cudaLaunchKernel_ptsz(const void* func, dim3 gridDim, dim3 blockDim, void** args,
                                            size_t sharedMem, cudaStream_t stream) {
    ENTER();
    return launch(findKernel(func), gridDim, blockDim, args, sharedMem, stream ? stream : CU_STREAM_PER_THREAD);
}

cudaError_t CUDARTAPI __cudaGetKernel(cudaKernel_t* kernel, const void* hostFun) {
    VULKANIZER_TRACE_API();
    if (!kernel) return record(cudaErrorInvalidValue);
    *kernel = static_cast<CUkern_st*>(findKernel(hostFun));
    return *kernel ? cudaSuccess : record(cudaErrorInvalidDeviceFunction);
}

cudaError_t CUDARTAPI __cudaLaunchKernel(cudaKernel_t kernel, dim3 gridDim, dim3 blockDim, void** args,
                                         size_t sharedMem, cudaStream_t stream) {
    ENTER();
    return launch(kernel, gridDim, blockDim, args, sharedMem, stream);
}

cudaError_t CUDARTAPI __cudaLaunchKernel_ptsz(cudaKernel_t kernel, dim3 gridDim, dim3 blockDim, void** args,
                                              size_t sharedMem, cudaStream_t stream) {
    ENTER();
    return launch(kernel, gridDim, blockDim, args, sharedMem, stream ? stream : CU_STREAM_PER_THREAD);
}

cudaError_t CUDARTAPI cudaFuncGetAttributes(cudaFuncAttributes* attr, const void* func) {
    ENTER();
    if (!attr) return record(cudaErrorInvalidValue);
    Kernel* k = findKernel(func);
    if (!k) return record(cudaErrorInvalidDeviceFunction);
    CUfunction f;
    if (cudaError_t e = resolve(k, f)) return record(e);
    std::memset(attr, 0, sizeof *attr);
    int v = 0;
    auto get = [&](CUfunction_attribute a) { return cuFuncGetAttribute(&v, a, f) == CUDA_SUCCESS ? v : 0; };
    attr->maxThreadsPerBlock = get(CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
    attr->sharedSizeBytes = size_t(get(CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES));
    attr->constSizeBytes = size_t(get(CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES));
    attr->localSizeBytes = size_t(get(CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES));
    attr->numRegs = get(CU_FUNC_ATTRIBUTE_NUM_REGS);
    attr->ptxVersion = get(CU_FUNC_ATTRIBUTE_PTX_VERSION);
    attr->binaryVersion = get(CU_FUNC_ATTRIBUTE_BINARY_VERSION);
    attr->maxDynamicSharedSizeBytes = get(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES);
    return cudaSuccess;
}

// Tuning hints without an effect on Vulkan.
cudaError_t CUDARTAPI cudaFuncSetAttribute(const void* func, int attr, int value) {
    ENTER();
    return findKernel(func) ? cudaSuccess : record(cudaErrorInvalidDeviceFunction);
}
cudaError_t CUDARTAPI cudaFuncSetCacheConfig(const void* func, int cacheConfig) {
    ENTER();
    return cudaSuccess;
}
cudaError_t CUDARTAPI cudaDeviceSetCacheConfig(int cacheConfig) {
    ENTER();
    return cudaSuccess;
}

// ---- errors -----------------------------------------------------------------------------------

cudaError_t CUDARTAPI cudaGetLastError(void) {
    const cudaError_t e = t_lastError;
    t_lastError = cudaSuccess;
    return e;
}

cudaError_t CUDARTAPI cudaPeekAtLastError(void) { return t_lastError; }

const char* CUDARTAPI cudaGetErrorName(cudaError_t error) {
    const ErrorText* t = findError(error);
    return t ? t->name : "cudaErrorUnknown";
}

const char* CUDARTAPI cudaGetErrorString(cudaError_t error) {
    const ErrorText* t = findError(error);
    return t ? t->text : "unrecognized error code";
}

// ---- versions and devices ---------------------------------------------------------------------

cudaError_t CUDARTAPI cudaRuntimeGetVersion(int* runtimeVersion) {
    if (!runtimeVersion) return record(cudaErrorInvalidValue);
    *runtimeVersion = kRuntimeVersion;
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaDriverGetVersion(int* driverVersion) {
    if (!driverVersion) return record(cudaErrorInvalidValue);
    return record(cuDriverGetVersion(driverVersion));
}

cudaError_t CUDARTAPI cudaGetDeviceCount(int* count) {
    VULKANIZER_TRACE_API();
    if (!count) return record(cudaErrorInvalidValue);
    *count = 0;
    if (cudaError_t e = enter(); e != cudaSuccess && e != cudaErrorInvalidDevice) return record(e);
    *count = Runtime::get().deviceCount;
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaGetDevice(int* device) {
    if (!device) return record(cudaErrorInvalidValue);
    *device = t_device;
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaSetDevice(int device) {
    VULKANIZER_TRACE_API();
    const int previous = t_device;
    t_device = device;
    if (cudaError_t e = enter()) {
        t_device = previous;
        return record(e);
    }
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device) {
    ENTER();
    if (!value) return record(cudaErrorInvalidValue);
    return record(cuDeviceGetAttribute(value, static_cast<CUdevice_attribute>(attr), device));
}

// CUDA 12 exports this as cudaGetDeviceProperties_v2 too; CUDA 13 only under the plain name, with
// the fields its cudaDeviceProp dropped (clock rates, compute mode, ...) removed.
cudaError_t CUDARTAPI cudaGetDeviceProperties(cudaDeviceProp* prop, int device) {
    ENTER();
    if (!prop) return record(cudaErrorInvalidValue);
    if (device < 0 || device >= Runtime::get().deviceCount) return record(cudaErrorInvalidDevice);
    cudaDeviceProp& p = *prop;
    std::memset(&p, 0, sizeof p);
    cuDeviceGetName(p.name, sizeof p.name, device);
    cuDeviceGetUuid(&p.uuid, device);
    cuDeviceTotalMem(&p.totalGlobalMem, device);
    const int d = device;
    p.sharedMemPerBlock = size_t(attr(8, d));
    p.regsPerBlock = attr(12, d);
    p.warpSize = attr(10, d);
    p.memPitch = size_t(attr(11, d));
    p.maxThreadsPerBlock = attr(1, d);
    for (int i = 0; i < 3; ++i) {
        p.maxThreadsDim[i] = attr(2 + i, d);
        p.maxGridSize[i] = attr(5 + i, d);
    }
#if VULKANIZER_CUDART_MAJOR < 13
    p.clockRate = attr(13, d);
    p.deviceOverlap = attr(15, d);
    p.kernelExecTimeoutEnabled = attr(17, d);
    p.computeMode = attr(20, d);
    p.memoryClockRate = attr(36, d);
#else
    p.deviceNumaId = -1;  // not a NUMA device
    p.hostNumaId = -1;
#endif
    p.totalConstMem = size_t(attr(9, d));
    p.major = attr(75, d);
    p.minor = attr(76, d);
    p.textureAlignment = size_t(attr(14, d));
    p.texturePitchAlignment = size_t(attr(51, d));
    p.maxTexture1D = attr(21, d);
#if VULKANIZER_CUDART_MAJOR < 13
    p.maxTexture1DLinear = attr(69, d);
#endif
    p.maxTexture2D[0] = attr(22, d), p.maxTexture2D[1] = attr(23, d);
    p.maxTexture3D[0] = attr(24, d), p.maxTexture3D[1] = attr(25, d), p.maxTexture3D[2] = attr(26, d);
    p.maxTexture2DLayered[0] = attr(27, d), p.maxTexture2DLayered[1] = attr(28, d), p.maxTexture2DLayered[2] = attr(29, d);
    p.maxTexture1DLayered[0] = attr(42, d), p.maxTexture1DLayered[1] = attr(43, d);
    p.maxTexture2DGather[0] = attr(45, d), p.maxTexture2DGather[1] = attr(46, d);
    p.maxTexture2DLinear[0] = attr(70, d), p.maxTexture2DLinear[1] = attr(71, d), p.maxTexture2DLinear[2] = attr(72, d);
    p.maxSurface1D = attr(55, d);
    p.maxSurface2D[0] = attr(56, d), p.maxSurface2D[1] = attr(57, d);
    p.maxSurface3D[0] = attr(58, d), p.maxSurface3D[1] = attr(59, d), p.maxSurface3D[2] = attr(60, d);
    p.surfaceAlignment = size_t(attr(30, d));
    p.multiProcessorCount = attr(16, d);
    p.integrated = attr(18, d);
    p.canMapHostMemory = attr(19, d);
    p.concurrentKernels = attr(31, d);
    p.ECCEnabled = attr(32, d);
    p.pciBusID = attr(33, d);
    p.pciDeviceID = attr(34, d);
    p.pciDomainID = attr(50, d);
    p.tccDriver = attr(35, d);
    p.asyncEngineCount = attr(40, d);
    p.unifiedAddressing = attr(41, d);
    p.memoryBusWidth = attr(37, d);
    p.l2CacheSize = attr(38, d);
    p.maxThreadsPerMultiProcessor = attr(39, d);
    p.streamPrioritiesSupported = attr(78, d);
    p.globalL1CacheSupported = attr(79, d);
    p.localL1CacheSupported = attr(80, d);
    p.sharedMemPerMultiprocessor = size_t(attr(81, d));
    p.regsPerMultiprocessor = attr(82, d);
    p.managedMemory = attr(83, d);
    p.isMultiGpuBoard = attr(84, d);
    p.cooperativeLaunch = attr(95, d);
    p.sharedMemPerBlockOptin = size_t(attr(97, d));
    p.maxBlocksPerMultiProcessor = attr(106, d);
    return cudaSuccess;
}

#if VULKANIZER_CUDART_MAJOR < 13
cudaError_t CUDARTAPI cudaGetDeviceProperties_v2(cudaDeviceProp* prop, int device) {
    return cudaGetDeviceProperties(prop, device);
}
#endif

cudaError_t CUDARTAPI cudaDeviceSynchronize(void) {
    ENTER();
    return record(cuCtxSynchronize());
}

cudaError_t CUDARTAPI cudaDeviceReset(void) {
    ENTER();
    Runtime& rt = Runtime::get();
    std::lock_guard<std::recursive_mutex> lock(rt.mutex);
    cuCtxSynchronize();
    for (auto& fb : rt.fatbins) {
        auto it = fb->modules.find(t_device);
        if (it == fb->modules.end()) continue;
        if (it->second) cuModuleUnload(it->second);
        fb->modules.erase(it);
        for (auto& k : fb->kernels) k->functions.erase(t_device);
    }
    if (rt.primary[t_device]) {
        cuCtxSetCurrent(nullptr);
        cuDevicePrimaryCtxRelease(t_device);
        rt.primary[t_device] = nullptr;
    }
    return cudaSuccess;
}

// ---- memory -----------------------------------------------------------------------------------

cudaError_t CUDARTAPI cudaMalloc(void** devPtr, size_t size) {
    ENTER();
    if (!devPtr) return record(cudaErrorInvalidValue);
    if (size == 0) {
        *devPtr = nullptr;
        return cudaSuccess;
    }
    CUdeviceptr p = 0;
    if (CUresult r = cuMemAlloc(&p, size)) return record(r == CUDA_ERROR_OUT_OF_MEMORY ? cudaErrorMemoryAllocation : toCudaError(r));
    *devPtr = reinterpret_cast<void*>(p);
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaFree(void* devPtr) {
    ENTER();
    if (!devPtr) return cudaSuccess;  // cudaFree(0) is the idiomatic way to initialize the runtime
    return record(cuMemFree(reinterpret_cast<CUdeviceptr>(devPtr)));
}

// Stream-ordered allocation (the driver allocates right away and frees once the stream's
// work is done, without waiting).
cudaError_t CUDARTAPI cudaMallocFromPoolAsync(void** devPtr, size_t size, cudaMemPool_t pool, cudaStream_t stream) {
    ENTER();
    if (!devPtr) return record(cudaErrorInvalidValue);
    if (size == 0) {
        *devPtr = nullptr;
        return cudaSuccess;
    }
    CUdeviceptr p = 0;
    const CUresult r = pool ? cuMemAllocFromPoolAsync(&p, size, pool, stream) : cuMemAllocAsync(&p, size, stream);
    if (r) return record(r == CUDA_ERROR_OUT_OF_MEMORY ? cudaErrorMemoryAllocation : toCudaError(r));
    *devPtr = reinterpret_cast<void*>(p);
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaMallocAsync(void** devPtr, size_t size, cudaStream_t stream) {
    return cudaMallocFromPoolAsync(devPtr, size, nullptr, stream);
}

cudaError_t CUDARTAPI cudaFreeAsync(void* devPtr, cudaStream_t stream) {
    ENTER();
    if (!devPtr) return cudaSuccess;
    return record(cuMemFreeAsync(reinterpret_cast<CUdeviceptr>(devPtr), stream));
}

cudaError_t CUDARTAPI cudaMallocAsync_ptsz(void** devPtr, size_t size, cudaStream_t stream) {
    return cudaMallocAsync(devPtr, size, stream);
}
cudaError_t CUDARTAPI cudaMallocFromPoolAsync_ptsz(void** devPtr, size_t size, cudaMemPool_t pool, cudaStream_t stream) {
    return cudaMallocFromPoolAsync(devPtr, size, pool, stream);
}
cudaError_t CUDARTAPI cudaFreeAsync_ptsz(void* devPtr, cudaStream_t stream) { return cudaFreeAsync(devPtr, stream); }

cudaError_t CUDARTAPI cudaMallocHost(void** ptr, size_t size) {
    ENTER();
    return record(cuMemAllocHost(ptr, size));
}

// The flags (cudaHostAllocPortable/Mapped/WriteCombined) have the driver's values.
cudaError_t CUDARTAPI cudaHostAlloc(void** pHost, size_t size, unsigned int flags) {
    ENTER();
    return record(cuMemHostAlloc(pHost, size, flags));
}

cudaError_t CUDARTAPI cudaHostGetFlags(unsigned int* pFlags, void* pHost) {
    ENTER();
    return record(cuMemHostGetFlags(pFlags, pHost));
}

cudaError_t CUDARTAPI cudaFreeHost(void* ptr) {
    ENTER();
    return record(cuMemFreeHost(ptr));
}

cudaError_t CUDARTAPI cudaMemGetInfo(size_t* free, size_t* total) {
    ENTER();
    return record(cuMemGetInfo(free, total));
}

static cudaError_t memcpyImpl(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream,
                              bool async) {
    if (count == 0) return cudaSuccess;
    const auto d = reinterpret_cast<CUdeviceptr>(dst);
    const auto s = reinterpret_cast<CUdeviceptr>(src);
    CUresult r;
    switch (kind) {
    case cudaMemcpyHostToHost:
        std::memcpy(dst, src, count);
        return cudaSuccess;
    case cudaMemcpyHostToDevice:
        r = async ? cuMemcpyHtoDAsync(d, src, count, stream) : cuMemcpyHtoD(d, src, count);
        break;
    case cudaMemcpyDeviceToHost:
        r = async ? cuMemcpyDtoHAsync(dst, s, count, stream) : cuMemcpyDtoH(dst, s, count);
        break;
    case cudaMemcpyDeviceToDevice:
        r = async ? cuMemcpyDtoDAsync(d, s, count, stream) : cuMemcpyDtoD(d, s, count);
        break;
    case cudaMemcpyDefault:
        r = async ? cuMemcpyAsync(d, s, count, stream) : cuMemcpy(d, s, count);
        break;
    default:
        return record(cudaErrorInvalidMemcpyDirection);
    }
    return record(r);
}

cudaError_t CUDARTAPI cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
    ENTER();
    return memcpyImpl(dst, src, count, kind, nullptr, false);
}

cudaError_t CUDARTAPI cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind,
                                      cudaStream_t stream) {
    ENTER();
    return memcpyImpl(dst, src, count, kind, stream, true);
}

cudaError_t CUDARTAPI cudaMemset(void* devPtr, int value, size_t count) {
    ENTER();
    return record(cuMemsetD8(reinterpret_cast<CUdeviceptr>(devPtr), static_cast<unsigned char>(value), count));
}

cudaError_t CUDARTAPI cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream) {
    ENTER();
    return record(
        cuMemsetD8Async(reinterpret_cast<CUdeviceptr>(devPtr), static_cast<unsigned char>(value), count, stream));
}

cudaError_t CUDARTAPI cudaGetSymbolAddress(void** devPtr, const void* symbol) {
    ENTER();
    if (!devPtr) return record(cudaErrorInvalidValue);
    CUdeviceptr p;
    size_t size;
    if (cudaError_t e = symbolAddress(symbol, p, size)) return record(e);
    *devPtr = reinterpret_cast<void*>(p);
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaGetSymbolSize(size_t* size, const void* symbol) {
    ENTER();
    if (!size) return record(cudaErrorInvalidValue);
    CUdeviceptr p;
    return record(symbolAddress(symbol, p, *size));
}

static cudaError_t symbolCopy(const void* symbol, size_t count, size_t offset, CUdeviceptr& ptr) {
    size_t size;
    if (cudaError_t e = symbolAddress(symbol, ptr, size)) return e;
    if (offset > size || count > size - offset) return cudaErrorInvalidValue;
    ptr += offset;
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaMemcpyToSymbol(const void* symbol, const void* src, size_t count, size_t offset,
                                         cudaMemcpyKind kind) {
    ENTER();
    CUdeviceptr p;
    if (cudaError_t e = symbolCopy(symbol, count, offset, p)) return record(e);
    if (kind != cudaMemcpyHostToDevice && kind != cudaMemcpyDeviceToDevice && kind != cudaMemcpyDefault)
        return record(cudaErrorInvalidMemcpyDirection);
    return memcpyImpl(reinterpret_cast<void*>(p), src, count, kind == cudaMemcpyDefault ? cudaMemcpyDefault : kind,
                      nullptr, false);
}

cudaError_t CUDARTAPI cudaMemcpyFromSymbol(void* dst, const void* symbol, size_t count, size_t offset,
                                           cudaMemcpyKind kind) {
    ENTER();
    CUdeviceptr p;
    if (cudaError_t e = symbolCopy(symbol, count, offset, p)) return record(e);
    if (kind != cudaMemcpyDeviceToHost && kind != cudaMemcpyDeviceToDevice && kind != cudaMemcpyDefault)
        return record(cudaErrorInvalidMemcpyDirection);
    return memcpyImpl(dst, reinterpret_cast<void*>(p), count, kind, nullptr, false);
}

cudaError_t CUDARTAPI cudaMemcpyToSymbolAsync(const void* symbol, const void* src, size_t count, size_t offset,
                                              cudaMemcpyKind kind, cudaStream_t stream) {
    ENTER();
    CUdeviceptr p;
    if (cudaError_t e = symbolCopy(symbol, count, offset, p)) return record(e);
    return memcpyImpl(reinterpret_cast<void*>(p), src, count, kind, stream, true);
}

cudaError_t CUDARTAPI cudaMemcpyFromSymbolAsync(void* dst, const void* symbol, size_t count, size_t offset,
                                                cudaMemcpyKind kind, cudaStream_t stream) {
    ENTER();
    CUdeviceptr p;
    if (cudaError_t e = symbolCopy(symbol, count, offset, p)) return record(e);
    return memcpyImpl(dst, reinterpret_cast<void*>(p), count, kind, stream, true);
}

cudaError_t CUDARTAPI cudaMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                                        size_t height, cudaMemcpyKind kind, cudaStream_t stream) {
    ENTER();
    if (width > dpitch || width > spitch) return record(cudaErrorInvalidPitchValue);
    if (dpitch == width && spitch == width) return memcpyImpl(dst, src, width * height, kind, stream, true);
    for (size_t row = 0; row < height; ++row) {
        if (cudaError_t e = memcpyImpl(static_cast<char*>(dst) + row * dpitch,
                                       static_cast<const char*>(src) + row * spitch, width, kind, stream, true))
            return e;
    }
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                                   size_t height, cudaMemcpyKind kind) {
    return cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, nullptr);
}

cudaError_t CUDARTAPI cudaMemcpyPeerAsync(void* dst, int dstDevice, const void* src, int srcDevice, size_t count,
                                          cudaStream_t stream) {
    ENTER();
    if (dstDevice == srcDevice) return memcpyImpl(dst, src, count, cudaMemcpyDeviceToDevice, stream, true);
    CUcontext dstCtx, srcCtx;  // between devices: the driver copies through host memory
    if (cudaError_t e = primaryContext(dstDevice, dstCtx)) return record(e);
    if (cudaError_t e = primaryContext(srcDevice, srcCtx)) return record(e);
    return record(cuMemcpyPeerAsync(CUdeviceptr(dst), dstCtx, CUdeviceptr(src), srcCtx, count, stream));
}

cudaError_t CUDARTAPI cudaMemcpyPeer(void* dst, int dstDevice, const void* src, int srcDevice, size_t count) {
    return cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, count, nullptr);
}

// Managed, pinned and registered memory: host pages the device uses at the same address
// (driver api_host.cpp). Flags and advice values are the driver's.
cudaError_t CUDARTAPI cudaMallocManaged(void** devPtr, size_t size, unsigned int flags) {
    ENTER();
    if (!devPtr) return record(cudaErrorInvalidValue);
    CUdeviceptr p = 0;
    const CUresult r = cuMemAllocManaged(&p, size, flags);
    *devPtr = reinterpret_cast<void*>(p);
    return record(r);
}
cudaError_t CUDARTAPI cudaHostRegister(void* ptr, size_t size, unsigned int flags) {
    ENTER();
    return record(cuMemHostRegister(ptr, size, flags));
}
cudaError_t CUDARTAPI cudaHostUnregister(void* ptr) {
    ENTER();
    return record(cuMemHostUnregister(ptr));
}
cudaError_t CUDARTAPI cudaHostGetDevicePointer(void** pDevice, void* pHost, unsigned int flags) {
    ENTER();
    if (!pDevice) return record(cudaErrorInvalidValue);
    CUdeviceptr p = 0;
    const CUresult r = cuMemHostGetDevicePointer(&p, pHost, flags);
    *pDevice = reinterpret_cast<void*>(p);
    return record(r);
}

// Managed memory never migrates: prefetches and attachments only check their arguments.
#if VULKANIZER_CUDART_MAJOR >= 13
cudaError_t CUDARTAPI cudaMemPrefetchAsync(const void* devPtr, size_t count, cudaMemLocation location,
                                           unsigned int flags, cudaStream_t stream) {
#else
cudaError_t CUDARTAPI cudaMemPrefetchAsync(const void* devPtr, size_t count, int dstDevice, cudaStream_t stream) {
#endif
    ENTER();
    return record(cuMemPrefetchAsync(reinterpret_cast<CUdeviceptr>(devPtr), count, 0, stream));
}
#if VULKANIZER_CUDART_MAJOR < 13
cudaError_t CUDARTAPI cudaMemPrefetchAsync_v2(const void* devPtr, size_t count, cudaMemLocation location,
                                              unsigned int flags, cudaStream_t stream) {
    ENTER();
    return record(cuMemPrefetchAsync(reinterpret_cast<CUdeviceptr>(devPtr), count, 0, stream));
}
#endif
cudaError_t CUDARTAPI cudaStreamAttachMemAsync(cudaStream_t stream, void* devPtr, size_t length, unsigned int flags) {
    ENTER();
    return record(cuStreamAttachMemAsync(stream, reinterpret_cast<CUdeviceptr>(devPtr), length, flags));
}
#if VULKANIZER_CUDART_MAJOR >= 13
cudaError_t CUDARTAPI cudaMemAdvise(const void* devPtr, size_t count, int advice, cudaMemLocation location) {
#else
cudaError_t CUDARTAPI cudaMemAdvise(const void* devPtr, size_t count, int advice, int device) {
#endif
    ENTER();
    return cudaSuccess;  // a hint
}

// ---- devices: peers, flags, occupancy ---------------------------------------------------------

cudaError_t CUDARTAPI cudaSetDeviceFlags(unsigned int flags) {
    ENTER();
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaGetDeviceFlags(unsigned int* flags) {
    ENTER();
    if (!flags) return record(cudaErrorInvalidValue);
    *flags = 0;
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaDeviceCanAccessPeer(int* canAccessPeer, int device, int peerDevice) {
    ENTER();
    if (!canAccessPeer) return record(cudaErrorInvalidValue);
    *canAccessPeer = 0;
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags) {
    ENTER();
    return record(cudaErrorPeerAccessUnsupported);
}

cudaError_t CUDARTAPI cudaDeviceDisablePeerAccess(int peerDevice) {
    ENTER();
    return record(cudaErrorPeerAccessNotEnabled);
}

cudaError_t CUDARTAPI cudaDeviceGetPCIBusId(char* pciBusId, int len, int device) {
    ENTER();
    if (!pciBusId || len <= 0) return record(cudaErrorInvalidValue);
    std::snprintf(pciBusId, size_t(len), "%04x:%02x:%02x.0", attr(50, device), attr(33, device), attr(34, device));
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, const void* func,
                                                                            int blockSize, size_t dynamicSMemSize,
                                                                            unsigned int flags) {
    ENTER();
    Kernel* k = findKernel(func);
    if (!k) return record(cudaErrorInvalidDeviceFunction);
    CUfunction f;
    if (cudaError_t e = resolve(k, f)) return record(e);
    // The driver's numbers: cooperative launches are checked against them.
    return record(cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, f, blockSize, dynamicSMemSize, flags));
}

cudaError_t CUDARTAPI cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, const void* func, int blockSize,
                                                                   size_t dynamicSMemSize) {
    return cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, func, blockSize, dynamicSMemSize, 0);
}

// ---- extended launches ------------------------------------------------------------------------

cudaError_t CUDARTAPI cudaLaunchKernelExC(const cudaLaunchConfig_t* config, const void* func, void** args) {
    ENTER();
    if (!config) return record(cudaErrorInvalidValue);
    if (config->numAttrs && !config->attrs) return record(cudaErrorInvalidValue);
    return launch(findKernel(func), config->gridDim, config->blockDim, args, config->dynamicSmemBytes, config->stream,
                  config->attrs, config->numAttrs);
}

cudaError_t CUDARTAPI cudaLaunchKernelExC_ptsz(const cudaLaunchConfig_t* config, const void* func, void** args) {
    return cudaLaunchKernelExC(config, func, args);
}

// Grid-wide synchronization (cooperative groups): the driver checks that all blocks fit at once.
cudaError_t CUDARTAPI cudaLaunchCooperativeKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args,
                                                  size_t sharedMem, cudaStream_t stream) {
    ENTER();
    Kernel* k = findKernel(func);
    if (!k) return record(cudaErrorInvalidDeviceFunction);
    CUfunction f;
    if (cudaError_t e = resolve(k, f)) return record(e);
    const CUresult r = cuLaunchCooperativeKernel(f, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z,
                                                 unsigned(sharedMem), stream, args);
    return record(r == CUDA_ERROR_INVALID_VALUE ? cudaErrorInvalidConfiguration : toCudaError(r));
}

cudaError_t CUDARTAPI cudaLaunchCooperativeKernel_ptsz(const void* func, dim3 gridDim, dim3 blockDim, void** args,
                                                       size_t sharedMem, cudaStream_t stream) {
    return cudaLaunchCooperativeKernel(func, gridDim, blockDim, args, sharedMem, stream);
}

cudaError_t CUDARTAPI cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void* userData) {
    ENTER();
    return record(cuLaunchHostFunc(stream, fn, userData));  // captured, or run after the stream's work
}

// ---- graphs and stream capture ------------------------------------------------------------------
// The driver keeps the graphs (api_graph.cpp): capture works for every launch path, the runtime's
// and the Driver API's alike. Kernel nodes name host functions here and CUfunctions there.

}  // extern "C"

namespace {

cudaError_t toDriverCopy(const cudaMemcpy3DParms* p, CUDA_MEMCPY3D& c);  // with the copies below

cudaError_t toDriverKernelParams(const cudaKernelNodeParams* p, CUDA_KERNEL_NODE_PARAMS_v2& out) {
    if (!p || !p->func) return cudaErrorInvalidValue;
    Kernel* k = findKernel(p->func);
    if (!k) return cudaErrorInvalidDeviceFunction;
    CUfunction f;
    if (cudaError_t e = resolve(k, f)) return e;
    out = {f, p->gridDim.x, p->gridDim.y, p->gridDim.z, p->blockDim.x, p->blockDim.y, p->blockDim.z,
           p->sharedMemBytes, p->kernelParams, p->extra, nullptr, nullptr};
    return cudaSuccess;
}

// cudaGraphNodeParams as the driver's CUgraphNodeParams (the outputs are copied back by addNode).
cudaError_t toDriverNodeParams(const cudaGraphNodeParams* p, CUgraphNodeParams& out) {
    if (!p) return cudaErrorInvalidValue;
    out = CUgraphNodeParams{};
    out.type = static_cast<CUgraphNodeType>(p->type);
    switch (p->type) {
        case cudaGraphNodeTypeKernel: {
            const cudaKernelNodeParamsV2& k = p->kernel;
#if VULKANIZER_CUDART_MAJOR >= 13
            if (k.functionType == cudaKernelFunctionTypeKernel || k.functionType == cudaKernelFunctionTypeFunction) {
                out.kernel = {static_cast<CUfunction>(k.func), k.gridDim.x, k.gridDim.y, k.gridDim.z, k.blockDim.x,
                              k.blockDim.y, k.blockDim.z, k.sharedMemBytes, k.kernelParams, k.extra, nullptr, nullptr};
                return cudaSuccess;
            }
#endif
            const cudaKernelNodeParams v1{k.func, k.gridDim, k.blockDim, k.sharedMemBytes, k.kernelParams, k.extra};
            return toDriverKernelParams(&v1, out.kernel);
        }
        case cudaGraphNodeTypeMemcpy: return toDriverCopy(&p->memcpy.copyParams, out.memcpy.copyParams);
        case cudaGraphNodeTypeMemset:
            out.memset = {reinterpret_cast<CUdeviceptr>(p->memset.dst), p->memset.pitch, p->memset.value,
                          p->memset.elementSize, p->memset.width, p->memset.height, nullptr};
            return cudaSuccess;
        case cudaGraphNodeTypeHost: out.host = {p->host.fn, p->host.userData, 0, nullptr}; return cudaSuccess;
        case cudaGraphNodeTypeGraph:
            out.graph = {p->graph.graph, static_cast<CUgraphChildGraphNodeOwnership>(p->graph.ownership)};
            return cudaSuccess;
        case cudaGraphNodeTypeEmpty: return cudaSuccess;
        case cudaGraphNodeTypeWaitEvent: out.eventWait.event = p->eventWait.event; return cudaSuccess;
        case cudaGraphNodeTypeEventRecord: out.eventRecord.event = p->eventRecord.event; return cudaSuccess;
        case cudaGraphNodeTypeMemAlloc: out.alloc = p->alloc; return cudaSuccess;
        case cudaGraphNodeTypeMemFree: out.free.dptr = reinterpret_cast<CUdeviceptr>(p->free.dptr); return cudaSuccess;
        case cudaGraphNodeTypeConditional:
            out.conditional = {p->conditional.handle, static_cast<CUgraphConditionalNodeType>(p->conditional.type),
                               p->conditional.size, p->conditional.phGraph_out, nullptr};
            return cudaSuccess;
        default: return cudaErrorNotSupported;
    }
}

cudaError_t addNode(cudaGraphNode_t* node, cudaGraph_t graph, const cudaGraphNode_t* deps, const void* edgeData,
                    size_t numDeps, cudaGraphNodeParams* params) {
    CUgraphNodeParams p;
    if (cudaError_t e = toDriverNodeParams(params, p)) return e;
    if (CUresult r = cuGraphAddNode_v2(node, graph, deps, static_cast<const CUgraphEdgeData*>(edgeData), numDeps, &p))
        return toCudaError(r);
    if (params->type == cudaGraphNodeTypeMemAlloc) params->alloc.dptr = p.alloc.dptr;  // out
    if (params->type == cudaGraphNodeTypeConditional) params->conditional.phGraph_out = p.conditional.phGraph_out;
    return cudaSuccess;
}

}  // namespace

extern "C" {

cudaError_t CUDARTAPI cudaGraphCreate(cudaGraph_t* graph, unsigned int flags) {
    ENTER();
    return record(cuGraphCreate(graph, flags));
}

cudaError_t CUDARTAPI cudaGraphAddKernelNode(cudaGraphNode_t* node, cudaGraph_t graph, const cudaGraphNode_t* deps,
                                             size_t numDeps, const cudaKernelNodeParams* params) {
    ENTER();
    CUDA_KERNEL_NODE_PARAMS_v2 p;
    if (cudaError_t e = toDriverKernelParams(params, p)) return record(e);
    return record(cuGraphAddKernelNode_v2(node, graph, deps, numDeps, &p));
}

// The runtime's parameter structs of these nodes (cudaMemsetParams, cudaHostNodeParams,
// cudaMemAllocNodeParams) have the driver's layouts.
cudaError_t CUDARTAPI cudaGraphAddEmptyNode(cudaGraphNode_t* node, cudaGraph_t graph, const cudaGraphNode_t* deps,
                                            size_t numDeps) {
    ENTER();
    return record(cuGraphAddEmptyNode(node, graph, deps, numDeps));
}
cudaError_t CUDARTAPI cudaGraphAddHostNode(cudaGraphNode_t* node, cudaGraph_t graph, const cudaGraphNode_t* deps,
                                           size_t numDeps, const void* params) {
    ENTER();
    return record(cuGraphAddHostNode(node, graph, deps, numDeps, static_cast<const CUDA_HOST_NODE_PARAMS*>(params)));
}
cudaError_t CUDARTAPI cudaGraphAddMemsetNode(cudaGraphNode_t* node, cudaGraph_t graph, const cudaGraphNode_t* deps,
                                             size_t numDeps, const void* params) {
    ENTER();
    CUcontext ctx = nullptr;
    cuCtxGetCurrent(&ctx);
    return record(cuGraphAddMemsetNode(node, graph, deps, numDeps, static_cast<const CUDA_MEMSET_NODE_PARAMS*>(params), ctx));
}
cudaError_t CUDARTAPI cudaGraphAddMemAllocNode(cudaGraphNode_t* node, cudaGraph_t graph, const cudaGraphNode_t* deps,
                                               size_t numDeps, void* params) {
    ENTER();
    return record(cuGraphAddMemAllocNode(node, graph, deps, numDeps, static_cast<CUDA_MEM_ALLOC_NODE_PARAMS*>(params)));
}
cudaError_t CUDARTAPI cudaGraphMemAllocNodeGetParams(cudaGraphNode_t node, void* params) {
    ENTER();
    return record(cuGraphMemAllocNodeGetParams(node, static_cast<CUDA_MEM_ALLOC_NODE_PARAMS*>(params)));
}
cudaError_t CUDARTAPI cudaGraphAddMemFreeNode(cudaGraphNode_t* node, cudaGraph_t graph, const cudaGraphNode_t* deps,
                                              size_t numDeps, void* dptr) {
    ENTER();
    return record(cuGraphAddMemFreeNode(node, graph, deps, numDeps, reinterpret_cast<CUdeviceptr>(dptr)));
}
cudaError_t CUDARTAPI cudaGraphMemFreeNodeGetParams(cudaGraphNode_t node, void* dptr) {
    ENTER();
    return record(cuGraphMemFreeNodeGetParams(node, static_cast<CUdeviceptr*>(dptr)));
}
cudaError_t CUDARTAPI cudaGraphAddChildGraphNode(cudaGraphNode_t* node, cudaGraph_t graph, const cudaGraphNode_t* deps,
                                                 size_t numDeps, cudaGraph_t childGraph) {
    ENTER();
    return record(cuGraphAddChildGraphNode(node, graph, deps, numDeps, childGraph));
}
cudaError_t CUDARTAPI cudaDeviceGraphMemTrim(int device) {
    ENTER();
    return record(cuDeviceGraphMemTrim(device));
}

cudaError_t CUDARTAPI cudaGraphKernelNodeSetParams(cudaGraphNode_t node, const cudaKernelNodeParams* params) {
    ENTER();
    CUDA_KERNEL_NODE_PARAMS_v2 p;
    if (cudaError_t e = toDriverKernelParams(params, p)) return record(e);
    return record(cuGraphKernelNodeSetParams_v2(node, &p));
}

cudaError_t CUDARTAPI cudaGraphKernelNodeGetParams(cudaGraphNode_t node, cudaKernelNodeParams* params) {
    ENTER();
    if (!params) return record(cudaErrorInvalidValue);
    CUDA_KERNEL_NODE_PARAMS_v2 p;
    if (CUresult r = cuGraphKernelNodeGetParams_v2(node, &p)) return record(r);
    params->func = nullptr;  // the host function the CUfunction was registered for
    Runtime& rt = Runtime::get();
    std::lock_guard<std::recursive_mutex> lock(rt.mutex);
    for (const auto& [hostFun, k] : rt.kernelsByHostFun)
        for (const auto& [device, f] : k->functions)
            if (f == p.func) params->func = const_cast<void*>(hostFun);
    params->gridDim = dim3(p.gridDimX, p.gridDimY, p.gridDimZ);
    params->blockDim = dim3(p.blockDimX, p.blockDimY, p.blockDimZ);
    params->sharedMemBytes = p.sharedMemBytes;
    params->kernelParams = p.kernelParams;
    params->extra = nullptr;
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaGraphInstantiate(cudaGraphExec_t* exec, cudaGraph_t graph, unsigned long long flags) {
    ENTER();
    return record(cuGraphInstantiateWithFlags(exec, graph, flags));
}

cudaError_t CUDARTAPI cudaGraphInstantiateWithFlags(cudaGraphExec_t* exec, cudaGraph_t graph,
                                                    unsigned long long flags) {
    return cudaGraphInstantiate(exec, graph, flags);
}

cudaError_t CUDARTAPI cudaGraphExecKernelNodeSetParams(cudaGraphExec_t exec, cudaGraphNode_t node,
                                                       const cudaKernelNodeParams* params) {
    ENTER();
    CUDA_KERNEL_NODE_PARAMS_v2 p;
    if (cudaError_t e = toDriverKernelParams(params, p)) return record(e);
    return record(cuGraphExecKernelNodeSetParams_v2(exec, node, &p));
}

// CUDA 12 form: cudaGraphExecUpdateResultInfo has the driver struct's layout.
cudaError_t CUDARTAPI cudaGraphExecUpdate(cudaGraphExec_t exec, cudaGraph_t graph, void* resultInfo) {
    ENTER();
    return record(cuGraphExecUpdate_v2(exec, graph, static_cast<CUgraphExecUpdateResultInfo*>(resultInfo)));
}

cudaError_t CUDARTAPI cudaGraphUpload(cudaGraphExec_t exec, cudaStream_t stream) {
    ENTER();
    return record(cuGraphUpload(exec, stream));
}

cudaError_t CUDARTAPI cudaGraphLaunch(cudaGraphExec_t exec, cudaStream_t stream) {
    ENTER();
    return record(cuGraphLaunch(exec, stream));
}

cudaError_t CUDARTAPI cudaGraphExecDestroy(cudaGraphExec_t exec) {
    ENTER();
    return record(cuGraphExecDestroy(exec));
}

cudaError_t CUDARTAPI cudaGraphDestroy(cudaGraph_t graph) {
    ENTER();
    return record(cuGraphDestroy(graph));
}

cudaError_t CUDARTAPI cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t* nodes, size_t* numNodes) {
    ENTER();
    return record(cuGraphGetNodes(graph, nodes, numNodes));
}

#if VULKANIZER_CUDART_MAJOR >= 13
cudaError_t CUDARTAPI cudaGraphNodeGetDependencies(cudaGraphNode_t node, cudaGraphNode_t* deps, void* edgeData,
                                                   size_t* numDeps) {
    ENTER();
    return record(cuGraphNodeGetDependencies_v2(node, deps, static_cast<CUgraphEdgeData*>(edgeData), numDeps));
}
#else
cudaError_t CUDARTAPI cudaGraphNodeGetDependencies(cudaGraphNode_t node, cudaGraphNode_t* deps, size_t* numDeps) {
    ENTER();
    return record(cuGraphNodeGetDependencies_v2(node, deps, nullptr, numDeps));
}
#endif

cudaError_t CUDARTAPI cudaGraphDebugDotPrint(cudaGraph_t graph, const char* path, unsigned int flags) {
    ENTER();
    return record(cudaErrorNotSupported);
}

cudaError_t CUDARTAPI cudaStreamIsCapturing(cudaStream_t stream, enum cudaStreamCaptureStatus* status) {
    ENTER();
    return record(cuStreamIsCapturing(stream, reinterpret_cast<CUstreamCaptureStatus*>(status)));
}

cudaError_t CUDARTAPI cudaStreamBeginCapture(cudaStream_t stream, int mode) {
    ENTER();
    return record(cuStreamBeginCapture_v2(stream, static_cast<CUstreamCaptureMode>(mode)));
}

cudaError_t CUDARTAPI cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* graph) {
    ENTER();
    return record(cuStreamEndCapture(stream, graph));
}

// CUDA 13's cudaStreamGetCaptureInfo is CUDA 12's _v3.
#if VULKANIZER_CUDART_MAJOR >= 13
cudaError_t CUDARTAPI cudaStreamGetCaptureInfo(cudaStream_t stream, enum cudaStreamCaptureStatus* status,
#else
cudaError_t CUDARTAPI cudaStreamGetCaptureInfo_v3(cudaStream_t stream, enum cudaStreamCaptureStatus* status,
#endif
                                                  unsigned long long* id, cudaGraph_t* graph,
                                                  const cudaGraphNode_t** deps, const void** edgeData,
                                                  size_t* numDeps) {
    ENTER();
    return record(cuStreamGetCaptureInfo_v3(stream, reinterpret_cast<CUstreamCaptureStatus*>(status),
                                            reinterpret_cast<cuuint64_t*>(id), graph, deps,
                                            reinterpret_cast<const CUgraphEdgeData**>(edgeData), numDeps));
}

#if VULKANIZER_CUDART_MAJOR < 13
cudaError_t CUDARTAPI cudaStreamGetCaptureInfo_v2(cudaStream_t stream, enum cudaStreamCaptureStatus* status,
                                                  unsigned long long* id, cudaGraph_t* graph,
                                                  const cudaGraphNode_t** deps, size_t* numDeps) {
    return cudaStreamGetCaptureInfo_v3(stream, status, id, graph, deps, nullptr, numDeps);
}
#endif

cudaError_t CUDARTAPI cudaThreadExchangeStreamCaptureMode(int* mode) {
    ENTER();
    return record(cuThreadExchangeStreamCaptureMode(reinterpret_cast<CUstreamCaptureMode*>(mode)));
}

cudaError_t CUDARTAPI cudaStreamBeginCaptureToGraph(cudaStream_t stream, cudaGraph_t graph,
                                                    const cudaGraphNode_t* deps, const void* edgeData,
                                                    size_t numDeps, int mode) {
    ENTER();
    return record(cuStreamBeginCaptureToGraph(stream, graph, deps, static_cast<const CUgraphEdgeData*>(edgeData), numDeps,
                                              static_cast<CUstreamCaptureMode>(mode)));
}
cudaError_t CUDARTAPI cudaStreamBeginCaptureToGraph_ptsz(cudaStream_t stream, cudaGraph_t graph,
                                                         const cudaGraphNode_t* deps, const void* edgeData,
                                                         size_t numDeps, int mode) {
    return cudaStreamBeginCaptureToGraph(stream, graph, deps, edgeData, numDeps, mode);
}

// CUDA 12's _v2 forms are CUDA 13's plain names.
#if VULKANIZER_CUDART_MAJOR >= 13
#define VULKANIZER_V2(name) name
#else
#define VULKANIZER_V2(name) name##_v2
#endif

cudaError_t CUDARTAPI VULKANIZER_V2(cudaStreamUpdateCaptureDependencies)(cudaStream_t stream, cudaGraphNode_t* deps,
                                                                        const void* edgeData, size_t numDeps,
                                                                        unsigned int flags) {
    ENTER();
    return record(cuStreamUpdateCaptureDependencies_v2(stream, deps, static_cast<const CUgraphEdgeData*>(edgeData),
                                                       numDeps, flags));
}

#if VULKANIZER_CUDART_MAJOR < 13
cudaError_t CUDARTAPI cudaStreamUpdateCaptureDependencies(cudaStream_t stream, cudaGraphNode_t* deps, size_t numDeps,
                                                          unsigned int flags) {
    ENTER();
    return record(cuStreamUpdateCaptureDependencies_v2(stream, deps, nullptr, numDeps, flags));
}
#endif

cudaError_t CUDARTAPI VULKANIZER_V2(cudaGraphAddNode)(cudaGraphNode_t* node, cudaGraph_t graph, const cudaGraphNode_t* deps,
                                                      const void* edgeData, size_t numDeps, cudaGraphNodeParams* params) {
    ENTER();
    return record(addNode(node, graph, deps, edgeData, numDeps, params));
}
#if VULKANIZER_CUDART_MAJOR < 13
cudaError_t CUDARTAPI cudaGraphAddNode(cudaGraphNode_t* node, cudaGraph_t graph, const cudaGraphNode_t* deps,
                                       size_t numDeps, cudaGraphNodeParams* params) {
    ENTER();
    return record(addNode(node, graph, deps, nullptr, numDeps, params));
}
#endif

cudaError_t CUDARTAPI cudaGraphNodeSetParams(cudaGraphNode_t node, cudaGraphNodeParams* params) {
    ENTER();
    CUgraphNodeParams p;
    if (cudaError_t e = toDriverNodeParams(params, p)) return record(e);
    return record(cuGraphNodeSetParams(node, &p));
}

cudaError_t CUDARTAPI cudaGraphExecNodeSetParams(cudaGraphExec_t exec, cudaGraphNode_t node, cudaGraphNodeParams* params) {
    ENTER();
    CUgraphNodeParams p;
    if (cudaError_t e = toDriverNodeParams(params, p)) return record(e);
    return record(cuGraphExecNodeSetParams(exec, node, &p));
}

cudaError_t CUDARTAPI cudaGraphConditionalHandleCreate(cudaGraphConditionalHandle* handle, cudaGraph_t graph,
                                                       unsigned int defaultValue, unsigned int flags) {
    ENTER();
    return record(cuGraphConditionalHandleCreate(reinterpret_cast<CUgraphConditionalHandle*>(handle), graph, nullptr, defaultValue, flags));
}
#if VULKANIZER_CUDART_MAJOR >= 13
cudaError_t CUDARTAPI cudaGraphConditionalHandleCreate_v2(cudaGraphConditionalHandle* handle, cudaGraph_t graph,
                                                          void* ctx, unsigned int defaultValue, unsigned int flags) {
    return cudaGraphConditionalHandleCreate(handle, graph, defaultValue, flags);
}
#endif

// Memcpy, event nodes; memset and host nodes' parameters have the driver's layouts.
cudaError_t CUDARTAPI cudaGraphAddMemcpyNode(cudaGraphNode_t* node, cudaGraph_t graph, const cudaGraphNode_t* deps,
                                             size_t numDeps, const cudaMemcpy3DParms* params) {
    ENTER();
    CUDA_MEMCPY3D c;
    if (cudaError_t e = toDriverCopy(params, c)) return record(e);
    return record(cuGraphAddMemcpyNode(node, graph, deps, numDeps, &c, nullptr));
}
static CUDA_MEMCPY3D linearCopy(void* dst, const void* src, size_t count) {
    CUDA_MEMCPY3D c{};
    c.srcMemoryType = c.dstMemoryType = CU_MEMORYTYPE_UNIFIED;
    c.srcDevice = reinterpret_cast<CUdeviceptr>(src);
    c.dstDevice = reinterpret_cast<CUdeviceptr>(dst);
    c.WidthInBytes = count;
    c.Height = c.Depth = 1;
    return c;
}
cudaError_t CUDARTAPI cudaGraphAddMemcpyNode1D(cudaGraphNode_t* node, cudaGraph_t graph, const cudaGraphNode_t* deps,
                                               size_t numDeps, void* dst, const void* src, size_t count, int) {
    ENTER();
    const CUDA_MEMCPY3D c = linearCopy(dst, src, count);
    return record(cuGraphAddMemcpyNode(node, graph, deps, numDeps, &c, nullptr));
}
cudaError_t CUDARTAPI cudaGraphMemcpyNodeSetParams(cudaGraphNode_t node, const cudaMemcpy3DParms* params) {
    ENTER();
    CUDA_MEMCPY3D c;
    if (cudaError_t e = toDriverCopy(params, c)) return record(e);
    return record(cuGraphMemcpyNodeSetParams(node, &c));
}
cudaError_t CUDARTAPI cudaGraphMemcpyNodeSetParams1D(cudaGraphNode_t node, void* dst, const void* src, size_t count,
                                                     int) {
    ENTER();
    const CUDA_MEMCPY3D c = linearCopy(dst, src, count);
    return record(cuGraphMemcpyNodeSetParams(node, &c));
}
cudaError_t CUDARTAPI cudaGraphMemsetNodeGetParams(cudaGraphNode_t node, void* params) {
    ENTER();
    return record(cuGraphMemsetNodeGetParams(node, static_cast<CUDA_MEMSET_NODE_PARAMS*>(params)));
}
cudaError_t CUDARTAPI cudaGraphMemsetNodeSetParams(cudaGraphNode_t node, const void* params) {
    ENTER();
    return record(cuGraphMemsetNodeSetParams(node, static_cast<const CUDA_MEMSET_NODE_PARAMS*>(params)));
}
cudaError_t CUDARTAPI cudaGraphHostNodeGetParams(cudaGraphNode_t node, void* params) {
    ENTER();
    return record(cuGraphHostNodeGetParams(node, static_cast<CUDA_HOST_NODE_PARAMS*>(params)));
}
cudaError_t CUDARTAPI cudaGraphHostNodeSetParams(cudaGraphNode_t node, const void* params) {
    ENTER();
    return record(cuGraphHostNodeSetParams(node, static_cast<const CUDA_HOST_NODE_PARAMS*>(params)));
}
cudaError_t CUDARTAPI cudaGraphAddEventRecordNode(cudaGraphNode_t* node, cudaGraph_t graph, const cudaGraphNode_t* deps,
                                                  size_t numDeps, cudaEvent_t event) {
    ENTER();
    return record(cuGraphAddEventRecordNode(node, graph, deps, numDeps, event));
}
cudaError_t CUDARTAPI cudaGraphAddEventWaitNode(cudaGraphNode_t* node, cudaGraph_t graph, const cudaGraphNode_t* deps,
                                                size_t numDeps, cudaEvent_t event) {
    ENTER();
    return record(cuGraphAddEventWaitNode(node, graph, deps, numDeps, event));
}
cudaError_t CUDARTAPI cudaGraphEventRecordNodeGetEvent(cudaGraphNode_t node, cudaEvent_t* event) {
    ENTER();
    return record(cuGraphEventRecordNodeGetEvent(node, event));
}
cudaError_t CUDARTAPI cudaGraphEventRecordNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event) {
    ENTER();
    return record(cuGraphEventRecordNodeSetEvent(node, event));
}
cudaError_t CUDARTAPI cudaGraphEventWaitNodeGetEvent(cudaGraphNode_t node, cudaEvent_t* event) {
    ENTER();
    return record(cuGraphEventWaitNodeGetEvent(node, event));
}
cudaError_t CUDARTAPI cudaGraphEventWaitNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event) {
    ENTER();
    return record(cuGraphEventWaitNodeSetEvent(node, event));
}
cudaError_t CUDARTAPI cudaGraphNodeGetType(cudaGraphNode_t node, int* type) {
    ENTER();
    return record(cuGraphNodeGetType(node, reinterpret_cast<CUgraphNodeType*>(type)));
}

// Edges and cloning. CUDA 13's forms take edge data (CUDA 12's _v2).
cudaError_t CUDARTAPI cudaGraphGetRootNodes(cudaGraph_t graph, cudaGraphNode_t* nodes, size_t* numNodes) {
    ENTER();
    return record(cuGraphGetRootNodes(graph, nodes, numNodes));
}
cudaError_t CUDARTAPI VULKANIZER_V2(cudaGraphGetEdges)(cudaGraph_t graph, cudaGraphNode_t* from, cudaGraphNode_t* to,
                                                       void* edgeData, size_t* numEdges) {
    ENTER();
    return record(cuGraphGetEdges_v2(graph, from, to, static_cast<CUgraphEdgeData*>(edgeData), numEdges));
}
cudaError_t CUDARTAPI VULKANIZER_V2(cudaGraphAddDependencies)(cudaGraph_t graph, const cudaGraphNode_t* from,
                                                              const cudaGraphNode_t* to, const void* edgeData,
                                                              size_t numDeps) {
    ENTER();
    return record(cuGraphAddDependencies_v2(graph, from, to, static_cast<const CUgraphEdgeData*>(edgeData), numDeps));
}
cudaError_t CUDARTAPI VULKANIZER_V2(cudaGraphRemoveDependencies)(cudaGraph_t graph, const cudaGraphNode_t* from,
                                                                 const cudaGraphNode_t* to, const void* edgeData,
                                                                 size_t numDeps) {
    ENTER();
    return record(cuGraphRemoveDependencies_v2(graph, from, to, static_cast<const CUgraphEdgeData*>(edgeData), numDeps));
}
cudaError_t CUDARTAPI VULKANIZER_V2(cudaGraphNodeGetDependentNodes)(cudaGraphNode_t node, cudaGraphNode_t* nodes,
                                                                    void* edgeData, size_t* numNodes) {
    ENTER();
    return record(cuGraphNodeGetDependentNodes_v2(node, nodes, static_cast<CUgraphEdgeData*>(edgeData), numNodes));
}
#if VULKANIZER_CUDART_MAJOR < 13
cudaError_t CUDARTAPI cudaGraphGetEdges(cudaGraph_t graph, cudaGraphNode_t* from, cudaGraphNode_t* to, size_t* numEdges) {
    return cudaGraphGetEdges_v2(graph, from, to, nullptr, numEdges);
}
cudaError_t CUDARTAPI cudaGraphAddDependencies(cudaGraph_t graph, const cudaGraphNode_t* from, const cudaGraphNode_t* to,
                                               size_t numDeps) {
    return cudaGraphAddDependencies_v2(graph, from, to, nullptr, numDeps);
}
cudaError_t CUDARTAPI cudaGraphRemoveDependencies(cudaGraph_t graph, const cudaGraphNode_t* from,
                                                  const cudaGraphNode_t* to, size_t numDeps) {
    return cudaGraphRemoveDependencies_v2(graph, from, to, nullptr, numDeps);
}
cudaError_t CUDARTAPI cudaGraphNodeGetDependentNodes(cudaGraphNode_t node, cudaGraphNode_t* nodes, size_t* numNodes) {
    return cudaGraphNodeGetDependentNodes_v2(node, nodes, nullptr, numNodes);
}
cudaError_t CUDARTAPI cudaGraphNodeGetDependencies_v2(cudaGraphNode_t node, cudaGraphNode_t* deps, void* edgeData,
                                                      size_t* numDeps) {
    ENTER();
    return record(cuGraphNodeGetDependencies_v2(node, deps, static_cast<CUgraphEdgeData*>(edgeData), numDeps));
}
#endif
cudaError_t CUDARTAPI cudaGraphDestroyNode(cudaGraphNode_t node) {
    ENTER();
    return record(cuGraphDestroyNode(node));
}
cudaError_t CUDARTAPI cudaGraphClone(cudaGraph_t* clone, cudaGraph_t original) {
    ENTER();
    return record(cuGraphClone(clone, original));
}
cudaError_t CUDARTAPI cudaGraphNodeFindInClone(cudaGraphNode_t* node, cudaGraphNode_t original, cudaGraph_t clone) {
    ENTER();
    return record(cuGraphNodeFindInClone(node, original, clone));
}

cudaError_t CUDARTAPI cudaUserObjectCreate(cudaUserObject_t* object, void* ptr, cudaHostFn_t destroy,
                                           unsigned int initialRefcount, unsigned int flags) {
    ENTER();
    return record(cuUserObjectCreate(object, ptr, destroy, initialRefcount, flags));
}
cudaError_t CUDARTAPI cudaUserObjectRetain(cudaUserObject_t object, unsigned int count) {
    ENTER();
    return record(cuUserObjectRetain(object, count));
}
cudaError_t CUDARTAPI cudaUserObjectRelease(cudaUserObject_t object, unsigned int count) {
    ENTER();
    return record(cuUserObjectRelease(object, count));
}
cudaError_t CUDARTAPI cudaGraphRetainUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned int count,
                                                unsigned int flags) {
    ENTER();
    return record(cuGraphRetainUserObject(graph, object, count, flags));
}
cudaError_t CUDARTAPI cudaGraphReleaseUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned int count) {
    ENTER();
    return record(cuGraphReleaseUserObject(graph, object, count));
}

// Graphs allocate no memory of their own: every attribute (usage, high-water marks) is 0.
cudaError_t CUDARTAPI cudaDeviceGetGraphMemAttribute(int device, int attr, void* value) {
    ENTER();
    if (!value) return record(cudaErrorInvalidValue);
    std::memset(value, 0, sizeof(uint64_t));
    return cudaSuccess;
}
cudaError_t CUDARTAPI cudaDeviceSetGraphMemAttribute(int device, int attr, void* value) {
    ENTER();
    return value ? cudaSuccess : record(cudaErrorInvalidValue);
}

// ---- libraries: modules loaded at run time, kernels looked up by name ---------------------------

}  // extern "C"

struct CUlib_st {
    CUmodule module = nullptr;
    int device = 0;
    std::vector<std::unique_ptr<CUkern_st>> kernels;
};

extern "C" {

cudaError_t CUDARTAPI cudaLibraryLoadData(cudaLibrary_t* library, const void* code, void* jitOptions,
                                          void** jitOptionValues, unsigned int numJitOptions, void* libraryOptions,
                                          void** libraryOptionValues, unsigned int numLibraryOptions) {
    ENTER();
    if (!library || !code) return record(cudaErrorInvalidValue);
    auto lib = std::make_unique<CUlib_st>();
    lib->device = t_device;
    if (CUresult r = cuModuleLoadData(&lib->module, code)) return record(r);
    *library = lib.release();
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaLibraryGetKernel(cudaKernel_t* kernel, cudaLibrary_t library, const char* name) {
    ENTER();
    if (!kernel || !library || !name) return record(cudaErrorInvalidValue);
    for (const auto& k : library->kernels)
        if (k->name == name) {
            *kernel = k.get();
            return cudaSuccess;
        }
    CUfunction f;
    if (CUresult r = cuModuleGetFunction(&f, library->module, name))
        return record(r == CUDA_ERROR_NOT_FOUND ? cudaErrorSymbolNotFound : toCudaError(r));
    auto k = std::make_unique<CUkern_st>();
    k->name = name;
    k->functions[library->device] = f;
    Runtime& rt = Runtime::get();
    {
        std::lock_guard<std::recursive_mutex> lock(rt.mutex);
        rt.kernelsByHostFun[k.get()] = k.get();  // launchable as cudaLaunchKernel((const void*)kernel, ...)
    }
    *kernel = k.get();
    library->kernels.push_back(std::move(k));
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaLibraryUnload(cudaLibrary_t library) {
    ENTER();
    if (!library) return record(cudaErrorInvalidValue);
    Runtime& rt = Runtime::get();
    {
        std::lock_guard<std::recursive_mutex> lock(rt.mutex);
        for (const auto& k : library->kernels) rt.kernelsByHostFun.erase(k.get());
    }
    const CUresult r = cuModuleUnload(library->module);
    delete library;
    return record(r);
}

cudaError_t CUDARTAPI cudaKernelSetAttributeForDevice(cudaKernel_t kernel, int attr, int value, int device) {
    ENTER();
    return kernel ? cudaSuccess : record(cudaErrorInvalidValue);  // as cudaFuncSetAttribute: hints only
}

// ---- arrays, pitched memory, 2D/3D copies, textures and surfaces (the driver's) ----------------

}  // extern "C"

namespace {

// A channel format as the driver's array format and channel count (formats of 3 channels and
// the packed/normalized kinds are not supported).
bool driverFormat(const cudaChannelFormatDesc& d, CUarray_format& format, unsigned& channels) {
    channels = (d.x > 0) + (d.y > 0) + (d.z > 0) + (d.w > 0);
    if (channels != 1 && channels != 2 && channels != 4) return false;
    const int bits = d.x;
    switch (d.f) {
    case cudaChannelFormatKindSigned:
        format = bits == 8 ? CU_AD_FORMAT_SIGNED_INT8 : bits == 16 ? CU_AD_FORMAT_SIGNED_INT16 : CU_AD_FORMAT_SIGNED_INT32;
        return bits == 8 || bits == 16 || bits == 32;
    case cudaChannelFormatKindUnsigned:
        format = bits == 8 ? CU_AD_FORMAT_UNSIGNED_INT8 : bits == 16 ? CU_AD_FORMAT_UNSIGNED_INT16 : CU_AD_FORMAT_UNSIGNED_INT32;
        return bits == 8 || bits == 16 || bits == 32;
    case cudaChannelFormatKindFloat:
        format = bits == 16 ? CU_AD_FORMAT_HALF : CU_AD_FORMAT_FLOAT;
        return bits == 16 || bits == 32;
    default: return false;
    }
}

cudaChannelFormatDesc channelDesc(CUarray_format f, unsigned channels) {
    cudaChannelFormatDesc d{};
    int bits = 32;
    switch (f) {
    case CU_AD_FORMAT_UNSIGNED_INT8: d.f = cudaChannelFormatKindUnsigned, bits = 8; break;
    case CU_AD_FORMAT_UNSIGNED_INT16: d.f = cudaChannelFormatKindUnsigned, bits = 16; break;
    case CU_AD_FORMAT_UNSIGNED_INT32: d.f = cudaChannelFormatKindUnsigned; break;
    case CU_AD_FORMAT_SIGNED_INT8: d.f = cudaChannelFormatKindSigned, bits = 8; break;
    case CU_AD_FORMAT_SIGNED_INT16: d.f = cudaChannelFormatKindSigned, bits = 16; break;
    case CU_AD_FORMAT_SIGNED_INT32: d.f = cudaChannelFormatKindSigned; break;
    case CU_AD_FORMAT_HALF: d.f = cudaChannelFormatKindFloat, bits = 16; break;
    default: d.f = cudaChannelFormatKindFloat; break;
    }
    d.x = bits;
    d.y = channels > 1 ? bits : 0;
    d.z = channels > 2 ? bits : 0;
    d.w = channels > 3 ? bits : 0;
    return d;
}

// cudaResourceDesc -> CUDA_RESOURCE_DESC
cudaError_t driverResource(const cudaResourceDesc* r, CUDA_RESOURCE_DESC& out) {
    if (!r) return cudaErrorInvalidValue;
    out = {};
    unsigned channels = 0;
    CUarray_format format{};
    switch (r->resType) {
    case cudaResourceTypeArray:
        out.resType = CU_RESOURCE_TYPE_ARRAY;
        out.res.array.hArray = r->res.array.array;
        return cudaSuccess;
    case cudaResourceTypeLinear:
        if (!driverFormat(r->res.linear.desc, format, channels)) return cudaErrorInvalidChannelDescriptor;
        out.resType = CU_RESOURCE_TYPE_LINEAR;
        out.res.linear = {reinterpret_cast<CUdeviceptr>(r->res.linear.devPtr), format, channels, r->res.linear.sizeInBytes};
        return cudaSuccess;
    case cudaResourceTypePitch2D:
        if (!driverFormat(r->res.pitch2D.desc, format, channels)) return cudaErrorInvalidChannelDescriptor;
        out.resType = CU_RESOURCE_TYPE_PITCH2D;
        out.res.pitch2D = {reinterpret_cast<CUdeviceptr>(r->res.pitch2D.devPtr), format, channels, r->res.pitch2D.width,
                           r->res.pitch2D.height, r->res.pitch2D.pitchInBytes};
        return cudaSuccess;
    default: return cudaErrorNotSupported;  // mipmapped arrays
    }
}

// The two sides of an array copy: an array at (x bytes, y), or unified memory with a pitch.
cudaError_t copy2DArray(CUarray dstArray, size_t dstX, size_t dstY, void* dst, size_t dpitch, CUarray srcArray,
                        size_t srcX, size_t srcY, const void* src, size_t spitch, size_t width, size_t height,
                        cudaStream_t stream, bool async) {
    CUDA_MEMCPY2D p{};
    p.srcXInBytes = srcX, p.srcY = srcY;
    if (srcArray) p.srcMemoryType = CU_MEMORYTYPE_ARRAY, p.srcArray = srcArray;
    else p.srcMemoryType = CU_MEMORYTYPE_UNIFIED, p.srcDevice = reinterpret_cast<CUdeviceptr>(src), p.srcPitch = spitch;
    p.dstXInBytes = dstX, p.dstY = dstY;
    if (dstArray) p.dstMemoryType = CU_MEMORYTYPE_ARRAY, p.dstArray = dstArray;
    else p.dstMemoryType = CU_MEMORYTYPE_UNIFIED, p.dstDevice = reinterpret_cast<CUdeviceptr>(dst), p.dstPitch = dpitch;
    p.WidthInBytes = width;
    p.Height = height;
    const CUresult r = async ? cuMemcpy2DAsync(&p, stream) : cuMemcpy2D(&p);
    return r ? toCudaError(r) : cudaSuccess;
}

size_t arrayTexelBytes(CUarray a) {
    CUDA_ARRAY3D_DESCRIPTOR d{};
    if (cuArray3DGetDescriptor(&d, a)) return 0;
    const cudaChannelFormatDesc c = channelDesc(d.Format, d.NumChannels);
    return size_t(c.x / 8) * d.NumChannels;
}

}  // namespace

extern "C" {

cudaChannelFormatDesc CUDARTAPI cudaCreateChannelDesc(int x, int y, int z, int w, cudaChannelFormatKind f) {
    return cudaChannelFormatDesc{x, y, z, w, f};
}

cudaError_t CUDARTAPI cudaMalloc3DArray(cudaArray_t* array, const cudaChannelFormatDesc* desc, cudaExtent extent,
                                        unsigned int flags) {
    ENTER();
    if (!array || !desc) return record(cudaErrorInvalidValue);
    CUDA_ARRAY3D_DESCRIPTOR d{};
    if (!driverFormat(*desc, d.Format, d.NumChannels)) return record(cudaErrorInvalidChannelDescriptor);
    d.Width = extent.width, d.Height = extent.height, d.Depth = extent.depth;
    d.Flags = flags;  // cudaArrayLayered, SurfaceLoadStore, Cubemap, TextureGather: the driver's values
    return record(cuArray3DCreate(array, &d));
}

cudaError_t CUDARTAPI cudaMallocArray(cudaArray_t* array, const cudaChannelFormatDesc* desc, size_t width,
                                      size_t height, unsigned int flags) {
    return cudaMalloc3DArray(array, desc, cudaExtent{width, height, 0}, flags);
}

cudaError_t CUDARTAPI cudaFreeArray(cudaArray_t array) {
    ENTER();
    if (!array) return cudaSuccess;
    return record(cuArrayDestroy(array));
}

cudaError_t CUDARTAPI cudaArrayGetInfo(cudaChannelFormatDesc* desc, cudaExtent* extent, unsigned int* flags,
                                       cudaArray_t array) {
    ENTER();
    CUDA_ARRAY3D_DESCRIPTOR d{};
    if (CUresult r = cuArray3DGetDescriptor(&d, array)) return record(r);
    if (desc) *desc = channelDesc(d.Format, d.NumChannels);
    if (extent) *extent = {d.Width, d.Height, d.Depth};
    if (flags) *flags = d.Flags;
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaGetChannelDesc(cudaChannelFormatDesc* desc, cudaArray_const_t array) {
    ENTER();
    if (!desc) return record(cudaErrorInvalidValue);
    CUDA_ARRAY3D_DESCRIPTOR d{};
    if (CUresult r = cuArray3DGetDescriptor(&d, const_cast<CUarray>(array))) return record(r);
    *desc = channelDesc(d.Format, d.NumChannels);
    return cudaSuccess;
}

// Pitched allocations: rows 256-byte aligned (textures need texturePitchAlignment).
cudaError_t CUDARTAPI cudaMallocPitch(void** devPtr, size_t* pitch, size_t width, size_t height) {
    ENTER();
    if (!devPtr || !pitch) return record(cudaErrorInvalidValue);
    *pitch = (std::max<size_t>(width, 1) + 255) / 256 * 256;
    CUdeviceptr p = 0;
    if (CUresult r = cuMemAlloc(&p, *pitch * std::max<size_t>(height, 1))) return record(r);
    *devPtr = reinterpret_cast<void*>(p);
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaMalloc3D(cudaPitchedPtr* pitchedDevPtr, cudaExtent extent) {
    ENTER();
    if (!pitchedDevPtr) return record(cudaErrorInvalidValue);
    size_t pitch = 0;
    void* p = nullptr;
    if (cudaError_t e = cudaMallocPitch(&p, &pitch, extent.width, extent.height * std::max<size_t>(extent.depth, 1)))
        return e;
    *pitchedDevPtr = {p, pitch, extent.width, extent.height};
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaMemset2DAsync(void* devPtr, size_t pitch, int value, size_t width, size_t height,
                                        cudaStream_t stream) {
    ENTER();
    if (width > pitch) return record(cudaErrorInvalidPitchValue);
    for (size_t row = 0; row < height; ++row)
        if (CUresult r = cuMemsetD8Async(reinterpret_cast<CUdeviceptr>(devPtr) + row * pitch, (unsigned char)value,
                                         width, stream))
            return record(r);
    return cudaSuccess;
}
cudaError_t CUDARTAPI cudaMemset2D(void* devPtr, size_t pitch, int value, size_t width, size_t height) {
    if (cudaError_t e = cudaMemset2DAsync(devPtr, pitch, value, width, height, nullptr)) return e;
    return record(cuStreamSynchronize(nullptr));
}
cudaError_t CUDARTAPI cudaMemset3DAsync(cudaPitchedPtr p, int value, cudaExtent extent, cudaStream_t stream) {
    for (size_t z = 0; z < extent.depth; ++z)
        if (cudaError_t e = cudaMemset2DAsync(static_cast<char*>(p.ptr) + z * p.pitch * p.ysize, p.pitch, value,
                                              extent.width, extent.height, stream))
            return e;
    return cudaSuccess;
}
cudaError_t CUDARTAPI cudaMemset3D(cudaPitchedPtr p, int value, cudaExtent extent) {
    if (cudaError_t e = cudaMemset3DAsync(p, value, extent, nullptr)) return e;
    return record(cuStreamSynchronize(nullptr));
}

cudaError_t CUDARTAPI cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src,
                                               size_t spitch, size_t width, size_t height, cudaMemcpyKind,
                                               cudaStream_t stream) {
    ENTER();
    return record(copy2DArray(dst, wOffset, hOffset, nullptr, 0, nullptr, 0, 0, src, spitch, width, height, stream, true));
}
cudaError_t CUDARTAPI cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src,
                                          size_t spitch, size_t width, size_t height, cudaMemcpyKind) {
    ENTER();
    return record(copy2DArray(dst, wOffset, hOffset, nullptr, 0, nullptr, 0, 0, src, spitch, width, height, nullptr, false));
}
cudaError_t CUDARTAPI cudaMemcpy2DFromArrayAsync(void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset,
                                                 size_t hOffset, size_t width, size_t height, cudaMemcpyKind,
                                                 cudaStream_t stream) {
    ENTER();
    return record(copy2DArray(nullptr, 0, 0, dst, dpitch, const_cast<CUarray>(src), wOffset, hOffset, nullptr, 0, width,
                              height, stream, true));
}
cudaError_t CUDARTAPI cudaMemcpy2DFromArray(void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset,
                                            size_t hOffset, size_t width, size_t height, cudaMemcpyKind) {
    ENTER();
    return record(copy2DArray(nullptr, 0, 0, dst, dpitch, const_cast<CUarray>(src), wOffset, hOffset, nullptr, 0, width,
                              height, nullptr, false));
}
cudaError_t CUDARTAPI cudaMemcpy2DArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst,
                                               cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc,
                                               size_t width, size_t height, cudaMemcpyKind) {
    ENTER();
    return record(copy2DArray(dst, wOffsetDst, hOffsetDst, nullptr, 0, const_cast<CUarray>(src), wOffsetSrc, hOffsetSrc,
                              nullptr, 0, width, height, nullptr, false));
}
// Deprecated linear forms: `count` bytes from (wOffset, hOffset), wrapping rows.
cudaError_t CUDARTAPI cudaMemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src,
                                        size_t count, cudaMemcpyKind kind) {
    ENTER();
    const size_t texel = arrayTexelBytes(dst);
    CUDA_ARRAY3D_DESCRIPTOR d{};
    if (!texel || cuArray3DGetDescriptor(&d, dst)) return record(cudaErrorInvalidValue);
    const size_t offset = hOffset * d.Width * texel + wOffset;
    if (offset == 0 && (count % (d.Width * texel)) == 0)
        return cudaMemcpy2DToArray(dst, 0, 0, src, d.Width * texel, d.Width * texel, count / (d.Width * texel), kind);
    return record(cuMemcpyHtoA(dst, offset, src, count));
}
cudaError_t CUDARTAPI cudaMemcpyFromArray(void* dst, cudaArray_const_t src, size_t wOffset, size_t hOffset,
                                          size_t count, cudaMemcpyKind) {
    ENTER();
    const size_t texel = arrayTexelBytes(const_cast<CUarray>(src));
    CUDA_ARRAY3D_DESCRIPTOR d{};
    if (!texel || cuArray3DGetDescriptor(&d, const_cast<CUarray>(src))) return record(cudaErrorInvalidValue);
    return record(cuMemcpyAtoD(reinterpret_cast<CUdeviceptr>(dst), const_cast<CUarray>(src),
                               hOffset * d.Width * texel + wOffset, count));
}

}  // extern "C"

namespace {

cudaError_t toDriverCopy(const cudaMemcpy3DParms* p, CUDA_MEMCPY3D& c) {
    if (!p) return cudaErrorInvalidValue;
    c = CUDA_MEMCPY3D{};
    // extents are in elements for arrays, in bytes otherwise
    const size_t texel = p->srcArray ? arrayTexelBytes(p->srcArray) : p->dstArray ? arrayTexelBytes(p->dstArray) : 1;
    if (!texel) return cudaErrorInvalidValue;
    if (p->srcArray) {
        c.srcMemoryType = CU_MEMORYTYPE_ARRAY, c.srcArray = p->srcArray;
        c.srcXInBytes = p->srcPos.x * texel;
    } else {
        c.srcMemoryType = CU_MEMORYTYPE_UNIFIED, c.srcDevice = reinterpret_cast<CUdeviceptr>(p->srcPtr.ptr);
        c.srcPitch = p->srcPtr.pitch, c.srcHeight = p->srcPtr.ysize;
        c.srcXInBytes = p->srcPos.x;
    }
    c.srcY = p->srcPos.y, c.srcZ = p->srcPos.z;
    if (p->dstArray) {
        c.dstMemoryType = CU_MEMORYTYPE_ARRAY, c.dstArray = p->dstArray;
        c.dstXInBytes = p->dstPos.x * texel;
    } else {
        c.dstMemoryType = CU_MEMORYTYPE_UNIFIED, c.dstDevice = reinterpret_cast<CUdeviceptr>(p->dstPtr.ptr);
        c.dstPitch = p->dstPtr.pitch, c.dstHeight = p->dstPtr.ysize;
        c.dstXInBytes = p->dstPos.x;
    }
    c.dstY = p->dstPos.y, c.dstZ = p->dstPos.z;
    c.WidthInBytes = (p->srcArray || p->dstArray) ? p->extent.width * texel : p->extent.width;
    c.Height = std::max<size_t>(p->extent.height, 1);
    c.Depth = std::max<size_t>(p->extent.depth, 1);
    return cudaSuccess;
}

}  // namespace

extern "C" {

cudaError_t CUDARTAPI cudaMemcpy3DAsync(const cudaMemcpy3DParms* p, cudaStream_t stream) {
    ENTER();
    CUDA_MEMCPY3D c;
    if (cudaError_t e = toDriverCopy(p, c)) return record(e);
    return record(stream ? cuMemcpy3DAsync(&c, stream) : cuMemcpy3D(&c));
}
cudaError_t CUDARTAPI cudaMemcpy3D(const cudaMemcpy3DParms* p) {
    if (cudaError_t e = cudaMemcpy3DAsync(p, nullptr)) return e;
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaCreateTextureObject(cudaTextureObject_t* texObject, const cudaResourceDesc* resDesc,
                                              const cudaTextureDesc* texDesc, const cudaResourceViewDesc* viewDesc) {
    ENTER();
    if (!texObject || !texDesc) return record(cudaErrorInvalidValue);
    if (viewDesc) return record(cudaErrorNotSupported);
    CUDA_RESOURCE_DESC r;
    if (cudaError_t e = driverResource(resDesc, r)) return record(e);
    CUDA_TEXTURE_DESC t{};
    for (int i = 0; i < 3; ++i) t.addressMode[i] = CUaddress_mode(texDesc->addressMode[i]);
    t.filterMode = CUfilter_mode(texDesc->filterMode);
    if (texDesc->readMode == cudaReadModeElementType) t.flags |= CU_TRSF_READ_AS_INTEGER;
    if (texDesc->normalizedCoords) t.flags |= CU_TRSF_NORMALIZED_COORDINATES;
    if (texDesc->sRGB) t.flags |= CU_TRSF_SRGB;
    std::memcpy(t.borderColor, texDesc->borderColor, sizeof t.borderColor);
    return record(cuTexObjectCreate(texObject, &r, &t, nullptr));
}
cudaError_t CUDARTAPI cudaDestroyTextureObject(cudaTextureObject_t texObject) {
    ENTER();
    return record(cuTexObjectDestroy(texObject));
}
cudaError_t CUDARTAPI cudaCreateSurfaceObject(cudaSurfaceObject_t* surfObject, const cudaResourceDesc* resDesc) {
    ENTER();
    if (!surfObject) return record(cudaErrorInvalidValue);
    CUDA_RESOURCE_DESC r;
    if (cudaError_t e = driverResource(resDesc, r)) return record(e);
    return record(cuSurfObjectCreate(surfObject, &r));
}
cudaError_t CUDARTAPI cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject) {
    ENTER();
    return record(cuSurfObjectDestroy(surfObject));
}

// ---- driver entry points, profiler, IPC ---------------------------------------------------------

cudaError_t CUDARTAPI cudaGetDriverEntryPointByVersion(const char* symbol, void** funcPtr, unsigned int cudaVersion,
                                                       unsigned long long flags,
                                                       enum cudaDriverEntryPointQueryResult* status) {
    ENTER();
    if (!symbol || !funcPtr) return record(cudaErrorInvalidValue);
    CUdriverProcAddressQueryResult r = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
    const CUresult e = cuGetProcAddress_v2(symbol, funcPtr, int(cudaVersion), flags, &r);
    if (status)
        *status = r == CU_GET_PROC_ADDRESS_SUCCESS ? cudaDriverEntryPointSuccess : cudaDriverEntryPointSymbolNotFound;
    // A missing symbol is reported through `status`, not as an error.
    return e == CUDA_SUCCESS || e == CUDA_ERROR_NOT_FOUND ? cudaSuccess : record(e);
}

cudaError_t CUDARTAPI cudaGetDriverEntryPoint(const char* symbol, void** funcPtr, unsigned long long flags,
                                              enum cudaDriverEntryPointQueryResult* status) {
    return cudaGetDriverEntryPointByVersion(symbol, funcPtr, kRuntimeVersion, flags, status);
}

cudaError_t CUDARTAPI cudaProfilerStart(void) { return cudaSuccess; }
cudaError_t CUDARTAPI cudaProfilerStop(void) { return cudaSuccess; }

// Interprocess sharing: the driver's (api_ipc.cpp); the runtime's handles have the same 64 bytes.
static_assert(sizeof(cudaIpcMemHandle_t) == sizeof(CUipcMemHandle) && sizeof(cudaIpcEventHandle_t) == sizeof(CUipcEventHandle));

cudaError_t CUDARTAPI cudaIpcGetMemHandle(cudaIpcMemHandle_t* handle, void* devPtr) {
    ENTER();
    if (!handle) return record(cudaErrorInvalidValue);
    CUipcMemHandle h;
    if (CUresult r = cuIpcGetMemHandle(&h, reinterpret_cast<CUdeviceptr>(devPtr))) return record(r);
    std::memcpy(handle, &h, sizeof h);
    return cudaSuccess;
}
cudaError_t CUDARTAPI cudaIpcOpenMemHandle(void** devPtr, cudaIpcMemHandle_t handle, unsigned int flags) {
    ENTER();
    if (!devPtr) return record(cudaErrorInvalidValue);
    CUipcMemHandle h;
    std::memcpy(&h, &handle, sizeof h);
    CUdeviceptr p = 0;
    if (CUresult r = cuIpcOpenMemHandle_v2(&p, h, flags)) return record(r);
    *devPtr = reinterpret_cast<void*>(p);
    return cudaSuccess;
}
cudaError_t CUDARTAPI cudaIpcCloseMemHandle(void* devPtr) {
    ENTER();
    return record(cuIpcCloseMemHandle(reinterpret_cast<CUdeviceptr>(devPtr)));
}
cudaError_t CUDARTAPI cudaIpcGetEventHandle(cudaIpcEventHandle_t* handle, cudaEvent_t event) {
    ENTER();
    if (!handle) return record(cudaErrorInvalidValue);
    CUipcEventHandle h;
    if (CUresult r = cuIpcGetEventHandle(&h, event)) return record(r);
    std::memcpy(handle, &h, sizeof h);
    return cudaSuccess;
}
cudaError_t CUDARTAPI cudaIpcOpenEventHandle(cudaEvent_t* event, cudaIpcEventHandle_t handle) {
    ENTER();
    CUipcEventHandle h;
    std::memcpy(&h, &handle, sizeof h);
    return record(cuIpcOpenEventHandle(event, h));
}

// ---- memory pools (the driver's; the runtime types share the driver's layouts) --------------------

cudaError_t CUDARTAPI cudaDeviceGetDefaultMemPool(cudaMemPool_t* pool, int device) {
    ENTER();
    return record(cuDeviceGetDefaultMemPool(pool, device));
}
cudaError_t CUDARTAPI cudaDeviceGetMemPool(cudaMemPool_t* pool, int device) {
    ENTER();
    return record(cuDeviceGetMemPool(pool, device));
}
cudaError_t CUDARTAPI cudaDeviceSetMemPool(int device, cudaMemPool_t pool) {
    ENTER();
    return record(cuDeviceSetMemPool(device, pool));
}
cudaError_t CUDARTAPI cudaMemPoolCreate(cudaMemPool_t* pool, const CUmemPoolProps* props) {
    ENTER();
    return record(cuMemPoolCreate(pool, props));
}
cudaError_t CUDARTAPI cudaMemPoolDestroy(cudaMemPool_t pool) {
    ENTER();
    return record(cuMemPoolDestroy(pool));
}
cudaError_t CUDARTAPI cudaMemPoolGetAttribute(cudaMemPool_t pool, int attr, void* value) {
    ENTER();
    return record(cuMemPoolGetAttribute(pool, CUmemPool_attribute(attr), value));
}
cudaError_t CUDARTAPI cudaMemPoolSetAttribute(cudaMemPool_t pool, int attr, void* value) {
    ENTER();
    return record(cuMemPoolSetAttribute(pool, CUmemPool_attribute(attr), value));
}
cudaError_t CUDARTAPI cudaMemPoolSetAccess(cudaMemPool_t pool, const CUmemAccessDesc* descList, size_t count) {
    ENTER();
    return record(cuMemPoolSetAccess(pool, descList, count));
}
cudaError_t CUDARTAPI cudaMemPoolGetAccess(int* flags, cudaMemPool_t pool, CUmemLocation* location) {
    ENTER();
    CUmemAccess_flags f;
    if (CUresult r = cuMemPoolGetAccess(&f, pool, location)) return record(r);
    if (flags) *flags = int(f);
    return cudaSuccess;
}
cudaError_t CUDARTAPI cudaMemPoolTrimTo(cudaMemPool_t pool, size_t minBytesToKeep) {
    ENTER();
    return record(cuMemPoolTrimTo(pool, minBytesToKeep));
}

cudaError_t CUDARTAPI cudaPointerGetAttributes(struct cudaPointerAttributes* attributes, const void* ptr) {
    ENTER();
    if (!attributes) return record(cudaErrorInvalidValue);
    *attributes = {};
    const CUdeviceptr p = reinterpret_cast<CUdeviceptr>(ptr);
    unsigned memoryType = 0;
    int managed = 0;
    if (ptr && cuPointerGetAttribute(&memoryType, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, p) == CUDA_SUCCESS &&
        cuPointerGetAttribute(&managed, CU_POINTER_ATTRIBUTE_IS_MANAGED, p) == CUDA_SUCCESS) {
        attributes->type = managed ? cudaMemoryTypeManaged
                           : memoryType == CU_MEMORYTYPE_HOST ? cudaMemoryTypeHost
                                                              : cudaMemoryTypeDevice;
        int dev = 0;  // the memory's device (unified addressing: maybe not the current one)
        if (cuPointerGetAttribute(&dev, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, p) != CUDA_SUCCESS) {
            CUdevice current = 0;
            cuCtxGetDevice(&current);
            dev = current;
        }
        attributes->device = dev;
        CUdeviceptr d = 0;  // pinned memory the device cannot use has no device pointer
        if (cuPointerGetAttribute(&d, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, p) == CUDA_SUCCESS)
            attributes->devicePointer = reinterpret_cast<void*>(d);
        if (attributes->type != cudaMemoryTypeDevice) attributes->hostPointer = const_cast<void*>(ptr);
    } else {
        attributes->type = cudaMemoryTypeUnregistered;  // plain host memory
        attributes->device = -2;
        attributes->hostPointer = const_cast<void*>(ptr);
    }
    return cudaSuccess;
}

// ---- streams and events -----------------------------------------------------------------------

cudaError_t CUDARTAPI cudaDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority) {
    ENTER();
    if (leastPriority) *leastPriority = 0;  // one queue: priorities have no effect
    if (greatestPriority) *greatestPriority = 0;
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaStreamGetPriority(cudaStream_t stream, int* priority) {
    ENTER();
    if (!priority) return record(cudaErrorInvalidValue);
    *priority = 0;
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream, unsigned int flags) {
    ENTER();
    return record(cuEventRecord(event, stream));
}

cudaError_t CUDARTAPI cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void* userData,
                                            unsigned int flags) {
    ENTER();
    if (!callback) return record(cudaErrorInvalidValue);
    const CUresult r = cuStreamSynchronize(stream);  // runs once the stream's preceding work is done
    callback(stream, toCudaError(r), userData);
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned int flags) {
    ENTER();
    return record(cuStreamCreate(stream, flags));
}

cudaError_t CUDARTAPI cudaStreamCreate(cudaStream_t* stream) { return cudaStreamCreateWithFlags(stream, 0); }

cudaError_t CUDARTAPI cudaStreamCreateWithPriority(cudaStream_t* stream, unsigned int flags, int priority) {
    return cudaStreamCreateWithFlags(stream, flags);
}

cudaError_t CUDARTAPI cudaStreamDestroy(cudaStream_t stream) {
    ENTER();
    return record(cuStreamDestroy(stream));
}

cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t stream) {
    ENTER();
    return record(cuStreamSynchronize(stream));
}

cudaError_t CUDARTAPI cudaStreamQuery(cudaStream_t stream) {
    ENTER();
    return record(cuStreamQuery(stream));
}

cudaError_t CUDARTAPI cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags) {
    ENTER();
    return record(cuStreamWaitEvent(stream, event, flags));
}

cudaError_t CUDARTAPI cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags) {
    ENTER();
    return record(cuEventCreate(event, flags));
}

cudaError_t CUDARTAPI cudaEventCreate(cudaEvent_t* event) { return cudaEventCreateWithFlags(event, 0); }

cudaError_t CUDARTAPI cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    ENTER();
    return record(cuEventRecord(event, stream));
}

cudaError_t CUDARTAPI cudaEventSynchronize(cudaEvent_t event) {
    ENTER();
    return record(cuEventSynchronize(event));
}

cudaError_t CUDARTAPI cudaEventQuery(cudaEvent_t event) {
    ENTER();
    return record(cuEventQuery(event));
}

cudaError_t CUDARTAPI cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end) {
    ENTER();
    return record(cuEventElapsedTime(ms, start, end));
}

cudaError_t CUDARTAPI cudaEventDestroy(cudaEvent_t event) {
    ENTER();
    return record(cuEventDestroy(event));
}

}  // extern "C"
