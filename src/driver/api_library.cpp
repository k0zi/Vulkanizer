// Libraries (CUDA 12's context-independent modules) and the module and function queries next to
// them. A library is a module loaded into the context current at load time (the first device's
// primary context if there is none); its kernels are that module's functions, so a CUkernel is a
// CUfunction and launches as one (cuLaunchKernel takes either, as in CUDA).

#include "driver/driver.h"

#include "common/log.h"

#include <fstream>
#include <map>
#include <sstream>

using namespace vulkanizer;
using namespace vulkanizer::driver;

struct CUlib_st {
    static constexpr uint32_t kMagic = 0x42494c53;  // 'SLIB'
    uint32_t magic = kMagic;
    CUmodule module = nullptr;
};

namespace {

// kernel (function) -> its library, for cuKernelGetLibrary. Under the driver lock.
std::map<CUfunction, CUlibrary>& libraryOf() {
    static std::map<CUfunction, CUlibrary> m;
    return m;
}

CUfunction functionOf(CUkernel k) { return reinterpret_cast<CUfunction>(k); }

}  // namespace

extern "C" {

CUresult CUDAAPI cuModuleGetFunctionCount(unsigned int* count, CUmodule mod) {
    if (!count) return CUDA_ERROR_INVALID_VALUE;
    if (!validHandle(mod)) return CUDA_ERROR_INVALID_HANDLE;
    *count = unsigned(mod->functions.size());
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuModuleEnumerateFunctions(CUfunction* functions, unsigned int numFunctions, CUmodule mod) {
    if (!functions && numFunctions) return CUDA_ERROR_INVALID_VALUE;
    if (!validHandle(mod)) return CUDA_ERROR_INVALID_HANDLE;
    for (unsigned i = 0; i < numFunctions && i < mod->functions.size(); ++i) functions[i] = mod->functions[i].get();
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuFuncGetName(const char** name, CUfunction hfunc) {
    if (!name) return CUDA_ERROR_INVALID_VALUE;
    if (!validHandle(hfunc)) return CUDA_ERROR_INVALID_HANDLE;
    *name = hfunc->entry->name.c_str();
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuFuncGetModule(CUmodule* hmod, CUfunction hfunc) {
    if (!hmod) return CUDA_ERROR_INVALID_VALUE;
    if (!validHandle(hfunc)) return CUDA_ERROR_INVALID_HANDLE;
    *hmod = hfunc->module;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuLibraryLoadData(CUlibrary* library, const void* code, CUjit_option* jitOptions,
                                   void** jitOptionsValues, unsigned int numJitOptions, CUlibraryOption*, void**,
                                   unsigned int) {
    VULKANIZER_TRACE_API();
    if (!library || !code) return CUDA_ERROR_INVALID_VALUE;
    if (CUresult r = cuInit(0)) return r;
    CUcontext current = nullptr, primary = nullptr;
    cuCtxGetCurrent(&current);
    if (!current) {  // no context yet: the first device's primary one, as the runtime would use
        if (CUresult r = cuDevicePrimaryCtxRetain(&primary, 0)) return r;
        cuCtxPushCurrent(primary);
    }
    CUmodule module = nullptr;
    const CUresult r = cuModuleLoadDataEx(&module, code, numJitOptions, jitOptions, jitOptionsValues);
    if (primary) {
        CUcontext popped;
        cuCtxPopCurrent(&popped);
    }
    if (r) return r;
    auto* lib = new CUlib_st();
    lib->module = module;
    std::lock_guard<std::recursive_mutex> lock(Driver::get().mutex());
    Driver::get().registerHandle(lib);
    for (auto& f : module->functions) libraryOf()[f.get()] = lib;
    *library = lib;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuLibraryLoadFromFile(CUlibrary* library, const char* fileName, CUjit_option* jitOptions,
                                       void** jitOptionsValues, unsigned int numJitOptions,
                                       CUlibraryOption* libraryOptions, void** libraryOptionValues,
                                       unsigned int numLibraryOptions) {
    if (!fileName) return CUDA_ERROR_INVALID_VALUE;
    std::ifstream in(fileName, std::ios::binary);
    if (!in) return CUDA_ERROR_FILE_NOT_FOUND;
    std::ostringstream ss;
    ss << in.rdbuf();
    const std::string image = ss.str();  // PTX text gets its terminator from the string
    return cuLibraryLoadData(library, image.c_str(), jitOptions, jitOptionsValues, numJitOptions, libraryOptions,
                             libraryOptionValues, numLibraryOptions);
}

CUresult CUDAAPI cuLibraryUnload(CUlibrary library) {
    VULKANIZER_TRACE_API();
    std::lock_guard<std::recursive_mutex> lock(Driver::get().mutex());
    if (!validHandle(library)) return CUDA_ERROR_INVALID_HANDLE;
    for (auto& f : library->module->functions) libraryOf().erase(f.get());
    const CUresult r = cuModuleUnload(library->module);
    Driver::get().unregisterHandle(library);
    delete library;
    return r;
}

CUresult CUDAAPI cuLibraryGetKernel(CUkernel* pKernel, CUlibrary library, const char* name) {
    if (!pKernel || !name) return CUDA_ERROR_INVALID_VALUE;
    if (!validHandle(library)) return CUDA_ERROR_INVALID_HANDLE;
    CUfunction f;
    if (CUresult r = cuModuleGetFunction(&f, library->module, name)) return r;
    *pKernel = reinterpret_cast<CUkernel>(f);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuLibraryGetKernelCount(unsigned int* count, CUlibrary lib) {
    if (!validHandle(lib)) return CUDA_ERROR_INVALID_HANDLE;
    return cuModuleGetFunctionCount(count, lib->module);
}

CUresult CUDAAPI cuLibraryEnumerateKernels(CUkernel* kernels, unsigned int numKernels, CUlibrary lib) {
    if (!validHandle(lib)) return CUDA_ERROR_INVALID_HANDLE;
    return cuModuleEnumerateFunctions(reinterpret_cast<CUfunction*>(kernels), numKernels, lib->module);
}

CUresult CUDAAPI cuLibraryGetModule(CUmodule* pMod, CUlibrary library) {
    if (!pMod) return CUDA_ERROR_INVALID_VALUE;
    if (!validHandle(library)) return CUDA_ERROR_INVALID_HANDLE;
    *pMod = library->module;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuLibraryGetGlobal(CUdeviceptr* dptr, size_t* bytes, CUlibrary library, const char* name) {
    if (!validHandle(library)) return CUDA_ERROR_INVALID_HANDLE;
    return cuModuleGetGlobal_v2(dptr, bytes, library->module, name);
}

// __managed__ variables are ordinary globals here.
CUresult CUDAAPI cuLibraryGetManaged(CUdeviceptr* dptr, size_t* bytes, CUlibrary library, const char* name) {
    return cuLibraryGetGlobal(dptr, bytes, library, name);
}

CUresult CUDAAPI cuKernelGetFunction(CUfunction* pFunc, CUkernel kernel) {
    if (!pFunc) return CUDA_ERROR_INVALID_VALUE;
    if (!validHandle(functionOf(kernel))) return CUDA_ERROR_INVALID_HANDLE;
    *pFunc = functionOf(kernel);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuKernelGetLibrary(CUlibrary* pLib, CUkernel kernel) {
    if (!pLib) return CUDA_ERROR_INVALID_VALUE;
    std::lock_guard<std::recursive_mutex> lock(Driver::get().mutex());
    auto it = libraryOf().find(functionOf(kernel));
    if (it == libraryOf().end()) return CUDA_ERROR_INVALID_HANDLE;
    *pLib = it->second;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuKernelGetAttribute(int* pi, CUfunction_attribute attrib, CUkernel kernel, CUdevice) {
    return cuFuncGetAttribute(pi, attrib, functionOf(kernel));
}

CUresult CUDAAPI cuKernelSetAttribute(CUfunction_attribute attrib, int val, CUkernel kernel, CUdevice) {
    return cuFuncSetAttribute(functionOf(kernel), attrib, val);
}

CUresult CUDAAPI cuKernelSetCacheConfig(CUkernel kernel, CUfunc_cache config, CUdevice) {
    return cuFuncSetCacheConfig(functionOf(kernel), config);
}

CUresult CUDAAPI cuKernelGetName(const char** name, CUkernel hfunc) { return cuFuncGetName(name, functionOf(hfunc)); }

CUresult CUDAAPI cuKernelGetParamInfo(CUkernel kernel, size_t paramIndex, size_t* paramOffset, size_t* paramSize) {
    return cuFuncGetParamInfo(functionOf(kernel), paramIndex, paramOffset, paramSize);
}

}  // extern "C"
