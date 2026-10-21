/*
 * Vulkanizer - clean-room CUDA Driver API header (subset).
 *
 * Written from the public CUDA Driver API documentation. Type layouts and enum
 * values are ABI and must match NVIDIA's numbering, since unmodified CUDA
 * applications call into libcuda.so with them. Only the subset Vulkanizer
 * implements (or is about to) is declared here.
 */
#ifndef VULKANIZER_CUDA_H
#define VULKANIZER_CUDA_H

#include <stddef.h>
#include <stdint.h>

#define CUDA_VERSION 12040

#ifdef __cplusplus
extern "C" {
#endif

#define CUDAAPI

/* ------------------------------------------------------------------------ */
/* Types                                                                    */
/* ------------------------------------------------------------------------ */

typedef uint32_t cuuint32_t;
typedef uint64_t cuuint64_t;

typedef unsigned long long CUdeviceptr_v2;
typedef CUdeviceptr_v2 CUdeviceptr;
typedef int CUdevice_v1;
typedef CUdevice_v1 CUdevice;

typedef struct CUctx_st* CUcontext;
typedef struct CUmod_st* CUmodule;
typedef struct CUfunc_st* CUfunction;
typedef struct CUstream_st* CUstream;
typedef struct CUevent_st* CUevent;

typedef struct CUuuid_st {
    char bytes[16];
} CUuuid;

typedef enum cudaError_enum {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_INVALID_VALUE = 1,
    CUDA_ERROR_OUT_OF_MEMORY = 2,
    CUDA_ERROR_NOT_INITIALIZED = 3,
    CUDA_ERROR_DEINITIALIZED = 4,
    CUDA_ERROR_PROFILER_DISABLED = 5,
    CUDA_ERROR_STUB_LIBRARY = 34,
    CUDA_ERROR_DEVICE_UNAVAILABLE = 46,
    CUDA_ERROR_NO_DEVICE = 100,
    CUDA_ERROR_INVALID_DEVICE = 101,
    CUDA_ERROR_DEVICE_NOT_LICENSED = 102,
    CUDA_ERROR_INVALID_IMAGE = 200,
    CUDA_ERROR_INVALID_CONTEXT = 201,
    CUDA_ERROR_CONTEXT_ALREADY_CURRENT = 202,
    CUDA_ERROR_MAP_FAILED = 205,
    CUDA_ERROR_UNMAP_FAILED = 206,
    CUDA_ERROR_ARRAY_IS_MAPPED = 207,
    CUDA_ERROR_ALREADY_MAPPED = 208,
    CUDA_ERROR_NO_BINARY_FOR_GPU = 209,
    CUDA_ERROR_ALREADY_ACQUIRED = 210,
    CUDA_ERROR_NOT_MAPPED = 211,
    CUDA_ERROR_UNSUPPORTED_LIMIT = 215,
    CUDA_ERROR_CONTEXT_ALREADY_IN_USE = 216,
    CUDA_ERROR_INVALID_PTX = 218,
    CUDA_ERROR_JIT_COMPILER_NOT_FOUND = 221,
    CUDA_ERROR_UNSUPPORTED_PTX_VERSION = 222,
    CUDA_ERROR_INVALID_SOURCE = 300,
    CUDA_ERROR_FILE_NOT_FOUND = 301,
    CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,
    CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303,
    CUDA_ERROR_OPERATING_SYSTEM = 304,
    CUDA_ERROR_INVALID_HANDLE = 400,
    CUDA_ERROR_ILLEGAL_STATE = 401,
    CUDA_ERROR_NOT_FOUND = 500,
    CUDA_ERROR_NOT_READY = 600,
    CUDA_ERROR_ILLEGAL_ADDRESS = 700,
    CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701,
    CUDA_ERROR_LAUNCH_TIMEOUT = 702,
    CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704,
    CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = 705,
    CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = 708,
    CUDA_ERROR_CONTEXT_IS_DESTROYED = 709,
    CUDA_ERROR_ASSERT = 710,
    CUDA_ERROR_LAUNCH_FAILED = 719,
    CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE = 720,
    CUDA_ERROR_NOT_PERMITTED = 800,
    CUDA_ERROR_NOT_SUPPORTED = 801,
    CUDA_ERROR_UNKNOWN = 999
} CUresult;

typedef enum CUdevice_attribute_enum {
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
    CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,
    CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,
    CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,
    CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,
    CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,
    CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
    CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17,
    CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,
    CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19,
    CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31,
    CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32,
    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,
    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,
    CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35,
    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36,
    CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37,
    CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38,
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39,
    CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40,
    CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,
    CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
    CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78,
    CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79,
    CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82,
    CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97
} CUdevice_attribute;

typedef enum CUfunction_attribute_enum {
    CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0,
    CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1,
    CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2,
    CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3,
    CU_FUNC_ATTRIBUTE_NUM_REGS = 4,
    CU_FUNC_ATTRIBUTE_PTX_VERSION = 5,
    CU_FUNC_ATTRIBUTE_BINARY_VERSION = 6,
    CU_FUNC_ATTRIBUTE_CACHE_MODE_CA = 7,
    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8,
    CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 9
} CUfunction_attribute;

typedef enum CUjit_option_enum {
    CU_JIT_MAX_REGISTERS = 0,
    CU_JIT_THREADS_PER_BLOCK = 1,
    CU_JIT_WALL_TIME = 2,
    CU_JIT_INFO_LOG_BUFFER = 3,
    CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4,
    CU_JIT_ERROR_LOG_BUFFER = 5,
    CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6,
    CU_JIT_OPTIMIZATION_LEVEL = 7
} CUjit_option;

typedef enum CUctx_flags_enum {
    CU_CTX_SCHED_AUTO = 0x00,
    CU_CTX_SCHED_SPIN = 0x01,
    CU_CTX_SCHED_YIELD = 0x02,
    CU_CTX_SCHED_BLOCKING_SYNC = 0x04,
    CU_CTX_MAP_HOST = 0x08,
    CU_CTX_LMEM_RESIZE_TO_MAX = 0x10
} CUctx_flags;

typedef enum CUstream_flags_enum {
    CU_STREAM_DEFAULT = 0x0,
    CU_STREAM_NON_BLOCKING = 0x1
} CUstream_flags;

typedef enum CUevent_flags_enum {
    CU_EVENT_DEFAULT = 0x0,
    CU_EVENT_BLOCKING_SYNC = 0x1,
    CU_EVENT_DISABLE_TIMING = 0x2,
    CU_EVENT_INTERPROCESS = 0x4
} CUevent_flags;

typedef enum CUdriverProcAddress_flags_enum {
    CU_GET_PROC_ADDRESS_DEFAULT = 0,
    CU_GET_PROC_ADDRESS_LEGACY_STREAM = 1 << 0,
    CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM = 1 << 1
} CUdriverProcAddress_flags;

typedef enum CUdriverProcAddressQueryResult_enum {
    CU_GET_PROC_ADDRESS_SUCCESS = 0,
    CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND = 1,
    CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT = 2
} CUdriverProcAddressQueryResult;

#define CU_STREAM_LEGACY ((CUstream)0x1)
#define CU_STREAM_PER_THREAD ((CUstream)0x2)

#define CU_LAUNCH_PARAM_END ((void*)0x00)
#define CU_LAUNCH_PARAM_BUFFER_POINTER ((void*)0x01)
#define CU_LAUNCH_PARAM_BUFFER_SIZE ((void*)0x02)

/* ------------------------------------------------------------------------ */
/* Versioned entry points: the plain names map to the current ABI versions. */
/* ------------------------------------------------------------------------ */

#define cuGetProcAddress cuGetProcAddress_v2
#define cuDeviceTotalMem cuDeviceTotalMem_v2
#define cuDevicePrimaryCtxRelease cuDevicePrimaryCtxRelease_v2
#define cuDevicePrimaryCtxSetFlags cuDevicePrimaryCtxSetFlags_v2
#define cuDevicePrimaryCtxReset cuDevicePrimaryCtxReset_v2
#define cuCtxCreate cuCtxCreate_v2
#define cuCtxDestroy cuCtxDestroy_v2
#define cuCtxPushCurrent cuCtxPushCurrent_v2
#define cuCtxPopCurrent cuCtxPopCurrent_v2
#define cuMemAlloc cuMemAlloc_v2
#define cuMemFree cuMemFree_v2
#define cuMemGetInfo cuMemGetInfo_v2
#define cuMemGetAddressRange cuMemGetAddressRange_v2
#define cuModuleGetGlobal cuModuleGetGlobal_v2
#define cuMemAllocHost cuMemAllocHost_v2
#define cuMemcpyHtoD cuMemcpyHtoD_v2
#define cuMemcpyDtoH cuMemcpyDtoH_v2
#define cuMemcpyDtoD cuMemcpyDtoD_v2
#define cuMemcpyHtoDAsync cuMemcpyHtoDAsync_v2
#define cuMemcpyDtoHAsync cuMemcpyDtoHAsync_v2
#define cuMemcpyDtoDAsync cuMemcpyDtoDAsync_v2
#define cuMemsetD8 cuMemsetD8_v2
#define cuMemsetD16 cuMemsetD16_v2
#define cuMemsetD32 cuMemsetD32_v2
#define cuStreamDestroy cuStreamDestroy_v2
#define cuEventDestroy cuEventDestroy_v2

/* ------------------------------------------------------------------------ */
/* Initialization, version, errors                                          */
/* ------------------------------------------------------------------------ */

CUresult CUDAAPI cuInit(unsigned int Flags);
CUresult CUDAAPI cuDriverGetVersion(int* driverVersion);
CUresult CUDAAPI cuGetErrorName(CUresult error, const char** pStr);
CUresult CUDAAPI cuGetErrorString(CUresult error, const char** pStr);
CUresult CUDAAPI cuGetProcAddress(const char* symbol, void** pfn, int cudaVersion, cuuint64_t flags,
                                  CUdriverProcAddressQueryResult* symbolStatus);

/* ------------------------------------------------------------------------ */
/* Device management                                                        */
/* ------------------------------------------------------------------------ */

CUresult CUDAAPI cuDeviceGet(CUdevice* device, int ordinal);
CUresult CUDAAPI cuDeviceGetCount(int* count);
CUresult CUDAAPI cuDeviceGetName(char* name, int len, CUdevice dev);
CUresult CUDAAPI cuGetExportTable(const void** ppExportTable, const CUuuid* pExportTableId);
CUresult CUDAAPI cuDeviceGetUuid(CUuuid* uuid, CUdevice dev);
CUresult CUDAAPI cuDeviceGetUuid_v2(CUuuid* uuid, CUdevice dev);
CUresult CUDAAPI cuDeviceTotalMem(size_t* bytes, CUdevice dev);
CUresult CUDAAPI cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev);
CUresult CUDAAPI cuDeviceComputeCapability(int* major, int* minor, CUdevice dev);

/* ------------------------------------------------------------------------ */
/* Context management                                                       */
/* ------------------------------------------------------------------------ */

CUresult CUDAAPI cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev);
CUresult CUDAAPI cuDevicePrimaryCtxRelease(CUdevice dev);
CUresult CUDAAPI cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags);
CUresult CUDAAPI cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int* flags, int* active);
CUresult CUDAAPI cuDevicePrimaryCtxReset(CUdevice dev);

CUresult CUDAAPI cuCtxCreate(CUcontext* pctx, unsigned int flags, CUdevice dev);
CUresult CUDAAPI cuCtxDestroy(CUcontext ctx);
CUresult CUDAAPI cuCtxPushCurrent(CUcontext ctx);
CUresult CUDAAPI cuCtxPopCurrent(CUcontext* pctx);
CUresult CUDAAPI cuCtxSetCurrent(CUcontext ctx);
CUresult CUDAAPI cuCtxGetCurrent(CUcontext* pctx);
CUresult CUDAAPI cuCtxGetDevice(CUdevice* device);
CUresult CUDAAPI cuCtxGetFlags(unsigned int* flags);
CUresult CUDAAPI cuCtxSynchronize(void);
CUresult CUDAAPI cuCtxGetApiVersion(CUcontext ctx, unsigned int* version);

/* ------------------------------------------------------------------------ */
/* Memory management                                                        */
/* ------------------------------------------------------------------------ */

CUresult CUDAAPI cuMemGetInfo(size_t* free, size_t* total);
CUresult CUDAAPI cuMemAlloc(CUdeviceptr* dptr, size_t bytesize);
CUresult CUDAAPI cuMemFree(CUdeviceptr dptr);
CUresult CUDAAPI cuMemGetAddressRange(CUdeviceptr* pbase, size_t* psize, CUdeviceptr dptr);
CUresult CUDAAPI cuMemAllocHost(void** pp, size_t bytesize);
CUresult CUDAAPI cuMemFreeHost(void* p);

CUresult CUDAAPI cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount);
CUresult CUDAAPI cuMemcpyHtoD(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount);
CUresult CUDAAPI cuMemcpyDtoH(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount);
CUresult CUDAAPI cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount);
CUresult CUDAAPI cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream);
CUresult CUDAAPI cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream hStream);
CUresult CUDAAPI cuMemcpyDtoHAsync(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream);
CUresult CUDAAPI cuMemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream);

CUresult CUDAAPI cuMemsetD8(CUdeviceptr dstDevice, unsigned char uc, size_t N);
CUresult CUDAAPI cuMemsetD16(CUdeviceptr dstDevice, unsigned short us, size_t N);
CUresult CUDAAPI cuMemsetD32(CUdeviceptr dstDevice, unsigned int ui, size_t N);
CUresult CUDAAPI cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream);
CUresult CUDAAPI cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream);
CUresult CUDAAPI cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream);

/* ------------------------------------------------------------------------ */
/* Modules and execution                                                    */
/* ------------------------------------------------------------------------ */

CUresult CUDAAPI cuModuleLoad(CUmodule* module, const char* fname);
CUresult CUDAAPI cuModuleLoadData(CUmodule* module, const void* image);
CUresult CUDAAPI cuModuleLoadDataEx(CUmodule* module, const void* image, unsigned int numOptions,
                                    CUjit_option* options, void** optionValues);
CUresult CUDAAPI cuModuleLoadFatBinary(CUmodule* module, const void* fatCubin);
CUresult CUDAAPI cuModuleUnload(CUmodule hmod);
CUresult CUDAAPI cuModuleGetGlobal(CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, const char* name);
CUresult CUDAAPI cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name);

CUresult CUDAAPI cuFuncGetAttribute(int* pi, CUfunction_attribute attrib, CUfunction hfunc);
CUresult CUDAAPI cuFuncGetParamInfo(CUfunction func, size_t paramIndex, size_t* paramOffset, size_t* paramSize);

CUresult CUDAAPI cuLaunchKernel(CUfunction f,
                                unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                                unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                                unsigned int sharedMemBytes, CUstream hStream,
                                void** kernelParams, void** extra);

/* ------------------------------------------------------------------------ */
/* Streams and events                                                       */
/* ------------------------------------------------------------------------ */

CUresult CUDAAPI cuStreamCreate(CUstream* phStream, unsigned int Flags);
CUresult CUDAAPI cuStreamDestroy(CUstream hStream);
CUresult CUDAAPI cuStreamSynchronize(CUstream hStream);
CUresult CUDAAPI cuStreamQuery(CUstream hStream);
CUresult CUDAAPI cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags);

CUresult CUDAAPI cuEventCreate(CUevent* phEvent, unsigned int Flags);
CUresult CUDAAPI cuEventRecord(CUevent hEvent, CUstream hStream);
CUresult CUDAAPI cuEventSynchronize(CUevent hEvent);
CUresult CUDAAPI cuEventQuery(CUevent hEvent);
CUresult CUDAAPI cuEventElapsedTime(float* pMilliseconds, CUevent hStart, CUevent hEnd);
CUresult CUDAAPI cuEventDestroy(CUevent hEvent);

#ifdef __cplusplus
}
#endif

#endif /* VULKANIZER_CUDA_H */
