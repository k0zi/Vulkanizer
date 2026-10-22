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
typedef struct CUmemPoolHandle_st* CUmemoryPool;
typedef unsigned long long CUmemGenericAllocationHandle_v1;
typedef CUmemGenericAllocationHandle_v1 CUmemGenericAllocationHandle;

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
    CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY = 224,
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
    CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712,
    CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 713,
    CUDA_ERROR_LAUNCH_FAILED = 719,
    CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE = 720,
    CUDA_ERROR_NOT_PERMITTED = 800,
    CUDA_ERROR_NOT_SUPPORTED = 801,
    CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED = 900,
    CUDA_ERROR_STREAM_CAPTURE_INVALIDATED = 901,
    CUDA_ERROR_STREAM_CAPTURE_MERGE = 902,
    CUDA_ERROR_STREAM_CAPTURE_UNMATCHED = 903,
    CUDA_ERROR_STREAM_CAPTURE_UNJOINED = 904,
    CUDA_ERROR_STREAM_CAPTURE_ISOLATION = 905,
    CUDA_ERROR_STREAM_CAPTURE_IMPLICIT = 906,
    CUDA_ERROR_CAPTURED_EVENT = 907,
    CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD = 908,
    CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE = 910,
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
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89,
    CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 101,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84,
    CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
    CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED = 99,
    CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 102,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED = 103,
    CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED = 107,
    CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED = 113,
    CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED = 115,
    CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES = 119
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
    CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 9,
    CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET = 10,
    CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH = 11,
    CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT = 12,
    CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH = 13,
    CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED = 14,
    CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE = 15
} CUfunction_attribute;

typedef enum CUfunc_cache_enum {
    CU_FUNC_CACHE_PREFER_NONE = 0x00,
    CU_FUNC_CACHE_PREFER_SHARED = 0x01,
    CU_FUNC_CACHE_PREFER_L1 = 0x02,
    CU_FUNC_CACHE_PREFER_EQUAL = 0x03
} CUfunc_cache;

typedef enum CUlimit_enum {
    CU_LIMIT_STACK_SIZE = 0x00,
    CU_LIMIT_PRINTF_FIFO_SIZE = 0x01,
    CU_LIMIT_MALLOC_HEAP_SIZE = 0x02,
    CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH = 0x03,
    CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT = 0x04,
    CU_LIMIT_MAX_L2_FETCH_GRANULARITY = 0x05,
    CU_LIMIT_PERSISTING_L2_CACHE_SIZE = 0x06
} CUlimit;

typedef enum CUmemorytype_enum {
    CU_MEMORYTYPE_HOST = 0x01,
    CU_MEMORYTYPE_DEVICE = 0x02,
    CU_MEMORYTYPE_ARRAY = 0x03,
    CU_MEMORYTYPE_UNIFIED = 0x04
} CUmemorytype;

typedef enum CUpointer_attribute_enum {
    CU_POINTER_ATTRIBUTE_CONTEXT = 1,
    CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 2,
    CU_POINTER_ATTRIBUTE_DEVICE_POINTER = 3,
    CU_POINTER_ATTRIBUTE_HOST_POINTER = 4,
    CU_POINTER_ATTRIBUTE_P2P_TOKENS = 5,
    CU_POINTER_ATTRIBUTE_SYNC_MEMOPS = 6,
    CU_POINTER_ATTRIBUTE_BUFFER_ID = 7,
    CU_POINTER_ATTRIBUTE_IS_MANAGED = 8,
    CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL = 9,
    CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE = 10,
    CU_POINTER_ATTRIBUTE_RANGE_START_ADDR = 11,
    CU_POINTER_ATTRIBUTE_RANGE_SIZE = 12,
    CU_POINTER_ATTRIBUTE_MAPPED = 13,
    CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES = 14,
    CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE = 15,
    CU_POINTER_ATTRIBUTE_ACCESS_FLAGS = 16,
    CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE = 17,
    CU_POINTER_ATTRIBUTE_MAPPING_SIZE = 18,
    CU_POINTER_ATTRIBUTE_MAPPING_BASE_ADDR = 19,
    CU_POINTER_ATTRIBUTE_MEMORY_BLOCK_ID = 20
} CUpointer_attribute;

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

/* Virtual memory management and stream-ordered allocation */

typedef enum CUmemLocationType_enum {
    CU_MEM_LOCATION_TYPE_INVALID = 0x0,
    CU_MEM_LOCATION_TYPE_DEVICE = 0x1,
    CU_MEM_LOCATION_TYPE_HOST = 0x2,
    CU_MEM_LOCATION_TYPE_HOST_NUMA = 0x3,
    CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT = 0x4
} CUmemLocationType;

typedef enum CUmemAllocationType_enum {
    CU_MEM_ALLOCATION_TYPE_INVALID = 0x0,
    CU_MEM_ALLOCATION_TYPE_PINNED = 0x1
} CUmemAllocationType;

typedef enum CUmemAllocationHandleType_enum {
    CU_MEM_HANDLE_TYPE_NONE = 0x0,
    CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 0x1,
    CU_MEM_HANDLE_TYPE_WIN32 = 0x2,
    CU_MEM_HANDLE_TYPE_WIN32_KMT = 0x4,
    CU_MEM_HANDLE_TYPE_FABRIC = 0x8
} CUmemAllocationHandleType;

typedef enum CUmemAccess_flags_enum {
    CU_MEM_ACCESS_FLAGS_PROT_NONE = 0x0,
    CU_MEM_ACCESS_FLAGS_PROT_READ = 0x1,
    CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 0x3
} CUmemAccess_flags;

typedef enum CUmemAllocationGranularity_flags_enum {
    CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0x0,
    CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = 0x1
} CUmemAllocationGranularity_flags;

typedef enum CUmemPool_attribute_enum {
    CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES = 1,
    CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC = 2,
    CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES = 3,
    CU_MEMPOOL_ATTR_RELEASE_THRESHOLD = 4,
    CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT = 5,
    CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH = 6,
    CU_MEMPOOL_ATTR_USED_MEM_CURRENT = 7,
    CU_MEMPOOL_ATTR_USED_MEM_HIGH = 8
} CUmemPool_attribute;

typedef struct CUmemLocation_st {
    CUmemLocationType type;
    int id;
} CUmemLocation_v1;
typedef CUmemLocation_v1 CUmemLocation;

typedef struct CUmemAllocationProp_st {
    CUmemAllocationType type;
    CUmemAllocationHandleType requestedHandleTypes;
    CUmemLocation location;
    void* win32HandleMetaData;
    struct {
        unsigned char compressionType;
        unsigned char gpuDirectRDMACapable;
        unsigned short usage;
        unsigned char reserved[4];
    } allocFlags;
} CUmemAllocationProp_v1;
typedef CUmemAllocationProp_v1 CUmemAllocationProp;

typedef struct CUmemAccessDesc_st {
    CUmemLocation location;
    CUmemAccess_flags flags;
} CUmemAccessDesc_v1;
typedef CUmemAccessDesc_v1 CUmemAccessDesc;

typedef struct CUmemPoolProps_st {
    CUmemAllocationType allocType;
    CUmemAllocationHandleType handleTypes;
    CUmemLocation location;
    void* win32SecurityAttributes;
    size_t maxSize;
    unsigned short usage;
    unsigned char reserved[54];
} CUmemPoolProps_v1;
typedef CUmemPoolProps_v1 CUmemPoolProps;

#define CU_MEMHOSTREGISTER_PORTABLE 0x01
#define CU_MEMHOSTREGISTER_DEVICEMAP 0x02
#define CU_MEMHOSTREGISTER_IOMEMORY 0x04
#define CU_MEMHOSTREGISTER_READ_ONLY 0x08
#define CU_MEMHOSTALLOC_PORTABLE 0x01
#define CU_MEMHOSTALLOC_DEVICEMAP 0x02
#define CU_MEMHOSTALLOC_WRITECOMBINED 0x04
typedef enum CUmemAttach_flags_enum {
    CU_MEM_ATTACH_GLOBAL = 0x1,
    CU_MEM_ATTACH_HOST = 0x2,
    CU_MEM_ATTACH_SINGLE = 0x4
} CUmemAttach_flags;
typedef enum CUmem_advise_enum {
    CU_MEM_ADVISE_SET_READ_MOSTLY = 1,
    CU_MEM_ADVISE_UNSET_READ_MOSTLY = 2,
    CU_MEM_ADVISE_SET_PREFERRED_LOCATION = 3,
    CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION = 4,
    CU_MEM_ADVISE_SET_ACCESSED_BY = 5,
    CU_MEM_ADVISE_UNSET_ACCESSED_BY = 6
} CUmem_advise;

/* Extended launches (cuLaunchKernelEx) */

typedef enum CUlaunchAttributeID_enum {
    CU_LAUNCH_ATTRIBUTE_IGNORE = 0,
    CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW = 1,
    CU_LAUNCH_ATTRIBUTE_COOPERATIVE = 2,
    CU_LAUNCH_ATTRIBUTE_SYNCHRONIZATION_POLICY = 3,
    CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION = 4,
    CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE = 5,
    CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION = 6,
    CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT = 7,
    CU_LAUNCH_ATTRIBUTE_PRIORITY = 8,
    CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP = 9,
    CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN = 10,
    CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT = 12,
    CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE = 13,
    CU_LAUNCH_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 14
} CUlaunchAttributeID;

typedef enum CUclusterSchedulingPolicy_enum {
    CU_CLUSTER_SCHEDULING_POLICY_DEFAULT = 0,
    CU_CLUSTER_SCHEDULING_POLICY_SPREAD = 1,
    CU_CLUSTER_SCHEDULING_POLICY_LOAD_BALANCING = 2
} CUclusterSchedulingPolicy;

typedef enum CUaccessProperty_enum {
    CU_ACCESS_PROPERTY_NORMAL = 0,
    CU_ACCESS_PROPERTY_STREAMING = 1,
    CU_ACCESS_PROPERTY_PERSISTING = 2
} CUaccessProperty;

typedef struct CUaccessPolicyWindow_st {
    void* base_ptr;
    size_t num_bytes;
    float hitRatio;
    CUaccessProperty hitProp;
    CUaccessProperty missProp;
} CUaccessPolicyWindow_v1;
typedef CUaccessPolicyWindow_v1 CUaccessPolicyWindow;

typedef union CUlaunchAttributeValue_union {
    char pad[64];
    CUaccessPolicyWindow accessPolicyWindow;
    int cooperative;
    int syncPolicy;
    struct {
        unsigned int x;
        unsigned int y;
        unsigned int z;
    } clusterDim;
    CUclusterSchedulingPolicy clusterSchedulingPolicyPreference;
    int programmaticStreamSerializationAllowed;
    struct {
        CUevent event;
        int flags;
        int triggerAtBlockStart;
    } programmaticEvent;
    int priority;
    unsigned int sharedMemCarveout;
} CUlaunchAttributeValue;

typedef struct CUlaunchAttribute_st {
    CUlaunchAttributeID id;
    char pad[8 - sizeof(CUlaunchAttributeID)];
    CUlaunchAttributeValue value;
} CUlaunchAttribute;

// cuCtxCreate_v3/_v4 parameters (execution affinity and CUDA-in-graphics are not supported).
typedef enum CUexecAffinityType_enum {
    CU_EXEC_AFFINITY_TYPE_SM_COUNT = 0,
    CU_EXEC_AFFINITY_TYPE_MAX
} CUexecAffinityType;
typedef struct CUexecAffinitySmCount_st {
    unsigned int val;
} CUexecAffinitySmCount;
typedef struct CUexecAffinityParam_st {
    CUexecAffinityType type;
    union {
        CUexecAffinitySmCount smCount;
    } param;
} CUexecAffinityParam;
typedef enum CUcigDataType_enum {
    CIG_DATA_TYPE_D3D12_COMMAND_QUEUE = 0x1,
    CIG_DATA_TYPE_NV_BLOB = 0x2,
} CUcigDataType;
typedef struct CUctxCigParam_st {
    CUcigDataType sharedDataType;
    void* sharedData;
} CUctxCigParam;
typedef struct CUctxCreateParams_st {
    CUexecAffinityParam* execAffinityParams;
    int numExecAffinityParams;
    CUctxCigParam* cigParams;
} CUctxCreateParams;

// ---- CUDA graphs and stream capture -----------------------------------------------------------
typedef void (*CUhostFn)(void* userData);
typedef struct CUkern_st* CUkernel;
typedef struct CUgraph_st* CUgraph;
typedef struct CUgraphNode_st* CUgraphNode;
typedef struct CUgraphExec_st* CUgraphExec;
typedef enum CUgraphNodeType_enum {
    CU_GRAPH_NODE_TYPE_KERNEL = 0,
    CU_GRAPH_NODE_TYPE_MEMCPY = 1,
    CU_GRAPH_NODE_TYPE_MEMSET = 2,
    CU_GRAPH_NODE_TYPE_HOST = 3,
    CU_GRAPH_NODE_TYPE_GRAPH = 4,
    CU_GRAPH_NODE_TYPE_EMPTY = 5,
    CU_GRAPH_NODE_TYPE_WAIT_EVENT = 6,
    CU_GRAPH_NODE_TYPE_EVENT_RECORD = 7,
    CU_GRAPH_NODE_TYPE_MEM_ALLOC = 10,
    CU_GRAPH_NODE_TYPE_MEM_FREE = 11,
} CUgraphNodeType;
typedef enum CUstreamCaptureStatus_enum {
    CU_STREAM_CAPTURE_STATUS_NONE = 0,
    CU_STREAM_CAPTURE_STATUS_ACTIVE = 1,
    CU_STREAM_CAPTURE_STATUS_INVALIDATED = 2
} CUstreamCaptureStatus;
typedef enum CUstreamCaptureMode_enum {
    CU_STREAM_CAPTURE_MODE_GLOBAL = 0,
    CU_STREAM_CAPTURE_MODE_THREAD_LOCAL = 1,
    CU_STREAM_CAPTURE_MODE_RELAXED = 2
} CUstreamCaptureMode;
typedef struct CUDA_KERNEL_NODE_PARAMS_st {  // _v1
    CUfunction func;
    unsigned int gridDimX, gridDimY, gridDimZ;
    unsigned int blockDimX, blockDimY, blockDimZ;
    unsigned int sharedMemBytes;
    void** kernelParams;
    void** extra;
} CUDA_KERNEL_NODE_PARAMS_v1;
typedef struct CUDA_KERNEL_NODE_PARAMS_v2_st {
    CUfunction func;
    unsigned int gridDimX, gridDimY, gridDimZ;
    unsigned int blockDimX, blockDimY, blockDimZ;
    unsigned int sharedMemBytes;
    void** kernelParams;
    void** extra;
    CUkernel kern;  // not supported: must be NULL
    CUcontext ctx;
} CUDA_KERNEL_NODE_PARAMS_v2;
typedef CUDA_KERNEL_NODE_PARAMS_v2 CUDA_KERNEL_NODE_PARAMS;
typedef struct CUDA_MEMSET_NODE_PARAMS_st {
    CUdeviceptr dst;
    size_t pitch;
    unsigned int value;
    unsigned int elementSize;
    size_t width;
    size_t height;
} CUDA_MEMSET_NODE_PARAMS;
typedef struct CUDA_HOST_NODE_PARAMS_st {
    CUhostFn fn;
    void* userData;
} CUDA_HOST_NODE_PARAMS;
typedef enum CUgraphExecUpdateResult_enum {
    CU_GRAPH_EXEC_UPDATE_SUCCESS = 0x0,
    CU_GRAPH_EXEC_UPDATE_ERROR = 0x1,
    CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED = 0x2,
    CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED = 0x3,
    CU_GRAPH_EXEC_UPDATE_ERROR_FUNCTION_CHANGED = 0x4,
    CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED = 0x5,
    CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED = 0x6,
    CU_GRAPH_EXEC_UPDATE_ERROR_UNSUPPORTED_FUNCTION_CHANGE = 0x7,
    CU_GRAPH_EXEC_UPDATE_ERROR_ATTRIBUTES_CHANGED = 0x8
} CUgraphExecUpdateResult;
typedef struct CUgraphExecUpdateResultInfo_st {
    CUgraphExecUpdateResult result;
    CUgraphNode errorNode;
    CUgraphNode errorFromNode;
} CUgraphExecUpdateResultInfo;
typedef struct CUgraphEdgeData_st {
    unsigned char from_port;
    unsigned char to_port;
    unsigned char type;
    unsigned char reserved[5];
} CUgraphEdgeData;

typedef struct CUlaunchConfig_st {
    unsigned int gridDimX;
    unsigned int gridDimY;
    unsigned int gridDimZ;
    unsigned int blockDimX;
    unsigned int blockDimY;
    unsigned int blockDimZ;
    unsigned int sharedMemBytes;
    CUstream hStream;
    CUlaunchAttribute* attrs;
    unsigned int numAttrs;
} CUlaunchConfig;

/* Tensor maps (TMA, sm_90+): declared for source compatibility, not supported */

typedef struct CUtensorMap_st {
#if defined(__cplusplus)
    alignas(64)
#else
    _Alignas(64)
#endif
        cuuint64_t opaque[16];
} CUtensorMap;

typedef enum CUtensorMapDataType_enum {
    CU_TENSOR_MAP_DATA_TYPE_UINT8 = 0,
    CU_TENSOR_MAP_DATA_TYPE_UINT16,
    CU_TENSOR_MAP_DATA_TYPE_UINT32,
    CU_TENSOR_MAP_DATA_TYPE_INT32,
    CU_TENSOR_MAP_DATA_TYPE_UINT64,
    CU_TENSOR_MAP_DATA_TYPE_INT64,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT64,
    CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ,
    CU_TENSOR_MAP_DATA_TYPE_TFLOAT32,
    CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ,
    CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B,
    CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,
    CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B
} CUtensorMapDataType;

typedef enum CUtensorMapInterleave_enum {
    CU_TENSOR_MAP_INTERLEAVE_NONE = 0,
    CU_TENSOR_MAP_INTERLEAVE_16B,
    CU_TENSOR_MAP_INTERLEAVE_32B
} CUtensorMapInterleave;

typedef enum CUtensorMapSwizzle_enum {
    CU_TENSOR_MAP_SWIZZLE_NONE = 0,
    CU_TENSOR_MAP_SWIZZLE_32B,
    CU_TENSOR_MAP_SWIZZLE_64B,
    CU_TENSOR_MAP_SWIZZLE_128B,
    CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B,
    CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B_FLIP_8B,
    CU_TENSOR_MAP_SWIZZLE_128B_ATOM_64B
} CUtensorMapSwizzle;

typedef enum CUtensorMapL2promotion_enum {
    CU_TENSOR_MAP_L2_PROMOTION_NONE = 0,
    CU_TENSOR_MAP_L2_PROMOTION_L2_64B,
    CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
    CU_TENSOR_MAP_L2_PROMOTION_L2_256B
} CUtensorMapL2promotion;

typedef enum CUtensorMapFloatOOBfill_enum {
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE = 0,
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA
} CUtensorMapFloatOOBfill;

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
#define cuMemHostRegister cuMemHostRegister_v2
#define cuMemHostGetDevicePointer cuMemHostGetDevicePointer_v2
#define cuStreamDestroy cuStreamDestroy_v2
#define cuEventDestroy cuEventDestroy_v2
#define cuStreamBeginCapture cuStreamBeginCapture_v2
#define cuStreamGetCaptureInfo cuStreamGetCaptureInfo_v2
#define cuGraphInstantiate cuGraphInstantiateWithFlags
#define cuGraphExecUpdate cuGraphExecUpdate_v2
#define cuGraphAddKernelNode cuGraphAddKernelNode_v2
#define cuGraphKernelNodeGetParams cuGraphKernelNodeGetParams_v2
#define cuGraphKernelNodeSetParams cuGraphKernelNodeSetParams_v2
#define cuGraphExecKernelNodeSetParams cuGraphExecKernelNodeSetParams_v2

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
CUresult CUDAAPI cuCtxCreate_v3(CUcontext* pctx, CUexecAffinityParam* paramsArray, int numParams, unsigned int flags,
                                CUdevice dev);
CUresult CUDAAPI cuCtxCreate_v4(CUcontext* pctx, CUctxCreateParams* ctxCreateParams, unsigned int flags, CUdevice dev);
CUresult CUDAAPI cuCtxGetDevice_v2(CUdevice* device, CUcontext ctx);  // CUDA 13: of ctx, or the current one if NULL
CUresult CUDAAPI cuCtxSynchronize_v2(CUcontext ctx);
CUresult CUDAAPI cuCtxDestroy(CUcontext ctx);
CUresult CUDAAPI cuCtxPushCurrent(CUcontext ctx);
CUresult CUDAAPI cuCtxPopCurrent(CUcontext* pctx);
CUresult CUDAAPI cuCtxSetCurrent(CUcontext ctx);
CUresult CUDAAPI cuCtxGetCurrent(CUcontext* pctx);
CUresult CUDAAPI cuCtxGetDevice(CUdevice* device);
CUresult CUDAAPI cuCtxGetFlags(unsigned int* flags);
CUresult CUDAAPI cuCtxSynchronize(void);
CUresult CUDAAPI cuCtxGetApiVersion(CUcontext ctx, unsigned int* version);
CUresult CUDAAPI cuCtxSetLimit(CUlimit limit, size_t value);
CUresult CUDAAPI cuCtxGetLimit(size_t* pvalue, CUlimit limit);

/* ------------------------------------------------------------------------ */
/* Memory management                                                        */
/* ------------------------------------------------------------------------ */

CUresult CUDAAPI cuMemGetInfo(size_t* free, size_t* total);
CUresult CUDAAPI cuMemAlloc(CUdeviceptr* dptr, size_t bytesize);
CUresult CUDAAPI cuMemFree(CUdeviceptr dptr);
CUresult CUDAAPI cuMemGetAddressRange(CUdeviceptr* pbase, size_t* psize, CUdeviceptr dptr);
CUresult CUDAAPI cuMemAllocHost(void** pp, size_t bytesize);
CUresult CUDAAPI cuMemFreeHost(void* p);
/* Pinned, registered and managed memory are host pages the device accesses at the same
   address (VK_EXT_external_memory_host); advice and prefetches are accepted and ignored. */
CUresult CUDAAPI cuMemHostAlloc(void** pp, size_t bytesize, unsigned int Flags);
CUresult CUDAAPI cuMemHostGetFlags(unsigned int* pFlags, void* p);
CUresult CUDAAPI cuMemAllocManaged(CUdeviceptr* dptr, size_t bytesize, unsigned int flags);
CUresult CUDAAPI cuMemAdvise(CUdeviceptr devPtr, size_t count, CUmem_advise advice, CUdevice device);
CUresult CUDAAPI cuMemAdvise_v2(CUdeviceptr devPtr, size_t count, CUmem_advise advice, CUmemLocation location);
CUresult CUDAAPI cuMemPrefetchAsync(CUdeviceptr devPtr, size_t count, CUdevice dstDevice, CUstream hStream);
CUresult CUDAAPI cuMemPrefetchAsync_v2(CUdeviceptr devPtr, size_t count, CUmemLocation location, unsigned int flags,
                                       CUstream hStream);
CUresult CUDAAPI cuStreamAttachMemAsync(CUstream hStream, CUdeviceptr dptr, size_t length, unsigned int flags);

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

CUresult CUDAAPI cuPointerGetAttribute(void* data, CUpointer_attribute attribute, CUdeviceptr ptr);
CUresult CUDAAPI cuPointerGetAttributes(unsigned int numAttributes, CUpointer_attribute* attributes, void** data,
                                        CUdeviceptr ptr);

CUresult CUDAAPI cuMemHostRegister(void* p, size_t bytesize, unsigned int Flags);
CUresult CUDAAPI cuMemHostUnregister(void* p);
CUresult CUDAAPI cuMemHostGetDevicePointer(CUdeviceptr* pdptr, void* p, unsigned int Flags);

/* Stream-ordered allocation and memory pools */
CUresult CUDAAPI cuMemAllocAsync(CUdeviceptr* dptr, size_t bytesize, CUstream hStream);
CUresult CUDAAPI cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream);
CUresult CUDAAPI cuMemAllocFromPoolAsync(CUdeviceptr* dptr, size_t bytesize, CUmemoryPool pool, CUstream hStream);
CUresult CUDAAPI cuMemPoolCreate(CUmemoryPool* pool, const CUmemPoolProps* poolProps);
CUresult CUDAAPI cuMemPoolDestroy(CUmemoryPool pool);
CUresult CUDAAPI cuMemPoolTrimTo(CUmemoryPool pool, size_t minBytesToKeep);
CUresult CUDAAPI cuMemPoolSetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void* value);
CUresult CUDAAPI cuMemPoolGetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void* value);
CUresult CUDAAPI cuMemPoolSetAccess(CUmemoryPool pool, const CUmemAccessDesc* map, size_t count);
CUresult CUDAAPI cuMemPoolGetAccess(CUmemAccess_flags* flags, CUmemoryPool memPool, CUmemLocation* location);
CUresult CUDAAPI cuDeviceGetDefaultMemPool(CUmemoryPool* pool_out, CUdevice dev);
CUresult CUDAAPI cuDeviceGetMemPool(CUmemoryPool* pool, CUdevice dev);
CUresult CUDAAPI cuDeviceSetMemPool(CUdevice dev, CUmemoryPool pool);

/* Virtual memory management */
CUresult CUDAAPI cuMemAddressReserve(CUdeviceptr* ptr, size_t size, size_t alignment, CUdeviceptr addr,
                                     unsigned long long flags);
CUresult CUDAAPI cuMemAddressFree(CUdeviceptr ptr, size_t size);
CUresult CUDAAPI cuMemCreate(CUmemGenericAllocationHandle* handle, size_t size, const CUmemAllocationProp* prop,
                             unsigned long long flags);
CUresult CUDAAPI cuMemRelease(CUmemGenericAllocationHandle handle);
CUresult CUDAAPI cuMemMap(CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle,
                          unsigned long long flags);
CUresult CUDAAPI cuMemUnmap(CUdeviceptr ptr, size_t size);
CUresult CUDAAPI cuMemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc* desc, size_t count);
CUresult CUDAAPI cuMemGetAccess(unsigned long long* flags, const CUmemLocation* location, CUdeviceptr ptr);
CUresult CUDAAPI cuMemGetAllocationGranularity(size_t* granularity, const CUmemAllocationProp* prop,
                                               CUmemAllocationGranularity_flags option);
CUresult CUDAAPI cuMemGetAllocationPropertiesFromHandle(CUmemAllocationProp* prop, CUmemGenericAllocationHandle handle);
CUresult CUDAAPI cuMemRetainAllocationHandle(CUmemGenericAllocationHandle* handle, void* addr);

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
CUresult CUDAAPI cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int value);
CUresult CUDAAPI cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config);

CUresult CUDAAPI cuOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, CUfunction func, int blockSize,
                                                             size_t dynamicSMemSize);
CUresult CUDAAPI cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, CUfunction func, int blockSize,
                                                                      size_t dynamicSMemSize, unsigned int flags);
CUresult CUDAAPI cuOccupancyMaxActiveClusters(int* numClusters, CUfunction func, const CUlaunchConfig* config);

CUresult CUDAAPI cuTensorMapEncodeTiled(CUtensorMap* tensorMap, CUtensorMapDataType tensorDataType,
                                        cuuint32_t tensorRank, void* globalAddress, const cuuint64_t* globalDim,
                                        const cuuint64_t* globalStrides, const cuuint32_t* boxDim,
                                        const cuuint32_t* elementStrides, CUtensorMapInterleave interleave,
                                        CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion,
                                        CUtensorMapFloatOOBfill oobFill);
CUresult CUDAAPI cuTensorMapEncodeIm2col(CUtensorMap* tensorMap, CUtensorMapDataType tensorDataType,
                                         cuuint32_t tensorRank, void* globalAddress, const cuuint64_t* globalDim,
                                         const cuuint64_t* globalStrides, const int* pixelBoxLowerCorner,
                                         const int* pixelBoxUpperCorner, cuuint32_t channelsPerPixel,
                                         cuuint32_t pixelsPerColumn, const cuuint32_t* elementStrides,
                                         CUtensorMapInterleave interleave, CUtensorMapSwizzle swizzle,
                                         CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill);

CUresult CUDAAPI cuLaunchKernel(CUfunction f,
                                unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                                unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                                unsigned int sharedMemBytes, CUstream hStream,
                                void** kernelParams, void** extra);
CUresult CUDAAPI cuLaunchKernelEx(const CUlaunchConfig* config, CUfunction f, void** kernelParams, void** extra);

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

/* Stream capture and graphs. Captured work is recorded in order and replayed in that order on
   the launch stream (a valid topological order: capture follows the streams' order). */
CUresult CUDAAPI cuLaunchHostFunc(CUstream hStream, CUhostFn fn, void* userData);
CUresult CUDAAPI cuStreamBeginCapture(CUstream hStream, CUstreamCaptureMode mode);
CUresult CUDAAPI cuStreamBeginCapture_v2_ptsz(CUstream hStream, CUstreamCaptureMode mode);
CUresult CUDAAPI cuStreamEndCapture(CUstream hStream, CUgraph* phGraph);
CUresult CUDAAPI cuStreamIsCapturing(CUstream hStream, CUstreamCaptureStatus* captureStatus);
CUresult CUDAAPI cuStreamGetCaptureInfo(CUstream hStream, CUstreamCaptureStatus* captureStatus_out,
                                        cuuint64_t* id_out, CUgraph* graph_out,
                                        const CUgraphNode** dependencies_out, size_t* numDependencies_out);
CUresult CUDAAPI cuStreamGetCaptureInfo_v3(CUstream hStream, CUstreamCaptureStatus* captureStatus_out,
                                           cuuint64_t* id_out, CUgraph* graph_out,
                                           const CUgraphNode** dependencies_out,
                                           const CUgraphEdgeData** edgeData_out, size_t* numDependencies_out);
CUresult CUDAAPI cuThreadExchangeStreamCaptureMode(CUstreamCaptureMode* mode);
CUresult CUDAAPI cuStreamUpdateCaptureDependencies_v2(CUstream hStream, CUgraphNode* dependencies,
                                                      const CUgraphEdgeData* dependencyData, size_t numDependencies,
                                                      unsigned int flags);
CUresult CUDAAPI cuGraphCreate(CUgraph* phGraph, unsigned int flags);
CUresult CUDAAPI cuGraphDestroy(CUgraph hGraph);
CUresult CUDAAPI cuGraphAddKernelNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies,
                                      size_t numDependencies, const CUDA_KERNEL_NODE_PARAMS* nodeParams);
CUresult CUDAAPI cuGraphKernelNodeGetParams(CUgraphNode hNode, CUDA_KERNEL_NODE_PARAMS* nodeParams);
CUresult CUDAAPI cuGraphKernelNodeSetParams(CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS* nodeParams);
CUresult CUDAAPI cuGraphAddMemsetNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies,
                                      size_t numDependencies, const CUDA_MEMSET_NODE_PARAMS* memsetParams,
                                      CUcontext ctx);
CUresult CUDAAPI cuGraphAddHostNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies,
                                    size_t numDependencies, const CUDA_HOST_NODE_PARAMS* nodeParams);
CUresult CUDAAPI cuGraphAddEmptyNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies,
                                     size_t numDependencies);
CUresult CUDAAPI cuGraphGetNodes(CUgraph hGraph, CUgraphNode* nodes, size_t* numNodes);
CUresult CUDAAPI cuGraphNodeGetType(CUgraphNode hNode, CUgraphNodeType* type);
CUresult CUDAAPI cuGraphNodeGetDependencies_v2(CUgraphNode hNode, CUgraphNode* dependencies, CUgraphEdgeData* edgeData,
                                               size_t* numDependencies);
CUresult CUDAAPI cuGraphInstantiate(CUgraphExec* phGraphExec, CUgraph hGraph, unsigned long long flags);
CUresult CUDAAPI cuGraphExecKernelNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode,
                                                const CUDA_KERNEL_NODE_PARAMS* nodeParams);
CUresult CUDAAPI cuGraphExecUpdate(CUgraphExec hGraphExec, CUgraph hGraph, CUgraphExecUpdateResultInfo* resultInfo);
CUresult CUDAAPI cuGraphUpload(CUgraphExec hGraphExec, CUstream hStream);
CUresult CUDAAPI cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream);
CUresult CUDAAPI cuGraphExecDestroy(CUgraphExec hGraphExec);

#ifdef __cplusplus
}
#endif

#endif /* VULKANIZER_CUDA_H */
