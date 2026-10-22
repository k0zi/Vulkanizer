// Driver API values and layouts that must match NVIDIA's cuda.h (see main.cpp).
#pragma once

#include <cstddef>
#include <vector>

#define VULKANIZER_CU_VALUES(X)                                                                   \
    X(CUDA_ERROR_ALREADY_MAPPED) X(CUDA_ERROR_NOT_MAPPED) X(CUDA_ERROR_UNSUPPORTED_LIMIT)         \
    X(CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY) X(CU_EXEC_AFFINITY_TYPE_SM_COUNT) X(CIG_DATA_TYPE_NV_BLOB) \
    X(CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED) X(CUDA_ERROR_STREAM_CAPTURE_IMPLICIT) X(CUDA_ERROR_CAPTURED_EVENT) \
    X(CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE) X(CU_GRAPH_NODE_TYPE_MEM_FREE) X(CU_GRAPH_NODE_TYPE_EVENT_RECORD) \
    X(CU_STREAM_CAPTURE_MODE_RELAXED) X(CU_STREAM_CAPTURE_STATUS_INVALIDATED)                   \
    X(CU_GRAPH_EXEC_UPDATE_ERROR_ATTRIBUTES_CHANGED)                                            \
    X(CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED) X(CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED)         \
    X(CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM)                                \
    X(CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED)                                                \
    X(CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED)                                    \
    X(CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED)                            \
    X(CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED)                                          \
    X(CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED)                                      \
    X(CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED) X(CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES) \
    X(CU_MEM_LOCATION_TYPE_DEVICE) X(CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT)                      \
    X(CU_MEM_ALLOCATION_TYPE_PINNED) X(CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR)                  \
    X(CU_MEM_HANDLE_TYPE_FABRIC) X(CU_MEM_ACCESS_FLAGS_PROT_READ) X(CU_MEM_ACCESS_FLAGS_PROT_READWRITE) \
    X(CU_MEM_ALLOC_GRANULARITY_RECOMMENDED) X(CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES)     \
    X(CU_MEMPOOL_ATTR_RELEASE_THRESHOLD) X(CU_MEMPOOL_ATTR_USED_MEM_HIGH)                         \
    X(CU_MEMHOSTREGISTER_DEVICEMAP) X(CU_MEMHOSTREGISTER_READ_ONLY)                               \
    X(CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED)                                        \
    X(CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE) X(CU_FUNC_CACHE_PREFER_SHARED)      \
    X(CU_LIMIT_PRINTF_FIFO_SIZE) X(CU_LIMIT_PERSISTING_L2_CACHE_SIZE) X(CU_MEMORYTYPE_DEVICE)     \
    X(CU_POINTER_ATTRIBUTE_DEVICE_POINTER) X(CU_POINTER_ATTRIBUTE_RANGE_SIZE)                     \
    X(CU_POINTER_ATTRIBUTE_ACCESS_FLAGS) X(CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE)                   \
    X(CU_POINTER_ATTRIBUTE_MEMORY_BLOCK_ID) X(CU_LAUNCH_ATTRIBUTE_COOPERATIVE)                    \
    X(CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION) X(CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE) \
    X(CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION) X(CU_LAUNCH_ATTRIBUTE_PRIORITY)      \
    X(CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT) X(CU_LAUNCH_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT) \
    X(CU_CLUSTER_SCHEDULING_POLICY_SPREAD) X(CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ)                \
    X(CU_TENSOR_MAP_SWIZZLE_128B) X(CU_TENSOR_MAP_L2_PROMOTION_L2_256B)                           \
    X(CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA) X(CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)

#define VULKANIZER_CU_LAYOUTS(X)                                                                  \
    X(sizeof(CUmemLocation)) X(sizeof(CUmemAllocationProp))                                       \
    X(offsetof(CUmemAllocationProp, location)) X(offsetof(CUmemAllocationProp, win32HandleMetaData)) \
    X(offsetof(CUmemAllocationProp, allocFlags)) X(sizeof(CUmemAccessDesc))                       \
    X(offsetof(CUmemAccessDesc, flags)) X(sizeof(CUmemPoolProps)) X(offsetof(CUmemPoolProps, location)) \
    X(offsetof(CUmemPoolProps, win32SecurityAttributes)) X(offsetof(CUmemPoolProps, maxSize))     \
    X(offsetof(CUmemPoolProps, usage)) X(sizeof(CUaccessPolicyWindow))                            \
    X(sizeof(CUlaunchAttributeValue)) X(sizeof(CUlaunchAttribute)) X(offsetof(CUlaunchAttribute, value)) \
    X(sizeof(CUlaunchConfig)) X(offsetof(CUlaunchConfig, hStream)) X(offsetof(CUlaunchConfig, attrs)) \
    X(offsetof(CUlaunchConfig, numAttrs)) X(sizeof(CUtensorMap)) X(alignof(CUtensorMap))             \
    X(sizeof(CUexecAffinityParam)) X(sizeof(CUctxCigParam)) X(sizeof(CUctxCreateParams))         \
    X(offsetof(CUctxCreateParams, numExecAffinityParams)) X(offsetof(CUctxCreateParams, cigParams)) \
    X(sizeof(CUDA_KERNEL_NODE_PARAMS_v1)) X(sizeof(CUDA_KERNEL_NODE_PARAMS_v2))                 \
    X(offsetof(CUDA_KERNEL_NODE_PARAMS_v2, kernelParams)) X(offsetof(CUDA_KERNEL_NODE_PARAMS_v2, ctx)) \
    X(sizeof(CUDA_MEMSET_NODE_PARAMS)) X(offsetof(CUDA_MEMSET_NODE_PARAMS, value))             \
    X(offsetof(CUDA_MEMSET_NODE_PARAMS, height)) X(sizeof(CUDA_HOST_NODE_PARAMS))              \
    X(sizeof(CUgraphExecUpdateResultInfo)) X(sizeof(CUgraphEdgeData))


