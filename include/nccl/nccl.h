#pragma once

// Clean-room declarations of the NCCL API subset implemented by Vulkanizer's libnccl.so.2, written
// from the public NCCL documentation. Enum values follow the documented ABI.

#include <stddef.h>
#include <stdint.h>

#if __has_include(<cuda_runtime_api.h>)
#include <cuda_runtime_api.h>
#else
typedef struct CUstream_st* cudaStream_t;
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define NCCL_MAJOR 2
#define NCCL_MINOR 32
#define NCCL_PATCH 3
#define NCCL_VERSION(X, Y, Z) ((X) * 10000 + (Y) * 100 + (Z))
#define NCCL_VERSION_CODE NCCL_VERSION(NCCL_MAJOR, NCCL_MINOR, NCCL_PATCH)
#define NCCL_SPLIT_NOCOLOR -1
#define NCCL_UNIQUE_ID_BYTES 128

typedef struct ncclComm* ncclComm_t;
typedef struct {
    char internal[NCCL_UNIQUE_ID_BYTES];
} ncclUniqueId;

typedef enum {
    ncclSuccess = 0,
    ncclUnhandledCudaError = 1,
    ncclSystemError = 2,
    ncclInternalError = 3,
    ncclInvalidArgument = 4,
    ncclInvalidUsage = 5,
    ncclRemoteError = 6,
    ncclInProgress = 7,
    ncclNumResults = 8
} ncclResult_t;

typedef enum { ncclSum = 0, ncclProd = 1, ncclMax = 2, ncclMin = 3, ncclAvg = 4, ncclNumOps = 5, ncclMaxRedOp = 0x7fffffff } ncclRedOp_t;

typedef enum {
    ncclInt8 = 0,
    ncclChar = 0,
    ncclUint8 = 1,
    ncclInt32 = 2,
    ncclInt = 2,
    ncclUint32 = 3,
    ncclInt64 = 4,
    ncclUint64 = 5,
    ncclFloat16 = 6,
    ncclHalf = 6,
    ncclFloat32 = 7,
    ncclFloat = 7,
    ncclFloat64 = 8,
    ncclDouble = 8,
    ncclBfloat16 = 9,
    ncclFloat8e4m3 = 10,
    ncclFloat8e5m2 = 11,
    ncclNumTypes = 12
} ncclDataType_t;

typedef enum { ncclScalarDevice = 0, ncclScalarHostImmediate = 1 } ncclScalarResidence_t;

// Communicator options: accepted and ignored (the layout is the caller's concern).
typedef struct ncclConfig_v22800 ncclConfig_t;

ncclResult_t ncclGetVersion(int* version);
ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId);
ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank);
ncclResult_t ncclCommInitRankConfig(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank, ncclConfig_t* config);
ncclResult_t ncclCommInitRankScalable(ncclComm_t* newcomm, int nranks, int myrank, int nId, ncclUniqueId* commIds,
                                      ncclConfig_t* config);
ncclResult_t ncclCommInitAll(ncclComm_t* comms, int ndev, const int* devlist);
ncclResult_t ncclCommFinalize(ncclComm_t comm);
ncclResult_t ncclCommDestroy(ncclComm_t comm);
ncclResult_t ncclCommAbort(ncclComm_t comm);
ncclResult_t ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t* newcomm, ncclConfig_t* config);
const char* ncclGetErrorString(ncclResult_t result);
const char* ncclGetLastError(ncclComm_t comm);
ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t* asyncError);
ncclResult_t ncclCommCount(const ncclComm_t comm, int* count);
ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int* device);
ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank);
ncclResult_t ncclCommRegister(const ncclComm_t comm, void* buff, size_t size, void** handle);
ncclResult_t ncclCommDeregister(const ncclComm_t comm, void* handle);
ncclResult_t ncclMemAlloc(void** ptr, size_t size);
ncclResult_t ncclMemFree(void* ptr);
ncclResult_t ncclRedOpCreatePreMulSum(ncclRedOp_t* op, void* scalar, ncclDataType_t datatype,
                                      ncclScalarResidence_t residence, ncclComm_t comm);
ncclResult_t ncclRedOpDestroy(ncclRedOp_t op, ncclComm_t comm);

ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op,
                        int root, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm,
                       cudaStream_t stream);
ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
                           ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                           ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount, ncclDataType_t datatype,
                               ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype,
                           ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclAlltoAll(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                          ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclGather(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
                        ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm,
                      cudaStream_t stream);
ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm,
                      cudaStream_t stream);
ncclResult_t ncclGroupStart(void);
ncclResult_t ncclGroupEnd(void);

#ifdef __cplusplus
}
#endif
