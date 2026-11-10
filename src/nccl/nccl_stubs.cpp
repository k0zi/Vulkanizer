// NCCL entry points PyTorch links against that Vulkanizer does not implement: symmetric memory
// windows, device communicators, one-sided signals, elastic communicators (shrink, grow, suspend).
// Each one reports ncclInvalidUsage, with a warning the first time. They ignore their arguments,
// which the C calling convention allows, so one parameterless definition serves every signature.

#include <nccl/nccl.h>

#include "common/log.h"

#define VULKANIZER_NCCL_UNSUPPORTED(X) \
    X(ncclCommGetUniqueId) \
    X(ncclCommGrow) \
    X(ncclCommMemStats) \
    X(ncclCommResume) \
    X(ncclCommRevoke) \
    X(ncclCommShrink) \
    X(ncclCommSuspend) \
    X(ncclCommWindowDeregister) \
    X(ncclCommWindowRegister) \
    X(ncclDevCommCreate) \
    X(ncclDevCommDestroy) \
    X(ncclGetLsaMultimemDevicePointer) \
    X(ncclGetPeerDevicePointer) \
    X(ncclPutSignal) \
    X(ncclSignal) \
    X(ncclWaitSignal) \
    X(ncclWinGetUserPtr) \


#define VULKANIZER_DEFINE_STUB(name)                                      \
    extern "C" ncclResult_t name() {                                     \
        VULKANIZER_WARN_ONCE("nccl: %s is not implemented", #name);      \
        return ncclInvalidUsage;                                          \
    }

VULKANIZER_NCCL_UNSUPPORTED(VULKANIZER_DEFINE_STUB)
