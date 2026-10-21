// Primary and user contexts, per-thread context stack.

#include "driver/driver.h"

#include "common/log.h"

#include <algorithm>

using namespace vulkanizer;
using namespace vulkanizer::driver;

static CUresult requireInit() {
    return Driver::get().initialized() ? CUDA_SUCCESS : CUDA_ERROR_NOT_INITIALIZED;
}

extern "C" {

CUresult CUDAAPI cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev) {
    VULKANIZER_TRACE_API();
    if (!pctx) return CUDA_ERROR_INVALID_VALUE;
    if (CUresult r = requireInit()) return r;
    Driver& d = Driver::get();
    std::lock_guard<std::recursive_mutex> lock(d.mutex());
    DeviceSlot* s = d.slot(dev);
    if (!s) return CUDA_ERROR_INVALID_DEVICE;
    if (CUresult r = d.ensureDevice(*s)) return r;
    ++s->primaryRefs;
    *pctx = &s->primary;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDevicePrimaryCtxRelease_v2(CUdevice dev) {
    VULKANIZER_TRACE_API();
    if (CUresult r = requireInit()) return r;
    Driver& d = Driver::get();
    std::lock_guard<std::recursive_mutex> lock(d.mutex());
    DeviceSlot* s = d.slot(dev);
    if (!s) return CUDA_ERROR_INVALID_DEVICE;
    if (s->primaryRefs <= 0) return CUDA_ERROR_INVALID_CONTEXT;
    // TODO: free the primary context's resources when the last reference goes away.
    --s->primaryRefs;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDevicePrimaryCtxSetFlags_v2(CUdevice dev, unsigned int flags) {
    VULKANIZER_TRACE_API();
    if (CUresult r = requireInit()) return r;
    Driver& d = Driver::get();
    std::lock_guard<std::recursive_mutex> lock(d.mutex());
    DeviceSlot* s = d.slot(dev);
    if (!s) return CUDA_ERROR_INVALID_DEVICE;
    s->primary.flags = flags;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int* flags, int* active) {
    VULKANIZER_TRACE_API();
    if (!flags || !active) return CUDA_ERROR_INVALID_VALUE;
    if (CUresult r = requireInit()) return r;
    Driver& d = Driver::get();
    std::lock_guard<std::recursive_mutex> lock(d.mutex());
    DeviceSlot* s = d.slot(dev);
    if (!s) return CUDA_ERROR_INVALID_DEVICE;
    *flags = s->primary.flags;
    *active = s->primaryRefs > 0;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDevicePrimaryCtxReset_v2(CUdevice dev) {
    VULKANIZER_TRACE_API();
    if (CUresult r = requireInit()) return r;
    Driver& d = Driver::get();
    std::lock_guard<std::recursive_mutex> lock(d.mutex());
    DeviceSlot* s = d.slot(dev);
    if (!s) return CUDA_ERROR_INVALID_DEVICE;
    s->primaryRefs = 0;
    s->primary.flags = 0;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxCreate_v2(CUcontext* pctx, unsigned int flags, CUdevice dev) {
    VULKANIZER_TRACE_API();
    if (!pctx) return CUDA_ERROR_INVALID_VALUE;
    if (CUresult r = requireInit()) return r;
    Driver& d = Driver::get();
    std::lock_guard<std::recursive_mutex> lock(d.mutex());
    DeviceSlot* s = d.slot(dev);
    if (!s) return CUDA_ERROR_INVALID_DEVICE;
    if (CUresult r = d.ensureDevice(*s)) return r;
    auto* ctx = new CUctx_st();
    ctx->device = dev;
    ctx->flags = flags;
    if (!contextStack().push(ctx)) {
        delete ctx;
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    d.registerHandle(ctx);
    *pctx = ctx;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxDestroy_v2(CUcontext ctx) {
    VULKANIZER_TRACE_API();
    if (CUresult r = requireInit()) return r;
    if (!validHandle(ctx) || ctx->primary) return CUDA_ERROR_INVALID_CONTEXT;
    // TODO: release modules, streams and allocations owned by the context.
    Driver::get().unregisterHandle(ctx);
    contextStack().remove(ctx);
    delete ctx;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxGetCurrent(CUcontext* pctx) {
    VULKANIZER_TRACE_API();
    if (!pctx) return CUDA_ERROR_INVALID_VALUE;
    if (CUresult r = requireInit()) return r;
    *pctx = currentContext();
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxSetCurrent(CUcontext ctx) {
    VULKANIZER_TRACE_API();
    if (CUresult r = requireInit()) return r;
    auto& stack = contextStack();
    if (!ctx) {
        if (!stack.empty()) stack.pop();
        return CUDA_SUCCESS;
    }
    if (!validHandle(ctx)) return CUDA_ERROR_INVALID_CONTEXT;
    if (stack.empty()) return stack.push(ctx) ? CUDA_SUCCESS : CUDA_ERROR_OUT_OF_MEMORY;
    stack.back() = ctx;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxPushCurrent_v2(CUcontext ctx) {
    VULKANIZER_TRACE_API();
    if (CUresult r = requireInit()) return r;
    if (!validHandle(ctx)) return CUDA_ERROR_INVALID_CONTEXT;
    return contextStack().push(ctx) ? CUDA_SUCCESS : CUDA_ERROR_OUT_OF_MEMORY;
}

CUresult CUDAAPI cuCtxPopCurrent_v2(CUcontext* pctx) {
    VULKANIZER_TRACE_API();
    if (CUresult r = requireInit()) return r;
    auto& stack = contextStack();
    if (stack.empty()) return CUDA_ERROR_INVALID_CONTEXT;
    if (pctx) *pctx = stack.back();
    stack.pop();
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxGetDevice(CUdevice* device) {
    VULKANIZER_TRACE_API();
    if (!device) return CUDA_ERROR_INVALID_VALUE;
    if (CUresult r = requireInit()) return r;
    CUcontext ctx = currentContext();
    if (!validHandle(ctx)) return CUDA_ERROR_INVALID_CONTEXT;
    *device = ctx->device;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxGetFlags(unsigned int* flags) {
    VULKANIZER_TRACE_API();
    if (!flags) return CUDA_ERROR_INVALID_VALUE;
    if (CUresult r = requireInit()) return r;
    CUcontext ctx = currentContext();
    if (!validHandle(ctx)) return CUDA_ERROR_INVALID_CONTEXT;
    *flags = ctx->flags;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxSynchronize(void) {
    VULKANIZER_TRACE_API();
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    return toCuResult(s.dev->finish());
}

CUresult CUDAAPI cuCtxGetApiVersion(CUcontext ctx, unsigned int* version) {
    VULKANIZER_TRACE_API();
    if (!version) return CUDA_ERROR_INVALID_VALUE;
    if (CUresult r = requireInit()) return r;
    if (!ctx) ctx = currentContext();
    if (!validHandle(ctx)) return CUDA_ERROR_INVALID_CONTEXT;
    *version = 3020;
    return CUDA_SUCCESS;
}

// Limits are remembered and reported back; the kernels do not depend on them (no device-side
// malloc or stack, and device printf has its own buffer).
static size_t g_limits[7] = {1024, 1 << 20, 8 << 20, 2, 2048, 64, 0};

CUresult CUDAAPI cuCtxSetLimit(CUlimit limit, size_t value) {
    VULKANIZER_TRACE_API();
    if (CUresult r = requireInit()) return r;
    if (!validHandle(currentContext())) return CUDA_ERROR_INVALID_CONTEXT;
    if (unsigned(limit) >= sizeof g_limits / sizeof g_limits[0]) return CUDA_ERROR_UNSUPPORTED_LIMIT;
    g_limits[limit] = value;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxGetLimit(size_t* pvalue, CUlimit limit) {
    VULKANIZER_TRACE_API();
    if (!pvalue) return CUDA_ERROR_INVALID_VALUE;
    if (CUresult r = requireInit()) return r;
    if (!validHandle(currentContext())) return CUDA_ERROR_INVALID_CONTEXT;
    if (unsigned(limit) >= sizeof g_limits / sizeof g_limits[0]) return CUDA_ERROR_UNSUPPORTED_LIMIT;
    *pvalue = g_limits[limit];
    return CUDA_SUCCESS;
}

}  // extern "C"
