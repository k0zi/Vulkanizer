// Streams and events. All streams of a context share the device's single queue, so work is
// ordered across streams (stricter than CUDA requires, never wrong); kernels run
// asynchronously in batches (vk::Device). Events track the batch serial of the work recorded
// before them and, unless timing is disabled, a GPU timestamp.

#include "driver/driver.h"

#include "common/log.h"

using namespace vulkanizer;
using namespace vulkanizer::driver;

extern "C" {

CUresult CUDAAPI cuStreamCreate(CUstream* phStream, unsigned int Flags) {
    VULKANIZER_TRACE_API();
    if (!phStream) return CUDA_ERROR_INVALID_VALUE;
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    auto* stream = new CUstream_st();
    stream->ctx = s.ctx;
    stream->flags = Flags;
    Driver::get().registerHandle(stream);
    *phStream = stream;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuStreamDestroy_v2(CUstream hStream) {
    VULKANIZER_TRACE_API();
    if (!validHandle(hStream)) return CUDA_ERROR_INVALID_HANDLE;
    Driver::get().unregisterHandle(hStream);
    delete hStream;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuStreamSynchronize(CUstream hStream) {
    VULKANIZER_TRACE_API();
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!validStream(hStream)) return CUDA_ERROR_INVALID_HANDLE;
    return toCuResult(s.dev->finish());
}

CUresult CUDAAPI cuStreamQuery(CUstream hStream) {
    VULKANIZER_TRACE_API();
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!validStream(hStream)) return CUDA_ERROR_INVALID_HANDLE;
    const uint64_t serial = s.dev->currentSerial();
    if (VkResult vr = s.dev->flush()) return toCuResult(vr);
    return s.dev->isComplete(serial) ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
}

CUresult CUDAAPI cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags) {
    VULKANIZER_TRACE_API();
    if (!validStream(hStream) || !validHandle(hEvent)) return CUDA_ERROR_INVALID_HANDLE;
    return CUDA_SUCCESS;  // one queue: everything recorded later already runs after the event
}

CUresult CUDAAPI cuEventCreate(CUevent* phEvent, unsigned int Flags) {
    VULKANIZER_TRACE_API();
    if (!phEvent) return CUDA_ERROR_INVALID_VALUE;
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    auto* event = new CUevent_st();
    event->ctx = s.ctx;
    event->flags = Flags;
    if (!(Flags & CU_EVENT_DISABLE_TIMING)) event->timestamp = s.dev->allocateTimestamp();
    Driver::get().registerHandle(event);
    *phEvent = event;
    return CUDA_SUCCESS;
}

static vk::Device* deviceOf(CUevent e) { return Driver::get().slot(e->ctx->device)->vk.get(); }

CUresult CUDAAPI cuEventRecord(CUevent hEvent, CUstream hStream) {
    VULKANIZER_TRACE_API();
    if (!validHandle(hEvent) || !validStream(hStream)) return CUDA_ERROR_INVALID_HANDLE;
    vk::Device* dev = deviceOf(hEvent);
    if (hEvent->timestamp >= 0) {
        if (VkResult vr = dev->recordTimestamp(hEvent->timestamp, hEvent->serial)) return toCuResult(vr);
    } else {
        hEvent->serial = dev->currentSerial();
    }
    hEvent->recorded = true;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuEventSynchronize(CUevent hEvent) {
    VULKANIZER_TRACE_API();
    if (!validHandle(hEvent)) return CUDA_ERROR_INVALID_HANDLE;
    return toCuResult(deviceOf(hEvent)->wait(hEvent->serial));
}

CUresult CUDAAPI cuEventQuery(CUevent hEvent) {
    VULKANIZER_TRACE_API();
    if (!validHandle(hEvent)) return CUDA_ERROR_INVALID_HANDLE;
    vk::Device* dev = deviceOf(hEvent);
    if (VkResult vr = dev->flush()) return toCuResult(vr);  // otherwise it would never complete
    return dev->isComplete(hEvent->serial) ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
}

CUresult CUDAAPI cuEventElapsedTime(float* pMilliseconds, CUevent hStart, CUevent hEnd) {
    VULKANIZER_TRACE_API();
    if (!pMilliseconds) return CUDA_ERROR_INVALID_VALUE;
    if (!validHandle(hStart) || !validHandle(hEnd)) return CUDA_ERROR_INVALID_HANDLE;
    if (hStart->timestamp < 0 || hEnd->timestamp < 0) return CUDA_ERROR_INVALID_HANDLE;
    if (!hStart->recorded || !hEnd->recorded) return CUDA_ERROR_INVALID_HANDLE;
    double start, end;
    if (!deviceOf(hStart)->readTimestampNs(hStart->timestamp, start) ||
        !deviceOf(hEnd)->readTimestampNs(hEnd->timestamp, end))
        return CUDA_ERROR_NOT_READY;
    *pMilliseconds = float((end - start) * 1e-6);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuEventDestroy_v2(CUevent hEvent) {
    VULKANIZER_TRACE_API();
    if (!validHandle(hEvent)) return CUDA_ERROR_INVALID_HANDLE;
    if (hEvent->timestamp >= 0) deviceOf(hEvent)->freeTimestamp(hEvent->timestamp);
    Driver::get().unregisterHandle(hEvent);
    delete hEvent;
    return CUDA_SUCCESS;
}

}  // extern "C"
