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
    if (CUresult r = refuseIfCapturing(hStream)) return r;
    if (VkResult vr = s.dev->finish()) return toCuResult(vr);
    return checkDeviceAssert(*s.slot, true);
}

CUresult CUDAAPI cuStreamQuery(CUstream hStream) {
    VULKANIZER_TRACE_API();
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!validStream(hStream)) return CUDA_ERROR_INVALID_HANDLE;
    if (CUresult r = refuseIfCapturing(hStream)) return r;
    const uint64_t serial = s.dev->currentSerial();
    if (VkResult vr = s.dev->flush()) return toCuResult(vr);
    if (!s.dev->isComplete(serial)) return CUDA_ERROR_NOT_READY;
    return checkDeviceAssert(*s.slot, false);
}

CUresult CUDAAPI cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags) {
    VULKANIZER_TRACE_API();
    if (!validStream(hStream) || !validHandle(hEvent)) return CUDA_ERROR_INVALID_HANDLE;
    // An event recorded during a capture forks or joins it: the waiting stream captures too.
    if (hEvent->captureGraph) return joinCapture(hStream, hEvent);
    // One queue per device: on the event's own device everything recorded later already runs after
    // it. Another device's event (multi-GPU) is waited for on the host.
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    const bool realStream = hStream && hStream != CU_STREAM_LEGACY && hStream != CU_STREAM_PER_THREAD;
    const CUdevice waiting = realStream ? hStream->ctx->device : s.ctx->device;
    if (hEvent->recorded && hEvent->ctx->device != waiting)
        return toCuResult(Driver::get().slot(hEvent->ctx->device)->vk->wait(hEvent->serial));
    return CUDA_SUCCESS;
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

// A failed device-side assert of the work before a completed event (or of later work, already).
static CUresult eventAssert(CUevent e) {
    std::lock_guard<std::recursive_mutex> lock(Driver::get().mutex());
    return checkDeviceAssert(*Driver::get().slot(e->ctx->device), false);
}

CUresult CUDAAPI cuEventRecord(CUevent hEvent, CUstream hStream) {
    VULKANIZER_TRACE_API();
    if (!validHandle(hEvent) || !validStream(hStream)) return CUDA_ERROR_INVALID_HANDLE;
    if (CUstream cs = capturingStream(hStream)) {  // marks a point of the capture, runs nothing
        hEvent->captureGraph = cs->capture;
        hEvent->captureDeps = cs->captureDeps;
        return CUDA_SUCCESS;
    }
    hEvent->captureGraph = nullptr;
    vk::Device* dev = deviceOf(hEvent);
    if (hEvent->timestamp >= 0) {
        if (VkResult vr = dev->recordTimestamp(hEvent->timestamp, hEvent->serial)) return toCuResult(vr);
    } else {
        hEvent->serial = dev->currentSerial();
    }
    hEvent->recorded = true;
    // Other processes wait for an interprocess event without seeing this device's progress
    // (api_ipc.cpp): it completes here.
    if (hEvent->flags & CU_EVENT_INTERPROCESS) return toCuResult(dev->wait(hEvent->serial));
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuEventSynchronize(CUevent hEvent) {
    VULKANIZER_TRACE_API();
    if (!validHandle(hEvent)) return CUDA_ERROR_INVALID_HANDLE;
    if (VkResult vr = deviceOf(hEvent)->wait(hEvent->serial)) return toCuResult(vr);
    return eventAssert(hEvent);
}

CUresult CUDAAPI cuEventQuery(CUevent hEvent) {
    VULKANIZER_TRACE_API();
    if (!validHandle(hEvent)) return CUDA_ERROR_INVALID_HANDLE;
    vk::Device* dev = deviceOf(hEvent);
    if (VkResult vr = dev->flush()) return toCuResult(vr);  // otherwise it would never complete
    if (!dev->isComplete(hEvent->serial)) return CUDA_ERROR_NOT_READY;
    return eventAssert(hEvent);
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

// CUDA 12.8+: the same, under the name cuda.h 13 maps cuEventElapsedTime to.
CUresult CUDAAPI cuEventElapsedTime_v2(float* pMilliseconds, CUevent hStart, CUevent hEnd) {
    return cuEventElapsedTime(pMilliseconds, hStart, hEnd);
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
