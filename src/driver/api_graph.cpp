// CUDA graphs and stream capture. All streams share one queue, so a graph runs as its nodes in
// the order they were added, which respects their dependencies: explicitly added nodes can only
// depend on earlier ones, and capture appends work in the order the streams issued it. Kernel
// nodes keep their parameters packed (packParams), so a replay skips the launch bookkeeping.
//
// Capturing: kernel launches, copies, memsets, host functions, event records/waits, stream-ordered
// allocations and frees, and graph launches on a capturing stream become nodes (or dependency
// edges) instead of running. A stream joins another one's capture by waiting on an event recorded
// there (fork/join, as CUDA does it). Synchronizing a capturing stream invalidates its capture.
//
// Memory nodes allocate when they are created (GraphMemory); a child graph node holds a copy of the
// child's nodes and runs them in place.

#include "driver/driver.h"

#include "common/log.h"

#include <algorithm>
#include <atomic>

using namespace vulkanizer;
using namespace vulkanizer::driver;

namespace {

thread_local CUstreamCaptureMode t_captureMode = CU_STREAM_CAPTURE_MODE_GLOBAL;
std::atomic<uint64_t> g_nextCaptureId{1};

bool isRealStream(CUstream s) { return s && s != CU_STREAM_LEGACY && s != CU_STREAM_PER_THREAD && validHandle(s); }

void endCaptureOf(CUgraph g) {
    for (CUstream s : g->capturing) {
        if (!validHandle(s)) continue;
        s->capture = nullptr;
        s->captureStatus = CU_STREAM_CAPTURE_STATUS_NONE;
        s->captureOrigin = false;
        s->captureDeps.clear();
    }
    g->capturing.clear();
}

CUgraphNode addNode(CUgraph g, std::unique_ptr<CUgraphNode_st> node, const CUgraphNode* deps, size_t numDeps) {
    node->graph = g;
    node->deps.assign(deps, deps + numDeps);
    CUgraphNode n = node.get();
    Driver::get().registerHandle(n);
    g->nodes.push_back(std::move(node));
    return n;
}

void destroyGraph(CUgraph g) {
    Driver& d = Driver::get();
    for (const auto& n : g->nodes) d.unregisterHandle(n.get());
    d.unregisterHandle(g);
    delete g;
}

bool validDeps(const CUgraphNode* deps, size_t numDeps, CUgraph g) {
    if (numDeps && !deps) return false;
    for (size_t i = 0; i < numDeps; ++i)
        if (!validHandle(deps[i]) || deps[i]->graph != g) return false;
    return true;
}

// A kernel node's state from launch parameters (the v1 fields; v2 adds kern and ctx).
CUresult setKernel(CUgraphNode_st& n, CUfunction f, const unsigned grid[3], const unsigned block[3],
                   unsigned sharedMem, void** kernelParams, void** extra) {
    if (!validHandle(f)) return CUDA_ERROR_INVALID_HANDLE;
    std::vector<uint8_t> params;
    if (CUresult r = packParams(*f->entry, kernelParams, extra, params)) return r;
    n.type = CU_GRAPH_NODE_TYPE_KERNEL;
    n.func = f;
    std::copy(grid, grid + 3, n.grid);
    std::copy(block, block + 3, n.block);
    n.sharedMem = sharedMem;
    n.params = std::move(params);
    n.paramPointers.clear();
    return CUDA_SUCCESS;
}

template <class P>
CUresult setKernelFrom(CUgraphNode_st& n, const P* p) {
    if (!p) return CUDA_ERROR_INVALID_VALUE;
    const unsigned grid[3] = {p->gridDimX, p->gridDimY, p->gridDimZ}, block[3] = {p->blockDimX, p->blockDimY, p->blockDimZ};
    return setKernel(n, p->func, grid, block, p->sharedMemBytes, p->kernelParams, p->extra);
}

template <class P>
void getKernel(CUgraphNode_st& n, P* p) {
    n.paramPointers.clear();
    for (const ptx::Param& param : n.func->entry->params) n.paramPointers.push_back(n.params.data() + param.offset);
    p->func = n.func;
    p->gridDimX = n.grid[0], p->gridDimY = n.grid[1], p->gridDimZ = n.grid[2];
    p->blockDimX = n.block[0], p->blockDimY = n.block[1], p->blockDimZ = n.block[2];
    p->sharedMemBytes = n.sharedMem;
    p->kernelParams = n.paramPointers.data();
    p->extra = nullptr;
}

CUresult runNode(ApiScope& s, const CUgraphNode_st& n) {
    switch (n.type) {
        case CU_GRAPH_NODE_TYPE_GRAPH:
            for (const CUgraphNode_st& child : n.children)
                if (CUresult r = runNode(s, child)) return r;
            return CUDA_SUCCESS;
        case CU_GRAPH_NODE_TYPE_KERNEL:
            return launchKernel(s, n.func, n.grid, n.block, n.sharedMem, n.params, n.cooperative);
        case CU_GRAPH_NODE_TYPE_MEMCPY:
            return cuMemcpy(n.dst, n.src, n.bytes);
        case CU_GRAPH_NODE_TYPE_MEMSET:
            for (size_t row = 0; row < n.memset.height; ++row) {
                const CUdeviceptr dst = n.memset.dst + row * n.memset.pitch;
                CUresult r = n.memset.elementSize == 1   ? cuMemsetD8_v2(dst, (unsigned char)n.memset.value, n.memset.width)
                             : n.memset.elementSize == 2 ? cuMemsetD16_v2(dst, (unsigned short)n.memset.value, n.memset.width)
                                                         : cuMemsetD32_v2(dst, n.memset.value, n.memset.width);
                if (r) return r;
            }
            return CUDA_SUCCESS;
        case CU_GRAPH_NODE_TYPE_HOST:
            if (VkResult vr = s.dev->finish()) return toCuResult(vr);  // after the work before it
            n.fn(n.userData);
            return CUDA_SUCCESS;
        default:  // empty, event record/wait: ordering the single queue gives anyway; memory nodes allocated
                  // when they were created
            return CUDA_SUCCESS;
    }
}

// A memory node's allocation, owned by the graph.
CUresult allocateGraphMemory(ApiScope& s, CUgraph g, size_t bytes, CUdeviceptr& out) {
    if (bytes == 0) return CUDA_ERROR_INVALID_VALUE;
    vk::Buffer buf;
    if (VkResult vr = s.dev->allocateBuffer(bytes, buf)) return toCuResult(vr);
    s.slot->allocations[buf.address] = buf;
    s.slot->allocatedBytes += buf.size;
    s.slot->graphAllocations[buf.address] = GraphAllocation{};
    if (!g->memory) {
        g->memory = std::make_shared<GraphMemory>();
        g->memory->device = s.slot->ordinal;
    }
    g->memory->allocations.push_back(buf.address);
    out = buf.address;
    return CUDA_SUCCESS;
}

// A child graph node of `child`'s nodes and memory (a graph, or an executable graph's copy).
std::unique_ptr<CUgraphNode_st> childNode(const std::vector<CUgraphNode_st>& nodes,
                                          std::vector<std::shared_ptr<GraphMemory>> memory) {
    auto node = std::make_unique<CUgraphNode_st>();
    node->type = CU_GRAPH_NODE_TYPE_GRAPH;
    node->children = nodes;
    node->childMemory = std::move(memory);
    return node;
}

}  // namespace

namespace vulkanizer::driver {

GraphMemory::~GraphMemory() {
    std::lock_guard<std::recursive_mutex> lock(Driver::get().mutex());
    DeviceSlot* slot = Driver::get().slot(device);
    if (!slot || !slot->vk) return;
    for (CUdeviceptr p : allocations) {
        auto state = slot->graphAllocations.find(p);
        if (state == slot->graphAllocations.end()) continue;
        if (!state->second.released) {  // the application's now
            state->second.owned = false;
            continue;
        }
        slot->graphAllocations.erase(state);
        auto it = slot->allocations.find(p);
        if (it == slot->allocations.end()) continue;
        slot->allocatedBytes -= it->second.size;
        slot->vk->freeBufferAfter(it->second, slot->vk->currentSerial());  // after the launches recorded so far
        slot->allocations.erase(it);
    }
}

CUresult captureMemAlloc(ApiScope& s, CUstream cs, size_t bytes, CUdeviceptr& out) {
    if (cs->captureStatus == CU_STREAM_CAPTURE_STATUS_INVALIDATED) return CUDA_ERROR_STREAM_CAPTURE_INVALIDATED;
    if (CUresult r = allocateGraphMemory(s, cs->capture, bytes, out)) return r;
    auto node = std::make_unique<CUgraphNode_st>();
    node->type = CU_GRAPH_NODE_TYPE_MEM_ALLOC;
    node->dst = out;
    node->bytes = bytes;
    return captureNode(cs, std::move(node));
}

bool releaseGraphMemory(DeviceSlot& slot, CUdeviceptr dptr) {
    auto it = slot.graphAllocations.find(dptr);
    if (it == slot.graphAllocations.end()) return false;
    if (it->second.owned) {
        it->second.released = true;
        return true;
    }
    slot.graphAllocations.erase(it);  // no graph left: an ordinary free
    return false;
}

CUresult captureMemFree(CUstream cs, CUdeviceptr dptr) {
    if (DeviceSlot* slot = Driver::get().slot(cs->ctx ? cs->ctx->device : 0))
        if (auto it = slot->graphAllocations.find(dptr); it != slot->graphAllocations.end()) it->second.released = true;
    auto node = std::make_unique<CUgraphNode_st>();
    node->type = CU_GRAPH_NODE_TYPE_MEM_FREE;
    node->dst = dptr;
    return captureNode(cs, std::move(node));
}

CUstream capturingStream(CUstream hStream) {
    return isRealStream(hStream) && hStream->capture ? hStream : nullptr;
}

CUresult captureNode(CUstream hStream, std::unique_ptr<CUgraphNode_st> node) {
    if (hStream->captureStatus == CU_STREAM_CAPTURE_STATUS_INVALIDATED) return CUDA_ERROR_STREAM_CAPTURE_INVALIDATED;
    CUgraphNode n = addNode(hStream->capture, std::move(node), hStream->captureDeps.data(), hStream->captureDeps.size());
    hStream->captureDeps.assign(1, n);
    return CUDA_SUCCESS;
}

CUresult refuseIfCapturing(CUstream hStream) {
    CUstream s = capturingStream(hStream);
    if (!s) return CUDA_SUCCESS;
    for (CUstream c : s->capture->capturing)
        if (validHandle(c)) c->captureStatus = CU_STREAM_CAPTURE_STATUS_INVALIDATED;
    VULKANIZER_DEBUG("stream capture invalidated by a synchronizing call");
    return CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED;
}

CUresult joinCapture(CUstream hStream, CUevent hEvent) {
    if (!isRealStream(hStream)) return CUDA_ERROR_STREAM_CAPTURE_IMPLICIT;
    CUgraph g = hEvent->captureGraph;
    if (hStream->capture && hStream->capture != g) return CUDA_ERROR_STREAM_CAPTURE_MERGE;
    if (!hStream->capture) {
        hStream->capture = g;
        hStream->captureStatus = CU_STREAM_CAPTURE_STATUS_ACTIVE;
        hStream->captureOrigin = false;
        hStream->captureDeps.clear();
        g->capturing.push_back(hStream);
        for (CUstream c : g->capturing)
            if (validHandle(c) && c->captureOrigin) hStream->captureMode = c->captureMode;
    }
    for (CUgraphNode n : hEvent->captureDeps)
        if (std::find(hStream->captureDeps.begin(), hStream->captureDeps.end(), n) == hStream->captureDeps.end())
            hStream->captureDeps.push_back(n);
    return CUDA_SUCCESS;
}

}  // namespace vulkanizer::driver

extern "C" {

// ---- host functions ---------------------------------------------------------------------------

CUresult CUDAAPI cuLaunchHostFunc(CUstream hStream, CUhostFn fn, void* userData) {
    VULKANIZER_TRACE_API();
    if (!fn) return CUDA_ERROR_INVALID_VALUE;
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!validStream(hStream)) return CUDA_ERROR_INVALID_HANDLE;
    if (CUstream cs = capturingStream(hStream)) {
        auto node = std::make_unique<CUgraphNode_st>();
        node->type = CU_GRAPH_NODE_TYPE_HOST;
        node->fn = fn;
        node->userData = userData;
        return captureNode(cs, std::move(node));
    }
    if (VkResult vr = s.dev->finish()) return toCuResult(vr);  // runs after the stream's work so far
    fn(userData);
    return CUDA_SUCCESS;
}
CUresult CUDAAPI cuLaunchHostFunc_ptsz(CUstream hStream, CUhostFn fn, void* userData) {
    return cuLaunchHostFunc(hStream, fn, userData);
}
// CUDA 13.1: a synchronization mode, which only matters for blocking host functions.
CUresult CUDAAPI cuLaunchHostFunc_v2(CUstream hStream, CUhostFn fn, void* userData, unsigned int) {
    return cuLaunchHostFunc(hStream, fn, userData);
}
CUresult CUDAAPI cuLaunchHostFunc_v2_ptsz(CUstream hStream, CUhostFn fn, void* userData, unsigned int) {
    return cuLaunchHostFunc(hStream, fn, userData);
}

// ---- stream capture ---------------------------------------------------------------------------

CUresult CUDAAPI cuStreamBeginCapture_v2(CUstream hStream, CUstreamCaptureMode mode) {
    VULKANIZER_TRACE_API();
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!isRealStream(hStream)) return validStream(hStream) ? CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED : CUDA_ERROR_INVALID_HANDLE;
    if (hStream->capture) return CUDA_ERROR_ILLEGAL_STATE;
    if (mode > CU_STREAM_CAPTURE_MODE_RELAXED) return CUDA_ERROR_INVALID_VALUE;
    auto* g = new CUgraph_st();
    g->ctx = hStream->ctx;
    g->captureId = g_nextCaptureId++;
    g->capturing.push_back(hStream);
    Driver::get().registerHandle(g);
    hStream->capture = g;
    hStream->captureStatus = CU_STREAM_CAPTURE_STATUS_ACTIVE;
    hStream->captureMode = mode;
    hStream->captureOrigin = true;
    hStream->captureDeps.clear();
    return CUDA_SUCCESS;
}
CUresult CUDAAPI cuStreamBeginCapture_v2_ptsz(CUstream hStream, CUstreamCaptureMode mode) {
    return cuStreamBeginCapture_v2(hStream, mode);
}
#undef cuStreamBeginCapture
CUresult CUDAAPI cuStreamBeginCapture(CUstream hStream);  // CUDA 10.0: global mode
CUresult CUDAAPI cuStreamBeginCapture(CUstream hStream) {
    return cuStreamBeginCapture_v2(hStream, CU_STREAM_CAPTURE_MODE_GLOBAL);
}
CUresult CUDAAPI cuStreamBeginCapture_ptsz(CUstream hStream) {
    return cuStreamBeginCapture_v2(hStream, CU_STREAM_CAPTURE_MODE_GLOBAL);
}

CUresult CUDAAPI cuStreamEndCapture(CUstream hStream, CUgraph* phGraph) {
    VULKANIZER_TRACE_API();
    if (!phGraph) return CUDA_ERROR_INVALID_VALUE;
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!isRealStream(hStream) || !hStream->capture) return CUDA_ERROR_ILLEGAL_STATE;
    if (!hStream->captureOrigin) return CUDA_ERROR_STREAM_CAPTURE_UNMATCHED;
    CUgraph g = hStream->capture;
    const bool invalidated = hStream->captureStatus == CU_STREAM_CAPTURE_STATUS_INVALIDATED;
    endCaptureOf(g);
    if (invalidated) {
        destroyGraph(g);
        *phGraph = nullptr;
        return CUDA_ERROR_STREAM_CAPTURE_INVALIDATED;
    }
    *phGraph = g;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI cuStreamEndCapture_ptsz(CUstream hStream, CUgraph* phGraph) {
    return cuStreamEndCapture(hStream, phGraph);
}

CUresult CUDAAPI cuStreamIsCapturing(CUstream hStream, CUstreamCaptureStatus* captureStatus) {
    if (!captureStatus) return CUDA_ERROR_INVALID_VALUE;
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!validStream(hStream)) return CUDA_ERROR_INVALID_HANDLE;
    CUstream cs = capturingStream(hStream);
    *captureStatus = cs ? cs->captureStatus : CU_STREAM_CAPTURE_STATUS_NONE;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI cuStreamIsCapturing_ptsz(CUstream hStream, CUstreamCaptureStatus* captureStatus) {
    return cuStreamIsCapturing(hStream, captureStatus);
}

CUresult CUDAAPI cuStreamGetCaptureInfo_v3(CUstream hStream, CUstreamCaptureStatus* captureStatus_out,
                                           cuuint64_t* id_out, CUgraph* graph_out,
                                           const CUgraphNode** dependencies_out,
                                           const CUgraphEdgeData** edgeData_out, size_t* numDependencies_out) {
    if (!captureStatus_out) return CUDA_ERROR_INVALID_VALUE;
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!validStream(hStream)) return CUDA_ERROR_INVALID_HANDLE;
    CUstream cs = capturingStream(hStream);
    *captureStatus_out = cs ? cs->captureStatus : CU_STREAM_CAPTURE_STATUS_NONE;
    if (id_out) *id_out = cs ? cs->capture->captureId : 0;
    if (graph_out) *graph_out = cs ? cs->capture : nullptr;
    if (dependencies_out) *dependencies_out = cs && !cs->captureDeps.empty() ? cs->captureDeps.data() : nullptr;
    if (edgeData_out) *edgeData_out = nullptr;  // all edges are default ones
    if (numDependencies_out) *numDependencies_out = cs ? cs->captureDeps.size() : 0;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI cuStreamGetCaptureInfo_v3_ptsz(CUstream hStream, CUstreamCaptureStatus* captureStatus_out,
                                                cuuint64_t* id_out, CUgraph* graph_out,
                                                const CUgraphNode** dependencies_out,
                                                const CUgraphEdgeData** edgeData_out, size_t* numDependencies_out) {
    return cuStreamGetCaptureInfo_v3(hStream, captureStatus_out, id_out, graph_out, dependencies_out, edgeData_out,
                                     numDependencies_out);
}
CUresult CUDAAPI cuStreamGetCaptureInfo_v2(CUstream hStream, CUstreamCaptureStatus* captureStatus_out,
                                           cuuint64_t* id_out, CUgraph* graph_out,
                                           const CUgraphNode** dependencies_out, size_t* numDependencies_out) {
    return cuStreamGetCaptureInfo_v3(hStream, captureStatus_out, id_out, graph_out, dependencies_out, nullptr,
                                     numDependencies_out);
}
CUresult CUDAAPI cuStreamGetCaptureInfo_v2_ptsz(CUstream hStream, CUstreamCaptureStatus* captureStatus_out,
                                                cuuint64_t* id_out, CUgraph* graph_out,
                                                const CUgraphNode** dependencies_out, size_t* numDependencies_out) {
    return cuStreamGetCaptureInfo_v3(hStream, captureStatus_out, id_out, graph_out, dependencies_out, nullptr,
                                     numDependencies_out);
}
#undef cuStreamGetCaptureInfo
CUresult CUDAAPI cuStreamGetCaptureInfo(CUstream hStream, CUstreamCaptureStatus* captureStatus_out, cuuint64_t* id_out);
CUresult CUDAAPI cuStreamGetCaptureInfo(CUstream hStream, CUstreamCaptureStatus* captureStatus_out, cuuint64_t* id_out) {
    return cuStreamGetCaptureInfo_v3(hStream, captureStatus_out, id_out, nullptr, nullptr, nullptr, nullptr);
}
CUresult CUDAAPI cuStreamGetCaptureInfo_ptsz(CUstream hStream, CUstreamCaptureStatus* captureStatus_out,
                                             cuuint64_t* id_out) {
    return cuStreamGetCaptureInfo_v3(hStream, captureStatus_out, id_out, nullptr, nullptr, nullptr, nullptr);
}

// flags: CU_STREAM_ADD_CAPTURE_DEPENDENCIES (0) or CU_STREAM_SET_CAPTURE_DEPENDENCIES (1)
CUresult CUDAAPI cuStreamUpdateCaptureDependencies(CUstream hStream, CUgraphNode* dependencies, size_t numDependencies,
                                                   unsigned int flags) {
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    CUstream cs = capturingStream(hStream);
    if (!cs) return CUDA_ERROR_ILLEGAL_STATE;
    if (flags > 1 || !validDeps(dependencies, numDependencies, cs->capture)) return CUDA_ERROR_INVALID_VALUE;
    if (flags == 1) cs->captureDeps.clear();
    for (size_t i = 0; i < numDependencies; ++i)
        if (std::find(cs->captureDeps.begin(), cs->captureDeps.end(), dependencies[i]) == cs->captureDeps.end())
            cs->captureDeps.push_back(dependencies[i]);
    return CUDA_SUCCESS;
}
CUresult CUDAAPI cuStreamUpdateCaptureDependencies_v2(CUstream hStream, CUgraphNode* dependencies,
                                                      const CUgraphEdgeData*, size_t numDependencies,
                                                      unsigned int flags) {
    return cuStreamUpdateCaptureDependencies(hStream, dependencies, numDependencies, flags);
}

// Capture modes restrict what other threads may call during a capture; nothing here depends on
// it, so the mode is only kept for the caller.
CUresult CUDAAPI cuThreadExchangeStreamCaptureMode(CUstreamCaptureMode* mode) {
    if (!mode || *mode > CU_STREAM_CAPTURE_MODE_RELAXED) return CUDA_ERROR_INVALID_VALUE;
    std::swap(*mode, t_captureMode);
    return CUDA_SUCCESS;
}

// ---- graphs -----------------------------------------------------------------------------------

CUresult CUDAAPI cuGraphCreate(CUgraph* phGraph, unsigned int flags) {
    VULKANIZER_TRACE_API();
    if (!phGraph || flags) return CUDA_ERROR_INVALID_VALUE;
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    auto* g = new CUgraph_st();
    g->ctx = s.ctx;
    Driver::get().registerHandle(g);
    *phGraph = g;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGraphDestroy(CUgraph hGraph) {
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!validHandle(hGraph)) return CUDA_ERROR_INVALID_HANDLE;
    if (!hGraph->capturing.empty()) return CUDA_ERROR_ILLEGAL_STATE;
    destroyGraph(hGraph);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGraphAddKernelNode_v2(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies,
                                         size_t numDependencies, const CUDA_KERNEL_NODE_PARAMS_v2* nodeParams) {
    VULKANIZER_TRACE_API();
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!phGraphNode || !nodeParams || !validHandle(hGraph)) return CUDA_ERROR_INVALID_VALUE;
    if (!validDeps(dependencies, numDependencies, hGraph)) return CUDA_ERROR_INVALID_VALUE;
    if (nodeParams->kern) return CUDA_ERROR_NOT_SUPPORTED;  // CUkernel (library) handles
    auto node = std::make_unique<CUgraphNode_st>();
    if (CUresult r = setKernelFrom(*node, nodeParams)) return r;
    *phGraphNode = addNode(hGraph, std::move(node), dependencies, numDependencies);
    return CUDA_SUCCESS;
}
#undef cuGraphAddKernelNode
CUresult CUDAAPI cuGraphAddKernelNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies,
                                      size_t numDependencies, const CUDA_KERNEL_NODE_PARAMS_v1* nodeParams);
CUresult CUDAAPI cuGraphAddKernelNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies,
                                      size_t numDependencies, const CUDA_KERNEL_NODE_PARAMS_v1* nodeParams) {
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!phGraphNode || !nodeParams || !validHandle(hGraph)) return CUDA_ERROR_INVALID_VALUE;
    if (!validDeps(dependencies, numDependencies, hGraph)) return CUDA_ERROR_INVALID_VALUE;
    auto node = std::make_unique<CUgraphNode_st>();
    if (CUresult r = setKernelFrom(*node, nodeParams)) return r;
    *phGraphNode = addNode(hGraph, std::move(node), dependencies, numDependencies);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGraphKernelNodeGetParams_v2(CUgraphNode hNode, CUDA_KERNEL_NODE_PARAMS_v2* nodeParams) {
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!validHandle(hNode) || !nodeParams || hNode->type != CU_GRAPH_NODE_TYPE_KERNEL) return CUDA_ERROR_INVALID_VALUE;
    getKernel(*hNode, nodeParams);
    nodeParams->kern = nullptr;
    nodeParams->ctx = hNode->graph->ctx;
    return CUDA_SUCCESS;
}
#undef cuGraphKernelNodeGetParams
CUresult CUDAAPI cuGraphKernelNodeGetParams(CUgraphNode hNode, CUDA_KERNEL_NODE_PARAMS_v1* nodeParams);
CUresult CUDAAPI cuGraphKernelNodeGetParams(CUgraphNode hNode, CUDA_KERNEL_NODE_PARAMS_v1* nodeParams) {
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!validHandle(hNode) || !nodeParams || hNode->type != CU_GRAPH_NODE_TYPE_KERNEL) return CUDA_ERROR_INVALID_VALUE;
    getKernel(*hNode, nodeParams);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGraphKernelNodeSetParams_v2(CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS_v2* nodeParams) {
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!validHandle(hNode) || hNode->type != CU_GRAPH_NODE_TYPE_KERNEL) return CUDA_ERROR_INVALID_VALUE;
    if (nodeParams && nodeParams->kern) return CUDA_ERROR_NOT_SUPPORTED;
    return setKernelFrom(*hNode, nodeParams);
}
#undef cuGraphKernelNodeSetParams
CUresult CUDAAPI cuGraphKernelNodeSetParams(CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS_v1* nodeParams);
CUresult CUDAAPI cuGraphKernelNodeSetParams(CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS_v1* nodeParams) {
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!validHandle(hNode) || hNode->type != CU_GRAPH_NODE_TYPE_KERNEL) return CUDA_ERROR_INVALID_VALUE;
    return setKernelFrom(*hNode, nodeParams);
}

CUresult CUDAAPI cuGraphAddMemsetNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies,
                                      size_t numDependencies, const CUDA_MEMSET_NODE_PARAMS* memsetParams,
                                      CUcontext ctx) {
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!phGraphNode || !memsetParams || !validHandle(hGraph)) return CUDA_ERROR_INVALID_VALUE;
    if (!validDeps(dependencies, numDependencies, hGraph)) return CUDA_ERROR_INVALID_VALUE;
    const unsigned es = memsetParams->elementSize;
    if ((es != 1 && es != 2 && es != 4) || (memsetParams->height > 1 && memsetParams->pitch < memsetParams->width * es))
        return CUDA_ERROR_INVALID_VALUE;
    auto node = std::make_unique<CUgraphNode_st>();
    node->type = CU_GRAPH_NODE_TYPE_MEMSET;
    node->memset = *memsetParams;
    *phGraphNode = addNode(hGraph, std::move(node), dependencies, numDependencies);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGraphAddHostNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies,
                                    size_t numDependencies, const CUDA_HOST_NODE_PARAMS* nodeParams) {
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!phGraphNode || !nodeParams || !nodeParams->fn || !validHandle(hGraph)) return CUDA_ERROR_INVALID_VALUE;
    if (!validDeps(dependencies, numDependencies, hGraph)) return CUDA_ERROR_INVALID_VALUE;
    auto node = std::make_unique<CUgraphNode_st>();
    node->type = CU_GRAPH_NODE_TYPE_HOST;
    node->fn = nodeParams->fn;
    node->userData = nodeParams->userData;
    *phGraphNode = addNode(hGraph, std::move(node), dependencies, numDependencies);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGraphAddEmptyNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies,
                                     size_t numDependencies) {
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!phGraphNode || !validHandle(hGraph)) return CUDA_ERROR_INVALID_VALUE;
    if (!validDeps(dependencies, numDependencies, hGraph)) return CUDA_ERROR_INVALID_VALUE;
    *phGraphNode = addNode(hGraph, std::make_unique<CUgraphNode_st>(), dependencies, numDependencies);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGraphGetNodes(CUgraph hGraph, CUgraphNode* nodes, size_t* numNodes) {
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!validHandle(hGraph) || !numNodes) return CUDA_ERROR_INVALID_VALUE;
    if (nodes)
        for (size_t i = 0; i < *numNodes && i < hGraph->nodes.size(); ++i) nodes[i] = hGraph->nodes[i].get();
    *numNodes = hGraph->nodes.size();
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGraphNodeGetType(CUgraphNode hNode, CUgraphNodeType* type) {
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!validHandle(hNode) || !type) return CUDA_ERROR_INVALID_VALUE;
    *type = hNode->type;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGraphNodeGetDependencies_v2(CUgraphNode hNode, CUgraphNode* dependencies, CUgraphEdgeData* edgeData,
                                               size_t* numDependencies) {
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!validHandle(hNode) || !numDependencies) return CUDA_ERROR_INVALID_VALUE;
    for (size_t i = 0; i < *numDependencies && i < hNode->deps.size(); ++i) {
        if (dependencies) dependencies[i] = hNode->deps[i];
        if (edgeData) edgeData[i] = CUgraphEdgeData{};
    }
    *numDependencies = hNode->deps.size();
    return CUDA_SUCCESS;
}
CUresult CUDAAPI cuGraphNodeGetDependencies(CUgraphNode hNode, CUgraphNode* dependencies, size_t* numDependencies) {
    return cuGraphNodeGetDependencies_v2(hNode, dependencies, nullptr, numDependencies);
}

// ---- memory nodes and child graphs ------------------------------------------------------------

CUresult CUDAAPI cuGraphAddMemAllocNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies,
                                        size_t numDependencies, CUDA_MEM_ALLOC_NODE_PARAMS* nodeParams) {
    VULKANIZER_TRACE_API();
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!phGraphNode || !nodeParams || !validHandle(hGraph)) return CUDA_ERROR_INVALID_VALUE;
    if (!validDeps(dependencies, numDependencies, hGraph)) return CUDA_ERROR_INVALID_VALUE;
    if (nodeParams->poolProps.allocType != CU_MEM_ALLOCATION_TYPE_PINNED) return CUDA_ERROR_INVALID_VALUE;
    if (CUresult r = allocateGraphMemory(s, hGraph, nodeParams->bytesize, nodeParams->dptr)) return r;
    auto node = std::make_unique<CUgraphNode_st>();
    node->type = CU_GRAPH_NODE_TYPE_MEM_ALLOC;
    node->dst = nodeParams->dptr;
    node->bytes = nodeParams->bytesize;
    *phGraphNode = addNode(hGraph, std::move(node), dependencies, numDependencies);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGraphMemAllocNodeGetParams(CUgraphNode hNode, CUDA_MEM_ALLOC_NODE_PARAMS* params_out) {
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!validHandle(hNode) || !params_out || hNode->type != CU_GRAPH_NODE_TYPE_MEM_ALLOC) return CUDA_ERROR_INVALID_VALUE;
    *params_out = {};
    params_out->poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    params_out->poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    params_out->poolProps.location.id = s.slot->ordinal;
    params_out->bytesize = hNode->bytes;
    params_out->dptr = hNode->dst;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGraphAddMemFreeNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies,
                                       size_t numDependencies, CUdeviceptr dptr) {
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    auto state = s.slot->graphAllocations.find(dptr);
    if (!phGraphNode || !validHandle(hGraph) || state == s.slot->graphAllocations.end()) return CUDA_ERROR_INVALID_VALUE;
    if (!validDeps(dependencies, numDependencies, hGraph)) return CUDA_ERROR_INVALID_VALUE;
    state->second.released = true;
    auto node = std::make_unique<CUgraphNode_st>();
    node->type = CU_GRAPH_NODE_TYPE_MEM_FREE;
    node->dst = dptr;
    *phGraphNode = addNode(hGraph, std::move(node), dependencies, numDependencies);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGraphMemFreeNodeGetParams(CUgraphNode hNode, CUdeviceptr* dptr_out) {
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!validHandle(hNode) || !dptr_out || hNode->type != CU_GRAPH_NODE_TYPE_MEM_FREE) return CUDA_ERROR_INVALID_VALUE;
    *dptr_out = hNode->dst;
    return CUDA_SUCCESS;
}

// Graph memory is not cached beyond the graphs that own it: nothing to trim.
CUresult CUDAAPI cuDeviceGraphMemTrim(CUdevice device) {
    return Driver::get().slot(device) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_DEVICE;
}

CUresult CUDAAPI cuGraphAddChildGraphNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies,
                                          size_t numDependencies, CUgraph childGraph) {
    VULKANIZER_TRACE_API();
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!phGraphNode || !validHandle(hGraph) || !validHandle(childGraph) || childGraph == hGraph)
        return CUDA_ERROR_INVALID_VALUE;
    if (!validDeps(dependencies, numDependencies, hGraph)) return CUDA_ERROR_INVALID_VALUE;
    std::vector<CUgraphNode_st> nodes;
    for (const auto& n : childGraph->nodes) nodes.push_back(*n);
    std::vector<std::shared_ptr<GraphMemory>> memory;
    if (childGraph->memory) memory.push_back(childGraph->memory);
    *phGraphNode = addNode(hGraph, childNode(nodes, std::move(memory)), dependencies, numDependencies);
    return CUDA_SUCCESS;
}

// ---- executable graphs ------------------------------------------------------------------------

CUresult CUDAAPI cuGraphInstantiateWithFlags(CUgraphExec* phGraphExec, CUgraph hGraph, unsigned long long flags) {
    VULKANIZER_TRACE_API();
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!phGraphExec || !validHandle(hGraph)) return CUDA_ERROR_INVALID_VALUE;
    auto* e = new CUgraphExec_st();
    e->ctx = hGraph->ctx;
    for (const auto& n : hGraph->nodes) {
        e->index[n.get()] = e->nodes.size();
        e->nodes.push_back(*n);
    }
    if (hGraph->memory) e->memory.push_back(hGraph->memory);
    Driver::get().registerHandle(e);
    *phGraphExec = e;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGraphExecKernelNodeSetParams_v2(CUgraphExec hGraphExec, CUgraphNode hNode,
                                                   const CUDA_KERNEL_NODE_PARAMS_v2* nodeParams) {
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!validHandle(hGraphExec) || !validHandle(hNode)) return CUDA_ERROR_INVALID_VALUE;
    auto it = hGraphExec->index.find(hNode);
    if (it == hGraphExec->index.end() || hGraphExec->nodes[it->second].type != CU_GRAPH_NODE_TYPE_KERNEL)
        return CUDA_ERROR_INVALID_VALUE;
    if (nodeParams && nodeParams->kern) return CUDA_ERROR_NOT_SUPPORTED;
    return setKernelFrom(hGraphExec->nodes[it->second], nodeParams);
}
#undef cuGraphExecKernelNodeSetParams
CUresult CUDAAPI cuGraphExecKernelNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode,
                                                const CUDA_KERNEL_NODE_PARAMS_v1* nodeParams);
CUresult CUDAAPI cuGraphExecKernelNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode,
                                                const CUDA_KERNEL_NODE_PARAMS_v1* nodeParams) {
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!validHandle(hGraphExec) || !validHandle(hNode)) return CUDA_ERROR_INVALID_VALUE;
    auto it = hGraphExec->index.find(hNode);
    if (it == hGraphExec->index.end() || hGraphExec->nodes[it->second].type != CU_GRAPH_NODE_TYPE_KERNEL)
        return CUDA_ERROR_INVALID_VALUE;
    return setKernelFrom(hGraphExec->nodes[it->second], nodeParams);
}

// Takes over the parameters of a graph with the same shape: same node count and types, in the
// same order (what re-capturing the same work gives). Kernel nodes may change function.
CUresult CUDAAPI cuGraphExecUpdate_v2(CUgraphExec hGraphExec, CUgraph hGraph, CUgraphExecUpdateResultInfo* resultInfo) {
    VULKANIZER_TRACE_API();
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!validHandle(hGraphExec) || !validHandle(hGraph)) return CUDA_ERROR_INVALID_VALUE;
    auto fail = [&](CUgraphExecUpdateResult why, CUgraphNode node) {
        if (resultInfo) *resultInfo = {why, node, nullptr};
        return CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE;
    };
    if (hGraph->nodes.size() != hGraphExec->nodes.size()) return fail(CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED, nullptr);
    for (size_t i = 0; i < hGraph->nodes.size(); ++i)
        if (hGraph->nodes[i]->type != hGraphExec->nodes[i].type)
            return fail(CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED, hGraph->nodes[i].get());
    for (size_t i = 0; i < hGraph->nodes.size(); ++i) {
        hGraphExec->nodes[i] = *hGraph->nodes[i];
        hGraphExec->index[hGraph->nodes[i].get()] = i;
    }
    if (hGraph->memory) hGraphExec->memory.push_back(hGraph->memory);  // the new nodes may use it
    if (resultInfo) *resultInfo = {CU_GRAPH_EXEC_UPDATE_SUCCESS, nullptr, nullptr};
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGraphUpload(CUgraphExec hGraphExec, CUstream hStream) {
    if (!validStream(hStream)) return CUDA_ERROR_INVALID_HANDLE;
    return validHandle(hGraphExec) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;  // nothing to prepare
}
CUresult CUDAAPI cuGraphUpload_ptsz(CUgraphExec hGraphExec, CUstream hStream) { return cuGraphUpload(hGraphExec, hStream); }

CUresult CUDAAPI cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream) {
    VULKANIZER_TRACE_API();
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!validHandle(hGraphExec)) return CUDA_ERROR_INVALID_VALUE;
    if (!validStream(hStream)) return CUDA_ERROR_INVALID_HANDLE;
    if (CUstream cs = capturingStream(hStream))  // a child graph node of the capture
        return captureNode(cs, childNode(hGraphExec->nodes, hGraphExec->memory));
    for (const CUgraphNode_st& n : hGraphExec->nodes)
        if (CUresult r = runNode(s, n)) return r;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI cuGraphLaunch_ptsz(CUgraphExec hGraphExec, CUstream hStream) { return cuGraphLaunch(hGraphExec, hStream); }

CUresult CUDAAPI cuGraphExecDestroy(CUgraphExec hGraphExec) {
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!validHandle(hGraphExec)) return CUDA_ERROR_INVALID_VALUE;
    Driver::get().unregisterHandle(hGraphExec);
    delete hGraphExec;
    return CUDA_SUCCESS;
}

}  // extern "C"
