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
// child's nodes and runs them in place. Instantiation orders the nodes topologically (edges added
// later with cuGraphAddDependencies may point backwards), stable in the order of creation.
//
// Conditional nodes (if, while, switch) own their body graphs. The condition is a 32-bit value in
// device memory (ConditionSlot) that kernels set with cudaGraphSetConditional; when a launch reaches
// the node, the host waits for the work before it and reads the value, so a graph with conditional
// nodes runs its bodies, and returns from cuGraphLaunch, only after the GPU got there.

#include "driver/driver.h"

#include "common/log.h"

#include <algorithm>
#include <atomic>
#include <set>

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

// Drops count references; the last one runs the destructor, after the work queued so far.
void releaseUserObject(CUuserObject o, unsigned int count) {
    if (o->refs.fetch_sub(count) != count) return;
    Driver::get().unregisterHandle(o);
    CUcontext ctx = nullptr;
    if (cuCtxGetCurrent(&ctx) == CUDA_SUCCESS && ctx) cuCtxSynchronize();
    if (o->destroy) o->destroy(o->ptr);
    delete o;
}

void releaseUserObjects(const std::map<CUuserObject, unsigned int>& objects) {
    for (const auto& [o, count] : objects) releaseUserObject(o, count);
}

void retainUserObjects(const std::map<CUuserObject, unsigned int>& objects) {
    for (const auto& [o, count] : objects) o->refs += count;
}

void destroyGraph(CUgraph g) {
    Driver& d = Driver::get();
    for (const auto& n : g->nodes) d.unregisterHandle(n.get());
    d.unregisterHandle(g);
    releaseUserObjects(g->userObjects);
    delete g;
}

// A conditional node's body graph, owned by the node (and its copies).
std::shared_ptr<CUgraph_st> makeBody(CUcontext ctx) {
    auto* g = new CUgraph_st();
    g->ctx = ctx;
    Driver::get().registerHandle(g);
    return std::shared_ptr<CUgraph_st>(g, [](CUgraph_st* b) { destroyGraph(b); });
}

// The graph's nodes in a topological order, stable in the order they were added.
std::vector<const CUgraphNode_st*> ordered(const CUgraph_st& g) {
    std::unordered_map<const CUgraphNode_st*, size_t> pending;  // unfinished dependencies
    std::unordered_map<const CUgraphNode_st*, std::vector<size_t>> dependents;
    std::unordered_map<const CUgraphNode_st*, size_t> position;
    for (size_t i = 0; i < g.nodes.size(); ++i) position[g.nodes[i].get()] = i;
    for (const auto& n : g.nodes) {
        pending[n.get()] = n->deps.size();
        for (CUgraphNode d : n->deps) dependents[d].push_back(position[n.get()]);
    }
    std::vector<const CUgraphNode_st*> out;
    std::set<size_t> ready;
    for (size_t i = 0; i < g.nodes.size(); ++i)
        if (!pending[g.nodes[i].get()]) ready.insert(i);
    while (!ready.empty()) {
        const CUgraphNode_st* n = g.nodes[*ready.begin()].get();
        ready.erase(ready.begin());
        out.push_back(n);
        for (size_t i : dependents[n])
            if (--pending[g.nodes[i].get()] == 0) ready.insert(i);
    }
    return out;  // shorter than the graph if it has a cycle
}

// Fills in what an executable copy of the node needs: the bodies of conditional nodes (also nested
// ones and those in child graphs) as node lists, and the memory they use.
void freeze(CUgraphNode_st& n, std::vector<std::shared_ptr<GraphMemory>>& memory) {
    if (n.type == CU_GRAPH_NODE_TYPE_GRAPH)
        for (CUgraphNode_st& c : n.children) freeze(c, memory);
    if (n.type != CU_GRAPH_NODE_TYPE_CONDITIONAL) return;
    n.bodyNodes.clear();
    n.bodyConditions.clear();
    for (const auto& body : n.bodies) {
        std::vector<CUgraphNode_st> nodes;
        for (const CUgraphNode_st* c : ordered(*body)) nodes.push_back(*c);
        for (CUgraphNode_st& c : nodes) freeze(c, memory);
        n.bodyNodes.push_back(std::move(nodes));
        n.bodyConditions.push_back(body->conditions);
        if (body->memory) memory.push_back(body->memory);
    }
}

// Executable copies of a graph's nodes, in order.
CUresult instantiateNodes(const CUgraph_st& g, std::vector<CUgraphNode_st>& nodes,
                          std::unordered_map<CUgraphNode, size_t>& index,
                          std::vector<std::shared_ptr<GraphMemory>>& memory) {
    const auto order = ordered(g);
    if (order.size() != g.nodes.size()) return CUDA_ERROR_INVALID_VALUE;  // a cycle
    nodes.clear();
    index.clear();
    for (const CUgraphNode_st* n : order) {
        index[const_cast<CUgraphNode>(n)] = nodes.size();
        nodes.push_back(*n);
        freeze(nodes.back(), memory);
    }
    if (g.memory) memory.push_back(g.memory);
    return CUDA_SUCCESS;
}

// Assigns the default values of the handles that ask for it, in stream order.
CUresult resetConditions(const Conditions& conditions) {
    for (const auto& c : conditions)
        if (c->assignDefault)
            if (CUresult r = cuMemsetD32_v2(c->device, c->defaultValue, 1)) return r;
    return CUDA_SUCCESS;
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

CUfunction functionOf(const CUDA_KERNEL_NODE_PARAMS_v1* p) { return p->func; }
CUfunction functionOf(const CUDA_KERNEL_NODE_PARAMS_v2* p) {
    return p->func ? p->func : reinterpret_cast<CUfunction>(p->kern);  // a CUkernel is its CUfunction
}

template <class P>
CUresult setKernelFrom(CUgraphNode_st& n, const P* p) {
    if (!p) return CUDA_ERROR_INVALID_VALUE;
    const unsigned grid[3] = {p->gridDimX, p->gridDimY, p->gridDimZ}, block[3] = {p->blockDimX, p->blockDimY, p->blockDimZ};
    return setKernel(n, functionOf(p), grid, block, p->sharedMemBytes, p->kernelParams, p->extra);
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

CUresult runNodes(ApiScope& s, CUstream stream, const std::vector<CUgraphNode_st>& nodes, const Conditions& conditions);

// A conditional node: the condition as the work before it left it.
CUresult runConditional(ApiScope& s, CUstream stream, const CUgraphNode_st& n) {
    auto value = [&](uint32_t& v) { return cuMemcpyDtoH_v2(&v, n.condition->device, 4); };  // waits for the GPU
    auto body = [&](size_t i) { return runNodes(s, stream, n.bodyNodes[i], n.bodyConditions[i]); };
    uint32_t v = 0;
    if (CUresult r = value(v)) return r;
    switch (n.condType) {
        case CU_GRAPH_COND_TYPE_IF:
            if (v) return body(0);
            return n.bodyNodes.size() > 1 ? body(1) : CUDA_SUCCESS;
        case CU_GRAPH_COND_TYPE_WHILE:
            while (v) {
                if (CUresult r = body(0)) return r;
                if (CUresult r = value(v)) return r;
            }
            return CUDA_SUCCESS;
        case CU_GRAPH_COND_TYPE_SWITCH:
            return v < n.bodyNodes.size() ? body(v) : CUDA_SUCCESS;
    }
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult runNode(ApiScope& s, CUstream stream, const CUgraphNode_st& n) {
    switch (n.type) {
        case CU_GRAPH_NODE_TYPE_GRAPH:
            return runNodes(s, stream, n.children, n.childConditions);
        case CU_GRAPH_NODE_TYPE_CONDITIONAL:
            return runConditional(s, stream, n);
        case CU_GRAPH_NODE_TYPE_KERNEL:
            return launchKernel(s, n.func, n.grid, n.block, n.sharedMem, n.params, n.cooperative);
        case CU_GRAPH_NODE_TYPE_MEMCPY:
            if (n.copy3D) return cuMemcpy3DAsync_v2(&n.copy, stream);
            return cuMemcpy(n.dst, n.src, n.bytes);
        case CU_GRAPH_NODE_TYPE_EVENT_RECORD:
            return cuEventRecord(n.event, stream);
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
        default:  // empty, event wait: ordering the single queue gives anyway; memory nodes allocated when
                  // they were created
            return CUDA_SUCCESS;
    }
}

CUresult runNodes(ApiScope& s, CUstream stream, const std::vector<CUgraphNode_st>& nodes, const Conditions& conditions) {
    if (CUresult r = resetConditions(conditions)) return r;
    for (const CUgraphNode_st& n : nodes)
        if (CUresult r = runNode(s, stream, n)) return r;
    return CUDA_SUCCESS;
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

// A child graph node of `child`'s nodes, memory and handles (a graph, or an executable graph's copy).
std::unique_ptr<CUgraphNode_st> childNode(const std::vector<CUgraphNode_st>& nodes,
                                          std::vector<std::shared_ptr<GraphMemory>> memory, Conditions conditions) {
    auto node = std::make_unique<CUgraphNode_st>();
    node->type = CU_GRAPH_NODE_TYPE_GRAPH;
    node->children = nodes;
    node->childMemory = std::move(memory);
    node->childConditions = std::move(conditions);
    return node;
}

CUresult setMemset(CUgraphNode_st& n, const CUDA_MEMSET_NODE_PARAMS* p) {
    if (!p) return CUDA_ERROR_INVALID_VALUE;
    const unsigned es = p->elementSize;
    if ((es != 1 && es != 2 && es != 4) || (p->height > 1 && p->pitch < p->width * es)) return CUDA_ERROR_INVALID_VALUE;
    n.type = CU_GRAPH_NODE_TYPE_MEMSET;
    n.memset = *p;
    return CUDA_SUCCESS;
}

CUresult setMemcpy(CUgraphNode_st& n, const CUDA_MEMCPY3D* p) {
    if (!p) return CUDA_ERROR_INVALID_VALUE;
    n.type = CU_GRAPH_NODE_TYPE_MEMCPY;
    n.copy3D = true;
    n.copy = *p;
    return CUDA_SUCCESS;
}

CUresult setHost(CUgraphNode_st& n, const CUDA_HOST_NODE_PARAMS* p) {
    if (!p || !p->fn) return CUDA_ERROR_INVALID_VALUE;
    n.type = CU_GRAPH_NODE_TYPE_HOST;
    n.fn = p->fn;
    n.userData = p->userData;
    return CUDA_SUCCESS;
}

// Changes a node's parameters to those of a node of the same type (cuGraphNodeSetParams).
CUresult setParams(CUgraphNode_st& n, const CUgraphNodeParams* p) {
    if (!p || p->type != n.type) return CUDA_ERROR_INVALID_VALUE;
    switch (p->type) {
        case CU_GRAPH_NODE_TYPE_KERNEL: return setKernelFrom(n, &p->kernel);
        case CU_GRAPH_NODE_TYPE_MEMCPY: return setMemcpy(n, &p->memcpy.copyParams);
        case CU_GRAPH_NODE_TYPE_MEMSET: return setMemset(n, reinterpret_cast<const CUDA_MEMSET_NODE_PARAMS*>(&p->memset));
        case CU_GRAPH_NODE_TYPE_HOST: return setHost(n, reinterpret_cast<const CUDA_HOST_NODE_PARAMS*>(&p->host));
        case CU_GRAPH_NODE_TYPE_EVENT_RECORD:
            if (!validHandle(p->eventRecord.event)) return CUDA_ERROR_INVALID_VALUE;
            n.event = p->eventRecord.event;
            return CUDA_SUCCESS;
        case CU_GRAPH_NODE_TYPE_WAIT_EVENT:
            if (!validHandle(p->eventWait.event)) return CUDA_ERROR_INVALID_VALUE;
            n.event = p->eventWait.event;
            return CUDA_SUCCESS;
        case CU_GRAPH_NODE_TYPE_EMPTY: return CUDA_SUCCESS;
        default: return CUDA_ERROR_NOT_SUPPORTED;
    }
}

// The node at hNode in an executable graph.
CUgraphNode_st* execNode(CUgraphExec e, CUgraphNode hNode) {
    if (!validHandle(e) || !validHandle(hNode)) return nullptr;
    auto it = e->index.find(hNode);
    return it == e->index.end() ? nullptr : &e->nodes[it->second];
}

// A copy of the graph (its body graphs copied too); the original nodes' copies in clonedFrom.
CUgraph cloneGraph(const CUgraph_st& g) {
    auto* c = new CUgraph_st();
    c->ctx = g.ctx;
    c->conditions = g.conditions;
    c->memory = g.memory;
    c->userObjects = g.userObjects;
    retainUserObjects(c->userObjects);
    Driver::get().registerHandle(c);
    for (const auto& n : g.nodes) {
        auto copy = std::make_unique<CUgraphNode_st>(*n);
        for (auto& body : copy->bodies) {
            CUgraph b = cloneGraph(*body);
            body = std::shared_ptr<CUgraph_st>(b, [](CUgraph_st* x) { destroyGraph(x); });
        }
        const std::vector<CUgraphNode> deps = copy->deps;  // the originals, mapped below
        c->clonedFrom[n.get()] = addNode(c, std::move(copy), deps.data(), deps.size());
    }
    for (const auto& n : c->nodes)
        for (CUgraphNode& d : n->deps) d = c->clonedFrom.at(d);
    return c;
}

}  // namespace

namespace vulkanizer::driver {

ConditionSlot::~ConditionSlot() {
    if (!device || cuCtxPushCurrent_v2(ctx) != CUDA_SUCCESS) return;
    cuMemFree_v2(device);  // after the work queued so far
    CUcontext popped;
    cuCtxPopCurrent_v2(&popped);
}

CUresult captureCopy3D(CUstream hStream, const CUDA_MEMCPY3D& copy) {
    auto node = std::make_unique<CUgraphNode_st>();
    if (CUresult r = setMemcpy(*node, &copy)) return r;
    return captureNode(hStream, std::move(node));
}

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
    g->fromCapture = true;
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
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!isRealStream(hStream) || !hStream->capture) return CUDA_ERROR_ILLEGAL_STATE;
    if (!hStream->captureOrigin) return CUDA_ERROR_STREAM_CAPTURE_UNMATCHED;
    CUgraph g = hStream->capture;
    if (!phGraph && g->fromCapture) return CUDA_ERROR_INVALID_VALUE;  // only a given graph needs no output
    const bool invalidated = hStream->captureStatus == CU_STREAM_CAPTURE_STATUS_INVALIDATED;
    endCaptureOf(g);
    if (invalidated) {
        if (g->fromCapture) destroyGraph(g);  // not one given to cuStreamBeginCaptureToGraph
        if (phGraph) *phGraph = nullptr;
        return CUDA_ERROR_STREAM_CAPTURE_INVALIDATED;
    }
    if (phGraph) *phGraph = g;
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
    auto node = std::make_unique<CUgraphNode_st>();
    if (CUresult r = setMemset(*node, memsetParams)) return r;
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
    if (CUresult r = setHost(*node, nodeParams)) return r;
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
    const auto order = ordered(*childGraph);
    if (order.size() != childGraph->nodes.size()) return CUDA_ERROR_INVALID_VALUE;  // a cycle
    std::vector<CUgraphNode_st> nodes;
    for (const CUgraphNode_st* n : order) nodes.push_back(*n);
    std::vector<std::shared_ptr<GraphMemory>> memory;
    if (childGraph->memory) memory.push_back(childGraph->memory);
    *phGraphNode = addNode(hGraph, childNode(nodes, std::move(memory), childGraph->conditions), dependencies,
                           numDependencies);
    return CUDA_SUCCESS;
}

// ---- executable graphs ------------------------------------------------------------------------

CUresult CUDAAPI cuGraphInstantiateWithFlags(CUgraphExec* phGraphExec, CUgraph hGraph, unsigned long long flags) {
    VULKANIZER_TRACE_API();
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!phGraphExec || !validHandle(hGraph)) return CUDA_ERROR_INVALID_VALUE;
    auto e = std::make_unique<CUgraphExec_st>();
    e->ctx = hGraph->ctx;
    if (CUresult r = instantiateNodes(*hGraph, e->nodes, e->index, e->memory)) return r;
    e->conditions = hGraph->conditions;
    e->userObjects = hGraph->userObjects;
    retainUserObjects(e->userObjects);
    Driver::get().registerHandle(e.get());
    *phGraphExec = e.release();
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
    std::vector<CUgraphNode_st> nodes;
    std::unordered_map<CUgraphNode, size_t> index;
    std::vector<std::shared_ptr<GraphMemory>> memory = hGraphExec->memory;  // the old nodes' stays too
    if (CUresult r = instantiateNodes(*hGraph, nodes, index, memory)) return r;
    if (nodes.size() != hGraphExec->nodes.size()) return fail(CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED, nullptr);
    for (size_t i = 0; i < nodes.size(); ++i)
        if (nodes[i].type != hGraphExec->nodes[i].type) {
            CUgraphNode which = nullptr;
            for (const auto& [n, at] : index)
                if (at == i) which = n;
            return fail(CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED, which);
        }
    hGraphExec->nodes = std::move(nodes);
    hGraphExec->index = std::move(index);
    hGraphExec->memory = std::move(memory);
    hGraphExec->conditions = hGraph->conditions;
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
        return captureNode(cs, childNode(hGraphExec->nodes, hGraphExec->memory, hGraphExec->conditions));
    return runNodes(s, hStream, hGraphExec->nodes, hGraphExec->conditions);
}
CUresult CUDAAPI cuGraphLaunch_ptsz(CUgraphExec hGraphExec, CUstream hStream) { return cuGraphLaunch(hGraphExec, hStream); }

CUresult CUDAAPI cuGraphExecDestroy(CUgraphExec hGraphExec) {
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!validHandle(hGraphExec)) return CUDA_ERROR_INVALID_VALUE;
    Driver::get().unregisterHandle(hGraphExec);
    releaseUserObjects(hGraphExec->userObjects);
    delete hGraphExec;
    return CUDA_SUCCESS;
}

// ---- memcpy, memset, host and event nodes -----------------------------------------------------

CUresult CUDAAPI cuGraphAddMemcpyNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies,
                                      size_t numDependencies, const CUDA_MEMCPY3D* copyParams, CUcontext) {
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!phGraphNode || !validHandle(hGraph) || !validDeps(dependencies, numDependencies, hGraph))
        return CUDA_ERROR_INVALID_VALUE;
    auto node = std::make_unique<CUgraphNode_st>();
    if (CUresult r = setMemcpy(*node, copyParams)) return r;
    *phGraphNode = addNode(hGraph, std::move(node), dependencies, numDependencies);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGraphMemcpyNodeGetParams(CUgraphNode hNode, CUDA_MEMCPY3D* nodeParams) {
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!validHandle(hNode) || !nodeParams || hNode->type != CU_GRAPH_NODE_TYPE_MEMCPY) return CUDA_ERROR_INVALID_VALUE;
    if (hNode->copy3D) {
        *nodeParams = hNode->copy;
        return CUDA_SUCCESS;
    }
    *nodeParams = CUDA_MEMCPY3D{};  // a captured linear copy
    nodeParams->srcMemoryType = nodeParams->dstMemoryType = CU_MEMORYTYPE_UNIFIED;
    nodeParams->srcDevice = hNode->src;
    nodeParams->dstDevice = hNode->dst;
    nodeParams->WidthInBytes = hNode->bytes;
    nodeParams->Height = nodeParams->Depth = 1;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGraphMemcpyNodeSetParams(CUgraphNode hNode, const CUDA_MEMCPY3D* nodeParams) {
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!validHandle(hNode) || hNode->type != CU_GRAPH_NODE_TYPE_MEMCPY) return CUDA_ERROR_INVALID_VALUE;
    return setMemcpy(*hNode, nodeParams);
}

CUresult CUDAAPI cuGraphMemsetNodeGetParams(CUgraphNode hNode, CUDA_MEMSET_NODE_PARAMS* nodeParams) {
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!validHandle(hNode) || !nodeParams || hNode->type != CU_GRAPH_NODE_TYPE_MEMSET) return CUDA_ERROR_INVALID_VALUE;
    *nodeParams = hNode->memset;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGraphMemsetNodeSetParams(CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS* nodeParams) {
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!validHandle(hNode) || hNode->type != CU_GRAPH_NODE_TYPE_MEMSET) return CUDA_ERROR_INVALID_VALUE;
    return setMemset(*hNode, nodeParams);
}

CUresult CUDAAPI cuGraphHostNodeGetParams(CUgraphNode hNode, CUDA_HOST_NODE_PARAMS* nodeParams) {
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!validHandle(hNode) || !nodeParams || hNode->type != CU_GRAPH_NODE_TYPE_HOST) return CUDA_ERROR_INVALID_VALUE;
    *nodeParams = {hNode->fn, hNode->userData};
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGraphHostNodeSetParams(CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS* nodeParams) {
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!validHandle(hNode) || hNode->type != CU_GRAPH_NODE_TYPE_HOST) return CUDA_ERROR_INVALID_VALUE;
    return setHost(*hNode, nodeParams);
}

static CUresult addEventNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies,
                             size_t numDependencies, CUevent event, CUgraphNodeType type) {
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!phGraphNode || !validHandle(hGraph) || !validHandle(event) || !validDeps(dependencies, numDependencies, hGraph))
        return CUDA_ERROR_INVALID_VALUE;
    auto node = std::make_unique<CUgraphNode_st>();
    node->type = type;
    node->event = event;
    *phGraphNode = addNode(hGraph, std::move(node), dependencies, numDependencies);
    return CUDA_SUCCESS;
}

static CUresult nodeEvent(CUgraphNode hNode, CUgraphNodeType type, CUevent* get, CUevent set) {
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!validHandle(hNode) || hNode->type != type) return CUDA_ERROR_INVALID_VALUE;
    if (get) {
        *get = hNode->event;
        return CUDA_SUCCESS;
    }
    if (!validHandle(set)) return CUDA_ERROR_INVALID_VALUE;
    hNode->event = set;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGraphAddEventRecordNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies,
                                           size_t numDependencies, CUevent event) {
    return addEventNode(phGraphNode, hGraph, dependencies, numDependencies, event, CU_GRAPH_NODE_TYPE_EVENT_RECORD);
}
CUresult CUDAAPI cuGraphAddEventWaitNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies,
                                         size_t numDependencies, CUevent event) {
    return addEventNode(phGraphNode, hGraph, dependencies, numDependencies, event, CU_GRAPH_NODE_TYPE_WAIT_EVENT);
}
CUresult CUDAAPI cuGraphEventRecordNodeGetEvent(CUgraphNode hNode, CUevent* event_out) {
    if (!event_out) return CUDA_ERROR_INVALID_VALUE;
    return nodeEvent(hNode, CU_GRAPH_NODE_TYPE_EVENT_RECORD, event_out, nullptr);
}
CUresult CUDAAPI cuGraphEventRecordNodeSetEvent(CUgraphNode hNode, CUevent event) {
    return nodeEvent(hNode, CU_GRAPH_NODE_TYPE_EVENT_RECORD, nullptr, event);
}
CUresult CUDAAPI cuGraphEventWaitNodeGetEvent(CUgraphNode hNode, CUevent* event_out) {
    if (!event_out) return CUDA_ERROR_INVALID_VALUE;
    return nodeEvent(hNode, CU_GRAPH_NODE_TYPE_WAIT_EVENT, event_out, nullptr);
}
CUresult CUDAAPI cuGraphEventWaitNodeSetEvent(CUgraphNode hNode, CUevent event) {
    return nodeEvent(hNode, CU_GRAPH_NODE_TYPE_WAIT_EVENT, nullptr, event);
}

// ---- conditional nodes, generic node parameters ------------------------------------------------

CUresult CUDAAPI cuGraphConditionalHandleCreate(CUgraphConditionalHandle* pHandle_out, CUgraph hGraph, CUcontext,
                                                unsigned int defaultLaunchValue, unsigned int flags) {
    VULKANIZER_TRACE_API();
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!pHandle_out || !validHandle(hGraph) || (flags & ~CU_GRAPH_COND_ASSIGN_DEFAULT)) return CUDA_ERROR_INVALID_VALUE;
    auto slot = std::make_shared<ConditionSlot>();
    slot->ctx = s.ctx;
    slot->defaultValue = defaultLaunchValue;
    slot->assignDefault = flags & CU_GRAPH_COND_ASSIGN_DEFAULT;
    if (CUresult r = cuMemAlloc_v2(&slot->device, 4)) return r;
    if (CUresult r = cuMemsetD32_v2(slot->device, defaultLaunchValue, 1)) return r;
    hGraph->conditions.push_back(slot);
    *pHandle_out = slot->device;
    return CUDA_SUCCESS;
}

static CUresult addConditionalNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies,
                                   size_t numDependencies, CUDA_CONDITIONAL_NODE_PARAMS& p) {
    std::shared_ptr<ConditionSlot> slot;
    for (const auto& c : hGraph->conditions)
        if (c->device == p.handle) slot = c;
    const bool sizeOk = p.type == CU_GRAPH_COND_TYPE_IF       ? p.size == 1 || p.size == 2
                        : p.type == CU_GRAPH_COND_TYPE_WHILE  ? p.size == 1
                        : p.type == CU_GRAPH_COND_TYPE_SWITCH ? p.size >= 1
                                                              : false;
    if (!slot || !sizeOk) return CUDA_ERROR_INVALID_VALUE;
    auto node = std::make_unique<CUgraphNode_st>();
    node->type = CU_GRAPH_NODE_TYPE_CONDITIONAL;
    node->condition = slot;
    node->condType = p.type;
    for (unsigned i = 0; i < p.size; ++i) {
        node->bodies.push_back(makeBody(hGraph->ctx));
        node->bodyHandles.push_back(node->bodies.back().get());
    }
    p.phGraph_out = node->bodyHandles.data();  // the node's array, as in CUDA
    *phGraphNode = addNode(hGraph, std::move(node), dependencies, numDependencies);
    return CUDA_SUCCESS;
}

// Edge data: only the default edges (full dependencies) are supported.
static bool defaultEdges(const CUgraphEdgeData* edgeData, size_t count) {
    for (size_t i = 0; edgeData && i < count; ++i)
        if (edgeData[i].type || edgeData[i].from_port || edgeData[i].to_port) return false;
    return true;
}

CUresult CUDAAPI cuGraphAddNode_v2(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies,
                                   const CUgraphEdgeData* dependencyData, size_t numDependencies,
                                   CUgraphNodeParams* nodeParams) {
    VULKANIZER_TRACE_API();
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!phGraphNode || !nodeParams || !validHandle(hGraph) || !validDeps(dependencies, numDependencies, hGraph))
        return CUDA_ERROR_INVALID_VALUE;
    if (!defaultEdges(dependencyData, numDependencies)) return CUDA_ERROR_NOT_SUPPORTED;
    CUgraphNodeParams& p = *nodeParams;
    const CUgraphNode* deps = dependencies;
    const size_t n = numDependencies;
    switch (p.type) {
        case CU_GRAPH_NODE_TYPE_KERNEL: return cuGraphAddKernelNode_v2(phGraphNode, hGraph, deps, n, &p.kernel);
        case CU_GRAPH_NODE_TYPE_MEMCPY:
            return cuGraphAddMemcpyNode(phGraphNode, hGraph, deps, n, &p.memcpy.copyParams, p.memcpy.copyCtx);
        case CU_GRAPH_NODE_TYPE_MEMSET:
            return cuGraphAddMemsetNode(phGraphNode, hGraph, deps, n,
                                        reinterpret_cast<const CUDA_MEMSET_NODE_PARAMS*>(&p.memset), p.memset.ctx);
        case CU_GRAPH_NODE_TYPE_HOST:
            return cuGraphAddHostNode(phGraphNode, hGraph, deps, n, reinterpret_cast<const CUDA_HOST_NODE_PARAMS*>(&p.host));
        case CU_GRAPH_NODE_TYPE_GRAPH: {
            if (CUresult r = cuGraphAddChildGraphNode(phGraphNode, hGraph, deps, n, p.graph.graph)) return r;
            if (p.graph.ownership == CU_GRAPH_CHILD_GRAPH_OWNERSHIP_MOVE) destroyGraph(p.graph.graph);
            return CUDA_SUCCESS;
        }
        case CU_GRAPH_NODE_TYPE_EMPTY: return cuGraphAddEmptyNode(phGraphNode, hGraph, deps, n);
        case CU_GRAPH_NODE_TYPE_EVENT_RECORD: return cuGraphAddEventRecordNode(phGraphNode, hGraph, deps, n, p.eventRecord.event);
        case CU_GRAPH_NODE_TYPE_WAIT_EVENT: return cuGraphAddEventWaitNode(phGraphNode, hGraph, deps, n, p.eventWait.event);
        case CU_GRAPH_NODE_TYPE_MEM_ALLOC: return cuGraphAddMemAllocNode(phGraphNode, hGraph, deps, n, &p.alloc);
        case CU_GRAPH_NODE_TYPE_MEM_FREE: return cuGraphAddMemFreeNode(phGraphNode, hGraph, deps, n, p.free.dptr);
        case CU_GRAPH_NODE_TYPE_CONDITIONAL: return addConditionalNode(phGraphNode, hGraph, deps, n, p.conditional);
        default: return CUDA_ERROR_NOT_SUPPORTED;  // external semaphores, batched memory operations
    }
}
CUresult CUDAAPI cuGraphAddNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies,
                                size_t numDependencies, CUgraphNodeParams* nodeParams);
CUresult CUDAAPI cuGraphAddNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies,
                                size_t numDependencies, CUgraphNodeParams* nodeParams) {
    return cuGraphAddNode_v2(phGraphNode, hGraph, dependencies, nullptr, numDependencies, nodeParams);
}

CUresult CUDAAPI cuGraphNodeSetParams(CUgraphNode hNode, CUgraphNodeParams* nodeParams) {
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!validHandle(hNode)) return CUDA_ERROR_INVALID_VALUE;
    return setParams(*hNode, nodeParams);
}

CUresult CUDAAPI cuGraphExecNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, CUgraphNodeParams* nodeParams) {
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    CUgraphNode_st* n = execNode(hGraphExec, hNode);
    if (!n) return CUDA_ERROR_INVALID_VALUE;
    return setParams(*n, nodeParams);
}

CUresult CUDAAPI cuStreamBeginCaptureToGraph(CUstream hStream, CUgraph hGraph, const CUgraphNode* dependencies,
                                             const CUgraphEdgeData* dependencyData, size_t numDependencies,
                                             CUstreamCaptureMode mode) {
    VULKANIZER_TRACE_API();
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!isRealStream(hStream)) return validStream(hStream) ? CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED : CUDA_ERROR_INVALID_HANDLE;
    if (hStream->capture) return CUDA_ERROR_ILLEGAL_STATE;
    if (mode > CU_STREAM_CAPTURE_MODE_RELAXED || !validHandle(hGraph) || !hGraph->capturing.empty() ||
        !validDeps(dependencies, numDependencies, hGraph))
        return CUDA_ERROR_INVALID_VALUE;
    if (!defaultEdges(dependencyData, numDependencies)) return CUDA_ERROR_NOT_SUPPORTED;
    if (!hGraph->captureId) hGraph->captureId = g_nextCaptureId++;
    hGraph->capturing.push_back(hStream);
    hStream->capture = hGraph;
    hStream->captureStatus = CU_STREAM_CAPTURE_STATUS_ACTIVE;
    hStream->captureMode = mode;
    hStream->captureOrigin = true;
    hStream->captureDeps.assign(dependencies, dependencies + numDependencies);
    return CUDA_SUCCESS;
}
CUresult CUDAAPI cuStreamBeginCaptureToGraph_ptsz(CUstream hStream, CUgraph hGraph, const CUgraphNode* dependencies,
                                                  const CUgraphEdgeData* dependencyData, size_t numDependencies,
                                                  CUstreamCaptureMode mode) {
    return cuStreamBeginCaptureToGraph(hStream, hGraph, dependencies, dependencyData, numDependencies, mode);
}

// ---- edges, cloning ---------------------------------------------------------------------------

CUresult CUDAAPI cuGraphGetRootNodes(CUgraph hGraph, CUgraphNode* rootNodes, size_t* numRootNodes) {
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!validHandle(hGraph) || !numRootNodes) return CUDA_ERROR_INVALID_VALUE;
    size_t count = 0;
    for (const auto& n : hGraph->nodes)
        if (n->deps.empty()) {
            if (rootNodes && count < *numRootNodes) rootNodes[count] = n.get();
            ++count;
        }
    *numRootNodes = count;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGraphGetEdges_v2(CUgraph hGraph, CUgraphNode* from, CUgraphNode* to, CUgraphEdgeData* edgeData,
                                    size_t* numEdges) {
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!validHandle(hGraph) || !numEdges) return CUDA_ERROR_INVALID_VALUE;
    size_t count = 0;
    for (const auto& n : hGraph->nodes)
        for (CUgraphNode d : n->deps) {
            if (count < *numEdges) {
                if (from) from[count] = d;
                if (to) to[count] = n.get();
                if (edgeData) edgeData[count] = CUgraphEdgeData{};
            }
            ++count;
        }
    *numEdges = count;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI cuGraphGetEdges(CUgraph hGraph, CUgraphNode* from, CUgraphNode* to, size_t* numEdges) {
    return cuGraphGetEdges_v2(hGraph, from, to, nullptr, numEdges);
}

static bool inGraph(CUgraphNode n, CUgraph g) { return validHandle(n) && n->graph == g; }

CUresult CUDAAPI cuGraphAddDependencies_v2(CUgraph hGraph, const CUgraphNode* from, const CUgraphNode* to,
                                           const CUgraphEdgeData* edgeData, size_t numDependencies) {
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!validHandle(hGraph) || (numDependencies && (!from || !to))) return CUDA_ERROR_INVALID_VALUE;
    if (!defaultEdges(edgeData, numDependencies)) return CUDA_ERROR_NOT_SUPPORTED;
    for (size_t i = 0; i < numDependencies; ++i)
        if (!inGraph(from[i], hGraph) || !inGraph(to[i], hGraph) || from[i] == to[i]) return CUDA_ERROR_INVALID_VALUE;
    for (size_t i = 0; i < numDependencies; ++i) {
        auto& deps = to[i]->deps;
        if (std::find(deps.begin(), deps.end(), from[i]) == deps.end()) deps.push_back(from[i]);
    }
    return CUDA_SUCCESS;
}
CUresult CUDAAPI cuGraphAddDependencies(CUgraph hGraph, const CUgraphNode* from, const CUgraphNode* to,
                                        size_t numDependencies) {
    return cuGraphAddDependencies_v2(hGraph, from, to, nullptr, numDependencies);
}

CUresult CUDAAPI cuGraphRemoveDependencies_v2(CUgraph hGraph, const CUgraphNode* from, const CUgraphNode* to,
                                              const CUgraphEdgeData*, size_t numDependencies) {
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!validHandle(hGraph) || (numDependencies && (!from || !to))) return CUDA_ERROR_INVALID_VALUE;
    for (size_t i = 0; i < numDependencies; ++i) {
        if (!inGraph(from[i], hGraph) || !inGraph(to[i], hGraph)) return CUDA_ERROR_INVALID_VALUE;
        auto& deps = to[i]->deps;
        auto it = std::find(deps.begin(), deps.end(), from[i]);
        if (it == deps.end()) return CUDA_ERROR_INVALID_VALUE;
        deps.erase(it);
    }
    return CUDA_SUCCESS;
}
CUresult CUDAAPI cuGraphRemoveDependencies(CUgraph hGraph, const CUgraphNode* from, const CUgraphNode* to,
                                           size_t numDependencies) {
    return cuGraphRemoveDependencies_v2(hGraph, from, to, nullptr, numDependencies);
}

CUresult CUDAAPI cuGraphNodeGetDependentNodes_v2(CUgraphNode hNode, CUgraphNode* dependentNodes,
                                                 CUgraphEdgeData* edgeData, size_t* numDependentNodes) {
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!validHandle(hNode) || !numDependentNodes) return CUDA_ERROR_INVALID_VALUE;
    size_t count = 0;
    for (const auto& n : hNode->graph->nodes)
        if (std::find(n->deps.begin(), n->deps.end(), hNode) != n->deps.end()) {
            if (count < *numDependentNodes) {
                if (dependentNodes) dependentNodes[count] = n.get();
                if (edgeData) edgeData[count] = CUgraphEdgeData{};
            }
            ++count;
        }
    *numDependentNodes = count;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI cuGraphNodeGetDependentNodes(CUgraphNode hNode, CUgraphNode* dependentNodes, size_t* numDependentNodes) {
    return cuGraphNodeGetDependentNodes_v2(hNode, dependentNodes, nullptr, numDependentNodes);
}

CUresult CUDAAPI cuGraphDestroyNode(CUgraphNode hNode) {
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!validHandle(hNode)) return CUDA_ERROR_INVALID_VALUE;
    CUgraph g = hNode->graph;
    for (auto& n : g->nodes) n->deps.erase(std::remove(n->deps.begin(), n->deps.end(), hNode), n->deps.end());
    for (auto& [orig, copy] : g->clonedFrom)
        if (copy == hNode) copy = nullptr;
    Driver::get().unregisterHandle(hNode);
    g->nodes.erase(std::find_if(g->nodes.begin(), g->nodes.end(), [&](const auto& n) { return n.get() == hNode; }));
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGraphClone(CUgraph* phGraphClone, CUgraph originalGraph) {
    VULKANIZER_TRACE_API();
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!phGraphClone || !validHandle(originalGraph)) return CUDA_ERROR_INVALID_VALUE;
    *phGraphClone = cloneGraph(*originalGraph);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGraphNodeFindInClone(CUgraphNode* phNode, CUgraphNode hOriginalNode, CUgraph hClonedGraph) {
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!phNode || !validHandle(hClonedGraph)) return CUDA_ERROR_INVALID_VALUE;
    auto it = hClonedGraph->clonedFrom.find(hOriginalNode);
    if (it == hClonedGraph->clonedFrom.end() || !it->second) return CUDA_ERROR_INVALID_VALUE;
    *phNode = it->second;
    return CUDA_SUCCESS;
}

// ---- user objects -----------------------------------------------------------------------------

CUresult CUDAAPI cuUserObjectCreate(CUuserObject* object_out, void* ptr, CUhostFn destroy, unsigned int initialRefcount,
                                    unsigned int flags) {
    if (!object_out || !destroy || !initialRefcount || flags != CU_USER_OBJECT_NO_DESTRUCTOR_SYNC)
        return CUDA_ERROR_INVALID_VALUE;
    auto* o = new CUuserObject_st();
    o->ptr = ptr;
    o->destroy = destroy;
    o->refs = initialRefcount;
    Driver::get().registerHandle(o);
    *object_out = o;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuUserObjectRetain(CUuserObject object, unsigned int count) {
    if (!validHandle(object) || !count) return CUDA_ERROR_INVALID_VALUE;
    object->refs += count;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuUserObjectRelease(CUuserObject object, unsigned int count) {
    if (!validHandle(object) || !count || count > object->refs) return CUDA_ERROR_INVALID_VALUE;
    releaseUserObject(object, count);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGraphRetainUserObject(CUgraph graph, CUuserObject object, unsigned int count, unsigned int flags) {
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!validHandle(graph) || !validHandle(object) || !count || (flags & ~CU_GRAPH_USER_OBJECT_MOVE))
        return CUDA_ERROR_INVALID_VALUE;
    if (!(flags & CU_GRAPH_USER_OBJECT_MOVE)) object->refs += count;  // else the caller's references move
    graph->userObjects[object] += count;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGraphReleaseUserObject(CUgraph graph, CUuserObject object, unsigned int count) {
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!validHandle(graph) || !validHandle(object) || !count) return CUDA_ERROR_INVALID_VALUE;
    auto it = graph->userObjects.find(object);
    if (it == graph->userObjects.end() || it->second < count) return CUDA_ERROR_INVALID_VALUE;
    if ((it->second -= count) == 0) graph->userObjects.erase(it);
    releaseUserObject(object, count);
    return CUDA_SUCCESS;
}

}  // extern "C"
