#pragma once

#include <cuda.h>

#include "compiler/ptx_to_spirv.h"
#include "ptx/parser.h"
#include "vk/device.h"
#include "vk/instance.h"

#include <array>
#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Handle objects behind the opaque CUDA handle types. Each carries a magic value
// and is registered in Driver's live-handle set so stale handles are rejected.

struct CUctx_st {
    static constexpr uint32_t kMagic = 0x58544353;  // 'SCTX'
    uint32_t magic = kMagic;
    CUdevice device = 0;
    unsigned int flags = 0;
    bool primary = false;
};

struct CUfunc_st {
    static constexpr uint32_t kMagic = 0x434e5553;  // 'SUNC'
    uint32_t magic = kMagic;
    CUmod_st* module = nullptr;
    const vulkanizer::ptx::Entry* entry = nullptr;
    bool compileAttempted = false;
    bool compiled = false;
    std::string compileError;
    vulkanizer::compiler::KernelBinary binary;
    // keyed by the specialization: blockDim.x/y/z, dynamic shared memory words
    std::map<std::array<uint32_t, 5>, vulkanizer::vk::ComputePipeline> pipelines;  // by spec constants
};

struct CUmod_st {
    static constexpr uint32_t kMagic = 0x444f4d53;  // 'SMOD'
    uint32_t magic = kMagic;
    CUcontext ctx = nullptr;
    std::string source;       // PTX text; `ptx` holds views into it, so it must never be moved
    vulkanizer::ptx::Module ptx;
    CUdeviceptr globals = 0;  // buffer of the module's .global/.const variables (0 if none)
    std::vector<uint8_t> globalsImage;  // their initial contents (printf format strings)
    std::string sourceHash;             // SHA-256 of `source`, for the kernel cache (computed lazily)
    std::vector<std::unique_ptr<CUfunc_st>> functions;
};

struct CUstream_st {
    static constexpr uint32_t kMagic = 0x52545353;  // 'SSTR'
    uint32_t magic = kMagic;
    CUcontext ctx = nullptr;
    unsigned int flags = 0;
    // Stream capture (api_graph.cpp): work goes into `capture` instead of running. A stream joins
    // another one's capture by waiting on an event recorded there.
    CUgraph capture = nullptr;
    CUstreamCaptureStatus captureStatus = CU_STREAM_CAPTURE_STATUS_NONE;
    CUstreamCaptureMode captureMode = CU_STREAM_CAPTURE_MODE_GLOBAL;
    bool captureOrigin = false;               // the stream that began the capture
    std::vector<CUgraphNode> captureDeps;     // what the next captured node depends on
};

struct CUevent_st {
    static constexpr uint32_t kMagic = 0x54564553;  // 'SEVT'
    uint32_t magic = kMagic;
    CUcontext ctx = nullptr;
    unsigned int flags = 0;
    bool recorded = false;
    uint64_t serial = 0;      // device work the event completes with
    int32_t timestamp = -1;   // GPU timestamp slot, -1 without timing
    CUgraph captureGraph = nullptr;           // recorded on a capturing stream (fork/join point)
    std::vector<CUgraphNode> captureDeps;
};

namespace vulkanizer::driver {
// Graph memory (memory nodes: cuMemAllocAsync while capturing, cuGraphAddMemAllocNode): allocated
// when the node is created, at a fixed address, and owned by the graph, its executable graphs and
// the graphs containing it as a child, together. Freeing it while they exist (a free node, or a
// free outside the graph) only marks it released: a graph is relaunchable any time. Once the last
// of them is destroyed, released memory is freed (after the work recorded so far); the rest stays
// allocated until the application frees it, as in CUDA.
struct GraphAllocation {
    bool owned = true;      // a graph owning it still exists
    bool released = false;  // freed by the graph or the application
};
struct GraphMemory {
    CUdevice device = 0;
    std::vector<CUdeviceptr> allocations;
    ~GraphMemory();
};

// A conditional node's handle (CUgraphConditionalHandle is `device`): a 32-bit value in device
// memory that kernels set with cudaGraphSetConditional (a store, ptx::defineDeviceRuntimeBuiltins).
// With CU_GRAPH_COND_ASSIGN_DEFAULT it is reset to defaultValue whenever its graph starts.
struct ConditionSlot {
    CUcontext ctx = nullptr;
    CUdeviceptr device = 0;
    unsigned int defaultValue = 0;
    bool assignDefault = false;
    ~ConditionSlot();
};
using Conditions = std::vector<std::shared_ptr<ConditionSlot>>;
}  // namespace vulkanizer::driver

// A user object (cuUserObjectCreate): its destructor runs when the last reference is released.
struct CUuserObject_st {
    static constexpr uint32_t kMagic = 0x4a425553;  // 'SUBJ'
    uint32_t magic = kMagic;
    void* ptr = nullptr;
    CUhostFn destroy = nullptr;
    std::atomic<unsigned int> refs{0};
};

// A node of a CUDA graph. Kernel nodes keep their parameters packed as cuLaunchKernel would.
struct CUgraphNode_st {
    static constexpr uint32_t kMagic = 0x444e4753;  // 'SGND'
    uint32_t magic = kMagic;
    CUgraph graph = nullptr;
    CUgraphNodeType type = CU_GRAPH_NODE_TYPE_EMPTY;
    std::vector<CUgraphNode> deps;
    // kernel
    CUfunction func = nullptr;
    unsigned int grid[3] = {1, 1, 1}, block[3] = {1, 1, 1}, sharedMem = 0;
    std::vector<uint8_t> params;
    std::vector<void*> paramPointers;  // for cuGraphKernelNodeGetParams: into `params`
    bool cooperative = false;          // grid-wide synchronization (cuLaunchCooperativeKernel)
    // memcpy (device or host addresses, as cuMemcpy takes them)
    CUdeviceptr dst = 0, src = 0;
    size_t bytes = 0;
    // memset
    CUDA_MEMSET_NODE_PARAMS memset{};
    // host
    CUhostFn fn = nullptr;
    void* userData = nullptr;
    // child graph: its nodes in order, its memory and conditional handles; memory nodes: dst and bytes
    std::vector<CUgraphNode_st> children;
    std::vector<std::shared_ptr<vulkanizer::driver::GraphMemory>> childMemory;
    vulkanizer::driver::Conditions childConditions;
    // memcpy given as CUDA_MEMCPY3D (cuGraphAddMemcpyNode, 2D/3D copies), instead of dst/src/bytes
    bool copy3D = false;
    CUDA_MEMCPY3D copy{};
    // event record/wait
    CUevent event = nullptr;
    // conditional: the handle, the kind and the body graphs (owned by the node and its copies); in an
    // executable graph also each body's nodes in order and its handles (frozen when instantiated)
    std::shared_ptr<vulkanizer::driver::ConditionSlot> condition;
    CUgraphConditionalNodeType condType = CU_GRAPH_COND_TYPE_IF;
    std::vector<std::shared_ptr<CUgraph_st>> bodies;
    std::vector<CUgraph> bodyHandles;  // the array cuGraphAddNode hands out (phGraph_out)
    std::vector<std::vector<CUgraphNode_st>> bodyNodes;
    std::vector<vulkanizer::driver::Conditions> bodyConditions;
};

struct CUgraph_st {
    static constexpr uint32_t kMagic = 0x50524753;  // 'SGRP'
    uint32_t magic = kMagic;
    CUcontext ctx = nullptr;
    std::vector<std::unique_ptr<CUgraphNode_st>> nodes;  // in an order that respects the dependencies
    std::vector<CUstream> capturing;                       // streams capturing into this graph
    uint64_t captureId = 0;                                // cuStreamGetCaptureInfo's id
    std::shared_ptr<vulkanizer::driver::GraphMemory> memory;  // of its memory nodes (created with the first)
    bool fromCapture = false;                                 // made by cuStreamBeginCapture
    vulkanizer::driver::Conditions conditions;                // handles created for it
    std::map<CUuserObject, unsigned int> userObjects;         // references it holds
    std::unordered_map<CUgraphNode, CUgraphNode> clonedFrom;  // a clone: original node -> its copy
};

// An instantiated graph: its own copy of the nodes, run in order on the launch stream.
struct CUgraphExec_st {
    static constexpr uint32_t kMagic = 0x58454753;  // 'SGEX'
    uint32_t magic = kMagic;
    CUcontext ctx = nullptr;
    std::vector<CUgraphNode_st> nodes;
    std::unordered_map<CUgraphNode, size_t> index;  // graph node -> position in `nodes`
    std::vector<std::shared_ptr<vulkanizer::driver::GraphMemory>> memory;  // of the graphs it was made from
    vulkanizer::driver::Conditions conditions;                              // the graph's handles
    std::map<CUuserObject, unsigned int> userObjects;                       // references it holds
};

// A memory pool (stream-ordered allocation). Allocations come straight from the device; the pool
// keeps the statistics and attributes applications read back.
struct CUmemPoolHandle_st {
    static constexpr uint32_t kMagic = 0x4c4f5053;  // 'SPOL'
    uint32_t magic = kMagic;
    CUdevice device = 0;
    CUmemPoolProps props{};
    uint64_t releaseThreshold = 0;
    int reuseFlags[3] = {1, 1, 1};  // CU_MEMPOOL_ATTR_REUSE_*
    uint64_t usedBytes = 0, usedHigh = 0;
};

// Physical memory of the virtual memory API (cuMemCreate); the CUmemGenericAllocationHandle
// points to it. Freed when it is released and no longer mapped.
struct CUmemAllocation_st {
    static constexpr uint32_t kMagic = 0x4d454d53;  // 'SMEM'
    uint32_t magic = kMagic;
    CUdevice device = 0;
    CUmemAllocationProp prop{};
    VkDeviceMemory memory = VK_NULL_HANDLE;
    size_t size = 0;
    int refs = 1;  // the handle itself plus one per mapping
};

namespace vulkanizer::driver {

// An address range of cuMemAddressReserve: sparse buffers of `chunkSize` bytes side by side
// (one buffer is limited to maxBufferSize, 4 GB on RADV).
struct VmReservation {
    CUdeviceptr base = 0;
    size_t size = 0;
    VkDeviceSize chunkSize = 0;
    std::vector<vk::Buffer> chunks;
};

struct VmMapping {
    size_t size = 0;
    CUmemAllocation_st* allocation = nullptr;
    size_t offset = 0;  // into the allocation
};

struct DeviceSlot {
    CUdevice ordinal = 0;
    const vk::PhysicalDeviceInfo* info = nullptr;
    std::unique_ptr<vk::Device> vk;  // created with the first context that needs it
    CUctx_st primary;
    int primaryRefs = 0;
    std::map<CUdeviceptr, vk::Buffer> allocations;  // keyed by base address
    size_t allocatedBytes = 0;
    std::map<CUdeviceptr, CUmemPoolHandle_st*> poolAllocations;  // stream-ordered ones, by pool
    CUmemPoolHandle_st defaultPool;
    CUmemPoolHandle_st* currentPool = &defaultPool;
    std::map<CUdeviceptr, VmReservation> reservations;  // keyed by base address
    std::map<CUdeviceptr, VmMapping> mappings;          // keyed by mapped address
    // Device memory of other processes (cuIpcOpenMemHandle), by (exporter pid, memory id): the base
    // of the imported buffer (in `allocations`) and the number of opens.
    struct IpcImport {
        CUdeviceptr base = 0;
        int refs = 0;
    };
    std::map<std::pair<int32_t, uint64_t>, IpcImport> ipcImports;
    std::unordered_map<CUdeviceptr, GraphAllocation> graphAllocations;  // graph memory (GraphMemory), in `allocations`
    vk::Buffer printfBuffer;  // device printf records of the current launch
    // Device-side assert(): the record the first failing thread fills in (KernelBinary::kAssertRecordBytes),
    // checked at synchronization once a kernel with an assert ran (checkDeviceAssert). An assert is
    // sticky like in CUDA: every later launch and synchronization fails until the context is reset.
    vk::Buffer assertRecord;
    bool assertArmed = false;
    CUresult stickyError = CUDA_SUCCESS;
    vk::Buffer gridWorkspace;  // cooperative launches' {size, barrier} (cooperative groups' grid.sync())
};

class Driver {
public:
    // Never destroyed (see vk::Instance).
    static Driver& get();

    CUresult init();
    bool initialized() const { return initialized_.load(std::memory_order_acquire); }

    int deviceCount() const { return static_cast<int>(slots_.size()); }
    DeviceSlot* slot(CUdevice dev) {
        return dev >= 0 && dev < deviceCount() ? slots_[static_cast<size_t>(dev)].get() : nullptr;
    }
    // Lazily creates the Vulkan logical device. Caller holds mutex().
    CUresult ensureDevice(DeviceSlot& slot);

    std::recursive_mutex& mutex() { return mutex_; }

    void registerHandle(const void* h);
    void unregisterHandle(const void* h);
    bool isLive(const void* h);

private:
    Driver() = default;

    std::recursive_mutex mutex_;
    std::atomic<bool> initialized_{false};
    CUresult initResult_ = CUDA_SUCCESS;
    bool initAttempted_ = false;
    std::vector<std::unique_ptr<DeviceSlot>> slots_;
    std::unordered_set<const void*> liveHandles_;
};

template <class T>
bool validHandle(T* h) {
    return h && Driver::get().isLive(h) && h->magic == T::kMagic;
}

// Per-thread context stack. Fixed capacity and trivially destructible, so it stays valid
// for atexit handlers (e.g. libcudart unregistering its modules) that run after the
// thread_local destructors of the main thread.
class ContextStack {
public:
    static constexpr size_t kCapacity = 64;
    bool empty() const { return size_ == 0; }
    CUcontext& back() { return items_[size_ - 1]; }
    bool push(CUcontext c) {
        if (size_ == kCapacity) return false;
        items_[size_++] = c;
        return true;
    }
    void pop() { --size_; }
    void remove(CUcontext c) {
        size_t n = 0;
        for (size_t i = 0; i < size_; ++i)
            if (items_[i] != c) items_[n++] = items_[i];
        size_ = n;
    }

private:
    CUcontext items_[kCapacity];
    size_t size_;
};

CUcontext currentContext();
ContextStack& contextStack();

// Common prologue for calls that need a current context: checks initialization,
// takes the driver lock and makes sure the Vulkan device exists.
struct ApiScope {
    std::unique_lock<std::recursive_mutex> lock;
    CUcontext ctx = nullptr;
    DeviceSlot* slot = nullptr;
    vk::Device* dev = nullptr;
};
CUresult enterContext(ApiScope& scope);

// The default (0), legacy and per-thread streams, or a live stream of the context.
bool validStream(CUstream stream);

struct AllocRef {
    vk::Buffer* buffer = nullptr;
    CUdeviceptr base = 0;
    VkDeviceSize offset = 0;
};
// Finds the allocation containing [ptr, ptr + bytes). Caller holds the driver lock.
bool findAllocation(DeviceSlot& slot, CUdeviceptr ptr, size_t bytes, AllocRef& out);

// [ptr, ptr + bytes) as consecutive buffer ranges: a virtual memory reservation consists of
// several buffers. `at` is the piece's offset within the range.
struct AllocPiece {
    AllocRef ref;
    size_t at = 0, size = 0;
};
bool findPieces(DeviceSlot& slot, CUdeviceptr ptr, size_t bytes, std::vector<AllocPiece>& out);
// Unified addressing over the devices: the device whose memory holds [ptr, ptr + bytes) (`preferred`
// first, as separate Vulkan devices' address ranges may overlap), or nullptr. Caller holds the driver lock.
DeviceSlot* findPiecesOnAnyDevice(DeviceSlot& preferred, CUdeviceptr ptr, size_t bytes, std::vector<AllocPiece>& out);

// The reservation containing ptr, or nullptr.
VmReservation* findReservation(DeviceSlot& slot, CUdeviceptr ptr);

CUresult toCuResult(VkResult r);

// Kernel launches (api_module.cpp): the parameter buffer as cuLaunchKernel builds it, and the
// launch of packed parameters (cuLaunchKernel, graph replay). Caller holds the driver lock.
CUresult packParams(const ptx::Entry& e, void** kernelParams, void** extra, std::vector<uint8_t>& out);
CUresult launchKernel(ApiScope& s, CUfunction f, const unsigned grid[3], const unsigned block[3],
                      unsigned sharedMemBytes, const std::vector<uint8_t>& params, bool cooperative = false);

// CUDA_ERROR_ASSERT (printing the assert's message once) when a kernel's assert() failed, else
// CUDA_SUCCESS. `waited`: the device finished all recorded work, so an unmapped record may be read
// too (it is read in queue order). Caller holds the driver lock.
CUresult checkDeviceAssert(DeviceSlot& slot, bool waited);
// Clears the assert state of a device whose primary context is reset.
void resetDeviceAssert(DeviceSlot& slot);

// cuGetExportTable's private interfaces (export_tables.cpp, VULKANIZER_EXPORT_TABLES): nullptr when
// the interface is not provided.
const void* exportTable(const CUuuid& id);

// Stream capture (api_graph.cpp): `hStream` if it is capturing, else nullptr; captureNode appends
// a node after the stream's current dependencies (invalidated captures report the error).
CUstream capturingStream(CUstream hStream);
CUresult captureNode(CUstream hStream, std::unique_ptr<CUgraphNode_st> node);
// A 2D/3D copy on a capturing stream: a memcpy node.
CUresult captureCopy3D(CUstream hStream, const CUDA_MEMCPY3D& copy);
// Synchronizing a capturing stream is illegal: it ends the capture as invalidated.
CUresult refuseIfCapturing(CUstream hStream);
// Stream-ordered allocation and free on a capturing stream (`cs`): memory nodes (graph memory).
CUresult captureMemAlloc(ApiScope& s, CUstream cs, size_t bytes, CUdeviceptr& out);
CUresult captureMemFree(CUstream cs, CUdeviceptr dptr);
// A free of graph memory outside a graph (cuMemFree, cuMemFreeAsync): true if it only marks it
// released (its graph still exists); false if the memory is to be freed now (no graph owns it).
bool releaseGraphMemory(DeviceSlot& slot, CUdeviceptr dptr);
// cuStreamWaitEvent on an event recorded during a capture: `hStream` joins that capture.
CUresult joinCapture(CUstream hStream, CUevent hEvent);

// Pinned, registered and managed host memory (api_host.cpp). Caller holds the driver lock.
struct HostMemoryInfo {
    uintptr_t base = 0;
    size_t size = 0;
    bool managed = false;
    bool deviceAccessible = false;  // imported: the device uses it at the same address
};
bool findHostMemory(CUdeviceptr ptr, HostMemoryInfo& info);
// cuMemFree of cuMemAllocManaged memory; CUDA_ERROR_NOT_FOUND for any other pointer.
CUresult freeManaged(ApiScope& s, CUdeviceptr ptr);

// Compute capability reported to applications (default 7.5; VULKANIZER_CC=86 overrides).
void computeCapability(int& major, int& minor);

}  // namespace vulkanizer::driver
