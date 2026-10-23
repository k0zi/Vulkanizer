#pragma once

#include "common/device_address.h"
#include "vk/instance.h"

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace vulkanizer::vk {

using common::kAddressMask;  // device address conventions: common/device_address.h

// A device allocation addressable from shaders through its buffer device address.
struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    VkDeviceSize offset = 0;  // where this allocation starts in `buffer` (sub-allocations)
    VkDeviceAddress address = 0;
    void* mapped = nullptr;  // set when the memory is host-visible and coherent (at `offset`)
    bool subAllocated = false;  // part of a shared block (allocateBuffer): never destroyed on its own
    bool hostCached = false;  // host reads of `mapped` are cached (otherwise write-combined: very slow)
    bool hostAddressed = false;  // host memory at its own address (address == mapped)
    bool placedMapping = false;  // our memory, mapped at a chosen address (allocateHostMemory)
};

struct ComputePipeline {
    VkShaderModule module = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    uint32_t pushConstantSize = 0;
};

// Optional capabilities that were enabled on the logical device.
struct Features {
    bool shaderInt8 = false;
    bool shaderInt16 = false;
    bool shaderFloat16 = false;
    bool shaderFloat64 = false;
    bool storageBuffer8Bit = false;
    bool storageBuffer16Bit = false;
    bool bufferInt64Atomics = false;
    bool sharedInt64Atomics = false;
    bool vulkanMemoryModel = false;
    bool subgroupSizeControl = false;
    bool computeFullSubgroups = false;
    bool timelineSemaphore = false;
    bool integerDotProduct = false;  // SPV_KHR_integer_dot_product (Vulkan 1.3)
    bool fusedFma = false;           // fma() is evaluated with a single rounding (probed)
    bool fusedFma64 = false;         // the same for f64 fma()
    bool sparseBuffers = false;      // sparse residency buffers on the compute queue (CUDA VMM)
    bool addressCaptureReplay = false;  // buffers can be placed at a requested device address
    bool hostMemoryImport = false;  // host memory at device address h | hostAddressBits() (importHostMemory)
    bool hostMemoryAlloc = false;   // the same for memory of our own (allocateHostMemory)
};

// One page range of a sparse buffer bound to `memory` (or unbound if VK_NULL_HANDLE).
struct SparseBind {
    const Buffer* buffer = nullptr;
    VkDeviceSize offset = 0, size = 0;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize memoryOffset = 0;
};

// Logical device with a single compute queue. Kernel dispatches are recorded into batches
// (one command buffer each, a memory barrier between commands) and submitted lazily:
// when a batch fills up, at flush()/finish(), or before any operation that touches memory from
// the host (write/read/copy/fill wait for all recorded work first). A batch also closes once its
// estimated GPU time reaches VULKANIZER_BATCH_MS: the kernel driver resets the GPU when a
// submission runs too long (amdgpu: 2 s). All methods are thread-safe.
class Device {
public:
    static VkResult create(const PhysicalDeviceInfo& info, std::unique_ptr<Device>& out);
    ~Device();

    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    const PhysicalDeviceInfo& info() const { return info_; }
    const Features& features() const { return features_; }
    VkDevice handle() const { return device_; }

    // Small allocations (up to kSubAllocLimit) are carved out of shared blocks: creating and
    // freeing Vulkan memory costs tens of microseconds, and the kernel zeroes every new page.
    // Offsets passed to write/read/copy/fill are relative to the allocation.
    VkResult allocateBuffer(VkDeviceSize size, Buffer& out);
    // Waits for recorded work first. A sub-allocation is free again at once: all later device
    // work is queued after the work recorded so far, and host access waits for that work.
    void freeBuffer(Buffer& buf);
    // Frees the buffer once the work up to `serial` has executed, without waiting for it.
    void freeBufferAfter(Buffer& buf, uint64_t serial);

    // Virtual memory: sparse buffers (no memory of their own) whose pages are bound to memory
    // objects. `opaqueAddress` places the buffer (0: anywhere; needs addressCaptureReplay), and
    // `opaqueOut` receives the value that places another buffer right here.
    VkDeviceSize sparsePageSize() const { return sparsePageSize_; }
    VkDeviceSize maxSparseBufferSize() const { return maxSparseBufferSize_; }
    VkDeviceSize hostImportAlignment() const { return hostImportAlignment_; }
    // Bits 63..47 of this device's buffer addresses, which kernels set again on every address
    // (specialization constant 5 of every pipeline, see common/device_address.h): 0xFFFF8000... on
    // RADV, 0 on llvmpipe (CPU pointers). Host memory used in place sits at h | hostAddressBits().
    uint64_t hostAddressBits() const { return uint64_t(addressHigh_) << 32; }
    // Makes [host, host + size) device-accessible at device address host | hostAddressBits() (a
    // sparse buffer placed there, with the host pages imported through VK_EXT_external_memory_host
    // bound to it). `host` and `size` must be multiples of hostImportAlignment(). out.address and
    // out.mapped are `host`. Fails if the device address range is taken.
    VkResult importHostMemory(void* host, VkDeviceSize size, Buffer& out);
    // Host-cached memory of the device, mapped at a host address h of its choosing and usable by
    // the device at h | hostAddressBits(), like imported memory (out.address = out.mapped = h). Unlike
    // imported memory (userptr), it costs nothing per submit: the kernel revalidates every
    // imported page on each submission. `size` is rounded up to whole sparsePageSize() units.
    VkResult allocateHostMemory(VkDeviceSize size, Buffer& out);
    VkResult createSparseBuffer(VkDeviceSize size, uint64_t opaqueAddress, Buffer& out, uint64_t& opaqueOut);
    VkResult allocateMemory(VkDeviceSize size, VkDeviceMemory& out);  // for sparse binding
    void freeMemory(VkDeviceMemory memory);
    // Takes effect for the work recorded afterwards; unbinding first waits for the recorded work.
    VkResult bindSparse(const std::vector<SparseBind>& binds);

    // All of these run in queue order with the recorded work. read waits for it and returns the
    // data; write returns once `src` was consumed (small writes are staged and recorded, large
    // ones wait and copy); copy and fill are recorded without waiting.
    VkResult write(const Buffer& dst, VkDeviceSize offset, const void* src, VkDeviceSize size);
    VkResult read(const Buffer& src, VkDeviceSize offset, void* dst, VkDeviceSize size);
    VkResult copy(const Buffer& src, VkDeviceSize srcOffset, const Buffer& dst, VkDeviceSize dstOffset,
                  VkDeviceSize size);
    // offset and size must be multiples of 4.
    VkResult fill(const Buffer& dst, VkDeviceSize offset, VkDeviceSize size, uint32_t pattern);

    // spec feeds specialization constants 0..4: blockDim.x/y/z, the dynamic shared memory size
    // in 32-bit words and the number of packed blocks (KernelBinary::packable).
    VkResult createComputePipeline(const uint32_t* code, size_t wordCount, const char* entryPoint,
                                   uint32_t pushConstantSize, const uint32_t spec[5], ComputePipeline& out);
    void destroyComputePipeline(ComputePipeline& p);  // waits for recorded work first

    // Asynchronous: recorded into the current batch. A grid that would run longer than the batch
    // budget is split over several batches (CUDA blocks are independent).
    // `label` names the kernel in VULKANIZER_PROFILE=1 reports (must outlive the device).
    // `params` (optional) is uploaded into each batch the dispatch is recorded in, and its address
    // goes into the first 8 bytes of the push constants.
    // `whole`: one dispatch, never split (all workgroups must be able to run at once: grid-wide
    // synchronization of cooperative launches).
    VkResult dispatch(const ComputePipeline& p, uint32_t groupsX, uint32_t groupsY, uint32_t groupsZ,
                      const void* pushData, uint32_t pushSize, const char* label = nullptr,
                      const void* params = nullptr, size_t paramsSize = 0, bool whole = false);
    // Copies data into memory that stays valid until the work recorded next has executed.
    VkResult upload(const void* data, VkDeviceSize size, VkDeviceAddress& address);

    // Progress is tracked by serial numbers: every recorded command belongs to a batch serial.
    uint64_t currentSerial();             // serial of the work recorded so far
    VkResult flush();                     // submits the open batch
    VkResult wait(uint64_t serial);       // flushes if needed and waits for that serial
    VkResult finish();                    // waits for everything recorded
    bool isComplete(uint64_t serial);
    bool idle();                          // nothing recorded or running

    // GPU timestamps (for CUDA events). Slots come from a fixed pool; -1 if unsupported.
    int32_t allocateTimestamp();
    void freeTimestamp(int32_t slot);
    VkResult recordTimestamp(int32_t slot, uint64_t& serial);
    bool readTimestampNs(int32_t slot, double& ns);  // false until the timestamp's batch completed

private:
    struct Timed {  // a dispatch with start/end timestamps
        const char* label;
        VkPipeline pipeline;
        uint64_t groups;
        double estimateNs;  // 0: not estimated (a pipeline's first launch)
    };
    struct Batch {
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        uint64_t serial = 0;
        bool pending = false;  // submitted, fence not yet observed
        Buffer arena;          // upload memory of this batch
        VkDeviceSize arenaUsed = 0;
        VkQueryPool timing = VK_NULL_HANDLE;  // start/end timestamps per dispatch
        std::vector<Timed> timed;
        double estimateNs = 0;  // expected GPU time of the recorded dispatches
        std::vector<const char*> kernels;  // dispatched kernels, named when the device is lost
    };
    static constexpr uint32_t kBatches = 8;
    static constexpr uint32_t kMaxCommandsPerBatch = 64;
    static constexpr uint32_t kTimestamps = 1024;
    static constexpr VkDeviceSize kAsyncWriteLimit = 256 * 1024;
    // Reads from uncached mapped memory larger than this go through readStaging_ (a GPU copy into
    // host-cached memory), in chunks of kReadChunk: uncached host reads run at ~0.2 GB/s.
    static constexpr VkDeviceSize kDirectReadLimit = 4 * 1024;
    static constexpr VkDeviceSize kReadChunk = 8 << 20;
    // A pipeline's first launch runs this many workgroups on their own to measure it. Rates are
    // taken per this many workgroups at least, as smaller grids leave the GPU partly idle.
    static constexpr uint64_t kRateGroups = 256;

    explicit Device(const PhysicalDeviceInfo& info) : info_(info) {}
    VkResult init();
    VkResult createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                          std::initializer_list<VkMemoryPropertyFlags> preferences, Buffer& out);
    int32_t findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags required) const;
    // Runs commands immediately and waits (after all recorded work).
    VkResult submit(const std::function<void(VkCommandBuffer)>& record);
    // Records into the open batch, starting one if needed.
    VkResult record(const std::function<void(VkCommandBuffer)>& commands);
    bool probeFusedFma();
    bool probeFusedFma64();
    void probeSparse();
    void destroyBuffer(Buffer& buf);
    void reapDeferredFrees();
    VkResult beginBatch();
    bool idleLocked();  // no open batch and every submitted one finished
    VkResult flushLocked();
    VkResult waitBatch(Batch& b);
    bool hasRecordedWork() const;
    void collectTimings(Batch& b);
    VkResult recordDispatch(const ComputePipeline& p, const uint32_t base[3], const uint32_t count[3],
                            const void* pushData, uint32_t pushSize, const char* label, double estimateNs);
    // Uploads `params` into the open batch unless already done (uploadedSerial) and patches the
    // address into push[0..8).
    VkResult uploadParams(const void* params, size_t size, std::vector<uint8_t>& push, uint64_t& uploadedSerial);

    PhysicalDeviceInfo info_;
    Features features_;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    Batch batches_[kBatches];
    int32_t open_ = -1;  // index of the batch being recorded
    uint32_t nextBatch_ = 0;
    uint32_t commands_ = 0;
    uint64_t nextSerial_ = 1;
    uint64_t completedSerial_ = 0;
    VkQueryPool timestamps_ = VK_NULL_HANDLE;
    double timestampPeriodNs_ = 0;
    std::vector<int32_t> freeTimestamps_;
    std::vector<uint64_t> timestampSerial_;
    bool timing_ = false;     // dispatches are timed (batch budget, profiling)
    bool profiling_ = false;  // VULKANIZER_PROFILE
    bool deviceLost_ = false;  // reported once
    bool launchBlocking_ = false;  // CUDA_LAUNCH_BLOCKING=1
    double batchBudgetNs_ = 0;  // 0: no limit
    // GPU time per workgroup of each pipeline: the highest seen, so estimates err on the long side.
    std::unordered_map<VkPipeline, double> nsPerGroup_;
    std::vector<std::pair<uint64_t, Buffer>> deferredFrees_;  // (serial, buffer)
    // Sub-allocation (VULKANIZER_NO_SUBALLOC=1 turns it off).
    static constexpr VkDeviceSize kSubAllocLimit = 4 << 20;
    static constexpr VkDeviceSize kBlockSize = 64 << 20;
    static constexpr VkDeviceSize kSubAllocAlign = 256;  // as cudaMalloc guarantees
    struct Block {
        Buffer buf;
        std::map<VkDeviceSize, VkDeviceSize> free;  // offset -> size, coalesced
        VkDeviceSize used = 0;
    };
    std::vector<std::unique_ptr<Block>> blocks_;
    bool subAlloc_ = true;
    VkResult allocateDedicated(VkDeviceSize size, Buffer& out);
    VkResult subAllocate(VkDeviceSize size, Buffer& out);
    void releaseSubAllocation(Buffer& buf);
    Buffer readStaging_;  // host-cached, two kReadChunk halves (read)
    VkDeviceSize sparsePageSize_ = 0;
    VkDeviceSize maxSparseBufferSize_ = 0;
    uint32_t sparseMemoryTypes_ = 0;
    VkDeviceSize hostImportAlignment_ = 0;
    uint32_t addressHigh_ = 0;  // hostAddressBits() >> 32
    PFN_vkGetMemoryHostPointerPropertiesEXT getHostPointerProperties_ = nullptr;
    PFN_vkMapMemory2KHR mapMemory2_ = nullptr;
    // VULKANIZER_PIPELINE_STATS=1: the driver's statistics of every pipeline (registers, spills...)
    PFN_vkGetPipelineExecutableStatisticsKHR getPipelineStats_ = nullptr;
    void printPipelineStats(VkPipeline pipeline, const char* entryPoint, const uint32_t spec[5]);
    PFN_vkUnmapMemory2KHR unmapMemory2_ = nullptr;
    // A sparse buffer of `size` at device address h | hostAddressBits() (fails if that is taken).
    VkResult createAtHostAddress(uint64_t h, VkDeviceSize size, bool imported, Buffer& out);
    // allocateHostMemory's buffer and host-visible memory at h | hostAddressBits(), not yet mapped:
    // a plain buffer on memory placed there, or a sparse buffer placed there.
    VkResult placeHostMemory(uint64_t h, VkDeviceSize size, bool sparse, Buffer& out);
    std::recursive_mutex mutex_;
};

}  // namespace vulkanizer::vk
