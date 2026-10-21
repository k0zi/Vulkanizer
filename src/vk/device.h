#pragma once

#include "vk/instance.h"

#include <functional>
#include <memory>
#include <mutex>
#include <vector>

namespace vulkanizer::vk {

// A device allocation addressable from shaders through its buffer device address.
struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    VkDeviceAddress address = 0;
    void* mapped = nullptr;  // set when the memory is host-visible and coherent
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
};

// Logical device with a single compute queue. Kernel dispatches are recorded into batches
// (one command buffer each, a memory barrier between commands) and submitted lazily:
// when a batch fills up, at flush()/finish(), or before any operation that touches memory from
// the host (write/read/copy/fill wait for all recorded work first). All methods are thread-safe.
class Device {
public:
    static VkResult create(const PhysicalDeviceInfo& info, std::unique_ptr<Device>& out);
    ~Device();

    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    const PhysicalDeviceInfo& info() const { return info_; }
    const Features& features() const { return features_; }
    VkDevice handle() const { return device_; }

    VkResult allocateBuffer(VkDeviceSize size, Buffer& out);
    void freeBuffer(Buffer& buf);  // waits for recorded work first

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

    // Asynchronous: recorded into the current batch.
    // `label` names the kernel in VULKANIZER_PROFILE=1 reports (must outlive the device).
    VkResult dispatch(const ComputePipeline& p, uint32_t groupsX, uint32_t groupsY, uint32_t groupsZ,
                      const void* pushData, uint32_t pushSize, const char* label = nullptr);
    // Copies data into memory that stays valid until the work recorded next has executed.
    VkResult upload(const void* data, VkDeviceSize size, VkDeviceAddress& address);

    // Progress is tracked by serial numbers: every recorded command belongs to a batch serial.
    uint64_t currentSerial();             // serial of the work recorded so far
    VkResult flush();                     // submits the open batch
    VkResult wait(uint64_t serial);       // flushes if needed and waits for that serial
    VkResult finish();                    // waits for everything recorded
    bool isComplete(uint64_t serial);

    // GPU timestamps (for CUDA events). Slots come from a fixed pool; -1 if unsupported.
    int32_t allocateTimestamp();
    void freeTimestamp(int32_t slot);
    VkResult recordTimestamp(int32_t slot, uint64_t& serial);
    bool readTimestampNs(int32_t slot, double& ns);  // false until the timestamp's batch completed

private:
    struct Batch {
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        uint64_t serial = 0;
        bool pending = false;  // submitted, fence not yet observed
        Buffer arena;          // upload memory of this batch
        VkDeviceSize arenaUsed = 0;
        VkQueryPool profile = VK_NULL_HANDLE;  // VULKANIZER_PROFILE: start/end timestamps per dispatch
        std::vector<const char*> labels;
    };
    static constexpr uint32_t kBatches = 8;
    static constexpr uint32_t kMaxCommandsPerBatch = 64;
    static constexpr uint32_t kTimestamps = 1024;
    static constexpr VkDeviceSize kAsyncWriteLimit = 256 * 1024;

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
    VkResult beginBatch();
    bool idleLocked();  // no open batch and every submitted one finished
    VkResult flushLocked();
    VkResult waitBatch(Batch& b);
    bool hasRecordedWork() const;
    void collectProfile(Batch& b);

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
    bool profiling_ = false;
    std::recursive_mutex mutex_;
};

}  // namespace vulkanizer::vk
