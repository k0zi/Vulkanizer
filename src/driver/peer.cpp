// Copies between devices (multi-GPU, plan.md 8.3) and cross-device waits (8.4).
//
// Separate Vulkan devices share no memory, and peer access stays off (cuDeviceCanAccessPeer = 0):
// a copy goes through staging memory both devices reach. The source device allocates system memory
// exportable as a dma-buf (vk::Device::allocateShared) and the destination imports it
// (importShared), once per ordered device pair (PeerLink). Each chunk of a copy is then two device
// copies, in queue order on both devices:
//
//   source queue:       ... earlier work | [wait: slot free] | src -> slot
//   destination queue:  ... earlier work | [wait: source copy done] | slot -> dst | later work
//
// The waits are host steps (vk::Device::enqueueHost): the device's worker thread waits for the
// other device's timeline value, then lets its own queue go on. The caller never waits (unless the
// copy is synchronous), the host never touches the data, and a large copy streams through the
// slots with both devices busy. Where the devices cannot share memory (no dma-buf, or the import
// fails) the data goes through the host as before.

#include "driver/driver.h"

#include "common/log.h"

#include <algorithm>
#include <cstdlib>

namespace vulkanizer::driver {

namespace {

// VULKANIZER_PEER_STAGING_MB: staging per device pair and direction (default 64 MiB, 4 slots).
constexpr uint32_t kSlots = 4;
VkDeviceSize stagingBytes() {
    static const VkDeviceSize bytes = [] {
        const char* env = std::getenv("VULKANIZER_PEER_STAGING_MB");
        const long mb = env && *env ? std::atol(env) : 64;
        return VkDeviceSize(std::max(4L, mb)) << 20;
    }();
    return bytes;
}

PeerLink* link(DeviceSlot& from, DeviceSlot& to) {
    std::unique_ptr<PeerLink>& l = from.peerLinks[to.ordinal];
    if (l) return l->failed ? nullptr : l.get();
    l = std::make_unique<PeerLink>();
    vk::Device &a = *from.vk, &b = *to.vk;
    const VkDeviceSize total = stagingBytes() / (kSlots * 256) * (kSlots * 256);
    int fd = -1;
    VkResult r = a.allocateShared(total, l->src, fd);
    if (r == VK_SUCCESS) r = b.importShared(fd, total, l->dst);  // takes the fd
    if (r != VK_SUCCESS) {
        if (l->src.buffer) a.freeBuffer(l->src);
        l->failed = true;
        VULKANIZER_INFO("peer copies %s -> %s go through the host (no shared staging memory: %d)", a.info().name(),
                        b.info().name(), int(r));
        return nullptr;
    }
    l->chunk = total / kSlots;
    l->chunks = kSlots;
    l->used.assign(kSlots, 0);
    VULKANIZER_DEBUG("peer staging %s -> %s: %u x %llu bytes", a.info().name(), b.info().name(), kSlots,
                     static_cast<unsigned long long>(l->chunk));
    return l.get();
}

// The device copies between the range [at, at + n) of a piece list and `staging` from `offset`.
VkResult copyRange(vk::Device& dev, const std::vector<AllocPiece>& pieces, size_t at, size_t n,
                   const vk::Buffer& staging, VkDeviceSize offset, bool toStaging) {
    for (const AllocPiece& p : pieces) {
        const size_t lo = std::max(at, p.at), hi = std::min(at + n, p.at + p.size);
        if (lo >= hi) continue;
        const VkDeviceSize bufOffset = p.ref.offset + (lo - p.at), stOffset = offset + (lo - at);
        const VkResult r = toStaging ? dev.copy(*p.ref.buffer, bufOffset, staging, stOffset, hi - lo)
                                     : dev.copy(staging, stOffset, *p.ref.buffer, bufOffset, hi - lo);
        if (r) return r;
    }
    return VK_SUCCESS;
}

// Through the host (no shared memory): the read waits for the source device's work so far, the
// write is queued on the destination device.
CUresult copyThroughHost(DeviceSlot& to, const std::vector<AllocPiece>& dst, DeviceSlot& from,
                         const std::vector<AllocPiece>& src, size_t bytes, bool sync) {
    std::vector<char> staging(bytes);
    for (const AllocPiece& p : src)
        if (VkResult r = from.vk->read(*p.ref.buffer, p.ref.offset, staging.data() + p.at, p.size)) return toCuResult(r);
    for (const AllocPiece& p : dst)
        if (VkResult r = to.vk->write(*p.ref.buffer, p.ref.offset, staging.data() + p.at, p.size)) return toCuResult(r);
    return sync ? toCuResult(to.vk->finish()) : CUDA_SUCCESS;
}

}  // namespace

CUresult waitForOtherDevice(vk::Device& waiting, vk::Device& other, uint64_t serial) {
    if (&waiting == &other || other.isComplete(serial)) return CUDA_SUCCESS;
    if (VkResult r = other.flush()) return toCuResult(r);  // waitSubmitted needs it submitted
    vk::Device* o = &other;
    return toCuResult(waiting.enqueueHost([o, serial] {
        if (VkResult r = o->waitSubmitted(serial)) VULKANIZER_WARN("waiting for another device failed (%d)", int(r));
    }));
}

CUresult copyBetweenDevices(DeviceSlot& to, const std::vector<AllocPiece>& dst, DeviceSlot& from,
                            const std::vector<AllocPiece>& src, size_t bytes, bool sync) {
    if (bytes == 0) return CUDA_SUCCESS;
    PeerLink* l = link(from, to);
    if (!l) return copyThroughHost(to, dst, from, src, bytes, sync);
    vk::Device &a = *from.vk, &b = *to.vk;
    for (size_t at = 0; at < bytes; at += l->chunk) {
        const size_t n = std::min<size_t>(l->chunk, bytes - at);
        const uint32_t k = l->next;
        l->next = (k + 1) % l->chunks;
        const VkDeviceSize slot = k * l->chunk;
        // the slot's last copy out (on the destination) must be done before the source overwrites it
        if (CUresult r = waitForOtherDevice(a, b, l->used[k])) return r;
        if (VkResult r = copyRange(a, src, at, n, l->src, slot, true)) return toCuResult(r);
        const uint64_t copiedIn = a.currentSerial();
        if (CUresult r = waitForOtherDevice(b, a, copiedIn)) return r;
        if (VkResult r = copyRange(b, dst, at, n, l->dst, slot, false)) return toCuResult(r);
        l->used[k] = b.currentSerial();
    }
    if (VkResult r = a.flush()) return toCuResult(r);
    if (VkResult r = sync ? b.wait(b.currentSerial()) : b.flush()) return toCuResult(r);
    return CUDA_SUCCESS;
}

}  // namespace vulkanizer::driver
