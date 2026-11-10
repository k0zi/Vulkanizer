// CUDA IPC: device memory and events shared between processes.
//
// Device allocations are exportable Vulkan memory (vk::Device::exportMemory). A memory handle names
// the exporting process, its memory and the position in it; it is plain data and carries no file
// descriptor, so the exporting process serves descriptors of its exported memory on an abstract
// Unix socket ("vulkanizer-ipc-<uid>-<pid>", to processes of the same user only), started with the
// first cuIpcGetMemHandle. The importing process asks for the descriptor, imports the whole memory
// once (more opens of handles into the same memory share it) and gets the address in it.
//
// An interprocess event completes when it is recorded (cuEventRecord waits for the device), so the
// event another process opens from its handle never has anything to wait for.

#include "driver/driver.h"

#include "common/log.h"

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstring>
#include <string>
#include <thread>

using namespace vulkanizer;
using namespace vulkanizer::driver;

namespace {

constexpr uint32_t kMemMagic = 0x50494b56;    // 'VKIP'
constexpr uint32_t kEventMagic = 0x45494b56;  // 'VKIE'
constexpr uint16_t kVersion = 1;

struct MemHandle {
    uint32_t magic;
    uint16_t version;
    uint16_t ordinal;  // the exporter's device
    int32_t pid;
    uint32_t memoryType;
    uint64_t memoryId;
    uint64_t offset;      // of the shared address in the memory
    uint64_t bufferSize;  // the exporter's buffer and memory (import parameters)
    uint64_t memorySize;
    uint64_t reserved;
    uint8_t uuid[8];  // the device (first bytes of its UUID)
};
static_assert(sizeof(MemHandle) == CU_IPC_HANDLE_SIZE);

struct EventHandle {
    uint32_t magic;
    int32_t pid;
    uint64_t event;
    uint8_t unused[48];
};
static_assert(sizeof(EventHandle) == CU_IPC_HANDLE_SIZE);

struct Request {
    uint32_t magic;
    uint32_t ordinal;
    uint64_t memoryId;
};

sockaddr_un socketAddress(int32_t pid, socklen_t& length) {
    sockaddr_un a{};
    a.sun_family = AF_UNIX;
    const std::string name = "vulkanizer-ipc-" + std::to_string(getuid()) + "-" + std::to_string(pid);
    std::memcpy(a.sun_path + 1, name.data(), name.size());  // abstract: sun_path[0] = 0
    length = socklen_t(offsetof(sockaddr_un, sun_path) + 1 + name.size());
    return a;
}

// Sends one descriptor (or none, with an error status) over a connected socket.
void reply(int sock, int32_t status, int fd) {
    iovec iov{&status, sizeof status};
    msghdr msg{};
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    alignas(cmsghdr) char control[CMSG_SPACE(sizeof(int))] = {};
    if (fd >= 0) {
        msg.msg_control = control;
        msg.msg_controllen = sizeof control;
        cmsghdr* c = CMSG_FIRSTHDR(&msg);
        c->cmsg_level = SOL_SOCKET;
        c->cmsg_type = SCM_RIGHTS;
        c->cmsg_len = CMSG_LEN(sizeof(int));
        std::memcpy(CMSG_DATA(c), &fd, sizeof fd);
    }
    (void)sendmsg(sock, &msg, MSG_NOSIGNAL);
}

void serve(int listener) {
    for (;;) {
        const int sock = accept4(listener, nullptr, nullptr, SOCK_CLOEXEC);
        if (sock < 0) {
            if (errno == EINTR || errno == ECONNABORTED) continue;
            VULKANIZER_WARN("IPC: accept failed: %s", std::strerror(errno));
            return;
        }
        ucred peer{};
        socklen_t len = sizeof peer;
        Request req{};
        if (getsockopt(sock, SOL_SOCKET, SO_PEERCRED, &peer, &len) != 0 || peer.uid != getuid()) {
            reply(sock, -1, -1);
        } else if (recv(sock, &req, sizeof req, MSG_WAITALL) != ssize_t(sizeof req) || req.magic != kMemMagic) {
            reply(sock, -1, -1);
        } else {
            int fd = -1;
            DeviceSlot* slot = Driver::get().slot(CUdevice(req.ordinal));
            vk::Device* dev = slot ? slot->vk.get() : nullptr;
            const VkResult r = dev ? dev->exportMemory(req.memoryId, fd) : VK_ERROR_INITIALIZATION_FAILED;
            reply(sock, r == VK_SUCCESS ? 0 : int32_t(r), r == VK_SUCCESS ? fd : -1);
            if (fd >= 0) close(fd);
        }
        close(sock);
    }
}

// Starts this process's descriptor server once (again in a forked child: the thread is not inherited).
CUresult startServer() {
    static std::mutex mutex;
    static int32_t serverPid = 0;
    static int listener = -1;
    std::lock_guard<std::mutex> lock(mutex);
    const int32_t pid = getpid();
    if (serverPid == pid) return CUDA_SUCCESS;
    if (listener >= 0) close(listener);  // the parent's
    listener = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    socklen_t length;
    const sockaddr_un a = socketAddress(pid, length);
    if (listener < 0 || bind(listener, reinterpret_cast<const sockaddr*>(&a), length) != 0 || listen(listener, 64) != 0) {
        VULKANIZER_WARN("IPC: cannot listen for handle requests: %s", std::strerror(errno));
        if (listener >= 0) close(listener);
        listener = -1;
        return CUDA_ERROR_OPERATING_SYSTEM;
    }
    std::thread(serve, listener).detach();
    serverPid = pid;
    return CUDA_SUCCESS;
}

// Asks process `pid` for a descriptor of its memory `id`.
CUresult requestDescriptor(int32_t pid, uint32_t ordinal, uint64_t id, int& fd) {
    fd = -1;
    const int sock = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (sock < 0) return CUDA_ERROR_OPERATING_SYSTEM;
    timeval timeout{10, 0};
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof timeout);
    socklen_t length;
    const sockaddr_un a = socketAddress(pid, length);
    const Request req{kMemMagic, ordinal, id};
    int32_t status = -1;
    iovec iov{&status, sizeof status};
    msghdr msg{};
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    alignas(cmsghdr) char control[CMSG_SPACE(sizeof(int))] = {};
    msg.msg_control = control;
    msg.msg_controllen = sizeof control;
    if (connect(sock, reinterpret_cast<const sockaddr*>(&a), length) != 0 ||
        send(sock, &req, sizeof req, MSG_NOSIGNAL) != ssize_t(sizeof req) || recvmsg(sock, &msg, MSG_CMSG_CLOEXEC) <= 0) {
        VULKANIZER_WARN("IPC: process %d does not answer: %s", pid, std::strerror(errno));
        close(sock);
        return CUDA_ERROR_INVALID_HANDLE;
    }
    close(sock);
    for (cmsghdr* c = CMSG_FIRSTHDR(&msg); c; c = CMSG_NXTHDR(&msg, c))
        if (c->cmsg_level == SOL_SOCKET && c->cmsg_type == SCM_RIGHTS) std::memcpy(&fd, CMSG_DATA(c), sizeof fd);
    if (status != 0 || fd < 0) {
        if (fd >= 0) close(fd);
        return CUDA_ERROR_INVALID_HANDLE;  // freed by the exporter
    }
    return CUDA_SUCCESS;
}

}  // namespace

extern "C" {

CUresult CUDAAPI cuIpcGetMemHandle(CUipcMemHandle* pHandle, CUdeviceptr dptr) {
    VULKANIZER_TRACE_API();
    if (!pHandle) return CUDA_ERROR_INVALID_VALUE;
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    AllocRef ref;
    if (!findAllocation(*s.slot, dptr, 1, ref)) return CUDA_ERROR_INVALID_VALUE;
    const vk::Buffer& buf = *ref.buffer;
    if (!buf.exportable) return buf.hostAddressed ? CUDA_ERROR_INVALID_VALUE : CUDA_ERROR_NOT_SUPPORTED;
    MemHandle h{};
    if (VkResult vr = s.dev->exportId(buf, h.memoryId)) return toCuResult(vr);
    if (CUresult r = startServer()) return r;
    h.magic = kMemMagic;
    h.version = kVersion;
    h.ordinal = uint16_t(s.slot->ordinal);
    h.pid = getpid();
    h.memoryType = buf.memoryType;
    h.offset = buf.offset + ref.offset;  // ref.offset: dptr within the allocation
    h.bufferSize = buf.bufferSize;
    h.memorySize = buf.memorySize;
    std::memcpy(h.uuid, s.slot->info->uuid, sizeof h.uuid);
    std::memcpy(pHandle->reserved, &h, sizeof h);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuIpcOpenMemHandle_v2(CUdeviceptr* pdptr, CUipcMemHandle handle, unsigned int) {
    VULKANIZER_TRACE_API();
    if (!pdptr) return CUDA_ERROR_INVALID_VALUE;
    MemHandle h;
    std::memcpy(&h, handle.reserved, sizeof h);
    if (h.magic != kMemMagic || h.version != kVersion) return CUDA_ERROR_INVALID_VALUE;
    if (h.pid == getpid()) return CUDA_ERROR_INVALID_CONTEXT;  // as CUDA: not within one process
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (std::memcmp(h.uuid, s.slot->info->uuid, sizeof h.uuid) != 0) return CUDA_ERROR_INVALID_DEVICE;
    const auto key = std::make_pair(h.pid, h.memoryId);
    auto it = s.slot->ipcImports.find(key);
    if (it == s.slot->ipcImports.end()) {
        int fd;
        if (CUresult r = requestDescriptor(h.pid, h.ordinal, h.memoryId, fd)) return r;
        vk::Buffer buf;
        if (VkResult vr = s.dev->importMemory(fd, h.bufferSize, h.memorySize, h.memoryType, buf)) {
            close(fd);
            VULKANIZER_WARN("IPC: importing memory of process %d failed (%d)", h.pid, int(vr));
            return toCuResult(vr);
        }
        s.slot->allocations[buf.address] = buf;
        it = s.slot->ipcImports.emplace(key, DeviceSlot::IpcImport{buf.address, 0}).first;
    }
    ++it->second.refs;
    *pdptr = it->second.base + h.offset;
    return CUDA_SUCCESS;
}

// The unversioned name (cuda.h maps it to _v2), for binaries that link it.
#undef cuIpcOpenMemHandle
CUresult CUDAAPI cuIpcOpenMemHandle(CUdeviceptr* pdptr, CUipcMemHandle handle, unsigned int Flags);
CUresult CUDAAPI cuIpcOpenMemHandle(CUdeviceptr* pdptr, CUipcMemHandle handle, unsigned int Flags) {
    return cuIpcOpenMemHandle_v2(pdptr, handle, Flags);
}

CUresult CUDAAPI cuIpcCloseMemHandle(CUdeviceptr dptr) {
    VULKANIZER_TRACE_API();
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    for (auto it = s.slot->ipcImports.begin(); it != s.slot->ipcImports.end(); ++it) {
        auto alloc = s.slot->allocations.find(it->second.base);
        if (alloc == s.slot->allocations.end() || dptr < alloc->first || dptr - alloc->first >= alloc->second.size)
            continue;
        if (--it->second.refs == 0) {
            s.dev->freeBuffer(alloc->second);  // after the work recorded so far
            s.slot->allocations.erase(alloc);
            s.slot->ipcImports.erase(it);
        }
        return CUDA_SUCCESS;
    }
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult CUDAAPI cuIpcGetEventHandle(CUipcEventHandle* pHandle, CUevent event) {
    VULKANIZER_TRACE_API();
    if (!pHandle) return CUDA_ERROR_INVALID_VALUE;
    if (!validHandle(event)) return CUDA_ERROR_INVALID_HANDLE;
    if (!(event->flags & CU_EVENT_INTERPROCESS) || !(event->flags & CU_EVENT_DISABLE_TIMING))
        return CUDA_ERROR_INVALID_VALUE;
    EventHandle h{};
    h.magic = kEventMagic;
    h.pid = getpid();
    h.event = reinterpret_cast<uintptr_t>(event);
    std::memcpy(pHandle->reserved, &h, sizeof h);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuIpcOpenEventHandle(CUevent* phEvent, CUipcEventHandle handle) {
    VULKANIZER_TRACE_API();
    if (!phEvent) return CUDA_ERROR_INVALID_VALUE;
    EventHandle h;
    std::memcpy(&h, handle.reserved, sizeof h);
    if (h.magic != kEventMagic) return CUDA_ERROR_INVALID_VALUE;
    if (h.pid == getpid()) return CUDA_ERROR_INVALID_CONTEXT;
    // Already complete: its recording in the other process waited for that process's work.
    return cuEventCreate(phEvent, CU_EVENT_INTERPROCESS | CU_EVENT_DISABLE_TIMING);
}

}  // extern "C"
