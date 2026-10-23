// CUDA IPC between two processes: a parent exports device memory (a small, sub-allocated
// allocation, a large dedicated one, and a pointer inside one) and an interprocess event; a forked
// child opens the handles, checks the parent's data, overwrites it on the device and closes the
// handles; the parent then sees the child's data. Also: opening a handle in its own process fails,
// two opens of one memory share the mapping.

#include "test_util.h"

#include <sys/wait.h>
#include <unistd.h>

#include <cstring>
#include <vector>

namespace {

struct Message {
    CUipcMemHandle small, large, inner;
    CUipcEventHandle event;
};

constexpr size_t kSmall = 64 << 10, kLarge = 8 << 20, kInnerOffset = 4096;

void initContext(CUcontext& ctx) {
    CHECK_CU(cuInit(0));
    CUdevice dev;
    CHECK_CU(cuDeviceGet(&dev, 0));
    CHECK_CU(cuCtxCreate(&ctx, 0, dev));
}

bool expectWords(CUdeviceptr p, size_t bytes, uint32_t value, const char* what) {
    std::vector<uint32_t> v(bytes / 4);
    CHECK_CU(cuMemcpyDtoH(v.data(), p, bytes));
    for (size_t i = 0; i < v.size(); ++i)
        if (v[i] != value) {
            std::fprintf(stderr, "%s: word %zu is 0x%x, expected 0x%x\n", what, i, v[i], value);
            return false;
        }
    return true;
}

int child(int readFd) {
    Message m;
    if (read(readFd, &m, sizeof m) != ssize_t(sizeof m)) return 10;
    CUcontext ctx;
    initContext(ctx);
    CUdeviceptr small, large, inner, again;
    CHECK_CU(cuIpcOpenMemHandle(&small, m.small, CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS));
    CHECK_CU(cuIpcOpenMemHandle(&large, m.large, CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS));
    CHECK_CU(cuIpcOpenMemHandle(&inner, m.inner, CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS));
    CHECK_CU(cuIpcOpenMemHandle(&again, m.small, CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS));
    if (again != small) return 11;  // one mapping of the memory
    if (inner != large + kInnerOffset) return 12;
    CUevent event;
    CHECK_CU(cuIpcOpenEventHandle(&event, m.event));
    CHECK_CU(cuStreamWaitEvent(nullptr, event, 0));
    CHECK_CU(cuEventSynchronize(event));
    bool ok = expectWords(small, kSmall, 0x11111111, "child: small") &&
              expectWords(large, kLarge, 0x22222222, "child: large") &&
              expectWords(inner, 256, 0x22222222, "child: inner");
    // overwrite on the device: a memset kernel, and a host-to-device copy
    CHECK_CU(cuMemsetD32(small, 0x33333333, kSmall / 4));
    std::vector<uint32_t> words(kLarge / 4, 0x44444444);
    CHECK_CU(cuMemcpyHtoD(large, words.data(), kLarge));
    CHECK_CU(cuCtxSynchronize());
    CHECK_CU(cuIpcCloseMemHandle(again));
    CHECK_CU(cuIpcCloseMemHandle(small));
    CHECK_CU(cuIpcCloseMemHandle(inner));
    CHECK_CU(cuIpcCloseMemHandle(large));
    EXPECT_CU(CUDA_ERROR_INVALID_VALUE, cuIpcCloseMemHandle(large));  // all closed
    CHECK_CU(cuEventDestroy(event));
    CHECK_CU(cuCtxDestroy(ctx));
    return ok ? 0 : 13;
}

}  // namespace

int main() {
    int fds[2];
    EXPECT(pipe(fds) == 0);
    const pid_t pid = fork();  // before any CUDA call: each process has its own device
    EXPECT(pid >= 0);
    if (pid == 0) {
        close(fds[1]);
        _exit(child(fds[0]));
    }
    close(fds[0]);
    CUcontext ctx;
    initContext(ctx);
    CUdeviceptr small, large;
    CHECK_CU(cuMemAlloc(&small, kSmall));
    CHECK_CU(cuMemAlloc(&large, kLarge));
    CHECK_CU(cuMemsetD32(small, 0x11111111, kSmall / 4));
    CHECK_CU(cuMemsetD32(large, 0x22222222, kLarge / 4));
    Message m;
    CHECK_CU(cuIpcGetMemHandle(&m.small, small));
    CHECK_CU(cuIpcGetMemHandle(&m.large, large));
    CHECK_CU(cuIpcGetMemHandle(&m.inner, large + kInnerOffset));
    CUdeviceptr self;
    EXPECT_CU(CUDA_ERROR_INVALID_CONTEXT, cuIpcOpenMemHandle(&self, m.small, 0));
    CUevent event;
    CHECK_CU(cuEventCreate(&event, CU_EVENT_INTERPROCESS | CU_EVENT_DISABLE_TIMING));
    CUipcEventHandle timed;
    CUevent timing;
    CHECK_CU(cuEventCreate(&timing, 0));
    EXPECT_CU(CUDA_ERROR_INVALID_VALUE, cuIpcGetEventHandle(&timed, timing));  // needs the interprocess flags
    CHECK_CU(cuEventRecord(event, nullptr));
    CHECK_CU(cuIpcGetEventHandle(&m.event, event));
    EXPECT(write(fds[1], &m, sizeof m) == ssize_t(sizeof m));
    int status = 0;
    EXPECT(waitpid(pid, &status, 0) == pid);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        std::fprintf(stderr, "child failed (status %d)\n", WIFEXITED(status) ? WEXITSTATUS(status) : -1);
        return 1;
    }
    const bool ok = expectWords(small, kSmall, 0x33333333, "parent: small") &&
                    expectWords(large, kLarge, 0x44444444, "parent: large");
    CHECK_CU(cuEventDestroy(event));
    CHECK_CU(cuEventDestroy(timing));
    CHECK_CU(cuMemFree(small));
    CHECK_CU(cuMemFree(large));
    CHECK_CU(cuCtxDestroy(ctx));
    std::printf(ok ? "OK\n" : "FAILED\n");
    return ok ? 0 : 1;
}
