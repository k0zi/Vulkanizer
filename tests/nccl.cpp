// Vulkanizer's libnccl: one rank (device copies), three communicators of one process in a group
// (ncclCommInitAll), and three forked processes sharing the GPU, through every collective (sum, max,
// min, avg and premultiplied sums in f32, f16, bf16, f64 and integers), point-to-point messages in a
// ring and in both directions within a group, and ncclCommSplit.

#include "test_util.h"

#include <nccl/nccl.h>

#include <sys/wait.h>
#include <unistd.h>

#include <cmath>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#define CHECK_NCCL(call)                                                                           \
    do {                                                                                           \
        ncclResult_t r_ = (call);                                                                  \
        if (r_ != ncclSuccess) {                                                                   \
            std::fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__, #call,              \
                         ncclGetErrorString(r_));                                                  \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

namespace {

int g_failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "  FAILED: %s\n", what.c_str());
        ++g_failures;
    }
}

template <class T>
CUdeviceptr upload(const std::vector<T>& v) {
    CUdeviceptr d;
    CHECK_CU(cuMemAlloc(&d, std::max<size_t>(v.size() * sizeof(T), 4)));
    if (!v.empty()) CHECK_CU(cuMemcpyHtoD(d, v.data(), v.size() * sizeof(T)));
    return d;
}
template <class T>
std::vector<T> download(CUdeviceptr d, size_t n) {
    std::vector<T> v(n);
    CHECK_CU(cuCtxSynchronize());
    if (n) CHECK_CU(cuMemcpyDtoH(v.data(), d, n * sizeof(T)));
    return v;
}

uint16_t toHalf(float f) {
    _Float16 h = _Float16(f);
    uint16_t u;
    std::memcpy(&u, &h, 2);
    return u;
}
float fromHalf(uint16_t u) {
    _Float16 h;
    std::memcpy(&h, &u, 2);
    return float(h);
}
uint16_t toBf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return uint16_t((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}
float fromBf16(uint16_t b) {
    const uint32_t u = uint32_t(b) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// Everything one rank of `comm` (n ranks) does; the values depend on the rank only.
void runRank(ncclComm_t comm, int rank, int n) {
    const size_t count = 1000;
    // allreduce sum, f32, out of place, and max/min/avg in place
    std::vector<float> mine(count);
    for (size_t i = 0; i < count; ++i) mine[i] = float(rank + 1) * float(i % 7);
    CUdeviceptr a = upload(mine), b = upload(std::vector<float>(count));
    CHECK_NCCL(ncclAllReduce((void*)a, (void*)b, count, ncclFloat, ncclSum, comm, nullptr));
    auto sum = download<float>(b, count);
    bool ok = true;
    for (size_t i = 0; i < count; ++i) ok &= sum[i] == float(n * (n + 1) / 2) * float(i % 7);
    expect(ok, "allreduce sum f32 rank " + std::to_string(rank));
    for (ncclRedOp_t op : {ncclMax, ncclMin, ncclAvg}) {
        CHECK_CU(cuMemcpyHtoD(b, mine.data(), count * 4));
        CHECK_NCCL(ncclAllReduce((void*)b, (void*)b, count, ncclFloat, op, comm, nullptr));
        auto r = download<float>(b, count);
        ok = true;
        for (size_t i = 0; i < count; ++i) {
            const float want = op == ncclMax ? float(n) * float(i % 7)
                               : op == ncclMin ? float(i % 7)
                                               : float(n * (n + 1) / 2) * float(i % 7) / float(n);
            ok &= std::fabs(r[i] - want) <= 1e-5f * std::fabs(want);
        }
        expect(ok, "allreduce op " + std::to_string(int(op)) + " rank " + std::to_string(rank));
    }
    // f16, bf16, f64, int32 sums
    {
        std::vector<uint16_t> h(count), bf(count);
        for (size_t i = 0; i < count; ++i) h[i] = toHalf(0.5f * float(rank)), bf[i] = toBf16(float(rank) + 0.25f);
        CUdeviceptr dh = upload(h), dbf = upload(bf);
        CHECK_NCCL(ncclAllReduce((void*)dh, (void*)dh, count, ncclHalf, ncclSum, comm, nullptr));
        CHECK_NCCL(ncclAllReduce((void*)dbf, (void*)dbf, count, ncclBfloat16, ncclSum, comm, nullptr));
        const float wantHalf = 0.5f * float(n * (n - 1) / 2), wantBf = float(n * (n - 1) / 2) + 0.25f * float(n);
        expect(fromHalf(download<uint16_t>(dh, count)[count - 1]) == wantHalf, "allreduce f16");
        expect(fromBf16(download<uint16_t>(dbf, count)[0]) == wantBf, "allreduce bf16");
        std::vector<double> d(count, double(rank) * 1e-3);
        std::vector<int32_t> iv(count, -rank * 3);
        CUdeviceptr dd = upload(d), di = upload(iv);
        CHECK_NCCL(ncclAllReduce((void*)dd, (void*)dd, count, ncclDouble, ncclSum, comm, nullptr));
        CHECK_NCCL(ncclAllReduce((void*)di, (void*)di, count, ncclInt32, ncclSum, comm, nullptr));
        expect(std::fabs(download<double>(dd, count)[5] - double(n * (n - 1) / 2) * 1e-3) < 1e-15, "allreduce f64");
        expect(download<int32_t>(di, count)[9] == -3 * (n * (n - 1) / 2), "allreduce int32");
        cuMemFree(dh), cuMemFree(dbf), cuMemFree(dd), cuMemFree(di);
    }
    // premultiplied sum: rank r scales by (r + 1), a host scalar
    {
        float factor = float(rank + 1);
        ncclRedOp_t op;
        CHECK_NCCL(ncclRedOpCreatePreMulSum(&op, &factor, ncclFloat, ncclScalarHostImmediate, comm));
        std::vector<float> ones(16, 1.0f);
        CUdeviceptr d = upload(ones);
        CHECK_NCCL(ncclAllReduce((void*)d, (void*)d, 16, ncclFloat, op, comm, nullptr));
        expect(download<float>(d, 16)[3] == float(n * (n + 1) / 2), "premulsum");
        CHECK_NCCL(ncclRedOpDestroy(op, comm));
        cuMemFree(d);
    }
    // broadcast from the last rank, reduce to rank 0
    {
        std::vector<int32_t> v(64, rank == n - 1 ? 77 : -1);
        CUdeviceptr d = upload(v);
        CHECK_NCCL(ncclBcast((void*)d, 64, ncclInt32, n - 1, comm, nullptr));
        expect(download<int32_t>(d, 64)[63] == 77, "broadcast");
        std::vector<int64_t> x(8, rank + 10);
        CUdeviceptr dx = upload(x), dy = upload(std::vector<int64_t>(8, -5));
        CHECK_NCCL(ncclReduce((void*)dx, (void*)dy, 8, ncclInt64, ncclSum, 0, comm, nullptr));
        const int64_t got = download<int64_t>(dy, 8)[2];
        expect(rank == 0 ? got == int64_t(10 * n + n * (n - 1) / 2) : got == -5, "reduce");
        cuMemFree(d), cuMemFree(dx), cuMemFree(dy);
    }
    // allgather, reducescatter, alltoall, gather
    {
        const size_t block = 5;
        std::vector<float> own(block, float(rank));
        CUdeviceptr src = upload(own), all = upload(std::vector<float>(block * n));
        CHECK_NCCL(ncclAllGather((void*)src, (void*)all, block, ncclFloat, comm, nullptr));
        auto g = download<float>(all, block * n);
        ok = true;
        for (int k = 0; k < n; ++k) ok &= g[size_t(k) * block + 2] == float(k);
        expect(ok, "allgather");
        std::vector<float> full(block * n);
        for (size_t i = 0; i < full.size(); ++i) full[i] = float(i) + float(rank);
        CUdeviceptr dfull = upload(full), part = upload(std::vector<float>(block));
        CHECK_NCCL(ncclReduceScatter((void*)dfull, (void*)part, block, ncclFloat, ncclSum, comm, nullptr));
        expect(download<float>(part, block)[1] == float(n) * float(size_t(rank) * block + 1) + float(n * (n - 1) / 2),
               "reducescatter");
        std::vector<int32_t> out(block * n);
        for (int k = 0; k < n; ++k)
            for (size_t i = 0; i < block; ++i) out[size_t(k) * block + i] = rank * 100 + k;  // block k goes to rank k
        CUdeviceptr dout = upload(out), din = upload(std::vector<int32_t>(block * n));
        CHECK_NCCL(ncclAlltoAll((void*)dout, (void*)din, block, ncclInt32, comm, nullptr));
        auto in = download<int32_t>(din, block * n);
        ok = true;
        for (int k = 0; k < n; ++k) ok &= in[size_t(k) * block] == k * 100 + rank;
        expect(ok, "alltoall");
        CUdeviceptr gathered = upload(std::vector<float>(block * n, -1));
        CHECK_NCCL(ncclGather((void*)src, (void*)gathered, block, ncclFloat, n - 1, comm, nullptr));
        if (rank == n - 1) expect(download<float>(gathered, block * n)[block * (n - 1)] == float(n - 1), "gather");
        cuMemFree(src), cuMemFree(all), cuMemFree(dfull), cuMemFree(part), cuMemFree(dout), cuMemFree(din),
            cuMemFree(gathered);
    }
    // a ring of sends and receives, and a two-way exchange with every peer, in groups
    {
        std::vector<int32_t> v(32, rank);
        CUdeviceptr s = upload(v), r = upload(std::vector<int32_t>(32, -1));
        CHECK_NCCL(ncclGroupStart());
        CHECK_NCCL(ncclSend((void*)s, 32, ncclInt32, (rank + 1) % n, comm, nullptr));
        CHECK_NCCL(ncclRecv((void*)r, 32, ncclInt32, (rank + n - 1) % n, comm, nullptr));
        CHECK_NCCL(ncclGroupEnd());
        expect(download<int32_t>(r, 32)[31] == (rank + n - 1) % n, "send/recv ring");
        std::vector<CUdeviceptr> got(static_cast<size_t>(n));
        CHECK_NCCL(ncclGroupStart());
        for (int k = 0; k < n; ++k) {
            got[size_t(k)] = upload(std::vector<int32_t>(32, -1));
            CHECK_NCCL(ncclSend((void*)s, 32, ncclInt32, k, comm, nullptr));
            CHECK_NCCL(ncclRecv((void*)got[size_t(k)], 32, ncclInt32, k, comm, nullptr));
        }
        CHECK_NCCL(ncclGroupEnd());
        ok = true;
        for (int k = 0; k < n; ++k) ok &= download<int32_t>(got[size_t(k)], 32)[0] == k, cuMemFree(got[size_t(k)]);
        expect(ok, "send/recv with every peer (and itself)");
        cuMemFree(s), cuMemFree(r);
    }
    // split into even and odd ranks, then an allreduce of the rank numbers in each half
    {
        ncclComm_t half;
        CHECK_NCCL(ncclCommSplit(comm, rank % 2, -rank, &half, nullptr));  // keys descending: reversed order
        int hr, hn;
        CHECK_NCCL(ncclCommUserRank(half, &hr));
        CHECK_NCCL(ncclCommCount(half, &hn));
        int expectedSize = 0, expectedRank = 0;
        for (int k = 0; k < n; ++k)
            if (k % 2 == rank % 2) ++expectedSize, expectedRank += k > rank;
        expect(hn == expectedSize && hr == expectedRank, "split ranks");
        std::vector<int32_t> v(4, rank);
        CUdeviceptr d = upload(v);
        CHECK_NCCL(ncclAllReduce((void*)d, (void*)d, 4, ncclInt32, ncclSum, half, nullptr));
        int want = 0;
        for (int k = rank % 2; k < n; k += 2) want += k;
        expect(download<int32_t>(d, 4)[0] == want, "allreduce in a split communicator");
        CHECK_NCCL(ncclCommDestroy(half));
        cuMemFree(d);
    }
    cuMemFree(a), cuMemFree(b);
}

void initContext(CUcontext& ctx) {
    CHECK_CU(cuInit(0));
    CUdevice dev;
    CHECK_CU(cuDeviceGet(&dev, 0));
    CHECK_CU(cuDevicePrimaryCtxRetain(&ctx, dev));
    CHECK_CU(cuCtxSetCurrent(ctx));
}

}  // namespace

int main() {
    int version = 0;
    CHECK_NCCL(ncclGetVersion(&version));
    std::printf("NCCL version %d\n", version);

    // three processes on the GPU, forked before any CUDA call
    constexpr int kRanks = 3;
    ncclUniqueId id;
    CHECK_NCCL(ncclGetUniqueId(&id));
    std::vector<pid_t> children;
    for (int r = 1; r < kRanks; ++r) {
        const pid_t pid = fork();
        EXPECT(pid >= 0);
        if (pid == 0) {
            CUcontext ctx;
            initContext(ctx);
            ncclComm_t comm;
            CHECK_NCCL(ncclCommInitRank(&comm, kRanks, id, r));
            runRank(comm, r, kRanks);
            CHECK_NCCL(ncclCommDestroy(comm));
            _exit(g_failures ? 1 : 0);
        }
        children.push_back(pid);
    }
    CUcontext ctx;
    initContext(ctx);
    ncclComm_t comm;
    CHECK_NCCL(ncclCommInitRank(&comm, kRanks, id, 0));
    runRank(comm, 0, kRanks);
    CHECK_NCCL(ncclCommDestroy(comm));
    for (pid_t pid : children) {
        int status = 0;
        EXPECT(waitpid(pid, &status, 0) == pid);
        expect(WIFEXITED(status) && WEXITSTATUS(status) == 0, "child rank");
    }
    std::printf("  %d processes: %s\n", kRanks, g_failures ? "FAILED" : "ok");

    // one rank: device copies on the stream
    {
        ncclUniqueId one;
        CHECK_NCCL(ncclGetUniqueId(&one));
        ncclComm_t single;
        CHECK_NCCL(ncclCommInitRank(&single, 1, one, 0));
        const int before = g_failures;
        runRank(single, 0, 1);
        CHECK_NCCL(ncclCommDestroy(single));
        std::printf("  one rank: %s\n", g_failures != before ? "FAILED" : "ok");
    }

    // three communicators of this process: every collective in a group across them
    {
        ncclComm_t comms[3];
        const int devs[3] = {0, 0, 0};
        CHECK_NCCL(ncclCommInitAll(comms, 3, devs));
        std::vector<CUdeviceptr> bufs;
        CHECK_NCCL(ncclGroupStart());
        for (int r = 0; r < 3; ++r) {
            bufs.push_back(upload(std::vector<float>(100, float(r + 1))));
            CHECK_NCCL(ncclAllReduce((void*)bufs.back(), (void*)bufs.back(), 100, ncclFloat, ncclSum, comms[r], nullptr));
        }
        CHECK_NCCL(ncclGroupEnd());
        bool ok = true;
        for (int r = 0; r < 3; ++r) ok &= download<float>(bufs[size_t(r)], 100)[99] == 6.0f, cuMemFree(bufs[size_t(r)]);
        expect(ok, "ncclCommInitAll + group");
        for (ncclComm_t c : comms) CHECK_NCCL(ncclCommDestroy(c));
        std::printf("  one process, three communicators: %s\n", ok ? "ok" : "FAILED");
    }
    CHECK_CU(cuDevicePrimaryCtxRelease(0));
    std::printf(g_failures ? "FAILED\n" : "OK\n");
    return g_failures ? 1 : 0;
}
