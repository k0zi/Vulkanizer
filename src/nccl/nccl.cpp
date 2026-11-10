// Vulkanizer's libnccl: NCCL's collectives and point-to-point operations between the ranks of one
// machine (processes, or several communicators of one process), for PyTorch's torch.distributed
// "nccl" backend (DDP, FSDP...). The ranks may share one GPU.
//
// A communicator of one rank does every collective as a device copy on the stream. With more ranks,
// data goes through the host: every operation is posted, then completed. Posting waits for the
// stream and publishes this rank's contribution in a shared memory object of its own (POSIX shm,
// named after the communicator's unique id, the operation's sequence number and the rank) and never
// blocks; completing waits for the objects it needs, reduces them in rank order (every rank gets the
// same bits) and copies the result to the device. The last reader of an object unlinks it. Groups
// (ncclGroupStart/End) post all their operations before completing any, so the sends and receives
// of a group, and the ranks of one process, never wait for each other.
//
// Collectives must be called in the same order on every rank (as with NCCL); point-to-point
// messages are matched by their order per pair of ranks.

#include <nccl/nccl.h>
#include <cuda.h>

#include "common/log.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <map>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

struct ncclComm {
    uint64_t token[2] = {};  // names the communicator's shared memory objects
    int nranks = 1, rank = 0;
    CUdevice device = 0;
    CUcontext ctx = nullptr;
    uint64_t collectives = 0;              // sequence number of the next collective
    std::vector<uint64_t> sent, received;  // point-to-point messages per peer
    struct PreMulSum {
        ncclDataType_t type;
        bool device;
        double value;        // host immediate scalar
        CUdeviceptr scalar;  // or a device one, read when the operation runs
    };
    std::map<int, PreMulSum> premul;  // custom reduction operators (ncclNumOps and up)
    int nextOp = ncclNumOps;
    std::atomic<bool> aborted{false};
};

namespace {

constexpr uint32_t kIdMagic = 0x4c43434e;  // 'NCCL'

struct UniqueId {
    uint32_t magic;
    uint32_t reserved;
    uint64_t token[2];
};
static_assert(sizeof(UniqueId) <= NCCL_UNIQUE_ID_BYTES);

size_t typeSize(ncclDataType_t t) {
    switch (t) {
    case ncclInt8: case ncclUint8: case ncclFloat8e4m3: case ncclFloat8e5m2: return 1;
    case ncclFloat16: case ncclBfloat16: return 2;
    case ncclInt32: case ncclUint32: case ncclFloat32: return 4;
    case ncclInt64: case ncclUint64: case ncclFloat64: return 8;
    default: return 0;
    }
}

// ---- host arithmetic ---------------------------------------------------------------------------

float halfToFloat(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000) << 16, e = (h >> 10) & 0x1F, m = h & 0x3FF;
    float f;
    if (e == 0x1F) f = m ? NAN : INFINITY;
    else f = e ? std::ldexp(1.0f + float(m) / 1024.0f, int(e) - 15) : std::ldexp(float(m), -24);
    return sign ? -f : f;
}
uint16_t floatToHalf(float f) {  // round to nearest even
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t sign = (u >> 16) & 0x8000, a = u & 0x7FFFFFFF;
    if (a > 0x7F800000) return uint16_t(sign | 0x7E00);
    if (a >= 0x477FF000) return uint16_t(sign | 0x7C00);
    if (a < 0x38800000) return uint16_t(sign | uint32_t(std::nearbyint(std::ldexp(double(std::fabs(f)), 24))));
    const uint32_t x = a - (112u << 23);
    uint32_t h = x >> 13;
    const uint32_t rem = x & 0x1FFF;
    if (rem > 0x1000 || (rem == 0x1000 && (h & 1))) ++h;
    return uint16_t(sign | h);
}
float bf16ToFloat(uint16_t b) {
    const uint32_t u = uint32_t(b) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
uint16_t floatToBf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7FFFFFFF) > 0x7F800000) return 0x7FC0;
    return uint16_t((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}

// Element i of a typed array as the accumulation type (double, or int64/uint64 for integers).
template <class Acc>
Acc load(const void* p, size_t i, ncclDataType_t t) {
    const char* c = static_cast<const char*>(p);
    switch (t) {
    case ncclInt8: return Acc(reinterpret_cast<const int8_t*>(c)[i]);
    case ncclUint8: return Acc(reinterpret_cast<const uint8_t*>(c)[i]);
    case ncclInt32: return Acc(reinterpret_cast<const int32_t*>(c)[i]);
    case ncclUint32: return Acc(reinterpret_cast<const uint32_t*>(c)[i]);
    case ncclInt64: return Acc(reinterpret_cast<const int64_t*>(c)[i]);
    case ncclUint64: return Acc(reinterpret_cast<const uint64_t*>(c)[i]);
    case ncclFloat16: return Acc(halfToFloat(reinterpret_cast<const uint16_t*>(c)[i]));
    case ncclBfloat16: return Acc(bf16ToFloat(reinterpret_cast<const uint16_t*>(c)[i]));
    case ncclFloat32: return Acc(reinterpret_cast<const float*>(c)[i]);
    case ncclFloat64: return Acc(reinterpret_cast<const double*>(c)[i]);
    default: return Acc(0);
    }
}
template <class Acc>
void store(void* p, size_t i, ncclDataType_t t, Acc v) {
    char* c = static_cast<char*>(p);
    switch (t) {
    case ncclInt8: reinterpret_cast<int8_t*>(c)[i] = int8_t(v); break;
    case ncclUint8: reinterpret_cast<uint8_t*>(c)[i] = uint8_t(v); break;
    case ncclInt32: reinterpret_cast<int32_t*>(c)[i] = int32_t(v); break;
    case ncclUint32: reinterpret_cast<uint32_t*>(c)[i] = uint32_t(v); break;
    case ncclInt64: reinterpret_cast<int64_t*>(c)[i] = int64_t(v); break;
    case ncclUint64: reinterpret_cast<uint64_t*>(c)[i] = uint64_t(v); break;
    case ncclFloat16: reinterpret_cast<uint16_t*>(c)[i] = floatToHalf(float(v)); break;
    case ncclBfloat16: reinterpret_cast<uint16_t*>(c)[i] = floatToBf16(float(v)); break;
    case ncclFloat32: reinterpret_cast<float*>(c)[i] = float(v); break;
    case ncclFloat64: reinterpret_cast<double*>(c)[i] = double(v); break;
    default: break;
    }
}

bool isFloat(ncclDataType_t t) { return t == ncclFloat16 || t == ncclBfloat16 || t == ncclFloat32 || t == ncclFloat64; }
bool isSigned(ncclDataType_t t) { return t == ncclInt8 || t == ncclInt32 || t == ncclInt64; }

template <class Acc>
Acc combine(Acc a, Acc b, ncclRedOp_t op) {
    switch (op) {
    case ncclProd: return a * b;
    case ncclMax: return std::max(a, b);
    case ncclMin: return std::min(a, b);
    default: return a + b;  // sum, avg (divided at the end), premultiplied sum
    }
}

// out[i] = the reduction of parts[r][first + i] over r, in rank order.
template <class Acc>
void reduceAs(const std::vector<const void*>& parts, size_t first, size_t count, ncclDataType_t t, ncclRedOp_t op,
              void* out) {
    for (size_t i = 0; i < count; ++i) {
        Acc acc = load<Acc>(parts[0], first + i, t);
        for (size_t r = 1; r < parts.size(); ++r) acc = combine(acc, load<Acc>(parts[r], first + i, t), op);
        if (op == ncclAvg) acc = acc / Acc(parts.size());
        store<Acc>(out, i, t, acc);
    }
}

ncclResult_t reduce(const std::vector<const void*>& parts, size_t first, size_t count, ncclDataType_t t,
                    ncclRedOp_t op, void* out) {
    if (op >= ncclNumOps) op = ncclSum;  // premultiplied sums: the factors were applied when posting
    if (isFloat(t)) reduceAs<double>(parts, first, count, t, op, out);
    else if (isSigned(t)) reduceAs<int64_t>(parts, first, count, t, op, out);
    else if (typeSize(t) && t != ncclFloat8e4m3 && t != ncclFloat8e5m2) reduceAs<uint64_t>(parts, first, count, t, op, out);
    else return ncclInvalidArgument;  // fp8 reductions
    return ncclSuccess;
}

// ---- shared memory objects ---------------------------------------------------------------------

struct alignas(64) Header {
    std::atomic<uint32_t> ready;
    std::atomic<int32_t> readers;  // left to read it; the last one unlinks
    uint64_t bytes;
};
static_assert(std::atomic<uint32_t>::is_always_lock_free && std::atomic<int32_t>::is_always_lock_free);

std::string objectName(const ncclComm* c, const char* kind, uint64_t seq, int a, int b = -1) {
    char name[160];
    std::snprintf(name, sizeof name, "/vknccl-%016llx%016llx-%s%llu-%d-%d", (unsigned long long)c->token[0],
                  (unsigned long long)c->token[1], kind, (unsigned long long)seq, a, b);
    return name;
}

// Publishes `bytes` (filled by `fill`) for `readers` readers.
template <class Fill>
ncclResult_t publish(const std::string& name, size_t bytes, int readers, Fill fill) {
    const int fd = shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC, 0600);
    if (fd < 0) {
        VULKANIZER_WARN("nccl: cannot create %s: %s", name.c_str(), std::strerror(errno));
        return ncclSystemError;
    }
    const size_t size = sizeof(Header) + bytes;
    void* map = ftruncate(fd, off_t(size)) == 0 ? mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)
                                                 : MAP_FAILED;
    close(fd);
    if (map == MAP_FAILED) {
        shm_unlink(name.c_str());
        return ncclSystemError;
    }
    auto* h = static_cast<Header*>(map);
    h->bytes = bytes;
    h->readers.store(readers, std::memory_order_relaxed);
    ncclResult_t r = fill(static_cast<char*>(map) + sizeof(Header));
    h->ready.store(r == ncclSuccess ? 1 : 2, std::memory_order_release);  // 2: the publisher failed
    munmap(map, size);
    return r;
}

// A published object, mapped once it is ready.
struct Object {
    std::string name;
    void* map = nullptr;
    size_t size = 0;
    const void* data() const { return static_cast<const char*>(map) + sizeof(Header); }
    size_t bytes() const { return size - sizeof(Header); }
};

ncclResult_t open(const ncclComm* comm, const std::string& name, Object& out) {
    out.name = name;
    auto wait = std::chrono::microseconds(20);
    for (;;) {
        if (comm->aborted.load()) return ncclRemoteError;
        const int fd = shm_open(name.c_str(), O_RDWR | O_CLOEXEC, 0600);
        if (fd >= 0) {
            struct stat st;
            if (fstat(fd, &st) == 0 && size_t(st.st_size) >= sizeof(Header)) {
                void* map = mmap(nullptr, size_t(st.st_size), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
                if (map != MAP_FAILED) {
                    const uint32_t ready = static_cast<Header*>(map)->ready.load(std::memory_order_acquire);
                    if (ready) {
                        close(fd);
                        out.map = map;
                        out.size = size_t(st.st_size);
                        return ready == 1 ? ncclSuccess : ncclRemoteError;
                    }
                    munmap(map, size_t(st.st_size));
                }
            }
            close(fd);
        }
        std::this_thread::sleep_for(wait);
        wait = std::min(wait * 2, std::chrono::microseconds(2000));
    }
}

void release(Object& o) {
    if (!o.map) return;
    if (static_cast<Header*>(o.map)->readers.fetch_sub(1, std::memory_order_acq_rel) == 1) shm_unlink(o.name.c_str());
    munmap(o.map, o.size);
    o.map = nullptr;
}

// ---- operations --------------------------------------------------------------------------------

enum class Kind { AllReduce, Reduce, Broadcast, AllGather, ReduceScatter, AllToAll, Gather, Send, Recv };

struct Op {
    Kind kind;
    const void* send;
    void* recv;
    size_t count;  // elements: per rank's block for AllGather, ReduceScatter, AllToAll, Gather
    ncclDataType_t type;
    ncclRedOp_t op;
    int root;  // or the peer
    ncclComm_t comm;
    CUstream stream;
    uint64_t seq = 0;
};

// The current context is the communicator's while an operation runs.
struct ContextScope {
    bool pushed = false;
    explicit ContextScope(ncclComm_t c) {
        CUcontext cur = nullptr;
        cuCtxGetCurrent(&cur);
        if (cur != c->ctx) pushed = cuCtxPushCurrent(c->ctx) == CUDA_SUCCESS;
    }
    ~ContextScope() {
        if (pushed) {
            CUcontext dummy;
            cuCtxPopCurrent(&dummy);
        }
    }
};

ncclResult_t cuda(CUresult r) {
    if (r == CUDA_SUCCESS) return ncclSuccess;
    VULKANIZER_WARN("nccl: CUDA call failed (%d)", int(r));
    return ncclUnhandledCudaError;
}

// This rank's contribution, from the device, premultiplied for a premultiplied sum.
ncclResult_t contribution(const Op& o, const void* src, size_t elements, char* dst) {
    const size_t bytes = elements * typeSize(o.type);
    if (bytes) {
        if (ncclResult_t r = cuda(cuMemcpyDtoH(dst, CUdeviceptr(src), bytes))) return r;
    }
    auto it = o.comm->premul.find(int(o.op));
    if (it == o.comm->premul.end() || o.kind == Kind::AllGather || o.kind == Kind::Broadcast) return ncclSuccess;
    double factor = it->second.value;
    if (it->second.device) {
        alignas(8) char scalar[8] = {};
        if (ncclResult_t r = cuda(cuMemcpyDtoH(scalar, it->second.scalar, typeSize(it->second.type)))) return r;
        factor = load<double>(scalar, 0, it->second.type);
    }
    for (size_t i = 0; i < elements; ++i) store<double>(dst, i, o.type, load<double>(dst, i, o.type) * factor);
    return ncclSuccess;
}

// The number of elements each rank publishes, and which ranks read them.
size_t publishedElements(const Op& o) {
    switch (o.kind) {
    case Kind::ReduceScatter:
    case Kind::AllToAll: return o.count * size_t(o.comm->nranks);
    default: return o.count;
    }
}

ncclResult_t post(Op& o) {
    ncclComm_t c = o.comm;
    ContextScope scope(c);
    if (ncclResult_t r = cuda(cuStreamSynchronize(o.stream))) return r;  // the inputs are ready
    const size_t size = typeSize(o.type);
    if (o.kind == Kind::Recv) {
        o.seq = c->received[size_t(o.root)]++;
        return ncclSuccess;
    }
    if (o.kind == Kind::Send) {
        o.seq = c->sent[size_t(o.root)]++;
        return publish(objectName(c, "p", o.seq, c->rank, o.root), o.count * size, 1,
                       [&](char* dst) { return contribution(o, o.send, o.count, dst); });
    }
    o.seq = c->collectives++;
    // who reads this rank's contribution: the root alone (Reduce, Gather), every rank (the rest);
    // of a broadcast only the root's is published
    int readers = c->nranks;
    if (o.kind == Kind::Reduce || o.kind == Kind::Gather) readers = 1;
    if (o.kind == Kind::Broadcast && c->rank != o.root) return ncclSuccess;
    const size_t elements = publishedElements(o);
    return publish(objectName(c, "c", o.seq, c->rank), elements * size, readers,
                   [&](char* dst) { return contribution(o, o.send, elements, dst); });
}

ncclResult_t complete(Op& o) {
    ncclComm_t c = o.comm;
    ContextScope scope(c);
    const size_t size = typeSize(o.type);
    const int n = c->nranks;
    auto upload = [&](const void* data, size_t bytes, void* dst) {
        return bytes ? cuda(cuMemcpyHtoD(CUdeviceptr(dst), data, bytes)) : ncclSuccess;
    };
    if (o.kind == Kind::Send) return ncclSuccess;
    if (o.kind == Kind::Recv) {
        Object obj;
        ncclResult_t r = open(c, objectName(c, "p", o.seq, o.root, c->rank), obj);
        if (r == ncclSuccess) r = upload(obj.data(), std::min(obj.bytes(), o.count * size), o.recv);
        release(obj);
        return r;
    }
    const bool rootOnly = o.kind == Kind::Reduce || o.kind == Kind::Gather;
    if (rootOnly && c->rank != o.root) return ncclSuccess;
    std::vector<Object> objs(static_cast<size_t>(n));
    ncclResult_t r = ncclSuccess;
    const auto finish = [&](ncclResult_t result) {
        for (Object& obj : objs) release(obj);
        return result;
    };
    if (o.kind == Kind::Broadcast) {
        if ((r = open(c, objectName(c, "c", o.seq, o.root), objs[size_t(o.root)]))) return finish(r);
        return finish(upload(objs[size_t(o.root)].data(), o.count * size, o.recv));
    }
    std::vector<const void*> parts(static_cast<size_t>(n));
    for (int k = 0; k < n; ++k) {
        if ((r = open(c, objectName(c, "c", o.seq, k), objs[size_t(k)]))) return finish(r);
        parts[size_t(k)] = objs[size_t(k)].data();
    }
    std::vector<char> out;
    switch (o.kind) {
    case Kind::AllReduce:
    case Kind::Reduce:
        out.resize(o.count * size);
        if ((r = reduce(parts, 0, o.count, o.type, o.op, out.data()))) return finish(r);
        break;
    case Kind::ReduceScatter:  // this rank's block of the reduction
        out.resize(o.count * size);
        if ((r = reduce(parts, size_t(c->rank) * o.count, o.count, o.type, o.op, out.data()))) return finish(r);
        break;
    case Kind::AllGather:
    case Kind::Gather:  // every rank's block, in rank order
        out.resize(size_t(n) * o.count * size);
        for (int k = 0; k < n; ++k) std::memcpy(out.data() + size_t(k) * o.count * size, parts[size_t(k)], o.count * size);
        break;
    case Kind::AllToAll:  // block `rank` of every rank's buffer
        out.resize(size_t(n) * o.count * size);
        for (int k = 0; k < n; ++k)
            std::memcpy(out.data() + size_t(k) * o.count * size,
                        static_cast<const char*>(parts[size_t(k)]) + size_t(c->rank) * o.count * size, o.count * size);
        break;
    default: break;
    }
    return finish(upload(out.data(), out.size(), o.recv));
}

// A communicator of one rank: every collective is a copy, except premultiplied sums.
bool localCopy(const Op& o, ncclResult_t& result) {
    if (o.comm->nranks != 1 || o.kind == Kind::Send || o.kind == Kind::Recv || o.comm->premul.count(int(o.op)))
        return false;
    const size_t bytes = o.count * typeSize(o.type);
    ContextScope scope(o.comm);
    result = (o.send == o.recv || !bytes)
                 ? ncclSuccess
                 : cuda(cuMemcpyDtoDAsync(CUdeviceptr(o.recv), CUdeviceptr(o.send), bytes, o.stream));
    return true;
}

thread_local int g_groupDepth = 0;
thread_local std::vector<Op> g_group;

ncclResult_t run(Op o) {
    if (!o.comm) return ncclInvalidArgument;
    if (!typeSize(o.type)) return ncclInvalidArgument;
    if ((o.kind == Kind::Send || o.kind == Kind::Recv || o.kind == Kind::Reduce || o.kind == Kind::Broadcast ||
         o.kind == Kind::Gather) &&
        (o.root < 0 || o.root >= o.comm->nranks))
        return ncclInvalidArgument;
    if (o.op < 0 || (o.op >= ncclNumOps && !o.comm->premul.count(int(o.op)))) return ncclInvalidArgument;
    if (g_groupDepth > 0) {
        g_group.push_back(o);
        return ncclSuccess;
    }
    ncclResult_t r;
    if (localCopy(o, r)) return r;
    if ((r = post(o))) return r;
    return complete(o);
}

uint64_t randomToken() {
    std::random_device rd;
    return (uint64_t(rd()) << 32) ^ rd() ^ (uint64_t(getpid()) << 16) ^
           uint64_t(std::chrono::steady_clock::now().time_since_epoch().count());
}

ncclResult_t initComm(ncclComm_t* out, int nranks, const uint64_t token[2], int rank) {
    if (!out || nranks < 1 || rank < 0 || rank >= nranks) return ncclInvalidArgument;
    if (cuInit(0) != CUDA_SUCCESS) return ncclUnhandledCudaError;
    auto* c = new ncclComm();
    c->token[0] = token[0];
    c->token[1] = token[1];
    c->nranks = nranks;
    c->rank = rank;
    c->sent.assign(size_t(nranks), 0);
    c->received.assign(size_t(nranks), 0);
    if (cuCtxGetCurrent(&c->ctx) != CUDA_SUCCESS || !c->ctx) {
        // no current context: the first device's primary one, as the runtime would make current
        if (cuDevicePrimaryCtxRetain(&c->ctx, 0) != CUDA_SUCCESS) {
            delete c;
            return ncclUnhandledCudaError;
        }
    }
    {
        ContextScope scope(c);
        cuCtxGetDevice(&c->device);
    }
    *out = c;
    return ncclSuccess;
}

ncclResult_t tokenOf(const ncclUniqueId& id, uint64_t token[2]) {
    UniqueId u;
    std::memcpy(&u, id.internal, sizeof u);
    if (u.magic != kIdMagic) return ncclInvalidArgument;
    token[0] = u.token[0];
    token[1] = u.token[1];
    return ncclSuccess;
}

}  // namespace

extern "C" {

ncclResult_t ncclGetVersion(int* version) {
    if (!version) return ncclInvalidArgument;
    *version = NCCL_VERSION_CODE;
    return ncclSuccess;
}

ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId) {
    if (!uniqueId) return ncclInvalidArgument;
    std::memset(uniqueId, 0, sizeof *uniqueId);
    const UniqueId u{kIdMagic, 0, {randomToken(), randomToken()}};
    std::memcpy(uniqueId->internal, &u, sizeof u);
    return ncclSuccess;
}

ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank) {
    VULKANIZER_TRACE_API();
    uint64_t token[2];
    if (ncclResult_t r = tokenOf(commId, token)) return r;
    return initComm(comm, nranks, token, rank);
}

ncclResult_t ncclCommInitRankConfig(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank, ncclConfig_t*) {
    return ncclCommInitRank(comm, nranks, commId, rank);
}

ncclResult_t ncclCommInitRankScalable(ncclComm_t* newcomm, int nranks, int myrank, int nId, ncclUniqueId* commIds,
                                      ncclConfig_t*) {
    if (nId < 1 || !commIds) return ncclInvalidArgument;
    return ncclCommInitRank(newcomm, nranks, commIds[0], myrank);
}

ncclResult_t ncclCommInitAll(ncclComm_t* comms, int ndev, const int* devlist) {
    VULKANIZER_TRACE_API();
    if (!comms || ndev < 1) return ncclInvalidArgument;
    const uint64_t token[2] = {randomToken(), randomToken()};
    for (int i = 0; i < ndev; ++i) {
        CUcontext ctx = nullptr;
        if (cuInit(0) != CUDA_SUCCESS || cuDevicePrimaryCtxRetain(&ctx, devlist ? devlist[i] : i) != CUDA_SUCCESS)
            return ncclUnhandledCudaError;
        cuCtxPushCurrent(ctx);
        const ncclResult_t r = initComm(&comms[i], ndev, token, i);
        CUcontext dummy;
        cuCtxPopCurrent(&dummy);
        if (r) return r;
    }
    return ncclSuccess;
}

ncclResult_t ncclCommFinalize(ncclComm_t comm) { return comm ? ncclSuccess : ncclInvalidArgument; }

ncclResult_t ncclCommDestroy(ncclComm_t comm) {
    delete comm;
    return ncclSuccess;
}

// Called from another thread (PyTorch's watchdog) while operations may still wait on the
// communicator: they return ncclRemoteError, and the communicator is not freed.
ncclResult_t ncclCommAbort(ncclComm_t comm) {
    if (comm) comm->aborted = true;
    return ncclSuccess;
}

// Collective over `comm`: every rank learns every (color, key); the ranks of one color form a new
// communicator, ordered by key and then by their old rank.
ncclResult_t ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t* newcomm, ncclConfig_t*) {
    VULKANIZER_TRACE_API();
    if (!comm || !newcomm) return ncclInvalidArgument;
    const uint64_t seq = comm->collectives++;
    const int32_t mine[2] = {color, key};
    if (ncclResult_t r = publish(objectName(comm, "s", seq, comm->rank), sizeof mine, comm->nranks, [&](char* dst) {
            std::memcpy(dst, mine, sizeof mine);
            return ncclSuccess;
        }))
        return r;
    std::vector<std::pair<int, int>> members;  // (key, old rank) of this color
    for (int k = 0; k < comm->nranks; ++k) {
        Object obj;
        if (ncclResult_t r = open(comm, objectName(comm, "s", seq, k), obj)) return r;
        int32_t theirs[2];
        std::memcpy(theirs, obj.data(), sizeof theirs);
        release(obj);
        if (theirs[0] == color && color != NCCL_SPLIT_NOCOLOR) members.push_back({theirs[1], k});
    }
    if (color == NCCL_SPLIT_NOCOLOR) {
        *newcomm = nullptr;
        return ncclSuccess;
    }
    std::sort(members.begin(), members.end());
    const int rank = int(std::find_if(members.begin(), members.end(), [&](auto& m) { return m.second == comm->rank; }) -
                         members.begin());
    const uint64_t token[2] = {comm->token[0] ^ (seq * 0x9E3779B97F4A7C15ull),
                               comm->token[1] ^ (uint64_t(uint32_t(color)) * 0xC2B2AE3D27D4EB4Full + 1)};
    ContextScope scope(comm);
    return initComm(newcomm, int(members.size()), token, rank);
}

const char* ncclGetErrorString(ncclResult_t result) {
    switch (result) {
    case ncclSuccess: return "no error";
    case ncclUnhandledCudaError: return "unhandled cuda error (run with NCCL_DEBUG=INFO for details)";
    case ncclSystemError: return "unhandled system error (run with NCCL_DEBUG=INFO for details)";
    case ncclInternalError: return "internal error - please report this issue to the NCCL developers";
    case ncclInvalidArgument: return "invalid argument (run with NCCL_DEBUG=WARN for details)";
    case ncclInvalidUsage: return "invalid usage (run with NCCL_DEBUG=WARN for details)";
    case ncclRemoteError: return "remote process exited or there was a network error";
    case ncclInProgress: return "NCCL operation in progress";
    default: return "unknown result code";
    }
}

const char* ncclGetLastError(ncclComm_t) { return ""; }

ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t* asyncError) {
    if (!comm || !asyncError) return ncclInvalidArgument;
    *asyncError = comm->aborted ? ncclRemoteError : ncclSuccess;
    return ncclSuccess;
}

ncclResult_t ncclCommCount(const ncclComm_t comm, int* count) {
    if (!comm || !count) return ncclInvalidArgument;
    *count = comm->nranks;
    return ncclSuccess;
}

ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int* device) {
    if (!comm || !device) return ncclInvalidArgument;
    *device = comm->device;
    return ncclSuccess;
}

ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank) {
    if (!comm || !rank) return ncclInvalidArgument;
    *rank = comm->rank;
    return ncclSuccess;
}

// Buffer registration only speeds NCCL's transports up: nothing to do.
ncclResult_t ncclCommRegister(const ncclComm_t comm, void* buff, size_t, void** handle) {
    if (!comm || !handle) return ncclInvalidArgument;
    *handle = buff;
    return ncclSuccess;
}
ncclResult_t ncclCommDeregister(const ncclComm_t comm, void*) { return comm ? ncclSuccess : ncclInvalidArgument; }

ncclResult_t ncclMemAlloc(void** ptr, size_t size) {
    if (!ptr) return ncclInvalidArgument;
    CUdeviceptr p = 0;
    if (ncclResult_t r = cuda(cuMemAlloc(&p, size ? size : 1))) return r;
    *ptr = reinterpret_cast<void*>(p);
    return ncclSuccess;
}
ncclResult_t ncclMemFree(void* ptr) { return cuda(cuMemFree(CUdeviceptr(ptr))); }

ncclResult_t ncclRedOpCreatePreMulSum(ncclRedOp_t* op, void* scalar, ncclDataType_t datatype,
                                      ncclScalarResidence_t residence, ncclComm_t comm) {
    if (!op || !scalar || !comm || !typeSize(datatype)) return ncclInvalidArgument;
    ncclComm::PreMulSum p{datatype, residence == ncclScalarDevice, 0, 0};
    if (p.device) p.scalar = CUdeviceptr(scalar);
    else p.value = load<double>(scalar, 0, datatype);
    const int id = comm->nextOp++;
    comm->premul[id] = p;
    *op = ncclRedOp_t(id);
    return ncclSuccess;
}

ncclResult_t ncclRedOpDestroy(ncclRedOp_t op, ncclComm_t comm) {
    if (!comm || !comm->premul.erase(int(op))) return ncclInvalidArgument;
    return ncclSuccess;
}

ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op,
                        int root, ncclComm_t comm, cudaStream_t stream) {
    VULKANIZER_TRACE_API();
    return run({Kind::Reduce, sendbuff, recvbuff, count, datatype, op, root, comm, CUstream(stream)});
}
ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
                           ncclComm_t comm, cudaStream_t stream) {
    VULKANIZER_TRACE_API();
    return run({Kind::Broadcast, sendbuff, recvbuff, count, datatype, ncclSum, root, comm, CUstream(stream)});
}
ncclResult_t ncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm,
                       cudaStream_t stream) {
    return ncclBroadcast(buff, buff, count, datatype, root, comm, stream);
}
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                           ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) {
    VULKANIZER_TRACE_API();
    return run({Kind::AllReduce, sendbuff, recvbuff, count, datatype, op, 0, comm, CUstream(stream)});
}
ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount, ncclDataType_t datatype,
                               ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) {
    VULKANIZER_TRACE_API();
    return run({Kind::ReduceScatter, sendbuff, recvbuff, recvcount, datatype, op, 0, comm, CUstream(stream)});
}
ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype,
                           ncclComm_t comm, cudaStream_t stream) {
    VULKANIZER_TRACE_API();
    return run({Kind::AllGather, sendbuff, recvbuff, sendcount, datatype, ncclSum, 0, comm, CUstream(stream)});
}
ncclResult_t ncclAlltoAll(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                          ncclComm_t comm, cudaStream_t stream) {
    VULKANIZER_TRACE_API();
    return run({Kind::AllToAll, sendbuff, recvbuff, count, datatype, ncclSum, 0, comm, CUstream(stream)});
}
ncclResult_t ncclGather(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
                        ncclComm_t comm, cudaStream_t stream) {
    VULKANIZER_TRACE_API();
    return run({Kind::Gather, sendbuff, recvbuff, count, datatype, ncclSum, root, comm, CUstream(stream)});
}
ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm,
                      cudaStream_t stream) {
    VULKANIZER_TRACE_API();
    return run({Kind::Send, sendbuff, nullptr, count, datatype, ncclSum, peer, comm, CUstream(stream)});
}
ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm,
                      cudaStream_t stream) {
    VULKANIZER_TRACE_API();
    return run({Kind::Recv, nullptr, recvbuff, count, datatype, ncclSum, peer, comm, CUstream(stream)});
}

ncclResult_t ncclGroupStart(void) {
    ++g_groupDepth;
    return ncclSuccess;
}

// Posts every operation of the group, then completes them.
ncclResult_t ncclGroupEnd(void) {
    if (g_groupDepth == 0) return ncclInvalidUsage;
    if (--g_groupDepth > 0) return ncclSuccess;
    std::vector<Op> ops;
    ops.swap(g_group);
    ncclResult_t result = ncclSuccess;
    std::vector<bool> posted(ops.size(), false);
    for (size_t i = 0; i < ops.size(); ++i) {
        ncclResult_t r;
        if (localCopy(ops[i], r)) {
            if (r && !result) result = r;
            continue;
        }
        r = post(ops[i]);
        if (r && !result) result = r;
        posted[i] = r == ncclSuccess;
    }
    for (size_t i = 0; i < ops.size(); ++i) {
        if (!posted[i]) continue;
        const ncclResult_t r = complete(ops[i]);
        if (r && !result) result = r;
    }
    return result;
}

// Estimates a group's time instead of running it: its operations are dropped.
ncclResult_t ncclGroupSimulateEnd(void*) {
    if (g_groupDepth == 0) return ncclInvalidUsage;
    if (--g_groupDepth == 0) g_group.clear();
    return ncclSuccess;
}

}  // extern "C"
