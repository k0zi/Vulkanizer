// Fat binaries: PTX selection, LZ4-compressed entries, the __fatBinC_Wrapper_t form and
// cubin-only images. The images are built here from data/vector_add.ptx.

#include "test_util.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

struct Wrapper {  // __fatBinC_Wrapper_t
    uint32_t magic;
    uint32_t version;
    const void* data;
    const void* filename;
};

using Bytes = std::string;

template <typename T>
void put(Bytes& out, T v) {  // little-endian like the fat binary format
    out.append(reinterpret_cast<const char*>(&v), sizeof v);
}

// LZ4 block of `in` (greedy, 4-byte hash matches; the last 5 bytes stay literals, and no match
// starts in the last 12, as the format requires).
Bytes lz4Compress(const Bytes& in) {
    auto length = [](Bytes& out, size_t len) {
        for (len -= 15; len >= 255; len -= 255) out += char(255);
        out += char(len);
    };
    auto read4 = [&](size_t i) {
        uint32_t v;
        std::memcpy(&v, in.data() + i, 4);
        return v;
    };
    Bytes out;
    std::vector<size_t> table(1 << 12, SIZE_MAX);
    const size_t n = in.size();
    size_t anchor = 0, i = 0;
    while (n >= 12 && i + 12 <= n) {
        const uint32_t h = (read4(i) * 2654435761u) >> 20;
        const size_t cand = table[h];
        table[h] = i;
        if (cand == SIZE_MAX || i - cand > 65535 || read4(cand) != read4(i)) {
            ++i;
            continue;
        }
        size_t len = 4;
        while (i + len < n - 5 && in[cand + len] == in[i + len]) ++len;
        const size_t lit = i - anchor;
        out += char((std::min<size_t>(lit, 15) << 4) | std::min<size_t>(len - 4, 15));
        if (lit >= 15) length(out, lit);
        out.append(in, anchor, lit);
        put<uint16_t>(out, uint16_t(i - cand));
        if (len - 4 >= 15) length(out, len - 4);
        i += len;
        anchor = i;
    }
    const size_t lit = n - anchor;
    out += char(std::min<size_t>(lit, 15) << 4);
    if (lit >= 15) length(out, lit);
    out.append(in, anchor, lit);
    return out;
}

constexpr uint64_t kFlag64Bit = 0x1, kFlagLinux = 0x10, kFlagLz4 = 0x2000;

// Entry: u16 kind (1 PTX, 2 ELF), u16 version, u32 header size (64), u32 padded payload size,
// u32 0, u32 payload size, u32 0, u32 0, u32 sm version, u32 address bits, u32 0, u64 flags,
// u64 0, u64 uncompressed size; the payload follows, padded to 8 bytes.
Bytes entry(uint16_t kind, uint32_t sm, const Bytes& payload, uint64_t flags = kFlag64Bit | kFlagLinux,
            uint64_t uncompressed = 0) {
    const Bytes padded = payload + Bytes((8 - payload.size() % 8) % 8, '\0');
    Bytes e;
    put<uint16_t>(e, kind);
    put<uint16_t>(e, 0x0101);
    for (uint32_t v : {64u, uint32_t(padded.size()), 0u, uint32_t(payload.size()), 0u, 0u, sm, 64u, 0u}) put(e, v);
    for (uint64_t v : {flags, uint64_t(0), uncompressed}) put(e, v);
    return e + padded;
}

// Header: u32 magic, u16 version 1, u16 header size 16, u64 size of all entries. 8-byte aligned
// like nvcc's embedded data.
std::vector<uint64_t> fatbin(std::initializer_list<Bytes> entries) {
    Bytes body;
    for (const Bytes& e : entries) body += e;
    Bytes bytes;
    put<uint32_t>(bytes, 0xBA55ED50);
    put<uint16_t>(bytes, 1);
    put<uint16_t>(bytes, 16);
    put<uint64_t>(bytes, body.size());
    bytes += body;
    std::vector<uint64_t> image((bytes.size() + 7) / 8);
    std::memcpy(image.data(), bytes.data(), bytes.size());
    return image;
}

// Loads vecAdd from the module and checks a small launch.
void runVecAdd(CUmodule mod) {
    CUfunction fn;
    CHECK_CU(cuModuleGetFunction(&fn, mod, "vecAdd"));
    int n = 1000;
    std::vector<float> a(n), b(n), c(n);
    for (int i = 0; i < n; ++i) a[i] = float(i), b[i] = 2.0f;
    CUdeviceptr da, db, dc;
    CHECK_CU(cuMemAlloc(&da, n * sizeof(float)));
    CHECK_CU(cuMemAlloc(&db, n * sizeof(float)));
    CHECK_CU(cuMemAlloc(&dc, n * sizeof(float)));
    CHECK_CU(cuMemcpyHtoD(da, a.data(), n * sizeof(float)));
    CHECK_CU(cuMemcpyHtoD(db, b.data(), n * sizeof(float)));
    void* args[] = {&da, &db, &dc, &n};
    CHECK_CU(cuLaunchKernel(fn, (n + 127) / 128, 1, 1, 128, 1, 1, 0, nullptr, args, nullptr));
    CHECK_CU(cuMemcpyDtoH(c.data(), dc, n * sizeof(float)));
    for (int i = 0; i < n; ++i) EXPECT(c[i] == float(i) + 2.0f);
    CHECK_CU(cuMemFree(da));
    CHECK_CU(cuMemFree(db));
    CHECK_CU(cuMemFree(dc));
}

}  // namespace

int main() {
    CHECK_CU(cuInit(0));
    CUdevice dev;
    CHECK_CU(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CHECK_CU(cuCtxCreate(&ctx, 0, dev));
    CUmodule mod;

    const Bytes ptx = readTextFile(VULKANIZER_TEST_DATA_DIR "/vector_add.ptx") + Bytes(1, '\0');
    // a newer PTX that must not be picked for the default compute capability 7.5
    const Bytes newer = Bytes(".version 8.0\n.target sm_90\n.address_size 64\n"
                              ".visible .entry vecAdd(.param .u64 a, .param .u64 b, .param .u64 c, .param .u32 n)\n"
                              "{ .reg .b32 %r<2>; vulkanizer_wrong_ptx_selected.b32 %r1, 0; ret; }\n") + Bytes(1, '\0');
    const Bytes cubin = "\x7f" "ELF" + Bytes(60, '\0');
    const Bytes compressed = lz4Compress(ptx);
    EXPECT(compressed.size() < ptx.size() * 3 / 4);  // real matches, not just literals

    // cubin + LZ4-compressed sm_52 PTX + sm_90 PTX: the sm_52 one must be picked for CC 7.5
    const std::vector<uint64_t> lz4 = fatbin({entry(2, 75, cubin),
                                              entry(1, 52, compressed, kFlag64Bit | kFlagLinux | kFlagLz4, ptx.size()),
                                              entry(1, 90, newer)});
    CHECK_CU(cuModuleLoadData(&mod, lz4.data()));
    runVecAdd(mod);
    CHECK_CU(cuModuleUnload(mod));

    // the same through the wrapper struct, as __cudaRegisterFatBinary receives it
    const Wrapper wrapper{0x466243B1, 1, lz4.data(), nullptr};
    CHECK_CU(cuModuleLoadFatBinary(&mod, &wrapper));
    runVecAdd(mod);
    CHECK_CU(cuModuleUnload(mod));

    // uncompressed entries, newer one first
    const std::vector<uint64_t> plain = fatbin({entry(1, 90, newer), entry(1, 52, ptx)});
    CHECK_CU(cuModuleLoadData(&mod, plain.data()));
    runVecAdd(mod);
    CHECK_CU(cuModuleUnload(mod));

    // only native code: nothing to translate
    const std::vector<uint64_t> cubins = fatbin({entry(2, 75, cubin), entry(2, 86, cubin)});
    EXPECT_CU(CUDA_ERROR_NO_BINARY_FOR_GPU, cuModuleLoadData(&mod, cubins.data()));

    CHECK_CU(cuCtxDestroy(ctx));
    std::printf("OK\n");
    return 0;
}
