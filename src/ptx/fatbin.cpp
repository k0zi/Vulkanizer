#include "ptx/fatbin.h"

#include <algorithm>
#include <cstring>
#include <dlfcn.h>

namespace vulkanizer::fatbin {

namespace {

template <class T>
T read(const uint8_t* p) {
    T v;
    std::memcpy(&v, p, sizeof v);
    return v;
}

constexpr size_t kHeaderSize = 16;
constexpr size_t kEntryHeaderMin = 64;
constexpr uint64_t kMaxFatbinSize = uint64_t(1) << 32;

}  // namespace

bool parse(const void* image, std::vector<Entry>& out, std::string& error) {
    out.clear();
    auto* p = static_cast<const uint8_t*>(image);
    if (read<uint32_t>(p) == kWrapperMagic) {
        // { u32 magic; u32 version; const void* data; const void* filename/prelinked list }
        p = read<const uint8_t*>(p + 8);
        if (!p) {
            error = "fatbin wrapper without data";
            return false;
        }
    }
    if (read<uint32_t>(p) != kHeaderMagic) {
        error = "not a fat binary";
        return false;
    }
    const uint16_t headerSize = read<uint16_t>(p + 6);
    const uint64_t fatSize = read<uint64_t>(p + 8);
    if (headerSize < kHeaderSize || fatSize > kMaxFatbinSize) {
        error = "corrupt fat binary header";
        return false;
    }
    const uint8_t* e = p + headerSize;
    const uint8_t* end = e + fatSize;
    while (e + kEntryHeaderMin <= end) {
        Entry entry;
        entry.kind = read<uint16_t>(e);
        const uint32_t entryHeaderSize = read<uint32_t>(e + 4);
        const uint32_t paddedSize = read<uint32_t>(e + 8);
        entry.payloadSize = read<uint32_t>(e + 16);
        entry.smVersion = read<uint32_t>(e + 28);
        entry.flags = read<uint64_t>(e + 40);
        entry.uncompressedSize = read<uint64_t>(e + 56);
        if (entryHeaderSize < kEntryHeaderMin || e + entryHeaderSize + paddedSize > end ||
            entry.payloadSize > paddedSize) {
            error = "corrupt fat binary entry";
            return false;
        }
        if (!entry.payloadSize) entry.payloadSize = paddedSize;
        entry.payload = e + entryHeaderSize;
        out.push_back(entry);
        e += entryHeaderSize + paddedSize;
    }
    return true;
}

long lz4Decompress(const uint8_t* src, size_t srcSize, uint8_t* dst, size_t dstCapacity) {
    const uint8_t* ip = src;
    const uint8_t* const iend = src + srcSize;
    size_t op = 0;
    auto readLength = [&](size_t len) -> long {
        if (len != 15) return long(len);
        for (;;) {
            if (ip >= iend) return -1;
            const uint8_t b = *ip++;
            len += b;
            if (b != 255) return long(len);
        }
    };
    while (ip < iend) {
        const uint8_t token = *ip++;
        const long lit = readLength(token >> 4);
        if (lit < 0 || size_t(iend - ip) < size_t(lit) || dstCapacity - op < size_t(lit)) return -1;
        std::memcpy(dst + op, ip, size_t(lit));
        ip += lit;
        op += size_t(lit);
        if (ip >= iend) break;  // the last sequence has no match
        if (iend - ip < 2) return -1;
        const size_t offset = size_t(ip[0]) | size_t(ip[1]) << 8;
        ip += 2;
        if (offset == 0 || offset > op) return -1;
        long match = readLength(token & 15);
        if (match < 0) return -1;
        match += 4;
        if (dstCapacity - op < size_t(match)) return -1;
        for (long i = 0; i < match; ++i, ++op) dst[op] = dst[op - offset];  // may overlap
    }
    return long(op);
}

// zstd frames are decoded by the system's libzstd, loaded on first use.
static bool zstdDecompress(const uint8_t* src, size_t srcSize, uint8_t* dst, size_t dstCapacity, size_t& out,
                           std::string& error) {
    using Decompress = size_t (*)(void*, size_t, const void*, size_t);
    using IsError = unsigned (*)(size_t);
    using ErrorName = const char* (*)(size_t);
    static void* lib = [] {
        void* h = dlopen("libzstd.so.1", RTLD_NOW | RTLD_LOCAL);
        return h ? h : dlopen("libzstd.so", RTLD_NOW | RTLD_LOCAL);
    }();
    if (!lib) {
        error = "zstd-compressed fat binary entry, but libzstd.so.1 is not available";
        return false;
    }
    auto decompress = reinterpret_cast<Decompress>(dlsym(lib, "ZSTD_decompress"));
    auto isError = reinterpret_cast<IsError>(dlsym(lib, "ZSTD_isError"));
    auto errorName = reinterpret_cast<ErrorName>(dlsym(lib, "ZSTD_getErrorName"));
    if (!decompress || !isError) {
        error = "libzstd.so.1 lacks ZSTD_decompress";
        return false;
    }
    const size_t n = decompress(dst, dstCapacity, src, srcSize);
    if (isError(n)) {
        error = std::string("corrupt zstd data in fat binary: ") + (errorName ? errorName(n) : "?");
        return false;
    }
    out = n;
    return true;
}

std::string nativeTargets(const std::vector<Entry>& entries) {
    std::vector<uint32_t> sms;
    for (const Entry& e : entries)
        if (e.kind == uint16_t(Kind::Elf) && std::find(sms.begin(), sms.end(), e.smVersion) == sms.end())
            sms.push_back(e.smVersion);
    std::sort(sms.begin(), sms.end());
    std::string r;
    for (uint32_t sm : sms) r += (r.empty() ? "sm_" : ", sm_") + std::to_string(sm);
    return r.empty() ? "no architecture" : r;
}

bool extractPtx(const void* image, uint32_t cc, std::string& ptx, uint32_t& smVersion, std::string& error) {
    std::vector<Entry> entries;
    if (!parse(image, entries, error)) return false;
    const Entry* best = nullptr;
    const Entry* lowest = nullptr;
    for (const Entry& e : entries) {
        if (e.kind != uint16_t(Kind::Ptx)) continue;
        if (!lowest || e.smVersion < lowest->smVersion) lowest = &e;
        if (e.smVersion <= cc && (!best || e.smVersion > best->smVersion)) best = &e;
    }
    if (!best) best = lowest;
    if (!best) {
        error = "fat binary contains no PTX, only native code (SASS) for " + nativeTargets(entries) +
                ", which cannot be translated; rebuild with a virtual architecture (e.g. nvcc -arch=compute_61)";
        return false;
    }
    smVersion = best->smVersion;
    if (best->zstd()) {
        if (!best->uncompressedSize || best->uncompressedSize > kMaxFatbinSize) {
            error = "corrupt compressed fat binary entry";
            return false;
        }
        ptx.assign(size_t(best->uncompressedSize), '\0');
        size_t n = 0;
        if (!zstdDecompress(best->payload, best->payloadSize, reinterpret_cast<uint8_t*>(ptx.data()), ptx.size(), n,
                            error))
            return false;
        ptx.resize(n);
    } else if (best->lz4()) {
        if (!best->uncompressedSize || best->uncompressedSize > kMaxFatbinSize) {
            error = "corrupt compressed fat binary entry";
            return false;
        }
        ptx.assign(size_t(best->uncompressedSize), '\0');
        const long n = lz4Decompress(best->payload, best->payloadSize, reinterpret_cast<uint8_t*>(ptx.data()),
                                     ptx.size());
        if (n < 0) {
            error = "corrupt LZ4 data in fat binary";
            return false;
        }
        ptx.resize(size_t(n));
    } else {
        ptx.assign(reinterpret_cast<const char*>(best->payload), best->payloadSize);
    }
    if (const size_t nul = ptx.find('\0'); nul != std::string::npos) ptx.resize(nul);
    return true;
}

}  // namespace vulkanizer::fatbin
