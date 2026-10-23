#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// NVIDIA fat binaries: the container nvcc embeds in host executables. Layout as documented by
// open-source tools that read it (cuobjdump-compatible readers, ZLUDA): a 16-byte header
// followed by entries, each a header plus a PTX or ELF (cubin) payload, optionally compressed.
namespace vulkanizer::fatbin {

constexpr uint32_t kWrapperMagic = 0x466243B1;  // __fatBinC_Wrapper_t
constexpr uint32_t kHeaderMagic = 0xBA55ED50;

enum class Kind : uint16_t { Ptx = 1, Elf = 2 };

struct Entry {
    uint16_t kind = 0;
    uint32_t smVersion = 0;  // e.g. 52 for sm_52 / compute_52
    uint64_t flags = 0;
    const uint8_t* payload = nullptr;
    uint32_t payloadSize = 0;        // bytes stored (compressed size when compressed)
    uint64_t uncompressedSize = 0;   // 0 when not compressed
    bool lz4() const { return flags & 0x2000; }
    bool zstd() const { return flags & 0x8000; }
};

// Accepts a wrapper (0x466243B1) or a fat binary header (0xBA55ED50).
bool parse(const void* image, std::vector<Entry>& out, std::string& error);

// "sm_75, sm_86": the architectures of the native (cubin) entries, for error messages.
std::string nativeTargets(const std::vector<Entry>& entries);

// Picks the PTX entry with the highest sm_XX <= `cc` (or the lowest one if all are newer)
// and returns its text.
bool extractPtx(const void* image, uint32_t cc, std::string& ptx, uint32_t& smVersion, std::string& error);

// LZ4 block format. Returns the number of bytes written, or -1 on malformed input.
long lz4Decompress(const uint8_t* src, size_t srcSize, uint8_t* dst, size_t dstCapacity);

}  // namespace vulkanizer::fatbin
