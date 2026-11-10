#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace vulkanizer {

// SHA-256 (FIPS 180-4), for content-addressed caches.
class Sha256 {
public:
    Sha256();
    void update(const void* data, size_t size);
    void update(std::string_view s) { update(s.data(), s.size()); }
    std::string hexDigest();  // finalizes

private:
    void block(const uint8_t* p);
    uint32_t h_[8];
    uint8_t buf_[64];
    size_t used_ = 0;
    uint64_t total_ = 0;
};

}  // namespace vulkanizer
