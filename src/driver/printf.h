#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

namespace vulkanizer::driver {

// Formats one device printf record. `args` holds the argument block as CUDA lays it out:
// each argument aligned to its own size, floats promoted to double. `readString` fetches a
// device string for %s.
std::string formatDevicePrintf(const std::string& format, const uint8_t* args, size_t argBytes,
                               const std::function<std::string(uint64_t)>& readString);

}  // namespace vulkanizer::driver
