#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vulkanizer::compiler {

// Runs the external `spirv-val` (SPIRV-Tools) on a module. Returns false with the validator's
// output in `log` when the module is invalid or the tool cannot be run.
bool validateWithSpirvVal(const std::vector<uint32_t>& spirv, std::string& log);

// Writes the module to `<dir>/<name>.spv`. Returns false on I/O errors.
bool writeSpirv(const std::string& dir, const std::string& name, const std::vector<uint32_t>& spirv);

}  // namespace vulkanizer::compiler
