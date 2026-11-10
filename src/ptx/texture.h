#pragma once

#include <string>

namespace vulkanizer::ptx {

// Textures and surfaces in software: rewrites a module's tex, tld4, txq, suld, sust and suq
// instructions into calls of Vulkanizer's texture functions (texture_lib.cu, on the descriptors
// texture and surface object handles point to) and appends those functions to the module. Each
// instruction becomes a scoped block on its own line, so line numbers stay. Modules without such
// instructions are left as they are. False (with `error`) for forms that are not supported:
// texture references and formatted surface stores (sust.p).
bool rewriteTextureInstructions(std::string& source, std::string& error);

}  // namespace vulkanizer::ptx
