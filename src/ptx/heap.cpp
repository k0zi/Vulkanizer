#include "ptx/heap.h"

#include <regex>
#include <string_view>

namespace vulkanizer::ptx {

namespace {

// The heap functions, PTX generated from heap_lib.cu (tools/gen_heap_lib.sh).
const char kHeapLibrary[] =
#include "heap_lib.ptx.inc"
    ;

}  // namespace

void appendDeviceHeap(std::string& source) {
    if (source.find("malloc") == std::string::npos && source.find("free") == std::string::npos) return;
    static const std::regex decl(R"(\.extern\s+\.func\s*(\([^)]*\)\s*)?(malloc|free)\s*\()");
    if (!std::regex_search(source, decl)) return;
    // the library's variable and functions, without its header (.version, .target, .address_size)
    std::string_view lib = kHeapLibrary;
    const size_t first = lib.find(".visible");
    source += "\n";
    source += lib.substr(first == std::string_view::npos ? 0 : first);
}

}  // namespace vulkanizer::ptx
