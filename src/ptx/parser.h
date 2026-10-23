#pragma once

#include "ptx/lexer.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace vulkanizer::ptx {

// One kernel parameter, laid out exactly as in the CUDA parameter buffer.
struct Param {
    std::string name;
    uint32_t size = 0;
    uint32_t align = 0;
    uint32_t offset = 0;
};

struct Entry {
    std::string name;
    std::vector<Param> params;
    uint32_t paramBytes = 0;  // total size of the parameter buffer
    uint32_t maxntid[3] = {0, 0, 0};
    uint32_t reqntid[3] = {0, 0, 0};
    // The body's source text without the braces, tokenized only when the kernel is compiled
    // (modules from libraries are megabytes of PTX, most of it never needed).
    std::string_view body;
    uint32_t bodyLine = 0;  // line of body[0]
};

// Module-scope variable (.global / .const / .shared).
struct GlobalVar {
    std::string name;
    std::string space;  // ".global", ".const", ".shared"
    uint64_t size = 0;  // 0 for unsized extern arrays
    uint32_t align = 0;
    uint32_t elemSize = 0;
    bool isExtern = false;
    uint64_t offset = 0;        // .global/.const: offset in the module's variable buffer
    std::vector<uint8_t> init;  // initializer bytes (the rest is zero)
    // initializer elements naming functions (vtables, function pointer tables): byte offset and
    // name; parseModule writes the function's id (Module::functionId) there
    std::vector<std::pair<uint64_t, std::string>> functionRefs;
};

// The value a function's address has (mov of its name, initializers): an id, not an address. The
// inliner turns indirect calls into comparisons with the ids of the candidates.
constexpr uint64_t kFunctionIdBase = uint64_t(0xF0) << 56;

// Device function (.func). Functions are inlined at their call sites (ptx/instr.cpp).
struct DeviceFunction {
    std::string name;
    std::vector<Param> params;   // formal parameters
    std::vector<Param> returns;  // formal return values
    std::string_view body;       // as for Entry
    uint32_t bodyLine = 0;
    bool defined = false;        // false for declarations (.extern, prototypes)
};

struct Module {
    std::string version;  // e.g. "7.8"
    uint32_t versionMajor = 0;
    uint32_t versionMinor = 0;
    std::vector<std::string> targets;  // e.g. {"sm_52"}
    uint32_t addressSize = 64;
    std::vector<Entry> entries;
    std::vector<DeviceFunction> functions;
    std::vector<GlobalVar> globals;
    uint64_t globalsSize = 0;  // bytes of the .global/.const variable buffer

    const Entry* findEntry(std::string_view name) const;
    const GlobalVar* findGlobal(std::string_view name) const;
    const DeviceFunction* findFunction(std::string_view name) const;
    // kFunctionIdBase | (1 + the index of the name among the module's distinct function names), 0 if
    // `name` is not a function.
    uint64_t functionId(std::string_view name) const;
};

// Size in bytes of a PTX fundamental type directive (".u32" -> 4), 0 if `d` is not a type.
uint32_t typeSize(std::string_view d);

// Parses module structure: header directives, kernel signatures and module variables.
// Instruction bodies are kept as token ranges for the compiler. `source` must outlive `out`.
bool parseModule(std::string_view source, Module& out, ParseError& err);

}  // namespace vulkanizer::ptx
