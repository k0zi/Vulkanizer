#pragma once

#include "ptx/parser.h"

#include <cstdint>
#include <deque>
#include <string>
#include <string_view>
#include <vector>

namespace vulkanizer::ptx {

enum class OperandKind {
    Register,   // %r1 (name)
    Special,    // %tid.x (name "%tid", component 'x'), %laneid
    Immediate,  // 42, -1, 0xFF (imm holds the two's complement value)
    FloatImm,   // 0f3F800000, 1.5 (imm holds raw bits of floatWidth)
    Symbol,     // label or variable/parameter name (name)
    Address,    // [%rd1+8], [param_0], [sym+4] (name = base register or symbol, may be empty; offset)
    Vector,     // {%f1, %f2} (elems)
    Sink,       // _
};

struct Operand {
    OperandKind kind = OperandKind::Register;
    std::string_view name;
    std::string_view pairName;  // second destination of "%p|%q"
    char component = 0;         // Special: 'x', 'y', 'z' or 0
    bool negated = false;       // "!%p"
    bool baseIsRegister = false;  // Address: base is a register (otherwise a symbol or absolute)
    uint32_t floatWidth = 0;    // FloatImm: 32 or 64
    uint64_t imm = 0;
    int64_t offset = 0;         // Address
    std::vector<Operand> elems; // Vector
};

struct Instruction {
    std::string_view opcode;              // "ld"
    std::vector<std::string_view> mods;   // {".global", ".f32"}, in source order
    std::string_view guard;               // "@%p1" -> "%p1"
    bool guardNegated = false;            // "@!%p1"
    std::vector<Operand> operands;
    uint32_t line = 0;

    bool has(std::string_view mod) const;
};

struct RegDecl {
    std::string name;      // "%r3" (parameterized declarations are expanded)
    std::string_view type; // ".b32", ".pred", ...
};

struct Label {
    std::string_view name;
    size_t index;  // index of the instruction that follows the label
};

// Variables declared inside a kernel body (.local, .shared). Phase 3 gives them storage.
struct LocalVar {
    std::string_view name;
    std::string_view space;
    uint64_t size = 0;
    uint32_t align = 0;
    uint32_t line = 0;
};

struct Function {
    std::vector<RegDecl> regs;
    std::vector<Instruction> instructions;
    std::vector<Label> labels;
    std::vector<LocalVar> locals;  // .local, .shared and call-site .param variables
    std::deque<std::string> names;  // storage for names created by scoping and inlining (stable addresses)
};

// Parses the instruction stream of `entry`. Nested { } scopes are flattened (registers they
// redeclare are renamed) and calls to
// device functions (.func) are inlined; their .param arguments become local variables.
bool parseFunctionBody(const Module& module, const Entry& entry, Function& out, ParseError& err);

// Human-readable form of an instruction, for dumps and error messages.
std::string toString(const Instruction& ins);

}  // namespace vulkanizer::ptx
