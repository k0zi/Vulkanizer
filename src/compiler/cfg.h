#pragma once

#include "ptx/instr.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace vulkanizer::compiler {

// Basic-block graph of a PTX function. Branch and return instructions are
// removed from the blocks and become terminators.
struct BasicBlock {
    enum class Term { Branch, CondBranch, Return };
    size_t begin = 0, end = 0;  // instruction range [begin, end), terminators excluded
    Term term = Term::Return;
    uint32_t target = 0;        // Branch target, CondBranch "taken" target
    uint32_t fallthrough = 0;   // CondBranch "not taken" target
    std::string_view pred;      // CondBranch: taken when pred != predNegated
    bool predNegated = false;
    std::vector<uint32_t> succs, preds;
    std::string_view label;     // first label naming the block, for dumps
};

struct Cfg {
    std::vector<BasicBlock> blocks;  // block 0 is the entry
    std::vector<uint32_t> rpo;       // reachable blocks in reverse postorder
    std::vector<int32_t> rpoIndex;   // -1 for unreachable blocks
    std::vector<int32_t> idom;       // immediate dominator, -1 for the entry and unreachable blocks
};

bool buildCfg(const ptx::Function& fn, Cfg& out, std::string& error);

// Structured form of a reducible CFG (after Ramsey, "Beyond Relooper", 2022):
// Block and Loop are labelled constructs, Br jumps to the end of a Block or to
// the start of a Loop. Code runs a basic block's instructions (not its terminator).
struct SNode {
    enum class Kind { Code, If, Block, Loop, Br, Return };
    Kind kind = Kind::Code;
    uint32_t block = 0;      // Code
    uint32_t construct = 0;  // Block/Loop: own id; Br: target id
    std::string_view pred;   // If: condition register
    bool predNegated = false;
    std::vector<SNode> body;      // Block/Loop body, If "then"
    std::vector<SNode> elseBody;  // If "else"
};

struct Structured {
    std::vector<SNode> body;
    uint32_t constructCount = 0;  // construct ids are 1..constructCount
};

bool structurize(const Cfg& cfg, Structured& out, std::string& error);

std::string dump(const ptx::Function& fn, const Cfg& cfg);
std::string dump(const Structured& s);

}  // namespace vulkanizer::compiler
