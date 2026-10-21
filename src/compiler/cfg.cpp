#include "compiler/cfg.h"

#include <algorithm>
#include <map>

namespace vulkanizer::compiler {

namespace {

bool isBranch(const ptx::Instruction& ins) { return ins.opcode == "bra"; }
// trap aborts the kernel; here the thread just ends.
bool isReturn(const ptx::Instruction& ins) {
    return ins.opcode == "ret" || ins.opcode == "exit" || ins.opcode == "trap";
}

}  // namespace

bool buildCfg(const ptx::Function& fn, Cfg& out, std::string& error) {
    out = Cfg{};
    const size_t n = fn.instructions.size();
    std::map<std::string_view, size_t> labelAt;
    for (const ptx::Label& l : fn.labels) labelAt.emplace(l.name, l.index);

    std::vector<bool> leader(n + 1, false);
    leader[0] = true;
    for (const ptx::Label& l : fn.labels) leader[l.index] = true;
    for (size_t i = 0; i < n; ++i) {
        const ptx::Instruction& ins = fn.instructions[i];
        if (isBranch(ins) || isReturn(ins)) leader[i + 1] = true;
        if (ins.opcode == "brx" || ins.opcode == "call") {
            error = "line " + std::to_string(ins.line) + ": '" + std::string(ins.opcode) + "' is not supported yet";
            return false;
        }
    }

    // Blocks in source order; a synthetic empty exit block takes the end position.
    std::vector<int32_t> blockAt(n + 1, -1);
    for (size_t i = 0; i < n; ++i) {
        if (!leader[i]) continue;
        blockAt[i] = int32_t(out.blocks.size());
        BasicBlock b;
        b.begin = i;
        out.blocks.push_back(b);
    }
    const uint32_t exitBlock = uint32_t(out.blocks.size());
    blockAt[n] = int32_t(exitBlock);
    {
        BasicBlock b;
        b.begin = b.end = n;
        out.blocks.push_back(b);
    }
    if (n == 0) blockAt[0] = int32_t(exitBlock);
    for (const ptx::Label& l : fn.labels) {
        BasicBlock& b = out.blocks[blockAt[l.index]];
        if (b.label.empty()) b.label = l.name;
    }

    auto targetOf = [&](const ptx::Instruction& ins, uint32_t& block) {
        if (ins.operands.size() != 1 || ins.operands[0].kind != ptx::OperandKind::Symbol) {
            error = "line " + std::to_string(ins.line) + ": malformed branch";
            return false;
        }
        auto it = labelAt.find(ins.operands[0].name);
        if (it == labelAt.end()) {
            error = "line " + std::to_string(ins.line) + ": unknown label '" + std::string(ins.operands[0].name) + "'";
            return false;
        }
        block = uint32_t(blockAt[it->second]);
        return true;
    };

    for (uint32_t bi = 0; bi < exitBlock; ++bi) {
        BasicBlock& b = out.blocks[bi];
        size_t e = b.begin;
        while (e < n && (e == b.begin || !leader[e])) ++e;
        const uint32_t next = uint32_t(blockAt[e]);
        b.end = e;
        b.term = BasicBlock::Term::Branch;
        b.target = next;
        if (e == b.begin) continue;
        const ptx::Instruction& last = fn.instructions[e - 1];
        if (!isBranch(last) && !isReturn(last)) continue;
        b.end = e - 1;
        uint32_t taken = exitBlock;
        if (isBranch(last) && !targetOf(last, taken)) return false;
        if (last.guard.empty()) {
            b.term = isReturn(last) ? BasicBlock::Term::Return : BasicBlock::Term::Branch;
            b.target = taken;
        } else if (taken != next) {
            b.term = BasicBlock::Term::CondBranch;
            b.target = taken;
            b.fallthrough = next;
            b.pred = last.guard;
            b.predNegated = last.guardNegated;
        }
    }
    out.blocks[exitBlock].term = BasicBlock::Term::Return;

    for (uint32_t bi = 0; bi < out.blocks.size(); ++bi) {
        BasicBlock& b = out.blocks[bi];
        if (b.term == BasicBlock::Term::Branch) b.succs = {b.target};
        if (b.term == BasicBlock::Term::CondBranch) b.succs = {b.target, b.fallthrough};
    }

    // Reverse postorder from the entry (iterative DFS).
    const size_t count = out.blocks.size();
    std::vector<uint8_t> state(count, 0);
    std::vector<uint32_t> post;
    std::vector<std::pair<uint32_t, size_t>> stack{{0, 0}};
    state[0] = 1;
    while (!stack.empty()) {
        auto& [b, k] = stack.back();
        const std::vector<uint32_t>& succs = out.blocks[b].succs;
        if (k < succs.size()) {
            const uint32_t s = succs[k++];
            if (!state[s]) {
                state[s] = 1;
                stack.push_back({s, 0});
            }
        } else {
            post.push_back(b);
            stack.pop_back();
        }
    }
    out.rpo.assign(post.rbegin(), post.rend());
    out.rpoIndex.assign(count, -1);
    for (size_t i = 0; i < out.rpo.size(); ++i) out.rpoIndex[out.rpo[i]] = int32_t(i);
    for (uint32_t b : out.rpo)
        for (uint32_t s : out.blocks[b].succs) out.blocks[s].preds.push_back(b);

    // Dominators (Cooper, Harvey, Kennedy: "A Simple, Fast Dominance Algorithm").
    out.idom.assign(count, -1);
    out.idom[0] = 0;
    auto intersect = [&](int32_t a, int32_t b) {
        while (a != b) {
            while (out.rpoIndex[a] > out.rpoIndex[b]) a = out.idom[a];
            while (out.rpoIndex[b] > out.rpoIndex[a]) b = out.idom[b];
        }
        return a;
    };
    for (bool changed = true; changed;) {
        changed = false;
        for (size_t i = 1; i < out.rpo.size(); ++i) {
            const uint32_t b = out.rpo[i];
            int32_t d = -1;
            for (uint32_t p : out.blocks[b].preds) {
                if (out.idom[p] < 0) continue;
                d = d < 0 ? int32_t(p) : intersect(int32_t(p), d);
            }
            if (d != out.idom[b]) {
                out.idom[b] = d;
                changed = true;
            }
        }
    }
    out.idom[0] = -1;
    return true;
}

namespace {

class Structurizer {
public:
    Structurizer(const Cfg& cfg, Structured& out, std::string& error) : cfg_(cfg), out_(out), error_(error) {}

    bool run() {
        const size_t n = cfg_.blocks.size();
        if (!findLoops()) return false;

        forwardPreds_.assign(n, 0);
        loopExit_.assign(n, false);
        for (uint32_t t : cfg_.rpo) {
            for (uint32_t p : cfg_.blocks[t].preds) {
                if (rpo(p) < rpo(t)) ++forwardPreds_[t];
                if (innermost_[p] >= 0 && !inLoop(innermost_[p], t)) loopExit_[t] = true;
            }
        }

        // Each merge node is emitted right after a Block that encloses all its
        // predecessors: below its immediate dominator, but outside every loop
        // that does not contain the node itself.
        inner_.assign(n, {});
        outer_.assign(n, {});
        for (uint32_t t : cfg_.rpo) {
            if (t == 0 || !isMerge(t)) continue;
            uint32_t d = uint32_t(cfg_.idom[t]);
            bool outside = false;
            for (;;) {
                int32_t l = innermost_[d];
                if (outside) l = loopParent_[l];
                if (l < 0 || inLoop(l, t)) break;
                d = loops_[l].header;
                outside = true;
            }
            (outside ? outer_ : inner_)[d].push_back(t);
        }
        auto byRpoDesc = [&](uint32_t a, uint32_t b) { return rpo(a) > rpo(b); };
        for (auto& v : inner_) std::sort(v.begin(), v.end(), byRpoDesc);
        for (auto& v : outer_) std::sort(v.begin(), v.end(), byRpoDesc);

        std::vector<Ctx> ctx;
        return doTree(0, ctx, out_.body);
    }

private:
    struct Loop {
        uint32_t header;
        std::vector<bool> body;
        size_t size = 0;
    };
    struct Ctx {
        bool isLoop;     // LoopHeadedBy(node) or BlockFollowedBy(node)
        uint32_t node;
        uint32_t construct;
    };

    int32_t rpo(uint32_t b) const { return cfg_.rpoIndex[b]; }
    bool inLoop(int32_t l, uint32_t b) const { return loops_[l].body[b]; }
    bool isMerge(uint32_t b) const { return forwardPreds_[b] >= 2 || loopExit_[b]; }
    bool dominates(uint32_t a, uint32_t b) const {
        for (int32_t x = int32_t(b); x >= 0; x = cfg_.idom[x])
            if (uint32_t(x) == a) return true;
        return false;
    }

    bool findLoops() {
        const size_t n = cfg_.blocks.size();
        std::map<uint32_t, size_t> byHeader;
        for (uint32_t u : cfg_.rpo) {
            for (uint32_t h : cfg_.blocks[u].succs) {
                if (rpo(h) > rpo(u)) continue;
                if (!dominates(h, u)) {
                    error_ = "irreducible control flow is not supported yet";
                    return false;
                }
                auto it = byHeader.find(h);
                if (it == byHeader.end()) {
                    it = byHeader.emplace(h, loops_.size()).first;
                    loops_.push_back({h, std::vector<bool>(n, false)});
                    loops_.back().body[h] = true;
                }
                // Natural loop: everything that reaches u without passing through h.
                Loop& l = loops_[it->second];
                std::vector<uint32_t> work{u};
                while (!work.empty()) {
                    const uint32_t b = work.back();
                    work.pop_back();
                    if (l.body[b]) continue;
                    l.body[b] = true;
                    for (uint32_t p : cfg_.blocks[b].preds) work.push_back(p);
                }
            }
        }
        for (Loop& l : loops_) l.size = size_t(std::count(l.body.begin(), l.body.end(), true));

        innermost_.assign(n, -1);
        for (uint32_t b = 0; b < n; ++b)
            for (size_t i = 0; i < loops_.size(); ++i)
                if (loops_[i].body[b] && (innermost_[b] < 0 || loops_[i].size < loops_[innermost_[b]].size))
                    innermost_[b] = int32_t(i);
        loopParent_.assign(loops_.size(), -1);
        for (size_t i = 0; i < loops_.size(); ++i)
            for (size_t j = 0; j < loops_.size(); ++j)
                if (i != j && loops_[j].body[loops_[i].header] && loops_[j].size > loops_[i].size &&
                    (loopParent_[i] < 0 || loops_[j].size < loops_[loopParent_[i]].size))
                    loopParent_[i] = int32_t(j);
        loopIndex_.assign(n, -1);
        for (size_t i = 0; i < loops_.size(); ++i) loopIndex_[loops_[i].header] = int32_t(i);
        return true;
    }

    bool doTree(uint32_t x, std::vector<Ctx>& ctx, std::vector<SNode>& out) {
        if (loopIndex_[x] < 0) return nodeWithin(x, inner_[x], 0, ctx, out);
        return loopWithin(x, 0, ctx, out);
    }

    // Blocks for the loop's exit nodes wrap the Loop construct itself.
    bool loopWithin(uint32_t x, size_t k, std::vector<Ctx>& ctx, std::vector<SNode>& out) {
        const std::vector<uint32_t>& ys = outer_[x];
        if (k < ys.size()) return blockThen(ys[k], ctx, out, [&](std::vector<SNode>& body) {
            return loopWithin(x, k + 1, ctx, body);
        });
        SNode loop;
        loop.kind = SNode::Kind::Loop;
        loop.construct = ++out_.constructCount;
        ctx.push_back({true, x, loop.construct});
        const bool ok = nodeWithin(x, inner_[x], 0, ctx, loop.body);
        ctx.pop_back();
        out.push_back(std::move(loop));
        return ok;
    }

    template <class Inner>
    bool blockThen(uint32_t y, std::vector<Ctx>& ctx, std::vector<SNode>& out, Inner inner) {
        SNode block;
        block.kind = SNode::Kind::Block;
        block.construct = ++out_.constructCount;
        ctx.push_back({false, y, block.construct});
        const bool ok = inner(block.body);
        ctx.pop_back();
        out.push_back(std::move(block));
        return ok && doTree(y, ctx, out);
    }

    bool nodeWithin(uint32_t x, const std::vector<uint32_t>& ys, size_t k, std::vector<Ctx>& ctx,
                    std::vector<SNode>& out) {
        if (k < ys.size()) return blockThen(ys[k], ctx, out, [&](std::vector<SNode>& body) {
            return nodeWithin(x, ys, k + 1, ctx, body);
        });
        const BasicBlock& b = cfg_.blocks[x];
        if (b.end > b.begin) {
            SNode code;
            code.kind = SNode::Kind::Code;
            code.block = x;
            out.push_back(code);
        }
        switch (b.term) {
        case BasicBlock::Term::Return: {
            SNode r;
            r.kind = SNode::Kind::Return;
            out.push_back(r);
            return true;
        }
        case BasicBlock::Term::Branch:
            return doBranch(x, b.target, ctx, out);
        case BasicBlock::Term::CondBranch: {
            SNode s;
            s.kind = SNode::Kind::If;
            s.pred = b.pred;
            s.predNegated = b.predNegated;
            const bool ok = doBranch(x, b.target, ctx, s.body) && doBranch(x, b.fallthrough, ctx, s.elseBody);
            out.push_back(std::move(s));
            return ok;
        }
        }
        return true;
    }

    bool doBranch(uint32_t from, uint32_t to, std::vector<Ctx>& ctx, std::vector<SNode>& out) {
        const bool backward = rpo(to) <= rpo(from);
        if (backward || isMerge(to)) {
            for (auto it = ctx.rbegin(); it != ctx.rend(); ++it) {
                if (it->node == to && it->isLoop == backward) {
                    SNode br;
                    br.kind = SNode::Kind::Br;
                    br.construct = it->construct;
                    out.push_back(br);
                    return true;
                }
            }
            error_ = "internal error: no enclosing construct for branch target";
            return false;
        }
        return doTree(to, ctx, out);
    }

    const Cfg& cfg_;
    Structured& out_;
    std::string& error_;
    std::vector<Loop> loops_;
    std::vector<int32_t> innermost_, loopParent_, loopIndex_;
    std::vector<uint32_t> forwardPreds_;
    std::vector<bool> loopExit_;
    std::vector<std::vector<uint32_t>> inner_, outer_;
};

void dumpNodes(const std::vector<SNode>& nodes, int depth, std::string& s) {
    const std::string pad(size_t(depth) * 2, ' ');
    for (const SNode& n : nodes) {
        switch (n.kind) {
        case SNode::Kind::Code:
            s += pad + "code B" + std::to_string(n.block) + "\n";
            break;
        case SNode::Kind::Return:
            s += pad + "return\n";
            break;
        case SNode::Kind::Br:
            s += pad + "br #" + std::to_string(n.construct) + "\n";
            break;
        case SNode::Kind::Block:
        case SNode::Kind::Loop:
            s += pad + (n.kind == SNode::Kind::Block ? "block #" : "loop #") + std::to_string(n.construct) + "\n";
            dumpNodes(n.body, depth + 1, s);
            break;
        case SNode::Kind::If:
            s += pad + "if " + (n.predNegated ? "!" : "") + std::string(n.pred) + "\n";
            dumpNodes(n.body, depth + 1, s);
            if (!n.elseBody.empty()) {
                s += pad + "else\n";
                dumpNodes(n.elseBody, depth + 1, s);
            }
            break;
        }
    }
}

}  // namespace

bool structurize(const Cfg& cfg, Structured& out, std::string& error) {
    out = Structured{};
    Structurizer s(cfg, out, error);
    return s.run();
}

std::string dump(const ptx::Function& fn, const Cfg& cfg) {
    std::string s;
    for (uint32_t bi : cfg.rpo) {
        const BasicBlock& b = cfg.blocks[bi];
        s += "B" + std::to_string(bi);
        if (!b.label.empty()) (s += " (") += std::string(b.label) + ")";
        if (cfg.idom[bi] >= 0) s += "  idom B" + std::to_string(cfg.idom[bi]);
        s += "\n";
        for (size_t i = b.begin; i < b.end; ++i) s += "    " + ptx::toString(fn.instructions[i]) + "\n";
        switch (b.term) {
        case BasicBlock::Term::Return:
            s += "    -> return\n";
            break;
        case BasicBlock::Term::Branch:
            s += "    -> B" + std::to_string(b.target) + "\n";
            break;
        case BasicBlock::Term::CondBranch:
            s += "    -> " + std::string(b.predNegated ? "!" : "") + std::string(b.pred) + " ? B" +
                 std::to_string(b.target) + " : B" + std::to_string(b.fallthrough) + "\n";
            break;
        }
    }
    return s;
}

std::string dump(const Structured& st) {
    std::string s;
    dumpNodes(st.body, 0, s);
    return s;
}

}  // namespace vulkanizer::compiler
