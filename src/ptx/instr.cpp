#include "ptx/instr.h"

#include "common/log.h"

#include <algorithm>
#include <cstdio>
#include <unordered_map>
#include <unordered_set>

namespace vulkanizer::ptx {

bool Instruction::has(std::string_view mod) const {
    return std::find(mods.begin(), mods.end(), mod) != mods.end();
}

static bool isSpecialRegister(std::string_view name) {
    static constexpr std::string_view names[] = {
        "%tid",         "%ntid",        "%ctaid",       "%nctaid",      "%laneid",      "%warpid",
        "%nwarpid",     "%smid",        "%nsmid",       "%gridid",      "%lanemask_eq", "%lanemask_le",
        "%lanemask_lt", "%lanemask_ge", "%lanemask_gt", "%clock",       "%clock64",     "%globaltimer",
        "%globaltimer_lo", "%globaltimer_hi", "%dynamic_smem_size", "%total_smem_size", "%warpsz",
    };
    if (name.starts_with("%envreg")) return true;  // %envreg0..31
    return std::find(std::begin(names), std::end(names), name) != std::end(names);
}

namespace {

class BodyParser {
public:
    BodyParser(const std::vector<Token>& tokens, size_t begin, size_t end, Function& out, ParseError& err)
        : t_(tokens), pos_(begin), end_(end), f_(out), err_(err) {}

    bool run() {
        while (pos_ < end_) {
            if (!statement()) return false;
        }
        return true;
    }

private:
    const Token& peek(size_t k = 0) const { return pos_ + k < end_ ? t_[pos_ + k] : t_[end_]; }
    const Token& next() {
        const Token& tk = peek();
        if (pos_ < end_) ++pos_;
        return tk;
    }
    static bool isPunct(const Token& tk, char c) {
        return tk.kind == TokenKind::Punct && tk.text.size() == 1 && tk.text[0] == c;
    }
    bool fail(const Token& at, std::string msg) {
        err_ = {std::move(msg), at.line};
        return false;
    }
    bool expectPunct(char c) {
        const Token& tk = next();
        if (!isPunct(tk, c)) return fail(tk, std::string("expected '") + c + "'");
        return true;
    }
    bool atEnd() const { return pos_ >= end_; }

    bool statement() {
        const Token& tk = peek();
        if (isPunct(tk, '{')) {  // nested scope: its registers may shadow the enclosing ones
            next();
            scopes_.emplace_back();
            return true;
        }
        if (isPunct(tk, '}')) {
            next();
            if (!scopes_.empty()) scopes_.pop_back();
            return true;
        }
        if (tk.kind == TokenKind::Directive) return directive();
        if (tk.kind == TokenKind::Identifier && isPunct(peek(1), ':')) {
            f_.labels.push_back({tk.text, f_.instructions.size()});
            next();
            next();
            return true;
        }
        return instruction();
    }

    bool directive() {
        const Token& d = peek();
        if (d.text == ".reg") return regDecl();
        if (d.text == ".local" || d.text == ".shared" || d.text == ".param") return localDecl();
        if (d.text == ".loc" || d.text == ".file") {
            const uint32_t line = next().line;
            while (!atEnd() && peek().line == line) next();
            return true;
        }
        if (d.text == ".pragma") return skipToSemicolon();
        return fail(d, "unsupported directive in kernel body '" + std::string(d.text) + "'");
    }

    // .reg .b32 %r<6>, %x;   .reg .v4 .f32 %v;  (vector registers are not supported yet)
    bool regDecl() {
        next();
        std::string_view type;
        while (peek().kind == TokenKind::Directive) {
            const Token& d = next();
            if (d.text == ".pred" || typeSize(d.text)) type = d.text;
            else if (d.text == ".v2" || d.text == ".v4") return fail(d, "vector registers are not supported yet");
            else if (d.text == ".align") next();
            else return fail(d, "unsupported register qualifier '" + std::string(d.text) + "'");
        }
        if (type.empty()) return fail(peek(), "register declaration without a type");
        for (;;) {
            const Token& name = next();
            if (name.kind != TokenKind::Register && name.kind != TokenKind::Identifier)
                return fail(name, "expected register name");
            if (isPunct(peek(), '<')) {
                next();
                uint64_t count;
                const Token& n = next();
                if (n.kind != TokenKind::Integer || !parseInteger(n.text, count)) return fail(n, "expected count");
                if (!expectPunct('>')) return false;
                for (uint64_t i = 0; i < count; ++i) addReg(std::string(name.text) + std::to_string(i), type);
            } else {
                addReg(std::string(name.text), type);
            }
            if (!isPunct(peek(), ',')) break;
            next();
        }
        return expectPunct(';');
    }

    // Scopes are flattened into one register set. A register that a nested scope redeclares
    // gets a unique name there (the enclosing one stays live across the scope: CUDA's math
    // library declares %r0..%r3 in inline asm blocks of functions that use %r1..%r3 around them).
    void addReg(std::string name, std::string_view type) {
        if (!scopes_.empty() && regNames_.count(name)) {
            f_.names.push_back("$scope" + std::to_string(scopeNames_++) + "_" + name);
            const std::string_view fresh = f_.names.back();
            scopes_.back()[std::move(name)] = fresh;
            regNames_.insert(std::string(fresh));
            f_.regs.push_back({std::string(fresh), type});
            return;
        }
        if (regNames_.insert(name).second) f_.regs.push_back({std::move(name), type});
    }

    // The register a name refers to in the current scope.
    std::string_view resolve(std::string_view name) const {
        if (scopes_.empty()) return name;
        const std::string key(name);
        for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it)
            if (auto found = it->find(key); found != it->end()) return found->second;
        return name;
    }

    bool localDecl() {
        LocalVar v;
        v.line = peek().line;
        v.space = next().text;
        uint32_t elemSize = 0, vec = 1;
        while (peek().kind == TokenKind::Directive) {
            const Token& d = next();
            if (d.text == ".align") {
                uint64_t a;
                if (!parseInteger(next().text, a)) return fail(d, "invalid .align");
                v.align = uint32_t(a);
            } else if (d.text == ".v2") {
                vec = 2;
            } else if (d.text == ".v4") {
                vec = 4;
            } else if (uint32_t sz = typeSize(d.text)) {
                elemSize = sz;
            }
        }
        const Token& name = next();
        if (name.kind != TokenKind::Identifier) return fail(name, "expected variable name");
        v.name = name.text;
        v.size = uint64_t(elemSize) * vec;
        while (isPunct(peek(), '[')) {
            next();
            uint64_t dim = 0;
            if (!isPunct(peek(), ']') && !parseInteger(next().text, dim)) return fail(name, "invalid array size");
            v.size *= dim;
            if (!expectPunct(']')) return false;
        }
        if (!v.align) v.align = elemSize ? elemSize : 1;
        for (LocalVar& existing : f_.locals) {
            if (existing.name != v.name) continue;
            existing.size = std::max(existing.size, v.size);
            existing.align = std::max(existing.align, v.align);
            return skipToSemicolon();
        }
        f_.locals.push_back(v);
        return skipToSemicolon();
    }

    bool skipToSemicolon() {
        while (!atEnd()) {
            if (isPunct(next(), ';')) return true;
        }
        return fail(peek(), "expected ';'");
    }

    bool instruction() {
        Instruction ins;
        ins.line = peek().line;
        if (isPunct(peek(), '@')) {
            next();
            if (isPunct(peek(), '!')) {
                next();
                ins.guardNegated = true;
            }
            const Token& g = next();
            // plain names (no '%') are legal registers too, e.g. in inline asm scopes
            if (g.kind != TokenKind::Register && g.kind != TokenKind::Identifier)
                return fail(g, "expected predicate register after '@'");
            ins.guard = resolve(g.text);
        }
        const Token& op = next();
        if (op.kind != TokenKind::Identifier) return fail(op, "expected instruction, got '" + std::string(op.text) + "'");
        ins.opcode = op.text;
        while (peek().kind == TokenKind::Directive) {
            std::string_view mod = next().text;
            // Qualified modifiers: ".L1::evict_last", ".shared::cta" (the CTA's own shared memory,
            // the same as ".shared"), ".shared::cluster".
            while (isPunct(peek(), ':') && isPunct(peek(1), ':') && peek(2).kind == TokenKind::Identifier) {
                next();
                next();
                const Token& q = next();
                mod = std::string_view(mod.data(), size_t(q.text.data() + q.text.size() - mod.data()));
            }
            ins.mods.push_back(mod == ".shared::cta" ? std::string_view(".shared") : mod);
        }

        if (!isPunct(peek(), ';')) {
            for (;;) {
                Operand o;
                if (!operand(o)) return false;
                ins.operands.push_back(std::move(o));
                if (!isPunct(peek(), ',')) break;
                next();
            }
        }
        if (!expectPunct(';')) return false;
        f_.instructions.push_back(std::move(ins));
        return true;
    }

    bool signedInteger(Operand& o, bool negative) {
        const Token& tk = next();
        if (tk.kind == TokenKind::Integer) {
            if (!parseInteger(tk.text, o.imm)) return fail(tk, "invalid integer");
            o.kind = OperandKind::Immediate;
            if (negative) o.imm = uint64_t(0) - o.imm;
            return true;
        }
        if (tk.kind == TokenKind::Float) {
            if (!parseFloat(tk.text, o.imm, o.floatWidth)) return fail(tk, "invalid float literal");
            o.kind = OperandKind::FloatImm;
            if (negative) o.imm ^= uint64_t(1) << (o.floatWidth - 1);
            return true;
        }
        return fail(tk, "expected number");
    }

    bool operand(Operand& o) {
        const Token& tk = peek();
        if (isPunct(tk, '!')) {
            next();
            o.negated = true;
            return operand2(o);
        }
        return operand2(o);
    }

    bool operand2(Operand& o) {
        const Token& tk = peek();
        if (tk.kind == TokenKind::Register) {
            next();
            o.kind = isSpecialRegister(tk.text) ? OperandKind::Special : OperandKind::Register;
            o.name = o.kind == OperandKind::Register ? resolve(tk.text) : tk.text;
            if (peek().kind == TokenKind::Directive && peek().text.size() == 2) {
                const char c = peek().text[1];
                if (c == 'x' || c == 'y' || c == 'z' || c == 'w') {
                    o.component = c;
                    next();
                }
            }
            if (isPunct(peek(), '|')) {
                next();
                const Token& q = next();
                if (q.kind != TokenKind::Register && q.kind != TokenKind::Identifier)
                    return fail(q, "expected register after '|'");
                o.pairName = resolve(q.text);
            }
            return true;
        }
        if (tk.kind == TokenKind::Integer || tk.kind == TokenKind::Float) return signedInteger(o, false);
        if (isPunct(tk, '-')) {
            next();
            return signedInteger(o, true);
        }
        if (tk.kind == TokenKind::Identifier) {
            next();
            o.kind = tk.text == "_" ? OperandKind::Sink : OperandKind::Symbol;
            o.name = resolve(tk.text);
            if (isPunct(peek(), '|')) {  // "r0|p": plain register names
                next();
                const Token& q = next();
                if (q.kind != TokenKind::Register && q.kind != TokenKind::Identifier)
                    return fail(q, "expected register after '|'");
                o.pairName = resolve(q.text);
            }
            return true;
        }
        if (isPunct(tk, '[')) {
            next();
            return address(o);
        }
        if (isPunct(tk, '(')) {  // call argument / return lists
            next();
            o.kind = OperandKind::Vector;
            if (isPunct(peek(), ')')) return next(), true;
            for (;;) {
                Operand e;
                if (!operand(e)) return false;
                o.elems.push_back(std::move(e));
                if (!isPunct(peek(), ',')) break;
                next();
            }
            return expectPunct(')');
        }
        if (isPunct(tk, '{')) {
            next();
            o.kind = OperandKind::Vector;
            for (;;) {
                Operand e;
                if (!operand(e)) return false;
                o.elems.push_back(std::move(e));
                if (!isPunct(peek(), ',')) break;
                next();
            }
            return expectPunct('}');
        }
        return fail(tk, "unexpected operand '" + std::string(tk.text) + "'");
    }

    // [%rd1], [%rd1+8], [%rd1+-8], [sym], [sym+4], [0x100]
    bool address(Operand& o) {
        o.kind = OperandKind::Address;
        const Token& base = peek();
        if (base.kind == TokenKind::Register || base.kind == TokenKind::Identifier) {
            next();
            o.name = resolve(base.text);
            o.baseIsRegister = base.kind == TokenKind::Register;
        } else if (base.kind == TokenKind::Integer) {
            next();
            uint64_t v;
            if (!parseInteger(base.text, v)) return fail(base, "invalid address");
            o.offset = int64_t(v);
            return expectPunct(']');
        } else {
            return fail(base, "invalid address operand");
        }
        if (isPunct(peek(), '+') || isPunct(peek(), '-')) {
            bool negative = isPunct(next(), '-');
            if (isPunct(peek(), '-')) {
                next();
                negative = !negative;
            }
            const Token& off = next();
            uint64_t v;
            if (off.kind != TokenKind::Integer || !parseInteger(off.text, v)) return fail(off, "invalid address offset");
            o.offset = negative ? -int64_t(v) : int64_t(v);
        }
        return expectPunct(']');
    }

    const std::vector<Token>& t_;
    size_t pos_;
    size_t end_;
    Function& f_;
    ParseError& err_;
    std::unordered_set<std::string> regNames_;
    std::vector<std::unordered_map<std::string, std::string_view>> scopes_;  // renames of the open nested scopes
    unsigned scopeNames_ = 0;
};

void appendOperand(std::string& s, const Operand& o) {
    if (o.negated) s += '!';
    switch (o.kind) {
    case OperandKind::Register:
    case OperandKind::Symbol:
    case OperandKind::Sink:
        s += o.name;
        break;
    case OperandKind::Special:
        s += o.name;
        if (o.component) (s += '.') += o.component;
        break;
    case OperandKind::Immediate:
        s += std::to_string(int64_t(o.imm));
        break;
    case OperandKind::FloatImm: {
        char buf[32];
        std::snprintf(buf, sizeof buf, o.floatWidth == 32 ? "0f%08llX" : "0d%016llX", (unsigned long long)o.imm);
        s += buf;
        break;
    }
    case OperandKind::Address:
        s += '[';
        s += o.name;
        if (o.offset || o.name.empty()) {
            if (!o.name.empty() && o.offset >= 0) s += '+';
            s += std::to_string(o.offset);
        }
        s += ']';
        break;
    case OperandKind::Vector:
        s += '{';
        for (size_t i = 0; i < o.elems.size(); ++i) {
            if (i) s += ", ";
            appendOperand(s, o.elems[i]);
        }
        s += '}';
        break;
    }
    if (!o.pairName.empty()) (s += '|') += o.pairName;
}

}  // namespace

// Register names without a '%' prefix look like symbols to the parser; resolve them.
static void resolveRegisterNames(Function& f) {
    std::vector<std::string_view> plain;
    for (const RegDecl& r : f.regs)
        if (!r.name.empty() && r.name[0] != '%') plain.push_back(r.name);
    if (plain.empty()) return;
    std::sort(plain.begin(), plain.end());
    auto isReg = [&](std::string_view n) { return std::binary_search(plain.begin(), plain.end(), n); };
    auto fix = [&](auto& self, Operand& o) -> void {
        if (o.kind == OperandKind::Symbol && isReg(o.name)) o.kind = OperandKind::Register;
        if (o.kind == OperandKind::Address && !o.baseIsRegister && isReg(o.name)) o.baseIsRegister = true;
        for (Operand& e : o.elems) self(self, e);
    };
    for (Instruction& ins : f.instructions) {
        if (ins.opcode == "bra") continue;
        for (Operand& o : ins.operands) fix(fix, o);
    }
}

static bool parseBody(std::string_view body, uint32_t line, Function& out, ParseError& err) {
    out = Function{};
    std::vector<Token> tokens;  // views into the module source, like everything parsed from them
    if (!tokenize(body, tokens, err, line)) return false;
    BodyParser p(tokens, 0, tokens.size() - 1, out, err);  // without the End token
    if (!p.run()) return false;
    resolveRegisterNames(out);
    return true;
}

namespace {

// Inlines every `call` of a Function. Callee registers, labels and local variables get a
// unique prefix; formal parameters are renamed to the caller's argument variables; `ret`
// jumps to a label after the inlined body.
class Inliner {
public:
    // New names are stored in `arena` (the top-level function's), so views into it stay valid
    // across nested inlining.
    Inliner(const Module& m, std::deque<std::string>& arena, ParseError& err) : m_(m), arena_(arena), err_(err) {}

    bool run(Function& fn, int depth) {
        bool hasCalls = false;
        for (const Instruction& ins : fn.instructions) hasCalls |= ins.opcode == "call";
        if (!hasCalls) return true;
        if (depth > kMaxDepth) return fail(fn.instructions.front().line, "recursive device functions are not supported");

        std::vector<Instruction> out;
        std::vector<Label> labels;
        size_t nextLabel = 0;
        std::vector<Label> sorted = fn.labels;
        std::stable_sort(sorted.begin(), sorted.end(), [](const Label& a, const Label& b) { return a.index < b.index; });
        for (size_t i = 0; i <= fn.instructions.size(); ++i) {
            while (nextLabel < sorted.size() && sorted[nextLabel].index == i)
                labels.push_back({sorted[nextLabel++].name, out.size()});
            if (i == fn.instructions.size()) break;
            const Instruction& ins = fn.instructions[i];
            if (ins.opcode != "call") {
                out.push_back(ins);
                continue;
            }
            if (!expand(fn, ins, out, labels, depth)) return false;
        }
        fn.instructions = std::move(out);
        fn.labels = std::move(labels);
        return true;
    }

private:
    static constexpr int kMaxDepth = 32;

    bool fail(uint32_t line, std::string msg) {
        err_ = {std::move(msg), line};
        return false;
    }

    // call{.uni} {(ret, ...),} func{, (arg, ...)}
    bool expand(Function& fn, const Instruction& call, std::vector<Instruction>& out, std::vector<Label>& labels,
                int depth) {
        const Operand* rets = nullptr;
        const Operand* args = nullptr;
        std::string_view callee;
        for (const Operand& o : call.operands) {
            if (o.kind == OperandKind::Symbol && callee.empty()) callee = o.name;
            else if (o.kind == OperandKind::Vector && callee.empty()) rets = &o;
            else if (o.kind == OperandKind::Vector) args = &o;
            else return fail(call.line, "indirect calls are not supported");
        }
        if (callee.empty()) return fail(call.line, "indirect calls are not supported");
        if (!call.guard.empty()) return fail(call.line, "predicated calls are not supported yet");

        if (callee == "vprintf") {
            // Kept as a pseudo instruction for the compiler: vprintf [format param], [arguments param]
            if (!args || args->elems.size() != 2) return fail(call.line, "malformed vprintf call");
            Instruction p;
            p.opcode = "vprintf";
            p.line = call.line;
            for (const Operand& a : args->elems) {
                Operand addr;
                addr.kind = OperandKind::Address;
                addr.name = a.name;
                p.operands.push_back(addr);
            }
            out.push_back(std::move(p));
            if (rets)
                for (const Operand& r : rets->elems) out.push_back(storeParam(r.name, 0, call.line));
            return true;
        }
        if (callee == "__assertfail") {
            // A failed device-side assert(): the thread stops, like trap. (CUDA would also
            // report cudaErrorAssert and print the message; not modeled yet.)
            Instruction t;
            t.opcode = "trap";
            t.line = call.line;
            out.push_back(std::move(t));
            return true;
        }
        const DeviceFunction* f = m_.findFunction(callee);
        if (!f) return fail(call.line, "call to unknown function '" + std::string(callee) + "'");
        if (!f->defined) return fail(call.line, "call to external function '" + std::string(callee) + "'");
        if ((args ? args->elems.size() : 0) != f->params.size() || (rets ? rets->elems.size() : 0) != f->returns.size())
            return fail(call.line, "argument count mismatch in call to '" + std::string(callee) + "'");

        Function body;
        if (!parseBody(f->body, f->bodyLine, body, err_)) return false;
        if (!run(body, depth + 1)) return false;

        const std::string prefix = "$inl" + std::to_string(counter_++) + "_";
        std::unordered_map<std::string_view, std::string_view> rename;
        auto fresh = [&](std::string_view name) -> std::string_view {
            arena_.push_back(prefix + std::string(name));
            return arena_.back();
        };
        for (const RegDecl& r : body.regs) {
            const std::string_view n = fresh(r.name);
            rename[r.name] = n;
            fn.regs.push_back({std::string(n), r.type});
        }
        for (const LocalVar& v : body.locals) {
            if (v.space == ".shared") {  // one per block, shared by every call site
                if (std::none_of(fn.locals.begin(), fn.locals.end(), [&](const LocalVar& l) { return l.name == v.name; }))
                    fn.locals.push_back(v);
                continue;
            }
            LocalVar copy = v;
            copy.name = fresh(v.name);
            rename[v.name] = copy.name;
            fn.locals.push_back(copy);
        }
        for (const Label& l : body.labels) rename[l.name] = fresh(l.name);
        for (size_t i = 0; i < f->params.size(); ++i) rename[f->params[i].name] = args->elems[i].name;
        for (size_t i = 0; i < f->returns.size(); ++i) rename[f->returns[i].name] = rets->elems[i].name;
        const std::string_view after = fresh("return");

        auto mapName = [&](std::string_view n) {
            auto it = rename.find(n);
            return it == rename.end() ? n : it->second;
        };
        auto mapOperand = [&](auto& self, Operand& o) -> void {
            if (o.kind != OperandKind::Special && o.kind != OperandKind::Immediate && o.kind != OperandKind::FloatImm)
                o.name = mapName(o.name);
            if (!o.pairName.empty()) o.pairName = mapName(o.pairName);
            for (Operand& e : o.elems) self(self, e);
        };

        const size_t base = out.size();
        std::vector<Label> calleeLabels;
        for (const Label& l : body.labels) calleeLabels.push_back({mapName(l.name), base + l.index});
        for (Instruction ins : body.instructions) {
            if (!ins.guard.empty()) ins.guard = mapName(ins.guard);
            for (Operand& o : ins.operands) mapOperand(mapOperand, o);
            if (ins.opcode == "ret") {
                ins.opcode = "bra";
                ins.mods.clear();
                Operand target;
                target.kind = OperandKind::Symbol;
                target.name = after;
                ins.operands = {target};
            }
            out.push_back(std::move(ins));
        }
        labels.insert(labels.end(), calleeLabels.begin(), calleeLabels.end());
        labels.push_back({after, out.size()});
        return true;
    }

    // st.param.b32 [name+0], value
    static Instruction storeParam(std::string_view name, uint64_t value, uint32_t line) {
        Instruction st;
        st.opcode = "st";
        st.mods = {".param", ".b32"};
        st.line = line;
        Operand addr;
        addr.kind = OperandKind::Address;
        addr.name = name;
        Operand v;
        v.kind = OperandKind::Immediate;
        v.imm = value;
        st.operands = {addr, v};
        return st;
    }

    const Module& m_;
    std::deque<std::string>& arena_;
    ParseError& err_;
    unsigned counter_ = 0;
};

}  // namespace

bool parseFunctionBody(const Module& module, const Entry& entry, Function& out, ParseError& err) {
    if (!parseBody(entry.body, entry.bodyLine, out, err)) return false;
    Inliner inliner(module, out.names, err);
    return inliner.run(out, 0);
}

std::string toString(const Instruction& ins) {
    std::string s;
    if (!ins.guard.empty()) {
        s += ins.guardNegated ? "@!" : "@";
        (s += ins.guard) += ' ';
    }
    s += ins.opcode;
    for (std::string_view m : ins.mods) s += m;
    for (size_t i = 0; i < ins.operands.size(); ++i) {
        s += i ? ", " : " ";
        appendOperand(s, ins.operands[i]);
    }
    return s;
}

}  // namespace vulkanizer::ptx
