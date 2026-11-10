#include "ptx/parser.h"

#include <algorithm>
#include <bit>
#include <charconv>

namespace vulkanizer::ptx {

const Entry* Module::findEntry(std::string_view name) const {
    for (const Entry& e : entries)
        if (e.name == name) return &e;
    return nullptr;
}

const DeviceFunction* Module::findFunction(std::string_view name) const {
    const DeviceFunction* found = nullptr;
    for (const DeviceFunction& f : functions)
        if (f.name == name && (!found || f.defined)) found = &f;
    return found;
}

uint64_t Module::functionId(std::string_view name) const {
    uint64_t index = 0;
    for (size_t i = 0; i < functions.size(); ++i) {
        bool first = true;  // declarations and the definition share one id
        for (size_t j = 0; j < i && first; ++j) first = functions[j].name != functions[i].name;
        if (!first) continue;
        ++index;
        if (functions[i].name == name) return kFunctionIdBase | index;
    }
    return 0;
}

const GlobalVar* Module::findGlobal(std::string_view name) const {
    for (const GlobalVar& g : globals)
        if (g.name == name) return &g;
    return nullptr;
}

// Size in bytes of a PTX fundamental type directive, 0 if `d` is not a type.
uint32_t typeSize(std::string_view d) {
    struct { std::string_view name; uint32_t size; } table[] = {
        {".b8", 1},     {".u8", 1},       {".s8", 1},     {".b16", 2},    {".u16", 2},
        {".s16", 2},    {".f16", 2},      {".bf16", 2},   {".b32", 4},    {".u32", 4},
        {".s32", 4},    {".f32", 4},      {".f16x2", 4},  {".bf16x2", 4}, {".b64", 8},
        {".u64", 8},    {".s64", 8},      {".f64", 8},    {".b128", 16},  {".texref", 8},
        {".samplerref", 8}, {".surfref", 8},
    };
    for (auto& t : table)
        if (t.name == d) return t.size;
    return 0;
}

static uint32_t alignUp(uint32_t v, uint32_t a) { return a ? (v + a - 1) / a * a : v; }

namespace {

class Parser {
public:
    // `tokens`: the module skeleton (function bodies reduced to their line breaks); `bodies`:
    // the bodies' source text, in order.
    Parser(Module& m, const std::vector<Token>& tokens, const std::vector<std::pair<std::string_view, uint32_t>>& bodies,
           ParseError& err)
        : m_(m), t_(tokens), bodies_(bodies), err_(err) {}

    bool run() {
        bool isExtern = false;
        while (peek().kind != TokenKind::End) {
            const Token& tk = peek();
            if (tk.kind != TokenKind::Directive) return fail(tk, "unexpected token '" + std::string(tk.text) + "'");
            const std::string_view d = tk.text;
            if (d == ".version") {
                if (!parseVersion()) return false;
            } else if (d == ".target") {
                if (!parseTarget()) return false;
            } else if (d == ".address_size") {
                next();
                uint64_t v;
                if (!parseUInt(next(), v)) return false;
                m_.addressSize = static_cast<uint32_t>(v);
            } else if (d == ".visible" || d == ".weak" || d == ".common") {
                next();
                continue;  // linkage prefix, keep isExtern as is
            } else if (d == ".extern") {
                next();
                isExtern = true;
                continue;
            } else if (d == ".entry") {
                if (!parseEntry()) return false;
            } else if (d == ".func") {
                if (!parseFunction()) return false;
            } else if (d == ".global" || d == ".const" || d == ".shared") {
                if (!parseGlobalVar(isExtern)) return false;
            } else if (d == ".file" || d == ".loc") {
                skipLine();
            } else if (d == ".section") {
                if (!skipSection()) return false;
            } else if (d == ".pragma" || d == ".alias") {
                if (!skipToSemicolon()) return false;
            } else {
                return fail(tk, "unsupported module-level directive '" + std::string(d) + "'");
            }
            isExtern = false;
        }
        return true;
    }

private:
    const Token& peek(size_t k = 0) const { return t_[std::min(pos_ + k, t_.size() - 1)]; }
    const Token& next() {
        const Token& tk = peek();
        if (tk.kind != TokenKind::End) ++pos_;
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
    bool parseUInt(const Token& tk, uint64_t& v) {
        if (tk.kind != TokenKind::Integer) return fail(tk, "expected integer");
        if (!parseInteger(tk.text, v)) return fail(tk, "invalid integer");
        return true;
    }

    bool parseVersion() {
        next();
        const Token& tk = next();
        if (tk.kind != TokenKind::Float && tk.kind != TokenKind::Integer) return fail(tk, "expected PTX version");
        m_.version = std::string(tk.text);
        const size_t dot = m_.version.find('.');
        auto toU = [](std::string_view s) {
            uint32_t v = 0;
            std::from_chars(s.data(), s.data() + s.size(), v);
            return v;
        };
        m_.versionMajor = toU(std::string_view(m_.version).substr(0, dot));
        m_.versionMinor = dot == std::string::npos ? 0 : toU(std::string_view(m_.version).substr(dot + 1));
        return true;
    }

    bool parseTarget() {
        next();
        for (;;) {
            const Token& tk = next();
            if (tk.kind != TokenKind::Identifier) return fail(tk, "expected target name");
            m_.targets.emplace_back(tk.text);
            if (!isPunct(peek(), ',')) return true;
            next();
        }
    }

    bool parseParam(std::vector<Param>& params) {
        const Token& start = next();
        if (start.kind != TokenKind::Directive || start.text != ".param")
            return fail(start, "expected .param in kernel signature");
        Param p;
        uint32_t elemSize = 0;
        bool pointer = false;
        while (peek().kind == TokenKind::Directive) {
            const Token& d = next();
            if (d.text == ".align") {
                uint64_t a;
                if (!parseUInt(next(), a)) return false;
                // after .ptr, .align is the alignment of the memory pointed to, not the parameter's
                if (!pointer) p.align = static_cast<uint32_t>(a);
            } else if (uint32_t sz = typeSize(d.text)) {
                elemSize = sz;
            } else if (d.text == ".ptr" || d.text == ".global" || d.text == ".const" || d.text == ".shared" ||
                       d.text == ".local") {
                pointer |= d.text == ".ptr";  // pointer attributes do not affect layout
            } else {
                return fail(d, "unsupported parameter qualifier '" + std::string(d.text) + "'");
            }
        }
        const Token& name = next();
        if (name.kind != TokenKind::Identifier) return fail(name, "expected parameter name");
        if (elemSize == 0) return fail(name, "parameter '" + std::string(name.text) + "' has no type");
        uint64_t count = 1;
        if (isPunct(peek(), '[')) {
            next();
            if (!parseUInt(next(), count)) return false;
            if (!expectPunct(']')) return false;
        }
        p.name = std::string(name.text);
        p.size = static_cast<uint32_t>(elemSize * count);
        if (p.align == 0) p.align = elemSize;
        params.push_back(std::move(p));
        return true;
    }

    bool parseParamList(std::vector<Param>& params) {
        if (!expectPunct('(')) return false;
        if (!isPunct(peek(), ')')) {
            for (;;) {
                if (!parseParam(params)) return false;
                if (!isPunct(peek(), ',')) break;
                next();
            }
        }
        return expectPunct(')');
    }

    static void layoutParams(std::vector<Param>& params, uint32_t* total) {
        uint32_t offset = 0;
        for (Param& p : params) {
            offset = alignUp(offset, p.align);
            p.offset = offset;
            offset += p.size;
        }
        if (total) *total = offset;
    }

    bool parseDims(uint32_t (&dims)[3]) {
        for (int i = 0; i < 3; ++i) {
            uint64_t v;
            if (!parseUInt(next(), v)) return false;
            dims[i] = static_cast<uint32_t>(v);
            if (!isPunct(peek(), ',')) break;
            next();
        }
        for (uint32_t& d : dims)
            if (d == 0) d = 1;
        return true;
    }

    bool parseEntry() {
        next();  // .entry
        const Token& name = next();
        if (name.kind != TokenKind::Identifier) return fail(name, "expected kernel name");
        Entry e;
        e.name = std::string(name.text);

        if (isPunct(peek(), '(') && !parseParamList(e.params)) return false;

        // Performance-tuning directives between the signature and the body.
        while (peek().kind == TokenKind::Directive) {
            const Token& d = next();
            if (d.text == ".maxntid") {
                if (!parseDims(e.maxntid)) return false;
            } else if (d.text == ".reqntid") {
                if (!parseDims(e.reqntid)) return false;
            } else {
                // .minnctapersm, .maxnreg, .maxclusterrank, ...: skip their integer arguments
                while (peek().kind == TokenKind::Integer || isPunct(peek(), ',')) next();
            }
        }

        if (isPunct(peek(), ';')) {  // declaration only
            next();
            return true;
        }
        if (!isPunct(peek(), '{')) return fail(peek(), "expected '{' to start kernel body");
        next();
        if (!skipToMatchingBrace()) return false;
        if (!takeBody(e.body, e.bodyLine)) return false;

        layoutParams(e.params, &e.paramBytes);
        m_.entries.push_back(std::move(e));
        return true;
    }

    bool parseGlobalVar(bool isExtern) {
        GlobalVar g;
        g.space = std::string(next().text);
        g.isExtern = isExtern;
        uint32_t vecWidth = 1;
        bool isFloat = false;
        while (peek().kind == TokenKind::Directive) {
            const Token& d = next();
            if (d.text == ".align") {
                uint64_t a;
                if (!parseUInt(next(), a)) return false;
                g.align = static_cast<uint32_t>(a);
            } else if (d.text == ".v2") {
                vecWidth = 2;
            } else if (d.text == ".v4") {
                vecWidth = 4;
            } else if (uint32_t sz = typeSize(d.text)) {
                g.elemSize = sz;
                isFloat = d.text[1] == 'f';
            }
        }
        const Token& name = next();
        if (name.kind != TokenKind::Identifier) return fail(name, "expected variable name");
        g.name = std::string(name.text);
        uint64_t size = uint64_t(g.elemSize) * vecWidth;
        bool unsized = false;
        while (isPunct(peek(), '[')) {
            next();
            if (isPunct(peek(), ']')) {
                unsized = true;  // extern array, or sized by its initializer
            } else {
                uint64_t dim;
                if (!parseUInt(next(), dim)) return false;
                size *= dim;
            }
            if (!expectPunct(']')) return false;
        }
        if (isPunct(peek(), '=')) {
            next();
            if (!parseInitializer(g, isFloat)) return false;
            if (unsized) size = g.init.size();
            unsized = false;
        }
        g.size = unsized ? 0 : size;
        if (g.init.size() > g.size) return fail(name, "initializer of '" + g.name + "' is too long");
        if (g.align == 0) g.align = g.elemSize ? g.elemSize : 1;
        m_.globals.push_back(std::move(g));
        return expectPunct(';');
    }

    // = 5;  = {1, 2, 3};  = {{1, 2}, {3, 4}};  elements are integer or float literals.
    bool parseInitializer(GlobalVar& g, bool isFloat) {
        int depth = 0;
        for (;;) {
            const Token& tk = peek();
            if (isPunct(tk, '{')) {
                next();
                ++depth;
                continue;
            }
            if (isPunct(tk, '}')) {
                next();
                if (--depth <= 0) return true;
                continue;
            }
            if (isPunct(tk, ',')) {
                next();
                continue;
            }
            if (depth == 0 && isPunct(tk, ';')) return true;
            bool negative = false;
            if (isPunct(tk, '-')) {
                next();
                negative = true;
            }
            const Token& v = next();
            uint64_t bits = 0;
            if (v.kind == TokenKind::Integer) {
                if (!parseInteger(v.text, bits)) return fail(v, "invalid integer in initializer");
                if (negative) bits = uint64_t(0) - bits;
                if (isFloat) {
                    const double d = negative ? -double(uint64_t(0) - bits) : double(bits);
                    bits = g.elemSize == 4 ? std::bit_cast<uint32_t>(float(d)) : std::bit_cast<uint64_t>(d);
                }
            } else if (v.kind == TokenKind::Float) {
                uint32_t width;
                if (!parseFloat(v.text, bits, width)) return fail(v, "invalid float in initializer");
                if (negative) bits ^= uint64_t(1) << (width - 1);
                if (g.elemSize == 4 && width == 64)
                    bits = std::bit_cast<uint32_t>(float(std::bit_cast<double>(bits)));
                else if (g.elemSize == 8 && width == 32)
                    bits = std::bit_cast<uint64_t>(double(std::bit_cast<float>(uint32_t(bits))));
            } else if (v.kind == TokenKind::Identifier && g.elemSize == 8 && !negative) {
                g.functionRefs.emplace_back(g.init.size(), std::string(v.text));  // resolved by parseModule
            } else {
                return fail(v, "unsupported initializer element '" + std::string(v.text) + "'");
            }
            for (uint32_t i = 0; i < g.elemSize; ++i) g.init.push_back(uint8_t(bits >> (8 * i)));
            if (depth == 0) return true;
        }
    }

    bool takeBody(std::string_view& body, uint32_t& line) {
        if (nextBody_ >= bodies_.size()) return fail(peek(), "function body not found");
        body = bodies_[nextBody_].first;
        line = bodies_[nextBody_].second;
        ++nextBody_;
        return true;
    }

    // Consumes tokens up to and including the brace that closes an already consumed '{'.
    bool skipToMatchingBrace() {
        int depth = 1;
        while (depth > 0) {
            const Token& tk = next();
            if (tk.kind == TokenKind::End) return fail(tk, "unterminated '{' block");
            if (isPunct(tk, '{')) ++depth;
            if (isPunct(tk, '}')) --depth;
        }
        return true;
    }

    bool skipToSemicolon() {
        int depth = 0;
        for (;;) {
            const Token& tk = next();
            if (tk.kind == TokenKind::End) return fail(tk, "expected ';'");
            if (isPunct(tk, '{')) ++depth;
            if (isPunct(tk, '}')) --depth;
            if (depth == 0 && isPunct(tk, ';')) return true;
        }
    }

    // .func {(returns)} name {(params)} {directives} ; | { body }
    bool parseFunction() {
        next();  // .func
        DeviceFunction f;
        if (isPunct(peek(), '(') && !parseParamList(f.returns)) return false;
        const Token& name = next();
        if (name.kind != TokenKind::Identifier) return fail(name, "expected function name");
        f.name = std::string(name.text);
        if (isPunct(peek(), '(') && !parseParamList(f.params)) return false;
        while (peek().kind == TokenKind::Directive) {  // .noreturn, .attribute(...), tuning directives
            next();
            if (isPunct(peek(), '(')) {
                while (!isPunct(next(), ')'))
                    if (peek().kind == TokenKind::End) return fail(peek(), "unterminated attribute");
            }
            while (peek().kind == TokenKind::Integer || isPunct(peek(), ',')) next();
        }
        layoutParams(f.params, nullptr);
        layoutParams(f.returns, nullptr);
        if (isPunct(peek(), ';')) {
            next();
        } else {
            if (!expectPunct('{')) return false;
            if (!skipToMatchingBrace()) return false;
            if (!takeBody(f.body, f.bodyLine)) return false;
            f.defined = true;
        }
        m_.functions.push_back(std::move(f));
        return true;
    }

    bool skipSection() {
        next();
        while (peek().kind != TokenKind::End && !isPunct(peek(), '{')) next();
        if (!expectPunct('{')) return false;
        return skipToMatchingBrace();
    }

    // .file / .loc have no terminator; they end at the end of the line.
    void skipLine() {
        const uint32_t line = next().line;
        while (peek().kind != TokenKind::End && peek().line == line) next();
    }

    Module& m_;
    const std::vector<Token>& t_;
    const std::vector<std::pair<std::string_view, uint32_t>>& bodies_;
    size_t nextBody_ = 0;
    ParseError& err_;
    size_t pos_ = 0;
};

}  // namespace

namespace {

// Splits the source into a skeleton for the module-level parser, where the body of every
// .entry/.func is reduced to its line breaks (so line numbers stay right), and the bodies.
// Only comments and string literals need care: PTX has no other way to hide a brace.
void splitBodies(std::string_view s, std::string& skeleton, std::vector<std::pair<std::string_view, uint32_t>>& bodies) {
    skeleton.clear();
    skeleton.reserve(s.size() / 8);
    const size_t n = s.size();
    uint32_t line = 1;
    bool function = false;  // a .entry/.func header since the last top-level ';' or '}'
    size_t i = 0;
    auto copyComment = [&](bool keep) {  // at "//" or "/*": skips (and optionally copies) it
        const size_t begin = i;
        if (s[i + 1] == '/') {
            while (i < n && s[i] != '\n') ++i;
        } else {
            i += 2;
            while (i + 1 < n && !(s[i] == '*' && s[i + 1] == '/')) {
                if (s[i] == '\n') ++line;
                ++i;
            }
            i = std::min(n, i + 2);
        }
        if (keep) skeleton.append(s.substr(begin, i - begin));
    };
    auto isWordChar = [](char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '.' || c == '$'; };
    while (i < n) {
        const char c = s[i];
        if (c == '/' && i + 1 < n && (s[i + 1] == '/' || s[i + 1] == '*')) {
            copyComment(true);
            continue;
        }
        if (c == '"') {
            const size_t begin = i++;
            while (i < n && s[i] != '"' && s[i] != '\n') i += s[i] == '\\' ? 2 : 1;
            i = std::min(n, i + 1);
            skeleton.append(s.substr(begin, i - begin));
            continue;
        }
        if (c == '.' && (i == 0 || !isWordChar(s[i - 1]))) {
            size_t j = i + 1;
            while (j < n && isWordChar(s[j])) ++j;
            const std::string_view word = s.substr(i, j - i);
            if (word == ".entry" || word == ".func") function = true;
            skeleton.append(word);
            i = j;
            continue;
        }
        if (c == ';' || c == '}') function = false;
        if (c == '{' && function) {  // a body: up to the matching brace
            skeleton.push_back('{');
            const size_t begin = ++i;
            const uint32_t bodyLine = line;
            int depth = 1;
            while (i < n && depth > 0) {
                const char b = s[i];
                if (b == '/' && i + 1 < n && (s[i + 1] == '/' || s[i + 1] == '*')) {
                    const uint32_t before = line;
                    copyComment(false);
                    for (uint32_t l = before; l < line; ++l) skeleton.push_back('\n');
                    continue;
                }
                if (b == '{') ++depth;
                else if (b == '}') --depth;
                else if (b == '\n') {
                    ++line;
                    skeleton.push_back('\n');
                }
                ++i;
            }
            bodies.push_back({s.substr(begin, (depth == 0 ? i - 1 : i) - begin), bodyLine});
            if (depth == 0) skeleton.push_back('}');
            function = false;
            continue;
        }
        if (c == '\n') ++line;
        skeleton.push_back(c);
        ++i;
    }
}

}  // namespace

bool parseModule(std::string_view source, Module& out, ParseError& err) {
    out = Module{};
    std::string skeleton;
    std::vector<std::pair<std::string_view, uint32_t>> bodies;
    splitBodies(source, skeleton, bodies);
    std::vector<Token> tokens;
    if (!tokenize(skeleton, tokens, err)) return false;
    Parser p(out, tokens, bodies, err);
    if (!p.run()) return false;
    // .global and .const variables share one buffer per module.
    uint64_t offset = 0;
    for (GlobalVar& g : out.globals) {
        if (g.isExtern || (g.space != ".global" && g.space != ".const")) continue;
        offset = (offset + g.align - 1) / g.align * g.align;
        g.offset = offset;
        offset += g.size;
    }
    out.globalsSize = offset;
    for (GlobalVar& g : out.globals)
        for (const auto& [at, name] : g.functionRefs) {
            const uint64_t id = out.functionId(name);
            if (!id) {
                err = {"unsupported initializer element '" + name + "' in '" + g.name + "'", 1};
                return false;
            }
            for (int i = 0; i < 8; ++i) g.init[at + i] = uint8_t(id >> (8 * i));
        }
    if (out.version.empty()) {
        err = {"missing .version directive", 1};
        return false;
    }
    return true;
}

}  // namespace vulkanizer::ptx
