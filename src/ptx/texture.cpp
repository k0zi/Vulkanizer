#include "ptx/texture.h"

#include <cctype>
#include <cstring>
#include <string_view>
#include <vector>

namespace vulkanizer::ptx {

namespace {

// The texture library, PTX generated from texture_lib.cu (tools/gen_texture_lib.sh).
const char kTextureLibrary[] =
#include "texture_lib.ptx.inc"
    ;

bool isIdentChar(char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '$' || c == '%'; }

std::string trim(std::string_view s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return std::string(s.substr(a, b - a));
}

// Splits at top-level commas (outside [], {}, ()).
std::vector<std::string> splitTop(std::string_view s) {
    std::vector<std::string> out;
    int depth = 0;
    size_t start = 0;
    for (size_t i = 0; i < s.size(); ++i) {
        const char c = s[i];
        if (c == '[' || c == '{' || c == '(') ++depth;
        else if (c == ']' || c == '}' || c == ')') --depth;
        else if (c == ',' && depth == 0) {
            out.push_back(trim(s.substr(start, i - start)));
            start = i + 1;
        }
    }
    const std::string last = trim(s.substr(start));
    if (!last.empty()) out.push_back(last);
    return out;
}

// "{a, b}" -> {a, b}; a plain operand -> {it}.
std::vector<std::string> elements(const std::string& s) {
    if (s.size() >= 2 && s.front() == '{' && s.back() == '}') return splitTop(std::string_view(s).substr(1, s.size() - 2));
    return {s};
}

struct Coords {
    std::string x, y, z, layer;  // empty: not given
};

// The coordinate vector of a geometry: 1d {x}, 2d {x, y}, 3d {x, y, z, _}, a1d {layer, x},
// a2d {layer, x, y, _}, cube {x, y, z, _} (a direction), acube {layer, x, y, z}.
bool parseCoords(const std::string& geom, const std::vector<std::string>& c, Coords& out) {
    auto at = [&](size_t i) { return i < c.size() && c[i] != "_" ? c[i] : std::string(); };
    if (geom == "1d") out.x = at(0);
    else if (geom == "2d") out.x = at(0), out.y = at(1);
    else if (geom == "3d") out.x = at(0), out.y = at(1), out.z = at(2);
    else if (geom == "a1d") out.layer = at(0), out.x = at(1);
    else if (geom == "a2d") out.layer = at(0), out.x = at(1), out.y = at(2);
    else if (geom == "cube") out.x = at(0), out.y = at(1), out.z = at(2);
    else if (geom == "acube") out.layer = at(0), out.x = at(1), out.y = at(2), out.z = at(3);
    else return false;
    return !out.x.empty();
}

class Rewriter {
public:
    explicit Rewriter(std::string& error) : error_(error) {}

    // The replacement of one statement (`guard` without '@', or empty), or false with error_ set.
    bool rewrite(const std::string& guard, const std::string& opcodeWithMods, const std::string& operands,
                 std::string& out) {
        std::vector<std::string> mods;
        size_t pos = 0;
        while (pos <= opcodeWithMods.size()) {
            const size_t dot = opcodeWithMods.find('.', pos);
            mods.push_back(opcodeWithMods.substr(pos, dot == std::string::npos ? std::string::npos : dot - pos));
            if (dot == std::string::npos) break;
            pos = dot + 1;
        }
        const std::string op = mods.front();
        mods.erase(mods.begin());
        const std::vector<std::string> ops = splitTop(operands);
        id_ = next_++;
        body_.clear();
        decls_.clear();
        bool ok = false;
        if (op == "tex") ok = tex(mods, ops);
        else if (op == "tld4") ok = tld4(mods, ops);
        else if (op == "txq" || op == "suq") ok = query(op, mods, ops);
        else if (op == "suld") ok = suld(mods, ops);
        else if (op == "sust") ok = sust(mods, ops);
        if (!ok) return false;
        std::string block = "{ " + decls_ + body_ + "}";
        if (!guard.empty()) {  // predicated: branch around the block (calls cannot be predicated)
            const bool negated = guard[0] == '!';
            const std::string pred = negated ? guard.substr(1) : guard;
            const std::string skip = name("skip");
            block = std::string("@") + (negated ? "" : "!") + pred + " bra " + skip + "; " + block + " " + skip + ":";
        }
        out = block;
        return true;
    }

private:
    std::string& error_;
    unsigned next_ = 0, id_ = 0;
    std::string decls_, body_;

    std::string name(const char* what) const { return "__vkt" + std::to_string(id_) + "_" + what; }

    bool fail(const std::string& message) {
        error_ = message;
        return false;
    }

    std::string param(const char* what, const char* type) {
        const std::string n = name(what);
        decls_ += std::string(".param .") + type + " " + n + "; ";
        return n;
    }

    void store(const std::string& p, const char* type, const std::string& value) {
        body_ += std::string("st.param.") + type + " [" + p + "], " + value + "; ";
    }

    // The texture handle: a register (unified texture mode); texture references are not supported.
    bool handle(const std::string& address, std::string& h, std::vector<std::string>& coords) {
        if (address.size() < 2 || address.front() != '[' || address.back() != ']') return fail("malformed texture address");
        const std::vector<std::string> parts = splitTop(std::string_view(address).substr(1, address.size() - 2));
        if (parts.empty() || parts[0].empty() || parts[0][0] != '%')
            return fail("texture references are not supported (only texture and surface objects)");
        h = parts[0];
        coords = parts.size() > 1 ? elements(parts[1]) : std::vector<std::string>{};
        return true;
    }

    // Destinations of a 4-vector result: `_` sinks get scratch registers.
    std::vector<std::string> destinations(const std::string& d, const char* regType, size_t count) {
        std::vector<std::string> out = elements(d);
        out.resize(count, "_");
        for (size_t i = 0; i < out.size(); ++i)
            if (out[i] == "_") {
                const std::string s = name("sink") + std::to_string(i);
                decls_ += std::string(".reg .") + regType + " " + s + "; ";
                out[i] = s;
            }
        return out;
    }

    static std::string vector(const std::vector<std::string>& v) {
        std::string s = "{";
        for (size_t i = 0; i < v.size(); ++i) s += (i ? ", " : "") + v[i];
        return s + "}";
    }

    // tex{.level,.grad}.geom.v4.dtype.ctype d, [h, c]{, lod or gradients}{, offset}{, depth}
    bool tex(const std::vector<std::string>& mods, const std::vector<std::string>& ops) {
        std::vector<std::string> m = mods;
        std::string lodMode;  // "level", "grad" or empty (base level)
        if (!m.empty() && (m[0] == "level" || m[0] == "grad" || m[0] == "base")) {
            if (m[0] != "base") lodMode = m[0];
            m.erase(m.begin());
        }
        if (m.size() != 4 || m[1] != "v4") return fail("unsupported tex form");
        if (!lodMode.empty() || m[0] == "cube" || m[0] == "acube") return texEx(m, lodMode, ops);
        const std::string &geom = m[0], &dtype = m[2], &ctype = m[3];
        if (ops.size() < 2) return fail("malformed tex");
        std::string h;
        std::vector<std::string> c;
        if (!handle(ops[1], h, c)) return false;
        Coords xyz;
        if (!parseCoords(geom, c, xyz)) return fail("unsupported texture geometry ." + geom);
        const bool floatResult = dtype == "f32";
        if (!floatResult && dtype != "s32" && dtype != "u32") return fail("unsupported tex result type ." + dtype);
        const bool fetch = ctype == "s32";
        if (!fetch && ctype != "f32") return fail("unsupported tex coordinate type ." + ctype);
        const char* ct = fetch ? "s32" : "f32";
        const std::string zero = fetch ? "0" : "0f00000000";
        const std::string ph = param("h", "b64"), px = param("x", ct), py = param("y", ct), pz = param("z", ct),
                          pl = param("l", "s32");
        decls_ += ".param .align 16 .b8 " + name("r") + "[16]; ";
        store(ph, "b64", h);
        store(px, ct, xyz.x);
        store(py, ct, xyz.y.empty() ? zero : xyz.y);
        store(pz, ct, xyz.z.empty() ? zero : xyz.z);
        store(pl, "s32", xyz.layer.empty() ? "0" : xyz.layer);
        const std::string fn = std::string(fetch ? "__vulkanizer_fetch_" : "__vulkanizer_tex_") + (floatResult ? "f" : "i");
        body_ += "call.uni (" + name("r") + "), " + fn + ", (" + ph + ", " + px + ", " + py + ", " + pz + ", " + pl + "); ";
        const std::vector<std::string> d = destinations(ops[0], floatResult ? "f32" : "b32", 4);
        body_ += std::string("ld.param.v4.") + (floatResult ? "f32" : "b32") + " " + vector(d) + ", [" + name("r") + "]; ";
        return true;
    }

    // tex.level / tex.grad, and cube maps: __vulkanizer_tex_ex_* with the level of detail or the
    // derivatives (gradient vectors have one element per dimension, a cube's three plus a pad)
    bool texEx(const std::vector<std::string>& m, const std::string& lodMode, const std::vector<std::string>& ops) {
        const std::string &geom = m[0], &dtype = m[2], &ctype = m[3];
        if (ctype != "f32") return fail("tex." + lodMode + " needs f32 coordinates");
        const bool floatResult = dtype == "f32";
        if (!floatResult && dtype != "s32" && dtype != "u32") return fail("unsupported tex result type ." + dtype);
        if (ops.size() < 2 + (lodMode == "level" ? 1 : lodMode == "grad" ? 2 : 0)) return fail("malformed tex." + lodMode);
        std::string h;
        std::vector<std::string> c;
        if (!handle(ops[1], h, c)) return false;
        Coords xyz;
        if (!parseCoords(geom, c, xyz)) return fail("unsupported texture geometry ." + geom);
        const std::string zero = "0f00000000";
        auto orZero = [&](const std::string& v) { return v.empty() || v == "_" ? zero : v; };
        std::vector<std::string> dx(3, zero), dy(3, zero);
        if (lodMode == "grad") {
            const std::vector<std::string> gx = elements(ops[2]), gy = elements(ops[3]);
            for (size_t i = 0; i < 3; ++i) {
                if (i < gx.size()) dx[i] = orZero(gx[i]);
                if (i < gy.size()) dy[i] = orZero(gy[i]);
            }
        }
        const std::string ph = param("h", "b64"), px = param("x", "f32"), py = param("y", "f32"), pz = param("z", "f32"),
                          pl = param("l", "s32"), pg = param("g", "s32"), plod = param("lod", "f32");
        std::string args = "(" + ph + ", " + px + ", " + py + ", " + pz + ", " + pl + ", " + pg + ", " + plod;
        store(ph, "b64", h);
        store(px, "f32", xyz.x);
        store(py, "f32", orZero(xyz.y));
        store(pz, "f32", orZero(xyz.z));
        store(pl, "s32", xyz.layer.empty() ? "0" : xyz.layer);
        store(pg, "s32", lodMode == "grad" ? "1" : "0");
        store(plod, "f32", lodMode == "level" ? ops[2] : zero);
        for (int k = 0; k < 6; ++k) {
            const std::string pd = param(("d" + std::to_string(k)).c_str(), "f32");
            store(pd, "f32", k < 3 ? dx[size_t(k)] : dy[size_t(k - 3)]);
            args += ", " + pd;
        }
        args += ")";
        decls_ += ".param .align 16 .b8 " + name("r") + "[16]; ";
        body_ += "call.uni (" + name("r") + "), " + std::string(floatResult ? "__vulkanizer_tex_ex_f" : "__vulkanizer_tex_ex_i") +
                 ", " + args + "; ";
        const std::vector<std::string> d = destinations(ops[0], floatResult ? "f32" : "b32", 4);
        body_ += std::string("ld.param.v4.") + (floatResult ? "f32" : "b32") + " " + vector(d) + ", [" + name("r") + "]; ";
        return true;
    }

    // tld4.comp.2d.v4.dtype.f32 d, [h, {x, y}]
    bool tld4(const std::vector<std::string>& mods, const std::vector<std::string>& ops) {
        if (mods.size() != 5 || mods[1] != "2d" || mods[2] != "v4" || mods[4] != "f32") return fail("unsupported tld4 form");
        const char* comps = "rgba";
        const char* found = std::strchr(comps, mods[0].empty() ? 0 : mods[0][0]);
        if (!found || mods[0].size() != 1) return fail("unsupported tld4 component");
        const bool floatResult = mods[3] == "f32";
        std::string h;
        std::vector<std::string> c;
        if (ops.size() < 2 || !handle(ops[1], h, c) || c.size() < 2) return fail("malformed tld4");
        const std::string ph = param("h", "b64"), pc = param("c", "s32"), px = param("x", "f32"), py = param("y", "f32");
        decls_ += ".param .align 16 .b8 " + name("r") + "[16]; ";
        store(ph, "b64", h);
        store(pc, "s32", std::to_string(found - comps));
        store(px, "f32", c[0]);
        store(py, "f32", c[1]);
        body_ += "call.uni (" + name("r") + "), __vulkanizer_tld4_" + (floatResult ? "f" : "i") + ", (" + ph + ", " + pc +
                 ", " + px + ", " + py + "); ";
        const std::vector<std::string> d = destinations(ops[0], floatResult ? "f32" : "b32", 4);
        body_ += std::string("ld.param.v4.") + (floatResult ? "f32" : "b32") + " " + vector(d) + ", [" + name("r") + "]; ";
        return true;
    }

    // txq{.level}.query.b32 d, [h]{, lod}; suq.query.b32 d, [h]
    bool query(const std::string& op, const std::vector<std::string>& mods, const std::vector<std::string>& ops) {
        std::vector<std::string> m = mods;
        const bool level = !m.empty() && m[0] == "level";
        if (level) m.erase(m.begin());
        if (m.size() != 2 || ops.size() < (level ? 3u : 2u)) return fail("malformed " + op);
        const std::string& q = m[0];
        if (q == "num_samples" || q == "memory_layout") {
            body_ += "mov.b32 " + ops[0] + ", 0; ";
            return true;
        }
        static const char* const kQueries[] = {"width", "height", "depth", "channel_data_type", "channel_order",
                                               "normalized_coords", "filter_mode", "addr_mode_0", "addr_mode_1",
                                               "addr_mode_2", "array_size", "num_mipmap_levels"};
        int index = -1;
        for (int i = 0; i < int(std::size(kQueries)); ++i)
            if (q == kQueries[i]) index = i;
        if (index < 0) return fail("unsupported " + op + " query ." + q);
        std::string h;
        std::vector<std::string> c;
        if (!handle(ops[1], h, c)) return false;
        const std::string ph = param("h", "b64"), pq = param("q", "s32"), pl = param("lvl", "s32");
        decls_ += ".param .b32 " + name("r") + "; ";
        store(ph, "b64", h);
        store(pq, "s32", std::to_string(index));
        store(pl, "s32", level ? ops[2] : "0");
        body_ += "call.uni (" + name("r") + "), __vulkanizer_query, (" + ph + ", " + pq + ", " + pl + "); ";
        body_ += "ld.param.b32 " + ops[0] + ", [" + name("r") + "]; ";
        return true;
    }

    // The geometry, vector width, element bits and clamp mode of suld.b / sust.b.
    struct SurfaceAccess {
        std::string geom;
        int count = 1, bits = 32, clamp = 0;
    };

    bool surfaceAccess(const char* op, const std::vector<std::string>& mods, SurfaceAccess& a) {
        if (mods.empty() || mods[0] != "b") return fail(std::string("formatted ") + op + ".p is not supported");
        for (size_t i = 1; i < mods.size(); ++i) {
            const std::string& x = mods[i];
            if (x == "1d" || x == "2d" || x == "3d" || x == "a1d" || x == "a2d") a.geom = x;
            else if (x == "v2") a.count = 2;
            else if (x == "v4") a.count = 4;
            else if (x == "b8") a.bits = 8;
            else if (x == "b16") a.bits = 16;
            else if (x == "b32") a.bits = 32;
            else if (x == "b64") a.bits = 64;
            else if (x == "trap") a.clamp = 0;
            else if (x == "clamp") a.clamp = 1;
            else if (x == "zero") a.clamp = 2;
            else if (x == "ca" || x == "cg" || x == "cs" || x == "cv" || x == "wb" || x == "wt") continue;
            else return fail(std::string("unsupported ") + op + " modifier ." + x);
        }
        if (a.geom.empty()) return fail(std::string("unsupported ") + op + " geometry");
        return true;
    }

    // Parameters common to the element accesses: handle, coordinates, mode.
    bool surfaceParams(const std::string& address, const SurfaceAccess& a, std::string& h, Coords& xyz) {
        std::vector<std::string> c;
        if (!handle(address, h, c)) return false;
        if (!parseCoords(a.geom, c, xyz)) return fail("malformed surface coordinates");
        return true;
    }

    void elementCall(const char* fn, const std::string& h, const SurfaceAccess& a, const Coords& xyz, int element,
                     const std::string* ret, const std::string* value) {
        const std::string suffix = std::to_string(element);
        const std::string ph = param(("h" + suffix).c_str(), "b64"), pb = param(("b" + suffix).c_str(), "s32"),
                          px = param(("x" + suffix).c_str(), "s32"), py = param(("y" + suffix).c_str(), "s32"),
                          pz = param(("z" + suffix).c_str(), "s32"), pl = param(("l" + suffix).c_str(), "s32"),
                          pm = param(("m" + suffix).c_str(), "s32");
        store(ph, "b64", h);
        store(pb, "s32", std::to_string(a.bits / 8));
        if (element == 0) {
            store(px, "s32", xyz.x);
        } else {  // the element's byte offset added to x
            const std::string r = name("xo") + suffix;
            decls_ += ".reg .s32 " + r + "; ";
            body_ += "add.s32 " + r + ", " + xyz.x + ", " + std::to_string(element * a.bits / 8) + "; ";
            store(px, "s32", r);
        }
        store(py, "s32", xyz.y.empty() ? "0" : xyz.y);
        store(pz, "s32", xyz.z.empty() ? "0" : xyz.z);
        store(pl, "s32", xyz.layer.empty() ? "0" : xyz.layer);
        store(pm, "s32", std::to_string(a.clamp));
        std::string args = "(" + ph + ", " + pb + ", " + px + ", " + py + ", " + pz + ", " + pl + ", " + pm;
        if (value) {
            const std::string pv = param(("v" + suffix).c_str(), "b64");
            store(pv, ("b" + std::to_string(a.bits)).c_str(), *value);
            args += ", " + pv;
        }
        args += ")";
        if (ret) {
            const std::string r = name("r") + suffix;
            decls_ += ".param .b64 " + r + "; ";
            body_ += "call.uni (" + r + "), " + fn + ", " + args + "; ";
            body_ += "ld.param.b" + std::to_string(a.bits) + " " + *ret + ", [" + r + "]; ";
        } else {
            body_ += std::string("call.uni ") + fn + ", " + args + "; ";
        }
    }

    // suld.b.geom{.cop}{.vec}.btype.clamp d, [h, coords]
    bool suld(const std::vector<std::string>& mods, const std::vector<std::string>& ops) {
        SurfaceAccess a;
        if (!surfaceAccess("suld", mods, a) || ops.size() < 2) return ops.size() < 2 ? fail("malformed suld") : false;
        std::string h;
        Coords xyz;
        if (!surfaceParams(ops[1], a, h, xyz)) return false;
        std::vector<std::string> d = elements(ops[0]);
        if (int(d.size()) != a.count) return fail("suld destination count mismatch");
        for (int i = 0; i < a.count; ++i) {
            if (d[size_t(i)] == "_") continue;
            elementCall("__vulkanizer_suld", h, a, xyz, i, &d[size_t(i)], nullptr);
        }
        return true;
    }

    // sust.b.geom{.cop}{.vec}.ctype.clamp [h, coords], value
    bool sust(const std::vector<std::string>& mods, const std::vector<std::string>& ops) {
        SurfaceAccess a;
        if (!surfaceAccess("sust", mods, a) || ops.size() < 2) return ops.size() < 2 ? fail("malformed sust") : false;
        std::string h;
        Coords xyz;
        if (!surfaceParams(ops[0], a, h, xyz)) return false;
        std::vector<std::string> v = elements(ops[1]);
        if (int(v.size()) != a.count) return fail("sust value count mismatch");
        for (int i = 0; i < a.count; ++i) elementCall("__vulkanizer_sust", h, a, xyz, i, nullptr, &v[size_t(i)]);
        return true;
    }
};

const char* const kOpcodes[] = {"tex", "tld4", "txq", "suld", "sust", "suq"};

// The texture opcode starting at s[i] (followed by '.'), or nullptr.
const char* opcodeAt(const std::string& s, size_t i) {
    for (const char* op : kOpcodes) {
        const size_t n = std::strlen(op);
        if (s.compare(i, n, op) == 0 && i + n < s.size() && s[i + n] == '.') return op;
    }
    return nullptr;
}

}  // namespace

bool rewriteTextureInstructions(std::string& source, std::string& error) {
    std::string out;
    out.reserve(source.size());
    Rewriter rewriter(error);
    bool any = false;
    size_t i = 0, copied = 0;
    while (i < source.size()) {
        // skip comments and strings
        if (source.compare(i, 2, "//") == 0) {
            i = source.find('\n', i);
            if (i == std::string::npos) break;
            continue;
        }
        if (source.compare(i, 2, "/*") == 0) {
            i = source.find("*/", i + 2);
            if (i == std::string::npos) break;
            i += 2;
            continue;
        }
        if (source[i] == '"') {
            i = source.find('"', i + 1);
            if (i == std::string::npos) break;
            ++i;
            continue;
        }
        const bool boundary = i == 0 || (!isIdentChar(source[i - 1]) && source[i - 1] != '.');
        if (!boundary || !opcodeAt(source, i)) {
            ++i;
            continue;
        }
        // the statement: [guard] opcode.mods operands;
        const size_t end = source.find(';', i);
        if (end == std::string::npos) break;
        size_t start = i;
        std::string guard;
        {  // a guard "@%p" or "@!%p" just before
            size_t j = i;
            while (j > 0 && std::isspace(static_cast<unsigned char>(source[j - 1]))) --j;
            size_t k = j;
            while (k > 0 && isIdentChar(source[k - 1])) --k;
            if (k > 0 && source[k - 1] == '!') --k;
            if (k > 0 && source[k - 1] == '@' && k < j) {
                guard = source.substr(k, j - k);
                start = k - 1;
            }
        }
        size_t opEnd = i;
        while (opEnd < end && !std::isspace(static_cast<unsigned char>(source[opEnd]))) ++opEnd;
        std::string replacement;
        if (!rewriter.rewrite(guard, source.substr(i, opEnd - i), source.substr(opEnd, end - opEnd), replacement)) {
            size_t line = 1;
            for (size_t k = 0; k < i; ++k) line += source[k] == '\n';
            error = "line " + std::to_string(line) + ": " + error;
            return false;
        }
        out.append(source, copied, start - copied);
        out += replacement;
        copied = end + 1;
        i = end + 1;
        any = true;
    }
    if (!any) return true;
    out.append(source, copied, std::string::npos);
    // the library's functions, without its header (.version, .target, .address_size)
    std::string_view lib = kTextureLibrary;
    const size_t first = lib.find(".visible");
    out += "\n";
    out += lib.substr(first == std::string_view::npos ? 0 : first);
    source = std::move(out);
    return true;
}

}  // namespace vulkanizer::ptx
