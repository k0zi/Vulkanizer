// Internal: PTX instruction parser, CFG construction and control-flow structurizer.

#include "common/sha256.h"
#include "compiler/cfg.h"
#include "ptx/instr.h"
#include "ptx/parser.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

using namespace vulkanizer;

#define EXPECT(cond)                                                                            \
    do {                                                                                        \
        if (!(cond)) {                                                                          \
            std::fprintf(stderr, "%s:%d: expectation failed: %s\n", __FILE__, __LINE__, #cond); \
            std::exit(1);                                                                       \
        }                                                                                       \
    } while (0)

namespace {

struct Parsed {
    std::string source;
    ptx::Module module;
    ptx::Function fn;
};

void parse(Parsed& p, const std::string& source, const char* entry = nullptr) {
    p.source = source;
    ptx::ParseError err;
    if (!ptx::parseModule(p.source, p.module, err)) {
        std::fprintf(stderr, "module parse error at line %u: %s\n", err.line, err.message.c_str());
        std::exit(1);
    }
    const ptx::Entry* e = entry ? p.module.findEntry(entry) : &p.module.entries.at(0);
    EXPECT(e != nullptr);
    if (!ptx::parseFunctionBody(p.module, *e, p.fn, err)) {
        std::fprintf(stderr, "body parse error at line %u: %s\n", err.line, err.message.c_str());
        std::exit(1);
    }
}

std::string kernel(const std::string& body) {
    return ".version 7.8\n.target sm_52\n.address_size 64\n.visible .entry k(.param .u64 k_p0)\n{\n" + body + "\n}\n";
}

std::string structured(const std::string& body, bool expectOk = true) {
    Parsed p;
    parse(p, kernel(body));
    compiler::Cfg cfg;
    std::string error;
    EXPECT(compiler::buildCfg(p.fn, cfg, error));
    compiler::Structured st;
    const bool ok = compiler::structurize(cfg, st, error);
    if (ok != expectOk) {
        std::fprintf(stderr, "structurize: %s\n", ok ? "unexpected success" : error.c_str());
        std::exit(1);
    }
    return ok ? compiler::dump(st) : error;
}

void testVectorAdd() {
    std::ifstream f(VULKANIZER_TEST_DATA_DIR "/vector_add.ptx");
    std::stringstream ss;
    ss << f.rdbuf();
    Parsed p;
    parse(p, ss.str(), "vecAdd");
    EXPECT(p.fn.regs.size() == 2 + 6 + 4 + 11);  // %p<2> %r<6> %f<4> %rd<11>
    EXPECT(p.fn.regs[0].name == "%p0" && p.fn.regs[0].type == ".pred");
    EXPECT(p.fn.instructions.size() == 22);
    EXPECT(p.fn.labels.size() == 1 && p.fn.labels[0].name == "$L__BB0_2" && p.fn.labels[0].index == 21);

    const ptx::Instruction& ld = p.fn.instructions[0];
    EXPECT(ld.opcode == "ld" && ld.mods.size() == 2 && ld.has(".param") && ld.has(".u64"));
    EXPECT(ld.operands[1].kind == ptx::OperandKind::Address && ld.operands[1].name == "vecAdd_param_0" &&
           !ld.operands[1].baseIsRegister);

    const ptx::Instruction& movTid = p.fn.instructions[6];
    EXPECT(movTid.operands[1].kind == ptx::OperandKind::Special && movTid.operands[1].name == "%tid" &&
           movTid.operands[1].component == 'x');

    const ptx::Instruction& bra = p.fn.instructions[9];
    EXPECT(bra.opcode == "bra" && bra.guard == "%p1" && !bra.guardNegated);
    EXPECT(ptx::toString(bra) == "@%p1 bra $L__BB0_2");

    const ptx::Instruction& mul = p.fn.instructions[11];
    EXPECT(mul.operands[2].kind == ptx::OperandKind::Immediate && mul.operands[2].imm == 4);
}

void testOperands() {
    Parsed p;
    parse(p, kernel(R"(
        .reg .pred %p<3>;
        .reg .b32 %r<4>;
        .reg .b64 %rd<3>;
        .reg .f32 %f<3>;
        .reg .b32 plain;
        ld.global.v2.f32 {%f1, %f2}, [%rd1+-8];
        st.global.u32 [%rd2+16], -5;
        mov.f32 %f1, 0f3F800000;
        mov.f64 %rd1, -1.5;
        setp.lt.and.s32 %p1|%p2, %r1, %r2, !%p0;
        @!%p1 add.s32 plain, plain, 0x10;
        mov.b64 {%r1, _}, %rd1;
        ld.param.u32 %r3, [k_p0+4];
    )"));
    const auto& in = p.fn.instructions;
    EXPECT(in.size() == 8);
    EXPECT(in[0].operands[0].kind == ptx::OperandKind::Vector && in[0].operands[0].elems.size() == 2);
    EXPECT(in[0].operands[1].baseIsRegister && in[0].operands[1].offset == -8);
    EXPECT(in[1].operands[0].offset == 16 && int64_t(in[1].operands[1].imm) == -5);
    EXPECT(in[2].operands[1].kind == ptx::OperandKind::FloatImm && in[2].operands[1].floatWidth == 32 &&
           in[2].operands[1].imm == 0x3F800000);
    EXPECT(in[3].operands[1].floatWidth == 64 && in[3].operands[1].imm == 0xBFF8000000000000ull);
    EXPECT(in[4].operands[0].name == "%p1" && in[4].operands[0].pairName == "%p2" && in[4].operands[3].negated);
    EXPECT(in[5].guard == "%p1" && in[5].guardNegated);
    EXPECT(in[5].operands[0].kind == ptx::OperandKind::Register && in[5].operands[0].name == "plain");
    EXPECT(in[5].operands[2].imm == 16);
    EXPECT(in[6].operands[0].elems[1].kind == ptx::OperandKind::Sink);
    EXPECT(in[7].operands[1].name == "k_p0" && in[7].operands[1].offset == 4);
}

// A nested scope redeclaring outer registers (CUDA's f64 trig reduction does this in inline
// asm): inside it the names refer to fresh registers, outside to the original ones.
void testShadowingScopes() {
    Parsed p;
    parse(p, kernel(R"(
        .reg .b32 %r<4>;
        mov.u32 %r3, 7;
        {
        .reg .u32 %r3, %t;
        mov.u32 %r3, 1;
        @%r3 add.u32 %t, %r3, %r1;
        {
        .reg .u32 %r3;
        mov.b64 {%r3, %t}, %r2;
        }
        add.u32 %r1, %r3, %t;
        }
        add.u32 %r2, %r3, 1;
    )"));
    const auto& in = p.fn.instructions;
    EXPECT(in.size() == 6);
    const std::string_view inner = in[1].operands[0].name, innermost = in[3].operands[0].elems[0].name;
    EXPECT(in[0].operands[0].name == "%r3");
    EXPECT(inner != "%r3" && innermost != "%r3" && inner != innermost);
    EXPECT(in[2].guard == inner && in[2].operands[1].name == inner && in[2].operands[0].name == "%t");
    EXPECT(in[2].operands[2].name == "%r1");  // not redeclared: the outer register
    EXPECT(in[4].operands[1].name == inner && in[4].operands[0].name == "%r1");
    EXPECT(in[5].operands[1].name == "%r3");  // the scopes are closed again
    EXPECT(p.fn.regs.size() == 4 + 3);       // %r0..%r3, %t and the two renamed %r3
}

void testStructurizer() {
    // if-then
    EXPECT(structured(R"(
        .reg .pred %p<2>;
        .reg .b32 %r<2>;
        setp.eq.s32 %p1, %r1, 0;
        @%p1 bra DONE;
        add.s32 %r1, %r1, 1;
    DONE:
        ret;
    )") == "block #1\n  code B0\n  if %p1\n    br #1\n  else\n    code B1\n    br #1\nreturn\n");

    // loop with the exit emitted after the loop
    const std::string loop = structured(R"(
        .reg .pred %p<2>;
        .reg .b32 %r<2>;
    LOOP:
        add.s32 %r1, %r1, 1;
        setp.lt.s32 %p1, %r1, 10;
        @%p1 bra LOOP;
        ret;
    )");
    EXPECT(loop == "block #1\n  loop #2\n    code B0\n    if %p1\n      br #2\n    else\n      br #1\nreturn\n");

    // early return inside a loop, guarded return
    structured(R"(
        .reg .pred %p<3>;
        .reg .b32 %r<2>;
    LOOP:
        add.s32 %r1, %r1, 1;
        setp.eq.s32 %p2, %r1, 5;
        @%p2 ret;
        setp.lt.s32 %p1, %r1, 10;
        @%p1 bra LOOP;
        ret;
    )");

    // irreducible: two entries into the same cycle
    const std::string err = structured(R"(
        .reg .pred %p<2>;
        .reg .b32 %r<2>;
        setp.eq.s32 %p1, %r1, 0;
        @%p1 bra B;
    A:
        add.s32 %r1, %r1, 1;
    B:
        add.s32 %r1, %r1, 2;
        bra.uni A;
    )", false);
    EXPECT(err.find("irreducible") != std::string::npos);
}

}  // namespace

void testSha256() {
    auto hex = [](const std::string& in) {
        Sha256 h;
        h.update(in);
        return h.hexDigest();
    };
    EXPECT(hex("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    EXPECT(hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    EXPECT(hex(std::string(1000, 'a')) == "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3");
}

// Bodies are split from the module text before tokenizing (comments, strings and initializers may
// hold braces) and keep their line numbers; `.align` after `.ptr` does not align the parameter.
void testModuleLayout() {
    const std::string src =
        ".version 7.8\n"                                                  // 1
        ".target sm_52\n"                                                 // 2
        ".address_size 64\n"                                              // 3
        ".file 1 \"odd { name }.cu\"\n"                                    // 4
        ".global .align 4 .b32 table[2] = {1, 2};\n"                      // 5
        ".extern .func (.param .b32 r) ext (.param .b32 a);\n"           // 6
        ".func (.param .b32 r) twice (.param .b32 a)\n"                  // 7
        "{ // a comment with { an unmatched brace\n"                      // 8
        "    .reg .b32 %t;\n"                                             // 9
        "    ld.param.b32 %t, [a]; /* and } another */\n"                 // 10
        "    add.s32 %t, %t, %t;\n"                                       // 11
        "    st.param.b32 [r], %t;\n"                                     // 12
        "    ret;\n"                                                      // 13
        "}\n"                                                             // 14
        ".visible .entry k(.param .u32 k_n, .param .u64 .ptr .align 1 k_p)\n"  // 15
        "{\n"                                                             // 16
        "    .reg .b32 %r<2>;\n"                                          // 17
        "    ld.param.u32 %r0, [k_n];\n"                                  // 18
        "    ret;\n"                                                      // 19
        "}\n";
    Parsed p;
    parse(p, src, "k");
    const ptx::Entry& e = *p.module.findEntry("k");
    EXPECT(e.params.size() == 2 && e.params[1].offset == 8 && e.paramBytes == 16);
    EXPECT(p.module.findGlobal("table") && p.module.findGlobal("table")->init.size() == 8);
    const ptx::DeviceFunction* twice = p.module.findFunction("twice");
    EXPECT(twice && twice->defined && !p.module.findFunction("ext")->defined);
    EXPECT(!p.fn.instructions.empty() && p.fn.instructions[0].line == 18);
}

// Function addresses: ids in initializers (a vtable) and as mov operands; an indirect call becomes
// a comparison with each candidate of its shape (here f and g, not h with two parameters).
void testIndirectCalls() {
    Parsed p;
    parse(p, R"(.version 7.8
.target sm_52
.address_size 64
.func (.param .b32 r) f(.param .b32 a) { .reg .b32 %x; ld.param.b32 %x, [a]; st.param.b32 [r], %x; ret; }
.func (.param .b32 r) g(.param .b32 a) { .reg .b32 %x; ld.param.b32 %x, [a]; add.s32 %x, %x, 1; st.param.b32 [r], %x; ret; }
.func (.param .b32 r) h(.param .b32 a, .param .b32 b) { st.param.b32 [r], 0; ret; }
.global .align 8 .u64 table[3] = {0, f, g};
.visible .entry k(.param .u64 k_p0)
{
    .reg .b64 %fp;
    .reg .b32 %v;
    mov.u64 %fp, g;
    {
    .param .b32 a0;
    st.param.b32 [a0], 5;
    .param .b32 r0;
    proto : .callprototype (.param .b32 _) _ (.param .b32 _);
    call (r0), %fp, (a0), proto;
    ld.param.b32 %v, [r0];
    }
    ret;
}
)");
    const ptx::GlobalVar* t = p.module.findGlobal("table");
    EXPECT(t && t->init.size() == 24);
    uint64_t ids[3];
    std::memcpy(ids, t->init.data(), sizeof ids);
    EXPECT(ids[0] == 0 && ids[1] == p.module.functionId("f") && ids[2] == p.module.functionId("g"));
    EXPECT(ids[1] != ids[2] && (ids[1] & ptx::kFunctionIdBase) == ptx::kFunctionIdBase && !p.module.functionId("table"));
    // mov of a function name is its id; two candidates, each compared and inlined
    EXPECT(p.fn.instructions[0].opcode == "mov" && p.fn.instructions[0].operands[1].kind == ptx::OperandKind::Immediate &&
           p.fn.instructions[0].operands[1].imm == ids[2]);
    int compares = 0;
    for (const ptx::Instruction& ins : p.fn.instructions) compares += ins.opcode == "setp";
    EXPECT(compares == 2);
}

int main() {
    testSha256();
    testModuleLayout();
    testVectorAdd();
    testOperands();
    testShadowingScopes();
    testStructurizer();
    testIndirectCalls();
    std::printf("OK\n");
    return 0;
}
