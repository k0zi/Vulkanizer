// Internal: PTX instruction parser, CFG construction and control-flow structurizer.

#include "common/sha256.h"
#include "compiler/cfg.h"
#include "ptx/instr.h"
#include "ptx/parser.h"

#include <cstdio>
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

int main() {
    testSha256();
    testModuleLayout();
    testVectorAdd();
    testOperands();
    testStructurizer();
    std::printf("OK\n");
    return 0;
}
