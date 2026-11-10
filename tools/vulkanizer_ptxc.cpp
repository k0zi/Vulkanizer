// Offline PTX -> SPIR-V compiler driver, the main development tool for src/compiler.
//
//   vulkanizer-ptxc kernel.ptx                  compile every entry to <stem>.<entry>.spv
//   vulkanizer-ptxc kernel.ptx -e vecAdd -o x.spv
//   vulkanizer-ptxc kernel.ptx --list           list entries and their parameter layout
//   vulkanizer-ptxc kernel.ptx --dis --val      also run spirv-dis / spirv-val on the output
//   vulkanizer-ptxc kernel.ptx --dump-ir        print the parsed instructions, CFG and structured form
//
// Target options (default: a generic subgroup-32 device):
//   --subgroup N  --push-constants N  --fp16  --fp64  --int8  --int16  --storage8  --storage16  --int64-atomics

#include "ptx/heap.h"
#include "compiler/cfg.h"
#include "compiler/ptx_to_spirv.h"
#include "ptx/instr.h"
#include "ptx/parser.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <sys/wait.h>
#include <string>
#include <vector>

using namespace vulkanizer;

namespace {

void usage() {
    std::fprintf(stderr,
                 "usage: vulkanizer-ptxc <file.ptx> [-e entry] [-o out.spv] [--list] [--dump-ir] [--dis] [--val]\n"
                 "                  [--subgroup N] [--push-constants N] [--fp16] [--fp64] [--int8] [--int16]\n"
                 "                  [--storage8] [--storage16] [--int64-atomics]\n");
}

bool readFile(const char* path, std::string& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::stringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

bool writeFile(const std::string& path, const std::vector<uint32_t>& words) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(words.data()), std::streamsize(words.size() * sizeof(uint32_t)));
    return bool(f);
}

std::string stemOf(const std::string& path) {
    size_t slash = path.find_last_of('/');
    std::string name = slash == std::string::npos ? path : path.substr(slash + 1);
    size_t dot = name.find_last_of('.');
    return dot == std::string::npos ? name : name.substr(0, dot);
}

int runTool(const char* tool, const std::string& args) {
    std::string cmd = std::string(tool) + " " + args;
    int rc = std::system(cmd.c_str());
    if (rc == -1 || (WIFEXITED(rc) && WEXITSTATUS(rc) == 127)) std::fprintf(stderr, "%s not found\n", tool);
    return rc;
}

void listEntries(const ptx::Module& m) {
    std::printf("PTX %s, target", m.version.c_str());
    for (const std::string& t : m.targets) std::printf(" %s", t.c_str());
    std::printf(", address size %u\n", m.addressSize);
    for (const ptx::Entry& e : m.entries) {
        std::printf("%s  (%u param bytes)\n", e.name.c_str(), e.paramBytes);
        for (const ptx::Param& p : e.params)
            std::printf("    +%-4u size %-3u align %-3u %s\n", p.offset, p.size, p.align, p.name.c_str());
    }
    for (const ptx::GlobalVar& g : m.globals)
        std::printf("var %s %s  size %llu align %u%s\n", g.space.c_str(), g.name.c_str(),
                    (unsigned long long)g.size, g.align, g.isExtern ? " extern" : "");
}

bool dumpEntry(const ptx::Module& m, const ptx::Entry& e) {
    ptx::Function fn;
    ptx::ParseError perr;
    if (!ptx::parseFunctionBody(m, e, fn, perr)) {
        std::fprintf(stderr, "%s: line %u: %s\n", e.name.c_str(), perr.line, perr.message.c_str());
        return false;
    }
    std::printf("==== %s: %zu registers, %zu instructions\n", e.name.c_str(), fn.regs.size(), fn.instructions.size());
    compiler::Cfg cfg;
    std::string error;
    if (!compiler::buildCfg(fn, cfg, error)) {
        std::fprintf(stderr, "%s: %s\n", e.name.c_str(), error.c_str());
        return false;
    }
    std::printf("---- CFG\n%s", compiler::dump(fn, cfg).c_str());
    compiler::Structured st;
    if (!compiler::structurize(cfg, st, error)) {
        std::fprintf(stderr, "%s: %s\n", e.name.c_str(), error.c_str());
        return false;
    }
    std::printf("---- structured\n%s", compiler::dump(st).c_str());
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    const char* input = nullptr;
    std::string entryName, output;
    bool list = false, dumpIr = false, dis = false, val = false;
    compiler::Target target;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs an argument\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "-e") entryName = next();
        else if (a == "-o") output = next();
        else if (a == "--list") list = true;
        else if (a == "--dump-ir") dumpIr = true;
        else if (a == "--dis") dis = true;
        else if (a == "--val") val = true;
        else if (a == "--subgroup") target.subgroupSize = uint32_t(std::atoi(next()));
        else if (a == "--push-constants") target.maxPushConstantsSize = uint32_t(std::atoi(next()));
        else if (a == "--fp16") target.float16 = true;
        else if (a == "--fp64") target.float64 = true;
        else if (a == "--int8") target.int8 = true;
        else if (a == "--int16") target.int16 = true;
        else if (a == "--storage8") target.storage8 = target.int8 = true;
        else if (a == "--storage16") target.storage16 = target.int16 = true;
        else if (a == "--int64-atomics") target.int64Atomics = true;
        else if (a == "--dot") target.integerDotProduct = true;
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else if (a[0] == '-' || input) { usage(); return 2; }
        else input = argv[i];
    }
    if (!input) {
        usage();
        return 2;
    }

    std::string source;
    if (!readFile(input, source)) {
        std::fprintf(stderr, "cannot read %s\n", input);
        return 1;
    }
    ptx::appendDeviceHeap(source);  // malloc and free, as the driver's loader does
    ptx::Module module;
    ptx::ParseError perr;
    if (!ptx::parseModule(source, module, perr)) {
        std::fprintf(stderr, "%s:%u: %s\n", input, perr.line, perr.message.c_str());
        return 1;
    }
    if (list) {
        listEntries(module);
        return 0;
    }

    std::vector<const ptx::Entry*> entries;
    if (!entryName.empty()) {
        const ptx::Entry* e = module.findEntry(entryName);
        if (!e) {
            std::fprintf(stderr, "no entry '%s' in %s\n", entryName.c_str(), input);
            return 1;
        }
        entries.push_back(e);
    } else {
        for (const ptx::Entry& e : module.entries) entries.push_back(&e);
    }
    if (entries.empty()) {
        std::fprintf(stderr, "%s has no kernels\n", input);
        return 1;
    }
    if (!output.empty() && entries.size() > 1) {
        std::fprintf(stderr, "-o needs a single entry; use -e (entries: %zu)\n", entries.size());
        return 2;
    }

    int status = 0;
    for (const ptx::Entry* e : entries) {
        if (dumpIr && !dumpEntry(module, *e)) status = 1;
        compiler::KernelBinary bin;
        std::string error;
        if (!compiler::compileKernel(module, *e, target, bin, error)) {
            std::fprintf(stderr, "%s: %s\n", e->name.c_str(), error.c_str());
            status = 1;
            continue;
        }
        std::string out = output.empty() ? stemOf(input) + "." + e->name + ".spv" : output;
        if (!writeFile(out, bin.spirv)) {
            std::fprintf(stderr, "cannot write %s\n", out.c_str());
            status = 1;
            continue;
        }
        std::printf("%s -> %s (%zu words, %u push constant bytes)\n", e->name.c_str(), out.c_str(), bin.spirv.size(),
                    bin.pushConstantSize);
        if (dis) runTool("spirv-dis", "'" + out + "'");
        if (val && runTool("spirv-val", "--target-env vulkan1.2 '" + out + "'") != 0) status = 1;
    }
    return status;
}
