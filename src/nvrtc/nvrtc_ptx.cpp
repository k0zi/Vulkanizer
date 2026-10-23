// An LD_AUDIT library (run_torch.sh, vulkanizer-run --torch): programs that compile CUDA C++ at run
// time and load the result as a cubin (PyTorch's jiterator, CuPy) get the program's PTX instead,
// which Vulkanizer translates; a cubin is native NVIDIA machine code it cannot run.
// nvrtcCompileProgram compiles for the virtual architecture (compute_XX for a requested sm_XX, which
// would make NVRTC keep machine code only), and nvrtcGetCUBINSize/nvrtcGetCUBIN are answered with
// nvrtcGetPTXSize/nvrtcGetPTX of the same program, or passed on where there is no PTX.
// An audit library because PyTorch looks NVRTC up with dlopen + dlsym, which a preloaded library
// cannot interpose: the dynamic linker reports every binding, dlsym's too, to la_symbind64.

#include <link.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

using SizeFn = int (*)(void* program, size_t* size);
using CodeFn = int (*)(void* program, char* code);
using CompileFn = int (*)(void* program, int numOptions, const char* const* options);

// The real NVRTC functions, recorded as their bindings come by.
CompileFn realCompile;
SizeFn realCubinSize, realPtxSize;
CodeFn realCubin, realPtx;

constexpr int kInternalError = 7;  // NVRTC_ERROR_INTERNAL_ERROR

const struct link_map* nvrtcLibrary;  // libnvrtc, once loaded

// A dynamic symbol of a loaded object, through its GNU hash table (the PTX getters are looked up
// lazily by PyTorch, so their bindings may not have come by yet).
uintptr_t lookup(const struct link_map* map, const char* name) {
    const ElfW(Sym)* symtab = nullptr;
    const char* strtab = nullptr;
    const uint32_t* gnuHash = nullptr;
    for (const ElfW(Dyn)* d = map->l_ld; d->d_tag != DT_NULL; ++d) {
        if (d->d_tag == DT_SYMTAB) symtab = reinterpret_cast<const ElfW(Sym)*>(d->d_un.d_ptr);
        else if (d->d_tag == DT_STRTAB) strtab = reinterpret_cast<const char*>(d->d_un.d_ptr);
        else if (d->d_tag == DT_GNU_HASH) gnuHash = reinterpret_cast<const uint32_t*>(d->d_un.d_ptr);
    }
    if (!symtab || !strtab || !gnuHash) return 0;
    uint32_t h = 5381;
    for (const unsigned char* c = reinterpret_cast<const unsigned char*>(name); *c; ++c) h = h * 33 + *c;
    const uint32_t nbuckets = gnuHash[0], symoffset = gnuHash[1], bloomSize = gnuHash[2];
    const uint32_t* buckets = gnuHash + 4 + bloomSize * (sizeof(ElfW(Addr)) / 4);
    const uint32_t* chain = buckets + nbuckets;
    for (uint32_t i = buckets[h % nbuckets]; i >= symoffset; ++i) {
        const uint32_t ch = chain[i - symoffset];
        if ((ch | 1) == (h | 1) && !std::strcmp(strtab + symtab[i].st_name, name) && symtab[i].st_value)
            return map->l_addr + symtab[i].st_value;
        if (ch & 1) break;
    }
    return 0;
}

void findPtxGetters() {
    if (!nvrtcLibrary) return;
    if (!realPtxSize) realPtxSize = reinterpret_cast<SizeFn>(lookup(nvrtcLibrary, "nvrtcGetPTXSize"));
    if (!realPtx) realPtx = reinterpret_cast<CodeFn>(lookup(nvrtcLibrary, "nvrtcGetPTX"));
}

int compileProgram(void* program, int numOptions, const char* const* options) {
    if (!realCompile) return kInternalError;
    std::vector<std::string> rewritten(options, options + (numOptions > 0 ? numOptions : 0));
    for (std::string& o : rewritten)
        for (const char* flag : {"-arch=sm_", "--gpu-architecture=sm_"})
            if (o.rfind(flag, 0) == 0) o.replace(o.find("sm_"), 3, "compute_");
    std::vector<const char*> pointers;
    for (const std::string& o : rewritten) pointers.push_back(o.c_str());
    return realCompile(program, numOptions, pointers.data());
}

bool hasPtx(void* program, size_t* size) {
    findPtxGetters();
    return realPtxSize && realPtx && realPtxSize(program, size) == 0 && *size > 1;
}

int cubinSize(void* program, size_t* size) {
    if (hasPtx(program, size)) return 0;
    return realCubinSize ? realCubinSize(program, size) : kInternalError;
}

int cubin(void* program, char* code) {
    size_t size = 0;
    if (hasPtx(program, &size)) return realPtx(program, code);
    return realCubin ? realCubin(program, code) : kInternalError;
}

}  // namespace

extern "C" {

unsigned la_version(unsigned) { return LAV_CURRENT; }

// Bindings to and from every object are reported (no PLT enter/exit hooks: no per-call cost).
unsigned la_objopen(struct link_map* map, Lmid_t, uintptr_t*) {
    if (map->l_name && std::strstr(map->l_name, "libnvrtc.so")) nvrtcLibrary = map;
    return LA_FLG_BINDTO | LA_FLG_BINDFROM;
}

uintptr_t la_symbind64(Elf64_Sym* sym, unsigned, uintptr_t*, uintptr_t*, unsigned*, const char* name) {
    if (std::strncmp(name, "nvrtc", 5) != 0) return sym->st_value;
    const uintptr_t real = sym->st_value;
    if (!std::strcmp(name, "nvrtcGetPTXSize")) realPtxSize = reinterpret_cast<SizeFn>(real);
    else if (!std::strcmp(name, "nvrtcGetPTX")) realPtx = reinterpret_cast<CodeFn>(real);
    else if (!std::strcmp(name, "nvrtcCompileProgram")) {
        realCompile = reinterpret_cast<CompileFn>(real);
        return reinterpret_cast<uintptr_t>(&compileProgram);
    } else if (!std::strcmp(name, "nvrtcGetCUBINSize")) {
        realCubinSize = reinterpret_cast<SizeFn>(real);
        return reinterpret_cast<uintptr_t>(&cubinSize);
    } else if (!std::strcmp(name, "nvrtcGetCUBIN")) {
        realCubin = reinterpret_cast<CodeFn>(real);
        return reinterpret_cast<uintptr_t>(&cubin);
    }
    return real;
}

}  // extern "C"
