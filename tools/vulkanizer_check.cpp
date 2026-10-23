// Tells whether CUDA programs and libraries can run on Vulkanizer, without running them.
//   vulkanizer-check [-v] [-q] PATH...   executables, shared libraries or directories (searched recursively)
//   vulkanizer-check --dump-ptx DIR [--cc 75] PATH...
//                                        also writes the PTX of every fat binary (the one Vulkanizer would pick
//                                        for that compute capability) to DIR/<file>.<n>.ptx, for vulkanizer-ptxc
//
// For every ELF file with CUDA device code or a CUDA runtime it reports:
//   ok       every fat binary has PTX
//   partial  some fat binaries have only native code (SASS): their kernels cannot be loaded
//   sass     only native code: nothing can run
//   static   the CUDA runtime is linked in statically: needs a rebuild with -cudart shared
//   replaced an NVIDIA library Vulkanizer brings its own of (vulkanizer-run and run_torch.sh load that one)
//   nvidia   NVIDIA cuDNN, which Vulkanizer does not provide (PyTorch runs without it)
// The exit status is 0 when nothing is in the way (replaced and cuDNN do not count), 1 otherwise.

#include "ptx/fatbin.h"

#include <elf.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;
namespace fatbin = vulkanizer::fatbin;

namespace {

struct Mapping {
    const uint8_t* data = nullptr;
    size_t size = 0;
    explicit Mapping(const char* path) {
        const int fd = open(path, O_RDONLY | O_CLOEXEC);
        if (fd < 0) return;
        struct stat st;
        if (fstat(fd, &st) == 0 && st.st_size > 0) {
            void* p = mmap(nullptr, size_t(st.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
            if (p != MAP_FAILED) data = static_cast<const uint8_t*>(p), size = size_t(st.st_size);
        }
        close(fd);
    }
    ~Mapping() {
        if (data) munmap(const_cast<uint8_t*>(data), size);
    }
    Mapping(const Mapping&) = delete;
    Mapping& operator=(const Mapping&) = delete;
};

struct Report {
    size_t modules = 0, withPtx = 0;
    std::set<uint32_t> ptxTargets, nativeTargets, nativeOnlyTargets;
    std::vector<std::string> needed;
    std::string soname;
    bool staticRuntime = false;
    std::vector<const uint8_t*> fatbins;  // headers of those with PTX
};

// NVIDIA libraries Vulkanizer has its own of; libnvrtc and libnvJitLink are redirected by libvulkanizer_nvrtc.
bool replaced(const std::string& name) {
    static const char* const kNames[] = {"libcuda.so",     "libcudart.so", "libcublas.so",   "libcublasLt.so",
                                         "libcusolver.so", "libcufft.so",  "libcusparse.so", "libcurand.so",
                                         "libnccl.so",     "libnvrtc.so",  "libnvrtc-builtins", "libnvJitLink.so"};
    return std::any_of(std::begin(kNames), std::end(kNames), [&](const char* n) { return name.rfind(n, 0) == 0; });
}

bool contains(const Mapping& m, std::string_view s) {
    return memmem(m.data, m.size, s.data(), s.size()) != nullptr;
}

// The fat binaries of a .nv_fatbin section: one per translation unit, each 8-byte aligned.
void scanFatbins(const uint8_t* p, size_t size, Report& r) {
    size_t at = 0;
    while (at + 16 <= size) {
        uint32_t magic;
        std::memcpy(&magic, p + at, 4);
        if (magic != fatbin::kHeaderMagic) {
            at += 8;
            continue;
        }
        uint16_t headerSize;
        uint64_t fatSize;
        std::memcpy(&headerSize, p + at + 6, 2);
        std::memcpy(&fatSize, p + at + 8, 8);
        if (headerSize < 16 || fatSize > size - at - headerSize) break;
        std::vector<fatbin::Entry> entries;
        std::string error;
        if (fatbin::parse(p + at, entries, error) && !entries.empty()) {
            ++r.modules;
            bool ptx = false;
            for (const fatbin::Entry& e : entries) {
                if (e.kind == uint16_t(fatbin::Kind::Ptx)) ptx = true, r.ptxTargets.insert(e.smVersion);
                else if (e.kind == uint16_t(fatbin::Kind::Elf)) r.nativeTargets.insert(e.smVersion);
            }
            if (ptx) ++r.withPtx, r.fatbins.push_back(p + at);
            else
                for (const fatbin::Entry& e : entries)
                    if (e.kind == uint16_t(fatbin::Kind::Elf)) r.nativeOnlyTargets.insert(e.smVersion);
        }
        at += (headerSize + fatSize + 7) & ~uint64_t(7);
    }
}

// False when the file is not a 64-bit little-endian ELF.
bool inspect(const Mapping& m, Report& r) {
    if (m.size < sizeof(Elf64_Ehdr) || std::memcmp(m.data, ELFMAG, SELFMAG) != 0 || m.data[EI_CLASS] != ELFCLASS64 ||
        m.data[EI_DATA] != ELFDATA2LSB)
        return false;
    Elf64_Ehdr eh;
    std::memcpy(&eh, m.data, sizeof eh);
    if (!eh.e_shoff || eh.e_shentsize != sizeof(Elf64_Shdr) || eh.e_shoff + uint64_t(eh.e_shnum) * sizeof(Elf64_Shdr) > m.size)
        return true;
    std::vector<Elf64_Shdr> sh(eh.e_shnum);
    std::memcpy(sh.data(), m.data + eh.e_shoff, sh.size() * sizeof(Elf64_Shdr));
    auto inFile = [&](const Elf64_Shdr& s) { return s.sh_type != SHT_NOBITS && s.sh_offset + s.sh_size <= m.size; };
    const Elf64_Shdr* names = eh.e_shstrndx < sh.size() && inFile(sh[eh.e_shstrndx]) ? &sh[eh.e_shstrndx] : nullptr;
    for (const Elf64_Shdr& s : sh) {
        if (!inFile(s)) continue;
        if (names && s.sh_name < names->sh_size) {
            const char* name = reinterpret_cast<const char*>(m.data + names->sh_offset + s.sh_name);
            if (!std::strcmp(name, ".nv_fatbin") || !std::strcmp(name, "__nv_relfatbin"))
                scanFatbins(m.data + s.sh_offset, s.sh_size, r);
        }
        if (s.sh_type == SHT_DYNAMIC && s.sh_link < sh.size() && inFile(sh[s.sh_link])) {
            const Elf64_Shdr& str = sh[s.sh_link];
            auto text = [&](uint64_t off) {
                return off < str.sh_size ? std::string(reinterpret_cast<const char*>(m.data + str.sh_offset + off),
                                                       strnlen(reinterpret_cast<const char*>(m.data + str.sh_offset + off),
                                                               str.sh_size - off))
                                         : std::string();
            };
            for (size_t i = 0; i < s.sh_size / sizeof(Elf64_Dyn); ++i) {
                Elf64_Dyn d;
                std::memcpy(&d, m.data + s.sh_offset + i * sizeof d, sizeof d);
                if (d.d_tag == DT_NULL) break;
                if (d.d_tag == DT_NEEDED) r.needed.push_back(text(d.d_un.d_val));
                else if (d.d_tag == DT_SONAME) r.soname = text(d.d_un.d_val);
            }
        }
    }
    const bool sharedRuntime = std::any_of(r.needed.begin(), r.needed.end(),
                                           [](const std::string& n) { return n.rfind("libcudart.so", 0) == 0; });
    const bool isRuntime = r.soname.rfind("libcudart.so", 0) == 0;
    // The runtime's error names (cudaGetErrorName) are in every copy of it.
    r.staticRuntime = !sharedRuntime && !isRuntime && contains(m, "cudaErrorInvalidDeviceFunction");
    return true;
}

std::string targets(const std::set<uint32_t>& sms, const char* prefix) {
    std::string s;
    for (uint32_t sm : sms) s += (s.empty() ? "" : ", ") + std::string(prefix) + std::to_string(sm);
    return s;
}

struct Options {
    bool verbose = false, quiet = false;
    fs::path dumpDir;
    uint32_t cc = 75;
};

void dumpPtx(const fs::path& path, const Report& r, const Options& opt) {
    for (size_t i = 0; i < r.fatbins.size(); ++i) {
        std::string ptx, error;
        uint32_t sm = 0;
        if (!fatbin::extractPtx(r.fatbins[i], opt.cc, ptx, sm, error)) {
            std::fprintf(stderr, "%s: fat binary %zu: %s\n", path.c_str(), i, error.c_str());
            continue;
        }
        const fs::path out = opt.dumpDir / (path.filename().string() + "." + std::to_string(i) + ".ptx");
        if (FILE* f = std::fopen(out.c_str(), "wb")) {
            std::fwrite(ptx.data(), 1, ptx.size(), f);
            std::fclose(f);
        }
    }
}

int problems = 0, checked = 0;

void check(const fs::path& path, const Options& opt) {
    const std::string file = path.filename().string();
    Mapping m(path.c_str());
    if (!m.data) return;
    Report r;
    if (!inspect(m, r)) return;
    const std::string& name = r.soname.empty() ? file : r.soname;
    const bool cudnn = name.rfind("libcudnn", 0) == 0;
    const bool replacedLib = replaced(name);
    if (!r.modules && !r.staticRuntime && !cudnn && !replacedLib) {
        if (opt.verbose) std::printf("-        %s (no CUDA device code)\n", path.c_str());
        return;
    }
    ++checked;
    const char* verdict;
    std::string detail;
    bool fine = false;
    if (replacedLib) {
        verdict = "replaced";
        detail = "NVIDIA's " + name + ": Vulkanizer's own is used instead (vulkanizer-run, run.sh, run_torch.sh)";
        fine = true;
    } else if (cudnn) {
        verdict = "nvidia";
        fine = true;
        detail = "NVIDIA cuDNN: not supported; PyTorch runs without it (vulkanizer-run --torch turns it off)";
    } else if (r.staticRuntime) {
        verdict = "static";
        detail = "CUDA runtime linked in statically (needs NVIDIA's driver): rebuild with -cudart shared";
    } else if (r.withPtx == r.modules) {
        verdict = "ok";
        fine = true;
        detail = std::to_string(r.modules) + (r.modules == 1 ? " module" : " modules") + ", PTX " +
                 targets(r.ptxTargets, "compute_");
    } else if (r.withPtx) {
        verdict = "partial";
        detail = std::to_string(r.modules - r.withPtx) + " of " + std::to_string(r.modules) +
                 " modules have only native code (" + targets(r.nativeOnlyTargets, "sm_") +
                 "): their kernels cannot be loaded";
    } else {
        verdict = "sass";
        detail = "only native code (" + targets(r.nativeTargets, "sm_") +
                 "), no PTX: rebuild with a virtual architecture (nvcc -arch=compute_61)";
    }
    if (!fine) ++problems;
    if (!opt.dumpDir.empty()) dumpPtx(path, r, opt);
    if (fine && opt.quiet) return;
    std::printf("%-8s %s\n         %s\n", verdict, path.c_str(), detail.c_str());
    if (opt.verbose && r.modules && !r.nativeTargets.empty())
        std::printf("         native code: %s\n", targets(r.nativeTargets, "sm_").c_str());
}

void walk(const fs::path& root, const Options& opt) {
    std::error_code ec;
    if (!fs::is_directory(root, ec)) {
        check(root, opt);
        return;
    }
    // Symbolic links are skipped: the versioned library they point to is visited anyway.
    for (fs::recursive_directory_iterator it(root, fs::directory_options::skip_permission_denied, ec), end;
         it != end; it.increment(ec)) {
        if (ec) break;
        if (it->is_symlink(ec) || !it->is_regular_file(ec)) continue;
        const std::string name = it->path().filename().string();
        const bool library = name.find(".so") != std::string::npos;
        const bool executable = (it->status(ec).permissions() & fs::perms::owner_exec) != fs::perms::none;
        if ((library || executable) && it->file_size(ec) >= sizeof(Elf64_Ehdr)) check(it->path(), opt);
    }
}

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    std::vector<fs::path> paths;
    bool help = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-v")) opt.verbose = true;
        else if (!std::strcmp(argv[i], "-q")) opt.quiet = true;
        else if (!std::strcmp(argv[i], "--dump-ptx") && i + 1 < argc) opt.dumpDir = argv[++i];
        else if (!std::strcmp(argv[i], "--cc") && i + 1 < argc) opt.cc = uint32_t(std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "-h") || !std::strcmp(argv[i], "--help")) help = true;
        else paths.emplace_back(argv[i]);
    }
    if (help || paths.empty()) {
        std::fprintf(stderr, "usage: vulkanizer-check [-v] [-q] [--dump-ptx DIR [--cc 75]] PATH...  (executables, "
                             "libraries or directories)\n");
        return 2;
    }
    if (!opt.dumpDir.empty()) {
        std::error_code ec;
        fs::create_directories(opt.dumpDir, ec);
    }
    for (const fs::path& p : paths) {
        std::error_code ec;
        if (!fs::exists(p, ec)) {
            std::fprintf(stderr, "%s: not found\n", p.c_str());
            ++problems;
            continue;
        }
        walk(p, opt);
    }
    if (!checked) std::printf("no CUDA device code or CUDA runtime found\n");
    else if (!problems) std::printf("everything can run on Vulkanizer\n");
    return problems ? 1 : 0;
}
