#include "driver/kernel_cache.h"

#include "common/log.h"
#include "common/sha256.h"

#include <dirent.h>
#include <dlfcn.h>
#include <link.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <vector>

namespace vulkanizer::driver {

namespace {

constexpr uint32_t kMagic = 0x335A4B56;  // "VKZ3"

// GNU build-id of the library containing this code: any rebuild invalidates the cache.
std::string buildId() {
    static const std::string id = [] {
        Dl_info info{};
        if (!dladdr(reinterpret_cast<void*>(&kernelCacheKey), &info)) return std::string();
        struct Ctx {
            const void* base;
            std::string id;
        } ctx{info.dli_fbase, {}};
        dl_iterate_phdr(
            [](dl_phdr_info* pi, size_t, void* data) {
                auto* c = static_cast<Ctx*>(data);
                if (reinterpret_cast<const void*>(pi->dlpi_addr) != c->base) return 0;
                for (int i = 0; i < pi->dlpi_phnum; ++i) {
                    const ElfW(Phdr)& ph = pi->dlpi_phdr[i];
                    if (ph.p_type != PT_NOTE) continue;
                    auto* p = reinterpret_cast<const uint8_t*>(pi->dlpi_addr + ph.p_vaddr);
                    const uint8_t* end = p + ph.p_memsz;
                    while (p + sizeof(ElfW(Nhdr)) <= end) {
                        auto* n = reinterpret_cast<const ElfW(Nhdr)*>(p);
                        const uint8_t* name = p + sizeof(*n);
                        const uint8_t* desc = name + ((n->n_namesz + 3) & ~3u);
                        if (n->n_type == NT_GNU_BUILD_ID && n->n_namesz == 4 && !std::memcmp(name, "GNU", 4)) {
                            static const char* hex = "0123456789abcdef";
                            for (uint32_t k = 0; k < n->n_descsz; ++k)
                                (c->id += hex[desc[k] >> 4]) += hex[desc[k] & 15];
                            return 1;
                        }
                        p = desc + ((n->n_descsz + 3) & ~3u);
                    }
                }
                return 1;
            },
            &ctx);
        return ctx.id;
    }();
    return id;
}

std::string cacheDir() {
    static const std::string dir = []() -> std::string {
        std::string base;
        if (const char* env = std::getenv("VULKANIZER_CACHE_DIR"); env && *env) {
            if (!std::strcmp(env, "off") || !std::strcmp(env, "0")) return {};
            base = env;
        } else if (const char* xdg = std::getenv("XDG_CACHE_HOME"); xdg && *xdg) {
            base = std::string(xdg) + "/vulkanizer";
        } else if (const char* home = std::getenv("HOME"); home && *home) {
            base = std::string(home) + "/.cache/vulkanizer";
        } else {
            return {};
        }
        if (buildId().empty()) {
            VULKANIZER_WARN("no build-id in libcuda.so: the kernel cache is disabled");
            return {};
        }
        // mkdir -p
        std::string path = base + "/kernels";
        for (size_t i = 1; i <= path.size(); ++i) {
            if (i == path.size() || path[i] == '/') {
                const std::string part = path.substr(0, i);
                if (mkdir(part.c_str(), 0755) != 0 && errno != EEXIST) {
                    VULKANIZER_WARN("kernel cache disabled: cannot create %s", part.c_str());
                    return {};
                }
            }
        }
        return path;
    }();
    return dir;
}

// The cache's size limit (VULKANIZER_CACHE_SIZE, MiB; default 256).
uint64_t cacheLimitBytes() {
    const char* env = std::getenv("VULKANIZER_CACHE_SIZE");
    const long long mib = env && *env ? std::atoll(env) : 256;
    return uint64_t(std::max(mib, 1LL)) << 20;
}

// Least recently used first (a hit refreshes a file's mtime), the cache shrinks to 3/4 of its
// limit once it is over it. Entries of earlier builds are never hit again, so they go first.
// Leftovers of interrupted writes (*.tmp<pid>) older than an hour go too.
void trimCache(const std::string& dir) {
    struct Entry {
        time_t mtime;
        uint64_t size;
        std::string path;
    };
    std::vector<Entry> entries;
    uint64_t total = 0;
    const time_t now = time(nullptr);
    DIR* d = opendir(dir.c_str());
    if (!d) return;
    while (const dirent* e = readdir(d)) {
        const std::string name = e->d_name, path = dir + "/" + name;
        struct stat st;
        if (stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) continue;
        if (name.find(".tmp") != std::string::npos) {
            if (now - st.st_mtime > 3600) unlink(path.c_str());
        } else if (name.size() > 4 && name.compare(name.size() - 4, 4, ".vkz") == 0) {
            entries.push_back({st.st_mtime, uint64_t(st.st_size), path});
            total += uint64_t(st.st_size);
        }
    }
    closedir(d);
    const uint64_t limit = cacheLimitBytes();
    if (total <= limit) return;
    std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) { return a.mtime < b.mtime; });
    size_t removed = 0;
    for (const Entry& e : entries) {
        if (total <= limit / 4 * 3) break;
        if (unlink(e.path.c_str()) == 0) ++removed;  // another process may have removed it already
        total -= e.size;
    }
    VULKANIZER_INFO("kernel cache over %llu MiB: removed %zu least recently used entries",
                    static_cast<unsigned long long>(limit >> 20), removed);
}

}  // namespace

std::string kernelCacheKey(const std::string& moduleHash, const std::string& entry, const compiler::Target& t) {
    Sha256 h;
    h.update(buildId());
    h.update(moduleHash);
    h.update(entry);
    const uint32_t fields[] = {t.subgroupSize, t.maxPushConstantsSize, t.maxSharedMemorySize,
                               uint32_t(t.subgroupBallot) | uint32_t(t.subgroupShuffle) << 1 |
                                   uint32_t(t.float16) << 2 | uint32_t(t.float64) << 3 | uint32_t(t.int8) << 4 |
                                   uint32_t(t.int16) << 5 | uint32_t(t.storage8) << 6 | uint32_t(t.storage16) << 7 |
                                   uint32_t(t.int64Atomics) << 8 | uint32_t(t.integerDotProduct) << 9 | uint32_t(t.fusedFma) << 10 | uint32_t(t.fmaInF64) << 11 | uint32_t(t.exactFtzFma) << 13 |
                                   uint32_t(t.fusedFma64) << 12};
    h.update(fields, sizeof fields);
    return h.hexDigest();
}

bool loadCachedKernel(const std::string& key, compiler::KernelBinary& out) {
    const std::string dir = cacheDir();
    if (dir.empty()) return false;
    const std::string path = dir + "/" + key + ".vkz";
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    uint32_t h[12];
    if (!f.read(reinterpret_cast<char*>(h), sizeof h) || h[0] != kMagic) return false;
    compiler::KernelBinary b;
    b.pushConstantSize = h[1];
    b.globalsOffset = h[2];
    b.gridOffset = h[3];
    b.printfOffset = h[4];
    b.paramsInBuffer = (h[5] & 1) != 0;
    b.packable = (h[5] & 2) != 0;
    b.staticSharedBytes = h[6];
    b.localBytes = h[7];
    b.entryPoint.resize(h[8]);
    b.spirv.resize(h[9]);
    b.workspaceOffset = h[10];
    b.assertOffset = h[11];
    if (!f.read(b.entryPoint.data(), std::streamsize(b.entryPoint.size())) ||
        !f.read(reinterpret_cast<char*>(b.spirv.data()), std::streamsize(b.spirv.size() * 4)))
        return false;
    out = std::move(b);
    utimes(path.c_str(), nullptr);  // recently used: trimCache keeps it
    return true;
}

void storeCachedKernel(const std::string& key, const compiler::KernelBinary& b) {
    const std::string dir = cacheDir();
    if (dir.empty()) return;
    const std::string path = dir + "/" + key + ".vkz";
    const std::string tmp = path + ".tmp" + std::to_string(getpid());
    {
        std::ofstream f(tmp, std::ios::binary);
        const uint32_t h[12] = {kMagic,       b.pushConstantSize,  b.globalsOffset,           b.gridOffset,
                                b.printfOffset, uint32_t(b.paramsInBuffer) | uint32_t(b.packable) << 1, b.staticSharedBytes, b.localBytes,
                                uint32_t(b.entryPoint.size()), uint32_t(b.spirv.size()), b.workspaceOffset, b.assertOffset};
        f.write(reinterpret_cast<const char*>(h), sizeof h);
        f.write(b.entryPoint.data(), std::streamsize(b.entryPoint.size()));
        f.write(reinterpret_cast<const char*>(b.spirv.data()), std::streamsize(b.spirv.size() * 4));
        if (!f) {
            std::remove(tmp.c_str());
            return;
        }
    }
    std::rename(tmp.c_str(), path.c_str());  // atomic: concurrent processes never see partial files
    static std::once_flag trimmed;  // once per process that compiles anything
    std::call_once(trimmed, [&] { trimCache(dir); });
}

}  // namespace vulkanizer::driver
