// Private interfaces NVIDIA's CUDA runtime asks the driver for with cuGetExportTable (plan.md 6.5):
// programs and libraries that link the runtime statically (nvcc's default, CuPy, NVIDIA's NCCL) do not
// start without them. They are undocumented. Shelved (plan.md 6.5): the runtime also requires an
// integrity check that cannot be met clean-room, so these tables do not get it running. Only built
// with the CMake option VULKANIZER_EXPORT_TABLES, and at run time off unless VULKANIZER_EXPORT_TABLES is set:
//   VULKANIZER_EXPORT_TABLES=1      the tables implemented so far
//   VULKANIZER_EXPORT_TABLES=probe  also, for every other interface, a table of logging functions: each
//                                   slot reports its number and register arguments, then fails, which
//                                   shows what the runtime calls (one slot per run)
// Clean-room: only what the runtime is seen to do with these tables, no NVIDIA code or disassembly.

#include "driver/driver.h"

#include "common/log.h"

#include <sys/uio.h>
#include <unistd.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace vulkanizer::driver {

namespace {

std::string uuidText(const CUuuid& id) {
    char hex[33];
    for (int i = 0; i < 16; ++i) std::snprintf(hex + 2 * i, 3, "%02x", static_cast<unsigned char>(id.bytes[i]));
    return hex;
}

// ---- probe tables ------------------------------------------------------------------------------

constexpr size_t kProbeSlots = 64, kProbeTables = 16;

std::array<std::string, kProbeTables> g_probed;  // the interface of each probe table

// VULKANIZER_EXPORT_PROBE_OK=<uuid prefix>:<slot>,...: those slots return 0 instead of failing, so the
// runtime gets on to its next call.
bool probeSucceeds(const std::string& uuid, size_t slot) {
    const char* env = std::getenv("VULKANIZER_EXPORT_PROBE_OK");
    if (!env) return false;
    std::string list = env;
    for (size_t at = 0; at < list.size();) {
        size_t end = list.find(',', at);
        if (end == std::string::npos) end = list.size();
        const std::string item = list.substr(at, end - at);
        const size_t colon = item.find(':');
        if (colon != std::string::npos && uuid.rfind(item.substr(0, colon), 0) == 0 &&
            std::strtoul(item.c_str() + colon + 1, nullptr, 10) == slot)
            return true;
        at = end + 1;
    }
    return false;
}

// The first bytes at `p` if it is readable memory of this process, for the probe log.
std::string peek(uintptr_t p) {
    if (p < 4096) return "";
    uint8_t bytes[32];
    iovec local{bytes, sizeof bytes}, remote{reinterpret_cast<void*>(p), sizeof bytes};
    if (process_vm_readv(getpid(), &local, 1, &remote, 1, 0) != ssize_t(sizeof bytes)) return "";
    std::string s = " [";
    for (size_t i = 0; i < sizeof bytes; i += 8) {
        uint64_t q;
        std::memcpy(&q, bytes + i, 8);
        char w[24];
        std::snprintf(w, sizeof w, "%s%#lx", i ? " " : "", (unsigned long)q);
        s += w;
    }
    return s + "]";
}

template <size_t T, size_t I>
uintptr_t probeSlot(uintptr_t a, uintptr_t b, uintptr_t c, uintptr_t d, uintptr_t e, uintptr_t f) {
    const bool ok = probeSucceeds(g_probed[T], I);
    std::fprintf(stderr, "[vulkanizer:probe] %s slot %zu (", g_probed[T].c_str(), I);
    const uintptr_t args[] = {a, b, c, d, e, f};
    for (int i = 0; i < 6; ++i) std::fprintf(stderr, "%s%#lx%s", i ? ", " : "", (unsigned long)args[i], peek(args[i]).c_str());
    std::fprintf(stderr, ") -> %s\n", ok ? "0" : "NOT_SUPPORTED");
    std::fflush(stderr);
    return ok ? 0 : CUDA_ERROR_NOT_SUPPORTED;
}

template <size_t T, size_t... I>
std::array<const void*, kProbeSlots> probeSlots(std::index_sequence<I...>) {
    return {reinterpret_cast<const void*>(&probeSlot<T, I>)...};
}

template <size_t... T>
std::array<std::array<const void*, kProbeSlots>, kProbeTables> probeTables(std::index_sequence<T...>) {
    return {probeSlots<T>(std::make_index_sequence<kProbeSlots>())...};
}

// Slot 0 of most tables is their size in bytes, which the runtime checks before using a slot.
const void* const* probeTable(const std::string& uuid) {
    static auto tables = probeTables(std::make_index_sequence<kProbeTables>());
    static std::map<std::string, size_t> assigned;
    auto it = assigned.find(uuid);
    if (it == assigned.end()) {
        if (assigned.size() == kProbeTables) return nullptr;
        it = assigned.emplace(uuid, assigned.size()).first;
        g_probed[it->second] = uuid;
        tables[it->second][0] = reinterpret_cast<const void*>(uintptr_t(kProbeSlots * sizeof(void*)));
    }
    return tables[it->second].data();
}

// ---- the runtime's interface (6bd5fb6c...) ---------------------------------------------------

// slot 2 (ctx out, device): the device's context for the runtime, retained.
CUresult runtimeContext(CUcontext* ctx, CUdevice dev) {
    VULKANIZER_DEBUG("export table: runtime context of device %d", dev);
    return cuDevicePrimaryCtxRetain(ctx, dev);
}

// ---- a094798c...: storage the runtime asks for ---------------------------------------------------

// slots 2 and 6 (void** area, size_t* bytes): a zeroed area for the runtime, one per slot, which it
// keeps (what it stores there is its own business; nothing here reads it).
template <size_t Slot>
void runtimeStorage(void** area, size_t* bytes) {
    alignas(64) static uint8_t storage[4096];
    VULKANIZER_DEBUG("export table: runtime storage %zu at %p (%zu bytes)", Slot, static_cast<void*>(storage),
                     sizeof storage);
    if (area) *area = storage;
    if (bytes) *bytes = sizeof storage;
}

// Implemented slots by interface; the others of an implemented interface stay probe functions
// (they log their call and fail).
struct Implemented {
    const char* uuid;
    std::vector<std::pair<size_t, const void*>> slots;
};

const std::vector<Implemented>& implemented() {
    static const std::vector<Implemented> tables = {
        {"6bd5fb6c5bf4e74a8987d93912fd9df9", {{2, reinterpret_cast<const void*>(&runtimeContext)}}},
        {"a094798c2e742e7493f20800200c0a66",
         {{2, reinterpret_cast<const void*>(&runtimeStorage<2>)}, {6, reinterpret_cast<const void*>(&runtimeStorage<6>)}}},
    };
    return tables;
}

}  // namespace

// The table for `id`, or nullptr (CUDA_ERROR_NOT_SUPPORTED) when it is not provided.
const void* exportTable(const CUuuid& id) {
    static const char* const mode = std::getenv("VULKANIZER_EXPORT_TABLES");
    if (!mode || !*mode || !std::strcmp(mode, "0")) return nullptr;
    const std::string text = uuidText(id);
    for (const Implemented& t : implemented()) {
        if (text != t.uuid) continue;
        const void* const* table = probeTable(text);
        if (!table) return nullptr;
        auto* slots = const_cast<const void**>(table);
        for (const auto& [slot, fn] : t.slots) slots[slot] = fn;
        return table;
    }
    if (!std::strcmp(mode, "probe")) {
        // VULKANIZER_EXPORT_PROBE_DENY=<uuid prefix>,...: refused like unknown interfaces
        if (const char* deny = std::getenv("VULKANIZER_EXPORT_PROBE_DENY")) {
            for (const char* p = deny; *p;) {
                const char* end = std::strchr(p, ',');
                const size_t n = end ? size_t(end - p) : std::strlen(p);
                if (n && text.compare(0, n, p, n) == 0) {
                    VULKANIZER_WARN("cuGetExportTable: refusing interface %s", text.c_str());
                    return nullptr;
                }
                p += n + (end ? 1 : 0);
            }
        }
        VULKANIZER_WARN("cuGetExportTable: probing interface %s", text.c_str());
        return probeTable(text);
    }
    VULKANIZER_WARN("cuGetExportTable: unknown interface %s", text.c_str());
    return nullptr;
}

}  // namespace vulkanizer::driver
