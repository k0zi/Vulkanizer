// AMD XDNA backend through XRT (amdxdna kernel driver + xrt_coreutil). Built only with
// -DVULKANIZER_NPU_XRT=ON. UNTESTED: written on a machine without an NPU, to be brought up on
// XDNA hardware following plan.md 7. (N1).
//
// The NPU has no JIT here: GEMMs run on precompiled kernels, one per size class, built with
// mlir-aie (IRON, programming_examples/basic/matrix_multiplication) into
//   <dir>/gemm_<in>_<out>_<M>x<K>x<N>.xclbin  and  <dir>/gemm_<in>_<out>_<M>x<K>x<N>.insts.bin
// in: bf16 (later i8), out: f32 or bf16; row-major lhs M x K, rhs K x N, result M x N; host
// signature (opcode 3, instructions, count, lhs, rhs, out, tmp, trace) as in the mlir-aie host
// examples (programming_examples/basic/matrix_multiplication/test.cpp; tmp and trace unused, 1 B / 4 B).
// <dir>: VULKANIZER_NPU_KERNELS, else $XDG_DATA_HOME/vulkanizer/npu (~/.local/share/...).
//
// A GEMM takes the smallest kernel that holds it, zero-padded (C^T = op(B)^T op(A)^T, see host.h).
// The operands are copied into XRT buffers here (N1); zero-copy dma-buf import is N2.

#include "npu/host.h"
#include "npu/npu.h"

#include "common/log.h"

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#if __has_include(<xrt/xrt_xclbin.h>)
#include <xrt/xrt_xclbin.h>
#else
#include <xrt/experimental/xrt_xclbin.h>
#endif

#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <optional>
#include <regex>
#include <string>
#include <vector>

namespace vulkanizer::npu {

namespace {

namespace fs = std::filesystem;

constexpr unsigned kOpcode = 3;  // mlir-aie: run the instruction sequence

struct KernelFile {
    DType in = DType::BF16, out = DType::F32;
    int64_t m = 0, k = 0, n = 0;
    fs::path xclbin, insts;
};

// A kernel's operand and result memory (sized for its M, K, N): anonymous pages, which the NPU takes
// as user pointer buffers and the GPU can import as well (zero copy, plan.md 7.3). Kept for the
// backend's lifetime, also when the kernel is unloaded: the GPU may have them registered.
using Pages = std::array<HostPages, 3>;  // lhs, rhs, out

// A kernel loaded on the NPU (a hardware context), with buffers over its Pages.
struct Loaded {
    Pages* pages = nullptr;
    uint64_t lastUse = 0;
    xrt::hw_context context;
    xrt::kernel kernel;
    xrt::bo instr, lhs, rhs, out, tmp, trace;
    uint32_t instrWords = 0;
    const KernelFile* file = nullptr;
};

std::optional<DType> parseType(const std::string& s) {
    if (s == "f32") return DType::F32;
    if (s == "bf16") return DType::BF16;
    if (s == "f16") return DType::F16;
    return std::nullopt;
}

fs::path kernelDir() {
    if (const char* d = std::getenv("VULKANIZER_NPU_KERNELS"); d && *d) return d;
    if (const char* x = std::getenv("XDG_DATA_HOME"); x && *x) return fs::path(x) / "vulkanizer" / "npu";
    if (const char* h = std::getenv("HOME"); h && *h) return fs::path(h) / ".local" / "share" / "vulkanizer" / "npu";
    return {};
}

std::vector<KernelFile> scan(const fs::path& dir) {
    static const std::regex name(R"(gemm_([a-z0-9]+)_([a-z0-9]+)_(\d+)x(\d+)x(\d+)\.xclbin)");
    std::vector<KernelFile> out;
    std::error_code ec;
    for (const fs::directory_entry& e : fs::directory_iterator(dir, ec)) {
        std::smatch mt;
        const std::string file = e.path().filename().string();
        if (!std::regex_match(file, mt, name)) continue;
        const auto in = parseType(mt[1]), o = parseType(mt[2]);
        fs::path insts = e.path();
        insts.replace_extension(".insts.bin");
        if (!in || !o || !fs::exists(insts)) {
            VULKANIZER_WARN("npu: %s: unknown types or no %s", file.c_str(), insts.filename().c_str());
            continue;
        }
        out.push_back({*in, *o, std::stoll(mt[3]), std::stoll(mt[4]), std::stoll(mt[5]), e.path(), insts});
    }
    return out;
}

std::vector<uint32_t> readWords(const fs::path& file) {
    std::ifstream in(file, std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::vector<uint32_t> words(bytes.size() / 4);
    std::memcpy(words.data(), bytes.data(), words.size() * 4);
    return words;
}

class XdnaBackend final : public Backend {
public:
    XdnaBackend(xrt::device device, std::vector<KernelFile> files) : device_(std::move(device)), files_(std::move(files)) {}

    const char* name() const override { return "xdna"; }

    int64_t volume(const GemmShape& s) const override {
        const KernelFile* f = pick(s);
        return f ? f->m * f->k * f->n : 0;
    }

    GemmTimes lastTimes() const override { return times_; }

    bool canStage() const override { return true; }

    bool stage(const GemmShape& s, Staging& st) override {
        const KernelFile* f = pick(s);
        if (!f) return false;
        std::lock_guard<std::mutex> lock(mutex_);
        try {
            Loaded& k = load(*f);
            const Pages& p = *k.pages;
            st = {f->in, f->out, f->m, f->k, f->n, p[0].data(), p[1].data(), p[2].data(), p[0].size(), p[1].size(),
                  p[2].size()};
            return true;
        } catch (const std::exception& e) {
            VULKANIZER_ERROR("npu: %s: %s", f->xclbin.filename().c_str(), e.what());
            return false;
        }
    }

    bool runStaged(const Staging& st) override {
        std::lock_guard<std::mutex> lock(mutex_);
        const KernelFile* f = nullptr;
        for (const KernelFile& file : files_)
            if (auto it = pages_.find(file.xclbin.string()); it != pages_.end() && it->second[0].data() == st.lhs) f = &file;
        if (!f) return false;
        Loaded* k = nullptr;
        try {
            k = &load(*f);  // usually loaded by stage(), unless evicted since
            using Clock = std::chrono::steady_clock;
            const Clock::time_point t0 = Clock::now();
            k->lhs.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            k->rhs.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            const Clock::time_point t1 = Clock::now();
            xrt::run run = k->kernel(kOpcode, k->instr, k->instrWords, k->lhs, k->rhs, k->out, k->tmp, k->trace);
            const ert_cmd_state state = run.wait();
            const Clock::time_point t2 = Clock::now();
            if (state != ERT_CMD_STATE_COMPLETED) {
                VULKANIZER_ERROR("npu: %s ended in state %d", k->file->xclbin.filename().c_str(), int(state));
                return false;
            }
            k->out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            const auto sec = [](Clock::duration d) { return std::chrono::duration<double>(d).count(); };
            times_ = {sec(t1 - t0), sec(t2 - t1), sec(Clock::now() - t2)};
            return true;
        } catch (const std::exception& e) {
            VULKANIZER_ERROR("npu: %s: %s", f->xclbin.filename().c_str(), e.what());
            return false;
        }
    }

    bool gemm(const GemmArgs& g) override {
        const GemmShape& s = g.shape;
        const KernelFile* f = pick(s);
        if (!f) return false;
        std::lock_guard<std::mutex> lock(mutex_);
        try {
            Loaded& k = load(*f);
            auto* lhs = k.lhs.map<char*>();
            auto* rhs = k.rhs.map<char*>();
            auto* out = k.out.map<char*>();
            const size_t sa = dtypeSize(s.a), sb = dtypeSize(s.b), sc = dtypeSize(s.c), so = dtypeSize(f->out);
            // On failure C must stay untouched when the GPU redoes it reading C (beta != 0): with several
            // batch entries the results are staged (n x m each) and C is written after the last run.
            const bool stage = s.batch > 1 && g.beta != 0.0f;
            std::vector<char> staged(stage ? size_t(s.batch * s.n * s.m) * so : 0);
            using Clock = std::chrono::steady_clock;
            const auto since = [](Clock::time_point& t) {
                const Clock::time_point now = Clock::now();
                const double d = std::chrono::duration<double>(now - t).count();
                t = now;
                return d;
            };
            GemmTimes times;
            Clock::time_point t = Clock::now();
            for (int64_t i = 0; i < s.batch; ++i) {
                // Zero padding: the kernel's M x K x N is at least n x k x m.
                zeroPadding(lhs, dtypeSize(f->in), s.n, s.k, f->k, f->m);
                zeroPadding(rhs, dtypeSize(f->in), s.k, s.m, f->n, f->k);
                packTransposed(static_cast<const char*>(g.b) + i * g.strideB * int64_t(sb), s.b, g.ldb, s.transB, s.k, s.n,
                               lhs, f->in, f->k);
                packTransposed(static_cast<const char*>(g.a) + i * g.strideA * int64_t(sa), s.a, g.lda, s.transA, s.m, s.k,
                               rhs, f->in, f->n);
                k.lhs.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                k.rhs.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                times.pack += since(t);
                xrt::run run = k.kernel(kOpcode, k.instr, k.instrWords, k.lhs, k.rhs, k.out, k.tmp, k.trace);
                const ert_cmd_state state = run.wait();
                if (state != ERT_CMD_STATE_COMPLETED) {
                    VULKANIZER_ERROR("npu: %s ended in state %d", f->xclbin.filename().c_str(), int(state));
                    return false;
                }
                times.run += since(t);
                k.out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                if (stage) {
                    char* dst = staged.data() + size_t(i * s.n * s.m) * so;
                    for (int64_t r = 0; r < s.n; ++r)
                        std::memcpy(dst + size_t(r * s.m) * so, out + size_t(r * f->n) * so, size_t(s.m) * so);
                    times.epilogue += since(t);
                    continue;
                }
                epilogue(out, f->out, f->n, g.alpha, g.beta, static_cast<char*>(g.c) + i * g.strideC * int64_t(sc), s.c,
                         g.ldc, s.m, s.n);
                times.epilogue += since(t);
            }
            for (int64_t i = 0; stage && i < s.batch; ++i)
                epilogue(staged.data() + size_t(i * s.n * s.m) * so, f->out, s.m, g.alpha, g.beta,
                         static_cast<char*>(g.c) + i * g.strideC * int64_t(sc), s.c, g.ldc, s.m, s.n);
            times.epilogue += since(t);
            times_ = times;
            return true;
        } catch (const std::exception& e) {
            VULKANIZER_ERROR("npu: %s: %s", f->xclbin.filename().c_str(), e.what());
            return false;
        }
    }

private:
    // The smallest kernel holding lhs n x k, rhs k x m, of an input type the operands may be
    // rounded to: bf16 inputs as they are, f32 only with reduced precision allowed; f16 needs an
    // f16 kernel (bf16 has 3 mantissa bits fewer).
    const KernelFile* pick(const GemmShape& s) const {
        const auto accepts = [&](DType kernel, DType operand) {
            return kernel == operand || (kernel == DType::BF16 && operand == DType::F32 && s.reducedPrecision);
        };
        const KernelFile* best = nullptr;
        for (const KernelFile& f : files_) {
            if (!accepts(f.in, s.a) || !accepts(f.in, s.b)) continue;
            if (f.m < s.n || f.k < s.k || f.n < s.m) continue;
            if (!best || f.m * f.k * f.n < best->m * best->k * best->n) best = &f;
        }
        return best;
    }

    // The kernel on the NPU. At most VULKANIZER_NPU_MAX_LOADED (default 4) kernels hold a hardware
    // context at a time (the NPU has few; other processes use them too), the least recently used
    // goes first; when the NPU refuses a context, all others are released and it is tried once more.
    Loaded& load(const KernelFile& f) {
        const std::string key = f.xclbin.string();
        if (auto it = loaded_.find(key); it != loaded_.end()) {
            it->second.lastUse = ++uses_;
            return it->second;
        }
        static const size_t maxLoaded = [] {
            const char* v = std::getenv("VULKANIZER_NPU_MAX_LOADED");
            const int n = v ? std::atoi(v) : 0;
            return size_t(n > 0 ? n : 4);
        }();
        while (loaded_.size() >= maxLoaded) evictOldest();
        try {
            return loadNew(f, key);
        } catch (const std::exception& e) {
            if (loaded_.empty()) throw;
            VULKANIZER_INFO("npu: %s: %s; unloading the other kernels", f.xclbin.filename().c_str(), e.what());
            loaded_.clear();
            return loadNew(f, key);
        }
    }

    void evictOldest() {
        auto oldest = loaded_.begin();
        for (auto it = loaded_.begin(); it != loaded_.end(); ++it)
            if (it->second.lastUse < oldest->second.lastUse) oldest = it;
        VULKANIZER_DEBUG("npu: unloading %s", oldest->second.file->xclbin.filename().c_str());
        loaded_.erase(oldest);
    }

    Loaded& loadNew(const KernelFile& f, const std::string& key) {
        xrt::xclbin xclbin(f.xclbin.string());
        std::string kernelName;
        for (const xrt::xclbin::kernel& k : xclbin.get_kernels())
            if (k.get_name().rfind("MLIR_AIE", 0) == 0) kernelName = k.get_name();
        if (kernelName.empty()) throw std::runtime_error("no MLIR_AIE kernel in the xclbin");
        device_.register_xclbin(xclbin);
        xrt::hw_context context(device_, xclbin.get_uuid());
        xrt::kernel kernel(context, kernelName);
        const std::vector<uint32_t> words = readWords(f.insts);
        auto pit = pages_.find(key);
        if (pit == pages_.end())
            pit = pages_
                      .emplace(key, Pages{HostPages(size_t(f.m * f.k) * dtypeSize(f.in)),
                                          HostPages(size_t(f.k * f.n) * dtypeSize(f.in)),
                                          HostPages(size_t(f.m * f.n) * dtypeSize(f.out))})
                      .first;
        Pages& pages = pit->second;
        const auto user = [&](HostPages& p, int arg) {
            return xrt::bo(device_, p.data(), p.size(), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(arg));
        };
        Loaded l{&pages,
                 ++uses_,
                 context,
                 kernel,
                 xrt::bo(device_, words.size() * 4, XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1)),
                 user(pages[0], 3),
                 user(pages[1], 4),
                 user(pages[2], 5),
                 xrt::bo(device_, 1, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6)),
                 xrt::bo(device_, 4, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(7)),
                 uint32_t(words.size()),
                 &f};
        l.instr.write(words.data());
        l.instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        VULKANIZER_INFO("npu: loaded %s (%lld x %lld x %lld)", f.xclbin.filename().c_str(), (long long)f.m,
                        (long long)f.k, (long long)f.n);
        return loaded_.emplace(key, std::move(l)).first->second;
    }

    xrt::device device_;
    std::vector<KernelFile> files_;
    std::mutex mutex_;
    std::map<std::string, Pages> pages_;  // declared before loaded_: outlives the buffers over it
    std::map<std::string, Loaded> loaded_;
    uint64_t uses_ = 0;
    GemmTimes times_;
};

}  // namespace

std::unique_ptr<Backend> createXdnaBackend(const DeviceInfo& info) {
    const fs::path dir = kernelDir();
    std::vector<KernelFile> files = scan(dir);
    if (files.empty()) {
        VULKANIZER_WARN("npu: %s: no GEMM kernels in %s (plan.md 7., N1)", info.node.c_str(), dir.c_str());
        return nullptr;
    }
    try {
        // XRT opens by PCI address (the accel node's parent device), so the right NPU is used when there are several.
        std::error_code ec;
        const fs::path pci = fs::canonical(fs::path("/sys/class/accel") / info.accel / "device", ec);
        xrt::device device = ec ? xrt::device(0u) : xrt::device(pci.filename().string());
        VULKANIZER_INFO("npu: xdna backend, %zu kernels from %s", files.size(), dir.c_str());
        return std::make_unique<XdnaBackend>(std::move(device), std::move(files));
    } catch (const std::exception& e) {
        VULKANIZER_WARN("npu: cannot open %s through XRT: %s", info.node.c_str(), e.what());
        return nullptr;
    }
}

}  // namespace vulkanizer::npu
