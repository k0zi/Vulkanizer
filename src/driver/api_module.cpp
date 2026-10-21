// Module loading (PTX), kernel lookup and launch.

#include "driver/driver.h"

#include "common/log.h"
#include "compiler/spirv_tools.h"
#include "common/sha256.h"
#include "driver/kernel_cache.h"
#include "driver/printf.h"
#include "vulkanizer/vulkanizer_ext.h"
#include "ptx/fatbin.h"

#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <fstream>
#include <sstream>

using namespace vulkanizer;
using namespace vulkanizer::driver;

namespace {

enum class ImageKind { Ptx, Cubin, FatBinary, FatBinaryWrapper, Empty };

bool hasMagic(const uint8_t* p, const uint8_t (&magic)[4]) {
    // Bytewise so we never read past the NUL terminator of a short PTX string.
    for (int i = 0; i < 4; ++i)
        if (p[i] != magic[i]) return false;
    return true;
}

ImageKind classifyImage(const void* image) {
    static constexpr uint8_t kElf[4] = {0x7f, 'E', 'L', 'F'};
    static constexpr uint8_t kFatBin[4] = {0x50, 0xed, 0x55, 0xba};         // 0xBA55ED50 little-endian
    static constexpr uint8_t kFatBinWrapper[4] = {0xb1, 0x43, 0x62, 0x46};  // 0x466243B1 little-endian
    auto* p = static_cast<const uint8_t*>(image);
    if (p[0] == 0) return ImageKind::Empty;
    if (hasMagic(p, kElf)) return ImageKind::Cubin;
    if (hasMagic(p, kFatBin)) return ImageKind::FatBinary;
    if (hasMagic(p, kFatBinWrapper)) return ImageKind::FatBinaryWrapper;
    return ImageKind::Ptx;
}

void writeLog(char* buffer, size_t size, const std::string& text) {
    if (!buffer || size == 0) return;
    std::strncpy(buffer, text.c_str(), size - 1);
    buffer[size - 1] = '\0';
}

compiler::Target targetFor(const DeviceSlot& slot);

constexpr uint32_t kPrintfBufferBytes = 1 << 20;

// Prints the records the kernel left in the printf buffer (see compiler/ptx_to_spirv.cpp).
void flushDevicePrintf(DeviceSlot& slot, const CUmod_st& mod) {
    uint32_t header[2];
    if (slot.vk->read(slot.printfBuffer, 0, header, sizeof header) != VK_SUCCESS) return;
    const uint32_t recordBytes = 8 + compiler::KernelBinary::kPrintfArgBytes;
    const uint32_t capacity = header[1];
    const uint32_t used = std::min(header[0], capacity / recordBytes * recordBytes);
    if (!used) return;
    std::vector<uint8_t> records(used);
    if (slot.vk->read(slot.printfBuffer, 16, records.data(), used) != VK_SUCCESS) return;
    auto readString = [&](uint64_t addr) -> std::string {
        if (addr >= mod.globals && addr < mod.globals + mod.globalsImage.size()) {
            const char* p = reinterpret_cast<const char*>(mod.globalsImage.data()) + (addr - mod.globals);
            return std::string(p, strnlen(p, mod.globalsImage.size() - (addr - mod.globals)));
        }
        AllocRef ref;
        if (!findAllocation(slot, addr, 1, ref)) return "<unknown string>";
        char tmp[256] = {};
        const size_t n = std::min<size_t>(sizeof tmp - 1, ref.buffer->size - ref.offset);
        slot.vk->read(*ref.buffer, ref.offset, tmp, n);
        return tmp;
    };
    std::string text;
    for (uint32_t at = 0; at + recordBytes <= used; at += recordBytes) {
        uint64_t format;
        std::memcpy(&format, records.data() + at, 8);
        text += formatDevicePrintf(readString(format), records.data() + at + 8, compiler::KernelBinary::kPrintfArgBytes,
                                   readString);
    }
    std::fwrite(text.data(), 1, text.size(), stdout);
    std::fflush(stdout);
    if (header[0] > capacity) VULKANIZER_WARN("device printf buffer overflowed; some output was dropped");
}

bool envFlag(const char* name) {
    const char* v = std::getenv(name);
    return v && *v && std::strcmp(v, "0") != 0;
}

// Translates one kernel for the context's device. VULKANIZER_DUMP=<dir> writes the SPIR-V out,
// VULKANIZER_SPIRV_VAL=1 runs spirv-val on it and rejects invalid modules.
void compileFunction(CUfunc_st& f, const DeviceSlot& slot) {
    f.compileAttempted = true;
    const compiler::Target target = targetFor(slot);
    CUmod_st& mod = *f.module;
    if (mod.sourceHash.empty()) {
        Sha256 h;
        h.update(mod.source);
        mod.sourceHash = h.hexDigest();
    }
    const std::string key = kernelCacheKey(mod.sourceHash, f.entry->name, target);
    const bool cached = loadCachedKernel(key, f.binary);
    VULKANIZER_DEBUG("kernel '%s': %s", f.entry->name.c_str(), cached ? "loaded from the kernel cache" : "translating");
    f.compiled = cached || compiler::compileKernel(mod.ptx, *f.entry, target, f.binary, f.compileError);
    if (!f.compiled) return;
    if (!cached) storeCachedKernel(key, f.binary);
    if (const char* dir = std::getenv("VULKANIZER_DUMP"); dir && *dir) {
        if (!compiler::writeSpirv(dir, f.entry->name, f.binary.spirv))
            VULKANIZER_WARN("VULKANIZER_DUMP: cannot write %s/%s.spv", dir, f.entry->name.c_str());
        // the module's PTX too, once per module (for vulkanizer-ptxc)
        const std::string ptxPath = std::string(dir) + "/" + mod.sourceHash.substr(0, 16) + ".ptx";
        if (FILE* fp = std::fopen(ptxPath.c_str(), "wx")) {
            std::fwrite(mod.source.data(), 1, mod.source.size(), fp);
            std::fclose(fp);
        }
    }
    if (envFlag("VULKANIZER_SPIRV_VAL")) {
        std::string log;
        if (!compiler::validateWithSpirvVal(f.binary.spirv, log)) {
            f.compiled = false;
            f.compileError = "generated SPIR-V is invalid: " + log;
        }
    }
}

CUresult loadModule(CUmodule* module, const void* image, std::string* errorLog) {
    if (!module || !image) return CUDA_ERROR_INVALID_VALUE;
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;

    auto mod = std::make_unique<CUmod_st>();
    mod->ctx = s.ctx;
    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();
    auto ms = [](Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    switch (classifyImage(image)) {
        case ImageKind::Empty: return CUDA_ERROR_INVALID_IMAGE;
        case ImageKind::Cubin:
            VULKANIZER_ERROR("cuModuleLoad: image is a cubin (native NVIDIA SASS) - only PTX can be translated");
            return CUDA_ERROR_NO_BINARY_FOR_GPU;
        case ImageKind::FatBinary:
        case ImageKind::FatBinaryWrapper: {
            int major, minor;
            computeCapability(major, minor);
            uint32_t sm = 0;
            std::string error;
            if (!fatbin::extractPtx(image, uint32_t(major * 10 + minor), mod->source, sm, error)) {
                VULKANIZER_ERROR("cuModuleLoad: %s", error.c_str());
                if (errorLog) *errorLog = error;
                return error.find("no PTX") != std::string::npos ? CUDA_ERROR_NO_BINARY_FOR_GPU
                                                                  : CUDA_ERROR_INVALID_IMAGE;
            }
            VULKANIZER_DEBUG("fatbin: using PTX for sm_%u (%zu bytes)", sm, mod->source.size());
            break;
        }
        case ImageKind::Ptx: mod->source = static_cast<const char*>(image); break;
    }

    const auto t1 = Clock::now();
    ptx::ParseError err;
    if (!ptx::parseModule(mod->source, mod->ptx, err)) {
        std::string msg = "PTX parse error at line " + std::to_string(err.line) + ": " + err.message;
        VULKANIZER_ERROR("%s", msg.c_str());
        if (errorLog) *errorLog = msg;
        return CUDA_ERROR_INVALID_PTX;
    }
    if (mod->ptx.globalsSize) {
        // One buffer for every .global/.const variable, initialized from the PTX.
        vk::Buffer buf;
        if (VkResult vr = s.dev->allocateBuffer(mod->ptx.globalsSize, buf)) return toCuResult(vr);
        std::vector<uint8_t> image(mod->ptx.globalsSize, 0);
        for (const ptx::GlobalVar& g : mod->ptx.globals)
            if (!g.init.empty()) std::memcpy(image.data() + g.offset, g.init.data(), g.init.size());
        if (VkResult vr = s.dev->write(buf, 0, image.data(), image.size())) {
            s.dev->freeBuffer(buf);
            return toCuResult(vr);
        }
        s.slot->allocations[buf.address] = buf;
        s.slot->allocatedBytes += buf.size;
        mod->globals = buf.address;
        mod->globalsImage = std::move(image);
    }
    const auto t2 = Clock::now();
    for (const ptx::Entry& e : mod->ptx.entries) {
        auto fn = std::make_unique<CUfunc_st>();
        fn->module = mod.get();
        fn->entry = &e;
        Driver::get().registerHandle(fn.get());
        mod->functions.push_back(std::move(fn));
    }
    // Small modules are translated eagerly so problems surface at load time (and in the JIT
    // log). Large ones (libraries ship hundreds of kernels, most never launched) are translated
    // on first launch; VULKANIZER_EAGER=1 forces eager translation. A kernel that cannot be
    // translated does not fail the module; launching it does.
    constexpr size_t kEagerLimit = 16;
    const bool eager = mod->functions.size() <= kEagerLimit || envFlag("VULKANIZER_EAGER");
    for (auto& fn : mod->functions) {
        if (!eager) break;
        compileFunction(*fn, *s.slot);
        if (fn->compiled) continue;
        VULKANIZER_WARN("kernel '%s' cannot be translated: %s", fn->entry->name.c_str(), fn->compileError.c_str());
        if (errorLog) *errorLog += fn->entry->name + ": " + fn->compileError + "\n";
    }
    const auto t3 = Clock::now();
    VULKANIZER_DEBUG("loaded PTX module (version %s, %zu kernel(s), %zu bytes): extract %.1f ms, parse %.1f ms, "
                     "globals %.1f ms, translate %.1f ms",
                     mod->ptx.version.c_str(), mod->ptx.entries.size(), mod->source.size(), ms(t0, t1), ms(t1, t2),
                     0.0, ms(t2, t3));
    Driver::get().registerHandle(mod.get());
    *module = mod.release();
    return CUDA_SUCCESS;
}

compiler::Target targetFor(const DeviceSlot& slot) {
    compiler::Target t;
    t.subgroupSize = slot.info->subgroupSize;
    t.subgroupBallot = slot.info->subgroupOps & VK_SUBGROUP_FEATURE_BALLOT_BIT;
    t.subgroupShuffle = slot.info->subgroupOps & VK_SUBGROUP_FEATURE_SHUFFLE_BIT;
    if (envFlag("VULKANIZER_EMULATE_WARPS")) t.subgroupShuffle = false;  // debugging: shared-memory warps
    t.maxPushConstantsSize = slot.info->props.limits.maxPushConstantsSize;
    t.maxSharedMemorySize = slot.info->props.limits.maxComputeSharedMemorySize;
    const vk::Features& f = slot.vk->features();
    t.float16 = f.shaderFloat16;
    t.float64 = f.shaderFloat64;
    t.int8 = f.shaderInt8;
    t.int16 = f.shaderInt16;
    t.storage8 = f.storageBuffer8Bit && f.shaderInt8;
    t.storage16 = f.storageBuffer16Bit && f.shaderInt16;
    t.int64Atomics = f.bufferInt64Atomics;
    t.integerDotProduct = f.integerDotProduct;
    t.fusedFma = f.fusedFma || envFlag("VULKANIZER_FAST_FMA");
    t.fmaInF64 = envFlag("VULKANIZER_FMA_F64");
    t.fusedFma64 = f.fusedFma64 || envFlag("VULKANIZER_FAST_FMA");
    return t;
}

CUresult packParams(const ptx::Entry& e, void** kernelParams, void** extra, std::vector<uint8_t>& out) {
    out.assign((e.paramBytes + 3u) & ~3u, 0);
    if (kernelParams) {
        for (size_t i = 0; i < e.params.size(); ++i) {
            if (!kernelParams[i]) return CUDA_ERROR_INVALID_VALUE;
            std::memcpy(out.data() + e.params[i].offset, kernelParams[i], e.params[i].size);
        }
        return CUDA_SUCCESS;
    }
    if (extra) {
        const void* buffer = nullptr;
        const size_t* size = nullptr;
        for (size_t i = 0; extra[i] != CU_LAUNCH_PARAM_END; i += 2) {
            if (extra[i] == CU_LAUNCH_PARAM_BUFFER_POINTER)
                buffer = extra[i + 1];
            else if (extra[i] == CU_LAUNCH_PARAM_BUFFER_SIZE)
                size = static_cast<const size_t*>(extra[i + 1]);
            else
                return CUDA_ERROR_INVALID_VALUE;
        }
        if (!buffer || !size || *size < e.paramBytes) return CUDA_ERROR_INVALID_VALUE;
        std::memcpy(out.data(), buffer, e.paramBytes);
        return CUDA_SUCCESS;
    }
    return e.params.empty() ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

}  // namespace

extern "C" {

CUresult CUDAAPI cuModuleLoadData(CUmodule* module, const void* image) {
    VULKANIZER_TRACE_API();
    return loadModule(module, image, nullptr);
}

CUresult CUDAAPI cuModuleLoadDataEx(CUmodule* module, const void* image, unsigned int numOptions,
                                    CUjit_option* options, void** optionValues) {
    VULKANIZER_TRACE_API();
    char* errorBuffer = nullptr;
    size_t errorBufferSize = 0;
    for (unsigned int i = 0; i < numOptions && options && optionValues; ++i) {
        if (options[i] == CU_JIT_ERROR_LOG_BUFFER)
            errorBuffer = static_cast<char*>(optionValues[i]);
        else if (options[i] == CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES)
            errorBufferSize = reinterpret_cast<uintptr_t>(optionValues[i]);
    }
    std::string errorLog;
    CUresult r = loadModule(module, image, &errorLog);
    writeLog(errorBuffer, errorBufferSize, errorLog);
    return r;
}

CUresult CUDAAPI cuModuleLoadFatBinary(CUmodule* module, const void* fatCubin) {
    VULKANIZER_TRACE_API();
    return loadModule(module, fatCubin, nullptr);
}

CUresult CUDAAPI cuModuleLoad(CUmodule* module, const char* fname) {
    VULKANIZER_TRACE_API();
    if (!module || !fname) return CUDA_ERROR_INVALID_VALUE;
    std::ifstream in(fname, std::ios::binary);
    if (!in) return CUDA_ERROR_FILE_NOT_FOUND;
    std::ostringstream ss;
    ss << in.rdbuf();
    const std::string data = ss.str();
    return loadModule(module, data.c_str(), nullptr);
}

CUresult CUDAAPI cuModuleUnload(CUmodule hmod) {
    VULKANIZER_TRACE_API();
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!validHandle(hmod)) return CUDA_ERROR_INVALID_HANDLE;
    DeviceSlot* owner = Driver::get().slot(hmod->ctx->device);
    for (auto& fn : hmod->functions) {
        for (auto& [dims, pipeline] : fn->pipelines) owner->vk->destroyComputePipeline(pipeline);
        Driver::get().unregisterHandle(fn.get());
    }
    if (hmod->globals) {
        auto it = owner->allocations.find(hmod->globals);
        if (it != owner->allocations.end()) {
            owner->allocatedBytes -= it->second.size;
            owner->vk->freeBuffer(it->second);
            owner->allocations.erase(it);
        }
    }
    Driver::get().unregisterHandle(hmod);
    delete hmod;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name) {
    VULKANIZER_TRACE_API();
    if (!hfunc || !name) return CUDA_ERROR_INVALID_VALUE;
    if (!Driver::get().initialized()) return CUDA_ERROR_NOT_INITIALIZED;
    if (!validHandle(hmod)) return CUDA_ERROR_INVALID_HANDLE;
    for (auto& fn : hmod->functions) {
        if (fn->entry->name == name) {
            *hfunc = fn.get();
            return CUDA_SUCCESS;
        }
    }
    return CUDA_ERROR_NOT_FOUND;
}

CUresult CUDAAPI vulkanizerCreateSpirvFunction(CUfunction* function, const uint32_t* spirv, size_t wordCount,
                                          const char* entryPoint, unsigned int paramBytes) {
    VULKANIZER_TRACE_API();
    if (!function || !spirv || !wordCount || !entryPoint) return CUDA_ERROR_INVALID_VALUE;
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (paramBytes > s.slot->info->props.limits.maxPushConstantsSize) return CUDA_ERROR_INVALID_VALUE;

    // A module with one synthetic entry whose single parameter is the whole push block.
    auto mod = std::make_unique<CUmod_st>();
    mod->ctx = s.ctx;
    ptx::Entry e;
    e.name = entryPoint;
    if (paramBytes) e.params.push_back({"params", paramBytes, 4, 0});
    e.paramBytes = paramBytes;
    mod->ptx.entries.push_back(std::move(e));

    auto fn = std::make_unique<CUfunc_st>();
    fn->module = mod.get();
    fn->entry = &mod->ptx.entries.back();
    fn->compileAttempted = fn->compiled = true;
    fn->binary.spirv.assign(spirv, spirv + wordCount);
    fn->binary.entryPoint = entryPoint;
    fn->binary.pushConstantSize = (paramBytes + 3u) & ~3u;
    *function = fn.get();
    Driver::get().registerHandle(fn.get());
    mod->functions.push_back(std::move(fn));
    Driver::get().registerHandle(mod.get());
    mod.release();  // owned by the context (never unloaded explicitly)
    return CUDA_SUCCESS;
}

CUresult CUDAAPI vulkanizerGetFeature(VulkanizerFeature feature, int* value) {
    VULKANIZER_TRACE_API();
    if (!value) return CUDA_ERROR_INVALID_VALUE;
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    switch (feature) {
    case VULKANIZER_FEATURE_FLOAT16_ARITHMETIC: *value = s.slot->vk->features().shaderFloat16; return CUDA_SUCCESS;
    }
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult CUDAAPI cuModuleGetGlobal_v2(CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, const char* name) {
    VULKANIZER_TRACE_API();
    if (!name) return CUDA_ERROR_INVALID_VALUE;
    if (!Driver::get().initialized()) return CUDA_ERROR_NOT_INITIALIZED;
    if (!validHandle(hmod)) return CUDA_ERROR_INVALID_HANDLE;
    const ptx::GlobalVar* g = hmod->ptx.findGlobal(name);
    if (!g || g->isExtern || (g->space != ".global" && g->space != ".const")) return CUDA_ERROR_NOT_FOUND;
    if (dptr) *dptr = hmod->globals + g->offset;
    if (bytes) *bytes = g->size;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuFuncGetParamInfo(CUfunction func, size_t paramIndex, size_t* paramOffset, size_t* paramSize) {
    VULKANIZER_TRACE_API();
    if (!validHandle(func)) return CUDA_ERROR_INVALID_HANDLE;
    const ptx::Entry& e = *func->entry;
    if (paramIndex >= e.params.size()) return CUDA_ERROR_INVALID_VALUE;
    if (paramOffset) *paramOffset = e.params[paramIndex].offset;
    if (paramSize) *paramSize = e.params[paramIndex].size;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuFuncGetAttribute(int* pi, CUfunction_attribute attrib, CUfunction hfunc) {
    VULKANIZER_TRACE_API();
    if (!pi) return CUDA_ERROR_INVALID_VALUE;
    if (!validHandle(hfunc)) return CUDA_ERROR_INVALID_HANDLE;
    const ptx::Entry& e = *hfunc->entry;
    const ptx::Module& m = hfunc->module->ptx;
    switch (attrib) {
        case CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK:
            *pi = e.maxntid[0] ? static_cast<int>(e.maxntid[0] * e.maxntid[1] * e.maxntid[2]) : 1024;
            break;
        // Both are architectures (major * 10 + minor), not the PTX ISA version: the PTX's virtual
        // architecture (.target sm_61 -> 61), and the device it was compiled for. Libraries such
        // as CUB pick their kernels by these.
        case CU_FUNC_ATTRIBUTE_PTX_VERSION: {
            int arch = 0;
            for (const std::string& t : m.targets)
                if (t.starts_with("sm_") || t.starts_with("compute_")) arch = std::atoi(t.c_str() + t.find('_') + 1);
            *pi = arch ? arch : 52;
            break;
        }
        case CU_FUNC_ATTRIBUTE_BINARY_VERSION: {
            int major, minor;
            computeCapability(major, minor);
            *pi = major * 10 + minor;
            break;
        }
        case CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES: *pi = hfunc->compiled ? int(hfunc->binary.staticSharedBytes) : 0; break;
        case CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES: *pi = hfunc->compiled ? int(hfunc->binary.localBytes) : 0; break;
        case CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: {
            const uint32_t limit = Driver::get().slot(hfunc->module->ctx->device)->info->props.limits.maxComputeSharedMemorySize;
            const uint32_t used = hfunc->compiled ? hfunc->binary.staticSharedBytes : 0;
            *pi = int(limit > used ? limit - used : 0);
            break;
        }
        case CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES:
        case CU_FUNC_ATTRIBUTE_NUM_REGS:
        case CU_FUNC_ATTRIBUTE_CACHE_MODE_CA:
        case CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT: *pi = 0; break;
        default: return CUDA_ERROR_INVALID_VALUE;
    }
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                                unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                                unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra) {
    VULKANIZER_TRACE_API();
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (!validHandle(f)) return CUDA_ERROR_INVALID_HANDLE;
    if (!validStream(hStream)) return CUDA_ERROR_INVALID_HANDLE;
    if (!gridDimX || !gridDimY || !gridDimZ || !blockDimX || !blockDimY || !blockDimZ) return CUDA_ERROR_INVALID_VALUE;

    const VkPhysicalDeviceLimits& lim = s.slot->info->props.limits;
    if (uint64_t(blockDimX) * blockDimY * blockDimZ > lim.maxComputeWorkGroupInvocations ||
        blockDimX > lim.maxComputeWorkGroupSize[0] || blockDimY > lim.maxComputeWorkGroupSize[1] ||
        blockDimZ > lim.maxComputeWorkGroupSize[2])
        return CUDA_ERROR_INVALID_VALUE;
    // x is split into several dispatches when needed (vk::Device::dispatch); y and z are 65535 in
    // CUDA, which every Vulkan device supports.
    if (gridDimX > 0x7FFFFFFFu || gridDimY > lim.maxComputeWorkGroupCount[1] ||
        gridDimZ > lim.maxComputeWorkGroupCount[2])
        return CUDA_ERROR_INVALID_VALUE;

    const ptx::Entry& e = *f->entry;
    std::vector<uint8_t> params;
    if (CUresult r = packParams(e, kernelParams, extra, params)) return r;

    if (!f->compileAttempted) compileFunction(*f, *s.slot);
    if (!f->compiled) {
        VULKANIZER_ERROR("cannot launch '%s': %s", e.name.c_str(), f->compileError.c_str());
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (uint64_t(f->binary.staticSharedBytes) + sharedMemBytes > lim.maxComputeSharedMemorySize) {
        VULKANIZER_ERROR("'%s': %u static + %u dynamic bytes of shared memory exceed the device limit (%u)", e.name.c_str(),
                    f->binary.staticSharedBytes, sharedMemBytes, lim.maxComputeSharedMemorySize);
        return CUDA_ERROR_INVALID_VALUE;
    }
    std::vector<uint8_t> push(f->binary.pushConstantSize, 0);
    if (f->binary.paramsInBuffer) {
        VkDeviceAddress addr;
        if (VkResult vr = s.dev->upload(params.data(), params.size(), addr)) return toCuResult(vr);
        std::memcpy(push.data(), &addr, sizeof addr);
    } else {
        std::memcpy(push.data(), params.data(), std::min(params.size(), push.size()));
    }
    if (f->binary.globalsOffset != compiler::KernelBinary::kNone)
        std::memcpy(push.data() + f->binary.globalsOffset, &f->module->globals, sizeof(CUdeviceptr));
    if (f->binary.gridOffset != compiler::KernelBinary::kNone) {
        const uint32_t grid[3] = {gridDimX, gridDimY, gridDimZ};
        std::memcpy(push.data() + f->binary.gridOffset, grid, sizeof grid);
    }
    const bool printf = f->binary.printfOffset != compiler::KernelBinary::kNone;
    if (printf) {
        if (!s.slot->printfBuffer.buffer) {
            if (VkResult vr = s.dev->allocateBuffer(kPrintfBufferBytes + 16, s.slot->printfBuffer)) return toCuResult(vr);
        }
        // The buffer is shared by all launches, so a printf kernel runs on its own.
        const uint32_t header[4] = {0, kPrintfBufferBytes, 0, 0};
        if (VkResult vr = s.dev->write(s.slot->printfBuffer, 0, header, sizeof header)) return toCuResult(vr);
        std::memcpy(push.data() + f->binary.printfOffset, &s.slot->printfBuffer.address, sizeof(VkDeviceAddress));
    }

    // Small blocks of a packable kernel share a workgroup (up to kPackThreads threads): fuller
    // waves, and more of them resident than the per-CU workgroup limit would allow. Only whole
    // warps are packed, so CUDA warps stay aligned to the hardware's.
    constexpr uint32_t kPackThreads = 256;
    const uint32_t threads = blockDimX * blockDimY * blockDimZ;
    uint32_t pack = 1;
    if (f->binary.packable && blockDimX % 32 == 0 && threads < kPackThreads && gridDimX > 1 && !envFlag("VULKANIZER_NO_PACK")) {
        pack = std::min<uint32_t>(kPackThreads / threads, gridDimX);
        pack = std::min(pack, lim.maxComputeWorkGroupSize[0] / blockDimX);
        pack = std::min(pack, lim.maxComputeWorkGroupInvocations / threads);
        pack = std::max(pack, 1u);
    }
    const std::array<uint32_t, 5> block{blockDimX, blockDimY, blockDimZ, (sharedMemBytes + 3) / 4, pack};
    auto it = f->pipelines.find(block);
    if (it == f->pipelines.end()) {
        vk::ComputePipeline pipeline;
        VkResult vr = s.dev->createComputePipeline(f->binary.spirv.data(), f->binary.spirv.size(),
                                                   f->binary.entryPoint.c_str(), f->binary.pushConstantSize,
                                                   block.data(), pipeline);
        if (vr != VK_SUCCESS) {
            VULKANIZER_ERROR("'%s': pipeline creation failed: %d", e.name.c_str(), vr);
            return toCuResult(vr);
        }
        if (pack > 1) VULKANIZER_DEBUG("'%s': %u blocks of %u threads per workgroup", e.name.c_str(), pack, threads);
        it = f->pipelines.emplace(block, pipeline).first;
    }
    const VkResult vr = s.dev->dispatch(it->second, (gridDimX + pack - 1) / pack, gridDimY, gridDimZ, push.data(),
                                        static_cast<uint32_t>(push.size()), f->entry->name.c_str());
    if (vr == VK_SUCCESS && printf) flushDevicePrintf(*s.slot, *f->module);  // read() waits for the kernel
    return toCuResult(vr);
}

CUresult CUDAAPI cuLaunchKernelEx(const CUlaunchConfig* config, CUfunction f, void** kernelParams, void** extra) {
    VULKANIZER_TRACE_API();
    if (!config || (config->numAttrs && !config->attrs)) return CUDA_ERROR_INVALID_VALUE;
    for (unsigned int i = 0; i < config->numAttrs; ++i) {
        const CUlaunchAttribute& a = config->attrs[i];
        if (a.id == CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION &&
            uint64_t(a.value.clusterDim.x) * a.value.clusterDim.y * a.value.clusterDim.z > 1)
            return CUDA_ERROR_NOT_SUPPORTED;  // thread block clusters (sm_90)
        if (a.id == CU_LAUNCH_ATTRIBUTE_COOPERATIVE && a.value.cooperative)
            VULKANIZER_WARN_ONCE("cuLaunchKernelEx: cooperative launches run without grid-wide synchronization");
        // Everything else is a scheduling hint, or ordering that the single queue gives anyway.
    }
    return cuLaunchKernel(f, config->gridDimX, config->gridDimY, config->gridDimZ, config->blockDimX, config->blockDimY,
                          config->blockDimZ, config->sharedMemBytes, config->hStream, kernelParams, extra);
}

CUresult CUDAAPI cuLaunchKernelEx_ptsz(const CUlaunchConfig* config, CUfunction f, void** kernelParams, void** extra) {
    return cuLaunchKernelEx(config, f, kernelParams, extra);
}

CUresult CUDAAPI cuLaunchKernel_ptsz(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                                     unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                                     unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra) {
    return cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream,
                          kernelParams, extra);
}

// The shared memory limit is the device's in any case (cuLaunchKernel checks it); the other
// attributes are hints.
CUresult CUDAAPI cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int value) {
    VULKANIZER_TRACE_API();
    if (!validHandle(hfunc)) return CUDA_ERROR_INVALID_HANDLE;
    if (attrib == CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES) {
        int limit = 0;
        if (CUresult r = cuFuncGetAttribute(&limit, attrib, hfunc)) return r;
        if (value < 0 || value > limit) return CUDA_ERROR_INVALID_VALUE;
    }
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config) {
    VULKANIZER_TRACE_API();
    return validHandle(hfunc) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_HANDLE;
}

// Resident blocks per multiprocessor by threads and shared memory, as the runtime computes it.
CUresult CUDAAPI cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, CUfunction func, int blockSize,
                                                                      size_t dynamicSMemSize, unsigned int flags) {
    VULKANIZER_TRACE_API();
    if (!numBlocks || blockSize <= 0) return CUDA_ERROR_INVALID_VALUE;
    if (!validHandle(func)) return CUDA_ERROR_INVALID_HANDLE;
    int threadsPerSm = 0, smemPerSm = 0, staticSmem = 0;
    const CUdevice dev = func->module->ctx->device;
    if (CUresult r = cuDeviceGetAttribute(&threadsPerSm, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, dev)) return r;
    if (CUresult r = cuDeviceGetAttribute(&smemPerSm, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, dev))
        return r;
    if (CUresult r = cuFuncGetAttribute(&staticSmem, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, func)) return r;
    constexpr int kBlocksPerSm = 32;
    int n = std::min(threadsPerSm / blockSize, kBlocksPerSm);
    if (const size_t smem = size_t(staticSmem) + dynamicSMemSize) n = std::min<int>(n, int(size_t(smemPerSm) / smem));
    *numBlocks = std::max(n, 0);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, CUfunction func, int blockSize,
                                                             size_t dynamicSMemSize) {
    return cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, func, blockSize, dynamicSMemSize, 0);
}

// Clusters of one block (the only size there is): one per multiprocessor at least.
CUresult CUDAAPI cuOccupancyMaxActiveClusters(int* numClusters, CUfunction func, const CUlaunchConfig* config) {
    VULKANIZER_TRACE_API();
    if (!numClusters || !config) return CUDA_ERROR_INVALID_VALUE;
    if (!validHandle(func)) return CUDA_ERROR_INVALID_HANDLE;
    for (unsigned int i = 0; i < config->numAttrs; ++i) {
        const CUlaunchAttribute& a = config->attrs[i];
        if (a.id == CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION &&
            uint64_t(a.value.clusterDim.x) * a.value.clusterDim.y * a.value.clusterDim.z > 1) {
            *numClusters = 0;
            return CUDA_SUCCESS;
        }
    }
    int perSm = 0, sms = 0;
    const unsigned threads = config->blockDimX * config->blockDimY * config->blockDimZ;
    if (CUresult r = cuOccupancyMaxActiveBlocksPerMultiprocessor(&perSm, func, int(threads), config->sharedMemBytes))
        return r;
    if (CUresult r = cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, func->module->ctx->device))
        return r;
    *numClusters = perSm * sms;
    return CUDA_SUCCESS;
}

// Tensor memory accelerator descriptors (sm_90+): never needed below compute capability 9.0.
CUresult CUDAAPI cuTensorMapEncodeTiled(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                                        const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                                        CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill) {
    VULKANIZER_TRACE_API();
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuTensorMapEncodeIm2col(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                                         const cuuint64_t*, const int*, const int*, cuuint32_t, cuuint32_t,
                                         const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                                         CUtensorMapL2promotion, CUtensorMapFloatOOBfill) {
    VULKANIZER_TRACE_API();
    return CUDA_ERROR_NOT_SUPPORTED;
}

}  // extern "C"
