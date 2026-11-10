// JIT linking (cuLink*): the inputs' PTX joined into one module, which is what the "cubin" of
// cuLinkComplete is here: Vulkanizer loads PTX, not machine code. One input stays as it is; with
// more, the first .version/.target/.address_size header is kept, and the others' header lines and
// the .extern declarations of functions another input defines are dropped. Fat binaries give their
// PTX; libraries (archives such as libcudadevrt.a) are skipped, as the device runtime's functions
// (cooperative groups) are built into the compiler; cubins and object code cannot be used.

#include "driver/driver.h"

#include "common/log.h"
#include "ptx/fatbin.h"

#include <cstring>
#include <fstream>
#include <regex>
#include <set>
#include <sstream>

using namespace vulkanizer;
using namespace vulkanizer::driver;

struct CUlinkState_st {
    static constexpr uint32_t kMagic = 0x4b4e4c53;  // 'SLNK'
    uint32_t magic = kMagic;
    std::vector<std::string> inputs;  // PTX texts
    std::string image;                // the linked module (NUL-terminated PTX)
    char* infoLog = nullptr;          // the caller's log buffers (CU_JIT_*_LOG_BUFFER), cleared
    char* errorLog = nullptr;
    size_t infoSize = 0, errorSize = 0;
};

namespace {

void takeLogs(CUlinkState state, unsigned numOptions, CUjit_option* options, void** values) {
    for (unsigned i = 0; i < numOptions && options && values; ++i) {
        const auto size = size_t(reinterpret_cast<uintptr_t>(values[i]));
        switch (options[i]) {
        case CU_JIT_INFO_LOG_BUFFER: state->infoLog = static_cast<char*>(values[i]); break;
        case CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES: state->infoSize = size; break;
        case CU_JIT_ERROR_LOG_BUFFER: state->errorLog = static_cast<char*>(values[i]); break;
        case CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES: state->errorSize = size; break;
        default: break;
        }
    }
}

void logError(CUlinkState state, const std::string& message) {
    VULKANIZER_ERROR("cuLink: %s", message.c_str());
    if (state->errorLog && state->errorSize) {
        std::snprintf(state->errorLog, state->errorSize, "%s", message.c_str());
    }
}

// The functions a PTX text defines (.func or .entry with a body).
std::set<std::string> definedFunctions(const std::string& ptx) {
    static const std::regex def(R"(\.(?:func|entry)\s+(?:\([^)]*\)\s*)?([A-Za-z_$%][\w$%]*)\s*\([^;{]*\)\s*(?:\.\w+[^{;]*)?\{)");
    std::set<std::string> names;
    for (std::sregex_iterator it(ptx.begin(), ptx.end(), def), end; it != end; ++it) names.insert((*it)[1]);
    return names;
}

std::string join(const std::vector<std::string>& inputs) {
    if (inputs.size() == 1) return inputs[0];
    std::set<std::string> defined;
    for (const std::string& p : inputs)
        for (const std::string& n : definedFunctions(p)) defined.insert(n);
    static const std::regex header(R"(^\s*\.(version|target|address_size)\b[^\n]*\n?)", std::regex::multiline);
    static const std::regex externFunc(R"(\.extern\s+\.func\s+(?:\([^)]*\)\s*)?([A-Za-z_$%][\w$%]*)[^;]*;)");
    std::string out;
    for (size_t i = 0; i < inputs.size(); ++i) {
        std::string body = inputs[i];
        if (i > 0) body = std::regex_replace(body, header, "");
        std::string kept;
        auto last = body.cbegin();
        for (std::sregex_iterator it(body.begin(), body.end(), externFunc), end; it != end; ++it) {
            kept.append(last, (*it)[0].first);
            if (!defined.count((*it)[1])) kept.append((*it)[0].first, (*it)[0].second);  // defined nowhere: keep
            last = (*it)[0].second;
        }
        kept.append(last, body.cend());
        out += kept;
        out += '\n';
    }
    return out;
}

}  // namespace

extern "C" {

CUresult CUDAAPI cuLinkCreate_v2(unsigned int numOptions, CUjit_option* options, void** optionValues,
                                 CUlinkState* stateOut) {
    VULKANIZER_TRACE_API();
    if (!stateOut) return CUDA_ERROR_INVALID_VALUE;
    auto* state = new CUlinkState_st();
    takeLogs(state, numOptions, options, optionValues);
    if (state->infoLog && state->infoSize) state->infoLog[0] = 0;
    if (state->errorLog && state->errorSize) state->errorLog[0] = 0;
    Driver::get().registerHandle(state);
    *stateOut = state;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuLinkAddData_v2(CUlinkState state, CUjitInputType type, void* data, size_t size, const char* name,
                                  unsigned int, CUjit_option*, void**) {
    VULKANIZER_TRACE_API();
    if (!validHandle(state)) return CUDA_ERROR_INVALID_HANDLE;
    if (!data) return CUDA_ERROR_INVALID_VALUE;
    switch (type) {
    case CU_JIT_INPUT_PTX: {
        const char* text = static_cast<const char*>(data);
        state->inputs.emplace_back(text, size ? strnlen(text, size) : std::strlen(text));
        return CUDA_SUCCESS;
    }
    case CU_JIT_INPUT_FATBINARY: {
        int major = 0, minor = 0;
        CUdevice dev = 0;
        cuCtxGetDevice(&dev);
        cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
        cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
        std::string ptx, error;
        uint32_t sm = 0;
        if (!fatbin::extractPtx(data, uint32_t(major * 10 + minor), ptx, sm, error)) {
            logError(state, std::string(name ? name : "fatbin") + ": " + error);
            return CUDA_ERROR_NO_BINARY_FOR_GPU;
        }
        state->inputs.push_back(std::move(ptx));
        return CUDA_SUCCESS;
    }
    case CU_JIT_INPUT_LIBRARY:
        VULKANIZER_DEBUG("cuLink: skipping library %s", name ? name : "");
        return CUDA_SUCCESS;
    default:
        logError(state, std::string(name ? name : "input") + ": only PTX (or a fat binary with PTX) can be linked");
        return CUDA_ERROR_NO_BINARY_FOR_GPU;
    }
}

CUresult CUDAAPI cuLinkAddFile_v2(CUlinkState state, CUjitInputType type, const char* path, unsigned int numOptions,
                                  CUjit_option* options, void** optionValues) {
    if (!validHandle(state)) return CUDA_ERROR_INVALID_HANDLE;
    if (!path) return CUDA_ERROR_INVALID_VALUE;
    std::ifstream in(path, std::ios::binary);
    if (!in) return CUDA_ERROR_FILE_NOT_FOUND;
    std::ostringstream ss;
    ss << in.rdbuf();
    std::string data = ss.str();
    data.push_back('\0');
    return cuLinkAddData_v2(state, type, data.data(), data.size() - 1, path, numOptions, options, optionValues);
}

CUresult CUDAAPI cuLinkComplete(CUlinkState state, void** cubinOut, size_t* sizeOut) {
    VULKANIZER_TRACE_API();
    if (!validHandle(state)) return CUDA_ERROR_INVALID_HANDLE;
    if (!cubinOut || !sizeOut) return CUDA_ERROR_INVALID_VALUE;
    if (state->inputs.empty()) {
        logError(state, "nothing to link");
        return CUDA_ERROR_INVALID_IMAGE;
    }
    state->image = join(state->inputs);
    *cubinOut = state->image.data();
    *sizeOut = state->image.size() + 1;  // with the terminator
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuLinkDestroy(CUlinkState state) {
    if (!validHandle(state)) return CUDA_ERROR_INVALID_HANDLE;
    Driver::get().unregisterHandle(state);
    delete state;
    return CUDA_SUCCESS;
}

// The unversioned names (cuda.h maps them to _v2), for binaries that link them.
#undef cuLinkCreate
#undef cuLinkAddData
#undef cuLinkAddFile
CUresult CUDAAPI cuLinkCreate(unsigned int numOptions, CUjit_option* options, void** optionValues, CUlinkState* stateOut);
CUresult CUDAAPI cuLinkCreate(unsigned int numOptions, CUjit_option* options, void** optionValues, CUlinkState* stateOut) {
    return cuLinkCreate_v2(numOptions, options, optionValues, stateOut);
}
CUresult CUDAAPI cuLinkAddData(CUlinkState state, CUjitInputType type, void* data, size_t size, const char* name,
                               unsigned int numOptions, CUjit_option* options, void** optionValues);
CUresult CUDAAPI cuLinkAddData(CUlinkState state, CUjitInputType type, void* data, size_t size, const char* name,
                               unsigned int numOptions, CUjit_option* options, void** optionValues) {
    return cuLinkAddData_v2(state, type, data, size, name, numOptions, options, optionValues);
}
CUresult CUDAAPI cuLinkAddFile(CUlinkState state, CUjitInputType type, const char* path, unsigned int numOptions,
                               CUjit_option* options, void** optionValues);
CUresult CUDAAPI cuLinkAddFile(CUlinkState state, CUjitInputType type, const char* path, unsigned int numOptions,
                               CUjit_option* options, void** optionValues) {
    return cuLinkAddFile_v2(state, type, path, numOptions, options, optionValues);
}

}  // extern "C"
