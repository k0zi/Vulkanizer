#include "compiler/spirv_tools.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sys/wait.h>
#include <unistd.h>

namespace vulkanizer::compiler {

bool writeSpirv(const std::string& dir, const std::string& name, const std::vector<uint32_t>& spirv) {
    std::ofstream f(dir + "/" + name + ".spv", std::ios::binary);
    f.write(reinterpret_cast<const char*>(spirv.data()), std::streamsize(spirv.size() * sizeof(uint32_t)));
    return bool(f);
}

bool validateWithSpirvVal(const std::vector<uint32_t>& spirv, std::string& log) {
    const char* tmp = std::getenv("TMPDIR");
    std::string path = std::string(tmp && *tmp ? tmp : "/tmp") + "/vulkanizer-XXXXXX.spv";
    const int fd = mkstemps(path.data(), 4);
    if (fd < 0) {
        log = "cannot create a temporary file for spirv-val";
        return false;
    }
    const size_t bytes = spirv.size() * sizeof(uint32_t);
    const bool written = write(fd, spirv.data(), bytes) == ssize_t(bytes);
    close(fd);
    if (!written) {
        unlink(path.c_str());
        log = "cannot write a temporary file for spirv-val";
        return false;
    }

    const std::string cmd = "spirv-val --target-env vulkan1.2 '" + path + "' 2>&1";
    log.clear();
    int status = -1;
    if (FILE* p = popen(cmd.c_str(), "r")) {
        char buf[512];
        while (size_t n = std::fread(buf, 1, sizeof buf, p)) log.append(buf, n);
        status = pclose(p);
    }
    unlink(path.c_str());
    if (status == -1 || (WIFEXITED(status) && WEXITSTATUS(status) == 127)) {
        log = "spirv-val not found (install spirv-tools)";
        return false;
    }
    while (!log.empty() && (log.back() == '\n' || log.back() == ' ')) log.pop_back();
    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

}  // namespace vulkanizer::compiler
