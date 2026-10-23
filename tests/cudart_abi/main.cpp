// Checks that Vulkanizer's clean-room CUDA Runtime types match the official headers
// (built only when a CUDA toolkit is present in .cuda/).
#include "abi_list.h"
#include "layout.h"

#include <cstdio>

Layout nvidiaLayout();
Layout vulkanizerLayout();

int main() {
    const Layout a = nvidiaLayout(), b = vulkanizerLayout();
    int bad = 0;
    auto check = [&](const char* what, size_t x, size_t y) {
        if (x != y) std::printf("MISMATCH %-40s nvidia %zu, vulkanizer %zu\n", what, x, y), ++bad;
    };
    check("sizeof(cudaDeviceProp)", a.propSize, b.propSize);
    check("sizeof(cudaFuncAttributes)", a.funcAttrSize, b.funcAttrSize);
    check("sizeof(cudaLaunchConfig_t)", a.launchConfigSize, b.launchConfigSize);
    size_t i = 0;
#define PROP(f) check("cudaDeviceProp::" #f, a.propOffsets[i], b.propOffsets[i]), ++i;
    VULKANIZER_DEVICE_PROP_FIELDS(PROP)
    i = 0;
#define FATTR(f) check("cudaFuncAttributes::" #f, a.funcAttrOffsets[i], b.funcAttrOffsets[i]), ++i;
    VULKANIZER_FUNC_ATTR_FIELDS(FATTR)
    i = 0;
#define LCFG(f) check("cudaLaunchConfig_t::" #f, a.launchConfigOffsets[i], b.launchConfigOffsets[i]), ++i;
    VULKANIZER_LAUNCH_CONFIG_FIELDS(LCFG)
    i = 0;
#define ERR(e) check(#e, size_t(a.errors[i]), size_t(b.errors[i])), ++i;
    VULKANIZER_ERROR_VALUES(ERR)
    i = 0;
#define TEX(e) check(#e, a.texture[i], b.texture[i]), ++i;
    VULKANIZER_TEXTURE_LAYOUT(TEX)
    VULKANIZER_GRAPH_LAYOUT(TEX)
    std::printf(bad ? "%d mismatch(es)\n" : "OK\n", bad);
    return bad ? 1 : 0;
}
