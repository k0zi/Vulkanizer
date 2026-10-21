// Layout of the CUDA Runtime types as seen through the official CUDA headers.
#include <driver_types.h>

#include "abi_list.h"
#include "layout.h"

#include <cstddef>

Layout nvidiaLayout() {
    Layout l;
    l.propSize = sizeof(cudaDeviceProp);
    l.funcAttrSize = sizeof(cudaFuncAttributes);
    l.launchConfigSize = sizeof(cudaLaunchConfig_t);
#define PROP(f) l.propOffsets.push_back(offsetof(cudaDeviceProp, f));
#define FATTR(f) l.funcAttrOffsets.push_back(offsetof(cudaFuncAttributes, f));
#define ERR(e) l.errors.push_back(long(e));
#define LCFG(f) l.launchConfigOffsets.push_back(offsetof(cudaLaunchConfig_t, f));
    VULKANIZER_LAUNCH_CONFIG_FIELDS(LCFG)
    VULKANIZER_DEVICE_PROP_FIELDS(PROP)
    VULKANIZER_FUNC_ATTR_FIELDS(FATTR)
    VULKANIZER_ERROR_VALUES(ERR)
    return l;
}
