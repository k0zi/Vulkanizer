#pragma once

// Vulkanizer extensions to the CUDA Driver API, exported by Vulkanizer's libcuda.so. They are not part
// of CUDA; Vulkanizer's own libraries (libcublas) use them to run hand-written Vulkan kernels.

#include <cuda.h>

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Creates a function from SPIR-V that follows Vulkanizer's kernel ABI: the parameter buffer is the
// push constant block (paramBytes, at most the device's push constant size), blockDim comes
// from specialization constants 0/1/2 (local_size_x_id = 0, ...). Launch it with
// cuLaunchKernel and CU_LAUNCH_PARAM_BUFFER_POINTER/SIZE. The function lives as long as the
// current context.
CUresult CUDAAPI vulkanizerCreateSpirvFunction(CUfunction* function, const uint32_t* spirv, size_t wordCount,
                                          const char* entryPoint, unsigned int paramBytes);

// Optional device capabilities Vulkanizer's libraries pick kernel variants by.
typedef enum VulkanizerFeature {
    VULKANIZER_FEATURE_FLOAT16_ARITHMETIC = 1,  // shaderFloat16: f16 math in shaders (not just storage)
    VULKANIZER_FEATURE_FLOAT64 = 2,             // shaderFloat64: f64 in shaders
} VulkanizerFeature;

// *value = 1 if the current context's device has the feature, else 0.
CUresult CUDAAPI vulkanizerGetFeature(VulkanizerFeature feature, int* value);

#ifdef __cplusplus
}
#endif
