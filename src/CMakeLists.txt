# Logging, shared by the driver and the runtime library.
add_library(vulkanizer_common STATIC common/log.cpp common/sha256.cpp)
target_include_directories(vulkanizer_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

find_program(GLSLC glslc HINTS ${Vulkan_GLSLC_EXECUTABLE} REQUIRED)
set(vk_gen ${CMAKE_CURRENT_BINARY_DIR}/vk)
foreach(probe fma_probe fma_probe_f64)
  add_custom_command(
    OUTPUT ${vk_gen}/${probe}.comp.inc
    COMMAND ${CMAKE_COMMAND} -E make_directory ${vk_gen}
    COMMAND ${GLSLC} --target-env=vulkan1.2 -O -mfmt=num -o ${vk_gen}/${probe}.comp.inc
            ${CMAKE_CURRENT_SOURCE_DIR}/vk/shaders/${probe}.comp
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/vk/shaders/${probe}.comp
    COMMENT "glslc ${probe}.comp"
  )
endforeach()

# Internal building blocks, linked statically into libcuda.so (and into internal tests).
add_library(vulkanizer_core STATIC
  ${vk_gen}/fma_probe.comp.inc
  ${vk_gen}/fma_probe_f64.comp.inc
  vk/instance.cpp
  vk/device.cpp
  ptx/lexer.cpp
  ptx/parser.cpp
  ptx/instr.cpp
  ptx/fatbin.cpp
  compiler/cfg.cpp
  compiler/spirv_builder.cpp
  compiler/spirv_tools.cpp
  compiler/ptx_to_spirv.cpp
)
target_include_directories(vulkanizer_core PRIVATE ${vk_gen})
target_include_directories(vulkanizer_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(vulkanizer_core PUBLIC vulkanizer_common Vulkan::Vulkan ${CMAKE_DL_LIBS})

# The drop-in CUDA driver: libcuda.so -> libcuda.so.1 -> libcuda.so.1.0.0
add_library(vulkanizer_cuda SHARED
  driver/driver.cpp
  driver/api_init.cpp
  driver/api_context.cpp
  driver/api_memory.cpp
  driver/api_mempool.cpp
  driver/api_vmm.cpp
  driver/api_module.cpp
  driver/api_stream.cpp
  driver/printf.cpp
  driver/kernel_cache.cpp
)
target_link_libraries(vulkanizer_cuda PRIVATE vulkanizer_core ${CMAKE_DL_LIBS})
target_include_directories(vulkanizer_cuda PUBLIC ${PROJECT_SOURCE_DIR}/include)
set_target_properties(vulkanizer_cuda PROPERTIES
  OUTPUT_NAME cuda
  VERSION 1.0.0
  SOVERSION 1
  LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/driver/libcuda.map
)
target_link_options(vulkanizer_cuda PRIVATE
  -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/driver/libcuda.map
  -Wl,--no-undefined
)

# The CUDA Runtime, implemented on the driver above: libcudart.so.12 and, from the same source with
# the CUDA 13 ABI (VULKANIZER_CUDART_MAJOR), libcudart.so.13 in lib/cuda13 (for cu13x PyTorch builds;
# a separate directory keeps the libcudart.so links apart).
function(vulkanizer_cudart target major version)
  set(VULKANIZER_CUDART_MAJOR ${major})  # for the version script template
  set(map ${CMAKE_CURRENT_BINARY_DIR}/libcudart${major}.map)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/runtime/libcudart.map.in ${map} @ONLY)
  add_library(${target} SHARED runtime/runtime.cpp)
  target_link_libraries(${target} PRIVATE vulkanizer_common vulkanizer_cuda)
  target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_definitions(${target} PRIVATE VULKANIZER_CUDART_MAJOR=${major})
  set_target_properties(${target} PROPERTIES OUTPUT_NAME cudart VERSION ${version} SOVERSION ${major} LINK_DEPENDS ${map})
  target_link_options(${target} PRIVATE -Wl,--version-script=${map} -Wl,--no-undefined)
endfunction()
vulkanizer_cudart(vulkanizer_cudart 12 12.9.0)
vulkanizer_cudart(vulkanizer_cudart13 13 13.4.0)
set_target_properties(vulkanizer_cudart13 PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/cuda13)

# cuBLAS: libcublas.so -> libcublas.so.12, GEMM kernels written in GLSL.
set(cublas_gen ${CMAKE_CURRENT_BINARY_DIR}/cublas)
set(cublas_shaders)
# name:source[:define] -- variants of one source compiled with different defines.
foreach(entry gemm:gemm trsm:trsm gemv:gemv epilogue:epilogue gemm_f16_tn:gemm_f16_tn gemm_f16_tn_h:gemm_f16_tn:C_F16
    gemm_f16_tn_m:gemm_f16_tn:F16_MATH gemm_f16_tn_mh:gemm_f16_tn:F16_MATH,C_F16)
  string(REPLACE ":" ";" parts ${entry})
  list(GET parts 0 shader)
  list(GET parts 1 source)
  set(defines)
  list(LENGTH parts nparts)
  if(nparts GREATER 2)
    list(GET parts 2 define_list)
    string(REPLACE "," ";" define_list ${define_list})
    foreach(define ${define_list})
      list(APPEND defines -D${define})
    endforeach()
  endif()
  add_custom_command(
    OUTPUT ${cublas_gen}/${shader}.comp.inc
    COMMAND ${CMAKE_COMMAND} -E make_directory ${cublas_gen}
    COMMAND ${GLSLC} --target-env=vulkan1.2 -O -mfmt=num ${defines} -o ${cublas_gen}/${shader}.comp.inc
            ${CMAKE_CURRENT_SOURCE_DIR}/cublas/shaders/${source}.comp
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cublas/shaders/${source}.comp
    COMMENT "glslc ${shader}.comp"
  )
  list(APPEND cublas_shaders ${cublas_gen}/${shader}.comp.inc)
endforeach()
# cuBLAS (GEMM kernels written in GLSL) and cuBLASLt (the matmul on top of it plus epilogues):
# libcublas.so.12 / libcublasLt.so.12, and the same sources as .so.13 in lib/cuda13. The API did not
# change between the two, only the symbol version.
function(vulkanizer_cublas suffix major version outdir)
  set(VULKANIZER_CUDA_MAJOR ${major})  # for the version script templates
  foreach(lib cublas cublasLt)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cublas/lib${lib}.map.in ${CMAKE_CURRENT_BINARY_DIR}/lib${lib}${major}.map @ONLY)
  endforeach()
  add_library(vulkanizer_cublas${suffix} SHARED cublas/cublas.cpp cublas/cublas_stubs.cpp ${cublas_shaders})
  add_library(vulkanizer_cublaslt${suffix} SHARED cublas/cublasLt.cpp ${cublas_gen}/epilogue.comp.inc)
  target_link_libraries(vulkanizer_cublas${suffix} PRIVATE vulkanizer_common vulkanizer_cuda)
  target_link_libraries(vulkanizer_cublaslt${suffix} PRIVATE vulkanizer_common vulkanizer_cuda vulkanizer_cublas${suffix})
  foreach(pair "vulkanizer_cublas${suffix};cublas" "vulkanizer_cublaslt${suffix};cublasLt")
    list(GET pair 0 target)
    list(GET pair 1 lib)
    set(map ${CMAKE_CURRENT_BINARY_DIR}/lib${lib}${major}.map)
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${cublas_gen}
      PUBLIC ${PROJECT_SOURCE_DIR}/include/cublas ${PROJECT_SOURCE_DIR}/include)
    set_target_properties(${target} PROPERTIES OUTPUT_NAME ${lib} VERSION ${version} SOVERSION ${major}
      LINK_DEPENDS ${map} LIBRARY_OUTPUT_DIRECTORY ${outdir})
    target_link_options(${target} PRIVATE -Wl,--version-script=${map} -Wl,--no-undefined)
  endforeach()
endfunction()
vulkanizer_cublas("" 12 12.9.0 ${CMAKE_BINARY_DIR}/lib)
vulkanizer_cublas(13 13 13.8.0 ${CMAKE_BINARY_DIR}/lib/cuda13)
