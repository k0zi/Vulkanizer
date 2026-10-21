# Logging, shared by the driver and the runtime library.
add_library(vulkanizer_common STATIC common/log.cpp common/sha256.cpp)
target_include_directories(vulkanizer_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

find_program(GLSLC glslc HINTS ${Vulkan_GLSLC_EXECUTABLE} REQUIRED)
set(vk_gen ${CMAKE_CURRENT_BINARY_DIR}/vk)
add_custom_command(
  OUTPUT ${vk_gen}/fma_probe.comp.inc
  COMMAND ${CMAKE_COMMAND} -E make_directory ${vk_gen}
  COMMAND ${GLSLC} --target-env=vulkan1.2 -O -mfmt=num -o ${vk_gen}/fma_probe.comp.inc
          ${CMAKE_CURRENT_SOURCE_DIR}/vk/shaders/fma_probe.comp
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/vk/shaders/fma_probe.comp
  COMMENT "glslc fma_probe.comp"
)

# Internal building blocks, linked statically into libcuda.so (and into internal tests).
add_library(vulkanizer_core STATIC
  ${vk_gen}/fma_probe.comp.inc
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

# The CUDA Runtime: libcudart.so -> libcudart.so.12, implemented on the driver above.
add_library(vulkanizer_cudart SHARED runtime/runtime.cpp)
target_link_libraries(vulkanizer_cudart PRIVATE vulkanizer_common vulkanizer_cuda)
target_include_directories(vulkanizer_cudart PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(vulkanizer_cudart PROPERTIES
  OUTPUT_NAME cudart
  VERSION 12.9.0
  SOVERSION 12
  LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/runtime/libcudart.map
)
target_link_options(vulkanizer_cudart PRIVATE
  -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/runtime/libcudart.map
  -Wl,--no-undefined
)

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
add_library(vulkanizer_cublas SHARED cublas/cublas.cpp cublas/cublas_stubs.cpp ${cublas_shaders})
target_link_libraries(vulkanizer_cublas PRIVATE vulkanizer_common vulkanizer_cuda)
target_include_directories(vulkanizer_cublas PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${cublas_gen}
  PUBLIC ${PROJECT_SOURCE_DIR}/include/cublas ${PROJECT_SOURCE_DIR}/include)
set_target_properties(vulkanizer_cublas PROPERTIES
  OUTPUT_NAME cublas
  VERSION 12.9.0
  SOVERSION 12
  LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cublas/libcublas.map
)
target_link_options(vulkanizer_cublas PRIVATE
  -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/cublas/libcublas.map
  -Wl,--no-undefined
)

# cuBLASLt: libcublasLt.so -> libcublasLt.so.12, the matmul on top of libcublas plus epilogues.
add_library(vulkanizer_cublaslt SHARED cublas/cublasLt.cpp ${cublas_gen}/epilogue.comp.inc)
target_link_libraries(vulkanizer_cublaslt PRIVATE vulkanizer_common vulkanizer_cuda vulkanizer_cublas)
target_include_directories(vulkanizer_cublaslt PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${cublas_gen}
  PUBLIC ${PROJECT_SOURCE_DIR}/include/cublas ${PROJECT_SOURCE_DIR}/include)
set_target_properties(vulkanizer_cublaslt PROPERTIES
  OUTPUT_NAME cublasLt
  VERSION 12.9.0
  SOVERSION 12
  LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cublas/libcublasLt.map
)
target_link_options(vulkanizer_cublaslt PRIVATE
  -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/cublas/libcublasLt.map
  -Wl,--no-undefined
)
