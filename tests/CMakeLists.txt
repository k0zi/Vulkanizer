find_program(GLSLC glslc HINTS ${Vulkan_GLSLC_EXECUTABLE} REQUIRED)

set(VULKANIZER_TEST_DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/data)
set(VULKANIZER_SHADER_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)

# GLSL test shaders -> SPIR-V
set(SHADER_OUTPUTS)
foreach(shader vec_add.comp)
  set(src ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${shader})
  set(out ${VULKANIZER_SHADER_DIR}/${shader}.spv)
  add_custom_command(
    OUTPUT ${out}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${VULKANIZER_SHADER_DIR}
    COMMAND ${GLSLC} --target-env=vulkan1.2 -o ${out} ${src}
    DEPENDS ${src}
    COMMENT "glslc ${shader}"
  )
  list(APPEND SHADER_OUTPUTS ${out})
endforeach()
add_custom_target(vulkanizer_test_shaders DEPENDS ${SHADER_OUTPUTS})

# Tests written against the public CUDA Driver API, linked to Vulkanizer's libcuda.so
# exactly like a real CUDA application would be.
function(vulkanizer_cuda_test name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE vulkanizer_cuda)
  target_compile_definitions(${name} PRIVATE VULKANIZER_TEST_DATA_DIR="${VULKANIZER_TEST_DATA_DIR}")
  add_test(NAME ${name} COMMAND ${name})
endfunction()

vulkanizer_cuda_test(device_query)
vulkanizer_cuda_test(memory)
vulkanizer_cuda_test(ptx_module)
vulkanizer_cuda_test(vector_add)
vulkanizer_cuda_test(kernels)
vulkanizer_cuda_test(fatbin)
vulkanizer_cuda_test(vmm)
vulkanizer_cuda_test(warp_matrix)
vulkanizer_cuda_test(cublas)
target_link_libraries(cublas PRIVATE vulkanizer_cublas vulkanizer_cublaslt)

# Internal tests against Vulkanizer's own building blocks.
add_executable(vk_dispatch vk_dispatch.cpp)
target_link_libraries(vk_dispatch PRIVATE vulkanizer_core)
target_compile_definitions(vk_dispatch PRIVATE VULKANIZER_SHADER_DIR="${VULKANIZER_SHADER_DIR}")
add_dependencies(vk_dispatch vulkanizer_test_shaders)
add_test(NAME vk_dispatch COMMAND vk_dispatch)

add_executable(ptx_parse ptx_parse.cpp)
target_link_libraries(ptx_parse PRIVATE vulkanizer_core)
target_compile_definitions(ptx_parse PRIVATE VULKANIZER_TEST_DATA_DIR="${VULKANIZER_TEST_DATA_DIR}")
add_test(NAME ptx_parse COMMAND ptx_parse)

# Tests that need a real CUDA toolkit in .cuda/ (tools/setup_cuda.sh): programs built with
# nvcc against the CUDA Runtime, run on Vulkanizer's libcudart + libcuda, and an ABI check of
# Vulkanizer's clean-room runtime types against the official headers.
set(VULKANIZER_CUDA_TOOLKIT ${PROJECT_SOURCE_DIR}/.cuda CACHE PATH "CUDA toolkit used to build the nvcc tests")
if(EXISTS ${VULKANIZER_CUDA_TOOLKIT}/bin/nvcc)
  find_program(VULKANIZER_NVCC_HOST_CXX NAMES g++-13 g++-12 g++-14 g++ REQUIRED)
  set(nvcc_out ${CMAKE_CURRENT_BINARY_DIR}/cuda)
  foreach(name rt_vector_add rt_features rt_kernels rt_printf rt_math_f64)
    set(src ${CMAKE_CURRENT_SOURCE_DIR}/cuda/${name}.cu)
    add_custom_command(
      OUTPUT ${nvcc_out}/${name}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${nvcc_out}
      # PTX only (no SASS) and a shared libcudart, like a JIT-friendly application build
      COMMAND ${VULKANIZER_CUDA_TOOLKIT}/bin/nvcc -ccbin ${VULKANIZER_NVCC_HOST_CXX} -O2
              -gencode arch=compute_75,code=compute_75 -cudart shared -o ${nvcc_out}/${name} ${src}
      DEPENDS ${src}
      COMMENT "nvcc ${name}.cu"
    )
    add_custom_target(${name}_nvcc ALL DEPENDS ${nvcc_out}/${name})
    add_dependencies(${name}_nvcc vulkanizer_cudart vulkanizer_cuda)
    add_test(NAME ${name} COMMAND ${nvcc_out}/${name})
    set(env "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/lib")
    if(VULKANIZER_ASAN)  # nvcc's binaries are not instrumented, so the sanitizer runtime is preloaded
      execute_process(COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libasan.so
                      OUTPUT_VARIABLE asan_lib OUTPUT_STRIP_TRAILING_WHITESPACE)
      list(APPEND env "LD_PRELOAD=${asan_lib}")
    endif()
    set_tests_properties(${name} PROPERTIES ENVIRONMENT "${env}")
  endforeach()

  set_tests_properties(rt_printf PROPERTIES PASS_REGULAR_EXPRESSION
    "hello from the GPU\nint -7, unsigned 4000000000, hex 0xff, char Z\nlong -1234567890123, float 3.142, double 0.0025, padded \\[   42\\]\nstring 'device', percent 100%\nthread 3 of 32\n.*done: no error")

  add_executable(cudart_abi cudart_abi/main.cpp cudart_abi/nvidia.cpp cudart_abi/vulkanizer.cpp)
  set_source_files_properties(cudart_abi/nvidia.cpp PROPERTIES INCLUDE_DIRECTORIES ${VULKANIZER_CUDA_TOOLKIT}/include)
  target_include_directories(cudart_abi PRIVATE ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/include)
  add_test(NAME cudart_abi COMMAND cudart_abi)

  add_executable(cuda_abi cuda_abi/main.cpp cuda_abi/nvidia.cpp cuda_abi/vulkanizer.cpp)
  set_source_files_properties(cuda_abi/nvidia.cpp PROPERTIES INCLUDE_DIRECTORIES ${VULKANIZER_CUDA_TOOLKIT}/include)
  add_test(NAME cuda_abi COMMAND cuda_abi)
else()
  message(STATUS "No CUDA toolkit in ${VULKANIZER_CUDA_TOOLKIT}: nvcc tests disabled (run tools/setup_cuda.sh)")
endif()
