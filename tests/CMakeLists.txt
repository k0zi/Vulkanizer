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
vulkanizer_cuda_test(graph)
vulkanizer_cuda_test(host_memory)
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

# Tests that need a real CUDA toolkit (tools/setup_cuda.sh): programs built with nvcc against the
# CUDA Runtime, run on Vulkanizer's libcudart + libcuda, and an ABI check of Vulkanizer's clean-room
# runtime types against the official headers. Once per runtime major version: CUDA 12 in .cuda/
# (libcudart.so.12), CUDA 13 in .cuda13/ (libcudart.so.13, lib/cuda13); CUDA 13 tests end in _13.
function(vulkanizer_nvcc_tests toolkit major suffix libdir)
  find_program(VULKANIZER_NVCC_HOST_CXX NAMES g++-13 g++-12 g++-14 g++ REQUIRED)
  set(nvcc_out ${CMAKE_CURRENT_BINARY_DIR}/cuda${suffix})
  foreach(name rt_vector_add rt_features rt_kernels rt_printf rt_math_f64)
    set(src ${CMAKE_CURRENT_SOURCE_DIR}/cuda/${name}.cu)
    add_custom_command(
      OUTPUT ${nvcc_out}/${name}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${nvcc_out}
      # PTX only (no SASS) and a shared libcudart, like a JIT-friendly application build
      COMMAND ${toolkit}/bin/nvcc -ccbin ${VULKANIZER_NVCC_HOST_CXX} -O2
              -gencode arch=compute_75,code=compute_75 -cudart shared -o ${nvcc_out}/${name} ${src}
      DEPENDS ${src}
      COMMENT "nvcc ${name}.cu (CUDA ${major})"
    )
    add_custom_target(${name}${suffix}_nvcc ALL DEPENDS ${nvcc_out}/${name})
    add_dependencies(${name}${suffix}_nvcc vulkanizer_cudart vulkanizer_cudart13 vulkanizer_cuda)
    add_test(NAME ${name}${suffix} COMMAND ${nvcc_out}/${name})
    set(env "LD_LIBRARY_PATH=${libdir}:${CMAKE_BINARY_DIR}/lib")
    if(VULKANIZER_ASAN)  # nvcc's binaries are not instrumented, so the sanitizer runtime is preloaded
      execute_process(COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libasan.so
                      OUTPUT_VARIABLE asan_lib OUTPUT_STRIP_TRAILING_WHITESPACE)
      list(APPEND env "LD_PRELOAD=${asan_lib}")
    endif()
    set_tests_properties(${name}${suffix} PROPERTIES ENVIRONMENT "${env}")
  endforeach()

  set_tests_properties(rt_printf${suffix} PROPERTIES PASS_REGULAR_EXPRESSION
    "hello from the GPU\nint -7, unsigned 4000000000, hex 0xff, char Z\nlong -1234567890123, float 3.142, double 0.0025, padded \\[   42\\]\nstring 'device', percent 100%\nthread 3 of 32\n.*done: no error")

  # the official headers and Vulkanizer's clash, so each side is its own translation unit
  add_library(cudart_abi_nvidia${suffix} OBJECT cudart_abi/nvidia.cpp)
  target_include_directories(cudart_abi_nvidia${suffix} PRIVATE ${toolkit}/include)
  target_compile_definitions(cudart_abi_nvidia${suffix} PRIVATE VULKANIZER_CUDART_MAJOR=${major})
  add_executable(cudart_abi${suffix} cudart_abi/main.cpp cudart_abi/vulkanizer.cpp $<TARGET_OBJECTS:cudart_abi_nvidia${suffix}>)
  target_include_directories(cudart_abi${suffix} PRIVATE ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/include)
  target_compile_definitions(cudart_abi${suffix} PRIVATE VULKANIZER_CUDART_MAJOR=${major})
  add_test(NAME cudart_abi${suffix} COMMAND cudart_abi${suffix})
endfunction()

set(VULKANIZER_CUDA_TOOLKIT ${PROJECT_SOURCE_DIR}/.cuda CACHE PATH "CUDA 12 toolkit used to build the nvcc tests")
if(EXISTS ${VULKANIZER_CUDA_TOOLKIT}/bin/nvcc)
  vulkanizer_nvcc_tests(${VULKANIZER_CUDA_TOOLKIT} 12 "" ${CMAKE_BINARY_DIR}/lib)
  add_executable(cuda_abi cuda_abi/main.cpp cuda_abi/nvidia.cpp cuda_abi/vulkanizer.cpp)
  set_source_files_properties(cuda_abi/nvidia.cpp PROPERTIES INCLUDE_DIRECTORIES ${VULKANIZER_CUDA_TOOLKIT}/include)
  add_test(NAME cuda_abi COMMAND cuda_abi)
else()
  message(STATUS "No CUDA toolkit in ${VULKANIZER_CUDA_TOOLKIT}: nvcc tests disabled (run tools/setup_cuda.sh)")
endif()
set(VULKANIZER_CUDA13_TOOLKIT ${PROJECT_SOURCE_DIR}/.cuda13 CACHE PATH "CUDA 13 toolkit used to build the nvcc tests")
if(EXISTS ${VULKANIZER_CUDA13_TOOLKIT}/bin/nvcc)
  vulkanizer_nvcc_tests(${VULKANIZER_CUDA13_TOOLKIT} 13 _13 ${CMAKE_BINARY_DIR}/lib/cuda13)
  # the same libcuda.so.1 serves CUDA 13 programs: its types must match cuda.h 13 too
  add_library(cuda_abi_nvidia_13 OBJECT cuda_abi/nvidia.cpp)
  target_include_directories(cuda_abi_nvidia_13 PRIVATE ${VULKANIZER_CUDA13_TOOLKIT}/include)
  add_executable(cuda_abi_13 cuda_abi/main.cpp cuda_abi/vulkanizer.cpp $<TARGET_OBJECTS:cuda_abi_nvidia_13>)
  add_test(NAME cuda_abi_13 COMMAND cuda_abi_13)
else()
  message(STATUS "No CUDA 13 toolkit in ${VULKANIZER_CUDA13_TOOLKIT}: CUDA 13 tests disabled (tools/setup_cuda.sh --cuda13)")
endif()

# libtorch tests against the conda-forge PyTorch environments (tools/setup_torch.sh): torch_* against
# the CUDA 12 build in .work/torch-cf, torch13_* against the CUDA 13 one in .work/torch-cf13 (--cuda13).
# Not part of ctest: they need run_torch.sh's environment and take a while (./run_torch.sh torch_ops,
# ./run_torch.sh --env .work/torch-cf13 torch13_ops).
# torch_graph uses ATen's CUDA headers, which need the CUDA toolkit's of the same major version.
function(vulkanizer_torch_tests env prefix toolkit)
  if(NOT EXISTS ${env}/lib/libtorch_cuda.so)
    message(STATUS "No PyTorch in ${env}: ${prefix}_* tests disabled (run tools/setup_torch.sh)")
    return()
  endif()
  set(names ops mlp split)
  if(EXISTS ${toolkit}/include/cuda_runtime_api.h)
    list(APPEND names graph)
  endif()
  foreach(name ${names})
    set(target ${prefix}_${name})
    add_executable(${target} torch/torch_${name}.cpp)
    target_include_directories(${target} SYSTEM PRIVATE ${env}/include ${env}/include/torch/csrc/api/include)
    if(name STREQUAL graph)
      target_include_directories(${target} SYSTEM PRIVATE ${toolkit}/include)
      # the environment's libcudart for linking; run_torch.sh preloads Vulkanizer's in its place
      file(GLOB cudart ${env}/lib/libcudart.so.[0-9][0-9])
      target_link_libraries(${target} PRIVATE ${env}/lib/libc10_cuda.so ${cudart})
    endif()
    # libtorch_cuda registers the CUDA backend from static initializers: keep it although nothing references it
    target_link_libraries(${target} PRIVATE -Wl,--no-as-needed ${env}/lib/libtorch_cuda.so -Wl,--as-needed
                          ${env}/lib/libtorch.so ${env}/lib/libtorch_cpu.so ${env}/lib/libc10.so)
    set_target_properties(${target} PROPERTIES BUILD_RPATH ${env}/lib)
  endforeach()
endfunction()
set(VULKANIZER_TORCH_ENV ${PROJECT_SOURCE_DIR}/.work/torch-cf CACHE PATH "CUDA 12 PyTorch environment for the libtorch tests")
set(VULKANIZER_TORCH13_ENV ${PROJECT_SOURCE_DIR}/.work/torch-cf13 CACHE PATH "CUDA 13 PyTorch environment for the libtorch tests")
vulkanizer_torch_tests(${VULKANIZER_TORCH_ENV} torch ${VULKANIZER_CUDA_TOOLKIT})
vulkanizer_torch_tests(${VULKANIZER_TORCH13_ENV} torch13 ${VULKANIZER_CUDA13_TOOLKIT})
