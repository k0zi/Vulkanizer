add_executable(vulkanizer-info vulkanizer_info.cpp)
target_link_libraries(vulkanizer-info PRIVATE vulkanizer_core vulkanizer_npu)

add_executable(vulkanizer-check vulkanizer_check.cpp)
target_link_libraries(vulkanizer-check PRIVATE vulkanizer_core)

add_executable(vulkanizer-ptxc vulkanizer_ptxc.cpp)
target_link_libraries(vulkanizer-ptxc PRIVATE vulkanizer_core)

add_executable(vulkanizer-bandwidth vulkanizer_bandwidth.cpp)
target_link_libraries(vulkanizer-bandwidth PRIVATE vulkanizer_cuda)

if(VULKANIZER_NPU_XRT)
  add_executable(vulkanizer-npu-bench vulkanizer_npu_bench.cpp)
  target_link_libraries(vulkanizer-npu-bench PRIVATE vulkanizer_npu vulkanizer_cuda vulkanizer_cublas)
endif()
