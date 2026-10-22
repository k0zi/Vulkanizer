add_executable(vulkanizer-info vulkanizer_info.cpp)
target_link_libraries(vulkanizer-info PRIVATE vulkanizer_core)

add_executable(vulkanizer-ptxc vulkanizer_ptxc.cpp)
target_link_libraries(vulkanizer-ptxc PRIVATE vulkanizer_core)

add_executable(vulkanizer-bandwidth vulkanizer_bandwidth.cpp)
target_link_libraries(vulkanizer-bandwidth PRIVATE vulkanizer_cuda)
