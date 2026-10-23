# Installation and packages (cmake --install, tools/package.sh).
#
#   <prefix>/lib/vulkanizer/          libcuda.so.1, libcudart.so.12, libcublas(Lt).so.12
#   <prefix>/lib/vulkanizer/cuda13/   libcudart.so.13, libcublas(Lt).so.13
#   <prefix>/bin/                     vulkanizer-run, vulkanizer-info, vulkanizer-ptxc, vulkanizer-bandwidth
#   <prefix>/include/vulkanizer/      cuda.h, vulkanizer/vulkanizer_ext.h, cublas/ (-I<prefix>/include/vulkanizer)
#   <prefix>/share/vulkanizer/        PyTorch and Triton support files for vulkanizer-run --torch
#
# The libraries stay out of the linker's search path (no ld.so.conf entry): a libcuda.so.1 there
# would take the place of an NVIDIA driver for every program. vulkanizer-run puts them in front
# for the program it starts. The headers have their own root for the same reason (cuda.h).

include(GNUInstallDirs)

set(VULKANIZER_INSTALL_LIBDIR ${CMAKE_INSTALL_LIBDIR}/vulkanizer)
set(VULKANIZER_INSTALL_DATADIR ${CMAKE_INSTALL_DATADIR}/vulkanizer)
set(VULKANIZER_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_INCLUDEDIR}/vulkanizer)

# Each library finds its neighbours (libcudart -> libcuda) without LD_LIBRARY_PATH.
set_target_properties(vulkanizer_cuda vulkanizer_cudart vulkanizer_cublas vulkanizer_cublaslt vulkanizer_cusolver vulkanizer_cufft vulkanizer_cusparse
  PROPERTIES INSTALL_RPATH "$ORIGIN")
set_target_properties(vulkanizer_cudart13 vulkanizer_cublas13 vulkanizer_cublaslt13 vulkanizer_cusolver13 vulkanizer_cufft13 vulkanizer_cusparse13
  PROPERTIES INSTALL_RPATH "$ORIGIN:$ORIGIN/..")
file(RELATIVE_PATH lib_from_bin /${CMAKE_INSTALL_BINDIR} /${VULKANIZER_INSTALL_LIBDIR})
set_target_properties(vulkanizer-bandwidth PROPERTIES INSTALL_RPATH "$ORIGIN/${lib_from_bin}")

# libcuda.so too: Triton links its launcher with -lcuda (TRITON_LIBCUDA_PATH).
install(TARGETS vulkanizer_cuda LIBRARY DESTINATION ${VULKANIZER_INSTALL_LIBDIR})
install(TARGETS vulkanizer_cudart vulkanizer_cublas vulkanizer_cublaslt vulkanizer_cusolver vulkanizer_cufft vulkanizer_cusparse vulkanizer_nvrtc
  LIBRARY DESTINATION ${VULKANIZER_INSTALL_LIBDIR} NAMELINK_SKIP)
install(TARGETS vulkanizer_cudart13 vulkanizer_cublas13 vulkanizer_cublaslt13 vulkanizer_cusolver13 vulkanizer_cufft13 vulkanizer_cusparse13
  LIBRARY DESTINATION ${VULKANIZER_INSTALL_LIBDIR}/cuda13 NAMELINK_SKIP)
install(TARGETS vulkanizer-info vulkanizer-ptxc vulkanizer-bandwidth RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

configure_file(${PROJECT_SOURCE_DIR}/tools/vulkanizer-run.in ${PROJECT_BINARY_DIR}/vulkanizer-run @ONLY)
install(PROGRAMS ${PROJECT_BINARY_DIR}/vulkanizer-run DESTINATION ${CMAKE_INSTALL_BINDIR})

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION ${VULKANIZER_INSTALL_INCLUDEDIR})
install(FILES ${PROJECT_SOURCE_DIR}/tools/torch/sitecustomize.py DESTINATION ${VULKANIZER_INSTALL_DATADIR}/torch)
install(PROGRAMS ${PROJECT_SOURCE_DIR}/tools/triton/ptxas DESTINATION ${VULKANIZER_INSTALL_DATADIR}/triton)
install(FILES ${PROJECT_SOURCE_DIR}/README.md ${PROJECT_SOURCE_DIR}/LICENSE DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/vulkanizer)

set(CPACK_PACKAGE_NAME vulkanizer)
set(CPACK_PACKAGE_VENDOR "David Kozma")
set(CPACK_PACKAGE_CONTACT "David Kozma")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "CUDA driver, runtime and cuBLAS on Vulkan compute")
set(CPACK_PACKAGE_DESCRIPTION "Vulkanizer runs CUDA programs (llama.cpp, PyTorch, ...) on any Vulkan GPU:
a drop-in libcuda, libcudart and libcublas that compile PTX to SPIR-V. Start programs with vulkanizer-run.")
set(CPACK_RESOURCE_FILE_LICENSE ${PROJECT_SOURCE_DIR}/LICENSE)
set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${PROJECT_VERSION}-${CMAKE_SYSTEM_PROCESSOR}")
set(CPACK_GENERATOR DEB TGZ)
set(CPACK_INSTALL_DEFAULT_DIRECTORY_PERMISSIONS
  OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)
set(CPACK_DEBIAN_PACKAGE_SECTION utils)
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)  # libvulkan1, libstdc++6, ...
set(CPACK_DEBIAN_PACKAGE_RECOMMENDS "mesa-vulkan-drivers")
set(CPACK_DEBIAN_PACKAGE_HOMEPAGE "https://github.com/k0zi/Vulkanizer")
# The libraries are private (no ldconfig): no shlibs file offering them to other packages.
set(CPACK_DEBIAN_PACKAGE_GENERATE_SHLIBS OFF)
include(CPack)
