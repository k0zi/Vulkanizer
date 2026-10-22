// Generated from src/runtime/cudart_types.h: what tests/cudart_abi compares.
// VULKANIZER_CUDART_MAJOR (12 or 13) selects the fields of that runtime version.
#if VULKANIZER_CUDART_MAJOR >= 13
#define VULKANIZER_PROP_12(X, f)
#define VULKANIZER_PROP_13(X, f) X(f)
#else
#define VULKANIZER_PROP_12(X, f) X(f)
#define VULKANIZER_PROP_13(X, f)
#endif
#define VULKANIZER_DEVICE_PROP_FIELDS(X) \
    X(name) \
    X(uuid) \
    X(luid) \
    X(luidDeviceNodeMask) \
    X(totalGlobalMem) \
    X(sharedMemPerBlock) \
    X(regsPerBlock) \
    X(warpSize) \
    X(memPitch) \
    X(maxThreadsPerBlock) \
    X(maxThreadsDim) \
    X(maxGridSize) \
    VULKANIZER_PROP_12(X, clockRate) \
    X(totalConstMem) \
    X(major) \
    X(minor) \
    X(textureAlignment) \
    X(texturePitchAlignment) \
    VULKANIZER_PROP_12(X, deviceOverlap) \
    X(multiProcessorCount) \
    VULKANIZER_PROP_12(X, kernelExecTimeoutEnabled) \
    X(integrated) \
    X(canMapHostMemory) \
    VULKANIZER_PROP_12(X, computeMode) \
    X(maxTexture1D) \
    X(maxTexture1DMipmap) \
    VULKANIZER_PROP_12(X, maxTexture1DLinear) \
    X(maxTexture2D) \
    X(maxTexture2DMipmap) \
    X(maxTexture2DLinear) \
    X(maxTexture2DGather) \
    X(maxTexture3D) \
    X(maxTexture3DAlt) \
    X(maxTextureCubemap) \
    X(maxTexture1DLayered) \
    X(maxTexture2DLayered) \
    X(maxTextureCubemapLayered) \
    X(maxSurface1D) \
    X(maxSurface2D) \
    X(maxSurface3D) \
    X(maxSurface1DLayered) \
    X(maxSurface2DLayered) \
    X(maxSurfaceCubemap) \
    X(maxSurfaceCubemapLayered) \
    X(surfaceAlignment) \
    X(concurrentKernels) \
    X(ECCEnabled) \
    X(pciBusID) \
    X(pciDeviceID) \
    X(pciDomainID) \
    X(tccDriver) \
    X(asyncEngineCount) \
    X(unifiedAddressing) \
    VULKANIZER_PROP_12(X, memoryClockRate) \
    X(memoryBusWidth) \
    X(l2CacheSize) \
    X(persistingL2CacheMaxSize) \
    X(maxThreadsPerMultiProcessor) \
    X(streamPrioritiesSupported) \
    X(globalL1CacheSupported) \
    X(localL1CacheSupported) \
    X(sharedMemPerMultiprocessor) \
    X(regsPerMultiprocessor) \
    X(managedMemory) \
    X(isMultiGpuBoard) \
    X(multiGpuBoardGroupID) \
    X(hostNativeAtomicSupported) \
    VULKANIZER_PROP_12(X, singleToDoublePrecisionPerfRatio) \
    X(pageableMemoryAccess) \
    X(concurrentManagedAccess) \
    X(computePreemptionSupported) \
    X(canUseHostPointerForRegisteredMem) \
    X(cooperativeLaunch) \
    VULKANIZER_PROP_12(X, cooperativeMultiDeviceLaunch) \
    X(sharedMemPerBlockOptin) \
    X(pageableMemoryAccessUsesHostPageTables) \
    X(directManagedMemAccessFromHost) \
    X(maxBlocksPerMultiProcessor) \
    X(accessPolicyMaxWindowSize) \
    X(reservedSharedMemPerBlock) \
    X(hostRegisterSupported) \
    X(sparseCudaArraySupported) \
    X(hostRegisterReadOnlySupported) \
    X(timelineSemaphoreInteropSupported) \
    X(memoryPoolsSupported) \
    X(gpuDirectRDMASupported) \
    X(gpuDirectRDMAFlushWritesOptions) \
    X(gpuDirectRDMAWritesOrdering) \
    X(memoryPoolSupportedHandleTypes) \
    X(deferredMappingCudaArraySupported) \
    X(ipcEventSupported) \
    X(clusterLaunch) \
    X(unifiedFunctionPointers) \
    VULKANIZER_PROP_13(X, deviceNumaConfig) \
    VULKANIZER_PROP_13(X, deviceNumaId) \
    VULKANIZER_PROP_13(X, mpsEnabled) \
    VULKANIZER_PROP_13(X, hostNumaId) \
    VULKANIZER_PROP_13(X, gpuPciDeviceID) \
    VULKANIZER_PROP_13(X, gpuPciSubsystemID) \
    VULKANIZER_PROP_13(X, hostNumaMultinodeIpcSupported) \
    X(reserved)

#define VULKANIZER_FUNC_ATTR_FIELDS(X) \
    X(sharedSizeBytes) \
    X(constSizeBytes) \
    X(localSizeBytes) \
    X(maxThreadsPerBlock) \
    X(numRegs) \
    X(ptxVersion) \
    X(binaryVersion) \
    X(cacheModeCA) \
    X(maxDynamicSharedSizeBytes) \
    X(preferredShmemCarveout) \
    X(clusterDimMustBeSet) \
    X(requiredClusterWidth) \
    X(requiredClusterHeight) \
    X(requiredClusterDepth) \
    X(clusterSchedulingPolicyPreference) \
    X(nonPortableClusterSizeAllowed) \
    VULKANIZER_PROP_13(X, deviceNodeUpdateStatus) \
    VULKANIZER_PROP_13(X, sharedMemoryMode) \
    X(reserved)

#define VULKANIZER_ERROR_VALUES(X) \
    X(cudaSuccess) \
    X(cudaErrorInvalidValue) \
    X(cudaErrorMemoryAllocation) \
    X(cudaErrorInitializationError) \
    X(cudaErrorCudartUnloading) \
    X(cudaErrorProfilerDisabled) \
    X(cudaErrorInvalidConfiguration) \
    X(cudaErrorInvalidPitchValue) \
    X(cudaErrorInvalidSymbol) \
    X(cudaErrorInvalidHostPointer) \
    X(cudaErrorInvalidDevicePointer) \
    X(cudaErrorInvalidTexture) \
    X(cudaErrorInvalidChannelDescriptor) \
    X(cudaErrorInvalidMemcpyDirection) \
    X(cudaErrorInvalidFilterSetting) \
    X(cudaErrorInvalidNormSetting) \
    X(cudaErrorStubLibrary) \
    X(cudaErrorInsufficientDriver) \
    X(cudaErrorCallRequiresNewerDriver) \
    X(cudaErrorInvalidSurface) \
    X(cudaErrorDuplicateVariableName) \
    X(cudaErrorDuplicateTextureName) \
    X(cudaErrorDuplicateSurfaceName) \
    X(cudaErrorDevicesUnavailable) \
    X(cudaErrorIncompatibleDriverContext) \
    X(cudaErrorMissingConfiguration) \
    X(cudaErrorLaunchMaxDepthExceeded) \
    X(cudaErrorInvalidDeviceFunction) \
    X(cudaErrorNoDevice) \
    X(cudaErrorInvalidDevice) \
    X(cudaErrorDeviceNotLicensed) \
    X(cudaErrorSoftwareValidityNotEstablished) \
    X(cudaErrorStartupFailure) \
    X(cudaErrorInvalidKernelImage) \
    X(cudaErrorDeviceUninitialized) \
    X(cudaErrorMapBufferObjectFailed) \
    X(cudaErrorUnmapBufferObjectFailed) \
    X(cudaErrorArrayIsMapped) \
    X(cudaErrorAlreadyMapped) \
    X(cudaErrorNoKernelImageForDevice) \
    X(cudaErrorAlreadyAcquired) \
    X(cudaErrorNotMapped) \
    X(cudaErrorNotMappedAsArray) \
    X(cudaErrorNotMappedAsPointer) \
    X(cudaErrorECCUncorrectable) \
    X(cudaErrorUnsupportedLimit) \
    X(cudaErrorDeviceAlreadyInUse) \
    X(cudaErrorPeerAccessUnsupported) \
    X(cudaErrorInvalidPtx) \
    X(cudaErrorInvalidGraphicsContext) \
    X(cudaErrorNvlinkUncorrectable) \
    X(cudaErrorJitCompilerNotFound) \
    X(cudaErrorUnsupportedPtxVersion) \
    X(cudaErrorJitCompilationDisabled) \
    X(cudaErrorUnsupportedExecAffinity) \
    X(cudaErrorInvalidSource) \
    X(cudaErrorFileNotFound) \
    X(cudaErrorSharedObjectSymbolNotFound) \
    X(cudaErrorSharedObjectInitFailed) \
    X(cudaErrorOperatingSystem) \
    X(cudaErrorInvalidResourceHandle) \
    X(cudaErrorIllegalState) \
    X(cudaErrorSymbolNotFound) \
    X(cudaErrorNotReady) \
    X(cudaErrorIllegalAddress) \
    X(cudaErrorLaunchOutOfResources) \
    X(cudaErrorLaunchTimeout) \
    X(cudaErrorLaunchIncompatibleTexturing) \
    X(cudaErrorPeerAccessAlreadyEnabled) \
    X(cudaErrorPeerAccessNotEnabled) \
    X(cudaErrorSetOnActiveProcess) \
    X(cudaErrorContextIsDestroyed) \
    X(cudaErrorAssert) \
    X(cudaErrorTooManyPeers) \
    X(cudaErrorHostMemoryAlreadyRegistered) \
    X(cudaErrorHostMemoryNotRegistered) \
    X(cudaErrorHardwareStackError) \
    X(cudaErrorIllegalInstruction) \
    X(cudaErrorMisalignedAddress) \
    X(cudaErrorInvalidAddressSpace) \
    X(cudaErrorInvalidPc) \
    X(cudaErrorLaunchFailure) \
    X(cudaErrorCooperativeLaunchTooLarge) \
    X(cudaErrorNotPermitted) \
    X(cudaErrorNotSupported) \
    X(cudaErrorSystemNotReady) \
    X(cudaErrorSystemDriverMismatch) \
    X(cudaErrorCompatNotSupportedOnDevice) \
    X(cudaErrorStreamCaptureUnsupported) \
    X(cudaErrorStreamCaptureInvalidated) \
    X(cudaErrorStreamCaptureMerge) \
    X(cudaErrorStreamCaptureUnmatched) \
    X(cudaErrorStreamCaptureUnjoined) \
    X(cudaErrorStreamCaptureIsolation) \
    X(cudaErrorStreamCaptureImplicit) \
    X(cudaErrorCapturedEvent) \
    X(cudaErrorStreamCaptureWrongThread) \
    X(cudaErrorTimeout) \
    X(cudaErrorGraphExecUpdateFailure) \
    X(cudaErrorExternalDevice) \
    X(cudaErrorInvalidClusterSize) \
    X(cudaErrorUnknown)

#define VULKANIZER_LAUNCH_CONFIG_FIELDS(X) \
    X(gridDim) \
    X(blockDim) \
    X(dynamicSmemBytes) \
    X(stream) \
    X(attrs) \
    X(numAttrs)
