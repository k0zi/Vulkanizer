#pragma once

// Clean-room declarations of the cuFFT API subset implemented by Vulkanizer's libcufft.so, written
// from the public cuFFT documentation. Enum values follow the documented ABI; the data types are
// cuBLAS's (cublas_v2.h).

#include <cublas_v2.h>

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum cufftResult_t {
    CUFFT_SUCCESS = 0,
    CUFFT_INVALID_PLAN = 1,
    CUFFT_ALLOC_FAILED = 2,
    CUFFT_INVALID_TYPE = 3,
    CUFFT_INVALID_VALUE = 4,
    CUFFT_INTERNAL_ERROR = 5,
    CUFFT_EXEC_FAILED = 6,
    CUFFT_SETUP_FAILED = 7,
    CUFFT_INVALID_SIZE = 8,
    CUFFT_UNALIGNED_DATA = 9,
    CUFFT_INCOMPLETE_PARAMETER_LIST = 10,
    CUFFT_INVALID_DEVICE = 11,
    CUFFT_PARSE_ERROR = 12,
    CUFFT_NO_WORKSPACE = 13,
    CUFFT_NOT_IMPLEMENTED = 14,
    CUFFT_LICENSE_ERROR = 15,
    CUFFT_NOT_SUPPORTED = 16,
} cufftResult;

typedef int cufftHandle;
typedef float cufftReal;
typedef double cufftDoubleReal;
typedef struct {
    float x, y;
} cufftComplex;
typedef struct {
    double x, y;
} cufftDoubleComplex;

typedef enum cufftType_t {
    CUFFT_R2C = 0x2a,  // real to complex (forward)
    CUFFT_C2R = 0x2c,  // complex to real (inverse)
    CUFFT_C2C = 0x29,
    CUFFT_D2Z = 0x6a,
    CUFFT_Z2D = 0x6c,
    CUFFT_Z2Z = 0x69,
} cufftType;

#define CUFFT_FORWARD -1
#define CUFFT_INVERSE 1

cufftResult cufftCreate(cufftHandle* plan);
cufftResult cufftDestroy(cufftHandle plan);
cufftResult cufftSetStream(cufftHandle plan, cudaStream_t stream);
cufftResult cufftSetAutoAllocation(cufftHandle plan, int autoAllocate);
cufftResult cufftSetWorkArea(cufftHandle plan, void* workArea);
cufftResult cufftGetSize(cufftHandle plan, size_t* workSize);
cufftResult cufftGetVersion(int* version);
cufftResult cufftGetProperty(libraryPropertyType type, int* value);

cufftResult cufftPlan1d(cufftHandle* plan, int nx, cufftType type, int batch);
cufftResult cufftPlan2d(cufftHandle* plan, int nx, int ny, cufftType type);
cufftResult cufftPlan3d(cufftHandle* plan, int nx, int ny, int nz, cufftType type);
cufftResult cufftPlanMany(cufftHandle* plan, int rank, int* n, int* inembed, int istride, int idist, int* onembed,
                          int ostride, int odist, cufftType type, int batch);
cufftResult cufftMakePlan1d(cufftHandle plan, int nx, cufftType type, int batch, size_t* workSize);
cufftResult cufftMakePlan2d(cufftHandle plan, int nx, int ny, cufftType type, size_t* workSize);
cufftResult cufftMakePlan3d(cufftHandle plan, int nx, int ny, int nz, cufftType type, size_t* workSize);
cufftResult cufftMakePlanMany(cufftHandle plan, int rank, int* n, int* inembed, int istride, int idist,
                              int* onembed, int ostride, int odist, cufftType type, int batch, size_t* workSize);
cufftResult cufftMakePlanMany64(cufftHandle plan, int rank, long long int* n, long long int* inembed,
                                long long int istride, long long int idist, long long int* onembed,
                                long long int ostride, long long int odist, cufftType type, long long int batch,
                                size_t* workSize);

cufftResult cufftExecC2C(cufftHandle plan, cufftComplex* idata, cufftComplex* odata, int direction);
cufftResult cufftExecR2C(cufftHandle plan, cufftReal* idata, cufftComplex* odata);
cufftResult cufftExecC2R(cufftHandle plan, cufftComplex* idata, cufftReal* odata);
cufftResult cufftExecZ2Z(cufftHandle plan, cufftDoubleComplex* idata, cufftDoubleComplex* odata, int direction);
cufftResult cufftExecD2Z(cufftHandle plan, cufftDoubleReal* idata, cufftDoubleComplex* odata);
cufftResult cufftExecZ2D(cufftHandle plan, cufftDoubleComplex* idata, cufftDoubleReal* odata);

// cufftXt.h: the data-type-generic forms (f16, f32, f64)
cufftResult cufftXtMakePlanMany(cufftHandle plan, int rank, long long int* n, long long int* inembed,
                                long long int istride, long long int idist, cudaDataType inputtype,
                                long long int* onembed, long long int ostride, long long int odist,
                                cudaDataType outputtype, long long int batch, size_t* workSize,
                                cudaDataType executiontype);
cufftResult cufftXtExec(cufftHandle plan, void* input, void* output, int direction);

#ifdef __cplusplus
}
#endif
