#pragma once

// Clean-room declarations of the cuRAND host API implemented by Vulkanizer's libcurand.so, written
// from the public cuRAND documentation. Enum values follow the documented ABI; the data types are
// cuBLAS's (cublas_v2.h).

#include <cublas_v2.h>

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum curandStatus {
    CURAND_STATUS_SUCCESS = 0,
    CURAND_STATUS_VERSION_MISMATCH = 100,
    CURAND_STATUS_NOT_INITIALIZED = 101,
    CURAND_STATUS_ALLOCATION_FAILED = 102,
    CURAND_STATUS_TYPE_ERROR = 103,
    CURAND_STATUS_OUT_OF_RANGE = 104,
    CURAND_STATUS_LENGTH_NOT_MULTIPLE = 105,
    CURAND_STATUS_DOUBLE_PRECISION_REQUIRED = 106,
    CURAND_STATUS_LAUNCH_FAILURE = 201,
    CURAND_STATUS_PREEXISTING_FAILURE = 202,
    CURAND_STATUS_INITIALIZATION_FAILED = 203,
    CURAND_STATUS_ARCH_MISMATCH = 204,
    CURAND_STATUS_INTERNAL_ERROR = 999,
} curandStatus_t;

typedef enum curandRngType {
    CURAND_RNG_TEST = 0,
    CURAND_RNG_PSEUDO_DEFAULT = 100,  // XORWOW
    CURAND_RNG_PSEUDO_XORWOW = 101,
    CURAND_RNG_PSEUDO_MRG32K3A = 121,
    CURAND_RNG_PSEUDO_MTGP32 = 141,
    CURAND_RNG_PSEUDO_MT19937 = 142,
    CURAND_RNG_PSEUDO_PHILOX4_32_10 = 161,
    CURAND_RNG_QUASI_DEFAULT = 200,  // SOBOL32
    CURAND_RNG_QUASI_SOBOL32 = 201,
    CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 = 202,
    CURAND_RNG_QUASI_SOBOL64 = 203,
    CURAND_RNG_QUASI_SCRAMBLED_SOBOL64 = 204,
} curandRngType_t;

typedef enum curandOrdering {
    CURAND_ORDERING_PSEUDO_BEST = 100,
    CURAND_ORDERING_PSEUDO_DEFAULT = 101,
    CURAND_ORDERING_PSEUDO_SEEDED = 102,
    CURAND_ORDERING_PSEUDO_LEGACY = 103,
    CURAND_ORDERING_PSEUDO_DYNAMIC = 104,
    CURAND_ORDERING_QUASI_DEFAULT = 201,
} curandOrdering_t;

typedef enum curandDirectionVectorSet {
    CURAND_DIRECTION_VECTORS_32_JOEKUO6 = 101,
    CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6 = 102,
    CURAND_DIRECTION_VECTORS_64_JOEKUO6 = 103,
    CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6 = 104,
} curandDirectionVectorSet_t;

typedef enum curandMethod {
    CURAND_CHOOSE_BEST = 0,
    CURAND_ITR = 1,
    CURAND_KNUTH = 2,
    CURAND_HITR = 3,
    CURAND_M1 = 4,
    CURAND_M2 = 5,
    CURAND_BINARY_SEARCH = 6,
    CURAND_DISCRETE_GAUSS = 7,
    CURAND_REJECTION = 8,
    CURAND_DEVICE_API = 9,
    CURAND_FAST_REJECTION = 10,
    CURAND_3RD = 11,
    CURAND_DEFINITION = 12,
    CURAND_POISSON = 13,
} curandMethod_t;

typedef unsigned int curandDirectionVectors32_t[32];
typedef unsigned long long curandDirectionVectors64_t[64];

typedef struct curandGenerator_st* curandGenerator_t;
typedef struct curandDiscreteDistribution_st* curandDiscreteDistribution_t;

curandStatus_t curandCreateGenerator(curandGenerator_t* generator, curandRngType_t rng_type);
curandStatus_t curandCreateGeneratorHost(curandGenerator_t* generator, curandRngType_t rng_type);
curandStatus_t curandDestroyGenerator(curandGenerator_t generator);
curandStatus_t curandGetVersion(int* version);
curandStatus_t curandGetProperty(libraryPropertyType type, int* value);
curandStatus_t curandSetStream(curandGenerator_t generator, cudaStream_t stream);
curandStatus_t curandSetPseudoRandomGeneratorSeed(curandGenerator_t generator, unsigned long long seed);
curandStatus_t curandSetGeneratorOffset(curandGenerator_t generator, unsigned long long offset);
curandStatus_t curandSetGeneratorOrdering(curandGenerator_t generator, curandOrdering_t order);
curandStatus_t curandSetQuasiRandomGeneratorDimensions(curandGenerator_t generator, unsigned int num_dimensions);
curandStatus_t curandGenerateSeeds(curandGenerator_t generator);

curandStatus_t curandGenerate(curandGenerator_t generator, unsigned int* outputPtr, size_t num);
curandStatus_t curandGenerateLongLong(curandGenerator_t generator, unsigned long long* outputPtr, size_t num);
curandStatus_t curandGenerateUniform(curandGenerator_t generator, float* outputPtr, size_t num);
curandStatus_t curandGenerateUniformDouble(curandGenerator_t generator, double* outputPtr, size_t num);
curandStatus_t curandGenerateNormal(curandGenerator_t generator, float* outputPtr, size_t n, float mean, float stddev);
curandStatus_t curandGenerateNormalDouble(curandGenerator_t generator, double* outputPtr, size_t n, double mean,
                                          double stddev);
curandStatus_t curandGenerateLogNormal(curandGenerator_t generator, float* outputPtr, size_t n, float mean,
                                       float stddev);
curandStatus_t curandGenerateLogNormalDouble(curandGenerator_t generator, double* outputPtr, size_t n, double mean,
                                             double stddev);
curandStatus_t curandCreatePoissonDistribution(double lambda, curandDiscreteDistribution_t* discrete_distribution);
curandStatus_t curandDestroyDistribution(curandDiscreteDistribution_t discrete_distribution);
curandStatus_t curandGeneratePoisson(curandGenerator_t generator, unsigned int* outputPtr, size_t n, double lambda);
curandStatus_t curandGeneratePoissonMethod(curandGenerator_t generator, unsigned int* outputPtr, size_t n,
                                           double lambda, curandMethod_t method);

curandStatus_t curandGetDirectionVectors32(curandDirectionVectors32_t* vectors[], curandDirectionVectorSet_t set);
curandStatus_t curandGetDirectionVectors64(curandDirectionVectors64_t* vectors[], curandDirectionVectorSet_t set);
curandStatus_t curandGetScrambleConstants32(unsigned int** constants);
curandStatus_t curandGetScrambleConstants64(unsigned long long** constants);

#ifdef __cplusplus
}
#endif
