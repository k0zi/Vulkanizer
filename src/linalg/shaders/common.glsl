// Shared by the dense linear algebra kernels (cuSOLVER and cuBLAS factorizations): the element
// type (f32, or f64 with -DDOUBLE), column-major matrix access and batch addressing.

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#ifdef DOUBLE
#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require
#define real double
#define REAL_BYTES 8ul
layout(buffer_reference, std430, buffer_reference_align = 8) coherent buffer Real { double v; };
#else
#define real float
#define REAL_BYTES 4ul
layout(buffer_reference, std430, buffer_reference_align = 4) coherent buffer Real { float v; };
#endif
layout(buffer_reference, std430, buffer_reference_align = 4) coherent buffer Int { int v; };
layout(buffer_reference, std430, buffer_reference_align = 8) readonly buffer Ptr { uint64_t v; };

// The device's address bits 63..47 (vk::Device::hostAddressBits), specialization constant 5:
// addresses arrive without them.
layout(constant_id = 5) const uint kAddressHigh = 0xFFFF8000u;
uint64_t canon(uint64_t a) {
    uvec2 w = unpackUint2x32(a);
    w.y |= kAddressHigh;
    return packUint2x32(w);
}

// Push constants of every kernel (std430; linalg.cpp's Params).
layout(push_constant, std430) uniform Params {
    uint64_t a, b;        // matrices A and B: base addresses, or device arrays of pointers (kPtrA/kPtrB)
    uint64_t ipiv, info;  // int arrays (0: none)
    uint64_t tau;         // Householder scalars (real array), strideP apart per batch entry
    int64_t strideA, strideB;  // elements between batch entries (strided batches)
    int64_t strideP;           // pivots (or taus) between batch entries
    int m, n, nrhs, k;   // sizes; k: the column of this step
    int lda, ldb;
    uint flags;
    int batchOffset;
    uint64_t alphaBits;  // a scalar: f32 bits in the low word, or f64 bits (DOUBLE)
    uint64_t c;          // a third matrix (SVD: V), strideC elements apart per batch entry
    int64_t strideC;
    int ldc, pad;
} p;

real alpha() {
#ifdef DOUBLE
    return uint64BitsToDouble(p.alphaBits);
#else
    return uintBitsToFloat(uint(p.alphaBits));
#endif
}

const uint kPtrA = 1u, kPtrB = 2u, kUpper = 4u, kTrans = 8u, kUnit = 16u, kNoPivot = 32u, kPivotsBefore = 64u,
           kPivotsAfter = 128u, kRight = 256u, kPtrTau = 512u, kScale = 1024u, kNoVectors = 2048u, kVt = 4096u,
           kNoLong = 8192u, kNoShort = 16384u, kPtrC = 32768u, kTransB = 65536u;

int batchIndex() { return int(gl_WorkGroupID.z) + p.batchOffset; }

uint64_t matrixA(int batch) {
    return (p.flags & kPtrA) != 0u ? Ptr(canon(p.a + uint64_t(batch) * 8ul)).v : p.a + uint64_t(int64_t(batch) * p.strideA) * REAL_BYTES;
}
uint64_t matrixC(int batch) {
    return (p.flags & kPtrC) != 0u ? Ptr(canon(p.c + uint64_t(batch) * 8ul)).v : p.c + uint64_t(int64_t(batch) * p.strideC) * REAL_BYTES;
}
uint64_t matrixB(int batch) {
    return (p.flags & kPtrB) != 0u ? Ptr(canon(p.b + uint64_t(batch) * 8ul)).v : p.b + uint64_t(int64_t(batch) * p.strideB) * REAL_BYTES;
}

// Element (i, j) of a column-major matrix.
Real elem(uint64_t base, int i, int j, int ld) {
    return Real(canon(base + uint64_t(int64_t(i) + int64_t(j) * ld) * REAL_BYTES));
}

Int infoOf(int batch) { return Int(canon(p.info + uint64_t(batch) * 4ul)); }
Real tauOf(int batch, int k) {
    const uint64_t base = (p.flags & kPtrTau) != 0u ? Ptr(canon(p.tau + uint64_t(batch) * 8ul)).v
                                                     : p.tau + uint64_t(int64_t(batch) * p.strideP) * REAL_BYTES;
    return Real(canon(base + uint64_t(k) * REAL_BYTES));
}
Int pivotOf(int batch, int k) { return Int(canon(p.ipiv + uint64_t(int64_t(batch) * p.strideP + k) * 4ul)); }

// The SVD workspace of one batch entry (svd_*.comp; at B), for the r x c (r >= c) matrix G the
// one-sided Jacobi iteration orthogonalizes (A, or A^T if A is wide): G (r x c, ld r), the
// accumulated rotations V (c x c, ld c), then c + 1 reals for the ranks of the singular
// values, c for geqrf's tau, and c for the norms (later column signs).
uint64_t svdV(uint64_t W) { return W + uint64_t(p.m) * uint64_t(p.n) * REAL_BYTES; }
uint64_t svdRotations(uint64_t W) { return svdV(W) + uint64_t(p.n) * uint64_t(p.n) * REAL_BYTES; }
uint64_t svdTau(uint64_t W) { return svdRotations(W) + uint64_t(p.n + 1) * REAL_BYTES; }
uint64_t svdNorms(uint64_t W) { return svdTau(W) + uint64_t(p.n) * REAL_BYTES; }
