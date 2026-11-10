// Shared by the sparse kernels: the value type (f32, or f64 with -DDOUBLE), index arrays of 32 or
// 64 bits, dense matrices in either order, work item numbering and the push constants
// (cusparse.cpp's Params). Index arithmetic is 32-bit (sizes below 2^31).

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#ifdef DOUBLE
#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require
#define real double
#define REAL_BYTES 8ul
layout(buffer_reference, std430, buffer_reference_align = 8) buffer Real { double v; };
layout(buffer_reference, std430, buffer_reference_align = 8) coherent buffer SharedReal { double v; };
#else
#define real float
#define REAL_BYTES 4ul
layout(buffer_reference, std430, buffer_reference_align = 4) buffer Real { float v; };
layout(buffer_reference, std430, buffer_reference_align = 4) coherent buffer SharedReal { float v; };
#endif
layout(buffer_reference, std430, buffer_reference_align = 4) buffer Int { int v; };
layout(buffer_reference, std430, buffer_reference_align = 8) buffer Int64 { int64_t v; };

// The device's address bits 63..47 (vk::Device::hostAddressBits), specialization constant 5:
// addresses arrive without them.
layout(constant_id = 5) const uint kAddressHigh = 0xFFFF8000u;
uint64_t canon(uint64_t a) {
    uvec2 w = unpackUint2x32(a);
    w.y |= kAddressHigh;
    return packUint2x32(w);
}

layout(push_constant, std430) uniform Params {
    uint64_t ptr[8];     // the arrays, meaning per kernel
    uint64_t alphaBits;  // scalars: f32 bits in the low word, or f64 bits (DOUBLE)
    uint64_t betaBits;
    int stride[4];       // batch strides (elements), meaning per kernel
    int items;           // work items (or workgroups) of this dispatch
    int m, n, k;
    int ld0, ld1;
    int extra;
    uint flags;
} p;

// flags shared by the kernels
const uint kIndex64A = 1u, kIndex64B = 2u;      // the two index arrays (offsets/rows, columns) are 64-bit
const uint kRowMajor0 = 4u, kRowMajor1 = 8u;    // the dense operands' order
const uint kTrans0 = 16u, kTrans1 = 32u;        // transposed operands
const uint kBetaZero = 64u;
const uint kBaseShift = 8u;                     // index bases, 2 bits from here: A (bit 8), B (9), C (10)
const uint kModeShift = 16u;                    // kernel mode, 4 bits

real scalarOf(uint64_t bits) {
#ifdef DOUBLE
    return uint64BitsToDouble(bits);
#else
    return uintBitsToFloat(uint(bits));
#endif
}
real alpha() { return scalarOf(p.alphaBits); }
real beta() { return scalarOf(p.betaBits); }
uint mode() { return (p.flags >> kModeShift) & 15u; }
int baseOf(uint which) { return int((p.flags >> (kBaseShift + which)) & 1u); }

layout(local_size_x = 256) in;

int itemIndex() { return int((gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x) * 256u + gl_LocalInvocationID.x); }
int groupIndex() { return int(gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x); }

Real realAt(uint64_t base, int i) { return Real(canon(base + uint64_t(uint(i)) * REAL_BYTES)); }
// Coherent access, for values other invocations of the workgroup write (spsv.comp).
SharedReal sharedRealAt(uint64_t base, int i) { return SharedReal(canon(base + uint64_t(uint(i)) * REAL_BYTES)); }
Int intAt(uint64_t base, int i) { return Int(canon(base + uint64_t(uint(i)) * 4ul)); }
// An index array element, 32 or 64 bits.
int indexAt(uint64_t base, int i, bool wide) {
    return wide ? int(Int64(canon(base + uint64_t(uint(i)) * 8ul)).v) : Int(canon(base + uint64_t(uint(i)) * 4ul)).v;
}
// Element (i, j) of a dense matrix: column-major (i + j ld) or row-major (i ld + j).
int denseIndex(int i, int j, int ld, bool rowMajor) { return rowMajor ? i * ld + j : i + j * ld; }
