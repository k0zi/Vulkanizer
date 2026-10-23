// Shared by the FFT kernels: the complex element type (f32, or f64 with -DDOUBLE), the work item
// numbering and the push constants (cufft.cpp's Params).
//
// The transform works on the "full array": batch x n0 x n1 x n2 complex elements, contiguous,
// n2 fastest (ranks below 3 have leading 1s), in the plan's workspace. fft_gather.comp fills it
// from the caller's input, fft_pass.comp (and fft_bluestein.comp) transform it along one axis at
// a time, fft_scatter.comp writes the result to the caller's output.

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#ifdef DOUBLE
#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require
#define real double
#define cplx dvec2
#define CPLX_BYTES 16ul
layout(buffer_reference, std430, buffer_reference_align = 8) buffer Cplx { dvec2 v; };
layout(buffer_reference, std430, buffer_reference_align = 8) buffer Real { double v; };
#else
#define real float
#define cplx vec2
#define CPLX_BYTES 8ul
layout(buffer_reference, std430, buffer_reference_align = 8) buffer Cplx { vec2 v; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer Real { float v; };
#endif

// The device's address bits 63..47 (vk::Device::hostAddressBits), specialization constant 5:
// addresses arrive without them.
layout(constant_id = 5) const uint kAddressHigh = 0xFFFF8000u;
uint64_t canon(uint64_t a) {
    uvec2 w = unpackUint2x32(a);
    w.y |= kAddressHigh;
    return packUint2x32(w);
}

layout(push_constant, std430) uniform Params {
    uint64_t src, dst;
    uint64_t table;      // pass: the roots of unity exp(-2 pi i m / len); Bluestein: the chirp
    uint64_t aux;        // Bluestein: the transformed chirp filter, forward then inverse (len each)
    int64_t dist;        // gather/scatter: the caller's elements between batch entries
    int64_t stride[3];   // gather/scatter: the caller's element strides of the three dimensions
    int64_t items;       // work items of this dispatch
    int n[3];            // the full array's dimensions (without the batch)
    int len;             // pass: transform length; Bluestein: its padded length M
    int radix, span;     // pass: this pass's radix and the length done so far; Bluestein: span = N
    int inner;           // pass, Bluestein: element stride of the transformed axis
    uint flags;
} p;

// flags: the caller's element type (gather/scatter), the transform type, the direction and the
// Bluestein step
const uint kTypeMask = 3u, kF32 = 0u, kF64 = 1u, kF16 = 2u;
const uint kModeShift = 2u, kC2C = 0u, kR2C = 1u, kC2R = 2u;
const uint kInverse = 16u;
const uint kStepShift = 5u, kPre = 0u, kMultiply = 1u, kPost = 2u;

uint elementType() { return p.flags & kTypeMask; }
uint mode() { return (p.flags >> kModeShift) & 3u; }
bool inverse() { return (p.flags & kInverse) != 0u; }

layout(local_size_x = 256) in;

// Work items are numbered across a 2D grid of 256-thread workgroups. Index arithmetic is 32-bit
// (the full array has fewer than 2^31 elements): 64-bit division is slow on GPUs.
int itemIndex() {
    return int((gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x) * 256u + gl_LocalInvocationID.x);
}
int groupIndex() { return int(gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x); }

Cplx at(uint64_t base, int i) { return Cplx(canon(base + uint64_t(uint(i)) * CPLX_BYTES)); }
Cplx at64(uint64_t base, int64_t i) { return Cplx(canon(base + uint64_t(i) * CPLX_BYTES)); }
cplx cmul(cplx a, cplx b) { return cplx(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
cplx conjugate(cplx a) { return cplx(a.x, -a.y); }
