// The caller's arrays (fft_gather.comp, fft_scatter.comp): f32, f64 or f16 elements, real or
// complex, at element offsets.

#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require

layout(buffer_reference, std430, buffer_reference_align = 4) buffer F32 { float v; };
layout(buffer_reference, std430, buffer_reference_align = 8) buffer F32x2 { vec2 v; };
layout(buffer_reference, std430, buffer_reference_align = 2) buffer U16 { uint16_t v; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer U32 { uint v; };
#ifdef DOUBLE
layout(buffer_reference, std430, buffer_reference_align = 8) buffer F64 { double v; };
layout(buffer_reference, std430, buffer_reference_align = 8) buffer F64x2 { dvec2 v; };
#endif

cplx loadComplex(uint64_t base, int64_t i) {
#ifdef DOUBLE
    return F64x2(canon(base + uint64_t(i) * 16ul)).v;
#else
    if (elementType() == kF16) return unpackHalf2x16(U32(canon(base + uint64_t(i) * 4ul)).v);
    return F32x2(canon(base + uint64_t(i) * 8ul)).v;
#endif
}

real loadReal(uint64_t base, int64_t i) {
#ifdef DOUBLE
    return F64(canon(base + uint64_t(i) * 8ul)).v;
#else
    if (elementType() == kF16) return unpackHalf2x16(uint(U16(canon(base + uint64_t(i) * 2ul)).v)).x;
    return F32(canon(base + uint64_t(i) * 4ul)).v;
#endif
}

void storeComplex(uint64_t base, int64_t i, cplx v) {
#ifdef DOUBLE
    F64x2(canon(base + uint64_t(i) * 16ul)).v = v;
#else
    if (elementType() == kF16) U32(canon(base + uint64_t(i) * 4ul)).v = packHalf2x16(v);
    else F32x2(canon(base + uint64_t(i) * 8ul)).v = v;
#endif
}

void storeReal(uint64_t base, int64_t i, real v) {
#ifdef DOUBLE
    F64(canon(base + uint64_t(i) * 8ul)).v = v;
#else
    if (elementType() == kF16) U16(canon(base + uint64_t(i) * 2ul)).v = uint16_t(packHalf2x16(vec2(v, 0)) & 0xFFFFu);
    else F32(canon(base + uint64_t(i) * 4ul)).v = v;
#endif
}

// The caller's offset of element (b, i0, i1, i2).
int64_t offsetOf(int64_t b, int i0, int i1, int i2) {
    return b * p.dist + i0 * p.stride[0] + i1 * p.stride[1] + i2 * p.stride[2];
}
