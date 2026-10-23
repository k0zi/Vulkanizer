// Vulkanizer's texture and surface functions: what kernels' tex, tld4, txq, suld, sust and suq
// instructions become (src/ptx/texture.cpp rewrites them into calls of these). Textures are sampled
// in software from the descriptor a texture or surface object handle points to (VkTexDesc):
// coordinates are scaled and addressed (wrap, clamp, mirror, border), texels read from memory and
// converted from their format, and filtered bilinearly (trilinearly in 3D) with 8-bit fractional
// weights, as CUDA hardware does. Every call site inlines these, so the corners of the filter and
// the channels of a texel are loops (not unrolled): one copy of the texel code each.
//
// Compiled to PTX (texture_lib.ptx, which the build embeds) with
//   .cuda/bin/nvcc -ptx -rdc=true -arch=compute_61 -O3 -I src/common src/ptx/texture_lib.cu -o src/ptx/texture_lib.ptx
// (tools/gen_texture_lib.sh); -rdc keeps the functions although nothing here calls them.

#include <cuda_fp16.h>

#include "texture_desc.h"

namespace {

__device__ __forceinline__ const VkTexDesc* desc(unsigned long long h) { return reinterpret_cast<const VkTexDesc*>(h); }

// The address of texel (x, y, z) in `layer`, all in range.
__device__ __forceinline__ const char* texel(const VkTexDesc* d, int x, int y, int z, int layer) {
    return reinterpret_cast<const char*>(d->data) + size_t(layer) * d->slice + size_t(z) * d->slice +
           size_t(y) * d->pitch + size_t(x) * d->texelBytes;
}

__device__ __forceinline__ unsigned channelBytes(unsigned format) {
    switch (format) {
    case VK_FORMAT_U8: case VK_FORMAT_S8: return 1;
    case VK_FORMAT_U16: case VK_FORMAT_S16: case VK_FORMAT_HALF: return 2;
    default: return 4;
    }
}

// Channel c of a texel: its raw integer (sign-extended), and as a float (promoted to [0, 1] or
// [-1, 1] for 8- and 16-bit integers unless read as integer).
__device__ __forceinline__ int rawChannel(const VkTexDesc* d, const char* p, unsigned c) {
    if (c >= d->channels) return 0;
    const char* q = p + c * channelBytes(d->format);
    switch (d->format) {
    case VK_FORMAT_U8: return *reinterpret_cast<const unsigned char*>(q);
    case VK_FORMAT_S8: return *reinterpret_cast<const signed char*>(q);
    case VK_FORMAT_U16: return *reinterpret_cast<const unsigned short*>(q);
    case VK_FORMAT_S16: case VK_FORMAT_HALF: return *reinterpret_cast<const short*>(q);
    default: return *reinterpret_cast<const int*>(q);
    }
}

__device__ __forceinline__ float srgbToLinear(float v) {
    return v <= 0.04045f ? v / 12.92f : __powf((v + 0.055f) / 1.055f, 2.4f);
}

__device__ __forceinline__ float floatChannel(const VkTexDesc* d, const char* p, unsigned c) {
    if (c >= d->channels) return 0.0f;
    const int raw = rawChannel(d, p, c);
    const bool promote = !(d->flags & VK_TEX_READ_AS_INTEGER);
    float v;
    switch (d->format) {
    case VK_FORMAT_FLOAT: v = __int_as_float(raw); break;
    case VK_FORMAT_HALF: v = __half2float(__ushort_as_half((unsigned short)raw)); break;
    case VK_FORMAT_U8: v = promote ? float(raw) / 255.0f : float(raw); break;
    case VK_FORMAT_U16: v = promote ? float(raw) / 65535.0f : float(raw); break;
    case VK_FORMAT_S8: v = promote ? fmaxf(float(raw) / 127.0f, -1.0f) : float(raw); break;
    case VK_FORMAT_S16: v = promote ? fmaxf(float(raw) / 32767.0f, -1.0f) : float(raw); break;
    case VK_FORMAT_U32: v = float(unsigned(raw)); break;
    default: v = float(raw); break;
    }
    if ((d->flags & VK_TEX_SRGB) && c < 3) v = srgbToLinear(v);
    return v;
}

// An integer texel coordinate addressed into [0, size), or -1 outside (border).
__device__ __forceinline__ int address(int i, int size, unsigned mode) {
    if (i >= 0 && i < size) return i;
    switch (mode) {
    case VK_ADDRESS_WRAP: {
        const int m = i % size;
        return m < 0 ? m + size : m;
    }
    case VK_ADDRESS_MIRROR: {
        const int period = 2 * size;
        int m = i % period;
        if (m < 0) m += period;
        return m < size ? m : period - 1 - m;
    }
    case VK_ADDRESS_BORDER: return -1;
    default: return i < 0 ? 0 : size - 1;
    }
}

// The address mode of dimension k: wrap and mirror need normalized coordinates (clamp otherwise,
// as in CUDA).
__device__ __forceinline__ unsigned modeOf(const VkTexDesc* d, int k) {
    const unsigned m = d->address[k];
    return (d->flags & VK_TEX_NORMALIZED_COORDS) || m == VK_ADDRESS_BORDER ? m : VK_ADDRESS_CLAMP;
}

// The texel at integer coordinates, addressed; the border color outside.
__device__ float4 fetchFloat(const VkTexDesc* d, int x, int y, int z, int layer) {
    const int ax = address(x, d->width, modeOf(d, 0));
    const int ay = d->dimensions >= 2 ? address(y, d->height, modeOf(d, 1)) : 0;
    const int az = d->dimensions >= 3 ? address(z, d->depth, modeOf(d, 2)) : 0;
    if (ax < 0 || ay < 0 || az < 0) return make_float4(d->border[0], d->border[1], d->border[2], d->border[3]);
    const int l = layer < 0 ? 0 : (layer >= int(d->layers) ? int(d->layers) - 1 : layer);
    const char* p = texel(d, ax, ay, az, l);
    float v[4];
#pragma unroll 1
    for (unsigned c = 0; c < 4; ++c) v[c] = floatChannel(d, p, c);
    return make_float4(v[0], v[1], v[2], v[3]);
}

__device__ int4 fetchInt(const VkTexDesc* d, int x, int y, int z, int layer) {
    const int ax = address(x, d->width, modeOf(d, 0));
    const int ay = d->dimensions >= 2 ? address(y, d->height, modeOf(d, 1)) : 0;
    const int az = d->dimensions >= 3 ? address(z, d->depth, modeOf(d, 2)) : 0;
    if (ax < 0 || ay < 0 || az < 0)
        return make_int4(__float_as_int(d->border[0]), __float_as_int(d->border[1]), __float_as_int(d->border[2]),
                         __float_as_int(d->border[3]));
    const int l = layer < 0 ? 0 : (layer >= int(d->layers) ? int(d->layers) - 1 : layer);
    const char* p = texel(d, ax, ay, az, l);
    int v[4];
#pragma unroll 1
    for (unsigned c = 0; c < 4; ++c) v[c] = rawChannel(d, p, c);
    return make_int4(v[0], v[1], v[2], v[3]);
}

// A float coordinate in texel units (normalized coordinates scaled, wrap and mirror applied to
// them as CUDA does: on the normalized value).
__device__ __forceinline__ float texelCoordinate(const VkTexDesc* d, float u, unsigned size, unsigned mode) {
    if (!(d->flags & VK_TEX_NORMALIZED_COORDS)) return u;
    if (mode == VK_ADDRESS_WRAP) u -= floorf(u);
    else if (mode == VK_ADDRESS_MIRROR) {
        const float t = u - 2.0f * floorf(u * 0.5f);  // [0, 2)
        u = t <= 1.0f ? t : 2.0f - t;
    }
    return u * float(size);
}

// The 8-bit fixed-point fraction of the filter weights.
__device__ __forceinline__ float weight(float f) { return floorf(f * 256.0f + 0.5f) * (1.0f / 256.0f); }

__device__ float4 sample(const VkTexDesc* d, float u, float v, float w, int layer) {
    const float x = texelCoordinate(d, u, d->width, d->address[0]);
    const float y = d->dimensions >= 2 ? texelCoordinate(d, v, d->height, d->address[1]) : 0.0f;
    const float z = d->dimensions >= 3 ? texelCoordinate(d, w, d->depth, d->address[2]) : 0.0f;
    if (!(d->flags & VK_TEX_LINEAR_FILTER))
        return fetchFloat(d, int(floorf(x)), int(floorf(y)), int(floorf(z)), layer);
    const float fx = x - 0.5f, fy = y - 0.5f, fz = z - 0.5f;
    const int x0 = int(floorf(fx)), y0 = int(floorf(fy)), z0 = int(floorf(fz));
    const float a = weight(fx - floorf(fx)), b = weight(fy - floorf(fy)), c = weight(fz - floorf(fz));
    // the 2, 4 or 8 corners, weighted
    float4 r = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    const int corners = 1 << d->dimensions;
#pragma unroll 1
    for (int k = 0; k < corners; ++k) {
        const int i = k & 1, j = (k >> 1) & 1, l = (k >> 2) & 1;
        const float wt = (i ? a : 1.0f - a) * (d->dimensions >= 2 ? (j ? b : 1.0f - b) : 1.0f) *
                         (d->dimensions >= 3 ? (l ? c : 1.0f - c) : 1.0f);
        const float4 t = fetchFloat(d, x0 + i, y0 + j, z0 + l, layer);
        r.x += wt * t.x, r.y += wt * t.y, r.z += wt * t.z, r.w += wt * t.w;
    }
    return r;
}

// Surfaces: byte x coordinates; out of range a load gives 0 and a store does nothing (trap and
// zero mode), or the coordinate is clamped (clamp mode, 1).
__device__ __forceinline__ char* surfaceAddress(unsigned long long h, int bytes, int xb, int y, int z, int layer,
                                                int clampMode) {
    const VkTexDesc* d = desc(h);
    const int rowBytes = int(d->width * d->texelBytes);
    const int height = d->dimensions >= 2 ? int(d->height) : 1, depth = d->dimensions >= 3 ? int(d->depth) : 1;
    if (clampMode == 1) {
        xb = min(max(xb, 0), rowBytes - bytes);
        y = min(max(y, 0), height - 1);
        z = min(max(z, 0), depth - 1);
        layer = min(max(layer, 0), int(d->layers) - 1);
    } else if (xb < 0 || xb + bytes > rowBytes || y < 0 || y >= height || z < 0 || z >= depth || layer < 0 ||
               layer >= int(d->layers)) {
        return nullptr;
    }
    return reinterpret_cast<char*>(d->data) + size_t(layer) * d->slice + size_t(z) * d->slice + size_t(y) * d->pitch +
           size_t(xb);
}

}  // namespace

extern "C" {

// tex with float coordinates, float results (filtered as the texture says)
__device__ __noinline__ float4 __vulkanizer_tex_f(unsigned long long h, float x, float y, float z, int layer) {
    return sample(desc(h), x, y, z, layer);
}

// tex with float coordinates, integer results (point sampled, raw texels)
__device__ __noinline__ int4 __vulkanizer_tex_i(unsigned long long h, float x, float y, float z, int layer) {
    const VkTexDesc* d = desc(h);
    const float tx = texelCoordinate(d, x, d->width, d->address[0]);
    const float ty = d->dimensions >= 2 ? texelCoordinate(d, y, d->height, d->address[1]) : 0.0f;
    const float tz = d->dimensions >= 3 ? texelCoordinate(d, z, d->depth, d->address[2]) : 0.0f;
    return fetchInt(d, int(floorf(tx)), int(floorf(ty)), int(floorf(tz)), layer);
}

// tex with integer coordinates (tex1Dfetch on linear memory and texel fetches): no filtering, no
// normalization; out of range texels are 0
__device__ __noinline__ float4 __vulkanizer_fetch_f(unsigned long long h, int x, int y, int z, int layer) {
    const VkTexDesc* d = desc(h);
    if (x < 0 || x >= int(d->width) || (d->dimensions >= 2 && (y < 0 || y >= int(d->height))) ||
        (d->dimensions >= 3 && (z < 0 || z >= int(d->depth))))
        return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    return fetchFloat(d, x, y, z, layer);
}

__device__ __noinline__ int4 __vulkanizer_fetch_i(unsigned long long h, int x, int y, int z, int layer) {
    const VkTexDesc* d = desc(h);
    if (x < 0 || x >= int(d->width) || (d->dimensions >= 2 && (y < 0 || y >= int(d->height))) ||
        (d->dimensions >= 3 && (z < 0 || z >= int(d->depth))))
        return make_int4(0, 0, 0, 0);
    return fetchInt(d, x, y, z, layer);
}

// tld4: component `comp` of the four texels bilinear filtering at (x, y) would use, in CUDA's order
// (x: (0, 1), y: (1, 1), z: (1, 0), w: (0, 0) of the 2 x 2 footprint)
__device__ __noinline__ float4 __vulkanizer_tld4_f(unsigned long long h, int comp, float x, float y) {
    const VkTexDesc* d = desc(h);
    const float fx = texelCoordinate(d, x, d->width, d->address[0]) - 0.5f;
    const float fy = texelCoordinate(d, y, d->height, d->address[1]) - 0.5f;
    const int x0 = int(floorf(fx)), y0 = int(floorf(fy));
    float v[4];
#pragma unroll 1
    for (int k = 0; k < 4; ++k) {
        const float4 t = fetchFloat(d, x0 + (k == 1 || k == 2), y0 + (k < 2), 0, 0);
        v[k] = comp == 0 ? t.x : comp == 1 ? t.y : comp == 2 ? t.z : t.w;
    }
    return make_float4(v[0], v[1], v[2], v[3]);
}

__device__ __noinline__ int4 __vulkanizer_tld4_i(unsigned long long h, int comp, float x, float y) {
    const VkTexDesc* d = desc(h);
    const float fx = texelCoordinate(d, x, d->width, d->address[0]) - 0.5f;
    const float fy = texelCoordinate(d, y, d->height, d->address[1]) - 0.5f;
    const int x0 = int(floorf(fx)), y0 = int(floorf(fy));
    int v[4];
#pragma unroll 1
    for (int k = 0; k < 4; ++k) {
        const int4 t = fetchInt(d, x0 + (k == 1 || k == 2), y0 + (k < 2), 0, 0);
        v[k] = comp == 0 ? t.x : comp == 1 ? t.y : comp == 2 ? t.z : t.w;
    }
    return make_int4(v[0], v[1], v[2], v[3]);
}

// txq and suq: 0 width, 1 height, 2 depth, 3 channel data type (CUarray_format), 4 channel order
// (channel count), 5 normalized coordinates, 6 filter mode, 7..9 address modes, 10 array size
__device__ __noinline__ int __vulkanizer_query(unsigned long long h, int query) {
    const VkTexDesc* d = desc(h);
    switch (query) {
    case 0: return int(d->width);
    case 1: return int(d->height);
    case 2: return int(d->depth);
    case 3: return int(d->format);
    case 4: return int(d->channels);
    case 5: return (d->flags & VK_TEX_NORMALIZED_COORDS) ? 1 : 0;
    case 6: return (d->flags & VK_TEX_LINEAR_FILTER) ? 1 : 0;
    case 7: return int(d->address[0]);
    case 8: return int(d->address[1]);
    case 9: return int(d->address[2]);
    case 10: return int(d->layers);
    default: return 0;
    }
}

// suld.b: `bytes` (1, 2, 4 or 8) at byte coordinate xb
__device__ __noinline__ unsigned long long __vulkanizer_suld(unsigned long long h, int bytes, int xb, int y, int z,
                                                             int layer, int clampMode) {
    const char* p = surfaceAddress(h, bytes, xb, y, z, layer, clampMode);
    if (!p) return 0;
    switch (bytes) {
    case 1: return *reinterpret_cast<const unsigned char*>(p);
    case 2: return *reinterpret_cast<const unsigned short*>(p);
    case 4: return *reinterpret_cast<const unsigned*>(p);
    default: return *reinterpret_cast<const unsigned long long*>(p);
    }
}

// sust.b
__device__ __noinline__ void __vulkanizer_sust(unsigned long long h, int bytes, int xb, int y, int z, int layer,
                                               int clampMode, unsigned long long value) {
    char* p = surfaceAddress(h, bytes, xb, y, z, layer, clampMode);
    if (!p) return;
    switch (bytes) {
    case 1: *reinterpret_cast<unsigned char*>(p) = (unsigned char)value; break;
    case 2: *reinterpret_cast<unsigned short*>(p) = (unsigned short)value; break;
    case 4: *reinterpret_cast<unsigned*>(p) = (unsigned)value; break;
    default: *reinterpret_cast<unsigned long long*>(p) = value; break;
    }
}

}  // extern "C"
