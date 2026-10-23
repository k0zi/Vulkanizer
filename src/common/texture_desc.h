/* What a texture or surface object handle points to: a descriptor in device memory, read by the
   texture functions kernels' tex/suld/sust instructions become (src/ptx/texture_lib.cu). Written by
   the driver (api_texture.cpp); C, as the CUDA C texture library includes it too. */
#pragma once

#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif

enum {
    VK_TEX_NORMALIZED_COORDS = 1, /* coordinates in [0, 1) */
    VK_TEX_LINEAR_FILTER = 2,     /* bilinear (trilinear) filtering, else nearest */
    VK_TEX_READ_AS_INTEGER = 4,   /* 8/16-bit integers are not promoted to [0, 1] / [-1, 1] floats */
    VK_TEX_SRGB = 8,              /* sRGB to linear conversion of the color channels */
    VK_TEX_CUBE = 16,             /* a cube map: layers are faces (+x, -x, +y, -y, +z, -z), 6 per cube */
    VK_TEX_MIP_LINEAR = 32,       /* linear filtering between mipmap levels, else the nearest level */
};

enum { VK_ADDRESS_WRAP = 0, VK_ADDRESS_CLAMP = 1, VK_ADDRESS_MIRROR = 2, VK_ADDRESS_BORDER = 3 };

/* CUarray_format values */
enum {
    VK_FORMAT_U8 = 0x01,
    VK_FORMAT_U16 = 0x02,
    VK_FORMAT_U32 = 0x03,
    VK_FORMAT_S8 = 0x08,
    VK_FORMAT_S16 = 0x09,
    VK_FORMAT_S32 = 0x0a,
    VK_FORMAT_HALF = 0x10,
    VK_FORMAT_FLOAT = 0x20,
};

typedef struct VkTexDesc {
    uint64_t data;    /* texel (0, 0, 0, layer 0) */
    uint64_t pitch;   /* bytes between rows */
    uint64_t slice;   /* bytes between depth slices or layers */
    uint32_t width, height, depth, layers;  /* in texels; 1 for unused dimensions */
    uint32_t format;  /* CUarray_format of each channel */
    uint32_t channels;
    uint32_t texelBytes;
    uint32_t flags;   /* VK_TEX_* */
    uint32_t address[3];
    uint32_t dimensions; /* 1, 2 or 3 (layers are separate) */
    float border[4];
    /* Mipmaps: the descriptor of level l is l descriptors after level 0's; level 0's says how many
       there are (1: no mipmaps) and how the level is chosen: lod + bias, clamped to [minLod, maxLod]. */
    uint32_t levels;
    float lodBias, minLod, maxLod;
} VkTexDesc;
