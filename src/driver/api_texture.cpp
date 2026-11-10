// CUDA arrays, 2D/3D copies, texture and surface objects. Textures are sampled in software
// (ptx/texture.h): an array is plain device memory, rows of tightly packed texels, and a texture or
// surface object handle is the device address of a descriptor (common/texture_desc.h) the kernels'
// texture functions read. A cube map is a layered array of 6 faces per cube (VK_TEX_CUBE); a
// mipmapped array is one array per level, and its texture object one descriptor per level, side by
// side.

#include "driver/driver.h"

#include "common/log.h"
#include "common/texture_desc.h"

#include <cstring>
#include <map>
#include <vector>

using namespace vulkanizer;
using namespace vulkanizer::driver;

struct CUarray_st {
    static constexpr uint32_t kMagic = 0x52524153;  // 'SARR'
    uint32_t magic = kMagic;
    CUDA_ARRAY3D_DESCRIPTOR desc{};
    CUdeviceptr data = 0;
    size_t texelBytes = 0;
    size_t pitch = 0;  // bytes per row
    size_t slice = 0;  // bytes per depth slice or layer
    bool ownedByMipmap = false;  // a level of a mipmapped array: destroyed with it
};

struct CUmipmappedArray_st {
    static constexpr uint32_t kMagic = 0x504d4d53;  // 'SMMP'
    uint32_t magic = kMagic;
    CUDA_ARRAY3D_DESCRIPTOR desc{};  // of level 0
    std::vector<CUarray> levels;
};

namespace {

size_t channelBytes(CUarray_format f) {
    switch (f) {
    case CU_AD_FORMAT_UNSIGNED_INT8: case CU_AD_FORMAT_SIGNED_INT8: return 1;
    case CU_AD_FORMAT_UNSIGNED_INT16: case CU_AD_FORMAT_SIGNED_INT16: case CU_AD_FORMAT_HALF: return 2;
    case CU_AD_FORMAT_UNSIGNED_INT32: case CU_AD_FORMAT_SIGNED_INT32: case CU_AD_FORMAT_FLOAT: return 4;
    default: return 0;
    }
}

bool validChannels(unsigned n) { return n == 1 || n == 2 || n == 4; }

// The texture and surface objects: handle (device address of the descriptor) -> what made it.
struct TextureObject {
    CUDA_RESOURCE_DESC res{};
    CUDA_TEXTURE_DESC tex{};
    bool surface = false;
};
std::map<unsigned long long, TextureObject>& objects() {
    static std::map<unsigned long long, TextureObject> m;
    return m;
}

// One side of a 2D/3D copy: its base address (host or device, as cuMemcpy takes it), pitch and
// slice height.
struct Side {
    uintptr_t base = 0;
    size_t pitch = 0, height = 0;
};

CUresult sideOf(CUmemorytype type, const void* host, CUdeviceptr device, CUarray array, size_t pitch, size_t height,
                Side& out) {
    switch (type) {
    case CU_MEMORYTYPE_HOST: out = {reinterpret_cast<uintptr_t>(host), pitch, height}; return CUDA_SUCCESS;
    case CU_MEMORYTYPE_DEVICE:
    case CU_MEMORYTYPE_UNIFIED: out = {uintptr_t(device), pitch, height}; return CUDA_SUCCESS;
    case CU_MEMORYTYPE_ARRAY:
        if (!validHandle(array)) return CUDA_ERROR_INVALID_HANDLE;
        out = {uintptr_t(array->data), array->pitch, array->slice / std::max<size_t>(array->pitch, 1)};
        return CUDA_SUCCESS;
    default: return CUDA_ERROR_INVALID_VALUE;
    }
}

// Row by row (one copy when both sides are contiguous), on `stream` if async.
CUresult copy3D(const Side& src, size_t srcX, size_t srcY, size_t srcZ, const Side& dst, size_t dstX, size_t dstY,
                size_t dstZ, size_t width, size_t height, size_t depth, bool async, CUstream stream) {
    if (!width || !height || !depth) return CUDA_SUCCESS;
    const size_t sp = src.pitch ? src.pitch : width, dp = dst.pitch ? dst.pitch : width;
    const size_t sh = src.height ? src.height : height, dh = dst.height ? dst.height : height;
    if (width > sp || width > dp) return CUDA_ERROR_INVALID_VALUE;
    auto one = [&](uintptr_t d, uintptr_t s, size_t n) {
        return async ? cuMemcpyAsync(CUdeviceptr(d), CUdeviceptr(s), n, stream) : cuMemcpy(CUdeviceptr(d), CUdeviceptr(s), n);
    };
    const bool contiguous = sp == width && dp == width && srcX == 0 && dstX == 0 && sh == height && dh == height;
    if (contiguous) {
        return one(dst.base + (dstZ * dh + dstY) * dp, src.base + (srcZ * sh + srcY) * sp, width * height * depth);
    }
    for (size_t z = 0; z < depth; ++z)
        for (size_t y = 0; y < height; ++y)
            if (CUresult r = one(dst.base + ((dstZ + z) * dh + dstY + y) * dp + dstX,
                                 src.base + ((srcZ + z) * sh + srcY + y) * sp + srcX, width))
                return r;
    return CUDA_SUCCESS;
}

CUresult copy2D(const CUDA_MEMCPY2D* p, bool async, CUstream stream) {
    if (!p) return CUDA_ERROR_INVALID_VALUE;
    Side s, d;
    if (CUresult r = sideOf(p->srcMemoryType, p->srcHost, p->srcDevice, p->srcArray, p->srcPitch, 0, s)) return r;
    if (CUresult r = sideOf(p->dstMemoryType, p->dstHost, p->dstDevice, p->dstArray, p->dstPitch, 0, d)) return r;
    s.height = d.height = 0;
    return copy3D(s, p->srcXInBytes, p->srcY, 0, d, p->dstXInBytes, p->dstY, 0, p->WidthInBytes, p->Height, 1, async,
                  stream);
}

CUresult copy3DParams(const CUDA_MEMCPY3D* p, bool async, CUstream stream) {
    if (!p) return CUDA_ERROR_INVALID_VALUE;
    if (p->srcLOD || p->dstLOD) return CUDA_ERROR_NOT_SUPPORTED;
    Side s, d;
    if (CUresult r = sideOf(p->srcMemoryType, p->srcHost, p->srcDevice, p->srcArray, p->srcPitch, p->srcHeight, s))
        return r;
    if (CUresult r = sideOf(p->dstMemoryType, p->dstHost, p->dstDevice, p->dstArray, p->dstPitch, p->dstHeight, d))
        return r;
    return copy3D(s, p->srcXInBytes, p->srcY, p->srcZ, d, p->dstXInBytes, p->dstY, p->dstZ, p->WidthInBytes, p->Height,
                  p->Depth, async, stream);
}

CUresult arrayRange(CUarray a, size_t offset, size_t bytes, CUdeviceptr& out) {
    if (!validHandle(a)) return CUDA_ERROR_INVALID_HANDLE;
    if (offset + bytes > a->slice * std::max<size_t>(a->desc.Depth, 1)) return CUDA_ERROR_INVALID_VALUE;
    out = a->data + offset;
    return CUDA_SUCCESS;
}

// The descriptor of a resource; for textures with the sampling state.
CUresult describe(const CUDA_RESOURCE_DESC& res, const CUDA_TEXTURE_DESC* tex, VkTexDesc& d) {
    d = VkTexDesc{};
    d.layers = 1;
    switch (res.resType) {
    case CU_RESOURCE_TYPE_ARRAY: {
        CUarray a = res.res.array.hArray;
        if (!validHandle(a)) return CUDA_ERROR_INVALID_HANDLE;
        const bool cube = a->desc.Flags & CUDA_ARRAY3D_CUBEMAP;
        const bool layered = cube || (a->desc.Flags & CUDA_ARRAY3D_LAYERED);
        if (cube) d.flags |= VK_TEX_CUBE;
        d.data = a->data;
        d.pitch = a->pitch;
        d.slice = a->slice;
        d.width = uint32_t(a->desc.Width);
        d.height = uint32_t(std::max<size_t>(a->desc.Height, 1));
        d.depth = layered ? 1 : uint32_t(std::max<size_t>(a->desc.Depth, 1));
        d.layers = layered ? uint32_t(std::max<size_t>(a->desc.Depth, 1)) : 1;
        d.dimensions = a->desc.Height == 0 ? 1 : (a->desc.Depth == 0 || layered) ? 2 : 3;
        d.format = uint32_t(a->desc.Format);
        d.channels = a->desc.NumChannels;
        d.texelBytes = uint32_t(a->texelBytes);
        break;
    }
    case CU_RESOURCE_TYPE_LINEAR: {
        const size_t texel = channelBytes(res.res.linear.format) * res.res.linear.numChannels;
        if (!texel || !validChannels(res.res.linear.numChannels) || !res.res.linear.devPtr) return CUDA_ERROR_INVALID_VALUE;
        d.data = res.res.linear.devPtr;
        d.width = uint32_t(res.res.linear.sizeInBytes / texel);
        d.height = d.depth = 1;
        d.pitch = d.slice = res.res.linear.sizeInBytes;
        d.dimensions = 1;
        d.format = uint32_t(res.res.linear.format);
        d.channels = res.res.linear.numChannels;
        d.texelBytes = uint32_t(texel);
        break;
    }
    case CU_RESOURCE_TYPE_PITCH2D: {
        const auto& p = res.res.pitch2D;
        const size_t texel = channelBytes(p.format) * p.numChannels;
        if (!texel || !validChannels(p.numChannels) || !p.devPtr || p.pitchInBytes < p.width * texel)
            return CUDA_ERROR_INVALID_VALUE;
        d.data = p.devPtr;
        d.width = uint32_t(p.width);
        d.height = uint32_t(p.height);
        d.depth = 1;
        d.pitch = p.pitchInBytes;
        d.slice = p.pitchInBytes * p.height;
        d.dimensions = 2;
        d.format = uint32_t(p.format);
        d.channels = p.numChannels;
        d.texelBytes = uint32_t(texel);
        break;
    }
    default: return CUDA_ERROR_NOT_SUPPORTED;  // mipmapped arrays: describeAll
    }
    if (tex) {
        for (int i = 0; i < 3; ++i) d.address[i] = uint32_t(tex->addressMode[i]);
        if (tex->filterMode == CU_TR_FILTER_MODE_LINEAR) d.flags |= VK_TEX_LINEAR_FILTER;
        if (tex->flags & CU_TRSF_NORMALIZED_COORDINATES) d.flags |= VK_TEX_NORMALIZED_COORDS;
        if (tex->flags & CU_TRSF_READ_AS_INTEGER) d.flags |= VK_TEX_READ_AS_INTEGER;
        if (tex->flags & CU_TRSF_SRGB) d.flags |= VK_TEX_SRGB;
        if (tex->mipmapFilterMode == CU_TR_FILTER_MODE_LINEAR) d.flags |= VK_TEX_MIP_LINEAR;
        d.lodBias = tex->mipmapLevelBias;
        d.minLod = tex->minMipmapLevelClamp;
        d.maxLod = tex->maxMipmapLevelClamp;
        std::memcpy(d.border, tex->borderColor, sizeof d.border);
    } else {
        d.address[0] = d.address[1] = d.address[2] = VK_ADDRESS_CLAMP;
    }
    return CUDA_SUCCESS;
}

// The descriptors of a resource: one, or one per level of a mipmapped array (level 0 first, with
// the number of levels).
CUresult describeAll(const CUDA_RESOURCE_DESC& res, const CUDA_TEXTURE_DESC* tex, std::vector<VkTexDesc>& out) {
    out.clear();
    if (res.resType != CU_RESOURCE_TYPE_MIPMAPPED_ARRAY) {
        out.resize(1);
        if (CUresult r = describe(res, tex, out[0])) return r;
        out[0].levels = 1;
        if (!tex || out[0].maxLod < out[0].minLod) out[0].minLod = out[0].maxLod = 0;
        return CUDA_SUCCESS;
    }
    CUmipmappedArray m = res.res.mipmap.hMipmappedArray;
    if (!validHandle(m)) return CUDA_ERROR_INVALID_HANDLE;
    for (CUarray level : m->levels) {
        CUDA_RESOURCE_DESC one{};
        one.resType = CU_RESOURCE_TYPE_ARRAY;
        one.res.array.hArray = level;
        out.emplace_back();
        if (CUresult r = describe(one, tex, out.back())) return r;
    }
    out[0].levels = uint32_t(out.size());
    return CUDA_SUCCESS;
}

CUresult createObject(unsigned long long* handle, const CUDA_RESOURCE_DESC* res, const CUDA_TEXTURE_DESC* tex,
                      bool surface) {
    if (!handle || !res) return CUDA_ERROR_INVALID_VALUE;
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    if (surface && res->resType != CU_RESOURCE_TYPE_ARRAY) return CUDA_ERROR_INVALID_VALUE;
    std::vector<VkTexDesc> d;
    if (CUresult r = describeAll(*res, surface ? nullptr : tex, d)) return r;
    CUdeviceptr p;
    if (CUresult r = cuMemAlloc(&p, d.size() * sizeof(VkTexDesc))) return r;
    if (CUresult r = cuMemcpyHtoD(p, d.data(), d.size() * sizeof(VkTexDesc))) {
        cuMemFree(p);
        return r;
    }
    TextureObject o;
    o.res = *res;
    if (tex) o.tex = *tex;
    o.surface = surface;
    objects()[p] = o;
    *handle = p;
    return CUDA_SUCCESS;
}

CUresult destroyObject(unsigned long long handle, bool surface) {
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    auto it = objects().find(handle);
    if (it == objects().end() || it->second.surface != surface) return CUDA_ERROR_INVALID_VALUE;
    objects().erase(it);
    return cuMemFree(CUdeviceptr(handle));  // after the kernels recorded so far
}

const TextureObject* findObject(unsigned long long handle, bool surface) {
    auto it = objects().find(handle);
    return it == objects().end() || it->second.surface != surface ? nullptr : &it->second;
}

}  // namespace

extern "C" {

CUresult CUDAAPI cuArray3DCreate_v2(CUarray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* d) {
    VULKANIZER_TRACE_API();
    if (!pHandle || !d || !d->Width || !validChannels(d->NumChannels) || !channelBytes(d->Format))
        return CUDA_ERROR_INVALID_VALUE;
    // a cube map: square faces, 6 of them (a multiple of 6 when layered)
    if ((d->Flags & CUDA_ARRAY3D_CUBEMAP) &&
        (d->Width != d->Height || !d->Depth || d->Depth % 6 || (!(d->Flags & CUDA_ARRAY3D_LAYERED) && d->Depth != 6)))
        return CUDA_ERROR_INVALID_VALUE;
    if (d->Height == 0 && d->Depth && !(d->Flags & CUDA_ARRAY3D_LAYERED)) return CUDA_ERROR_INVALID_VALUE;
    ApiScope s;
    if (CUresult r = enterContext(s)) return r;
    auto* a = new CUarray_st();
    a->desc = *d;
    a->texelBytes = channelBytes(d->Format) * d->NumChannels;
    a->pitch = d->Width * a->texelBytes;
    a->slice = a->pitch * std::max<size_t>(d->Height, 1);
    if (CUresult r = cuMemAlloc(&a->data, a->slice * std::max<size_t>(d->Depth, 1))) {
        delete a;
        return r;
    }
    Driver::get().registerHandle(a);
    *pHandle = a;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuArrayCreate_v2(CUarray* pHandle, const CUDA_ARRAY_DESCRIPTOR* d) {
    if (!d) return CUDA_ERROR_INVALID_VALUE;
    const CUDA_ARRAY3D_DESCRIPTOR d3{d->Width, d->Height, 0, d->Format, d->NumChannels, 0};
    return cuArray3DCreate_v2(pHandle, &d3);
}

CUresult CUDAAPI cuArray3DGetDescriptor_v2(CUDA_ARRAY3D_DESCRIPTOR* out, CUarray hArray) {
    if (!out) return CUDA_ERROR_INVALID_VALUE;
    if (!validHandle(hArray)) return CUDA_ERROR_INVALID_HANDLE;
    *out = hArray->desc;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuArrayGetDescriptor_v2(CUDA_ARRAY_DESCRIPTOR* out, CUarray hArray) {
    if (!out) return CUDA_ERROR_INVALID_VALUE;
    if (!validHandle(hArray)) return CUDA_ERROR_INVALID_HANDLE;
    if (hArray->desc.Depth) return CUDA_ERROR_INVALID_VALUE;  // a 3D array
    *out = {hArray->desc.Width, hArray->desc.Height, hArray->desc.Format, hArray->desc.NumChannels};
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuArrayDestroy(CUarray hArray) {
    VULKANIZER_TRACE_API();
    if (!validHandle(hArray) || hArray->ownedByMipmap) return CUDA_ERROR_INVALID_HANDLE;
    const CUresult r = cuMemFree(hArray->data);
    Driver::get().unregisterHandle(hArray);
    delete hArray;
    return r;
}

// Mipmapped arrays: level l is an array of max(1, size >> l) in each dimension (layers and cube faces
// stay), all levels allocated up front.
CUresult CUDAAPI cuMipmappedArrayCreate(CUmipmappedArray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* d,
                                        unsigned int numMipmapLevels) {
    VULKANIZER_TRACE_API();
    if (!pHandle || !d || !numMipmapLevels) return CUDA_ERROR_INVALID_VALUE;
    const bool layered = d->Flags & (CUDA_ARRAY3D_LAYERED | CUDA_ARRAY3D_CUBEMAP);
    size_t largest = std::max<size_t>(d->Width, d->Height);
    if (!layered) largest = std::max<size_t>(largest, d->Depth);
    unsigned maxLevels = 1;
    while ((largest >> maxLevels) > 0) ++maxLevels;
    if (numMipmapLevels > maxLevels) return CUDA_ERROR_INVALID_VALUE;
    auto* m = new CUmipmappedArray_st();
    m->desc = *d;
    for (unsigned l = 0; l < numMipmapLevels; ++l) {
        CUDA_ARRAY3D_DESCRIPTOR ld = *d;
        auto shrink = [&](size_t v) { return v ? std::max<size_t>(v >> l, 1) : 0; };
        ld.Width = shrink(d->Width);
        ld.Height = shrink(d->Height);
        if (!layered) ld.Depth = shrink(d->Depth);
        CUarray a = nullptr;
        if (CUresult r = cuArray3DCreate_v2(&a, &ld)) {
            for (CUarray made : m->levels) made->ownedByMipmap = false, cuArrayDestroy(made);
            delete m;
            return r;
        }
        a->ownedByMipmap = true;
        m->levels.push_back(a);
    }
    Driver::get().registerHandle(m);
    *pHandle = m;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMipmappedArrayGetLevel(CUarray* pLevelArray, CUmipmappedArray hMipmappedArray, unsigned int level) {
    VULKANIZER_TRACE_API();
    if (!pLevelArray) return CUDA_ERROR_INVALID_VALUE;
    if (!validHandle(hMipmappedArray)) return CUDA_ERROR_INVALID_HANDLE;
    if (level >= hMipmappedArray->levels.size()) return CUDA_ERROR_INVALID_VALUE;
    *pLevelArray = hMipmappedArray->levels[level];
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMipmappedArrayDestroy(CUmipmappedArray hMipmappedArray) {
    VULKANIZER_TRACE_API();
    if (!validHandle(hMipmappedArray)) return CUDA_ERROR_INVALID_HANDLE;
    CUresult result = CUDA_SUCCESS;
    for (CUarray a : hMipmappedArray->levels) {
        a->ownedByMipmap = false;
        if (CUresult r = cuArrayDestroy(a)) result = r;
    }
    Driver::get().unregisterHandle(hMipmappedArray);
    delete hMipmappedArray;
    return result;
}

CUresult CUDAAPI cuMemcpy2D_v2(const CUDA_MEMCPY2D* pCopy) {
    VULKANIZER_TRACE_API();
    return copy2D(pCopy, false, nullptr);
}
CUresult CUDAAPI cuMemcpy2DUnaligned_v2(const CUDA_MEMCPY2D* pCopy) { return cuMemcpy2D_v2(pCopy); }
CUresult CUDAAPI cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D* pCopy, CUstream hStream) {
    VULKANIZER_TRACE_API();
    if (pCopy) {
        ApiScope s;
        if (CUresult r = enterContext(s)) return r;
        if (CUstream cs = capturingStream(hStream)) {
            CUDA_MEMCPY3D c{};
            c.srcXInBytes = pCopy->srcXInBytes, c.srcY = pCopy->srcY;
            c.srcMemoryType = pCopy->srcMemoryType, c.srcHost = pCopy->srcHost, c.srcDevice = pCopy->srcDevice;
            c.srcArray = pCopy->srcArray, c.srcPitch = pCopy->srcPitch;
            c.dstXInBytes = pCopy->dstXInBytes, c.dstY = pCopy->dstY;
            c.dstMemoryType = pCopy->dstMemoryType, c.dstHost = pCopy->dstHost, c.dstDevice = pCopy->dstDevice;
            c.dstArray = pCopy->dstArray, c.dstPitch = pCopy->dstPitch;
            c.WidthInBytes = pCopy->WidthInBytes, c.Height = pCopy->Height, c.Depth = 1;
            return captureCopy3D(cs, c);
        }
    }
    return copy2D(pCopy, true, hStream);
}
CUresult CUDAAPI cuMemcpy3D_v2(const CUDA_MEMCPY3D* pCopy) {
    VULKANIZER_TRACE_API();
    return copy3DParams(pCopy, false, nullptr);
}
CUresult CUDAAPI cuMemcpy3DAsync_v2(const CUDA_MEMCPY3D* pCopy, CUstream hStream) {
    VULKANIZER_TRACE_API();
    if (pCopy) {
        ApiScope s;
        if (CUresult r = enterContext(s)) return r;
        if (CUstream cs = capturingStream(hStream)) return captureCopy3D(cs, *pCopy);
    }
    return copy3DParams(pCopy, true, hStream);
}

CUresult CUDAAPI cuMemcpyHtoA_v2(CUarray dstArray, size_t dstOffset, const void* srcHost, size_t ByteCount) {
    CUdeviceptr d;
    if (CUresult r = arrayRange(dstArray, dstOffset, ByteCount, d)) return r;
    return cuMemcpyHtoD_v2(d, srcHost, ByteCount);
}
CUresult CUDAAPI cuMemcpyAtoH_v2(void* dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount) {
    CUdeviceptr s;
    if (CUresult r = arrayRange(srcArray, srcOffset, ByteCount, s)) return r;
    return cuMemcpyDtoH_v2(dstHost, s, ByteCount);
}
CUresult CUDAAPI cuMemcpyDtoA_v2(CUarray dstArray, size_t dstOffset, CUdeviceptr srcDevice, size_t ByteCount) {
    CUdeviceptr d;
    if (CUresult r = arrayRange(dstArray, dstOffset, ByteCount, d)) return r;
    return cuMemcpyDtoD_v2(d, srcDevice, ByteCount);
}
CUresult CUDAAPI cuMemcpyAtoD_v2(CUdeviceptr dstDevice, CUarray srcArray, size_t srcOffset, size_t ByteCount) {
    CUdeviceptr s;
    if (CUresult r = arrayRange(srcArray, srcOffset, ByteCount, s)) return r;
    return cuMemcpyDtoD_v2(dstDevice, s, ByteCount);
}
CUresult CUDAAPI cuMemcpyAtoA_v2(CUarray dstArray, size_t dstOffset, CUarray srcArray, size_t srcOffset,
                                 size_t ByteCount) {
    CUdeviceptr d, s;
    if (CUresult r = arrayRange(dstArray, dstOffset, ByteCount, d)) return r;
    if (CUresult r = arrayRange(srcArray, srcOffset, ByteCount, s)) return r;
    return cuMemcpyDtoD_v2(d, s, ByteCount);
}
CUresult CUDAAPI cuMemcpyHtoAAsync_v2(CUarray dstArray, size_t dstOffset, const void* srcHost, size_t ByteCount,
                                      CUstream hStream) {
    CUdeviceptr d;
    if (CUresult r = arrayRange(dstArray, dstOffset, ByteCount, d)) return r;
    return cuMemcpyHtoDAsync_v2(d, srcHost, ByteCount, hStream);
}
CUresult CUDAAPI cuMemcpyAtoHAsync_v2(void* dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount,
                                      CUstream hStream) {
    CUdeviceptr s;
    if (CUresult r = arrayRange(srcArray, srcOffset, ByteCount, s)) return r;
    return cuMemcpyDtoHAsync_v2(dstHost, s, ByteCount, hStream);
}

CUresult CUDAAPI cuTexObjectCreate(CUtexObject* pTexObject, const CUDA_RESOURCE_DESC* pResDesc,
                                   const CUDA_TEXTURE_DESC* pTexDesc, const CUDA_RESOURCE_VIEW_DESC*) {
    VULKANIZER_TRACE_API();
    if (!pTexDesc) return CUDA_ERROR_INVALID_VALUE;
    return createObject(pTexObject, pResDesc, pTexDesc, false);
}
CUresult CUDAAPI cuTexObjectDestroy(CUtexObject texObject) {
    VULKANIZER_TRACE_API();
    return destroyObject(texObject, false);
}
CUresult CUDAAPI cuTexObjectGetResourceDesc(CUDA_RESOURCE_DESC* pResDesc, CUtexObject texObject) {
    std::lock_guard<std::recursive_mutex> lock(Driver::get().mutex());
    const TextureObject* o = findObject(texObject, false);
    if (!o || !pResDesc) return CUDA_ERROR_INVALID_VALUE;
    *pResDesc = o->res;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI cuTexObjectGetTextureDesc(CUDA_TEXTURE_DESC* pTexDesc, CUtexObject texObject) {
    std::lock_guard<std::recursive_mutex> lock(Driver::get().mutex());
    const TextureObject* o = findObject(texObject, false);
    if (!o || !pTexDesc) return CUDA_ERROR_INVALID_VALUE;
    *pTexDesc = o->tex;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI cuTexObjectGetResourceViewDesc(CUDA_RESOURCE_VIEW_DESC*, CUtexObject) {
    return CUDA_ERROR_INVALID_VALUE;  // no views
}
CUresult CUDAAPI cuSurfObjectCreate(CUsurfObject* pSurfObject, const CUDA_RESOURCE_DESC* pResDesc) {
    VULKANIZER_TRACE_API();
    return createObject(pSurfObject, pResDesc, nullptr, true);
}
CUresult CUDAAPI cuSurfObjectDestroy(CUsurfObject surfObject) {
    VULKANIZER_TRACE_API();
    return destroyObject(surfObject, true);
}
CUresult CUDAAPI cuSurfObjectGetResourceDesc(CUDA_RESOURCE_DESC* pResDesc, CUsurfObject surfObject) {
    std::lock_guard<std::recursive_mutex> lock(Driver::get().mutex());
    const TextureObject* o = findObject(surfObject, true);
    if (!o || !pResDesc) return CUDA_ERROR_INVALID_VALUE;
    *pResDesc = o->res;
    return CUDA_SUCCESS;
}

// The unversioned names (cuda.h maps them to _v2), for binaries that link them.
#undef cuArrayCreate
#undef cuArrayGetDescriptor
#undef cuArray3DCreate
#undef cuArray3DGetDescriptor
#undef cuMemcpy2D
#undef cuMemcpy2DUnaligned
#undef cuMemcpy2DAsync
#undef cuMemcpy3D
#undef cuMemcpy3DAsync
CUresult CUDAAPI cuArrayCreate(CUarray* p, const CUDA_ARRAY_DESCRIPTOR* d);
CUresult CUDAAPI cuArrayCreate(CUarray* p, const CUDA_ARRAY_DESCRIPTOR* d) { return cuArrayCreate_v2(p, d); }
CUresult CUDAAPI cuArrayGetDescriptor(CUDA_ARRAY_DESCRIPTOR* d, CUarray a);
CUresult CUDAAPI cuArrayGetDescriptor(CUDA_ARRAY_DESCRIPTOR* d, CUarray a) { return cuArrayGetDescriptor_v2(d, a); }
CUresult CUDAAPI cuArray3DCreate(CUarray* p, const CUDA_ARRAY3D_DESCRIPTOR* d);
CUresult CUDAAPI cuArray3DCreate(CUarray* p, const CUDA_ARRAY3D_DESCRIPTOR* d) { return cuArray3DCreate_v2(p, d); }
CUresult CUDAAPI cuArray3DGetDescriptor(CUDA_ARRAY3D_DESCRIPTOR* d, CUarray a);
CUresult CUDAAPI cuArray3DGetDescriptor(CUDA_ARRAY3D_DESCRIPTOR* d, CUarray a) { return cuArray3DGetDescriptor_v2(d, a); }
CUresult CUDAAPI cuMemcpy2D(const CUDA_MEMCPY2D* p);
CUresult CUDAAPI cuMemcpy2D(const CUDA_MEMCPY2D* p) { return cuMemcpy2D_v2(p); }
CUresult CUDAAPI cuMemcpy2DUnaligned(const CUDA_MEMCPY2D* p);
CUresult CUDAAPI cuMemcpy2DUnaligned(const CUDA_MEMCPY2D* p) { return cuMemcpy2DUnaligned_v2(p); }
CUresult CUDAAPI cuMemcpy2DAsync(const CUDA_MEMCPY2D* p, CUstream s);
CUresult CUDAAPI cuMemcpy2DAsync(const CUDA_MEMCPY2D* p, CUstream s) { return cuMemcpy2DAsync_v2(p, s); }
CUresult CUDAAPI cuMemcpy3D(const CUDA_MEMCPY3D* p);
CUresult CUDAAPI cuMemcpy3D(const CUDA_MEMCPY3D* p) { return cuMemcpy3D_v2(p); }
CUresult CUDAAPI cuMemcpy3DAsync(const CUDA_MEMCPY3D* p, CUstream s);
CUresult CUDAAPI cuMemcpy3DAsync(const CUDA_MEMCPY3D* p, CUstream s) { return cuMemcpy3DAsync_v2(p, s); }

}  // extern "C"
