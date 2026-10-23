// Cube maps (texCubemap, layered, with a level of detail) and mipmaps (tex2DLod with nearest and
// linear level filtering, clamps and bias, tex2DGrad, integer texels).
#include <cstdio>
#include <cuda_runtime.h>

__global__ void cube(cudaTextureObject_t t, cudaTextureObject_t layered, float* out) {
    // the six faces, then texel (1, 3) of +x, then face +x of cube 1, then a level of detail
    const float3 dirs[7] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}, {1, -0.5f, 0.5f}};
    for (int i = 0; i < 7; ++i) out[i] = texCubemap<float>(t, dirs[i].x, dirs[i].y, dirs[i].z);
    out[7] = texCubemapLayered<float>(layered, 1, 0, 0, 1);
    out[8] = texCubemapLod<float>(t, 0, 0, -1, 0.0f);
}

__global__ void mip(cudaTextureObject_t point, cudaTextureObject_t linear, cudaTextureObject_t clamped,
                    cudaTextureObject_t ints, float* out, int* iout) {
    const float lods[5] = {0, 1, 2, 1.4f, 5};
    for (int i = 0; i < 5; ++i) out[i] = tex2DLod<float>(point, 0.5f, 0.5f, lods[i]);
    out[5] = tex2DLod<float>(linear, 0.5f, 0.5f, 0.5f);
    out[6] = tex2DLod<float>(clamped, 0.5f, 0.5f, 2.0f);  // max level 1
    out[7] = tex2DLod<float>(clamped, 0.5f, 0.5f, 0.0f);  // bias 1
    out[8] = tex2DGrad<float>(point, 0.5f, 0.5f, make_float2(1.0f / 8, 0), make_float2(0, 1.0f / 8));
    out[9] = tex2DGrad<float>(point, 0.5f, 0.5f, make_float2(2.0f / 8, 0), make_float2(0, 2.0f / 8));
    iout[0] = tex2DLod<int>(ints, 0.5f, 0.5f, 2.0f);
}

cudaTextureObject_t texture(const cudaResourceDesc& r, bool linearMip, float maxLevel, float bias) {
    cudaTextureDesc t{};
    t.addressMode[0] = t.addressMode[1] = t.addressMode[2] = cudaAddressModeClamp;
    t.filterMode = cudaFilterModePoint;
    t.readMode = cudaReadModeElementType;
    t.normalizedCoords = 1;
    t.mipmapFilterMode = linearMip ? cudaFilterModeLinear : cudaFilterModePoint;
    t.maxMipmapLevelClamp = maxLevel;
    t.mipmapLevelBias = bias;
    cudaTextureObject_t o = 0;
    cudaCreateTextureObject(&o, &r, &t, nullptr);
    return o;
}

int main() {
    int failures = 0;
    auto expect = [&](const char* what, float got, float want) {
        if (got != want) {
            std::printf("%s: %g, want %g\n", what, got, want);
            ++failures;
        }
    };
    const cudaChannelFormatDesc f1 = cudaCreateChannelDesc<float>();

    // cube maps: face f texel (i, j) = 10 f + i + 4 j; the layered one has 2 cubes, layer = 6 c + f
    float faces[12 * 16];
    for (int l = 0; l < 12; ++l)
        for (int j = 0; j < 4; ++j)
            for (int i = 0; i < 4; ++i) faces[l * 16 + j * 4 + i] = float(10 * l + i + 4 * j);
    cudaArray_t cubeArray, layeredArray;
    cudaMalloc3DArray(&cubeArray, &f1, make_cudaExtent(4, 4, 6), cudaArrayCubemap);
    cudaMalloc3DArray(&layeredArray, &f1, make_cudaExtent(4, 4, 12), cudaArrayCubemap | cudaArrayLayered);
    for (cudaArray_t a : {cubeArray, layeredArray}) {
        cudaMemcpy3DParms p{};
        p.srcPtr = make_cudaPitchedPtr(faces, 16, 4, 4);
        p.dstArray = a;
        p.extent = make_cudaExtent(4, 4, a == cubeArray ? 6 : 12);
        p.kind = cudaMemcpyHostToDevice;
        cudaMemcpy3D(&p);
    }
    cudaResourceDesc rc{}, rl{};
    rc.resType = rl.resType = cudaResourceTypeArray;
    rc.res.array.array = cubeArray;
    rl.res.array.array = layeredArray;
    float* d;
    int* di;
    cudaMalloc(&d, 16 * sizeof(float));
    cudaMalloc(&di, sizeof(int));
    cube<<<1, 1>>>(texture(rc, false, 0, 0), texture(rl, false, 0, 0), d);
    float got[16];
    cudaMemcpy(got, d, 9 * sizeof(float), cudaMemcpyDeviceToHost);
    // texel (2, 2) of each face (10 l + 10: the center rounds up), +x (1, 3), cube 1's +x, -z
    const float wantCube[9] = {10, 20, 30, 40, 50, 60, 13, 70, 60};
    for (int i = 0; i < 9; ++i) {
        char what[32];
        std::snprintf(what, sizeof what, "cube %d", i);
        expect(what, got[i], wantCube[i]);
    }

    // mipmaps: 8 x 8, 4 x 4, 2 x 2, 1 x 1 levels of value 100 (l + 1)
    cudaMipmappedArray_t mips, intMips;
    cudaMallocMipmappedArray(&mips, &f1, make_cudaExtent(8, 8, 0), 4);
    const cudaChannelFormatDesc i1 = cudaCreateChannelDesc<int>();
    cudaMallocMipmappedArray(&intMips, &i1, make_cudaExtent(8, 8, 0), 4);
    for (unsigned l = 0; l < 4; ++l) {
        const int n = 8 >> l;
        float v[64];
        int iv[64];
        for (int k = 0; k < n * n; ++k) v[k] = float(100 * (l + 1)), iv[k] = int(1000 + l);
        cudaArray_t level, intLevel;
        cudaGetMipmappedArrayLevel(&level, mips, l);
        cudaGetMipmappedArrayLevel(&intLevel, intMips, l);
        cudaMemcpy2DToArray(level, 0, 0, v, n * sizeof(float), n * sizeof(float), n, cudaMemcpyHostToDevice);
        cudaMemcpy2DToArray(intLevel, 0, 0, iv, n * sizeof(int), n * sizeof(int), n, cudaMemcpyHostToDevice);
    }
    cudaResourceDesc rm{}, ri{};
    rm.resType = ri.resType = cudaResourceTypeMipmappedArray;
    rm.res.mipmap.mipmap = mips;
    ri.res.mipmap.mipmap = intMips;
    mip<<<1, 1>>>(texture(rm, false, 3, 0), texture(rm, true, 3, 0), texture(rm, false, 1, 1), texture(ri, false, 3, 0),
                  d, di);
    int igot = 0;
    cudaMemcpy(got, d, 10 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&igot, di, sizeof igot, cudaMemcpyDeviceToHost);
    const float wantMip[10] = {100, 200, 300, 200, 400, 150, 200, 200, 100, 200};
    for (int i = 0; i < 10; ++i) {
        char what[32];
        std::snprintf(what, sizeof what, "mip %d", i);
        expect(what, got[i], wantMip[i]);
    }
    expect("int mip", float(igot), 1002);
    cudaFreeMipmappedArray(mips);
    cudaFreeMipmappedArray(intMips);
    const cudaError_t e = cudaDeviceSynchronize();
    std::printf("%s (%s)\n", failures ? "FAILED" : "OK", cudaGetErrorString(e));
    return failures || e != cudaSuccess;
}
