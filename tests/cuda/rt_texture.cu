// Textures and surfaces through the CUDA Runtime (sampled in software by Vulkanizer), against CPU
// references: 2D float arrays (point and linear filtering, unnormalized and normalized coordinates,
// wrap/clamp/mirror/border), uchar4 arrays read as integers and as normalized floats, tex1Dfetch on
// linear memory, pitch2D memory, 3D arrays (cudaMemcpy3D, trilinear), layered arrays, gathers,
// queries of surface data, and surfaces written and read by kernels.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

#define CHECK(call)                                                                            \
    do {                                                                                       \
        cudaError_t e = (call);                                                                \
        if (e != cudaSuccess) {                                                                \
            std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(e)); \
            return false;                                                                      \
        }                                                                                      \
    } while (0)

namespace {

int g_failures = 0;

bool report(const char* name, double err, double tol) {
    const bool ok = err <= tol;
    std::printf("%-40s max err %.3g%s\n", name, err, ok ? "" : "  FAILED");
    if (!ok) ++g_failures;
    return ok;
}

// ---- kernels ----------------------------------------------------------------------------------

__global__ void sample2D(cudaTextureObject_t t, float* out, int w, int h, float scale, float offset) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < w && y < h) out[y * w + x] = tex2D<float>(t, x * scale + offset, y * scale + offset);
}

__global__ void sample2DUchar4(cudaTextureObject_t ints, cudaTextureObject_t floats, uint4* outI, float4* outF, int w,
                               int h) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < w && y < h) {
        const uchar4 v = tex2D<uchar4>(ints, x + 0.5f, y + 0.5f);
        outI[y * w + x] = make_uint4(v.x, v.y, v.z, v.w);
        outF[y * w + x] = tex2D<float4>(floats, x + 0.5f, y + 0.5f);
    }
}

__global__ void fetch1D(cudaTextureObject_t t, float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = tex1Dfetch<float>(t, i) + tex1Dfetch<float>(t, i + n);  // the second: out of range, 0
}

__global__ void sample3D(cudaTextureObject_t t, float* out, int n, float shift) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n * n * n) {
        const int x = i % n, y = (i / n) % n, z = i / (n * n);
        out[i] = tex3D<float>(t, x + shift, y + shift, z + shift);
    }
}

__global__ void sampleLayered(cudaTextureObject_t t, float* out, int w, int h, int layers) {
    const int x = threadIdx.x, y = threadIdx.y, l = blockIdx.x;
    if (x < w && y < h && l < layers) out[(l * h + y) * w + x] = tex2DLayered<float>(t, x + 0.5f, y + 0.5f, l);
}

__global__ void gather(cudaTextureObject_t t, float4* out, float x, float y) { *out = tex2Dgather<float4>(t, x, y, 0); }

__global__ void surfaceKernel(cudaSurfaceObject_t s, int w, int h) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < w && y < h) {
        const float2 v = surf2Dread<float2>(s, x * sizeof(float2), y);
        surf2Dwrite(make_float2(v.x * 2.0f + 1.0f, float(x + y)), s, x * sizeof(float2), y);
    }
    if (x == 0 && y == 0) {  // out of range: a zero-mode read is 0, a store does nothing
        const float2 z = surf2Dread<float2>(s, w * sizeof(float2), 0, cudaBoundaryModeZero);
        surf2Dwrite(make_float2(-1.0f, -1.0f), s, -1 * int(sizeof(float2)), 0, cudaBoundaryModeZero);
        if (z.x != 0.0f || z.y != 0.0f) surf2Dwrite(make_float2(-7.0f, -7.0f), s, 0, 0);
    }
}

// ---- CPU references ---------------------------------------------------------------------------

int addressIndex(int i, int size, cudaTextureAddressMode m) {
    if (i >= 0 && i < size) return i;
    if (m == cudaAddressModeWrap) return ((i % size) + size) % size;
    if (m == cudaAddressModeMirror) {
        int p = ((i % (2 * size)) + 2 * size) % (2 * size);
        return p < size ? p : 2 * size - 1 - p;
    }
    if (m == cudaAddressModeBorder) return -1;
    return i < 0 ? 0 : size - 1;
}

float texelAt(const std::vector<float>& v, int w, int h, int x, int y, cudaTextureAddressMode m) {
    const int ax = addressIndex(x, w, m), ay = addressIndex(y, h, m);
    return ax < 0 || ay < 0 ? 0.0f : v[size_t(ay) * w + ax];
}

float reference2D(const std::vector<float>& v, int w, int h, float u, float t, bool linear, bool normalized,
                  cudaTextureAddressMode m) {
    if (normalized) {
        if (m == cudaAddressModeWrap) u -= std::floor(u), t -= std::floor(t);
        u *= w, t *= h;
    }
    if (!linear) return texelAt(v, w, h, int(std::floor(u)), int(std::floor(t)), m);
    const float fx = u - 0.5f, fy = t - 0.5f;
    const int x0 = int(std::floor(fx)), y0 = int(std::floor(fy));
    const float a = std::floor((fx - x0) * 256.0f + 0.5f) / 256.0f, b = std::floor((fy - y0) * 256.0f + 0.5f) / 256.0f;
    return (1 - a) * (1 - b) * texelAt(v, w, h, x0, y0, m) + a * (1 - b) * texelAt(v, w, h, x0 + 1, y0, m) +
           (1 - a) * b * texelAt(v, w, h, x0, y0 + 1, m) + a * b * texelAt(v, w, h, x0 + 1, y0 + 1, m);
}

// ---- tests ------------------------------------------------------------------------------------

bool testFloat2D() {
    const int w = 37, h = 23;
    std::vector<float> img(size_t(w) * h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) img[size_t(y) * w + x] = std::sin(0.3f * x) + 0.1f * y;
    cudaChannelFormatDesc ch = cudaCreateChannelDesc<float>();
    cudaArray_t arr;
    CHECK(cudaMallocArray(&arr, &ch, w, h));
    CHECK(cudaMemcpy2DToArray(arr, 0, 0, img.data(), w * sizeof(float), w * sizeof(float), h, cudaMemcpyHostToDevice));
    float* out;
    CHECK(cudaMalloc(&out, img.size() * sizeof(float)));
    struct Case {
        const char* name;
        cudaTextureFilterMode filter;
        cudaTextureAddressMode mode;
        bool normalized;
        float scale, offset;
    } cases[] = {
        {"2D float point", cudaFilterModePoint, cudaAddressModeClamp, false, 1.0f, 0.5f},
        {"2D float linear", cudaFilterModeLinear, cudaAddressModeClamp, false, 0.9f, 0.3f},
        {"2D float linear clamp outside", cudaFilterModeLinear, cudaAddressModeClamp, false, 1.3f, -4.0f},
        {"2D float point border", cudaFilterModePoint, cudaAddressModeBorder, false, 1.2f, -3.0f},
        {"2D float normalized wrap linear", cudaFilterModeLinear, cudaAddressModeWrap, true, 0.07f, 0.01f},
        {"2D float normalized mirror point", cudaFilterModePoint, cudaAddressModeMirror, true, 0.06f, -0.3f},
    };
    for (const Case& c : cases) {
        cudaResourceDesc res{};
        res.resType = cudaResourceTypeArray;
        res.res.array.array = arr;
        cudaTextureDesc td{};
        td.addressMode[0] = td.addressMode[1] = c.mode;
        td.filterMode = c.filter;
        td.readMode = cudaReadModeElementType;
        td.normalizedCoords = c.normalized;
        cudaTextureObject_t t;
        CHECK(cudaCreateTextureObject(&t, &res, &td, nullptr));
        sample2D<<<dim3((w + 15) / 16, (h + 15) / 16), dim3(16, 16)>>>(t, out, w, h, c.scale, c.offset);
        std::vector<float> got(img.size());
        CHECK(cudaMemcpy(got.data(), out, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
        double err = 0;
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                const float want = reference2D(img, w, h, x * c.scale + c.offset, y * c.scale + c.offset,
                                               c.filter == cudaFilterModeLinear, c.normalized, c.mode);
                err = std::max(err, double(std::fabs(got[size_t(y) * w + x] - want)));
            }
        report(c.name, err, 2e-5);
        CHECK(cudaDestroyTextureObject(t));
    }
    CHECK(cudaFree(out));
    CHECK(cudaFreeArray(arr));
    return true;
}

bool testUchar4() {
    const int w = 19, h = 11;
    std::vector<uchar4> img(size_t(w) * h);
    for (size_t i = 0; i < img.size(); ++i) img[i] = make_uchar4(i * 7 % 256, i * 13 % 256, 255 - i % 256, 128);
    cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
    cudaArray_t arr;
    CHECK(cudaMallocArray(&arr, &ch, w, h));
    CHECK(cudaMemcpy2DToArray(arr, 0, 0, img.data(), w * 4, w * 4, h, cudaMemcpyHostToDevice));
    cudaResourceDesc res{};
    res.resType = cudaResourceTypeArray;
    res.res.array.array = arr;
    cudaTextureDesc ti{}, tf{};
    ti.readMode = cudaReadModeElementType;
    tf.readMode = cudaReadModeNormalizedFloat;
    cudaTextureObject_t a, b;
    CHECK(cudaCreateTextureObject(&a, &res, &ti, nullptr));
    CHECK(cudaCreateTextureObject(&b, &res, &tf, nullptr));
    uint4* oi;
    float4* of;
    CHECK(cudaMalloc(&oi, img.size() * sizeof(uint4)));
    CHECK(cudaMalloc(&of, img.size() * sizeof(float4)));
    sample2DUchar4<<<dim3(2, 1), dim3(16, 16)>>>(a, b, oi, of, w, h);
    std::vector<uint4> gi(img.size());
    std::vector<float4> gf(img.size());
    CHECK(cudaMemcpy(gi.data(), oi, gi.size() * sizeof(uint4), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(gf.data(), of, gf.size() * sizeof(float4), cudaMemcpyDeviceToHost));
    double errI = 0, errF = 0;
    for (size_t i = 0; i < img.size(); ++i) {
        errI = std::max<double>({errI, double(std::abs(int(gi[i].x) - img[i].x)), double(std::abs(int(gi[i].w) - img[i].w))});
        errF = std::max<double>({errF, std::fabs(gf[i].x - img[i].x / 255.0f), std::fabs(gf[i].z - img[i].z / 255.0f)});
    }
    report("uchar4 read as integer", errI, 0);
    report("uchar4 read as normalized float", errF, 1e-6);
    CHECK(cudaDestroyTextureObject(a));
    CHECK(cudaDestroyTextureObject(b));
    CHECK(cudaFree(oi));
    CHECK(cudaFree(of));
    CHECK(cudaFreeArray(arr));
    return true;
}

bool testLinearAndPitch() {
    const int n = 300;
    std::vector<float> v(n);
    for (int i = 0; i < n; ++i) v[i] = 0.5f * i;
    float *lin, *out;
    CHECK(cudaMalloc(&lin, n * sizeof(float)));
    CHECK(cudaMalloc(&out, n * sizeof(float)));
    CHECK(cudaMemcpy(lin, v.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    cudaResourceDesc res{};
    res.resType = cudaResourceTypeLinear;
    res.res.linear.devPtr = lin;
    res.res.linear.desc = cudaCreateChannelDesc<float>();
    res.res.linear.sizeInBytes = n * sizeof(float);
    cudaTextureDesc td{};
    td.readMode = cudaReadModeElementType;
    cudaTextureObject_t t;
    CHECK(cudaCreateTextureObject(&t, &res, &td, nullptr));
    fetch1D<<<(n + 127) / 128, 128>>>(t, out, n);
    std::vector<float> got(n);
    CHECK(cudaMemcpy(got.data(), out, n * sizeof(float), cudaMemcpyDeviceToHost));
    double err = 0;
    for (int i = 0; i < n; ++i) err = std::max(err, double(std::fabs(got[i] - v[i])));
    report("tex1Dfetch linear memory", err, 0);
    CHECK(cudaDestroyTextureObject(t));

    // pitch2D: cudaMallocPitch'ed rows
    const int w = 29, h = 17;
    std::vector<float> img(size_t(w) * h);
    for (size_t i = 0; i < img.size(); ++i) img[i] = float(i % 97) - 40.0f;
    float* pitched;
    size_t pitch;
    CHECK(cudaMallocPitch(reinterpret_cast<void**>(&pitched), &pitch, w * sizeof(float), h));
    CHECK(cudaMemcpy2D(pitched, pitch, img.data(), w * sizeof(float), w * sizeof(float), h, cudaMemcpyHostToDevice));
    cudaResourceDesc pr{};
    pr.resType = cudaResourceTypePitch2D;
    pr.res.pitch2D.devPtr = pitched;
    pr.res.pitch2D.desc = cudaCreateChannelDesc<float>();
    pr.res.pitch2D.width = w;
    pr.res.pitch2D.height = h;
    pr.res.pitch2D.pitchInBytes = pitch;
    td.filterMode = cudaFilterModeLinear;
    CHECK(cudaCreateTextureObject(&t, &pr, &td, nullptr));
    float* out2;
    CHECK(cudaMalloc(&out2, img.size() * sizeof(float)));
    sample2D<<<dim3(2, 2), dim3(16, 16)>>>(t, out2, w, h, 0.97f, 0.2f);
    std::vector<float> g2(img.size());
    CHECK(cudaMemcpy(g2.data(), out2, g2.size() * sizeof(float), cudaMemcpyDeviceToHost));
    err = 0;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            err = std::max(err, double(std::fabs(g2[size_t(y) * w + x] -
                                                 reference2D(img, w, h, x * 0.97f + 0.2f, y * 0.97f + 0.2f, true, false,
                                                             cudaAddressModeClamp))));
    report("pitch2D linear", err, 1e-4);
    CHECK(cudaDestroyTextureObject(t));
    CHECK(cudaFree(lin));
    CHECK(cudaFree(out));
    CHECK(cudaFree(out2));
    CHECK(cudaFree(pitched));
    return true;
}

bool test3DAndLayered() {
    const int n = 8;
    std::vector<float> vol(size_t(n) * n * n);
    for (size_t i = 0; i < vol.size(); ++i) vol[i] = float(i % 11) * 0.25f;
    cudaChannelFormatDesc ch = cudaCreateChannelDesc<float>();
    cudaArray_t arr;
    CHECK(cudaMalloc3DArray(&arr, &ch, make_cudaExtent(n, n, n)));
    cudaMemcpy3DParms p{};
    p.srcPtr = make_cudaPitchedPtr(vol.data(), n * sizeof(float), n, n);
    p.dstArray = arr;
    p.extent = make_cudaExtent(n, n, n);
    p.kind = cudaMemcpyHostToDevice;
    CHECK(cudaMemcpy3D(&p));
    cudaResourceDesc res{};
    res.resType = cudaResourceTypeArray;
    res.res.array.array = arr;
    cudaTextureDesc td{};
    td.filterMode = cudaFilterModeLinear;
    td.addressMode[0] = td.addressMode[1] = td.addressMode[2] = cudaAddressModeClamp;
    cudaTextureObject_t t;
    CHECK(cudaCreateTextureObject(&t, &res, &td, nullptr));
    float* out;
    CHECK(cudaMalloc(&out, vol.size() * sizeof(float)));
    sample3D<<<(n * n * n + 127) / 128, 128>>>(t, out, n, 1.0f);  // halfway between texels
    std::vector<float> got(vol.size());
    CHECK(cudaMemcpy(got.data(), out, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
    auto at = [&](int x, int y, int z) {
        x = std::min(std::max(x, 0), n - 1), y = std::min(std::max(y, 0), n - 1), z = std::min(std::max(z, 0), n - 1);
        return vol[(size_t(z) * n + y) * n + x];
    };
    double err = 0;
    for (int z = 0; z < n; ++z)
        for (int y = 0; y < n; ++y)
            for (int x = 0; x < n; ++x) {
                float s = 0;  // coordinate x + 1 -> texels x and x + 1, weight 1/2 each
                for (int k = 0; k < 8; ++k) s += at(x + (k & 1), y + ((k >> 1) & 1), z + (k >> 2)) * 0.125f;
                err = std::max(err, double(std::fabs(got[(size_t(z) * n + y) * n + x] - s)));
            }
    report("3D float trilinear (cudaMemcpy3D)", err, 1e-5);
    CHECK(cudaDestroyTextureObject(t));
    CHECK(cudaFreeArray(arr));

    // layered: 3 layers of 6 x 5
    const int w = 6, h = 5, layers = 3;
    std::vector<float> lay(size_t(w) * h * layers);
    for (size_t i = 0; i < lay.size(); ++i) lay[i] = float(i);
    CHECK(cudaMalloc3DArray(&arr, &ch, make_cudaExtent(w, h, layers), cudaArrayLayered));
    p = cudaMemcpy3DParms{};
    p.srcPtr = make_cudaPitchedPtr(lay.data(), w * sizeof(float), w, h);
    p.dstArray = arr;
    p.extent = make_cudaExtent(w, h, layers);
    p.kind = cudaMemcpyHostToDevice;
    CHECK(cudaMemcpy3D(&p));
    res.res.array.array = arr;
    td = cudaTextureDesc{};
    CHECK(cudaCreateTextureObject(&t, &res, &td, nullptr));
    sampleLayered<<<layers, dim3(8, 8)>>>(t, out, w, h, layers);
    std::vector<float> gl(lay.size());
    CHECK(cudaMemcpy(gl.data(), out, gl.size() * sizeof(float), cudaMemcpyDeviceToHost));
    err = 0;
    for (size_t i = 0; i < lay.size(); ++i) err = std::max(err, double(std::fabs(gl[i] - lay[i])));
    report("2D layered", err, 0);
    CHECK(cudaDestroyTextureObject(t));
    CHECK(cudaFreeArray(arr));
    CHECK(cudaFree(out));
    return true;
}

bool testGather() {
    const int w = 4, h = 4;
    std::vector<float> img(16);
    for (int i = 0; i < 16; ++i) img[i] = float(i);
    cudaChannelFormatDesc ch = cudaCreateChannelDesc<float>();
    cudaArray_t arr;
    CHECK(cudaMallocArray(&arr, &ch, w, h, cudaArrayTextureGather));
    CHECK(cudaMemcpy2DToArray(arr, 0, 0, img.data(), w * 4, w * 4, h, cudaMemcpyHostToDevice));
    cudaResourceDesc res{};
    res.resType = cudaResourceTypeArray;
    res.res.array.array = arr;
    cudaTextureDesc td{};
    cudaTextureObject_t t;
    CHECK(cudaCreateTextureObject(&t, &res, &td, nullptr));
    float4* out;
    CHECK(cudaMalloc(&out, sizeof(float4)));
    gather<<<1, 1>>>(t, out, 2.0f, 2.0f);  // the footprint (1..2, 1..2)
    float4 g;
    CHECK(cudaMemcpy(&g, out, sizeof g, cudaMemcpyDeviceToHost));
    // x: (x0, y0 + 1), y: (x0 + 1, y0 + 1), z: (x0 + 1, y0), w: (x0, y0)
    const double err = std::max<double>({std::fabs(g.x - img[2 * w + 1]), std::fabs(g.y - img[2 * w + 2]),
                                 std::fabs(g.z - img[1 * w + 2]), std::fabs(g.w - img[1 * w + 1])});
    report("tex2Dgather", err, 0);
    CHECK(cudaDestroyTextureObject(t));
    CHECK(cudaFree(out));
    CHECK(cudaFreeArray(arr));
    return true;
}

bool testSurface() {
    const int w = 21, h = 13;
    std::vector<float2> img(size_t(w) * h);
    for (size_t i = 0; i < img.size(); ++i) img[i] = make_float2(float(i), -float(i));
    cudaChannelFormatDesc ch = cudaCreateChannelDesc<float2>();
    cudaArray_t arr;
    CHECK(cudaMallocArray(&arr, &ch, w, h, cudaArraySurfaceLoadStore));
    CHECK(cudaMemcpy2DToArray(arr, 0, 0, img.data(), w * sizeof(float2), w * sizeof(float2), h, cudaMemcpyHostToDevice));
    cudaResourceDesc res{};
    res.resType = cudaResourceTypeArray;
    res.res.array.array = arr;
    cudaSurfaceObject_t s;
    CHECK(cudaCreateSurfaceObject(&s, &res));
    surfaceKernel<<<dim3(2, 1), dim3(16, 16)>>>(s, w, h);
    std::vector<float2> got(img.size());
    CHECK(cudaMemcpy2DFromArray(got.data(), w * sizeof(float2), arr, 0, 0, w * sizeof(float2), h, cudaMemcpyDeviceToHost));
    double err = 0;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const float2 g = got[size_t(y) * w + x], v = img[size_t(y) * w + x];
            err = std::max<double>({err, std::fabs(g.x - (v.x * 2 + 1)), std::fabs(g.y - float(x + y))});
        }
    report("surf2Dread/surf2Dwrite (out of range: zero)", err, 0);
    CHECK(cudaDestroySurfaceObject(s));
    cudaChannelFormatDesc back;
    cudaExtent extent;
    unsigned flags;
    CHECK(cudaArrayGetInfo(&back, &extent, &flags, arr));
    report("cudaArrayGetInfo", (back.x == 32 && back.y == 32 && back.z == 0 && back.f == cudaChannelFormatKindFloat &&
                                extent.width == size_t(w) && extent.height == size_t(h) && flags == cudaArraySurfaceLoadStore)
                                   ? 0
                                   : 1,
           0);
    CHECK(cudaFreeArray(arr));
    return true;
}

}  // namespace

int main() {
    bool ok = testFloat2D() && testUchar4() && testLinearAndPitch() && test3DAndLayered() && testGather() && testSurface();
    ok = ok && g_failures == 0;
    std::printf(ok ? "OK\n" : "FAILED\n");
    return ok ? 0 : 1;
}
