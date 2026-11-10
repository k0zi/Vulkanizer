// NPU offload without an NPU (src/npu): the switches, the routing policy, sysfs discovery on a
// fake tree, the type conversions and the layout conversion of the backends.

#include "npu/host.h"
#include "npu/npu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

using namespace vulkanizer::npu;
namespace fs = std::filesystem;

#define EXPECT(cond)                                                                            \
    do {                                                                                        \
        if (!(cond)) {                                                                          \
            std::fprintf(stderr, "%s:%d: expectation failed: %s\n", __FILE__, __LINE__, #cond); \
            std::exit(1);                                                                       \
        }                                                                                       \
    } while (0)

static void testMode() {
    EXPECT(parseMode(nullptr) == Mode::Off);
    EXPECT(parseMode("") == Mode::Off);
    EXPECT(parseMode("off") == Mode::Off);
    EXPECT(parseMode("0") == Mode::Off);
    EXPECT(parseMode("auto") == Mode::Auto);
    EXPECT(parseMode("1") == Mode::Auto);
    EXPECT(parseMode("force") == Mode::Force);
    EXPECT(parseMode("bogus") == Mode::Off);
}

static void testRoute() {
    const Policy p;  // 2^24 .. 2^27, 64, 2x padding
    GemmShape big;  // 512^3 = 2^27, the largest the auto mode takes
    big.m = big.n = big.k = 512;
    const int64_t bigVolume = int64_t(1) << 27;
    GemmShape small;
    small.m = small.n = small.k = 128;
    const int64_t smallVolume = 128 * 128 * 128;
    GemmShape huge;
    huge.m = huge.n = huge.k = 1024;
    const int64_t hugeVolume = int64_t(1) << 30;
    GemmShape gemv;
    gemv.m = 4096, gemv.n = 1, gemv.k = 4096;
    EXPECT(route(big, Mode::Off, p, bigVolume) == Target::Gpu);
    EXPECT(route(big, Mode::Auto, p, 0) == Target::Gpu);
    EXPECT(route(big, Mode::Auto, p, bigVolume) == Target::Npu);
    EXPECT(route(small, Mode::Auto, p, smallVolume) == Target::Gpu);
    EXPECT(route(small, Mode::Force, p, smallVolume) == Target::Npu);
    EXPECT(route(huge, Mode::Auto, p, hugeVolume) == Target::Gpu);
    EXPECT(route(huge, Mode::Force, p, hugeVolume) == Target::Npu);
    EXPECT(route(gemv, Mode::Auto, p, 4096 * 64 * 4096) == Target::Gpu);
    GemmShape batched = small;  // 128^3 * 64 = 2^27
    batched.batch = 64;
    EXPECT(route(batched, Mode::Auto, p, smallVolume) == Target::Npu);
    batched.batch = 65;
    EXPECT(route(batched, Mode::Auto, p, smallVolume) == Target::Gpu);
    GemmShape empty = big;
    empty.k = 0;
    EXPECT(route(empty, Mode::Force, p, bigVolume) == Target::Gpu);
    // write-combined operands: GPU in auto mode, NPU when forced
    GemmShape uncached = big;
    uncached.cached = false;
    EXPECT(route(uncached, Mode::Auto, p, bigVolume) == Target::Gpu);
    EXPECT(route(uncached, Mode::Force, p, bigVolume) == Target::Npu);
    // padding: 2x is fine, more stays on the GPU in auto mode (force takes it anyway)
    EXPECT(route(big, Mode::Auto, p, 2 * bigVolume) == Target::Npu);
    EXPECT(route(big, Mode::Auto, p, 2 * bigVolume + 1) == Target::Gpu);
    EXPECT(route(big, Mode::Force, p, 8 * bigVolume) == Target::Npu);
}

static void writeFile(const fs::path& file, const char* text) {
    fs::create_directories(file.parent_path());
    std::ofstream(file) << text;
}

static void testDetect(const fs::path& tmp) {
    const fs::path root = tmp / "sys";
    fs::remove_all(root);
    EXPECT(detect(root.string()).empty());
    // accel0: XDNA2 (Strix), accel1: Meteor Lake, accel2: no driver bound
    const fs::path drivers = root / "bus" / "pci" / "drivers";
    fs::create_directories(drivers / "amdxdna");
    fs::create_directories(drivers / "intel_vpu");
    const char* ids[][3] = {{"accel0", "0x1022", "0x17f0"}, {"accel1", "0x8086", "0x7d1d"}, {"accel2", "0x1234", "0x0001"}};
    for (auto& id : ids) {
        const fs::path dev = root / "devices" / id[0];
        writeFile(dev / "vendor", id[1]);
        writeFile(dev / "device", id[2]);
        fs::create_directories(root / "class" / "accel" / id[0]);
        fs::create_directory_symlink(dev, root / "class" / "accel" / id[0] / "device");
    }
    fs::create_directory_symlink(drivers / "amdxdna", root / "devices" / "accel0" / "driver");
    fs::create_directory_symlink(drivers / "intel_vpu", root / "devices" / "accel1" / "driver");
    const auto found = detect(root.string());
    EXPECT(found.size() == 3);
    EXPECT(found[0].vendor == Vendor::AmdXdna && found[0].generation == "XDNA2" && found[0].node == "/dev/accel/accel0");
    EXPECT(found[0].pciVendor == 0x1022 && found[0].pciDevice == 0x17f0 && found[0].driver == "amdxdna");
    EXPECT(found[1].vendor == Vendor::IntelVpu && found[1].generation == "NPU3720");
    EXPECT(found[2].vendor == Vendor::Unknown && found[2].driver.empty() && found[2].generation.empty());
    fs::remove_all(root);
}

static void testConversions() {
    const float values[] = {0.0f, -0.0f, 1.0f, -2.5f, 65504.0f, 6.1035156e-05f, 5.9604645e-08f, 0.1f, 3.14159f};
    for (float v : values) {
        EXPECT(std::fabs(halfToFloat(floatToHalf(v)) - v) <= std::fabs(v) * 1e-3f);
        EXPECT(std::fabs(bf16ToFloat(floatToBf16(v)) - v) <= std::fabs(v) * 8e-3f);
    }
    EXPECT(floatToHalf(65520.0f) == 0x7C00);  // halfway past the largest half rounds to infinity
    EXPECT(floatToHalf(1.0f + 1.0f / 2048) == 0x3C00);  // tie to even
    EXPECT(floatToHalf(2.9802322e-08f) == 0);  // half the smallest subnormal: tie to even (0)
    EXPECT(floatToBf16(1.0f + 1.0f / 256) == 0x3F80);  // tie to even
    EXPECT(std::isnan(halfToFloat(floatToHalf(NAN))) && std::isnan(bf16ToFloat(floatToBf16(NAN))));
}

// The CPU backend against a direct column-major reference, all transpose combinations, a batch
// and beta, f32 and f16 storage.
static void testCpuBackend() {
    auto backend = createCpuBackend();
    const int m = 5, n = 4, k = 3, batch = 2, ld = 7;
    for (int t = 0; t < 4; ++t) {
        for (DType type : {DType::F32, DType::F16}) {
            const bool ta = t & 1, tb = t & 2;
            const size_t es = dtypeSize(type);
            const int64_t stride = ld * 8;
            std::vector<char> A(size_t(stride * batch) * es), B(A.size()), C(A.size());
            for (int64_t i = 0; i < stride * batch; ++i) {
                store(A.data(), type, i, float((i * 7) % 11) - 5);
                store(B.data(), type, i, float((i * 5) % 9) - 4);
                store(C.data(), type, i, float(i % 3));
            }
            const std::vector<char> C0 = C;
            GemmArgs g;
            g.shape.transA = ta, g.shape.transB = tb;
            g.shape.m = m, g.shape.n = n, g.shape.k = k, g.shape.batch = batch;
            g.shape.a = g.shape.b = g.shape.c = type;
            g.alpha = 0.5f, g.beta = 2.0f;
            g.a = A.data(), g.lda = ld, g.strideA = stride;
            g.b = B.data(), g.ldb = ld, g.strideB = stride;
            g.c = C.data(), g.ldc = ld, g.strideC = stride;
            EXPECT(backend->gemm(g));
            for (int bi = 0; bi < batch; ++bi)
                for (int j = 0; j < n; ++j)
                    for (int i = 0; i < m; ++i) {
                        float sum = 0;
                        for (int p = 0; p < k; ++p) {
                            const float a = load(A.data(), type, bi * stride + (ta ? p + i * ld : i + p * ld));
                            const float b = load(B.data(), type, bi * stride + (tb ? j + p * ld : p + j * ld));
                            sum += a * b;
                        }
                        const int64_t at = bi * stride + i + j * ld;
                        const float want = 0.5f * sum + 2.0f * load(C0.data(), type, at);
                        EXPECT(std::fabs(load(C.data(), type, at) - want) < 1e-3f);
                    }
            // outside the m x n window nothing changes
            for (int64_t i = 0; i < stride * batch; ++i) {
                const int64_t row = (i % stride) % ld, col = (i % stride) / ld;
                if (row >= m || col >= n) EXPECT(load(C.data(), type, i) == load(C0.data(), type, i));
            }
        }
    }
}

int main() {
    testMode();
    testRoute();
    testDetect(fs::temp_directory_path() / ("vulkanizer_npu_test_" + std::to_string(getpid())));
    testConversions();
    testCpuBackend();
    std::printf("OK\n");
    return 0;
}
