// Asynchronous NPU GEMMs (plan.md 7.4) on the host reference backend, slowed down
// (VULKANIZER_NPU=force, VULKANIZER_NPU_BACKEND=cpu, VULKANIZER_NPU_CPU_DELAY_MS): the cuBLAS call
// returns before the NPU run, and everything after it in stream order (events, other streams, a
// captured graph) sees the result; split GEMMs (GPU + NPU together) are right.

#include "test_util.h"

#include <cublas_v2.h>

#include <chrono>
#include <cstdlib>
#include <vector>

namespace {

#define CHECK_BLAS(call)                                                                                   \
    do {                                                                                                   \
        cublasStatus_t s_ = (call);                                                                        \
        if (s_ != CUBLAS_STATUS_SUCCESS) {                                                                 \
            std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #call, cublasGetStatusName(s_)); \
            std::exit(1);                                                                                  \
        }                                                                                                  \
    } while (0)

constexpr int kN = 64;  // m = n = k
constexpr int kBatch = 3;

std::vector<float> values(int count, int seed) {
    std::vector<float> v(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) v[size_t(i)] = float((i * 7 + seed * 13) % 9 - 4);  // exact in bf16 and f32
    return v;
}

// C = A * B, column-major n x n, batch entries kN * kN apart
std::vector<float> reference(const std::vector<float>& a, const std::vector<float>& b, int batch) {
    std::vector<float> c(size_t(kN * kN * batch), 0.0f);
    for (int e = 0; e < batch; ++e)
        for (int j = 0; j < kN; ++j)
            for (int p = 0; p < kN; ++p)
                for (int i = 0; i < kN; ++i)
                    c[size_t(e * kN * kN + j * kN + i)] += a[size_t(e * kN * kN + p * kN + i)] * b[size_t(e * kN * kN + j * kN + p)];
    return c;
}

CUdeviceptr upload(const std::vector<float>& v) {
    CUdeviceptr d;
    CHECK_CU(cuMemAlloc(&d, v.size() * 4));
    CHECK_CU(cuMemcpyHtoD(d, v.data(), v.size() * 4));
    return d;
}

std::vector<float> download(CUdeviceptr d, size_t count) {
    std::vector<float> v(count);
    CHECK_CU(cuMemcpyDtoH(v.data(), d, count * 4));
    return v;
}

void gemm(cublasHandle_t h, CUdeviceptr a, CUdeviceptr b, CUdeviceptr c, int batch = 1) {
    const float one = 1.0f, zero = 0.0f;
    const long long stride = kN * kN;
    CHECK_BLAS(cublasGemmStridedBatchedEx(h, CUBLAS_OP_N, CUBLAS_OP_N, kN, kN, kN, &one, reinterpret_cast<void*>(a),
                                          CUDA_R_32F, kN, stride, reinterpret_cast<void*>(b), CUDA_R_32F, kN, stride,
                                          &zero, reinterpret_cast<void*>(c), CUDA_R_32F, kN, stride, batch,
                                          CUBLAS_COMPUTE_32F_FAST_16BF, CUBLAS_GEMM_DEFAULT));
}

}  // namespace

int main() {
    const char* delay = std::getenv("VULKANIZER_NPU_CPU_DELAY_MS");
    const int delayMs = delay ? std::atoi(delay) : 0;
    EXPECT(delayMs >= 100);  // run by ctest with the environment set
    CHECK_CU(cuInit(0));
    CUdevice dev;
    CHECK_CU(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CHECK_CU(cuCtxCreate(&ctx, 0, dev));
    CUstream s1, s2;
    CHECK_CU(cuStreamCreate(&s1, CU_STREAM_NON_BLOCKING));
    CHECK_CU(cuStreamCreate(&s2, CU_STREAM_NON_BLOCKING));
    cublasHandle_t h1, h2;
    CHECK_BLAS(cublasCreate(&h1));
    CHECK_BLAS(cublasCreate(&h2));
    CHECK_BLAS(cublasSetStream(h1, s1));
    CHECK_BLAS(cublasSetStream(h2, s2));

    const size_t one = size_t(kN * kN), all = one * kBatch;
    const std::vector<float> a = values(int(all), 1), b = values(int(all), 2);
    const CUdeviceptr da = upload(a), db = upload(b);
    CUdeviceptr dc, dd;
    CHECK_CU(cuMemAlloc(&dc, all * 4));
    CHECK_CU(cuMemAlloc(&dd, all * 4));
    CHECK_CU(cuStreamSynchronize(nullptr));

    // 1. The call returns before the NPU run; an event after it completes with it; another stream
    //    waiting for the event reads C (a copy, then a GEMV kernel on the copy) only after the run.
    {
        const std::vector<float> ref = reference(a, b, 1);
        const auto t0 = std::chrono::steady_clock::now();
        gemm(h1, da, db, dc);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        EXPECT(ms < delayMs / 2);
        CUevent ev;
        CHECK_CU(cuEventCreate(&ev, CU_EVENT_DISABLE_TIMING));
        CHECK_CU(cuEventRecord(ev, s1));
        EXPECT_CU(CUDA_ERROR_NOT_READY, cuEventQuery(ev));
        EXPECT_CU(CUDA_ERROR_NOT_READY, cuStreamQuery(s1));
        CHECK_CU(cuStreamWaitEvent(s2, ev, 0));
        CHECK_CU(cuMemcpyDtoDAsync(dd, dc, one * 4, s2));
        // y = D * ones: the row sums of C, by a GPU kernel
        const CUdeviceptr ones = upload(std::vector<float>(kN, 1.0f));
        CUdeviceptr y;
        CHECK_CU(cuMemAlloc(&y, kN * 4));
        const float alpha = 1.0f, beta = 0.0f;
        CHECK_BLAS(cublasSgemv(h2, CUBLAS_OP_N, kN, kN, &alpha, reinterpret_cast<float*>(dd), kN,
                               reinterpret_cast<float*>(ones), 1, &beta, reinterpret_cast<float*>(y), 1));
        CHECK_CU(cuStreamSynchronize(s2));
        CHECK_CU(cuEventQuery(ev));
        const std::vector<float> d = download(dd, one), sums = download(y, kN);
        for (size_t i = 0; i < one; ++i) EXPECT(d[i] == ref[i]);
        for (int i = 0; i < kN; ++i) {
            float sum = 0;
            for (int j = 0; j < kN; ++j) sum += ref[size_t(i + j * kN)];
            EXPECT(sums[size_t(i)] == sum);
        }
        CHECK_CU(cuMemFree(ones));
        CHECK_CU(cuMemFree(y));
        CHECK_CU(cuEventDestroy(ev));
    }

    // 2. Two streams through the same NPU buffers, one of them batched: each entry is its own run.
    {
        CHECK_CU(cuMemsetD32(dc, 0, all));
        CHECK_CU(cuMemsetD32(dd, 0, all));
        gemm(h1, da, db, dc);
        gemm(h2, db, da, dd, kBatch);
        CHECK_CU(cuStreamSynchronize(s1));
        CHECK_CU(cuStreamSynchronize(s2));
        const std::vector<float> ab = reference(a, b, 1), ba = reference(b, a, kBatch);
        const std::vector<float> c = download(dc, one), d = download(dd, all);
        for (size_t i = 0; i < one; ++i) EXPECT(c[i] == ab[i]);
        for (size_t i = 0; i < all; ++i) EXPECT(d[i] == ba[i]);
    }

    // 3. Captured into a graph: the run is a host node, launched with the inputs of the time.
    {
        CHECK_CU(cuStreamBeginCapture(s1, CU_STREAM_CAPTURE_MODE_GLOBAL));
        gemm(h1, da, db, dc);
        CUgraph graph;
        CHECK_CU(cuStreamEndCapture(s1, &graph));
        CUgraphExec exec;
        CHECK_CU(cuGraphInstantiate(&exec, graph, 0));
        for (int seed = 3; seed < 5; ++seed) {
            const std::vector<float> a2 = values(int(one), seed);
            CHECK_CU(cuMemcpyHtoD(da, a2.data(), one * 4));
            CHECK_CU(cuMemsetD32(dc, 0, one));
            CHECK_CU(cuGraphLaunch(exec, s1));
            CHECK_CU(cuStreamSynchronize(s1));
            const std::vector<float> ref = reference(a2, b, 1), c = download(dc, one);
            for (size_t i = 0; i < one; ++i) EXPECT(c[i] == ref[i]);
        }
        CHECK_CU(cuGraphExecDestroy(exec));
        CHECK_CU(cuGraphDestroy(graph));
    }

    // 4. GPU + NPU split (7.5): a share of C's columns (rows, when the columns are too few) on the
    //    NPU, the rest on the GPU meanwhile; general shapes, transposes and beta, against a host GEMM.
    {
        setenv("VULKANIZER_NPU_SPLIT", "0.5", 1);
        struct Case { cublasOperation_t ta, tb; int m, n, k; };
        for (const Case c : {Case{CUBLAS_OP_N, CUBLAS_OP_N, 300, 768, 40}, Case{CUBLAS_OP_T, CUBLAS_OP_N, 768, 300, 40},
                             Case{CUBLAS_OP_N, CUBLAS_OP_T, 520, 600, 33}}) {
            const int lda = (c.ta == CUBLAS_OP_N ? c.m : c.k) + 3, ldb = (c.tb == CUBLAS_OP_N ? c.k : c.n) + 1, ldc = c.m + 2;
            const int colsA = c.ta == CUBLAS_OP_N ? c.k : c.m, colsB = c.tb == CUBLAS_OP_N ? c.n : c.k;
            const std::vector<float> ha = values(lda * colsA, 5), hb = values(ldb * colsB, 6), hc = values(ldc * c.n, 7);
            const CUdeviceptr pa = upload(ha), pb = upload(hb), pc = upload(hc);
            const float alpha = 2.0f, beta = -1.0f;
            const auto t0 = std::chrono::steady_clock::now();
            CHECK_BLAS(cublasGemmEx(h1, c.ta, c.tb, c.m, c.n, c.k, &alpha, reinterpret_cast<void*>(pa), CUDA_R_32F, lda,
                                    reinterpret_cast<void*>(pb), CUDA_R_32F, ldb, &beta, reinterpret_cast<void*>(pc),
                                    CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F_FAST_16BF, CUBLAS_GEMM_DEFAULT));
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            EXPECT(ms < delayMs / 2);
            EXPECT_CU(CUDA_ERROR_NOT_READY, cuStreamQuery(s1));  // the NPU part is running
            CHECK_CU(cuStreamSynchronize(s1));
            const std::vector<float> got = download(pc, size_t(ldc) * c.n);
            for (int j = 0; j < c.n; ++j)
                for (int i = 0; i < c.m; ++i) {
                    float sum = 0;
                    for (int p = 0; p < c.k; ++p) {
                        const float x = c.ta == CUBLAS_OP_N ? ha[size_t(i + p * lda)] : ha[size_t(p + i * lda)];
                        const float y = c.tb == CUBLAS_OP_N ? hb[size_t(p + j * ldb)] : hb[size_t(j + p * ldb)];
                        sum += x * y;
                    }
                    EXPECT(got[size_t(i + j * ldc)] == alpha * sum + beta * hc[size_t(i + j * ldc)]);
                }
            CHECK_CU(cuMemFree(pa));
            CHECK_CU(cuMemFree(pb));
            CHECK_CU(cuMemFree(pc));
        }
        unsetenv("VULKANIZER_NPU_SPLIT");
    }

    CHECK_BLAS(cublasDestroy(h1));
    CHECK_BLAS(cublasDestroy(h2));
    CHECK_CU(cuStreamDestroy(s1));
    CHECK_CU(cuStreamDestroy(s2));
    CHECK_CU(cuCtxDestroy(ctx));
    std::printf("npu_async: OK\n");
    return 0;
}
