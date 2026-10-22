// CUDA graphs through the Driver API: stream capture (kernels, memsets, copies, host functions,
// fork/join across streams), replay, invalidation, executable graph updates.

#include "test_util.h"

#include <cstring>
#include <vector>

namespace {

const char* kPtx = R"(
.version 7.0
.target sm_75
.address_size 64

// y[i] = y[i] * a + b
.visible .entry axpb(.param .u64 py, .param .u32 n, .param .f32 a, .param .f32 b)
{
    .reg .b32 %r<6>;
    .reg .f32 %f<5>;
    .reg .b64 %rd<4>;
    .reg .pred %p;
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.s32 %r4, %r1, %r2, %r3;
    ld.param.u32 %r5, [n];
    setp.ge.u32 %p, %r4, %r5;
    @%p bra DONE;
    ld.param.u64 %rd1, [py];
    mul.wide.u32 %rd2, %r4, 4;
    add.s64 %rd3, %rd1, %rd2;
    ld.global.f32 %f1, [%rd3];
    ld.param.f32 %f2, [a];
    ld.param.f32 %f3, [b];
    fma.rn.f32 %f4, %f1, %f2, %f3;
    st.global.f32 [%rd3], %f4;
DONE:
    ret;
}
)";

constexpr unsigned kN = 1000;
CUfunction g_axpb;

void axpb(CUstream s, CUdeviceptr y, float a, float b) {
    unsigned n = kN;
    void* args[] = {&y, &n, &a, &b};
    CHECK_CU(cuLaunchKernel(g_axpb, (kN + 127) / 128, 1, 1, 128, 1, 1, 0, s, args, nullptr));
}

std::vector<float> download(CUdeviceptr p) {
    std::vector<float> v(kN);
    CHECK_CU(cuMemcpyDtoH(v.data(), p, kN * 4));
    return v;
}

bool all(const std::vector<float>& v, float want) {
    for (float x : v)
        if (x != want) {
            std::fprintf(stderr, "got %g, want %g\n", x, want);
            return false;
        }
    return true;
}

int g_hostCalls = 0;
void countCall(void* p) { ++*static_cast<int*>(p); }

}  // namespace

int main() {
    CHECK_CU(cuInit(0));
    CUdevice dev;
    CHECK_CU(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CHECK_CU(cuCtxCreate(&ctx, 0, dev));
    CUmodule mod;
    CHECK_CU(cuModuleLoadData(&mod, kPtx));
    CHECK_CU(cuModuleGetFunction(&g_axpb, mod, "axpb"));
    CUstream s1, s2;
    CHECK_CU(cuStreamCreate(&s1, 0));
    CHECK_CU(cuStreamCreate(&s2, 0));
    CUdeviceptr y;
    CHECK_CU(cuMemAlloc(&y, kN * 4));
    const float one = 1.0f;
    unsigned oneBits;
    std::memcpy(&oneBits, &one, 4);

    // Capture: memset to 1, then *2 + 1 (= 3); nothing runs until the graph is launched.
    CHECK_CU(cuMemsetD32(y, 0, kN));
    CHECK_CU(cuStreamBeginCapture(s1, CU_STREAM_CAPTURE_MODE_RELAXED));
    CUstreamCaptureStatus status;
    CHECK_CU(cuStreamIsCapturing(s1, &status));
    EXPECT(status == CU_STREAM_CAPTURE_STATUS_ACTIVE);
    CHECK_CU(cuMemsetD32Async(y, oneBits, kN, s1));
    axpb(s1, y, 2.0f, 1.0f);
    cuuint64_t id = 0;
    const CUgraphNode* deps = nullptr;
    size_t numDeps = 0;
    CHECK_CU(cuStreamGetCaptureInfo(s1, &status, &id, nullptr, &deps, &numDeps));
    EXPECT(id != 0 && numDeps == 1);
    CHECK_CU(cuLaunchHostFunc(s1, countCall, &g_hostCalls));
    CUgraph graph;
    CHECK_CU(cuStreamEndCapture(s1, &graph));
    CHECK_CU(cuStreamIsCapturing(s1, &status));
    EXPECT(status == CU_STREAM_CAPTURE_STATUS_NONE);
    EXPECT(all(download(y), 0.0f) && g_hostCalls == 0);
    size_t numNodes = 0;
    CHECK_CU(cuGraphGetNodes(graph, nullptr, &numNodes));
    EXPECT(numNodes == 3);
    std::vector<CUgraphNode> nodes(numNodes);
    CHECK_CU(cuGraphGetNodes(graph, nodes.data(), &numNodes));
    CUgraphNodeType type;
    CHECK_CU(cuGraphNodeGetType(nodes[1], &type));
    EXPECT(type == CU_GRAPH_NODE_TYPE_KERNEL);
    CUDA_KERNEL_NODE_PARAMS kp;
    CHECK_CU(cuGraphKernelNodeGetParams(nodes[1], &kp));
    EXPECT(kp.func == g_axpb && kp.gridDimX == (kN + 127) / 128 && *static_cast<float*>(kp.kernelParams[2]) == 2.0f);

    CUgraphExec exec;
    CHECK_CU(cuGraphInstantiate(&exec, graph, 0));
    CHECK_CU(cuGraphLaunch(exec, s1));
    CHECK_CU(cuStreamSynchronize(s1));
    EXPECT(all(download(y), 3.0f) && g_hostCalls == 1);
    CHECK_CU(cuGraphLaunch(exec, s1));
    CHECK_CU(cuStreamSynchronize(s1));
    EXPECT(all(download(y), 3.0f) && g_hostCalls == 2);  // the memset starts it over

    // The same work with other parameters, taken over by the executable graph: 1 * 5 - 1 = 4.
    CHECK_CU(cuStreamBeginCapture(s1, CU_STREAM_CAPTURE_MODE_GLOBAL));
    CHECK_CU(cuMemsetD32Async(y, oneBits, kN, s1));
    axpb(s1, y, 5.0f, -1.0f);
    CHECK_CU(cuLaunchHostFunc(s1, countCall, &g_hostCalls));
    CUgraph graph2;
    CHECK_CU(cuStreamEndCapture(s1, &graph2));
    CUgraphExecUpdateResultInfo info;
    CHECK_CU(cuGraphExecUpdate(exec, graph2, &info));
    EXPECT(info.result == CU_GRAPH_EXEC_UPDATE_SUCCESS);
    CHECK_CU(cuGraphLaunch(exec, s1));
    CHECK_CU(cuStreamSynchronize(s1));
    EXPECT(all(download(y), 4.0f));

    // A different shape is refused.
    CUgraph empty;
    CHECK_CU(cuGraphCreate(&empty, 0));
    EXPECT_CU(CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE, cuGraphExecUpdate(exec, empty, &info));
    EXPECT(info.result == CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED);

    // Fork/join: s2 joins through an event, works, and s1 waits for it; then a copy to the host.
    std::vector<float> host(kN, -1.0f);
    CUevent fork, join;
    CHECK_CU(cuEventCreate(&fork, CU_EVENT_DISABLE_TIMING));
    CHECK_CU(cuEventCreate(&join, CU_EVENT_DISABLE_TIMING));
    CHECK_CU(cuStreamBeginCapture(s1, CU_STREAM_CAPTURE_MODE_GLOBAL));
    axpb(s1, y, 1.0f, 1.0f);  // 4 -> 5
    CHECK_CU(cuEventRecord(fork, s1));
    CHECK_CU(cuStreamWaitEvent(s2, fork, 0));
    CHECK_CU(cuStreamIsCapturing(s2, &status));
    EXPECT(status == CU_STREAM_CAPTURE_STATUS_ACTIVE);
    axpb(s2, y, 2.0f, 0.0f);  // -> 10
    CHECK_CU(cuEventRecord(join, s2));
    CHECK_CU(cuStreamWaitEvent(s1, join, 0));
    CHECK_CU(cuMemcpyDtoHAsync(host.data(), y, kN * 4, s1));
    CUgraph forked;
    CHECK_CU(cuStreamEndCapture(s1, &forked));
    CHECK_CU(cuStreamIsCapturing(s2, &status));
    EXPECT(status == CU_STREAM_CAPTURE_STATUS_NONE);
    EXPECT(all(host, -1.0f));
    CUgraphExec forkedExec;
    CHECK_CU(cuGraphInstantiate(&forkedExec, forked, 0));
    CHECK_CU(cuGraphLaunch(forkedExec, s1));
    CHECK_CU(cuStreamSynchronize(s1));
    EXPECT(all(host, 10.0f));

    // Synchronizing a capturing stream invalidates the capture.
    CHECK_CU(cuStreamBeginCapture(s1, CU_STREAM_CAPTURE_MODE_GLOBAL));
    axpb(s1, y, 1.0f, 1.0f);
    EXPECT_CU(CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED, cuStreamSynchronize(s1));
    CUgraph broken = reinterpret_cast<CUgraph>(1);
    EXPECT_CU(CUDA_ERROR_STREAM_CAPTURE_INVALIDATED, cuStreamEndCapture(s1, &broken));
    EXPECT(broken == nullptr);
    EXPECT(all(download(y), 10.0f));  // the captured kernel never ran

    // Entry points of a CUDA 12 caller.
    void* fn = nullptr;
    CUdriverProcAddressQueryResult q;
    CHECK_CU(cuGetProcAddress("cuGraphInstantiate", &fn, 12000, CU_GET_PROC_ADDRESS_DEFAULT, &q));
    EXPECT(fn == reinterpret_cast<void*>(&cuGraphInstantiateWithFlags));
    CHECK_CU(cuGetProcAddress("cuStreamBeginCapture", &fn, 12000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM, &q));
    EXPECT(fn == reinterpret_cast<void*>(&cuStreamBeginCapture_v2_ptsz));

    for (CUgraphExec e : {exec, forkedExec}) CHECK_CU(cuGraphExecDestroy(e));
    for (CUgraph g : {graph, graph2, empty, forked}) CHECK_CU(cuGraphDestroy(g));
    CHECK_CU(cuMemFree(y));
    CHECK_CU(cuCtxDestroy(ctx));
    std::printf("OK\n");
    return 0;
}
