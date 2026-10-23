// CUDA graphs through the Driver API, beyond capture: conditional nodes (if/else, while, switch,
// nested, default values, bodies filled by capture), cuGraphAddNode's generic parameters, node
// parameter updates in executable graphs, edges added in any order, cloning, user objects, and
// 2D copies captured as memcpy nodes.

#include "test_util.h"

#include <cstring>
#include <vector>

namespace {

// cudaGraphSetConditional as nvcc emits it: an .extern function the driver provides.
const char* kPtx = R"(
.version 7.0
.target sm_75
.address_size 64

.extern .func cudaGraphSetConditional
(
    .param .b64 cudaGraphSetConditional_param_0,
    .param .b32 cudaGraphSetConditional_param_1
)
;

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

// cudaGraphSetConditional(h, v)
.visible .entry setcond(.param .u64 h, .param .u32 v)
{
    .reg .b64 %rd1;
    .reg .b32 %r1;
    ld.param.u64 %rd1, [h];
    ld.param.u32 %r1, [v];
    {
    .param .b64 param0;
    st.param.b64 [param0+0], %rd1;
    .param .b32 param1;
    st.param.b32 [param1+0], %r1;
    call.uni cudaGraphSetConditional, (param0, param1);
    }
    ret;
}

// --*counter; cudaGraphSetConditional(h, *counter > 0)
.visible .entry countdown(.param .u64 pc, .param .u64 h)
{
    .reg .b64 %rd<3>;
    .reg .b32 %r<4>;
    .reg .pred %p;
    ld.param.u64 %rd1, [pc];
    ld.param.u64 %rd2, [h];
    ld.global.u32 %r1, [%rd1];
    sub.s32 %r2, %r1, 1;
    st.global.u32 [%rd1], %r2;
    setp.gt.s32 %p, %r2, 0;
    selp.u32 %r3, 1, 0, %p;
    {
    .param .b64 param0;
    st.param.b64 [param0+0], %rd2;
    .param .b32 param1;
    st.param.b32 [param1+0], %r3;
    call.uni cudaGraphSetConditional, (param0, param1);
    }
    ret;
}
)";

constexpr unsigned kN = 256;
CUfunction g_axpb, g_setcond, g_countdown;
CUdeviceptr g_y;

// Kernel node parameters with their own argument storage.
struct Kernel {
    CUdeviceptr ptr = 0, ptr2 = 0;
    unsigned n = kN, v = 0;
    float a = 1, b = 0;
    void* args[4] = {};
    CUgraphNodeParams params{};

    static Kernel axpb(float a, float b) {
        Kernel k;
        k.ptr = g_y, k.a = a, k.b = b;
        return k;
    }
    // Pointers into this object: call once it sits where it stays.
    CUgraphNodeParams* make(CUfunction f) {
        params = {};
        params.type = CU_GRAPH_NODE_TYPE_KERNEL;
        params.kernel.func = f;
        params.kernel.gridDimX = params.kernel.gridDimY = params.kernel.gridDimZ = 1;
        params.kernel.blockDimX = f == g_axpb ? kN : 1;
        params.kernel.blockDimY = params.kernel.blockDimZ = 1;
        if (f == g_axpb) args[0] = &ptr, args[1] = &n, args[2] = &a, args[3] = &b;
        if (f == g_setcond) args[0] = &ptr, args[1] = &v;
        if (f == g_countdown) args[0] = &ptr, args[1] = &ptr2;
        params.kernel.kernelParams = args;
        return &params;
    }
};

CUgraphNode add(CUgraph g, CUgraphNodeParams* p, std::vector<CUgraphNode> deps = {}) {
    CUgraphNode n;
    CHECK_CU(cuGraphAddNode_v2(&n, g, deps.data(), nullptr, deps.size(), p));
    return n;
}

CUgraphNode addKernel(CUgraph g, Kernel& k, CUfunction f, std::vector<CUgraphNode> deps = {}) {
    return add(g, k.make(f), std::move(deps));
}

// A conditional node of `size` bodies; their graphs in `bodies`.
CUgraphNode addConditional(CUgraph g, CUgraphConditionalHandle h, CUgraphConditionalNodeType type, unsigned size,
                           std::vector<CUgraph>& bodies, std::vector<CUgraphNode> deps = {}) {
    CUgraphNodeParams p{};
    p.type = CU_GRAPH_NODE_TYPE_CONDITIONAL;
    p.conditional.handle = h;
    p.conditional.type = type;
    p.conditional.size = size;
    CUgraphNode n = add(g, &p, std::move(deps));
    bodies.assign(p.conditional.phGraph_out, p.conditional.phGraph_out + size);  // the node's own array
    return n;
}

float y0() {
    float v;
    CHECK_CU(cuMemcpyDtoH(&v, g_y, 4));
    return v;
}

void setY(float v) {
    unsigned bits;
    std::memcpy(&bits, &v, 4);
    CHECK_CU(cuMemsetD32(g_y, bits, kN));
}

void run(CUgraphExec e, CUstream s) {
    CHECK_CU(cuGraphLaunch(e, s));
    CHECK_CU(cuStreamSynchronize(s));
}

int g_destroyed = 0;
void destroyed(void*) { ++g_destroyed; }
int g_hostCalls = 0;
void hostCall(void* p) { ++*static_cast<int*>(p); }

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
    CHECK_CU(cuModuleGetFunction(&g_setcond, mod, "setcond"));
    CHECK_CU(cuModuleGetFunction(&g_countdown, mod, "countdown"));
    CUstream s;
    CHECK_CU(cuStreamCreate(&s, 0));
    CHECK_CU(cuMemAlloc(&g_y, kN * 4));
    CUdeviceptr counter;
    CHECK_CU(cuMemAlloc(&counter, 4));

    {  // while: count down from 5, adding 1 each round; the default (1) starts it on every launch
        CUgraph g;
        CHECK_CU(cuGraphCreate(&g, 0));
        CUgraphConditionalHandle h;
        CHECK_CU(cuGraphConditionalHandleCreate(&h, g, ctx, 1, CU_GRAPH_COND_ASSIGN_DEFAULT));
        std::vector<CUgraph> bodies;
        addConditional(g, h, CU_GRAPH_COND_TYPE_WHILE, 1, bodies);
        Kernel down, plus = Kernel::axpb(1, 1);
        down.ptr = counter, down.ptr2 = h;
        CUgraphNode d = addKernel(bodies[0], down, g_countdown);
        addKernel(bodies[0], plus, g_axpb, {d});
        CUgraphExec e;
        CHECK_CU(cuGraphInstantiate(&e, g, 0));
        setY(0);
        CHECK_CU(cuMemsetD32(counter, 5, 1));
        run(e, s);
        unsigned c;
        CHECK_CU(cuMemcpyDtoH(&c, counter, 4));
        EXPECT(y0() == 5 && c == 0);
        CHECK_CU(cuMemsetD32(counter, 3, 1));
        run(e, s);
        EXPECT(y0() == 8);
        CHECK_CU(cuGraphExecDestroy(e));
        CHECK_CU(cuGraphDestroy(g));
        std::printf("while: OK\n");
    }

    {  // if/else, the condition set by a kernel whose parameter changes in the executable graph
        CUgraph g;
        CHECK_CU(cuGraphCreate(&g, 0));
        CUgraphConditionalHandle h;
        CHECK_CU(cuGraphConditionalHandleCreate(&h, g, ctx, 0, 0));
        Kernel set, then = Kernel::axpb(1, 10), otherwise = Kernel::axpb(1, 100);
        set.ptr = h, set.v = 1;
        CUgraphNode setNode = addKernel(g, set, g_setcond);
        std::vector<CUgraph> bodies;
        addConditional(g, h, CU_GRAPH_COND_TYPE_IF, 2, bodies, {setNode});
        addKernel(bodies[0], then, g_axpb);
        addKernel(bodies[1], otherwise, g_axpb);
        CUgraphExec e;
        CHECK_CU(cuGraphInstantiate(&e, g, 0));
        setY(0);
        run(e, s);
        EXPECT(y0() == 10);
        set.v = 0;
        CHECK_CU(cuGraphExecNodeSetParams(e, setNode, set.make(g_setcond)));
        run(e, s);
        EXPECT(y0() == 110);
        CHECK_CU(cuGraphExecDestroy(e));
        CHECK_CU(cuGraphDestroy(g));
        std::printf("if/else: OK\n");
    }

    {  // switch over 3 bodies: value 2, then one out of range; a nested while in body 2
        CUgraph g;
        CHECK_CU(cuGraphCreate(&g, 0));
        CUgraphConditionalHandle h;
        CHECK_CU(cuGraphConditionalHandleCreate(&h, g, ctx, 0, 0));
        Kernel set;
        set.ptr = h, set.v = 2;
        CUgraphNode setNode = addKernel(g, set, g_setcond);
        std::vector<CUgraph> bodies;
        addConditional(g, h, CU_GRAPH_COND_TYPE_SWITCH, 3, bodies, {setNode});
        Kernel k0 = Kernel::axpb(1, 1), k1 = Kernel::axpb(1, 2);
        addKernel(bodies[0], k0, g_axpb);
        addKernel(bodies[1], k1, g_axpb);
        CUgraphConditionalHandle inner;  // body 2: count down from 4, adding 1000
        CHECK_CU(cuGraphConditionalHandleCreate(&inner, bodies[2], ctx, 1, CU_GRAPH_COND_ASSIGN_DEFAULT));
        std::vector<CUgraph> loop;
        addConditional(bodies[2], inner, CU_GRAPH_COND_TYPE_WHILE, 1, loop);
        Kernel down, plus = Kernel::axpb(1, 1000);
        down.ptr = counter, down.ptr2 = inner;
        CUgraphNode d = addKernel(loop[0], down, g_countdown);
        addKernel(loop[0], plus, g_axpb, {d});
        CUgraphExec e;
        CHECK_CU(cuGraphInstantiate(&e, g, 0));
        setY(0);
        CHECK_CU(cuMemsetD32(counter, 4, 1));
        run(e, s);
        EXPECT(y0() == 4000);
        set.v = 7;
        CHECK_CU(cuGraphExecNodeSetParams(e, setNode, set.make(g_setcond)));
        run(e, s);
        EXPECT(y0() == 4000);
        CHECK_CU(cuGraphExecDestroy(e));
        CHECK_CU(cuGraphDestroy(g));
        std::printf("switch, nested while: OK\n");
    }

    {  // a body filled by stream capture; capture also into an existing graph after a node
        CUgraph g;
        CHECK_CU(cuGraphCreate(&g, 0));
        CUgraphConditionalHandle h;
        CHECK_CU(cuGraphConditionalHandleCreate(&h, g, ctx, 1, CU_GRAPH_COND_ASSIGN_DEFAULT));
        std::vector<CUgraph> bodies;
        CUgraphNode cond = addConditional(g, h, CU_GRAPH_COND_TYPE_IF, 1, bodies);
        CHECK_CU(cuStreamBeginCaptureToGraph(s, bodies[0], nullptr, nullptr, 0, CU_STREAM_CAPTURE_MODE_GLOBAL));
        Kernel k = Kernel::axpb(3, 0);
        CHECK_CU(cuLaunchKernel(g_axpb, 1, 1, 1, kN, 1, 1, 0, s, k.make(g_axpb)->kernel.kernelParams, nullptr));
        CUgraph captured;
        CHECK_CU(cuStreamEndCapture(s, &captured));
        EXPECT(captured == bodies[0]);
        CHECK_CU(cuStreamBeginCaptureToGraph(s, g, &cond, nullptr, 1, CU_STREAM_CAPTURE_MODE_GLOBAL));
        Kernel k2 = Kernel::axpb(1, 1);
        CHECK_CU(cuLaunchKernel(g_axpb, 1, 1, 1, kN, 1, 1, 0, s, k2.make(g_axpb)->kernel.kernelParams, nullptr));
        CHECK_CU(cuStreamEndCapture(s, &captured));
        EXPECT(captured == g);
        size_t n = 0;
        CHECK_CU(cuGraphGetNodes(g, nullptr, &n));
        EXPECT(n == 2);
        CUgraphExec e;
        CHECK_CU(cuGraphInstantiate(&e, g, 0));
        setY(2);
        run(e, s);
        EXPECT(y0() == 7);
        CHECK_CU(cuGraphExecDestroy(e));
        CHECK_CU(cuGraphDestroy(g));
        std::printf("capture into bodies: OK\n");
    }

    {  // generic nodes, edges added backwards, edge queries, clone, user objects
        CUgraph g;
        CHECK_CU(cuGraphCreate(&g, 0));
        Kernel twice = Kernel::axpb(2, 0);
        CUgraphNode kernel = addKernel(g, twice, g_axpb);  // added first, runs after the memset
        CUgraphNodeParams p{};
        p.type = CU_GRAPH_NODE_TYPE_MEMSET;
        const float three = 3;
        std::memcpy(&p.memset.value, &three, 4);
        p.memset.dst = g_y, p.memset.elementSize = 4, p.memset.width = kN, p.memset.height = 1;
        CUgraphNode memset = add(g, &p);
        CHECK_CU(cuGraphAddDependencies(g, &memset, &kernel, 1));
        CUdeviceptr copy;
        CHECK_CU(cuMemAlloc(&copy, kN * 4));
        p = {};
        p.type = CU_GRAPH_NODE_TYPE_MEMCPY;
        p.memcpy.copyParams.srcMemoryType = p.memcpy.copyParams.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        p.memcpy.copyParams.srcDevice = g_y, p.memcpy.copyParams.dstDevice = copy;
        p.memcpy.copyParams.WidthInBytes = kN * 4, p.memcpy.copyParams.Height = p.memcpy.copyParams.Depth = 1;
        CUgraphNode memcpy = add(g, &p, {kernel});
        p = {};
        p.type = CU_GRAPH_NODE_TYPE_HOST;
        p.host.fn = hostCall, p.host.userData = &g_hostCalls;
        CUgraphNode host = add(g, &p, {memcpy});
        CUevent ev;
        CHECK_CU(cuEventCreate(&ev, 0));
        p = {};
        p.type = CU_GRAPH_NODE_TYPE_EVENT_RECORD;
        p.eventRecord.event = ev;
        add(g, &p, {host});
        p = {};
        p.type = CU_GRAPH_NODE_TYPE_EMPTY;
        CUgraphNode empty = add(g, &p);

        size_t n = 0;
        CHECK_CU(cuGraphGetRootNodes(g, nullptr, &n));
        EXPECT(n == 2);  // memset, empty
        CHECK_CU(cuGraphGetEdges(g, nullptr, nullptr, &n));
        EXPECT(n == 4);
        CHECK_CU(cuGraphNodeGetDependentNodes(kernel, nullptr, &n));
        EXPECT(n == 1);
        CHECK_CU(cuGraphDestroyNode(empty));
        CHECK_CU(cuGraphGetRootNodes(g, nullptr, &n));
        EXPECT(n == 1);

        CUuserObject obj;
        CHECK_CU(cuUserObjectCreate(&obj, nullptr, destroyed, 1, CU_USER_OBJECT_NO_DESTRUCTOR_SYNC));
        CHECK_CU(cuGraphRetainUserObject(g, obj, 1, CU_GRAPH_USER_OBJECT_MOVE));

        CUgraph clone;
        CHECK_CU(cuGraphClone(&clone, g));
        CUgraphNode cloneKernel;
        CHECK_CU(cuGraphNodeFindInClone(&cloneKernel, kernel, clone));
        Kernel fourTimes = Kernel::axpb(4, 0);
        CHECK_CU(cuGraphNodeSetParams(cloneKernel, fourTimes.make(g_axpb)));

        CUgraphExec e, ce;
        CHECK_CU(cuGraphInstantiate(&e, g, 0));
        CHECK_CU(cuGraphInstantiate(&ce, clone, 0));
        setY(0);
        run(e, s);
        float c;
        CHECK_CU(cuMemcpyDtoH(&c, copy, 4));
        EXPECT(y0() == 6 && c == 6 && g_hostCalls == 1 && cuEventQuery(ev) == CUDA_SUCCESS);
        run(ce, s);
        EXPECT(y0() == 12 && g_hostCalls == 2);

        CHECK_CU(cuGraphDestroy(g));
        CHECK_CU(cuGraphDestroy(clone));
        CHECK_CU(cuGraphExecDestroy(ce));
        EXPECT(g_destroyed == 0);  // the executable graph still holds it
        CHECK_CU(cuGraphExecDestroy(e));
        EXPECT(g_destroyed == 1);
        CHECK_CU(cuEventDestroy(ev));
        CHECK_CU(cuMemFree(copy));
        std::printf("generic nodes, edges, clone, user objects: OK\n");
    }

    {  // a 2D copy on a capturing stream becomes a memcpy node
        CUdeviceptr src;
        CHECK_CU(cuMemAlloc(&src, kN * 4));
        CHECK_CU(cuMemsetD32(src, 7, kN));
        CHECK_CU(cuMemsetD32(g_y, 0, kN));
        CHECK_CU(cuStreamBeginCapture(s, CU_STREAM_CAPTURE_MODE_GLOBAL));
        CUDA_MEMCPY2D c{};
        c.srcMemoryType = c.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        c.srcDevice = src, c.dstDevice = g_y;
        c.srcPitch = c.dstPitch = 64;
        c.WidthInBytes = 32, c.Height = kN * 4 / 64;
        CHECK_CU(cuMemcpy2DAsync(&c, s));
        CUgraph g;
        CHECK_CU(cuStreamEndCapture(s, &g));
        std::vector<unsigned> v(kN);
        CHECK_CU(cuMemcpyDtoH(v.data(), g_y, kN * 4));
        EXPECT(v[0] == 0);
        CUgraphExec e;
        CHECK_CU(cuGraphInstantiate(&e, g, 0));
        run(e, s);
        CHECK_CU(cuMemcpyDtoH(v.data(), g_y, kN * 4));
        EXPECT(v[0] == 7 && v[7] == 7 && v[8] == 0 && v[16] == 7);
        CHECK_CU(cuGraphExecDestroy(e));
        CHECK_CU(cuGraphDestroy(g));
        CHECK_CU(cuMemFree(src));
        std::printf("captured 2D copy: OK\n");
    }

    CHECK_CU(cuMemFree(counter));
    CHECK_CU(cuMemFree(g_y));
    CHECK_CU(cuCtxDestroy(ctx));
    std::printf("OK\n");
    return 0;
}
