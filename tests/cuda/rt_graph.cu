// CUDA graphs through the Runtime API as the CUDA programming guide uses them: a while loop of
// conditional nodes set by cudaGraphSetConditional (body filled by cudaStreamBeginCaptureToGraph),
// an if/else node added in the middle of a stream capture, cudaGraphAddNode with generic node
// parameters (kernel, memset, 1D memcpy), node parameter updates in an executable graph, and user
// objects owned by graphs.
#include <cstdio>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

#define CHECK(call)                                                                            \
    do {                                                                                       \
        cudaError_t e = (call);                                                                \
        if (e != cudaSuccess) {                                                                \
            std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(e)); \
            ++g_failures;                                                                      \
            return false;                                                                      \
        }                                                                                      \
    } while (0)

// CUDA 13 merged the _v2 forms (with edge data) into these names.
#if CUDART_VERSION >= 13000
#define ADD_NODE(node, graph, deps, count, params) cudaGraphAddNode(node, graph, deps, nullptr, count, params)
#define CAPTURE_INFO(s, status, graph, deps, count) cudaStreamGetCaptureInfo(s, status, nullptr, graph, deps, nullptr, count)
#define SET_CAPTURE_DEPS(s, nodes, count) cudaStreamUpdateCaptureDependencies(s, nodes, nullptr, count, cudaStreamSetCaptureDependencies)
#else
#define ADD_NODE(node, graph, deps, count, params) cudaGraphAddNode(node, graph, deps, count, params)
#define CAPTURE_INFO(s, status, graph, deps, count) cudaStreamGetCaptureInfo(s, status, nullptr, graph, deps, count)
#define SET_CAPTURE_DEPS(s, nodes, count) cudaStreamUpdateCaptureDependencies(s, nodes, count, cudaStreamSetCaptureDependencies)
#endif

namespace {

int g_failures = 0;

bool report(const char* name, bool ok) {
    std::printf("%-44s %s\n", name, ok ? "OK" : "FAILED");
    if (!ok) ++g_failures;
    return ok;
}

constexpr int kN = 1024;

__global__ void addTo(float* y, float v) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < kN) y[i] += v;
}

// One loop round: count down, keep going while the counter is positive.
__global__ void loopBody(int* counter, cudaGraphConditionalHandle h) {
    if (threadIdx.x == 0 && blockIdx.x == 0) cudaGraphSetConditional(h, --*counter > 0);
}

__global__ void setCondition(cudaGraphConditionalHandle h, const float* y, float threshold) {
    if (threadIdx.x == 0 && blockIdx.x == 0) cudaGraphSetConditional(h, y[0] > threshold);
}

float first(const float* d) {
    float v = 0;
    cudaMemcpy(&v, d, 4, cudaMemcpyDeviceToHost);
    return v;
}

// While loop: 10 rounds, each adding 0.5.
bool whileLoop(float* y, int* counter) {
    cudaGraph_t graph;
    CHECK(cudaGraphCreate(&graph, 0));
    cudaGraphConditionalHandle handle;
    CHECK(cudaGraphConditionalHandleCreate(&handle, graph, 1, cudaGraphCondAssignDefault));
    cudaGraphNodeParams p = {cudaGraphNodeTypeConditional};
    p.conditional.handle = handle;
    p.conditional.type = cudaGraphCondTypeWhile;
    p.conditional.size = 1;
    cudaGraphNode_t node;
    CHECK(ADD_NODE(&node, graph, nullptr, 0, &p));
    cudaGraph_t body = p.conditional.phGraph_out[0];

    cudaStream_t s;
    CHECK(cudaStreamCreate(&s));
    CHECK(cudaStreamBeginCaptureToGraph(s, body, nullptr, nullptr, 0, cudaStreamCaptureModeRelaxed));
    loopBody<<<1, 1, 0, s>>>(counter, handle);
    addTo<<<kN / 256, 256, 0, s>>>(y, 0.5f);
    CHECK(cudaStreamEndCapture(s, nullptr));

    cudaGraphExec_t exec;
    CHECK(cudaGraphInstantiate(&exec, graph, 0));
    CHECK(cudaMemset(y, 0, kN * 4));
    const int ten = 10;
    CHECK(cudaMemcpy(counter, &ten, 4, cudaMemcpyHostToDevice));
    CHECK(cudaGraphLaunch(exec, s));
    CHECK(cudaStreamSynchronize(s));
    int left = -1;
    CHECK(cudaMemcpy(&left, counter, 4, cudaMemcpyDeviceToHost));
    report("while (10 rounds)", first(y) == 5.0f && left == 0);
    CHECK(cudaGraphExecDestroy(exec));
    CHECK(cudaGraphDestroy(graph));
    CHECK(cudaStreamDestroy(s));
    return true;
}

// If/else added during a capture: y[0] > 1 ? y += 10 : y += 100.
bool ifElseInCapture(float* y) {
    cudaStream_t s;
    CHECK(cudaStreamCreate(&s));
    cudaGraph_t graph;
    CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    cudaStreamCaptureStatus status;
    const cudaGraphNode_t* deps;
    size_t numDeps;
    CHECK(CAPTURE_INFO(s, &status, &graph, &deps, &numDeps));
    cudaGraphConditionalHandle handle;
    CHECK(cudaGraphConditionalHandleCreate(&handle, graph, 0, 0));
    setCondition<<<1, 1, 0, s>>>(handle, y, 1.0f);
    CHECK(CAPTURE_INFO(s, &status, &graph, &deps, &numDeps));
    cudaGraphNodeParams p = {cudaGraphNodeTypeConditional};
    p.conditional.handle = handle;
    p.conditional.type = cudaGraphCondTypeIf;
    p.conditional.size = 2;
    cudaGraphNode_t node;
    CHECK(ADD_NODE(&node, graph, deps, numDeps, &p));
    CHECK(SET_CAPTURE_DEPS(s, &node, 1));
    for (int i = 0; i < 2; ++i) {
        cudaStream_t b;
        CHECK(cudaStreamCreate(&b));
        CHECK(cudaStreamBeginCaptureToGraph(b, p.conditional.phGraph_out[i], nullptr, nullptr, 0,
                                            cudaStreamCaptureModeRelaxed));
        addTo<<<kN / 256, 256, 0, b>>>(y, i == 0 ? 10.0f : 100.0f);
        CHECK(cudaStreamEndCapture(b, nullptr));
        CHECK(cudaStreamDestroy(b));
    }
    addTo<<<kN / 256, 256, 0, s>>>(y, 1.0f);  // after the conditional node
    CHECK(cudaStreamEndCapture(s, &graph));

    cudaGraphExec_t exec;
    CHECK(cudaGraphInstantiate(&exec, graph, 0));
    CHECK(cudaMemset(y, 0, kN * 4));
    CHECK(cudaGraphLaunch(exec, s));  // 0: else, +100, +1
    CHECK(cudaGraphLaunch(exec, s));  // 101: then, +10, +1
    CHECK(cudaStreamSynchronize(s));
    report("if/else in a capture", first(y) == 112.0f);
    CHECK(cudaGraphExecDestroy(exec));
    CHECK(cudaGraphDestroy(graph));
    CHECK(cudaStreamDestroy(s));
    return true;
}

int g_destroyed = 0;
void destroyed(void*) { ++g_destroyed; }

// cudaGraphAddNode: memset (y = 2), kernel (+3), memcpy to z; then the kernel's value changed in the
// executable graph; a user object lives as long as the graphs holding it.
bool genericNodes(float* y, float* z) {
    cudaGraph_t graph;
    CHECK(cudaGraphCreate(&graph, 0));
    cudaGraphNodeParams p = {cudaGraphNodeTypeMemset};
    const float two = 2.0f;
    p.memset.dst = y;
    std::memcpy(&p.memset.value, &two, 4);
    p.memset.elementSize = 4;
    p.memset.width = kN;
    p.memset.height = 1;
    cudaGraphNode_t memset, kernel, copy;
    CHECK(ADD_NODE(&memset, graph, nullptr, 0, &p));
    float v = 3.0f;
    void* args[] = {&y, &v};
    p = {cudaGraphNodeTypeKernel};
    p.kernel.func = reinterpret_cast<void*>(addTo);
    p.kernel.gridDim = dim3(kN / 256);
    p.kernel.blockDim = dim3(256);
    p.kernel.kernelParams = args;
    CHECK(ADD_NODE(&kernel, graph, &memset, 1, &p));
    CHECK(cudaGraphAddMemcpyNode1D(&copy, graph, &kernel, 1, z, y, kN * 4, cudaMemcpyDeviceToDevice));

    cudaUserObject_t object;
    CHECK(cudaUserObjectCreate(&object, nullptr, destroyed, 1, cudaUserObjectNoDestructorSync));
    CHECK(cudaGraphRetainUserObject(graph, object, 1, cudaGraphUserObjectMove));

    cudaGraphExec_t exec;
    CHECK(cudaGraphInstantiate(&exec, graph, 0));
    CHECK(cudaGraphLaunch(exec, 0));
    CHECK(cudaDeviceSynchronize());
    const bool first5 = first(z) == 5.0f;
    v = 7.0f;
    CHECK(cudaGraphExecNodeSetParams(exec, kernel, &p));
    CHECK(cudaGraphLaunch(exec, 0));
    CHECK(cudaDeviceSynchronize());
    cudaGraphNodeType type;
    CHECK(cudaGraphNodeGetType(copy, &type));
    CHECK(cudaGraphDestroy(graph));
    const bool alive = g_destroyed == 0;
    CHECK(cudaGraphExecDestroy(exec));
    report("generic nodes, exec node update", first5 && first(z) == 9.0f && type == cudaGraphNodeTypeMemcpy);
    report("user object", alive && g_destroyed == 1);
    return true;
}

}  // namespace

int main() {
    float *y, *z;
    int* counter;
    if (cudaMalloc(&y, kN * 4) || cudaMalloc(&z, kN * 4) || cudaMalloc(&counter, 4)) return 1;
    whileLoop(y, counter);
    ifElseInCapture(y);
    genericNodes(y, z);
    cudaFree(y);
    cudaFree(z);
    cudaFree(counter);
    if (g_failures) {
        std::printf("%d FAILED\n", g_failures);
        return 1;
    }
    std::printf("ALL OK\n");
    return 0;
}
