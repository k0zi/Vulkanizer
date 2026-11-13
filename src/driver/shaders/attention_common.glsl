// Shared by attention_coopmat.comp and attention_scalar.comp: the launch arguments (driver/attention.cpp
// fills them from PyTorch's AttentionKernel::Params) and the per-row rules of the memory-efficient
// attention forward kernel (masks, bias, logsumexp).

layout(buffer_reference, std430, buffer_reference_align = 8) readonly buffer Args {
    uint64_t q, k, v, bias, o, lse;
    int64_t qStrideB, kStrideB, vStrideB, biasStrideB, biasStrideH;
    uint64_t seqstartQ, seqstartK, seqlenK;
    int qStrideM, kStrideM, vStrideM, biasStrideM, oStrideM, qStrideH, kStrideH, vStrideH;
    int headDim, headDimV, numQueries, numKeys, numHeads, qHeadsPerKv, maskType, causalOffset;
    int windowSize;
    float scale;
    int queriesPerBlock, gridX;  // the CUDA launch: logsumexp padding goes by its blocks
    int slices;                  // workgroups along y per head (attention_scalar.comp: value columns)
    uint flags;                  // kVectorLoads: q, k, v 16-byte aligned, strides and head_dim multiples of 8
};
const uint kVectorLoads = 1u;

layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer I32 { int v; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer F32 { float v; };

layout(push_constant, std430) uniform Push { uint64_t args; } pc;

// The device's address bits 63..47 (vk::Device::addressHighBits), specialization constant 5.
layout(constant_id = 5) const uint kAddressHigh = 0xFFFF8000u;
uint64_t canon(uint64_t a) {
    uvec2 w = unpackUint2x32(a);
    w.y |= kAddressHigh;
    return packUint2x32(w);
}

const float kLog2e = 1.4426950408889634;
const float kInf = 1.0 / 0.0;

// One (batch, head)'s problem, after advance_to_block (kernel_forward.h): element offsets of the
// first query / key / value / output row, the sequence lengths, the causal diagonal.
struct Problem {
    int64_t q, k, v, o, bias;  // element offsets
    int numQ, numK, diag;
    bool valid;
};

Problem problemFor(Args a, int batch, int head) {
    Problem p;
    p.valid = true;
    int64_t qStart = 0, kStart = 0;
    p.numQ = a.numQueries;
    p.numK = a.numKeys;
    const int kvHead = head / a.qHeadsPerKv;
    if (a.seqstartQ != 0ul) {  // variable sequence lengths
        const uint64_t sq = canon(a.seqstartQ) + uint64_t(batch) * 4ul, sk = canon(a.seqstartK) + uint64_t(batch) * 4ul;
        const int q0 = I32(sq).v, q1 = I32(sq + 4ul).v, k0 = I32(sk).v;
        // Loads behind a condition may be made unconditional by the shader compiler: every address
        // loaded from is a valid one (here and in the kernels), the condition picks the value.
        const bool lengths = a.seqlenK != 0ul;
        const int kNext = I32(lengths ? canon(a.seqlenK) + uint64_t(batch) * 4ul : sk + 4ul).v;
        const int k1 = lengths ? k0 + kNext : kNext;
        qStart = q0;
        kStart = k0;
        p.numQ = q1 - q0;
        p.numK = k1 - k0;
        p.q = 0;
        p.k = 0;
        p.v = 0;
        p.o = 0;
    } else {
        p.q = int64_t(batch) * a.qStrideB;
        p.k = int64_t(batch) * a.kStrideB;
        p.v = int64_t(batch) * a.vStrideB;
        p.o = int64_t(batch) * int64_t(a.numQueries) * int64_t(a.oStrideM);
    }
    p.q += qStart * a.qStrideM + int64_t(head) * a.qStrideH;
    p.k += kStart * a.kStrideM + int64_t(kvHead) * a.kStrideH;
    p.v += kStart * a.vStrideM + int64_t(kvHead) * a.vStrideH;
    p.o += qStart * a.oStrideM + int64_t(head) * a.headDimV;
    p.bias = int64_t(batch) * a.biasStrideB + int64_t(head) * a.biasStrideH;
    p.diag = a.maskType == 2 ? p.numK - p.numQ : a.causalOffset;
    return p;
}

// Keys a block of queries [q0, q1) can see: [begin, end).
void keyRange(Args a, Problem p, int q0, int q1, out int begin, out int end) {
    begin = 0;
    end = p.numK;
    if (a.maskType != 0) end = min(end, q1 - 1 + p.diag + 1);
    if (a.windowSize > 0) begin = max(0, q0 + p.diag - a.windowSize + 1);
}

// The masks, read from the arguments once (not in the kernels' loops).
struct Mask {
    int type, window;
};
Mask maskOf(Args a) {
    Mask m;
    m.type = a.maskType;
    m.window = max(a.windowSize, 0);
    return m;
}

// Masked out: -inf.
bool visible(Mask m, Problem p, int query, int key) {
    if (key >= p.numK) return false;
    if (m.type != 0 && key > query + p.diag) return false;
    if (m.window > 0 && key <= query + p.diag - m.window) return false;
    return true;
}

// logsumexp[batch, head, query] (padded to multiples of 32 with +inf per CUDA block, as the kernel does).
void storeLogsumexp(Args a, Problem p, int batch, int head, int query, float m2, float l) {
    if (a.lse == 0ul) return;
    const int block = query / a.queriesPerBlock, t = query % a.queriesPerBlock;
    if (block >= a.gridX) return;
    const int remaining = p.numQ - block * a.queriesPerBlock;
    float value;
    if (t < remaining)
        value = (m2 == -kInf ? 0.0 : m2 / kLog2e) + log(l == 0.0 ? 1.0 : l);
    else if (t < (remaining + 31) / 32 * 32)
        value = kInf;
    else
        return;
    const int lseDim = (a.numQueries + 31) / 32 * 32;
    const int64_t at = (int64_t(batch) * a.numHeads + head) * lseDim + query;
    F32(canon(a.lse) + uint64_t(at) * 4ul).v = value;
}
