// Warp-level matrix instructions (mma.sync, ldmatrix) as Triton emits them for tl.dot, against
// the fragment layouts of the PTX ISA computed on the host; warp reductions (redux.sync). The sm_80
// instructions of PyTorch's attention kernels: tf32 mma, cp.async, packed f16x2/bf16x2 cvt.

#include "test_util.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

const char* kPtx = R"(
.version 7.0
.target sm_75
.address_size 64

// d = a * b + c for one warp; a, b, c, d hold each lane's fragment registers in lane order.
.visible .entry mma_k16(.param .u64 pa, .param .u64 pb, .param .u64 pc, .param .u64 pd)
{
    .reg .b32 %a<4>, %b<2>, %t;
    .reg .f32 %c<4>, %d<4>;
    .reg .b64 %rd<12>;
    mov.u32 %t, %tid.x;
    ld.param.u64 %rd0, [pa];
    ld.param.u64 %rd1, [pb];
    ld.param.u64 %rd2, [pc];
    ld.param.u64 %rd3, [pd];
    mul.wide.u32 %rd4, %t, 16;
    mul.wide.u32 %rd5, %t, 8;
    add.u64 %rd6, %rd0, %rd4;
    add.u64 %rd7, %rd1, %rd5;
    add.u64 %rd8, %rd2, %rd4;
    add.u64 %rd9, %rd3, %rd4;
    ld.global.v4.b32 {%a0, %a1, %a2, %a3}, [%rd6];
    ld.global.v2.b32 {%b0, %b1}, [%rd7];
    ld.global.v4.f32 {%c0, %c1, %c2, %c3}, [%rd8];
    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%d0, %d1, %d2, %d3}, {%a0, %a1, %a2, %a3}, {%b0, %b1}, {%c0, %c1, %c2, %c3};
    st.global.v4.f32 [%rd9], {%d0, %d1, %d2, %d3};
    ret;
}

.visible .entry mma_tf32(.param .u64 pa, .param .u64 pb, .param .u64 pc, .param .u64 pd)
{
    .reg .b32 %a<4>, %b<2>, %t;
    .reg .f32 %c<4>, %d<4>;
    .reg .b64 %rd<12>;
    mov.u32 %t, %tid.x;
    ld.param.u64 %rd0, [pa];
    ld.param.u64 %rd1, [pb];
    ld.param.u64 %rd2, [pc];
    ld.param.u64 %rd3, [pd];
    mul.wide.u32 %rd4, %t, 16;
    mul.wide.u32 %rd5, %t, 8;
    add.u64 %rd6, %rd0, %rd4;
    add.u64 %rd7, %rd1, %rd5;
    add.u64 %rd8, %rd2, %rd4;
    add.u64 %rd9, %rd3, %rd4;
    ld.global.v4.b32 {%a0, %a1, %a2, %a3}, [%rd6];
    ld.global.v2.b32 {%b0, %b1}, [%rd7];
    ld.global.v4.f32 {%c0, %c1, %c2, %c3}, [%rd8];
    mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%d0, %d1, %d2, %d3}, {%a0, %a1, %a2, %a3}, {%b0, %b1}, {%c0, %c1, %c2, %c3};
    st.global.v4.f32 [%rd9], {%d0, %d1, %d2, %d3};
    ret;
}

// Lane t copies 16 bytes with src-size min(t, 16), and 8 bytes that odd lanes ignore (ignore-src);
// out[32 t ..] = the 16 bytes, out[32 t + 16 ..] = the 8.
.visible .entry cp_async(.param .u64 pin, .param .u64 pout)
{
    .reg .b32 %t, %s, %sa, %sb, %w<6>;
    .reg .b64 %rd<8>;
    .reg .pred %p;
    .shared .align 16 .b8 buf[768];
    mov.u32 %t, %tid.x;
    ld.param.u64 %rd0, [pin];
    ld.param.u64 %rd1, [pout];
    cvta.to.global.u64 %rd0, %rd0;
    mul.wide.u32 %rd2, %t, 16;
    add.u64 %rd3, %rd0, %rd2;
    mov.u32 %sa, buf;
    mad.lo.u32 %sa, %t, 16, %sa;
    min.u32 %s, %t, 16;
    cp.async.cg.shared.global.L2::128B [%sa], [%rd3], 16, %s;
    and.b32 %w0, %t, 1;
    setp.ne.u32 %p, %w0, 0;
    mov.u32 %sb, buf;
    add.u32 %sb, %sb, 512;
    mad.lo.u32 %sb, %t, 8, %sb;
    mul.wide.u32 %rd4, %t, 8;
    add.u64 %rd4, %rd0, %rd4;
    cp.async.ca.shared.global [%sb], [%rd4], 8, %p;
    cp.async.commit_group;
    cp.async.wait_group 0;
    bar.sync 0;
    ld.shared.v4.b32 {%w0, %w1, %w2, %w3}, [%sa];
    ld.shared.v2.b32 {%w4, %w5}, [%sb];
    mul.wide.u32 %rd5, %t, 32;
    add.u64 %rd6, %rd1, %rd5;
    st.global.v4.b32 [%rd6], {%w0, %w1, %w2, %w3};
    st.global.v2.b32 [%rd6+16], {%w4, %w5};
    ret;
}

// in: two floats per lane; out: f16x2, bf16x2 and relu bf16x2 of them, then padding
.visible .entry cvt_packed(.param .u64 pin, .param .u64 pout)
{
    .reg .b32 %t, %r<3>;
    .reg .f32 %a, %b;
    .reg .b64 %rd<6>;
    mov.u32 %t, %tid.x;
    ld.param.u64 %rd0, [pin];
    ld.param.u64 %rd1, [pout];
    mul.wide.u32 %rd2, %t, 8;
    add.u64 %rd3, %rd0, %rd2;
    ld.global.v2.f32 {%a, %b}, [%rd3];
    cvt.rn.f16x2.f32 %r0, %a, %b;
    cvt.rn.bf16x2.f32 %r1, %a, %b;
    cvt.rn.relu.bf16x2.f32 %r2, %a, %b;
    mul.wide.u32 %rd4, %t, 16;
    add.u64 %rd5, %rd1, %rd4;
    st.global.v4.b32 [%rd5], {%r0, %r1, %r2, %r2};
    ret;
}

.visible .entry mma_k8(.param .u64 pa, .param .u64 pb, .param .u64 pc, .param .u64 pd)
{
    .reg .b32 %a<2>, %b<1>, %t, %h<2>;
    .reg .f32 %c<4>;
    .reg .b64 %rd<12>;
    mov.u32 %t, %tid.x;
    ld.param.u64 %rd0, [pa];
    ld.param.u64 %rd1, [pb];
    ld.param.u64 %rd2, [pc];
    ld.param.u64 %rd3, [pd];
    mul.wide.u32 %rd4, %t, 8;
    mul.wide.u32 %rd5, %t, 4;
    mul.wide.u32 %rd10, %t, 16;
    add.u64 %rd6, %rd0, %rd4;
    add.u64 %rd7, %rd1, %rd5;
    add.u64 %rd8, %rd2, %rd10;
    add.u64 %rd9, %rd3, %rd4;
    ld.global.v2.b32 {%a0, %a1}, [%rd6];
    ld.global.b32 %b0, [%rd7];
    ld.global.v4.f32 {%c0, %c1, %c2, %c3}, [%rd8];
    // f16 result: two f16x2 registers per lane
    mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f32 {%h0, %h1}, {%a0, %a1}, {%b0}, {%c0, %c1, %c2, %c3};
    st.global.v2.b32 [%rd9], {%h0, %h1};
    ret;
}

// Four 8x8 b16 matrices (rows of 16 bytes, 32 bytes apart) through shared memory: out holds
// each lane's ldmatrix.x4 registers, then its ldmatrix.x2.trans registers.
.visible .entry ldmatrix(.param .u64 pin, .param .u64 pout)
{
    .shared .align 16 .b8 smem[1024];
    .reg .b32 %t, %w, %s, %addr, %j, %r, %d<6>;
    .reg .b64 %rd<8>;
    mov.u32 %t, %tid.x;
    ld.param.u64 %rd0, [pin];
    ld.param.u64 %rd1, [pout];
    mov.u32 %s, smem;
    // copy the 32 rows (4 words each): thread t copies row t
    mul.wide.u32 %rd2, %t, 16;
    add.u64 %rd3, %rd0, %rd2;
    ld.global.v4.b32 {%d0, %d1, %d2, %d3}, [%rd3];
    mad.lo.u32 %addr, %t, 32, %s;
    st.shared.v2.b32 [%addr], {%d0, %d1};
    st.shared.v2.b32 [%addr+8], {%d2, %d3};
    bar.sync 0;
    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%d0, %d1, %d2, %d3}, [%addr];
    ldmatrix.sync.aligned.m8n8.x2.trans.shared::cta.b16 {%d4, %d5}, [%addr];
    mul.wide.u32 %rd2, %t, 24;
    add.u64 %rd3, %rd1, %rd2;
    st.global.v4.b32 [%rd3], {%d0, %d1, %d2, %d3};
    st.global.v2.b32 [%rd3+16], {%d4, %d5};
    ret;
}

// out[lane * 8 + i]: the six reductions over all lanes, then add.s32 and min.u32 over the
// lanes in 0x00F0F00F, which are the only ones executing them (the rest keep 0).
.visible .entry redux(.param .u64 pin, .param .u64 pout)
{
    .reg .b32 %t, %v, %m, %d<8>;
    .reg .b64 %rd<4>;
    .reg .pred %p;
    mov.u32 %t, %tid.x;
    ld.param.u64 %rd0, [pin];
    ld.param.u64 %rd1, [pout];
    mul.wide.u32 %rd2, %t, 4;
    add.u64 %rd2, %rd0, %rd2;
    ld.global.b32 %v, [%rd2];
    redux.sync.add.s32 %d0, %v, -1;
    redux.sync.min.s32 %d1, %v, -1;
    redux.sync.max.u32 %d2, %v, -1;
    redux.sync.and.b32 %d3, %v, -1;
    redux.sync.or.b32 %d4, %v, -1;
    redux.sync.xor.b32 %d5, %v, -1;
    mov.u32 %d6, 0;
    mov.u32 %d7, 0;
    shr.u32 %m, 0x00F0F00F, %t;
    and.b32 %m, %m, 1;
    setp.ne.u32 %p, %m, 0;
    @%p redux.sync.add.s32 %d6, %v, 0x00F0F00F;
    @%p redux.sync.min.u32 %d7, %v, 0x00F0F00F;
    mul.wide.u32 %rd2, %t, 32;
    add.u64 %rd3, %rd1, %rd2;
    st.global.v4.b32 [%rd3], {%d0, %d1, %d2, %d3};
    st.global.v4.b32 [%rd3+16], {%d4, %d5, %d6, %d7};
    ret;
}
)";

uint16_t toHalf(float f) {  // exact for the small values used here
    if (f == 0) return std::signbit(f) ? 0x8000 : 0;
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t sign = (u >> 16) & 0x8000, exp = (u >> 23) & 0xFF, mant = u & 0x7FFFFF;
    return uint16_t(sign | ((exp - 127 + 15) << 10) | (mant >> 13));
}
float fromHalf(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000) << 16, exp = (h >> 10) & 0x1F, mant = h & 0x3FF;
    if (exp == 0 && mant == 0) {
        float z;
        std::memcpy(&z, &sign, 4);
        return z;
    }
    const uint32_t u = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

CUdeviceptr upload(const void* data, size_t bytes) {
    CUdeviceptr p;
    CHECK_CU(cuMemAlloc(&p, bytes));
    CHECK_CU(cuMemcpyHtoD(p, data, bytes));
    return p;
}

int g_failures = 0;
#define CHECK_EQ(what, got, want, ...)                                    \
    do {                                                                  \
        if ((got) != (want)) {                                            \
            std::fprintf(stderr, what ": got %g, want %g\n", __VA_ARGS__, double(got), double(want)); \
            ++g_failures;                                                 \
        }                                                                 \
    } while (0)

// m16n8kK: A is 16 x K (row major), B is K x 8, C and D 16 x 8. Fragment layouts per lane
// (groupID g = lane / 4, t = lane % 4) from the PTX ISA.
void testMma(CUmodule mod, int k) {
    float A[16][16], B[16][8], C[16][8];
    for (int r = 0; r < 16; ++r)
        for (int c = 0; c < k; ++c) A[r][c] = float((r * 3 + c * 5) % 7 - 3) * 0.5f;
    for (int r = 0; r < k; ++r)
        for (int c = 0; c < 8; ++c) B[r][c] = float((r * 2 + c * 7) % 5 - 2);
    for (int r = 0; r < 16; ++r)
        for (int c = 0; c < 8; ++c) C[r][c] = float(r - c);

    const int aRegs = k / 4, bRegs = k / 8;
    std::vector<uint32_t> a(32 * aRegs), b(32 * bRegs);
    std::vector<float> c(32 * 4);
    for (int lane = 0; lane < 32; ++lane) {
        const int g = lane / 4, t = lane % 4;
        for (int i = 0; i < 2 * aRegs; ++i) {  // a_i: row g + 8 ((i / 2) % 2), col 2t + i % 2 + 8 (i / 4)
            const uint16_t h = toHalf(A[g + 8 * ((i / 2) % 2)][2 * t + i % 2 + 8 * (i / 4)]);
            a[lane * aRegs + i / 2] |= uint32_t(h) << (16 * (i % 2));
        }
        for (int i = 0; i < 2 * bRegs; ++i) {  // b_i: row 2t + i % 2 + 8 (i / 2), col g
            const uint16_t h = toHalf(B[2 * t + i % 2 + 8 * (i / 2)][g]);
            b[lane * bRegs + i / 2] |= uint32_t(h) << (16 * (i % 2));
        }
        for (int i = 0; i < 4; ++i) c[lane * 4 + i] = C[g + 8 * (i / 2)][2 * t + i % 2];
    }
    CUdeviceptr da = upload(a.data(), a.size() * 4), db = upload(b.data(), b.size() * 4),
                dc = upload(c.data(), c.size() * 4), dd;
    const size_t outBytes = k == 16 ? 32 * 16 : 32 * 8;
    CHECK_CU(cuMemAlloc(&dd, outBytes));
    CUfunction f;
    CHECK_CU(cuModuleGetFunction(&f, mod, k == 16 ? "mma_k16" : "mma_k8"));
    void* args[] = {&da, &db, &dc, &dd};
    CHECK_CU(cuLaunchKernel(f, 1, 1, 1, 32, 1, 1, 0, nullptr, args, nullptr));
    std::vector<uint8_t> out(outBytes);
    CHECK_CU(cuMemcpyDtoH(out.data(), dd, outBytes));
    for (int lane = 0; lane < 32; ++lane) {
        const int g = lane / 4, t = lane % 4;
        for (int i = 0; i < 4; ++i) {  // d_i: row g + 8 (i / 2), col 2t + i % 2
            const int row = g + 8 * (i / 2), col = 2 * t + i % 2;
            float want = C[row][col];
            for (int kk = 0; kk < k; ++kk) want += A[row][kk] * B[kk][col];
            float got;
            if (k == 16) {
                std::memcpy(&got, &out[lane * 16 + i * 4], 4);
            } else {
                uint16_t h;
                std::memcpy(&h, &out[lane * 8 + i * 2], 2);
                got = fromHalf(h);
            }
            CHECK_EQ("mma k%d lane %d d%d", got, want, k, lane, i);
        }
    }
    for (CUdeviceptr p : {da, db, dc, dd}) CHECK_CU(cuMemFree(p));
}

// m16n8k8 tf32: lane (g, t) holds A[g][t], A[g+8][t], A[g][t+4], A[g+8][t+4] and B[t][g], B[t+4][g]
// as f32 bits; the low 13 mantissa bits are ignored, as by the tensor core.
void testMmaTf32(CUmodule mod) {
    float A[16][8], B[8][8], C[16][8];
    for (int r = 0; r < 16; ++r)
        for (int c = 0; c < 8; ++c) A[r][c] = float((r * 3 + c * 5) % 7 - 3) * 0.5f;
    for (int r = 0; r < 8; ++r)
        for (int c = 0; c < 8; ++c) B[r][c] = float((r * 2 + c * 7) % 5 - 2);
    for (int r = 0; r < 16; ++r)
        for (int c = 0; c < 8; ++c) C[r][c] = float(r - c);
    auto bitsOf = [](float f, uint32_t noise) {  // low bits the tensor core drops
        uint32_t u;
        std::memcpy(&u, &f, 4);
        return u | (noise & 0x1FFF);
    };
    std::vector<uint32_t> a(32 * 4), b(32 * 2);
    std::vector<float> c(32 * 4);
    for (int lane = 0; lane < 32; ++lane) {
        const int g = lane / 4, t = lane % 4;
        for (int i = 0; i < 4; ++i) a[lane * 4 + i] = bitsOf(A[g + 8 * (i % 2)][t + 4 * (i / 2)], uint32_t(lane * 37 + i));
        for (int i = 0; i < 2; ++i) b[lane * 2 + i] = bitsOf(B[t + 4 * i][g], uint32_t(lane * 11 + i));
        for (int i = 0; i < 4; ++i) c[lane * 4 + i] = C[g + 8 * (i / 2)][2 * t + i % 2];
    }
    CUdeviceptr da = upload(a.data(), a.size() * 4), db = upload(b.data(), b.size() * 4),
                dc = upload(c.data(), c.size() * 4), dd;
    CHECK_CU(cuMemAlloc(&dd, 32 * 16));
    CUfunction f;
    CHECK_CU(cuModuleGetFunction(&f, mod, "mma_tf32"));
    void* args[] = {&da, &db, &dc, &dd};
    CHECK_CU(cuLaunchKernel(f, 1, 1, 1, 32, 1, 1, 0, nullptr, args, nullptr));
    std::vector<float> out(32 * 4);
    CHECK_CU(cuMemcpyDtoH(out.data(), dd, out.size() * 4));
    for (int lane = 0; lane < 32; ++lane) {
        const int g = lane / 4, t = lane % 4;
        for (int i = 0; i < 4; ++i) {
            const int row = g + 8 * (i / 2), col = 2 * t + i % 2;
            float want = C[row][col];
            for (int kk = 0; kk < 8; ++kk) want += A[row][kk] * B[kk][col];
            CHECK_EQ("mma tf32 lane %d d%d", out[lane * 4 + i], want, lane, i);
        }
    }
    for (CUdeviceptr p : {da, db, dc, dd}) CHECK_CU(cuMemFree(p));
}

void testCpAsync(CUmodule mod) {
    std::vector<uint8_t> in(512);
    for (size_t i = 0; i < in.size(); ++i) in[i] = uint8_t(i * 7 + 1);
    CUdeviceptr din = upload(in.data(), in.size()), dout;
    CHECK_CU(cuMemAlloc(&dout, 32 * 32));
    CHECK_CU(cuMemsetD8(dout, 0xEE, 32 * 32));
    CUfunction f;
    CHECK_CU(cuModuleGetFunction(&f, mod, "cp_async"));
    void* args[] = {&din, &dout};
    CHECK_CU(cuLaunchKernel(f, 1, 1, 1, 32, 1, 1, 0, nullptr, args, nullptr));
    std::vector<uint8_t> out(32 * 32);
    CHECK_CU(cuMemcpyDtoH(out.data(), dout, out.size()));
    for (int t = 0; t < 32; ++t) {
        for (int i = 0; i < 16; ++i)  // src-size min(t, 16): the rest is zero-filled
            CHECK_EQ("cp.async lane %d byte %d", out[t * 32 + i], i < std::min(t, 16) ? in[t * 16 + i] : 0, t, i);
        for (int i = 0; i < 8; ++i)  // ignore-src on odd lanes
            CHECK_EQ("cp.async ignore-src lane %d byte %d", out[t * 32 + 16 + i], t % 2 ? 0 : in[t * 8 + i], t, i);
    }
    CHECK_CU(cuMemFree(din));
    CHECK_CU(cuMemFree(dout));
}

uint16_t toBf16(float f) {  // round to nearest even
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return uint16_t((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}

void testCvtPacked(CUmodule mod) {
    std::vector<float> in(64);
    for (int t = 0; t < 32; ++t) {
        in[2 * t] = float(t - 16) * 0.375f;         // exact in f16 and bf16, negative for relu
        in[2 * t + 1] = 1.0f + float(t) / 1024.0f;  // exact in f16, rounded (to even) in bf16
    }
    CUdeviceptr din = upload(in.data(), in.size() * 4), dout;
    CHECK_CU(cuMemAlloc(&dout, 32 * 16));
    CUfunction f;
    CHECK_CU(cuModuleGetFunction(&f, mod, "cvt_packed"));
    void* args[] = {&din, &dout};
    CHECK_CU(cuLaunchKernel(f, 1, 1, 1, 32, 1, 1, 0, nullptr, args, nullptr));
    std::vector<uint32_t> out(32 * 4);
    CHECK_CU(cuMemcpyDtoH(out.data(), dout, out.size() * 4));
    for (int t = 0; t < 32; ++t) {
        const float a = in[2 * t], b = in[2 * t + 1];
        const uint32_t f16 = uint32_t(toHalf(a)) << 16 | toHalf(b);  // a in the upper half
        const uint32_t bf16 = uint32_t(toBf16(a)) << 16 | toBf16(b);
        const uint32_t relu = uint32_t(toBf16(a < 0 ? 0.0f : a)) << 16 | toBf16(b);
        CHECK_EQ("cvt f16x2 lane %d", out[t * 4], f16, t);
        CHECK_EQ("cvt bf16x2 lane %d", out[t * 4 + 1], bf16, t);
        CHECK_EQ("cvt relu bf16x2 lane %d", out[t * 4 + 2], relu, t);
    }
    CHECK_CU(cuMemFree(din));
    CHECK_CU(cuMemFree(dout));
}

void testLdmatrix(CUmodule mod) {
    // matrix m, row r, column c holds m * 1000 + r * 10 + c
    std::vector<uint16_t> in(32 * 8);
    for (int row = 0; row < 32; ++row)
        for (int c = 0; c < 8; ++c) in[row * 8 + c] = uint16_t((row / 8) * 1000 + (row % 8) * 10 + c);
    CUdeviceptr din = upload(in.data(), in.size() * 2), dout;
    CHECK_CU(cuMemAlloc(&dout, 32 * 24));
    CUfunction f;
    CHECK_CU(cuModuleGetFunction(&f, mod, "ldmatrix"));
    void* args[] = {&din, &dout};
    CHECK_CU(cuLaunchKernel(f, 1, 1, 1, 32, 1, 1, 0, nullptr, args, nullptr));
    std::vector<uint16_t> out(32 * 12);
    CHECK_CU(cuMemcpyDtoH(out.data(), dout, out.size() * 2));
    auto at = [](int m, int r, int c) { return m * 1000 + r * 10 + c; };
    for (int lane = 0; lane < 32; ++lane) {
        for (int m = 0; m < 4; ++m)
            for (int h = 0; h < 2; ++h)  // row lane / 4, columns 2 (lane % 4) + h
                CHECK_EQ("ldmatrix lane %d matrix %d half %d", out[lane * 12 + m * 2 + h], at(m, lane / 4, 2 * (lane % 4) + h),
                         lane, m, h);
        for (int m = 0; m < 2; ++m)
            for (int h = 0; h < 2; ++h)  // transposed: row 2 (lane % 4) + h, column lane / 4
                CHECK_EQ("ldmatrix.trans lane %d matrix %d half %d", out[lane * 12 + 8 + m * 2 + h],
                         at(m, 2 * (lane % 4) + h, lane / 4), lane, m, h);
    }
    CHECK_CU(cuMemFree(din));
    CHECK_CU(cuMemFree(dout));
}

void testRedux(CUmodule mod) {
    std::vector<uint32_t> in(32);
    uint32_t seed = 7;
    for (uint32_t& v : in) {
        seed = seed * 1664525u + 1013904223u;
        v = seed ^ (seed >> 7);
    }
    in[5] = 0x80000001u;  // a negative minimum for min.s32
    CUdeviceptr din = upload(in.data(), in.size() * 4), dout;
    CHECK_CU(cuMemAlloc(&dout, 32 * 32));
    CUfunction f;
    CHECK_CU(cuModuleGetFunction(&f, mod, "redux"));
    void* args[] = {&din, &dout};
    CHECK_CU(cuLaunchKernel(f, 1, 1, 1, 32, 1, 1, 0, nullptr, args, nullptr));
    std::vector<uint32_t> out(32 * 8);
    CHECK_CU(cuMemcpyDtoH(out.data(), dout, out.size() * 4));
    const uint32_t part = 0x00F0F00Fu;
    uint32_t want[8] = {0, 0x7FFFFFFFu, 0, ~0u, 0, 0, 0, ~0u};
    for (int l = 0; l < 32; ++l) {
        const uint32_t v = in[l];
        want[0] += v;
        if (int32_t(v) < int32_t(want[1])) want[1] = v;
        want[2] = v > want[2] ? v : want[2];
        want[3] &= v;
        want[4] |= v;
        want[5] ^= v;
        if (part >> l & 1) {
            want[6] += v;
            want[7] = v < want[7] ? v : want[7];
        }
    }
    for (int lane = 0; lane < 32; ++lane)
        for (int i = 0; i < 8; ++i) {
            const uint32_t w = i < 6 || (part >> lane & 1) ? want[i] : 0;
            CHECK_EQ("redux lane %d result %d", out[lane * 8 + i], w, lane, i);
        }
    CHECK_CU(cuMemFree(din));
    CHECK_CU(cuMemFree(dout));
}

}  // namespace

int main() {
    CHECK_CU(cuInit(0));
    CUdevice dev;
    CHECK_CU(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CHECK_CU(cuCtxCreate(&ctx, 0, dev));
    CUmodule mod;
    CHECK_CU(cuModuleLoadData(&mod, kPtx));
    testMma(mod, 16);
    testMma(mod, 8);
    testMmaTf32(mod);
    testCpAsync(mod);
    testCvtPacked(mod);
    testLdmatrix(mod);
    testRedux(mod);
    CHECK_CU(cuModuleUnload(mod));
    CHECK_CU(cuCtxDestroy(ctx));
    if (g_failures) {
        std::printf("%d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
