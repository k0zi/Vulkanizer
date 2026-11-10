#pragma once

// Host-side matrix plumbing shared by the backends: the NPU kernels take row-major operands of
// their own type, cuBLAS hands over column-major ones. A column-major C (m x n, ldc) read
// row-major is C^T, so the backends compute C^T = op(B)^T op(A)^T: lhs = op(B)^T (n x k), rhs =
// op(A)^T (k x m), result n x m, all row-major.

#include "npu/npu.h"

#include <cstdint>

namespace vulkanizer::npu {

size_t dtypeSize(DType t);
float halfToFloat(uint16_t h);
uint16_t floatToHalf(float f);  // round to nearest even
float bf16ToFloat(uint16_t h);
uint16_t floatToBf16(float f);  // round to nearest even

float load(const void* base, DType t, int64_t index);
void store(void* base, DType t, int64_t index, float v);

// dst[c * dstLd + r] = op(X)[r][c] for op(X) rows x cols, X column-major with leading dimension
// ld (op = transpose when `trans`); dst is row-major cols x rows of type dt.
void packTransposed(const void* x, DType xt, int64_t ld, bool trans, int64_t rows, int64_t cols, void* dst, DType dt,
                    int64_t dstLd);

// Zeroes what packTransposed leaves of a row-major buffer of allRows x ld elements: columns cols..ld
// of rows 0..rows and the rows from `rows` on.
void zeroPadding(void* dst, size_t elem, int64_t rows, int64_t cols, int64_t ld, int64_t allRows);

// C[i + j * ldc] = alpha * D[j * dLd + i] + beta * C[i + j * ldc] for i < m, j < n; C is not read
// when beta is 0 (as in cuBLAS: NaNs in C do not propagate).
void epilogue(const void* d, DType dt, int64_t dLd, float alpha, float beta, void* c, DType ct, int64_t ldc, int64_t m,
              int64_t n);

}  // namespace vulkanizer::npu
