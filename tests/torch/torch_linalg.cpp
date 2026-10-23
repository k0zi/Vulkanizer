// torch.linalg on Vulkanizer's libcusolver and libcublas's batched LU, against the CPU: inverses,
// solves, determinants, Cholesky, LU and QR factorizations, symmetric eigenproblems and SVDs,
// single and batched, f32 and f64.

#include "torch_test.h"

using torch_test::check;

namespace {

torch::Tensor randn(std::vector<int64_t> shape, torch::Dtype dtype = torch::kFloat) {
    return torch::randn(shape, torch::TensorOptions().dtype(dtype));
}

// Well conditioned: random plus n on the diagonal.
torch::Tensor wellConditioned(std::vector<int64_t> shape, torch::Dtype dtype = torch::kFloat) {
    const int64_t n = shape.back();
    return randn(shape, dtype) + n * torch::eye(n, torch::TensorOptions().dtype(dtype));
}

// Symmetric positive definite: M M^T + n I.
torch::Tensor spd(std::vector<int64_t> shape, torch::Dtype dtype = torch::kFloat) {
    const torch::Tensor m = randn(shape, dtype);
    return m.matmul(m.transpose(-2, -1)) + shape.back() * torch::eye(shape.back(), torch::TensorOptions().dtype(dtype));
}

}  // namespace

int main() {
    torch_test::useVulkanizerDefaults();
    torch::manual_seed(0);

    for (torch::Dtype dtype : {torch::kFloat, torch::kDouble}) {
        const std::string t = dtype == torch::kFloat ? " f32" : " f64";
        const double tol = dtype == torch::kFloat ? 1e-3 : 1e-9;
        const torch::Tensor a = wellConditioned({16, 16}, dtype), batch = wellConditioned({8, 5, 5}, dtype);
        const torch::Tensor b = randn({16, 3}, dtype), bb = randn({8, 5, 2}, dtype);
        const torch::Tensor s = spd({12, 12}, dtype), sb = spd({6, 4, 4}, dtype);
        check("inv" + t, [&](torch::Device d) { return std::vector{torch::linalg_inv(a.to(d))}; }, tol);
        check("inv batched" + t, [&](torch::Device d) { return std::vector{torch::linalg_inv(batch.to(d))}; }, tol);
        check("solve" + t, [&](torch::Device d) { return std::vector{torch::linalg_solve(a.to(d), b.to(d))}; }, tol);
        check("solve batched" + t,
              [&](torch::Device d) { return std::vector{torch::linalg_solve(batch.to(d), bb.to(d))}; }, tol);
        check("solve left=false" + t,
              [&](torch::Device d) { return std::vector{torch::linalg_solve(a.to(d), b.t().to(d), false)}; }, tol);
        // a / 16: a determinant near 1, so the tolerance is relative
        check("det" + t, [&](torch::Device d) { return std::vector{torch::linalg_det(a.to(d) / 16)}; }, tol);
        check("slogdet" + t, [&](torch::Device d) {
            auto [sign, logabs] = torch::linalg_slogdet(batch.to(d));
            return std::vector{sign, logabs};
        }, tol);
        check("lu_factor + lu_solve" + t, [&](torch::Device d) {
            auto [lu, pivots] = torch::linalg_lu_factor(a.to(d));
            return std::vector{torch::linalg_lu_solve(lu, pivots, b.to(d))};
        }, tol);
        check("cholesky" + t, [&](torch::Device d) { return std::vector{torch::linalg_cholesky(s.to(d))}; }, tol);
        check("cholesky upper" + t,
              [&](torch::Device d) { return std::vector{torch::linalg_cholesky(s.to(d), true)}; }, tol);
        check("cholesky batched" + t, [&](torch::Device d) { return std::vector{torch::linalg_cholesky(sb.to(d))}; },
              tol);
        check("cholesky_solve" + t, [&](torch::Device d) {
            const torch::Tensor l = torch::linalg_cholesky(s.to(d));
            return std::vector{torch::cholesky_solve(randn({12, 2}, dtype).fill_(1).to(d), l)};
        }, tol);
        const torch::Tensor tall = randn({20, 7}, dtype), rhs = randn({20, 3}, dtype);
        check("qr reduced" + t, [&](torch::Device d) {
            auto [q, r] = torch::linalg_qr(tall.to(d));
            return std::vector{q, r};
        }, tol);
        check("qr complete" + t, [&](torch::Device d) {
            auto [q, r] = torch::linalg_qr(tall.to(d), "complete");
            return std::vector{q, r};
        }, tol);
        const torch::Tensor tallBatch = randn({4, 6, 5}, dtype);
        check("qr batched" + t, [&](torch::Device d) {
            auto [q, r] = torch::linalg_qr(tallBatch.to(d));
            return std::vector{q, r};
        }, tol);
        check("householder_product + ormqr" + t, [&](torch::Device d) {
            auto [a, tau] = torch::geqrf(tall.to(d));
            return std::vector{torch::linalg_householder_product(a, tau), torch::ormqr(a, tau, rhs.to(d), true, true)};
        }, tol);
        check("lstsq" + t, [&](torch::Device d) {
            return std::vector{std::get<0>(torch::linalg_lstsq(tall.to(d), rhs.to(d)))};
        }, tol);
        const torch::Tensor upper = torch::triu(wellConditioned({9, 9}, dtype)), rhs9 = randn({9, 4}, dtype);
        check("solve_triangular" + t, [&](torch::Device d) {
            return std::vector{torch::linalg_solve_triangular(upper.to(d), rhs9.to(d), true),
                               torch::linalg_solve_triangular(upper.to(d), rhs9.t().to(d), true, false),
                               torch::linalg_solve_triangular(upper.t().to(d), rhs9.to(d), false, true, true)};
        }, tol);
        // eigenvectors and singular vectors are unique only up to sign: compared through invariants
        const torch::Tensor sym = s - 12 * torch::eye(12, torch::TensorOptions().dtype(dtype));
        const torch::Tensor symBatch = sb.matmul(sb), symBig = spd({40, 40}, dtype) / 40;
        const auto eighCheck = [](const torch::Tensor& m, const std::string& uplo) {
            auto [w, v] = torch::linalg_eigh(m, uplo);
            return std::vector{w, v.matmul(torch::diag_embed(w)).matmul(v.transpose(-2, -1)),
                               v.transpose(-2, -1).matmul(v)};
        };
        check("eigh" + t, [&](torch::Device d) { return eighCheck(sym.to(d), "L"); }, tol);
        check("eigh upper" + t, [&](torch::Device d) { return eighCheck(sym.to(d), "U"); }, tol);
        check("eigh batched" + t, [&](torch::Device d) { return eighCheck(symBatch.to(d), "L"); }, tol);
        check("eigh 40" + t, [&](torch::Device d) { return eighCheck(symBig.to(d), "L"); }, tol);
        check("eigvalsh" + t, [&](torch::Device d) { return std::vector{torch::linalg_eigvalsh(symBatch.to(d))}; }, tol);
        const auto svdCheck = [](const torch::Tensor& m, bool full) {
            auto [u, sv, vh] = torch::linalg_svd(m, full);
            const int64_t k = sv.size(-1);
            const torch::Tensor uk = u.narrow(-1, 0, k), vk = vh.narrow(-2, 0, k);
            return std::vector{sv, uk.matmul(torch::diag_embed(sv)).matmul(vk), u.transpose(-2, -1).matmul(u),
                               vh.matmul(vh.transpose(-2, -1))};
        };
        const torch::Tensor wide = randn({5, 11}, dtype), smallBatch = randn({6, 4, 3}, dtype),
                            bigBatch = randn({3, 40, 36}, dtype);
        check("svd" + t, [&](torch::Device d) { return svdCheck(tall.to(d), false); }, tol);
        check("svd full" + t, [&](torch::Device d) { return svdCheck(tall.to(d), true); }, tol);
        check("svd wide full" + t, [&](torch::Device d) { return svdCheck(wide.to(d), true); }, tol);
        check("svd batched" + t, [&](torch::Device d) { return svdCheck(smallBatch.to(d), false); }, tol);
        check("svd big batched" + t, [&](torch::Device d) { return svdCheck(bigBatch.to(d), false); },
              tol * 10);
        check("svdvals" + t, [&](torch::Device d) { return std::vector{torch::linalg_svdvals(tallBatch.to(d))}; }, tol);
        check("pinv" + t, [&](torch::Device d) { return std::vector{torch::linalg_pinv(tall.to(d))}; }, tol);
        check("matrix_rank" + t, [&](torch::Device d) {
            const torch::Tensor low = tall.narrow(1, 0, 3).matmul(randn({3, 7}, dtype));
            return std::vector{torch::linalg_matrix_rank(low.to(d)).to(torch::kFloat)};
        }, 0);
        check("matrix_norm 2, nuc" + t, [&](torch::Device d) {
            return std::vector{torch::linalg_matrix_norm(tall.to(d), 2), torch::linalg_matrix_norm(tall.to(d), "nuc")};
        }, tol);
        check("cholesky_ex info" + t, [&](torch::Device d) {  // not positive definite: info = 2
            auto [l, info] = torch::linalg_cholesky_ex(-torch::eye(3, torch::TensorOptions().dtype(dtype)).to(d));
            return std::vector{info.to(torch::kFloat)};
        }, 0);
    }
    return torch_test::finish();
}
