// torch.sparse on Vulkanizer's libcusparse, against the CPU: sparse (COO and CSR) times dense,
// addmm, matrix-vector products, gradients (transposed products), sampled_addmm, CSR addition,
// sparse times sparse, sparse triangular solves, batched COO products and COO to CSR, f32 and f64.

#include "torch_test.h"

using torch_test::check;

namespace {

// A random sparse-ish dense matrix: about a quarter of the entries nonzero.
torch::Tensor sparseish(std::vector<int64_t> shape, torch::Dtype dtype) {
    const auto opts = torch::TensorOptions().dtype(dtype);
    return torch::randn(shape, opts) * (torch::rand(shape, opts) < 0.25).to(dtype);
}

}  // namespace

int main() {
    torch_test::useVulkanizerDefaults();
    torch::manual_seed(0);

    for (torch::Dtype dtype : {torch::kFloat, torch::kDouble}) {
        const std::string t = dtype == torch::kFloat ? " f32" : " f64";
        const double tol = dtype == torch::kFloat ? 1e-4 : 1e-10;
        const auto opts = torch::TensorOptions().dtype(dtype);
        const torch::Tensor a = sparseish({30, 20}, dtype), b = torch::randn({20, 8}, opts), c = torch::randn({30, 8}, opts);
        const torch::Tensor v = torch::randn({20}, opts);
        check("sparse.mm COO" + t, [&](torch::Device d) {
            return std::vector{torch::mm(a.to(d).to_sparse(), b.to(d))};
        }, tol);
        check("sparse.mm CSR" + t, [&](torch::Device d) {
            return std::vector{torch::mm(a.to(d).to_sparse_csr(), b.to(d))};
        }, tol);
        check("addmm COO" + t, [&](torch::Device d) {
            return std::vector{torch::addmm(c.to(d), a.to(d).to_sparse(), b.to(d), 0.5, 2.0)};
        }, tol);
        check("addmm CSR" + t, [&](torch::Device d) {
            return std::vector{torch::addmm(c.to(d), a.to(d).to_sparse_csr(), b.to(d), 0.5, 2.0)};
        }, tol);
        check("mv CSR" + t, [&](torch::Device d) { return std::vector{torch::mv(a.to(d).to_sparse_csr(), v.to(d))}; }, tol);
        check("mm COO backward" + t, [&](torch::Device d) {
            torch::Tensor dense = b.to(d).clone().requires_grad_(true);
            torch::Tensor out = torch::mm(a.to(d).to_sparse(), dense);
            out.sum().backward();
            return std::vector{out.detach(), dense.grad()};
        }, tol);
        check("mm dense @ CSR" + t, [&](torch::Device d) {
            return std::vector{torch::mm(c.to(d).t().contiguous(), a.to(d).to_sparse_csr())};
        }, tol);
        check("sampled_addmm" + t, [&](torch::Device d) {
            const torch::Tensor mask = a.to(d).to_sparse_csr();
            return std::vector{torch::sparse_sampled_addmm(mask, c.to(d), torch::randn({8, 20}, opts).fill_(0.5).to(d))
                                   .to_dense()};
        }, tol);
        const torch::Tensor a2 = sparseish({30, 20}, dtype);
        // the CPU references of these use dense math: PyTorch's sparse CPU kernels need MKL
        check("CSR + CSR" + t, [&](torch::Device d) {
            if (d.is_cpu()) return std::vector{a + a2};
            return std::vector{(a.to(d).to_sparse_csr() + a2.to(d).to_sparse_csr()).to_dense()};
        }, tol);
        const torch::Tensor s2 = sparseish({20, 15}, dtype);
        check("CSR @ CSR" + t, [&](torch::Device d) {
            if (d.is_cpu()) return std::vector{torch::mm(a, s2)};
            return std::vector{torch::mm(a.to(d).to_sparse_csr(), s2.to(d).to_sparse_csr()).to_dense()};
        }, tol);
        const torch::Tensor lower = torch::tril(sparseish({25, 25}, dtype)) + 3 * torch::eye(25, opts);
        const torch::Tensor rhs = torch::randn({25, 3}, opts);
        check("triangular_solve CSR" + t, [&](torch::Device d) {
            if (d.is_cpu()) return std::vector{std::get<0>(torch::triangular_solve(rhs, lower, false))};
            return std::vector{std::get<0>(torch::triangular_solve(rhs.to(d), lower.to(d).to_sparse_csr(), false))};
        }, tol);
        // batched CSR descriptors are disabled in PyTorch's CUDA build: batched COO (bmm per batch entry)
        const torch::Tensor batch = sparseish({4, 12, 10}, dtype), bb = torch::randn({4, 10, 6}, opts);
        check("bmm batched COO" + t, [&](torch::Device d) {
            return std::vector{torch::bmm(batch.to(d).to_sparse(), bb.to(d))};
        }, tol);
        check("COO to CSR" + t, [&](torch::Device d) {
            const torch::Tensor csr = a.to(d).to_sparse().to_sparse_csr();
            return std::vector{csr.crow_indices().to(torch::kFloat), csr.to_dense()};
        }, 0);
    }
    return torch_test::finish();
}
