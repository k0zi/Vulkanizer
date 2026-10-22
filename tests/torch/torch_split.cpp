// Kernels split over many batches (run with a tiny VULKANIZER_BATCH_MS, e.g.
// VULKANIZER_BATCH_MS=0.01 ./run_torch.sh torch_split): every chunk must see its own parameters,
// also kernels whose parameters live in a buffer (TensorIterator offset calculators of
// non-contiguous tensors), after the batch slots have been reused.
#include "torch_test.h"

using torch::indexing::None;
using torch::indexing::Slice;

int main() {
    torch_test::useVulkanizerDefaults();
    torch::manual_seed(0);
    const auto x = torch::randn({1, 4128, 3840});
    const auto xg = x.cuda().to(torch::kHalf), xh = x.to(torch::kHalf);
    // Launched back to back, without synchronizing in between: the later launches upload their
    // parameters while the chunks of the earlier ones are still waiting in reused batch slots.
    // Different sizes split into different numbers of batches, so the batch slots wrap around at
    // varying points; one bad launch is enough to fail.
    std::vector<std::function<torch::Tensor(const torch::Tensor&)>> views;
    for (int s = 0; s < 20; ++s)
        views.push_back([s](const torch::Tensor& t) { return t.index({Slice(), Slice(s, None, s + 1), Slice(None, None, s + 2)}); });
    views.push_back([](const torch::Tensor& t) { return t.transpose(1, 2); });
    views.push_back([](const torch::Tensor& t) { return t.view({1, 4128, 30, 128}).permute({0, 2, 3, 1}); });
    std::vector<torch::Tensor> out;
    for (int round = 0; round < 3; ++round)
        for (const auto& v : views) out.push_back(v(xg).contiguous());
    int bad = 0;
    for (size_t i = 0; i < out.size(); ++i) bad += !torch::equal(out[i].cpu(), views[i % views.size()](xh).contiguous());
    torch_test::report(std::to_string(out.size()) + " strided copies, mismatching", bad, 0);
    torch_test::report("layer_norm", torch_test::maxErr(torch::layer_norm(x, {3840}), torch::layer_norm(xg, {3840})), 2e-2);
    const auto a = torch::randn({512, 768}), b = torch::randn({768, 1024});
    torch_test::report("matmul", torch_test::maxErr(a.matmul(b), a.cuda().matmul(b.cuda())), 1e-3);
    return torch_test::finish();
}
