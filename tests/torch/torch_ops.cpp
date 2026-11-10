// PyTorch operations on the Vulkanizer GPU against the CPU: ./run_torch.sh torch_ops
#include "torch_test.h"

namespace F = torch::nn::functional;
using torch::Device;
using torch::indexing::None;
using torch::indexing::Slice;
using Tensors = std::vector<torch::Tensor>;
using torch_test::check;

int main() {
    torch_test::useVulkanizerDefaults();
    torch::manual_seed(0);

    const auto x = torch::randn({4, 3, 16, 16}), w = torch::randn({8, 3, 3, 3});
    check("conv2d", [&](Device d) -> Tensors { return {F::conv2d(x.to(d), w.to(d), F::Conv2dFuncOptions().padding(1))}; },
          1e-3);
    const auto a = torch::randn({8, 100});
    check("softmax", [&](Device d) -> Tensors { return {torch::softmax(a.to(d), -1)}; }, 1e-6);
    check("layer_norm", [&](Device d) -> Tensors { return {torch::layer_norm(a.to(d), {100})}; }, 1e-5);
    const auto emb = torch::randn({50, 16}), idx = torch::randint(0, 50, {20});
    check("embedding", [&](Device d) -> Tensors { return {torch::embedding(emb.to(d), idx.to(d))}; }, 0);
    const auto q = torch::randn({2, 4, 32, 16});
    check("sdpa", [&](Device d) -> Tensors { return {at::scaled_dot_product_attention(q.to(d), q.to(d), q.to(d))}; },
          1e-4);
    const auto m1 = torch::randn({64, 64}), m2 = torch::randn({64, 64});
    check("matmul f16", [&](Device d) -> Tensors {
        if (d.is_cpu()) return {m1.to(torch::kHalf).to(torch::kFloat).matmul(m2.to(torch::kHalf).to(torch::kFloat))};
        return {m1.to(d).to(torch::kHalf).matmul(m2.to(d).to(torch::kHalf))};
    }, 0.5);
    check("cumsum/sort/topk", [&](Device d) -> Tensors {
        const auto t = a.to(d);
        return {t.cumsum(1), std::get<0>(t.sort(1)), std::get<0>(t.topk(5, 1))};
    }, 1e-4);
    check("argmax/max", [&](Device d) -> Tensors {
        const auto t = a.to(d);
        return {t.argmax(1), std::get<0>(t.max(1))};
    }, 0);
    check("index/where", [&](Device d) -> Tensors {
        const auto t = a.to(d);
        return {torch::where(t > 0, t, 0), t.index({Slice(), Slice(None, None, 3)})};
    }, 0);
    torch::manual_seed(3);
    const auto src = torch::randn({10, 2, 32});  // (sequence, batch, features)
    check("transformer layer", [&](Device d) -> Tensors {
        torch::manual_seed(2);
        torch::nn::TransformerEncoderLayer layer(torch::nn::TransformerEncoderLayerOptions(32, 4).dim_feedforward(64).dropout(0.0));
        layer->to(d);
        return {layer->forward(src.to(d))};
    }, 1e-4);
    const auto vals = torch::randn({30, 8}), index = torch::randint(0, 5, {30});
    check("index_add f16", [&](Device d) -> Tensors {
        return {torch::zeros({5, 8}, torch::TensorOptions(d).dtype(torch::kHalf)).index_add_(0, index.to(d), vals.to(d).to(torch::kHalf))};
    }, 0.05);
    check("index_add f32", [&](Device d) -> Tensors {
        return {torch::zeros({5, 8}, torch::TensorOptions(d)).index_add_(0, index.to(d), vals.to(d))};
    }, 1e-5);
    return torch_test::finish();
}
