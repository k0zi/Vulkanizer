// Trains a small MLP on the CPU and on the Vulkanizer GPU from the same start and compares the
// losses, the parameters and the accuracy: ./run_torch.sh torch_mlp
#include "torch_test.h"

#include <chrono>

namespace {

struct Result {
    std::vector<double> losses;
    double accuracy;
    std::vector<torch::Tensor> params;
    double seconds;
};

Result train(torch::Device device, const torch::Tensor& X, const torch::Tensor& Y, int steps = 50) {
    torch::manual_seed(1);
    torch::nn::Sequential model(torch::nn::Linear(32, 64), torch::nn::ReLU(), torch::nn::Linear(64, 64),
                                torch::nn::GELU(), torch::nn::Linear(64, 1));
    model->to(device);
    torch::optim::Adam opt(model->parameters(), torch::optim::AdamOptions(1e-2));
    const auto x = X.to(device), y = Y.to(device);
    Result r;
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < steps; ++i) {
        opt.zero_grad();
        auto loss = torch::binary_cross_entropy_with_logits(model->forward(x), y);
        loss.backward();
        opt.step();
        r.losses.push_back(loss.item<double>());
    }
    r.accuracy = ((model->forward(x) > 0).to(torch::kFloat) == y).to(torch::kFloat).mean().item<double>();
    for (const auto& p : model->parameters()) r.params.push_back(p.detach().cpu());
    r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return r;
}

}  // namespace

int main() {
    torch_test::useVulkanizerDefaults();
    torch::manual_seed(0);
    const auto X = torch::randn({256, 32});
    const auto Y = (X.index({torch::indexing::Slice(), torch::indexing::Slice(0, 4)}).sum(1, true) > 0).to(torch::kFloat);
    const Result c = train(torch::kCPU, X, Y), g = train(torch::kCUDA, X, Y);
    std::printf("cpu  loss %.4f -> %.4f  acc %.3f  %.2fs\n", c.losses.front(), c.losses.back(), c.accuracy, c.seconds);
    std::printf("cuda loss %.4f -> %.4f  acc %.3f  %.2fs\n", g.losses.front(), g.losses.back(), g.accuracy, g.seconds);
    double lossDiff = 0, paramDiff = 0;
    for (size_t i = 0; i < c.losses.size(); ++i) lossDiff = std::max(lossDiff, std::abs(c.losses[i] - g.losses[i]));
    for (size_t i = 0; i < c.params.size(); ++i) paramDiff = std::max(paramDiff, torch_test::maxErr(c.params[i], g.params[i]));
    torch_test::report("losses", lossDiff, 1e-4);
    torch_test::report("parameters", paramDiff, 1e-4);
    torch_test::report("accuracy", std::abs(c.accuracy - g.accuracy), 0);
    return torch_test::finish();
}
