// PyTorch CUDA graphs on the Vulkanizer GPU: an MLP forward pass captured once with
// at::cuda::CUDAGraph (torch.cuda.CUDAGraph in Python), replayed for new inputs copied into its
// static input, against the CPU: ./run_torch.sh torch_graph
#include "torch_test.h"

#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

int main() {
    torch_test::useVulkanizerDefaults();
    torch::manual_seed(0);
    torch::nn::Sequential model(torch::nn::Linear(64, 128), torch::nn::GELU(), torch::nn::LayerNorm(std::vector<int64_t>{128}),
                                torch::nn::Linear(128, 32));
    model->eval();
    torch::NoGradGuard noGrad;
    torch::nn::Sequential gpu = std::dynamic_pointer_cast<torch::nn::SequentialImpl>(model->clone());
    gpu->to(torch::kCUDA);

    auto input = torch::zeros({16, 64}, torch::kCUDA);  // the graph's static input
    torch::Tensor output;
    at::cuda::CUDAGraph graph;
    {
        c10::cuda::CUDAStream stream = c10::cuda::getStreamFromPool();
        c10::cuda::CUDAStreamGuard guard(stream);
        for (int i = 0; i < 3; ++i) gpu->forward(input);  // warm-up, as PyTorch recommends
        stream.synchronize();
        graph.capture_begin();
        output = gpu->forward(input);
        graph.capture_end();
    }
    for (int round = 0; round < 3; ++round) {
        const auto x = torch::randn({16, 64});
        input.copy_(x);
        graph.replay();
        torch_test::report("replay " + std::to_string(round), torch_test::maxErr(model->forward(x), output), 1e-4);
    }
    return torch_test::finish();
}
