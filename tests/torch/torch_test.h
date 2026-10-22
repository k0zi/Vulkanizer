// Shared by the libtorch tests: the Vulkanizer GPU ("cuda") against the CPU. Run them through
// run_torch.sh (e.g. ./run_torch.sh torch_ops), which preloads Vulkanizer's CUDA libraries.
#pragma once

#include <torch/torch.h>

#include <cstdio>
#include <exception>
#include <functional>
#include <string>
#include <vector>

namespace torch_test {

inline int g_failures = 0;

// What run_torch.sh's sitecustomize.py does for Python programs: cuDNN and the fused attention
// kernels need NVIDIA hardware, so convolutions and scaled_dot_product_attention use PyTorch's own
// kernels and the math path.
inline void useVulkanizerDefaults() {
    auto& ctx = at::globalContext();
    ctx.setUserEnabledCuDNN(false);
    ctx.setSDPUseFlash(false);
    ctx.setSDPUseMemEfficient(false);
    ctx.setSDPUseCuDNN(false);
}

// Largest absolute difference of `got` (any device) from `want` (CPU), in float.
inline double maxErr(const torch::Tensor& want, const torch::Tensor& got) {
    return (want.to(torch::kFloat) - got.to(torch::kCPU, torch::kFloat)).abs().max().item<double>();
}

inline void report(const std::string& name, double err, double tol) {
    const bool ok = err <= tol;
    std::printf("%s %s: max err %.2e\n", ok ? "OK  " : "DIFF", name.c_str(), err);
    std::fflush(stdout);
    if (!ok) ++g_failures;
}

// Runs fn on the CPU and on the GPU and compares all tensors it returns.
inline void check(const std::string& name, const std::function<std::vector<torch::Tensor>(torch::Device)>& fn,
                  double tol) {
    try {
        const auto cpu = fn(torch::kCPU), gpu = fn(torch::kCUDA);
        double err = 0;
        for (size_t i = 0; i < cpu.size(); ++i) err = std::max(err, maxErr(cpu[i], gpu[i]));
        report(name, err, tol);
    } catch (const std::exception& e) {
        ++g_failures;
        const std::string what = e.what();
        std::printf("FAIL %s: %s\n", name.c_str(), what.substr(0, what.find('\n')).c_str());
    }
}

inline int finish() {
    std::printf("%s\n", g_failures ? "FAILED" : "ALL OK");
    return g_failures ? 1 : 0;
}

}  // namespace torch_test
