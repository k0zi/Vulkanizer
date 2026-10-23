// torch.fft on Vulkanizer's libcufft, against the CPU: complex and real transforms in one to three
// dimensions, batched, with lengths that are powers of two, mixed radix and prime, f32 and f64,
// plus f16 (powers of two only, as PyTorch requires) and the STFT.

#include "torch_test.h"

using torch_test::check;

namespace {

// Complex results compared as (real, imaginary) pairs.
torch::Tensor pairs(const torch::Tensor& t) { return t.is_complex() ? torch::view_as_real(t.resolve_conj()) : t; }

}  // namespace

int main() {
    torch_test::useVulkanizerDefaults();
    torch::manual_seed(0);

    for (torch::Dtype dtype : {torch::kFloat, torch::kDouble}) {
        const std::string t = dtype == torch::kFloat ? " f32" : " f64";
        const double tol = dtype == torch::kFloat ? 1e-3 : 1e-9;
        const auto opts = torch::TensorOptions().dtype(dtype);
        const torch::Tensor x = torch::randn({4, 96}, opts), prime = torch::randn({3, 37}, opts);
        const torch::Tensor image = torch::randn({2, 3, 24, 20}, opts), volume = torch::randn({5, 6, 7}, opts);
        const torch::Tensor cx = torch::complex(torch::randn({6, 50}, opts), torch::randn({6, 50}, opts));
        check("fft" + t, [&](torch::Device d) { return std::vector{pairs(torch::fft::fft(x.to(d)))}; }, tol);
        check("fft prime" + t, [&](torch::Device d) { return std::vector{pairs(torch::fft::fft(prime.to(d)))}; }, tol);
        check("ifft complex" + t, [&](torch::Device d) { return std::vector{pairs(torch::fft::ifft(cx.to(d)))}; }, tol);
        check("fft n=128 norm=ortho" + t, [&](torch::Device d) {
            return std::vector{pairs(torch::fft::fft(cx.to(d), 128, -1, "ortho"))};
        }, tol);
        check("fft dim 0" + t, [&](torch::Device d) { return std::vector{pairs(torch::fft::fft(x.to(d), {}, 0))}; }, tol);
        check("rfft + irfft" + t, [&](torch::Device d) {
            const torch::Tensor f = torch::fft::rfft(x.to(d));
            return std::vector{pairs(f), torch::fft::irfft(f, 96)};
        }, tol);
        check("rfft odd + irfft" + t, [&](torch::Device d) {
            const torch::Tensor f = torch::fft::rfft(prime.to(d));
            return std::vector{pairs(f), torch::fft::irfft(f, 37)};
        }, tol);
        check("fft2" + t, [&](torch::Device d) { return std::vector{pairs(torch::fft::fft2(image.to(d)))}; }, tol);
        check("rfft2 + irfft2" + t, [&](torch::Device d) {
            const torch::Tensor f = torch::fft::rfft2(image.to(d));
            return std::vector{pairs(f), torch::fft::irfft2(f, std::vector<int64_t>{24, 20})};
        }, tol);
        check("fftn 3d" + t, [&](torch::Device d) { return std::vector{pairs(torch::fft::fftn(volume.to(d)))}; }, tol);
        check("rfftn + irfftn 3d" + t, [&](torch::Device d) {
            const torch::Tensor f = torch::fft::rfftn(volume.to(d));
            return std::vector{pairs(f), torch::fft::irfftn(f, std::vector<int64_t>{5, 6, 7})};
        }, tol);
        check("hfft + ihfft" + t, [&](torch::Device d) {
            return std::vector{torch::fft::hfft(cx.to(d)), pairs(torch::fft::ihfft(x.to(d)))};
        }, tol);
        check("fft non-contiguous" + t, [&](torch::Device d) {
            return std::vector{pairs(torch::fft::fft(image.to(d).transpose(-1, -2)))};
        }, tol);
        check("stft + istft" + t, [&](torch::Device d) {
            const torch::Tensor signal = x.to(d).flatten(), window = torch::hann_window(64, opts.device(d));
            const torch::Tensor s = torch::stft(signal, 64, 16, 64, window, true, "reflect", false, {}, true);
            return std::vector{pairs(s), torch::istft(s, 64, 16, 64, window, true, false, {}, signal.size(0))};
        }, tol * 10);
    }
    // the CPU has no f16 FFT: its reference runs in f32 on the same (f16) values
    const torch::Tensor h = torch::randn({3, 64}).to(torch::kHalf);
    const auto halfInput = [&](torch::Device d) { return d.is_cpu() ? h.to(torch::kFloat) : h.to(d); };
    check("fft f16", [&](torch::Device d) {
        return std::vector{pairs(torch::fft::fft(halfInput(d)).to(torch::kComplexFloat))};
    }, 0.05);
    check("rfft + irfft f16", [&](torch::Device d) {
        const torch::Tensor f = torch::fft::rfft(halfInput(d));
        return std::vector{pairs(f.to(torch::kComplexFloat)), torch::fft::irfft(f, 64).to(torch::kFloat)};
    }, 0.05);
    return torch_test::finish();
}
