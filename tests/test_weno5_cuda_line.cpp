// One CUDA thread block is one WENO5 line, so a line longer than 1024 cells
// is an illegal launch. 1024 must match the CPU result. 1025 is the smallest
// line that does not.

#include <torch/torch.h>

#include <snap/recon/interpolation.hpp>

#include <gtest/gtest.h>

#include <exception>

using namespace snap;

namespace {

torch::Tensor weno5_left(torch::Tensor w, torch::Device device) {
  Weno5Interp recon;
  recon->to(device);
  auto out_sizes = w.sizes().vec();
  out_sizes.back() -= 4;
  auto out = torch::empty(out_sizes, w.options().device(device));
  recon->left(w.to(device), /*dim=*/3, out);
  if (device.is_cuda()) {
    torch::cuda::synchronize();
  }
  return out;
}

}  // namespace

TEST(weno5_cuda, line_above_1024_matches_cpu) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "no CUDA device";
  }
  auto opts = torch::TensorOptions().dtype(torch::kFloat64);
  for (int n : {1024, 1025}) {
    auto x = torch::arange(n, opts);
    auto w = torch::empty({5, 1, 1, n}, opts);
    for (int v = 0; v < 5; ++v) {
      w[v][0][0] = torch::sin((v + 1) * x / n);
    }
    torch::Tensor cpu, gpu;
    ASSERT_NO_THROW(cpu = weno5_left(w, torch::kCPU)) << "n=" << n;
    try {
      gpu = weno5_left(w, torch::kCUDA);
    } catch (std::exception const& err) {
      FAIL() << "CUDA WENO5 line of " << n
             << " cells failed: " << err.what();
    }
    auto diff = (gpu.cpu() - cpu).abs().max().item<double>();
    EXPECT_LT(diff, 1e-12) << "n=" << n << " max abs diff " << diff;
  }
}
