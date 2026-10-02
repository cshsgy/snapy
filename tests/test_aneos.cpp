// external
#include <gtest/gtest.h>

#include "cuda_test_gate.hpp"

// torch
#include <torch/torch.h>

// snap
#include <snap/eos/aneos.hpp>
#include <snap/eos/equation_of_state.hpp>

// time
#include <chrono>

using namespace snap;
using namespace torch::indexing;

TEST(TestANEOSThermo, cpu) {
  ANEOSThermo thermo("example.aneos");

  thermo->pretty_print(std::cout);

  auto rho = torch::ones({6000, 6000}, torch::kDouble) * 2.5e3;
  auto intEng = torch::ones({6000, 6000}, torch::kDouble) * 5.e9;  // J/m^3

  auto c0 = std::chrono::steady_clock::now();

  auto [pres, temp, cs] = thermo->compute("DU->PTL", {rho, intEng});
  std::cout << "pres = " << pres.index({0, Slice(0, 1)}) << std::endl;
  std::cout << "temp = " << temp.index({0, Slice(0, 1)}) << std::endl;
  std::cout << "cs = " << cs.index({0, Slice(0, 1)}) << std::endl;
  std::cout << "pres device = " << pres.device() << '\n';

  auto [temp2, intEng2, cs2] = thermo->compute("DP->TUL", {rho, pres});
  std::cout << "temp2 = " << temp2.index({0, Slice(0, 1)}) << std::endl;
  std::cout << "intEng2 = " << intEng2.index({0, Slice(0, 1)}) << std::endl;
  std::cout << "cs2 = " << cs2.index({0, Slice(0, 1)}) << std::endl;

  auto [pres2, intEng3, cs3] = thermo->compute("DT->PUL", {rho, temp});
  std::cout << "pres2 = " << pres2.index({0, Slice(0, 1)}) << std::endl;
  std::cout << "intEng3 = " << intEng3.index({0, Slice(0, 1)}) << std::endl;
  std::cout << "cs3 = " << cs3.index({0, Slice(0, 1)}) << std::endl;

  EXPECT_TRUE(torch::allclose(temp2, temp, 1e-8, 1e-8));
  EXPECT_TRUE(torch::allclose(intEng2, intEng, 1e-8, 1e-8));
  EXPECT_TRUE(torch::allclose(cs2, cs, 1e-8, 1e-8));
  EXPECT_TRUE(torch::allclose(pres2, pres, 1e-8, 1e-8));
  EXPECT_TRUE(torch::allclose(intEng3, intEng, 1e-8, 1e-8));
  EXPECT_TRUE(torch::allclose(cs3, cs, 1e-8, 1e-8));

  auto c1 = std::chrono::steady_clock::now();
  std::cout << "CPU wall-clock time: "
            << std::chrono::duration<double, std::milli>(c1 - c0).count()
            << " ms\n";
}

TEST(TestANEOSThermo, cuda) {
  if (!snapy_cuda_test_enabled()) {
    GTEST_SKIP() << "CUDA is not available, skipping test.";
  }

  ANEOSThermo thermo("example.aneos");

  thermo->pretty_print(std::cout);
  thermo->to(torch::kCUDA);

  auto opts = torch::dtype(torch::kDouble).device(torch::kCUDA);
  auto rho = torch::full({6000, 6000}, 2.5e3, opts);
  auto intEng = torch::full({6000, 6000}, 5.0e9, opts);

  auto c0 = std::chrono::steady_clock::now();

  auto [pres, temp, cs] = thermo->compute("DU->PTL", {rho, intEng});
  std::cout << "pres = " << pres.index({0, Slice(0, 1)}) << std::endl;
  std::cout << "temp = " << temp.index({0, Slice(0, 1)}) << std::endl;
  std::cout << "cs = " << cs.index({0, Slice(0, 1)}) << std::endl;
  std::cout << "pres device = " << pres.device() << '\n';

  thermo->cache["temp"] = temp;

  auto [temp2, intEng2, cs2] = thermo->compute("DP->TUL", {rho, pres});
  std::cout << "temp2 = " << temp2.index({0, Slice(0, 1)}) << std::endl;
  std::cout << "intEng2 = " << intEng2.index({0, Slice(0, 1)}) << std::endl;
  std::cout << "cs2 = " << cs2.index({0, Slice(0, 1)}) << std::endl;

  auto [pres2, intEng3, cs3] = thermo->compute("DT->PUL", {rho, temp});
  std::cout << "pres2 = " << pres2.index({0, Slice(0, 1)}) << std::endl;
  std::cout << "intEng3 = " << intEng3.index({0, Slice(0, 1)}) << std::endl;
  std::cout << "cs3 = " << cs3.index({0, Slice(0, 1)}) << std::endl;

  EXPECT_TRUE(torch::allclose(temp2, temp, 1e-8, 1e-8));
  EXPECT_TRUE(torch::allclose(intEng2, intEng, 1e-8, 1e-8));
  EXPECT_TRUE(torch::allclose(cs2, cs, 1e-8, 1e-8));
  EXPECT_TRUE(torch::allclose(pres2, pres, 1e-8, 1e-8));
  EXPECT_TRUE(torch::allclose(intEng3, intEng, 1e-8, 1e-8));
  EXPECT_TRUE(torch::allclose(cs3, cs, 1e-8, 1e-8));

  auto c1 = std::chrono::steady_clock::now();
  std::cout << "GPU wall-clock time: "
            << std::chrono::duration<double, std::milli>(c1 - c0).count()
            << " ms\n";
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
