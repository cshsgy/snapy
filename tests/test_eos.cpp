// external
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

// C/C++
#include <vector>

// torch
#include <torch/torch.h>

// kintera
#include <kintera/thermo/thermo.hpp>

// snapy
#include <snap/eos/fix_vapor_impl.h>
#include <snap/snap.h>

#include <cmath>
#include <snap/eos/ideal_moist.hpp>
#include <snap/mesh/meshblock.hpp>

// tests
#include "device_testing.hpp"

using namespace snap;

namespace {

std::shared_ptr<MeshBlockImpl> make_block(std::string eos_type) {
  auto options = MeshBlockOptionsImpl::from_yaml("test_eos.yaml");
  options->hydro()->eos()->type() = std::move(eos_type);
  return std::make_shared<MeshBlockImpl>(options);
}

}  // namespace

TEST(eos_limiter, accepts_zero_vapor_column) {
  std::vector<double> vapor(4, 0.);
  std::vector<double> major(4, 1.);

  EXPECT_EQ(fix_vapor_impl(vapor.data(), major.data(), vapor.size()), 0);
  for (double value : vapor) EXPECT_DOUBLE_EQ(value, 0.);

  vapor = {0.2, -0.1, 0., 0.};
  EXPECT_EQ(fix_vapor_impl(vapor.data(), major.data(), vapor.size()), 0);
  EXPECT_DOUBLE_EQ(vapor[0], 0.05);
  EXPECT_DOUBLE_EQ(vapor[1], 0.05);
  EXPECT_DOUBLE_EQ(vapor[2], 0.);
  EXPECT_DOUBLE_EQ(vapor[3], 0.);
}

namespace {

double column_sum(std::vector<double> const& v) {
  double s = 0.;
  for (double x : v) s += x;
  return s;
}

}  // namespace

// A bottom-cell deficit has nothing below it; the measured column
TEST(eos_limiter, repairs_bottom_cell_from_above) {
  std::vector<double> major = {1.00, 0.85, 0.72, 0.61};
  std::vector<double> vapor = {-1.7e-21, 4.4e-11, 1.0e-11, 1.0e-11};
  double before = column_sum(vapor);

  EXPECT_EQ(fix_vapor_impl(vapor.data(), major.data(), vapor.size()), 0);
  for (double value : vapor) EXPECT_GE(value, 0.);
  EXPECT_NEAR(column_sum(vapor), before, 1.e-14 * std::abs(before));

  // the repaired cell is zeroed, not left holding what it just borrowed
  EXPECT_DOUBLE_EQ(vapor[0], 0.);
  // Bounded transfer, asserted on the TRANSFER: the deficit is 1.7e-21 against
  // a vapor[1] of 4.4e-11, so any tolerance that survives round-off on the
  // latter also passes a supplier that gave nothing. The subtraction is exact
  // (Sterbenz), leaving only the one rounding in `vapor[1] -= take`.
  EXPECT_NEAR(4.4e-11 - vapor[1], 1.7e-21, 1.e-25);
  EXPECT_DOUBLE_EQ(vapor[2], 1.0e-11);
  EXPECT_DOUBLE_EQ(vapor[3], 1.0e-11);
}

// a column in net deficit is unrepairable: report it, leave it untouched
TEST(eos_limiter, rejects_column_in_net_deficit_without_writing) {
  std::vector<double> major(4, 1.);
  std::vector<double> vapor = {-0.3, 0.1, 0.1, 0.};
  std::vector<double> const expected = vapor;

  EXPECT_EQ(fix_vapor_impl(vapor.data(), major.data(), vapor.size()), 1);
  EXPECT_EQ(vapor, expected);
}

// nothing to borrow from in either direction
TEST(eos_limiter, rejects_single_cell_column) {
  std::vector<double> major = {1.};
  std::vector<double> vapor = {-1.e-30};

  EXPECT_EQ(fix_vapor_impl(vapor.data(), major.data(), 1), 1);
}

// a deficit the cells below can cover is still flattened as before
TEST(eos_limiter, downward_branch_is_unchanged) {
  std::vector<double> major = {1.00, 0.85, 0.72, 0.61};
  std::vector<double> vapor = {0.2, -0.1, 0., 0.};
  double before = column_sum(vapor);

  EXPECT_EQ(fix_vapor_impl(vapor.data(), major.data(), vapor.size()), 0);
  double const yfrac = 0.1 / 1.85;
  EXPECT_DOUBLE_EQ(vapor[0], yfrac * 1.00);
  EXPECT_DOUBLE_EQ(vapor[1], yfrac * 0.85);
  EXPECT_NEAR(column_sum(vapor), before, 1.e-14 * std::abs(before));
}

TEST_P(DeviceTest, moist_mixture) {
  auto block = make_block("moist-mixture");
  block->to(device, dtype);

  auto peos = block->phydro->peos;
  auto pcoord = block->pcoord;

  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();
  int nvar = peos->nvar();

  auto cons =
      torch::empty({nvar, nc3, nc2, nc1}, torch::device(device).dtype(dtype));

  cons.uniform_(0., 1.);

  cons[IDN].abs_();
  cons[IDN] += 1.E-6;  // avoid division by zero

  cons[IPR].abs_().mul_(10.);
  cons[IPR] += 0.5 * cons.narrow(0, IVX, 3).pow(2).sum() / cons[IDN];

  std::cout << "cons min = " << cons.min() << std::endl;
  std::cout << "cons max = " << cons.max() << std::endl;

  auto prim = peos->forward(cons);
  auto cons2 = peos->compute("W->U", {prim});

  EXPECT_TRUE(torch::allclose(cons, cons2, 1.E-6, 1.E-6));

  auto gamma = peos->compute("W->A", {prim});

  EXPECT_TRUE(gamma.allclose(torch::ones_like(gamma) * 1.4, 1.E-6, 1.E-6));

  auto cs = peos->compute("WA->L", {prim, gamma});
  std::cout << "cs min = " << cs.min() << std::endl;
  std::cout << "cs max = " << cs.max() << std::endl;

  // Cache validation must follow tensor identity and ATen's mutation version,
  // without retaining a full primitive-field copy. Mutating pressure must
  // invalidate the cached temperature, and a distinct tensor with identical
  // contents must still produce the same refreshed result.
  auto temp_before = peos->compute("W->T", {prim}).clone();
  prim[IPR].mul_(1.25);
  auto temp_after = peos->compute("W->T", {prim});
  auto temp_reference = peos->compute("W->T", {prim.clone()});
  EXPECT_FALSE(torch::allclose(temp_before, temp_after, 1.E-6, 1.E-6));
  EXPECT_TRUE(torch::allclose(temp_after, temp_reference, 1.E-6, 1.E-6));
}

TEST_P(DeviceTest, ideal_moist_internal_energy_offset) {
  auto block = make_block("ideal-moist");
  block->to(device, dtype);

  auto peos = block->phydro->peos;
  int nc1 = block->pcoord->options->nc1();
  int nc2 = block->pcoord->options->nc2();
  int nc3 = block->pcoord->options->nc3();
  int nvar = peos->nvar();

  auto hydro =
      torch::randn({nvar, nc3, nc2, nc1}, torch::device(device).dtype(dtype));
  auto offset = peos->internal_energy_offset(hydro);

  auto ideal_moist = std::dynamic_pointer_cast<IdealMoistImpl>(peos);
  ASSERT_TRUE(ideal_moist);

  auto expected = hydro[IDN] * ideal_moist->u0[0].to(dtype).to(device);
  EXPECT_TRUE(torch::allclose(offset, expected, 1.E-6, 1.E-6));

  auto tendency = 0.25 * hydro;
  auto tendency_offset = peos->internal_energy_offset(tendency);
  EXPECT_TRUE(torch::allclose(tendency_offset, 0.25 * offset, 1.E-6, 1.E-6));
}

TEST_P(DeviceTest, ideal_gas_internal_energy_offset_is_zero) {
  auto block = make_block("ideal-gas");
  block->to(device, dtype);

  auto peos = block->phydro->peos;
  int nc1 = block->pcoord->options->nc1();
  int nc2 = block->pcoord->options->nc2();
  int nc3 = block->pcoord->options->nc3();

  auto hydro = torch::randn({peos->nvar(), nc3, nc2, nc1},
                            torch::device(device).dtype(dtype));
  auto offset = peos->internal_energy_offset(hydro);
  EXPECT_TRUE(
      torch::allclose(offset, torch::zeros_like(hydro[IDN]), 1.E-12, 1.E-12));
}

/*TEST_P(DeviceTest, cons2prim_hydro_ideal_ncloud5) {
  int32_t NHYDRO = 14;
  int32_t ncloud = 5;
  int32_t nvapor = NHYDRO - ICY - ncloud;

  auto cons =
      torch::randn({NHYDRO, 1, 200, 200}, torch::device(device).dtype(dtype));
  auto gammad = torch::ones({1, 200, 200}, torch::device(device).dtype(dtype));
  auto rmu =
      torch::randn({nvapor + ncloud}, torch::device(device).dtype(dtype));
  auto rcv =
      torch::randn({nvapor + ncloud}, torch::device(device).dtype(dtype));

  gammad *= 1.4;
  rmu.normal_(0, 1);
  rcv.normal_(0, 1);

  auto start = std::chrono::high_resolution_clock::now();

  auto prim = eos_cons2prim_hydro_ideal(cons, gammad, rmu, rcv, ncloud);
  auto cons2 = eos_prim2cons_hydro_ideal(prim, gammad, rmu, rcv, ncloud);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Time taken by test body: " << elapsed.count() << " seconds"
            << std::endl;

  std::cout << (cons - cons2).min() << std::endl;
  std::cout << (cons - cons2).max() << std::endl;

  if (dtype == torch::kFloat32) {
    EXPECT_TRUE(torch::allclose(cons, cons2, 1.E-2, 1.E-2));
  } else {
    EXPECT_TRUE(torch::allclose(cons, cons2, 1.E-9, 1.E-9));
  }
}

TEST_P(DeviceTest, prim2cons_hydro_ideal_ncloud5) {
  int32_t NHYDRO = 14;
  int32_t ncloud = 5;
  int32_t nvapor = NHYDRO - ICY - ncloud;

  auto prim =
      torch::randn({NHYDRO, 1, 5, 5}, torch::device(device).dtype(dtype));
  auto gammad = torch::ones({1, 5, 5}, torch::device(device).dtype(dtype));
  auto rmu =
      torch::randn({nvapor + ncloud}, torch::device(device).dtype(dtype));
  auto rcv =
      torch::randn({nvapor + ncloud}, torch::device(device).dtype(dtype));

  gammad *= 1.4;
  rmu.normal_(0, 1);
  rcv.normal_(0, 1);

  auto cons = eos_prim2cons_hydro_ideal(prim, gammad, rmu, rcv, ncloud);
  auto prim2 = eos_cons2prim_hydro_ideal(cons, gammad, rmu, rcv, ncloud);

  std::cout << (prim - prim2).abs().max() << std::endl;

  if (dtype == torch::kFloat32) {
    EXPECT_TRUE(torch::allclose(prim, prim2, 1.E-4, 1.E-4));
  } else {
    EXPECT_TRUE(torch::allclose(prim, prim2, 1.E-12, 1.E-12));
  }
}*/

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
