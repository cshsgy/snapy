// C/C++
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>

// external
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

// snap
#include <snap/snap.h>

#include <snap/coord/coordinate.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/implicit/implicit_hydro.hpp>
#include <snap/mesh/meshblock.hpp>

using namespace snap;

namespace {
//! the one card this binary loads: kintera's species table is process-global
constexpr char const* kCard = "test_cycle_diagnostics.yaml";

//! read a `key=<double>` token off the cycle line
double read_token(std::string const& out, std::string const& key, bool* found) {
  auto pos = out.find(key);
  if (found) *found = pos != std::string::npos;
  if (pos == std::string::npos) return 0.;
  return std::stod(out.substr(pos + key.size()));
}

//! build a uniform block and return the ke it logs
double logged_ke(double rho, double vx, double vy, bool double_momentum) {
  auto options = MeshBlockOptionsImpl::from_yaml(kCard);
  auto block = std::make_shared<MeshBlockImpl>(options);
  auto coord = block->pcoord;

  auto w = torch::zeros({block->phydro->peos->nvar(), coord->options->nc3(),
                         coord->options->nc2(), coord->options->nc1()},
                        torch::kFloat64);
  w[IDN].fill_(rho);
  w[IVX].fill_(vx);
  w[IVY].fill_(vy);
  w[ICY].fill_(
      0.1);  // a real mixture, so the constituent sum is not a formality
  w[IPR].fill_(1.e5);

  Variables vars;
  vars["hydro_w"] = w;
  block->initialize(vars);
  block->pintg->options->ncycle_out(1);

  // a real cycle advances u only; hydro_w is a stage stale at print time
  if (double_momentum) {
    vars.at("hydro_u").narrow(0, IVX, 3).mul_(2.);
    vars.at("hydro_w").zero_();  // print_cycle_info must read u alone
  }

  testing::internal::CaptureStdout();
  block->print_cycle_info(vars, 0., 1.);
  std::string out = testing::internal::GetCapturedStdout();

  bool found = false;
  double ke = read_token(out, "ke=", &found);
  EXPECT_TRUE(found) << out;
  return ke;
}
}  // namespace

TEST(cycle_info, logged_ke_scales_with_density) {
  double ke1 = logged_ke(2.5, 3., 4., false);
  double ke2 = logged_ke(5.0, 3., 4., false);
  ASSERT_GT(ke1, 0.);
  EXPECT_NEAR(ke2, 2. * ke1, 1.e-9 * ke2)
      << "ke(rho=2.5)=" << ke1 << " ke(rho=5.0)=" << ke2;
}

TEST(cycle_info, logged_ke_reads_the_conserved_state) {
  double ke0 = logged_ke(2.5, 3., 4., false);
  double ke2 = logged_ke(2.5, 3., 4., true);
  ASSERT_GT(ke0, 0.);
  // ke = |p|^2/2rho quadruples; reading the zeroed hydro_w would log NaN
  EXPECT_NEAR(ke2, 4. * ke0, 1.e-9 * ke2)
      << "ke=" << ke0 << " after doubling momentum=" << ke2;
}

// The meters at the end of the cycle line accumulate over the whole run and
// nothing resets them, so the line has to say so before it prints them.
TEST(cycle_info, meter_group_is_marked_run_to_date) {
  auto options = MeshBlockOptionsImpl::from_yaml("test_cycle_diagnostics.yaml");
  auto block = std::make_shared<MeshBlockImpl>(options);
  auto coord = block->pcoord;

  auto w = torch::zeros({block->phydro->peos->nvar(), coord->options->nc3(),
                         coord->options->nc2(), coord->options->nc1()},
                        torch::kFloat64);
  w[IDN].fill_(1.5);
  w[ICY].fill_(0.1);
  w[IPR].fill_(1.e5);

  Variables vars;
  vars["hydro_w"] = w;
  block->initialize(vars);
  block->pintg->options->ncycle_out(1);

  // limcut= is printed only when the flux meter is nonzero
  block->phydro->lim_cut().fill_(3.);
  block->phydro->lim_flux().fill_(6.);

  testing::internal::CaptureStdout();
  block->print_cycle_info(vars, 0., 1.);
  std::string out = testing::internal::GetCapturedStdout();

  auto marker = out.find(" run-to-date:");
  ASSERT_NE(marker, std::string::npos)
      << "the cycle line must say that the meters after it are run-to-date "
         "accumulators, not per-cycle values\n"
      << out;

  // the marker introduces the meters only: dt is a genuine per-cycle value
  auto dt_pos = out.find(" dt=");
  ASSERT_NE(dt_pos, std::string::npos) << out;
  EXPECT_LT(dt_pos, marker) << out;

  // present AND after the marker: also pins the token spellings old logs use
  for (std::string token : {" limcut=", " thetamin=", " thetasevere="}) {
    auto pos = out.find(token);
    ASSERT_NE(pos, std::string::npos) << token << " missing from\n" << out;
    EXPECT_LT(marker, pos) << token << " printed ahead of the marker\n" << out;
  }
}

namespace {
struct Logged {
  double ke = 0.;
  double pe = 0.;
  bool pe_printed = false;
  //! interior sum of x1v * cell volume, read off the block's own grid
  double zvol = 0.;
};

//! Write a conserved state straight into hydro_u and return what the cycle line
//! prints for it, with cosine_cell_kj forced to `cth`. print_cycle_info reads
//! hydro_u alone, so nothing here depends on the EOS's W->U map.
Logged logged_state(double rho_dry, double rho_vapor, double rho_cloud,
                    double m1, double m2, double m3, double cth) {
  auto options = MeshBlockOptionsImpl::from_yaml(kCard);
  auto block = std::make_shared<MeshBlockImpl>(options);
  auto coord = block->pcoord;

  auto w = torch::zeros({block->phydro->peos->nvar(), coord->options->nc3(),
                         coord->options->nc2(), coord->options->nc1()},
                        torch::kFloat64);
  w[IDN].fill_(1.);
  w[ICY].fill_(0.1);
  w[IPR].fill_(1.e5);

  Variables vars;
  vars["hydro_w"] = w;
  block->initialize(vars);
  block->pintg->options->ncycle_out(1);

  auto u = vars.at("hydro_u");
  EXPECT_EQ(u.size(0) - ICY, 2) << "the fixture must carry condensate rows";
  u.zero_();
  u[IDN].fill_(rho_dry);
  u[ICY].fill_(rho_vapor);
  u[ICY + 1].fill_(rho_cloud);
  u[IVX].fill_(m1);
  u[IVY].fill_(m2);
  u[IVZ].fill_(m3);
  u[IPR].fill_(1.e5);
  coord->cosine_cell_kj.fill_(cth);

  testing::internal::CaptureStdout();
  block->print_cycle_info(vars, 0., 1.);
  std::string out = testing::internal::GetCapturedStdout();

  Logged got;
  bool has_ke = false;
  got.ke = read_token(out, "ke=", &has_ke);
  EXPECT_TRUE(has_ke) << out;
  got.pe = read_token(out, "pe=", &got.pe_printed);

  auto interior = block->part({0, 0, 0}, PartOptions().exterior(false));
  auto ones = torch::ones_like(u[IDN]).unsqueeze(0);
  got.zvol = (ones * coord->x1v * coord->cell_volume())
                 .index(interior)
                 .sum()
                 .item<double>();
  return got;
}
}  // namespace

// The denominator is the TOTAL density, dry plus every condensate row: two
// states with the same total mass and momentum must log the same ke, and a
// dry-density denominator reads the second of them 25% high.
TEST(cycle_info, logged_ke_sums_the_constituents) {
  double ke_total = logged_state(2.5, 0., 0., 0., 7.5, 0., 0.).ke;
  double ke_mixed = logged_state(2.0, 0.25, 0.25, 0., 7.5, 0., 0.).ke;
  double ke_dry_only = logged_state(2.0, 0., 0., 0., 7.5, 0., 0.).ke;
  ASSERT_GT(ke_total, 0.);
  EXPECT_NEAR(ke_mixed, ke_total, 1.e-9 * ke_total);
  EXPECT_NEAR(ke_dry_only, 1.25 * ke_total, 1.e-9 * ke_total);
}

// ke is a contraction with the metric, not a sum of squares, and only a nonzero
// cos(theta) between x2 and x3 can tell the two apart. With c = 0.5 and
// (m2, m3) = (3, 4): (m2^2 - 2 c m2 m3 + m3^2) / (1 - c^2) = 13 / 0.75,
// against 25 flat, i.e. 13/18.75 of the orthogonal answer.
TEST(cycle_info, logged_ke_raises_the_momentum_with_the_metric) {
  double ke_flat = logged_state(2.5, 0., 0., 0., 3., 4., 0.).ke;
  double ke_skew = logged_state(2.5, 0., 0., 0., 3., 4., 0.5).ke;
  ASSERT_GT(ke_flat, 0.);
  EXPECT_NEAR(ke_skew, ke_flat * 13. / 18.75, 1.e-9 * ke_flat)
      << "flat=" << ke_flat << " skew=" << ke_skew;
}

// `pe=` is printed only when the card carries a const-gravity forcing, and it
// weighs the total density, like ke. grav1 = -10 in the card.
TEST(cycle_info, logged_pe_is_the_column_geopotential) {
  auto dry = logged_state(2.5, 0., 0., 0., 0., 0., 0.);
  ASSERT_TRUE(dry.pe_printed) << "no pe= on the cycle line";
  ASSERT_GT(dry.zvol, 0.);
  EXPECT_NEAR(dry.pe, 2.5 * 10. * dry.zvol, 1.e-9 * std::abs(dry.pe))
      << "pe=" << dry.pe << " sum z V=" << dry.zvol;

  auto mixed = logged_state(2.0, 0.25, 0.25, 0., 0., 0., 0.);
  EXPECT_NEAR(mixed.pe, dry.pe, 1.e-9 * std::abs(dry.pe));
}

namespace {
//! kCard edited in place and written out: same species, so the table agrees
std::shared_ptr<MeshBlockImpl> block_from(YAML::Node card,
                                          std::string const& name,
                                          Variables* vars) {
  {
    std::ofstream(name) << card;
  }
  auto block =
      std::make_shared<MeshBlockImpl>(MeshBlockOptionsImpl::from_yaml(name));
  std::remove(name.c_str());

  auto coord = block->pcoord;
  auto w = torch::zeros({block->phydro->peos->nvar(), coord->options->nc3(),
                         coord->options->nc2(), coord->options->nc1()},
                        torch::kFloat64);
  w[IDN].fill_(1.);
  w[IPR].fill_(1.e5);
  w[ICY].fill_(0.01);
  w[ICY + 1].fill_(0.02);
  (*vars)["hydro_w"] = w;
  block->initialize(*vars);
  block->pintg->options->ncycle_out(1);
  return block;
}
}  // namespace

// A resting column of six unit cells, cloud settling at const-vsed = -2, one
// step of dt = 1: each face carries rho*y*vsed of the cell above it, so every
// cell but the sealed bottom one drains twice what it holds and theta =
// dx / (dt |vsed|) = 0.5; the bottom cell and the vapour keep theta = 1. The
// limiter then halves each of the five interior x1 faces. grav1 is -1e-12, not
// 0 (sedimentation is a null op at 0, and const-vsed does not read grav1), so
// the Riemann flux of the column is ~1e-15 of the settling flux.
TEST(cycle_info, positivity_meters_read_their_hand_computed_values) {
  auto card = YAML::LoadFile(kCard);
  card["forcing"]["const-gravity"]["grav1"] = -1.e-12;
  card["dynamics"]["equation-of-state"]["limiter"] = true;
  card["sedimentation"] =
      YAML::Load("{radius: {}, density: {}, const-vsed: {cloud: -2.}}");
  Variables vars;
  auto block = block_from(card, "test_cycle_diagnostics_meters.yaml", &vars);
  auto hydro = block->phydro;

  int il = block->pcoord->il();
  double rho_cloud = vars.at("hydro_u")[ICY + 1][0][0][il].item<double>();
  ASSERT_GT(rho_cloud, 0.);
  hydro->forward(1., vars.at("hydro_u"), vars);

  EXPECT_NEAR(hydro->positivity_min()[0].item<double>(), 0.5, 1.e-9);
  EXPECT_EQ(hydro->positivity_severe()[0].item<int64_t>(), 5);
  double flux = hydro->lim_flux()[0].item<double>();
  EXPECT_NEAR(flux, 5 * 2. * rho_cloud, 1.e-9 * flux);
  EXPECT_NEAR(hydro->lim_cut()[0].item<double>() / flux, 0.5, 1.e-9);

  testing::internal::CaptureStdout();
  block->print_cycle_info(vars, 0., 1.);
  std::string out = testing::internal::GetCapturedStdout();
  bool found = false;
  EXPECT_NEAR(read_token(out, " limcut=", &found), 0.5, 1.e-5) << out;
  EXPECT_TRUE(found) << out;
  EXPECT_NEAR(read_token(out, " thetamin=", &found), 0.5, 1.e-5) << out;
  EXPECT_TRUE(found) << out;
  EXPECT_EQ(read_token(out, " thetasevere=", &found), 5.) << out;
  EXPECT_TRUE(found) << out;
}

// Two cells, so one interior face carries the whole transfer M. Unclamped, a
// cell's constituents change by exactly M(i) - M(i+1) and the meter reads 0.
// With every constituent's availability driven negative in both cells, that
// face moves dry gas only (a fraction 1 - sum(y) of M), so each cell misses
// M * sum(y) against a scale of |M|: the meter reads sum(y) = 0.03, whatever M
// is.
TEST(cycle_info, vicclamp_reads_the_clamped_fraction) {
  auto card = YAML::LoadFile(kCard);
  card["geometry"]["bounds"]["x1max"] = 2.;
  card["geometry"]["cells"]["nx1"] = 2;
  card["integration"] = YAML::Load("{type: rk3, cfl: 0.9, implicit-scheme: 1}");
  Variables vars;
  auto block = block_from(card, "test_cycle_diagnostics_vic.yaml", &vars);
  auto picorr = block->phydro->picorr;
  ASSERT_TRUE(picorr);

  auto w = vars.at("hydro_w").clone();
  auto gamma = block->phydro->peos->compute("W->A", {w});
  int il = block->pcoord->il();
  auto solve = [&](bool starve) {
    auto du = torch::zeros_like(w);
    // an energy tendency in the lower cell only, so that M is not zero
    du[IPR].select(-1, il).fill_(1.e3);
    if (starve) {
      for (int n = ICY; n <= ICY + 1; ++n) du[n] = -2. * w[IDN] * w[n];
    }
    picorr->forward(du, w.clone(), gamma, 1.);
    return picorr->mass_correction()[IVX][0][0][il + 1].item<double>();
  };

  double m_free = solve(false);
  ASSERT_NE(m_free, 0.);
  EXPECT_LT(picorr->clamp_residual()[0].item<double>(), 1.e-12);

  double m_starved = solve(true);
  ASSERT_NE(m_starved, 0.);
  EXPECT_NEAR(picorr->clamp_residual()[0].item<double>(), 0.03, 1.e-12);

  testing::internal::CaptureStdout();
  block->print_cycle_info(vars, 0., 1.);
  std::string out = testing::internal::GetCapturedStdout();
  bool found = false;
  EXPECT_NEAR(read_token(out, " vicclamp=", &found), 0.03, 1.e-6) << out;
  EXPECT_TRUE(found) << out;
}
