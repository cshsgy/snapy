// Isothermal rigid-lid gravity mode through snapy's well-balanced x1 path.
// One mode, four density references, T1 vertical resolutions. The table is the
// result; a large frequency error is reported, not treated as a build failure.
// The run stops only when the resting column is not quiet or the initial
// projection is not the analytic mode.

#include <snap/snap.h>

#include <snap/mesh/meshblock.hpp>

#include <kintera/constants.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace snap;

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kH = 1.;
constexpr double kGamma = 1.4;
constexpr double kG = 1.;
constexpr double kL = 4.;
constexpr double kKx = 1.;
constexpr double kKz = kPi / kL;
constexpr double kCs2 = kGamma * kG * kH;
constexpr double kN2 = (kGamma - 1.) / kGamma * kG / kH;
constexpr double kAmp = 1.e-4;
constexpr int kNx2 = 64;
constexpr double kCfl = 0.4;
constexpr int kSamples = 40;
constexpr char kOut[] = "/tmp/snapy-250-t2-modes.md";

double gravity_omega() {
  double B = kCs2 * (kKx * kKx + kKz * kKz + 1. / (4. * kH * kH));
  double C = kN2 * kCs2 * kKx * kKx;
  double disc = std::sqrt(B * B - 4. * C);
  return std::sqrt(0.5 * (B - disc));
}

struct Sample {
  double time = 0.;
  double ac = 0.;
  double as = 0.;
  double phase = 0.;
  double eigen_rel = 0.;
  double speed = 0.;
  bool finite = true;
};

struct Row {
  std::string form;
  int nx1 = 0;
  double dz = 0.;
  double omega_true = 0.;
  double omega_fit = 0.;
  double rel_freq = 0.;
  double eigen_rel = 0.;
  double amp_ratio = 0.;
  double dt = 0.;
  int nstep = 0;
  double t_end = 0.;
  double max_speed = 0.;
  bool finite = false;
  std::string note;
};

int env_max_nx() {
  char const* v = std::getenv("SNAPY_250_MAX_NX");
  if (!v || !*v) return 1000000;
  return std::atoi(v);
}

bool rest_only() {
  char const* v = std::getenv("SNAPY_250_REST_ONLY");
  return v && std::string(v) == "1";
}

void write_table(std::vector<Row> const& rows, std::string const& preamble) {
  std::ofstream out(kOut);
  out << preamble;
  out << "\n| form | dz/H | nx1 | omega_true | omega_fit | rel_freq | eigen_rel "
         "| amp_ratio | dt | nstep | max_speed | note |\n";
  out << "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n";
  out << std::scientific << std::setprecision(6);
  for (auto const& r : rows) {
    out << "| " << r.form << " | " << r.dz << " | " << r.nx1 << " | "
        << r.omega_true << " | " << r.omega_fit << " | " << r.rel_freq << " | "
        << r.eigen_rel << " | " << r.amp_ratio << " | " << r.dt << " | "
        << r.nstep << " | " << r.max_speed << " | " << r.note << " |\n";
  }
  out.flush();
}

std::shared_ptr<MeshBlockImpl> make_block(int nx1, std::string const& form) {
  auto options = MeshBlockOptionsImpl::from_yaml("test_rho_ref_mode.yaml");
  auto coord = options->coord();
  coord->nx1(nx1);
  coord->global_nx1(nx1);
  coord->x1min(0.);
  coord->x1max(kL);
  coord->global_x1min(0.);
  coord->global_x1max(kL);
  coord->nx2(kNx2);
  coord->global_nx2(kNx2);
  coord->x2min(0.);
  coord->x2max(2. * kPi);
  coord->global_x2min(0.);
  coord->global_x2max(2. * kPi);
  options->hydro()->eos()->weight(kintera::constants::Rgas);
  options->hydro()->eos()->gammad(kGamma);
  options->hydro()->eos()->limiter(false);
  options->hydro()->rho_ref(form);
  options->hydro()->icorr() = nullptr;
  options->intg()->type("rk3");
  options->intg()->cfl(kCfl);
  auto block = std::make_shared<MeshBlockImpl>(options);
  block->to(torch::kCPU, torch::kFloat64);
  return block;
}

void fill_mode(MeshBlockImpl& block, torch::Tensor w, double amp, double omega) {
  auto coord = block.pcoord;
  auto z = coord->x1v;
  auto rho0 = torch::exp(-z / kH);
  auto wh = torch::exp(z / (2. * kH)) * torch::sin(kKz * z);
  auto wz = torch::exp(z / (2. * kH)) *
            (torch::sin(kKz * z) / (2. * kH) + kKz * torch::cos(kKz * z));
  auto real_factor =
      omega * rho0 * (kG * wh - kCs2 * wz) / (kCs2 * kKx * kKx - omega * omega);
  auto u_factor = kKx * real_factor / (omega * rho0);
  auto rho_factor = (-kKx * rho0 * u_factor + rho0 * (wh / kH - wz)) / omega;
  auto x = coord->x2v.view({1, coord->options->nc2(), 1});
  auto sink = torch::sin(kKx * x);
  auto cosk = torch::cos(kKx * x);
  w[IDN] = rho0 - amp * rho_factor * sink;
  w[IPR] = rho0 + amp * real_factor * sink;
  w[IVX] = amp * wh * cosk;
  w[IVY] = amp * u_factor * sink;
}

Sample project(MeshBlockImpl& block, Variables const& vars, double time,
               double omega, torch::Tensor const& mode_cos,
               torch::Tensor const& mode_sin) {
  Sample s;
  s.time = time;
  auto W = block.phydro->peos->compute("U->W", {vars.at("hydro_u")});
  auto coord = block.pcoord;
  int il = coord->il(), iu = coord->iu();
  int jl = coord->jl(), ju = coord->ju();
  auto vx = W[IVX].slice(-1, il, iu + 1).slice(-2, jl, ju + 1);
  auto vy = W[IVY].slice(-1, il, iu + 1).slice(-2, jl, ju + 1);
  s.finite = torch::isfinite(vx).all().item<bool>() &&
             torch::isfinite(vy).all().item<bool>();
  s.speed = std::max(vx.abs().max().item<double>(), vy.abs().max().item<double>());
  if (!s.finite) return s;
  auto wsim = vx;
  double aa = (mode_cos * mode_cos).sum().item<double>();
  double bb = (mode_sin * mode_sin).sum().item<double>();
  double ab = (mode_cos * mode_sin).sum().item<double>();
  double wa = (wsim * mode_cos).sum().item<double>();
  double wb = (wsim * mode_sin).sum().item<double>();
  double det = aa * bb - ab * ab;
  s.ac = (wa * bb - wb * ab) / det;
  s.as = (wb * aa - wa * ab) / det;
  auto resid = wsim - s.ac * mode_cos - s.as * mode_sin;
  double num = resid.square().sum().item<double>();
  double den = wsim.square().sum().item<double>();
  s.eigen_rel = den > 0. ? std::sqrt(num / den) : 0.;
  s.phase = std::atan2(s.as, s.ac);
  (void)omega;
  return s;
}

double fit_omega(std::vector<Sample> const& samples) {
  // Unwrap atan2, then least-squares slope against time.
  double acc = 0.;
  double last = 0.;
  double st = 0., sp = 0., stt = 0., stp = 0.;
  int n = 0;
  for (auto const& s : samples) {
    if (!s.finite) break;
    if (n == 0) {
      acc = s.phase;
    } else {
      double d = s.phase - last;
      while (d > kPi) d -= 2. * kPi;
      while (d < -kPi) d += 2. * kPi;
      acc += d;
    }
    last = s.phase;
    st += s.time;
    sp += acc;
    stt += s.time * s.time;
    stp += s.time * acc;
    ++n;
  }
  double det = n * stt - st * st;
  if (n < 2 || det == 0.) return 0.;
  return (n * stp - st * sp) / det;
}

torch::Tensor mode_pattern(MeshBlockImpl& block, bool sine) {
  auto coord = block.pcoord;
  int il = coord->il(), iu = coord->iu();
  int jl = coord->jl(), ju = coord->ju();
  auto z = coord->x1v.slice(0, il, iu + 1);
  auto x = coord->x2v.slice(0, jl, ju + 1).view({1, ju - jl + 1, 1});
  auto wh = torch::exp(z / (2. * kH)) * torch::sin(kKz * z);
  auto ang = sine ? torch::sin(kKx * x) : torch::cos(kKx * x);
  return wh * ang;
}

Row run_case(std::string const& form, int nx1, double amp, double omega,
             double t_target) {
  Row row;
  row.form = form;
  row.nx1 = nx1;
  row.dz = kL / nx1 / kH;
  row.omega_true = amp == 0. ? 0. : omega;
  auto block = make_block(nx1, form);
  auto coord = block->pcoord;
  auto w = torch::zeros({block->phydro->peos->nvar(), coord->options->nc3(),
                         coord->options->nc2(), coord->options->nc1()},
                        torch::kFloat64);
  fill_mode(*block, w, amp, omega);
  Variables vars;
  vars["hydro_w"] = w;
  block->initialize(vars);
  TORCH_CHECK(block->pintg->stages.size() == 3u, "expected rk3");
  row.dt = block->max_time_step(vars);
  row.nstep = std::max(1, (int)std::llround(t_target / row.dt));
  auto mode_cos = mode_pattern(*block, false);
  auto mode_sin = mode_pattern(*block, true);

  std::vector<Sample> samples;
  samples.push_back(project(*block, vars, 0., omega, mode_cos, mode_sin));
  double time = 0.;
  int stride = std::max(1, row.nstep / kSamples);
  for (int step = 1; step <= row.nstep; ++step) {
    for (int stage = 0; stage < 3; ++stage) {
      block->advance_local(vars, row.dt, stage);
      // Periodic x2 ghosts are a layout exchange, not a boundary function.
      // Without this, the next stage sees the initial ghosts against an
      // updated interior and a horizontal velocity grows from a resting column.
      block->exchange_ghost_zones(vars);
    }
    time += row.dt;
    if (step % stride == 0 || step == row.nstep) {
      auto s = project(*block, vars, time, omega, mode_cos, mode_sin);
      samples.push_back(s);
      if (!s.finite) break;
    }
  }
  row.t_end = time;
  row.finite = samples.back().finite;
  row.max_speed = 0.;
  for (auto const& s : samples) row.max_speed = std::max(row.max_speed, s.speed);
  if (amp == 0.) {
    row.note = row.finite ? "rest" : "rest non-finite";
    return row;
  }
  if (!samples.front().finite || samples.front().eigen_rel > 1.e-6 ||
      std::abs(std::hypot(samples.front().ac, samples.front().as) / amp - 1.) >
          1.e-6) {
    row.note = "t=0 projection is not the analytic mode";
    row.eigen_rel = samples.front().eigen_rel;
    row.amp_ratio =
        std::hypot(samples.front().ac, samples.front().as) / amp;
    return row;
  }
  if (!row.finite) {
    row.note = "non-finite";
    return row;
  }
  row.omega_fit = fit_omega(samples);
  row.rel_freq = (row.omega_fit - omega) / omega;
  row.eigen_rel = samples.back().eigen_rel;
  row.amp_ratio = std::hypot(samples.back().ac, samples.back().as) / amp;
  row.note = "ok";
  return row;
}

void print_row(Row const& r) {
  std::cerr << std::scientific << std::setprecision(6) << r.form << " nx1="
            << r.nx1 << " dz=" << r.dz << " rel_freq=" << r.rel_freq
            << " eigen_rel=" << r.eigen_rel << " amp_ratio=" << r.amp_ratio
            << " max_speed=" << r.max_speed << " dt=" << r.dt
            << " nstep=" << r.nstep << " " << r.note << "\n";
}

}  // namespace

TEST(hydro, rho_ref_isothermal_mode) {
  double omega = gravity_omega();
  EXPECT_NEAR(omega, 0.40403230080237695, 1.e-12);
  double t_target = 2. * (2. * kPi / omega);
  std::ostringstream head;
  head << std::scientific << std::setprecision(16);
  head << "# T2 step 3: one isothermal gravity mode\n\n";
  head << "L=4H, H=1, gamma=1.4, g=1, Rd=Rgas/weight=1, rigid x1 lids, "
          "periodic x2.\n";
  head << "n=1, kx=1, nx2=64, amp=1e-4, rk3, cfl=0.4, lmars, weno5, nghost=3, "
          "no sponge, no implicit correction.\n";
  head << "Background is the cell-center exponential. Ghosts are the "
          "reflecting and periodic boundaries, not an analytic continuation. "
          "Periodic x2 ghosts are exchanged after every stage.\n";
  head << "Rest control: amp=0, smooth5, nx1=16, same two periods. "
          "max_speed is the max interior |velocity|. The run stops if that "
          "exceeds 1e-4 or the state is non-finite. At dz/H=0.25 the vertical "
          "truncation sits near 1e-6, so 1e-8 is not the control threshold.\n";
  head << "omega_true=" << omega << " (gravity root). Two periods, dt fixed "
          "from the initial CFL.\n";
  head << "omega_fit is the slope of unwrapped atan2(as, ac). eigen_rel is the "
          "L2 of vertical velocity orthogonal to span{w(z)cos(kx x), "
          "w(z)sin(kx x)}, over ||w_sim||, at the end. amp_ratio is "
          "hypot(ac, as)/amp at the end.\n";

  std::vector<Row> rows;
  auto rest = run_case("smooth5", 16, 0., omega, t_target);
  print_row(rest);
  rows.push_back(rest);
  write_table(rows, head.str());
  if (!rest.finite || rest.max_speed > 1.e-4) {
    ADD_FAILURE() << "rest column is not quiet: max interior speed "
                  << rest.max_speed;
    return;
  }
  if (rest_only()) {
    SUCCEED();
    return;
  }

  int max_nx = env_max_nx();
  int resolutions[] = {8, 16, 32, 80, 200};
  char const* forms[] = {"smooth5", "isentrope", "none", "local_polytrope"};
  for (int nx1 : resolutions) {
    if (nx1 > max_nx) continue;
    for (char const* form : forms) {
      auto row = run_case(form, nx1, kAmp, omega, t_target);
      print_row(row);
      rows.push_back(row);
      write_table(rows, head.str());
      if (row.note.find("t=0") != std::string::npos) {
        ADD_FAILURE() << row.form << " nx1=" << row.nx1 << " " << row.note
                      << " eigen_rel=" << row.eigen_rel;
        return;
      }
    }
  }
}
