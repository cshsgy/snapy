// C/C++
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

// yaml
#include <yaml-cpp/yaml.h>

// kintera
#include <kintera/constants.h>

#include <kintera/thermo/thermo.hpp>

// snap
#include <snap/snap.h>

#include <snap/hydro/balance_column.hpp>
#include <snap/mesh/mesh.hpp>

using namespace snap;

namespace {

struct RunConfig {
  std::string input_file;
  std::string restart_file;
};

RunConfig ParseArguments(int argc, char** argv,
                         std::string const& default_input) {
  RunConfig cfg{default_input, ""};
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if ((arg == "-r" || arg == "--restart") && i + 1 < argc) {
      cfg.restart_file = argv[++i];
    } else if ((arg == "-i" || arg == "--input") && i + 1 < argc) {
      cfg.input_file = argv[++i];
    } else {
      cfg.input_file = arg;
    }
  }
  return cfg;
}

int species_offset(std::vector<std::string> const& species,
                   std::string const& name) {
  for (int n = 1; n < species.size(); ++n) {
    if (species[n] == name) return n - 1;
  }
  return -1;
}

torch::Tensor bryan_saturation_pressure(torch::Tensor const& temp) {
  constexpr double kT3 = 273.16;
  constexpr double kP3 = 611.7;
  constexpr double kBeta = 24.845;
  constexpr double kEps = 0.621;
  constexpr double kGamma = 1.4;
  constexpr double kRcpVapor = 1.166;
  constexpr double kRcpLiquid = 3.46;
  constexpr double kDelta =
      (kRcpLiquid - kRcpVapor) * kEps / (1. - 1. / kGamma);

  auto reduced_temp = temp / kT3;
  return kP3 * torch::exp(kBeta * (1. - 1. / reduced_temp) -
                          kDelta * torch::log(reduced_temp));
}

void set_user_output_callback(MeshBlock block,
                              std::vector<std::string> const& species,
                              double p0) {
  int iH2O = species_offset(species, "H2O");
  int iH2Oc = species_offset(species, "H2O(l)");

  block->user_output_callback = [iH2O, iH2Oc, p0](Variables const& vars) {
    constexpr double kRd = 287.;
    constexpr double kEps = 0.621;
    constexpr double kGamma = 1.4;
    constexpr double kRcpVapor = 1.166;
    constexpr double kRcpLiquid = 3.46;
    constexpr double kBeta = 24.845;
    constexpr double kT3 = 273.16;
    constexpr double kDelta =
        (kRcpLiquid - kRcpVapor) * kEps / (1. - 1. / kGamma);

    constexpr double kRv = kRd / kEps;
    constexpr double kCpd = kGamma / (kGamma - 1.) * kRd;
    constexpr double kCpLiquid = kRcpLiquid * kCpd;

    auto w = vars.at("hydro_w");
    auto qtol = torch::zeros_like(w[IDN]);
    auto qv = torch::zeros_like(w[IDN]);
    auto qc = torch::zeros_like(w[IDN]);

    if (iH2O >= 0) qv = w[ICY + iH2O];
    if (iH2Oc >= 0) qc = w[ICY + iH2Oc];
    qtol = qv + qc;

    auto qd = torch::clamp_min(1. - qtol, 1.e-12);
    auto feps = 1. + qv * (1. / kEps - 1.) - qc;
    auto temp = w[IPR] / (w[IDN] * kRd * feps);

    auto eta = qv / (qd * kEps);
    auto xgas = 1. + eta;
    auto pd = w[IPR] / xgas;
    auto pv = w[IPR] * eta / xgas;
    auto rh = torch::clamp_min(pv / bryan_saturation_pressure(temp), 1.e-12);

    auto cpt = kCpd * qd + kCpLiquid * qtol;
    auto lv = kRv * (kBeta * kT3 - kDelta * temp);
    auto theta_e = temp * torch::pow(p0 / pd, kRd * qd / cpt) *
                   torch::pow(rh, -kRv * qv / cpt) *
                   torch::exp(lv * qv / (cpt * temp));

    Variables out;
    out["qtol"] = qtol;
    out["theta_e"] = theta_e;
    return out;
  };
}

torch::Tensor surface_mass_fractions(std::vector<std::string> const& species,
                                     int nc3, int nc2, double qt,
                                     torch::TensorOptions const& options) {
  int ny = static_cast<int>(species.size()) - 1;
  auto yfrac = torch::zeros({ny, nc3, nc2}, options);

  int iH2O = species_offset(species, "H2O");
  if (iH2O >= 0) {
    yfrac[iH2O].fill_(qt);
  }

  return yfrac;
}

void solve_virtual_temperature_perturbation(
    kintera::ThermoX& thermo_x, torch::Tensor const& temp0,
    torch::Tensor const& pres, torch::Tensor const& xfrac0,
    torch::Tensor const& target_tv, torch::Tensor const& mask, double dT,
    double Rd, torch::Tensor& temp_out, torch::Tensor& xfrac_out) {
  auto temp_lo = temp0.clone();
  auto temp_hi = temp0 + std::max(5.0, 2.0 * std::abs(dT));

  for (int iter = 0; iter < 32; ++iter) {
    auto temp_mid = 0.5 * (temp_lo + temp_hi);
    auto xtrial = xfrac0.clone();
    thermo_x->forward(temp_mid, pres, xtrial);

    auto conc = thermo_x->compute("TPX->V", {temp_mid, pres, xtrial});
    auto dens = thermo_x->compute("V->D", {conc});
    auto tv_mid = pres / (dens * Rd);
    auto too_cold = torch::logical_and(mask, tv_mid < target_tv);

    temp_lo = torch::where(too_cold, temp_mid, temp_lo);
    temp_hi =
        torch::where(torch::logical_and(mask, torch::logical_not(too_cold)),
                     temp_mid, temp_hi);
  }

  temp_out = torch::where(mask, 0.5 * (temp_lo + temp_hi), temp0);
  xfrac_out = xfrac0.clone();
  thermo_x->forward(temp_out, pres, xfrac_out);
}

// Leftover |a|/g after the projection. 1e-14 * g * (10 H / c) stays under
// Mach 1e-12 on this column, so a pass is not the default 1e-10 cutoff.
constexpr double kBalanceRtol = 1.e-14;

struct BalanceReport {
  bool ran = false;
  double residual = 0.;
  int sweeps = 0;
};
BalanceReport g_balance;

struct MachSample {
  double mach = 0.;
  double mach_x = 0.;
  double mach_y = 0.;
  double mach_core = 0.;
  double mach_half = 0.;
  double mach_center = 0.;
  double vx = 0.;
  double vy = 0.;
  double z = 0.;
  double x = 0.;
  double core_z = 0.;
  double core_x = 0.;
  int cycle = 0;
  double time = 0.;
};

torch::Tensor interior_prim(MeshBlock block, torch::Tensor const& prim) {
  auto pcoord = block->pcoord;
  auto op = pcoord->options;
  return prim.narrow(1, pcoord->kl(), op->nx3())
      .narrow(2, pcoord->jl(), op->nx2())
      .narrow(3, pcoord->il(), op->nx1());
}

torch::Tensor primitives_of(MeshBlock block, Variables const& vars) {
  return block->phydro->peos->forward(vars.at("hydro_u").clone());
}

double sound_min(MeshBlock block, torch::Tensor const& prim) {
  auto wi = interior_prim(block, prim);
  auto gamma = block->phydro->peos->compute("W->A", {wi});
  auto cs = block->phydro->peos->compute("WA->L", {wi, gamma});
  return cs.min().item<double>();
}

MachSample mach_sample(MeshBlock block, torch::Tensor const& prim) {
  auto pcoord = block->pcoord;
  auto op = pcoord->options;
  auto wi = interior_prim(block, prim);
  auto gamma = block->phydro->peos->compute("W->A", {wi});
  auto cs = block->phydro->peos->compute("WA->L", {wi, gamma});
  auto mach_x = wi[IVX].abs() / cs;
  auto mach_y = wi[IVY].abs() / cs;
  auto speed = (wi[IVX].square() + wi[IVY].square() + wi[IVZ].square()).sqrt();
  auto mach = speed / cs;
  auto flat = mach.reshape({-1});
  int64_t idx = flat.argmax().item<int64_t>();
  int n1 = static_cast<int>(wi.size(3));
  int n2 = static_cast<int>(wi.size(2));
  int i = static_cast<int>(idx % n1);
  int j = static_cast<int>((idx / n1) % n2);
  int k = static_cast<int>(idx / (static_cast<int64_t>(n1) * n2));

  MachSample s;
  s.mach = flat[idx].item<double>();
  s.mach_x = mach_x.max().item<double>();
  s.mach_y = mach_y.max().item<double>();
  s.vx = wi[IVX][k][j][i].item<double>();
  s.vy = wi[IVY][k][j][i].item<double>();
  s.z = pcoord->x1v[pcoord->il() + i].item<double>();
  s.x = pcoord->x2v[pcoord->jl() + j].item<double>();
  // Drop the two x2 walls. A signal that lives only there is not the column.
  if (wi.size(2) > 2) {
    auto core = wi.narrow(2, 1, wi.size(2) - 2);
    auto gamma_c = block->phydro->peos->compute("W->A", {core});
    auto cs_c = block->phydro->peos->compute("WA->L", {core, gamma_c});
    auto mach_c =
        (core[IVX].square() + core[IVY].square() + core[IVZ].square()).sqrt() /
        cs_c;
    auto flat_c = mach_c.reshape({-1});
    int64_t idx_c = flat_c.argmax().item<int64_t>();
    int n1c = static_cast<int>(core.size(3));
    int n2c = static_cast<int>(core.size(2));
    int ic = static_cast<int>(idx_c % n1c);
    int jc = static_cast<int>((idx_c / n1c) % n2c);
    s.mach_core = flat_c[idx_c].item<double>();
    s.core_z = pcoord->x1v[pcoord->il() + ic].item<double>();
    s.core_x = pcoord->x2v[pcoord->jl() + 1 + jc].item<double>();
  }
  auto region_mach = [&](torch::Tensor region) {
    auto gamma_r = block->phydro->peos->compute("W->A", {region});
    auto cs_r = block->phydro->peos->compute("WA->L", {region, gamma_r});
    return ((region[IVX].square() + region[IVY].square() + region[IVZ].square())
                .sqrt() /
            cs_r)
        .max()
        .item<double>();
  };
  int nj = static_cast<int>(wi.size(2));
  if (nj >= 4) {
    s.mach_half = region_mach(wi.narrow(2, nj / 4, nj / 2));
    s.mach_center = region_mach(wi.narrow(2, nj / 2, 1));
  }
  return s;
}

void project_discrete_balance(MeshBlock block, torch::Tensor w, double grav) {
  auto pcoord = block->pcoord;
  auto op = pcoord->options;
  TORCH_CHECK(op->nx1() == op->global_nx1(),
              "discrete balance needs the whole x1 column in one block, nx1=",
              op->nx1(), " global_nx1=", op->global_nx1());
  TORCH_CHECK(block->phydro->options->wb_wall_clamp(),
              "discrete balance needs dynamics/wb-wall-clamp");
  int il = pcoord->il();
  int jl = pcoord->jl();
  int kl = pcoord->kl();
  int ni = op->nx1();
  int nj = op->nx2();
  int nk = op->nx3();
  auto col =
      w.narrow(1, kl, nk).narrow(2, jl, nj).narrow(3, il, ni).contiguous();
  auto dx = pcoord->dx1f.narrow(0, il, ni).contiguous();
  auto [balanced, err, sweeps] = snap::balance_column(
      col, dx, grav, /*wall_clamp=*/true, kBalanceRtol, /*max_iter=*/400);
  w.narrow(1, kl, nk).narrow(2, jl, nj).narrow(3, il, ni).copy_(balanced);
  g_balance.ran = true;
  g_balance.residual = std::max(g_balance.residual, err);
  g_balance.sweeps = std::max(g_balance.sweeps, sweeps);
  std::cout << std::scientific << std::setprecision(16)
            << "D1BALANCE sweeps=" << sweeps << " residual=" << err
            << " rtol=" << kBalanceRtol << std::endl;
}

void initialize_block(MeshBlock block, Variables& vars,
                      YAML::Node const& config, torch::Device const& device) {
  auto pcoord = block->pcoord;
  auto peos = block->phydro->peos;
  auto modules = block->named_modules();
  auto thermo_y = std::dynamic_pointer_cast<kintera::ThermoYImpl>(
      modules["hydro.eos.thermo"]);

  kintera::ThermoX thermo_x(thermo_y->options);
  thermo_x->to(device);

  auto const& species = thermo_y->options->species();
  int ny = static_cast<int>(species.size()) - 1;
  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();
  int il = pcoord->il();
  int iu = pcoord->iu();
  int nvar = peos->nvar();

  auto options = torch::TensorOptions().dtype(torch::kFloat64).device(device);
  auto w = torch::zeros({nvar, nc3, nc2, nc1}, options);

  double Ps = config["problem"]["p0"].as<double>();
  double Ts = config["problem"]["Ts"].as<double>();
  double xc = config["problem"]["xc"].as<double>();
  double zc = config["problem"]["zc"].as<double>();
  double xr = config["problem"]["xr"].as<double>();
  double zr = config["problem"]["zr"].as<double>();
  double dT = config["problem"]["dT"].as<double>();
  double qt = config["problem"]["qt"].as<double>();
  double grav = -config["forcing"]["const-gravity"]["grav1"].as<double>();

  auto temp_state = torch::zeros({nc3, nc2, nc1}, options);
  auto pres_state = torch::zeros({nc3, nc2, nc1}, options);
  auto xfrac_state = torch::zeros(
      std::vector<int64_t>{nc3, nc2, nc1, static_cast<int64_t>(species.size())},
      options);

  auto temp = Ts * torch::ones({nc3, nc2}, options);
  auto pres = Ps * torch::ones({nc3, nc2}, options);
  auto yfrac = surface_mass_fractions(species, nc3, nc2, qt, options);
  auto xfrac = thermo_y->compute("Y->X", {yfrac});
  thermo_x->forward(temp, pres, xfrac);

  double dz = pcoord->dx1f[il].item<double>();
  thermo_x->extrapolate_dz(
      temp, pres, xfrac,
      kintera::ExtrapOptions().dz(0.5 * dz).grav(grav).ds_dz(0.).rainout(
          false));

  for (int i = il; i <= iu; ++i) {
    temp_state.select(2, i).copy_(temp);
    pres_state.select(2, i).copy_(pres);
    xfrac_state.select(2, i).copy_(xfrac);

    if (i < iu) {
      dz = pcoord->dx1f[i].item<double>();
      thermo_x->extrapolate_dz(
          temp, pres, xfrac,
          kintera::ExtrapOptions().dz(dz).grav(grav).ds_dz(0.).rainout(false));
    }
  }

  auto x2 = pcoord->x2v.view({1, nc2}).expand({nc3, nc2});
  double Rd = kintera::constants::Rgas / thermo_x->mu[0].item<double>();

  for (int i = il; i <= iu; ++i) {
    double x1 = pcoord->x1v[i].item<double>();
    auto L = torch::sqrt(torch::square((x2 - xc) / xr) +
                         std::pow((x1 - zc) / zr, 2));
    auto mask = L < 1.;
    auto amp = dT * torch::square(torch::cos(0.5 * M_PI * L)) / 300.;

    auto temp_i = temp_state.select(2, i);
    auto pres_i = pres_state.select(2, i);
    auto xfrac_i = xfrac_state.select(2, i);
    auto conc_i = thermo_x->compute(
        "TPX->V", std::vector<torch::Tensor>{temp_i, pres_i, xfrac_i});
    auto dens_i = thermo_x->compute("V->D", std::vector<torch::Tensor>{conc_i});
    auto target_tv = pres_i / (dens_i * Rd) * (1. + amp);

    torch::Tensor temp_new;
    torch::Tensor xfrac_new;
    solve_virtual_temperature_perturbation(thermo_x, temp_i, pres_i, xfrac_i,
                                           target_tv, mask, dT, Rd, temp_new,
                                           xfrac_new);
    temp_i.copy_(temp_new);
    xfrac_i.copy_(xfrac_new);
  }

  for (int i = il; i <= iu; ++i) {
    auto temp_i = temp_state.select(2, i);
    auto pres_i = pres_state.select(2, i);
    auto xfrac_i = xfrac_state.select(2, i);
    auto conc_i = thermo_x->compute(
        "TPX->V", std::vector<torch::Tensor>{temp_i, pres_i, xfrac_i});

    w[IPR].select(2, i).copy_(pres_i);
    w[IDN].select(2, i).copy_(
        thermo_x->compute("V->D", std::vector<torch::Tensor>{conc_i}));
    w.narrow(0, ICY, ny)
        .select(3, i)
        .copy_(thermo_x->compute("X->Y", std::vector<torch::Tensor>{xfrac_i}));
  }

  if (config["problem"]["discrete-balance"].as<bool>(false)) {
    project_discrete_balance(block, w, grav);
  }

  vars["hydro_w"] = w;
}

}  // namespace

int main(int argc, char** argv) {
  torch::set_num_threads(1);
  torch::set_num_interop_threads(1);

  auto args = ParseArguments(argc, argv, "bryan.yaml");
  auto config = YAML::LoadFile(args.input_file);

  auto mesh = Mesh(MeshOptionsImpl::from_yaml(args.input_file));
  auto device = torch::Device(mesh->options->device_str());
  if (device.is_cuda()) {
    std::cout << "Running on CUDA" << std::endl;
  }
  mesh->to(device);

  MeshVariables vars(mesh->blocks.size());
  for (size_t i = 0; i < mesh->blocks.size(); ++i) {
    auto modules = mesh->blocks[i]->named_modules();
    auto thermo_y = std::dynamic_pointer_cast<kintera::ThermoYImpl>(
        modules["hydro.eos.thermo"]);
    set_user_output_callback(mesh->blocks[i], thermo_y->options->species(),
                             config["problem"]["p0"].as<double>());
    if (args.restart_file.empty()) {
      initialize_block(mesh->blocks[i], vars[i], config, device);
    }
  }

  double current_time = args.restart_file.empty()
                            ? mesh->initialize(vars)
                            : mesh->initialize(vars, args.restart_file.c_str());
  mesh->make_outputs(vars, current_time);

  int crossings = config["problem"]["sound-crossings"].as<int>(0);
  double t_cross = 0.;
  double tlim = mesh->blocks.front()->pintg->options->tlim();
  std::vector<MachSample> history;
  MachSample peak;
  MachSample peak_core;
  double peak_half = 0.;
  double peak_center = 0.;
  int first_above = -1;
  int first_core = -1;
  if (crossings > 0) {
    double cs_min = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < mesh->blocks.size(); ++i) {
      auto prim = primitives_of(mesh->blocks[i], vars[i]);
      cs_min = std::min(cs_min, sound_min(mesh->blocks[i], prim));
    }
    auto op = mesh->blocks.front()->pcoord->options;
    double height = op->global_x1max() - op->global_x1min();
    TORCH_CHECK(cs_min > 50. && cs_min < 2000.,
                "D1 sound speed out of range: ", cs_min);
    t_cross = height / cs_min;
    tlim = crossings * t_cross;
    for (auto& block : mesh->blocks) {
      block->pintg->options->tlim() = tlim;
    }
    std::cout << std::scientific << std::setprecision(16)
              << "D1WINDOW crossings=" << crossings << " height_m=" << height
              << " cs_min=" << cs_min << " t_cross=" << t_cross
              << " tlim=" << tlim << std::endl;
  }

  int cycle = mesh->blocks.front()->cycle;
  while (!mesh->blocks.front()->pintg->stop(cycle, current_time)) {
    ++cycle;
    mesh->set_cycle(cycle);

    auto dt = mesh->max_time_step(vars);
    mesh->print_cycle_info(vars, current_time, dt);

    for (int stage = 0; stage < mesh->blocks.front()->pintg->stages.size();
         ++stage) {
      mesh->forward(vars, dt, stage);
    }

    int redo = mesh->check_redo(vars);
    if (redo > 0) {
      cycle = mesh->blocks.front()->cycle;
      continue;
    }
    if (redo < 0) break;

    current_time += dt;
    if (crossings > 0) {
      MachSample step;
      for (size_t i = 0; i < mesh->blocks.size(); ++i) {
        auto prim = primitives_of(mesh->blocks[i], vars[i]);
        auto sample = mach_sample(mesh->blocks[i], prim);
        TORCH_CHECK(
            std::isfinite(sample.mach) && std::isfinite(sample.mach_core),
            "D1 mach is not finite at cycle ", cycle);
        if (sample.mach >= step.mach) step = sample;
      }
      step.cycle = cycle;
      step.time = current_time;
      history.push_back(step);
      if (step.mach > peak.mach) peak = step;
      if (step.mach_core > peak_core.mach_core) peak_core = step;
      peak_half = std::max(peak_half, step.mach_half);
      peak_center = std::max(peak_center, step.mach_center);
      if (first_above < 0 && step.mach >= 1.e-12) first_above = cycle;
      if (first_core < 0 && step.mach_core >= 1.e-12) first_core = cycle;
      int every = mesh->blocks.front()->pintg->options->ncycle_out();
      if (every > 0 && cycle % every == 0) {
        std::cout << std::scientific << std::setprecision(16)
                  << "D1MACH cycle=" << cycle << " time=" << step.time
                  << " mach=" << step.mach << " mach_x=" << step.mach_x
                  << " mach_y=" << step.mach_y << " peak=" << peak.mach
                  << std::endl;
      }
    }
    mesh->make_outputs(vars, current_time);
  }

  if (crossings > 0) {
    auto nearest = [&](int k) -> MachSample {
      if (history.empty()) return {};
      double target = k * t_cross;
      MachSample best = history.front();
      double err = std::abs(best.time - target);
      for (auto const& sample : history) {
        double e = std::abs(sample.time - target);
        if (e < err) {
          err = e;
          best = sample;
        }
      }
      return best;
    };
    auto end = history.empty() ? MachSample{} : history.back();
    auto c1 = nearest(1);
    auto c2 = nearest(2);
    auto c5 = nearest(5);
    auto c10 = nearest(10);
    double dx = mesh->blocks.front()
                    ->pcoord->dx1f
                    .narrow(0, mesh->blocks.front()->pcoord->il(),
                            mesh->blocks.front()->pcoord->options->nx1())
                    .mean()
                    .item<double>();
    bool complete = current_time + 1.e-8 >= tlim;
    bool roundoff = complete && peak.mach < 1.e-12;
    std::cout << std::scientific << std::setprecision(16) << "D1SUMMARY form="
              << mesh->blocks.front()->phydro->options->wb_density_ref()
              << " dx=" << dx << " peak_mach=" << peak.mach
              << " peak_cycle=" << peak.cycle << " peak_time=" << peak.time
              << " z=" << peak.z << " x=" << peak.x << " vx=" << peak.vx
              << " vy=" << peak.vy << " peak_mach_x=" << peak.mach_x
              << " peak_mach_y=" << peak.mach_y << " end_mach=" << end.mach
              << " end_time=" << end.time << " end_cycle=" << end.cycle
              << " first_cycle_ge_1e-12=" << first_above
              << " peak_mach_core=" << peak_core.mach_core
              << " core_cycle=" << peak_core.cycle
              << " core_time=" << peak_core.time
              << " core_z=" << peak_core.core_z
              << " core_x=" << peak_core.core_x
              << " first_core_cycle_ge_1e-12=" << first_core
              << " peak_mach_half=" << peak_half
              << " peak_mach_center=" << peak_center << " m1=" << c1.mach
              << " m1_core=" << c1.mach_core << " m2=" << c2.mach
              << " m2_core=" << c2.mach_core << " m5=" << c5.mach
              << " m5_core=" << c5.mach_core << " m10=" << c10.mach
              << " m10_core=" << c10.mach_core
              << " balance_sweeps=" << g_balance.sweeps
              << " balance_residual=" << g_balance.residual
              << " complete=" << (complete ? 1 : 0)
              << " result=" << (roundoff ? "pass" : "fail") << std::endl;
  }

  mesh->finalize(vars, current_time);
}
