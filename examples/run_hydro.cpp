// yaml
#include <yaml-cpp/yaml.h>

// kintera
#include <kintera/constants.h>

#include <kintera/kinetics/evolve_implicit.hpp>
#include <kintera/kinetics/kinetics.hpp>
#include <kintera/kinetics/kinetics_formatter.hpp>
#include <kintera/thermo/relative_humidity.hpp>

// snap
#include <snap/input/command_line.hpp>
#include <snap/mesh/meshblock.hpp>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

using namespace snap;

namespace {

// Same seed on main and arm A. uranus.yaml draws IVX/IVY with rand_like and
// does not seed, so an unseeded pair cannot report max|dT|.
constexpr uint64_t kGate2Seed = 236500;

void gate2_measure(MeshBlock block, Variables& vars, double current_time,
                   char const* tag) {
  auto hydro_u = vars.at("hydro_u");
  auto pcoord = block->pcoord;
  auto peos = block->phydro->peos;
  auto interior4 = block->part({0, 0, 0}, PartOptions().exterior(false));
  auto interior3 =
      block->part({0, 0, 0}, PartOptions().exterior(false).ndim(3));
  auto vol = pcoord->cell_volume();
  // Same reduction as MeshBlockImpl::print_cycle_info: sum u[IPR] * V
  // on interior cells. Ghosts are not in it.
  double energy = (hydro_u * vol)
                      .index(interior4)
                      .sum({1, 2, 3})[IPR]
                      .item<double>();
  int64_t hits = block->phydro->positivity_hits().item<int64_t>();

  // Clone first. W->T caches the tensor it is given; the live primitive and
  // the redo snapshot must stay out of that cache. At cycle 0 the primitive
  // is the IC. Later, kinetics has written species into hydro_u after the
  // last primitive update, so invert a clone of the conserved state.
  torch::Tensor w = (std::string(tag) == "cycle0")
                        ? vars.at("hydro_w").clone()
                        : peos->compute("U->W", {hydro_u.clone()});
  auto temp = peos->compute("W->T", {w}).clone();
  auto ti = temp.index(interior3).reshape({-1}).to(torch::kFloat64).cpu();

  if (char const* prefix = std::getenv("SNAPY_GATE2_DUMP")) {
    if (prefix[0] != '\0') {
      // Raw float64, length-prefixed. torch::save of a tensor is not a stable
      // thing to diff from Python here.
      std::ofstream out(std::string(prefix) + "_" + tag + "_T.bin",
                        std::ios::binary);
      int64_t n = ti.numel();
      out.write(reinterpret_cast<char const*>(&n), sizeof(n));
      out.write(reinterpret_cast<char const*>(ti.data_ptr<double>()),
                static_cast<std::streamsize>(n * sizeof(double)));
    }
  }

  std::cout << std::setprecision(17) << "GATE2_URANUS"
            << " tag=" << tag << " cycle=" << block->cycle
            << " time=" << current_time << " positivity_hits=" << hits
            << " energy=" << energy << " Tmin=" << ti.min().item<double>()
            << " Tmax=" << ti.max().item<double>()
            << " ncell=" << ti.numel() << std::endl;
}

}  // namespace

int main(int argc, char **argv) {
  torch::set_num_threads(1);
  torch::set_num_interop_threads(1);

  // read parameters
  auto cli = CommandLine::ParseArguments(argc, argv);
  if (!cli) return 0;

  // input file
  auto infile = std::string(cli->input_filename);

  auto config = YAML::LoadFile(infile);
  auto Ps = config["problem"]["Ps"].as<double>(1.e5);
  auto Ts = config["problem"]["Ts"].as<double>(300.);
  auto Tmin = config["problem"]["Tmin"].as<double>(200.);
  auto grav = -config["forcing"]["const-gravity"]["grav1"].as<double>();

  // initialize the block
  auto op_block = MeshBlockOptionsImpl::from_yaml(infile);
  auto block = MeshBlock(op_block);

  torch::Device device(torch::kCPU);
  if (torch::cuda::is_available() && op_block->layout()->device() == "cuda") {
    std::cout << "Running on CUDA" << std::endl;
    int device_id = op_block->layout()->device_id();
    if (device_id < 0) device_id = op_block->layout()->local_rank();
    device = torch::Device(torch::kCUDA, device_id);
  }

  block->to(device);

  // useful modules
  auto phydro = block->phydro;
  auto pcoord = block->pcoord;
  auto peos = phydro->peos;
  auto m = block->named_modules()["hydro.eos.thermo"];
  auto thermo_y = std::dynamic_pointer_cast<kintera::ThermoYImpl>(m);

  // dimensions and indices
  int nc3 = pcoord->x3v.size(0);
  int nc2 = pcoord->x2v.size(0);
  int nc1 = pcoord->x1v.size(0);
  int ny = thermo_y->options->species().size() - 1;
  int nvar = peos->nvar();

  // construct an adiabatic atmosphere
  kintera::ThermoX thermo_x(thermo_y->options);
  thermo_x->to(device);

  auto temp =
      Ts *
      torch::ones({nc3, nc2},
                  torch::TensorOptions().dtype(torch::kDouble).device(device));

  auto pres =
      Ps *
      torch::ones({nc3, nc2},
                  torch::TensorOptions().dtype(torch::kDouble).device(device));

  auto xfrac =
      torch::zeros({nc3, nc2, 1 + ny},
                   torch::TensorOptions().dtype(torch::kDouble).device(device));

  auto w = torch::zeros(
      {nvar, nc3, nc2, nc1},
      torch::TensorOptions().dtype(torch::kFloat64).device(device));

  // read in compositions
  for (int i = 1; i <= ny; ++i) {
    auto name = thermo_y->options->species()[i];
    auto xmixr = config["problem"]["x" + name].as<double>(0.);
    xfrac.select(2, i) = xmixr;
  }

  // dry air mole fraction
  xfrac.select(2, 0) = 1. - xfrac.narrow(-1, 1, ny).sum(-1);

  // adiabatic extrapolate half a grid to cell center
  int il = pcoord->il();
  int iu = pcoord->iu();
  auto dz = pcoord->dx1f[il].item<double>();
  thermo_x->extrapolate_dz(
      temp, pres, xfrac,
      kintera::ExtrapOptions().dz(dz / 2.).grav(grav).ds_dz(0.));

  int i = il;
  int nvapor = thermo_x->options->vapor_ids().size();
  int ncloud = thermo_x->options->cloud_ids().size();
  for (; i <= iu; ++i) {
    auto conc = thermo_x->compute("TPX->V", {temp, pres, xfrac});

    w[IPR].select(2, i) = pres;
    w[IDN].select(2, i) = thermo_x->compute("V->D", {conc});

    auto result = thermo_x->compute("X->Y", {xfrac});
    w.narrow(0, ICY, ny).select(3, i) = thermo_x->compute("X->Y", {xfrac});

    if ((temp < Tmin).any().item<double>()) break;
    dz = pcoord->dx1f[i].item<double>();
    thermo_x->extrapolate_dz(
        temp, pres, xfrac,
        kintera::ExtrapOptions().dz(dz).grav(grav).ds_dz(0.));
  }

  // isothermal extrapolation
  for (; i <= iu; ++i) {
    auto mu = (thermo_x->mu * xfrac).sum(-1);
    dz = pcoord->dx1f[i].item<double>();
    pres *= exp(-grav * mu * dz / (kintera::constants::Rgas * temp));
    auto conc = thermo_x->compute("TPX->V", {temp, pres, xfrac});
    w[IPR].select(2, i) = pres;
    w[IDN].select(2, i) = thermo_x->compute("V->D", {conc});
    w.narrow(0, ICY, ny).select(3, i) = thermo_x->compute("X->Y", {xfrac});
  }

  // add noise. Seed immediately before the draws so earlier RNG use in
  // construction cannot desynchronize main and arm A.
  torch::manual_seed(kGate2Seed);
  w[IVX] += 0.01 * torch::rand_like(w[IVX]);
  w[IVY] += 0.01 * torch::rand_like(w[IVY]);

  // initialize
  std::map<std::string, torch::Tensor> vars;
  vars["hydro_w"] = w;
  double current_time = block->initialize(vars, cli->restart_filename);
  std::cout << "GATE2_URANUS seed=" << kGate2Seed << std::endl;
  gate2_measure(block, vars, current_time, "cycle0");

  // user output variables
  // (1) total precipitable mass fraction [kg/kg]
  block->user_output_callback = [&](Variables const &vars) {
    auto w = vars.at("hydro_w");
    Variables out;
    out["qtol"] = w.narrow(0, ICY, ny).sum(0);
    return out;
  };

  // create kinetics model
  auto op_kinet = kintera::KineticsOptionsImpl::from_yaml(infile);
  auto kinet = kintera::Kinetics(op_kinet);
  kinet->to(device);

  // time loop
  if (cli->restart_filename == nullptr) {
    block->make_outputs(vars, current_time);
  }

  while (!block->pintg->stop(block->cycle, current_time)) {
    ++block->cycle;
    auto dt = block->max_time_step(vars);
    block->print_cycle_info(vars, current_time, dt);

    // evolve dynamics
    for (int stage = 0; stage < block->pintg->stages.size(); ++stage) {
      block->forward(vars, dt, stage);
    }

    // evolve kinetics
    auto &hydro_u = vars["hydro_u"];
    auto &hydro_w = vars["hydro_w"];

    // Kinetics needs the final dynamics state, including saturation adjustment.
    peos->forward(hydro_u, hydro_w);
    auto temp = peos->compute("W->T", {hydro_w});
    auto pres = hydro_w[IPR];
    auto xfrac = thermo_y->compute("Y->X", {hydro_w.narrow(0, ICY, ny)});
    auto conc = thermo_x->compute("TPX->V", {temp, pres, xfrac});
    auto cp_vol = thermo_x->compute("TV->cp", {temp, conc});

    // auto conc_kinet = kinet->options.narrow_copy(conc, thermo_y->options);
    auto conc_kinet = conc.slice(-1, 1, conc.size(-1));
    auto [rate, rc_ddC, rc_ddT] = kinet->forward(temp, pres, conc_kinet);
    auto jac = kinet->jacobian(temp, conc_kinet, cp_vol, rate, rc_ddC, rc_ddT);
    auto del_conc = kintera::evolve_implicit(rate, kinet->stoich, jac, dt);
    std::vector<int64_t> vec(del_conc.dim(), 1);
    vec[del_conc.dim() - 1] = -1;
    auto del_rho = del_conc / thermo_y->inv_mu.narrow(0, 1, ny).view(vec);
    hydro_u.narrow(0, ICY, ny) += del_rho.permute({3, 0, 1, 2});

    int err = block->check_redo(vars);
    if (err > 0) continue;  // redo this step with smaller dt
    if (err < 0) break;     // terminate simulation

    // make outputs
    current_time += dt;
    block->make_outputs(vars, current_time);
  }

  // cycle here is the integrator count: check_redo decrements it on a redo,
  // so a clean nlim of 500 is 500 successful forwards, not 500 attempts.
  gate2_measure(block, vars, current_time, "final");

  int status = block->finalize(vars, current_time);

  CommandLine::Destroy();
  return status;
}
