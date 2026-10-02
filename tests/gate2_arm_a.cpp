// One Gate 2 arm-A process: a single YAML card, one EOS, one case.
// Writes %.17g dumps of both arms and the limiter-off flux. kintera's species
// table is process-global, so a second card needs a second process.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

#include <kintera/constants.h>
#include <kintera/utils/molar_mass.hpp>

#include <snap/snap.h>

#include <snap/coord/coordinate.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/mesh/meshblock.hpp>

using namespace snap;

namespace {

struct Args {
  std::string card, eos, cas, out;
};

Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto need = [&](std::string const& name) {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << name << "\n";
        std::exit(2);
      }
      return std::string(argv[++i]);
    };
    if (k == "--card") a.card = need(k);
    else if (k == "--eos") a.eos = need(k);
    else if (k == "--case") a.cas = need(k);
    else if (k == "--out") a.out = need(k);
    else {
      std::cerr << "unknown arg " << k << "\n";
      std::exit(2);
    }
  }
  if (a.card.empty() || a.eos.empty() || a.cas.empty() || a.out.empty()) {
    std::cerr << "usage: gate2_arm_a --card F --eos ideal-moist|moist-mixture "
                 "--case NAME --out DIR\n";
    std::exit(2);
  }
  return a;
}

void write_tensor(std::ostream& os, char const* name, torch::Tensor t) {
  os << "BEGIN " << name;
  if (!t.defined() || t.numel() == 0) {
    os << " 0\nEND\n";
    return;
  }
  t = t.detach().to(torch::kCPU).to(torch::kFloat64).contiguous();
  for (auto s : t.sizes()) os << " " << s;
  os << "\n";
  double const* p = t.data_ptr<double>();
  for (int64_t i = 0; i < t.numel(); ++i) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.17g", p[i]);
    os << buf << "\n";
  }
  os << "END\n";
}

struct Prepared {
  YAML::Node card;
  double vx = 0, vy = 0, vz = 0;
  bool shape_donors = false;
  bool shape_mixed = false;
};

Prepared prepare(Args const& args) {
  Prepared p;
  p.card = YAML::LoadFile(args.card);
  p.card["dynamics"]["equation-of-state"]["type"] = args.eos;
  std::string c = args.cas;
  bool vapor = c.rfind("vapor-u0", 0) == 0;
  if (vapor) {
    bool found = false;
    for (auto sp : p.card["species"]) {
      if (sp["name"].as<std::string>() == "vapor") {
        double u0 = sp["u0_R"] ? sp["u0_R"].as<double>() : 0.;
        if (u0 != -1000.) {
          std::cerr << "vapor-u0 card must set vapor u0_R = -1000, got " << u0
                    << "\n";
          std::exit(2);
        }
        found = true;
      }
    }
    if (!found) {
      std::cerr << "vapor-u0 card has no vapor species\n";
      std::exit(2);
    }
  }
  auto set_rs = [&](std::string const& rs) {
    p.card["dynamics"]["riemann-solver"]["type"] = rs;
  };
  if (c == "adv-lmars" || c == "vapor-u0-lmars") {
    set_rs("lmars");
    p.vx = 2.;
    p.vy = 3.;
  } else if (c == "adv-hllc" || c == "vapor-u0-hllc") {
    set_rs("hllc");
    p.vx = 2.;
    p.vy = 3.;
  } else if (c == "settling") {
    set_rs("lmars");
    p.vy = 3.;
    p.card["sedimentation"] =
        YAML::Load("{radius: {}, density: {}, const-vsed: {cloud: -2.}}");
  } else if (c == "x2") {
    set_rs("lmars");
    p.vy = 2.;
    p.vz = 3.;
    p.card["geometry"]["bounds"]["x2max"] = 6.;
    p.card["geometry"]["cells"]["nx2"] = 6;
    p.card["boundary-condition"]["external"]["x2-inner"] = "reflecting";
    p.card["boundary-condition"]["external"]["x2-outer"] = "reflecting";
  } else if (c == "donors") {
    set_rs("lmars");
    p.shape_donors = true;
  } else if (c == "mixed") {
    set_rs("lmars");
    p.shape_mixed = true;
    p.card["sedimentation"] =
        YAML::Load("{radius: {}, density: {}, const-vsed: {cloud: -1.}}");
  } else {
    std::cerr << "unknown case " << c << "\n";
    std::exit(2);
  }
  return p;
}

void shape_column(torch::Tensor& w, int il, bool mixed) {
  double rho[6] = {1.00, 1.15, 0.90, 1.05, 0.85, 1.20};
  double vx[6] = {2.0, 2.4, 1.6, -1.6, -2.4, -2.0};
  double vy[6] = {3.0, 3.5, 2.5, 4.0, 2.0, 3.2};
  double vz[6] = {0.0, 0.5, -0.5, 1.0, -1.0, 0.3};
  if (mixed) {
    double mx[6] = {3.0, 3.4, 2.6, 3.2, 2.8, 3.1};
    for (int i = 0; i < 6; ++i) vx[i] = mx[i];
  }
  for (int c = 0; c < 6; ++c) {
    w[IDN].select(-1, il + c).fill_(rho[c]);
    w[IVX].select(-1, il + c).fill_(vx[c]);
    w[IVY].select(-1, il + c).fill_(vy[c]);
    w[IVZ].select(-1, il + c).fill_(vz[c]);
  }
}

struct Arm {
  std::shared_ptr<MeshBlockImpl> block;
  Variables vars;
  torch::Tensor du;
  torch::Tensor hook;
  torch::Tensor tcode;
  torch::Tensor hook_neighbor;
  torch::Tensor hook_vel;
  RepairCensus census{};
  bool census_valid = false;
  int repair_threw = 0;
  std::vector<std::array<int, 3>> bad_cells;
  int i2_neighbor_bitwise = -1;
  double i2_hke_over_s = -1.;
};

Arm run_arm(Prepared const& prep, std::string const& arm, int mutation,
            std::string const& yaml_path) {
  YAML::Node card = YAML::Clone(prep.card);
  bool limiter = arm != "off";
  card["dynamics"]["equation-of-state"]["limiter"] = limiter;
  card["dynamics"]["debug-disable-flux-positivity"] = arm == "b2";
  {
    std::ofstream(yaml_path) << card;
  }
  Arm out;
  out.block = std::make_shared<MeshBlockImpl>(
      MeshBlockOptionsImpl::from_yaml(yaml_path));
  std::remove(yaml_path.c_str());
  out.block->to(torch::kCPU);
  auto coord = out.block->pcoord;
  auto w = torch::zeros({out.block->phydro->peos->nvar(), coord->options->nc3(),
                         coord->options->nc2(), coord->options->nc1()},
                        torch::dtype(torch::kFloat64));
  w[IDN].fill_(1.);
  w[IPR].fill_(1.e5);
  w[IVX].fill_(prep.vx);
  w[IVY].fill_(prep.vy);
  w[IVZ].fill_(prep.vz);
  w[ICY].fill_(0.01);
  w[ICY + 1].fill_(0.02);
  if (prep.shape_donors || prep.shape_mixed) {
    shape_column(w, coord->il(), prep.shape_mixed);
  }
  out.vars["hydro_w"] = w;
  out.block->initialize(out.vars);
  auto peos = out.block->phydro->peos;
  peos->set_species_enthalpy_mutation(mutation);
  auto u = out.vars.at("hydro_u");
  out.du = out.block->phydro->forward(1., u, out.vars);
  out.hook = peos->species_enthalpy(out.vars.at("hydro_w")).contiguous();
  out.tcode = peos->compute("W->T", {out.vars.at("hydro_w")}).contiguous();

  if (mutation == 0 && arm == "a") {
    auto w0 = out.vars.at("hydro_w");
    auto wn = w0.clone();
    int i0 = coord->il(), j0 = coord->jl(), k0 = coord->kl();
    wn[IPR][k0][j0][i0] = wn[IPR][k0][j0][i0] + 2.e4;
    wn[IVY][k0][j0][i0] = wn[IVY][k0][j0][i0] + 5.;
    out.hook_neighbor = peos->species_enthalpy(wn).contiguous();
    auto eq = torch::eq(out.hook, out.hook_neighbor);
    eq.index_put_({torch::indexing::Slice(), k0, j0, i0}, true);
    out.i2_neighbor_bitwise = eq.all().item<bool>() ? 1 : 0;

    auto wv = w0.clone();
    wv[IVX] = wv[IVX] + 4.;
    out.hook_vel = peos->species_enthalpy(wv).contiguous();
    auto ke = 0.5 * (w0[IVX] * w0[IVX] + w0[IVY] * w0[IVY] + w0[IVZ] * w0[IVZ]);
    auto kev =
        0.5 * (wv[IVX] * wv[IVX] + wv[IVY] * wv[IVY] + wv[IVZ] * wv[IVZ]);
    auto dh = (out.hook - ke) - (out.hook_vel - kev);
    // Worst |h-KE difference| / S_n over interior cells. S_n uses YAML u0_R.
    auto th = peos->options->thermo();
    int ngas = static_cast<int>(th->vapor_ids().size());
    int ncloud = static_cast<int>(th->cloud_ids().size());
    int ny = ngas + ncloud - 1;
    double Tref = th->Tref();
    double R = kintera::constants::Rgas;
    auto interior3 =
        out.block->part({0, 0, 0}, PartOptions().exterior(false).ndim(3));
    auto interior4 = out.block->part({0, 0, 0}, PartOptions().exterior(false));
    auto T = out.tcode.index(interior3).contiguous();
    auto dh_i = dh.index(interior4).contiguous();
    auto ke_i = ke.index(interior3).contiguous();
    double worst = 0.;
    auto ts = T.sizes();
    for (int s = 0; s < ny; ++s) {
      int sp = s + 1;  // the hook drops dry
      double u0 = 0., cv = 0.;
      for (auto node : prep.card["species"]) {
        if (node["name"].as<std::string>() != th->names().at(sp)) continue;
        cv = node["cv_R"] ? node["cv_R"].as<double>() : 0.;
        u0 = node["u0_R"] ? node["u0_R"].as<double>() : 0.;
      }
      double M = peos->species_weight(sp);
      bool cloud = false;
      for (int id : th->cloud_ids())
        if (id == sp) cloud = true;
      double z = cloud ? 0. : 1.;
      auto one = dh_i[s].abs();
      for (int64_t k = 0; k < ts[0]; ++k) {
        for (int64_t j = 0; j < ts[1]; ++j) {
          for (int64_t i = 0; i < ts[2]; ++i) {
            double Tv = T[k][j][i].item<double>();
            double Sn = (R / M) * (std::abs(u0 - cv * Tref) + (cv + z) * Tv) +
                        std::abs(ke_i[k][j][i].item<double>());
            double ratio = one[k][j][i].item<double>() / std::max(Sn, 1.e-300);
            worst = std::max(worst, ratio);
          }
        }
      }
    }
    out.i2_hke_over_s = worst;
  }

  if (limiter) {
    auto upd = u.clone();
    auto interior = out.block->part({0, 0, 0}, PartOptions().exterior(false));
    upd.index(interior) += out.du.index(interior);
    int ny = peos->options->thermo()
                 ? static_cast<int>(peos->options->thermo()->vapor_ids().size() +
                                    peos->options->thermo()->cloud_ids().size()) -
                       1
                 : 0;
    auto bad = upd[IDN] < peos->options->density_floor();
    if (ny > 0) bad = bad.logical_or((upd.narrow(0, ICY, ny) < 0).any(0));
    auto interior3 =
        out.block->part({0, 0, 0}, PartOptions().exterior(false).ndim(3));
    auto ibad = bad.index(interior3);
    auto nc = ibad.sizes();
    for (int64_t k = 0; k < nc[0]; ++k)
      for (int64_t j = 0; j < nc[1]; ++j)
        for (int64_t i = 0; i < nc[2]; ++i)
          if (ibad[k][j][i].item<bool>())
            out.bad_cells.push_back(
                {static_cast<int>(k), static_cast<int>(j), static_cast<int>(i)});
    // The census is stored before any value changes. A column whose vapor
    // total is negative makes fix_vapor throw; that is a result, not a
    // reason to drop the fluxes already computed by forward().
    try {
      peos->apply_conserved_limiter_(upd);
    } catch (std::exception const& ex) {
      out.repair_threw = 1;
      std::cerr << "repair threw arm " << arm << " mutation " << mutation
                << ": " << ex.what() << "\n";
    }
    out.census = peos->repair_census();
    out.census_valid = true;
  }
  peos->set_species_enthalpy_mutation(0);
  return out;
}

void write_species(std::ostream& os, EquationOfStateImpl* peos,
                   YAML::Node const& card) {
  auto th = peos->options->thermo();
  os << "# use_nasa9_cp " << (th->use_nasa9_cp() ? 1 : 0) << "\n";
  os << "# use_h2_cp " << (th->use_h2_cp() ? 1 : 0) << "\n";
  os << "# h2_cp_mode " << th->h2_cp_mode() << "\n";
  os << "# Tref " << th->Tref() << "\n";
  os << "# Rgas " << kintera::constants::Rgas << "\n";
  int n = static_cast<int>(th->names().size());
  os << "# nspecies " << n << "\n";
  for (int i = 0; i < n; ++i) {
    double my = 0.;
    for (auto sp : card["species"]) {
      if (sp["name"].as<std::string>() != th->names()[i]) continue;
      if (sp["composition"]) {
        for (auto it : sp["composition"]) {
          my += it.second.as<double>() *
                kintera::atomic_mass(it.first.as<std::string>());
        }
      }
    }
    double u0 = 0., cv = 0.;
    if (i < static_cast<int>(th->uref_R().size())) {
      // YAML u0, not the shifted runtime intercept. Re-read from the card.
    }
    for (auto sp : card["species"]) {
      if (sp["name"].as<std::string>() != th->names()[i]) continue;
      cv = sp["cv_R"] ? sp["cv_R"].as<double>() : 0.;
      u0 = sp["u0_R"] ? sp["u0_R"].as<double>() : 0.;
    }
    int cloud = 0;
    for (int id : th->cloud_ids())
      if (id == i) cloud = 1;
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "# species %d %s M_code %.17g M_yaml %.17g cv_R %.17g "
                  "u0_R %.17g cloud %d\n",
                  i, th->names()[i].c_str(), peos->species_weight(i), my, cv,
                  u0, cloud);
    os << buf;
  }
}

void dump_arm(std::string const& path, Args const& args, std::string const& arm,
              int mutation, Arm const& a, YAML::Node const& card,
              int b2_eq_off, double max_abs_b2_off) {
  std::ofstream os(path);
  auto peos = a.block->phydro->peos;
  auto coord = a.block->pcoord;
  os << "# eos " << args.eos << "\n# case " << args.cas << "\n# arm " << arm
     << "\n# mutation " << mutation << "\n";
  os << "# index IDN " << IDN << " IVX " << IVX << " IVY " << IVY << " IVZ "
     << IVZ << " IPR " << IPR << " ICY " << ICY << "\n";
  os << "# kintera_Rgas " << kintera::constants::Rgas << "\n";
  write_species(os, peos.get(), card);
  os << "# il " << coord->il() << " iu " << coord->iu() << " jl " << coord->jl()
     << " ju " << coord->ju() << " kl " << coord->kl() << " ku " << coord->ku()
     << "\n";
  os << "# nghost " << coord->options->nghost() << "\n";
  os << "# census_valid " << (a.census_valid ? 1 : 0) << "\n";
  os << "# repair_threw " << a.repair_threw << "\n";
  os << "# nan_interior " << a.census.nan_interior << "\n";
  os << "# nan_ghost " << a.census.nan_ghost << "\n";
  os << "# clamp_interior " << a.census.clamp_interior << "\n";
  os << "# clamp_ghost " << a.census.clamp_ghost << "\n";
  os << "# bad_cells " << a.bad_cells.size() << "\n";
  for (auto const& c : a.bad_cells) {
    os << "# bad " << c[0] << " " << c[1] << " " << c[2] << "\n";
  }
  os << "# i2_neighbor_bitwise " << a.i2_neighbor_bitwise << "\n";
  {
    char ibuf[80];
    std::snprintf(ibuf, sizeof(ibuf), "# i2_hke_over_s %.17g\n", a.i2_hke_over_s);
    os << ibuf;
  }
  os << "# b2_eq_off " << b2_eq_off << "\n";
  char buf[80];
  std::snprintf(buf, sizeof(buf), "# max_abs_b2_off %.17g\n", max_abs_b2_off);
  os << buf;
  auto hydro = a.block->phydro;
  write_tensor(os, "W", a.vars.at("hydro_w"));
  write_tensor(os, "U", a.vars.at("hydro_u"));
  write_tensor(os, "DU", a.du);
  write_tensor(os, "FLUX1", hydro->flux1());
  write_tensor(os, "FLUX2", hydro->flux2());
  write_tensor(os, "FSED1", hydro->fsed1());
  write_tensor(os, "H", a.hook);
  write_tensor(os, "TCODE", a.tcode);
  write_tensor(os, "VOL", coord->cell_volume());
  write_tensor(os, "AREA1", coord->face_area1());
  write_tensor(os, "AREA2", coord->face_area2());
  write_tensor(os, "HNEIGH", a.hook_neighbor);
  write_tensor(os, "HVEL", a.hook_vel);
}

bool same_flux(torch::Tensor const& a, torch::Tensor const& b, double* max_abs) {
  bool ad = a.defined() && a.numel() > 0;
  bool bd = b.defined() && b.numel() > 0;
  if (!ad && !bd) {
    *max_abs = 0.;
    return true;
  }
  if (!ad || !bd || a.sizes() != b.sizes()) {
    *max_abs = std::numeric_limits<double>::infinity();
    return false;
  }
  auto d = (a - b).abs().max().item<double>();
  *max_abs = std::max(*max_abs, d);
  return torch::equal(a, b);
}

}  // namespace

int main(int argc, char** argv) {
  auto args = parse(argc, argv);
  auto prep = prepare(args);
  std::string base = args.out + "/" + args.eos + "__" + args.cas;
  std::string yaml_path = base + ".yaml";

  auto b2 = run_arm(prep, "b2", 0, yaml_path);
  auto off = run_arm(prep, "off", 0, yaml_path);
  double max_abs = 0.;
  bool eq = same_flux(b2.block->phydro->flux1(), off.block->phydro->flux1(),
                      &max_abs) &&
            same_flux(b2.block->phydro->flux2(), off.block->phydro->flux2(),
                      &max_abs) &&
            same_flux(b2.block->phydro->flux3(), off.block->phydro->flux3(),
                      &max_abs);
  int eq_flag = eq ? 1 : 0;

  auto a = run_arm(prep, "a", 0, yaml_path);
  auto row_equal = [](torch::Tensor const& f, torch::Tensor const& g, int row) {
    bool fd = f.defined() && f.numel() > 0;
    bool gd = g.defined() && g.numel() > 0;
    if (!fd && !gd) return true;
    if (!fd || !gd || f.sizes() != g.sizes()) return false;
    return torch::equal(f[row], g[row]);
  };
  int idn_a_b2 = (row_equal(a.block->phydro->flux1(), b2.block->phydro->flux1(), 0) &&
                  row_equal(a.block->phydro->flux2(), b2.block->phydro->flux2(), 0))
                     ? 1
                     : 0;
  int idn_a_off = (row_equal(a.block->phydro->flux1(), off.block->phydro->flux1(), 0) &&
                   row_equal(a.block->phydro->flux2(), off.block->phydro->flux2(), 0))
                      ? 1
                      : 0;
  {
    std::ofstream sm(base + ".summary");
    sm << "b2_eq_off " << eq_flag << "\n";
    sm << "max_abs_b2_off " << max_abs << "\n";
    sm << "idn_a_b2 " << idn_a_b2 << "\n";
    sm << "idn_a_off " << idn_a_off << "\n";
    sm << "i2_neighbor_bitwise " << a.i2_neighbor_bitwise << "\n";
    sm << "i2_hke_over_s " << a.i2_hke_over_s << "\n";
    sm << "census_a " << a.census.nan_interior << " " << a.census.nan_ghost << " "
       << a.census.clamp_interior << " " << a.census.clamp_ghost << "\n";
    sm << "census_b2 " << b2.census.nan_interior << " " << b2.census.nan_ghost
       << " " << b2.census.clamp_interior << " " << b2.census.clamp_ghost << "\n";
    sm << "bad_b2 " << b2.bad_cells.size() << "\n";
    for (auto const& c : b2.bad_cells)
      sm << "b2_bad " << c[0] << " " << c[1] << " " << c[2] << "\n";
    sm << "bad_a " << a.bad_cells.size() << "\n";
    sm << "repair_threw_a " << a.repair_threw << "\n";
    sm << "repair_threw_b2 " << b2.repair_threw << "\n";
    sm << "repair_threw_off " << off.repair_threw << "\n";
  }
  dump_arm(base + "__a__m0.csv", args, "a", 0, a, prep.card, eq_flag, max_abs);
  dump_arm(base + "__b2__m0.csv", args, "b2", 0, b2, prep.card, eq_flag, max_abs);
  dump_arm(base + "__off__m0.csv", args, "off", 0, off, prep.card, eq_flag,
           max_abs);

  auto th = a.block->phydro->peos->options->thermo();
  std::cout << "cp use_nasa9_cp=" << th->use_nasa9_cp()
            << " use_h2_cp=" << th->use_h2_cp()
            << " h2_cp_mode=" << th->h2_cp_mode() << " Tref=" << th->Tref()
            << " Rgas=" << kintera::constants::Rgas << "\n";
  std::cout << "b2_eq_off " << eq_flag << " max_abs " << max_abs << "\n";
  std::cout << "census A " << a.census.nan_interior << " " << a.census.nan_ghost
            << " " << a.census.clamp_interior << " " << a.census.clamp_ghost
            << " bad " << a.bad_cells.size() << "\n";
  std::cout << "census B2 " << b2.census.nan_interior << " "
            << b2.census.nan_ghost << " " << b2.census.clamp_interior << " "
            << b2.census.clamp_ghost << " bad " << b2.bad_cells.size() << "\n";
  for (auto const& c : b2.bad_cells) {
    std::cout << "b2_bad " << c[0] << " " << c[1] << " " << c[2] << "\n";
  }

  if (th->use_nasa9_cp() || th->use_h2_cp()) {
    std::cout << "FAIL cp flag on\n";
    return 3;
  }

  if (args.eos == "moist-mixture") {
    for (int m = 1; m <= 5; ++m) {
      auto am = run_arm(prep, "a", m, yaml_path);
      dump_arm(base + "__a__m" + std::to_string(m) + ".csv", args, "a", m, am,
               prep.card, eq_flag, max_abs);
      std::cout << "mutation " << m << " dumped\n";
    }
  }
  std::cout << "WROTE " << base << "\n";
  return 0;
}
