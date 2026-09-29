// Static x1 face-density error for snapy issue #250, test T1.
// The column is analytic and hydrostatic. Each cell is given the exact cell
// mass (p_left - p_right) / dz and the average pressure. Face densities come
// from HydroImpl::face_density_x1, which is the production WENO path.
// H = 1, g = 1, Rd = 1, T(0) = 1, p(0) = 1. dz/H is the interior cell width.

#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <snap/snap.h>

#include <snap/mesh/meshblock.hpp>

using namespace snap;

namespace {

struct State {
  double rho;
  double p;
  double T;
};

using StateFn = std::function<State(double)>;

double simpson(std::function<double(double)> const& f, double a, double b) {
  constexpr int n = 256;
  double h = (b - a) / n;
  double s = f(a) + f(b);
  for (int i = 1; i < n; ++i) s += ((i & 1) ? 4. : 2.) * f(a + i * h);
  return s * h / 3.;
}

struct Profile {
  std::string name;
  double Z;
  double gamma;
  StateFn at;
  double kink = -1.;  // split the pressure integral here when it falls in a cell
};

double kappa_of(double gamma) { return (gamma - 1.) / gamma; }

Profile dry_adiabat(std::string name, double gamma, double Z) {
  double kappa = kappa_of(gamma);
  return {name, Z, gamma, [=](double z) {
            double T = 1. - kappa * z;
            double p = std::pow(T, 1. / kappa);
            return State{p / T, p, T};
          }};
}

Profile isothermal(std::string name, double gamma, double Z) {
  return {name, Z, gamma, [](double z) {
            double p = std::exp(-z);
            return State{p, p, 1.};
          }};
}

// theta = exp(N^2 z), N^2 = kappa/2. Closed form, see the issue note in main().
Profile const_n2(std::string name, double gamma, double Z) {
  double kappa = kappa_of(gamma);
  double N2 = 0.5 * kappa;
  return {name, Z, gamma, [=](double z) {
            double pk = 1. - (kappa / N2) * (1. - std::exp(-N2 * z));
            double p = std::pow(pk, 1. / kappa);
            double T = std::exp(N2 * z) * pk;
            return State{p / T, p, T};
          }};
}

Profile inversion(std::string name, double gamma, double Z) {
  constexpr double G = 0.15;
  return {name, Z, gamma, [=](double z) {
            double T = 1. + G * z;
            double p = std::pow(T, -1. / G);
            return State{p / T, p, T};
          }};
}

Profile superadiabatic(std::string name, double gamma, double Z) {
  double G = 1.5 * kappa_of(gamma);
  return {name, Z, gamma, [=](double z) {
            double T = 1. - G * z;
            double p = std::pow(T, 1. / G);
            return State{p / T, p, T};
          }};
}

Profile sharp_tropopause(std::string name, double gamma, double Z) {
  double kappa = kappa_of(gamma);
  double zt = 1.;
  double Tt = 1. - kappa * zt;
  double pt = std::pow(Tt, 1. / kappa);
  return {name,
          Z,
          gamma,
          [=](double z) {
            if (z <= zt) {
              double T = 1. - kappa * z;
              double p = std::pow(T, 1. / kappa);
              return State{p / T, p, T};
            }
            double T = Tt;
            double p = pt * std::exp(-(z - zt) / Tt);
            return State{p / T, p, T};
          },
          zt};
}

// Lapse goes from adiabatic to isothermal across a tanh of width 0.25 H.
// The table covers the ghost halo, not only [0, Z]: holding p past the end
// made the outer ghost mass zero and poisoned the local polytrope.
Profile smooth_tropopause(std::string name, double gamma, double Z) {
  double kappa = kappa_of(gamma);
  double zt = 1.2;
  double delta = 0.25;
  auto T_of = [=](double z) {
    auto ant = [=](double s) {
      return 0.5 * kappa *
             (s - delta * std::log(std::cosh((s - zt) / delta)));
    };
    double T = 1. - (ant(z) - ant(0.));
    return T > 0.2 ? T : 0.2;
  };
  constexpr double z0 = -2.;
  constexpr int n = 40001;
  double z1 = Z + 2.;
  double h = (z1 - z0) / (n - 1);
  std::vector<double> pp(n), TT(n);
  int i0 = (int)std::lround((0. - z0) / h);
  pp[i0] = 1.;
  TT[i0] = T_of(0.);
  auto step = [&](int i, int dir) {
    double z = z0 + i * h;
    auto rhs = [&](double z_, double p) { return -p / T_of(z_); };
    double hh = dir * h;
    double k1 = rhs(z, pp[i]);
    double k2 = rhs(z + 0.5 * hh, pp[i] + 0.5 * hh * k1);
    double k3 = rhs(z + 0.5 * hh, pp[i] + 0.5 * hh * k2);
    double k4 = rhs(z + hh, pp[i] + hh * k3);
    pp[i + dir] = pp[i] + hh * (k1 + 2 * k2 + 2 * k3 + k4) / 6.;
    TT[i + dir] = T_of(z0 + (i + dir) * h);
  };
  for (int i = i0; i < n - 1; ++i) step(i, +1);
  for (int i = i0; i > 0; --i) step(i, -1);
  return {name, Z, gamma, [=](double z) {
            double x = std::min(std::max((z - z0) / h, 0.), (double)n - 1.0000001);
            int i = (int)x;
            double f = x - i;
            double p = pp[i] * (1. - f) + pp[i + 1] * f;
            double T = TT[i] * (1. - f) + TT[i + 1] * f;
            return State{p / T, p, T};
          }};
}

// Saturated pseudo-adiabat. q = epsilon * e_sat / (p - e_sat), dilute-capped.
// Constants are nondimensional and named in the regime; they are not a
// laboratory Clausius-Clapeyron fit.
Profile moist_sat(std::string name, double gamma, double Z, double epsilon,
                  double q_bottom, double L_over_Rd) {
  double esat0 = q_bottom / (epsilon + q_bottom);
  double cp = 1. / kappa_of(gamma);  // cp/Rd
  constexpr double z0 = -2.;
  constexpr int n = 40001;
  double z1 = Z + 2.;
  double h = (z1 - z0) / (n - 1);
  int i0 = (int)std::lround((0. - z0) / h);
  std::vector<double> pp(n), TT(n), qq(n);
  pp[i0] = 1.;
  TT[i0] = 1.;
  auto esat = [=](double T) {
    return esat0 * std::exp(L_over_Rd * epsilon * (1. - 1. / T));
  };
  auto q_of = [=](double T, double p) {
    double e = std::min(esat(T), 0.5 * p);
    return epsilon * e / (p - e);
  };
  auto Rfac = [=](double q) { return (1. - q) + q / epsilon; };
  qq[i0] = q_of(1., 1.);
  auto deriv = [&](double T, double p, double& dT, double& dp) {
    double q = q_of(T, p);
    dp = -p / (Rfac(q) * T);
    double num = 1. + L_over_Rd * q / T;
    double den = 1. + (L_over_Rd * L_over_Rd) * epsilon * q / (cp * T * T);
    dT = -kappa_of(gamma) * num / den;
    if (T < 0.25 && dT < 0.) dT = 0.;
    if (T > 5. && dT > 0.) dT = 0.;
  };
  auto step = [&](int i, int dir) {
    double hh = dir * h;
    double T = TT[i], p = pp[i];
    double dT1, dp1, dT2, dp2, dT3, dp3, dT4, dp4;
    deriv(T, p, dT1, dp1);
    deriv(T + 0.5 * hh * dT1, p + 0.5 * hh * dp1, dT2, dp2);
    deriv(T + 0.5 * hh * dT2, p + 0.5 * hh * dp2, dT3, dp3);
    deriv(T + hh * dT3, p + hh * dp3, dT4, dp4);
    TT[i + dir] = std::max(0.2, T + hh * (dT1 + 2 * dT2 + 2 * dT3 + dT4) / 6.);
    pp[i + dir] = p + hh * (dp1 + 2 * dp2 + 2 * dp3 + dp4) / 6.;
    qq[i + dir] = q_of(TT[i + dir], pp[i + dir]);
  };
  for (int i = i0; i < n - 1; ++i) step(i, +1);
  for (int i = i0; i > 0; --i) step(i, -1);
  return {name, Z, gamma, [=](double z) {
            double x =
                std::min(std::max((z - z0) / h, 0.), (double)n - 1.0000001);
            int i = (int)x;
            double f = x - i;
            double p = pp[i] * (1. - f) + pp[i + 1] * f;
            double T = TT[i] * (1. - f) + TT[i + 1] * f;
            double q = qq[i] * (1. - f) + qq[i + 1] * f;
            double Rf = (1. - q) + q / epsilon;
            return State{p / (Rf * T), p, T};
          }};
}

// q increases upward, so mu increases when the vapour is heavier than the
// background. T is a mild inversion. Not a saturated adiabat.
Profile heavy_vapour(std::string name, double gamma, double Z, double mu_d,
                     double mu_v) {
  double epsilon = mu_v / mu_d;
  constexpr double z0 = -2.;
  constexpr int n = 40001;
  double z1 = Z + 2.;
  double h = (z1 - z0) / (n - 1);
  int i0 = (int)std::lround((0. - z0) / h);
  std::vector<double> pp(n);
  auto q_of = [=](double z) {
    return std::min(0.4, std::max(1e-4, 0.01 + 0.20 * (z / Z)));
  };
  auto T_of = [](double z) { return std::max(0.3, 1. + 0.10 * z); };
  auto Rfac = [=](double q) { return (1. - q) + q / epsilon; };
  pp[i0] = 1.;
  auto step = [&](int i, int dir) {
    double z = z0 + i * h;
    double hh = dir * h;
    auto rhs = [&](double z_, double p) {
      return -p / (Rfac(q_of(z_)) * T_of(z_));
    };
    double k1 = rhs(z, pp[i]);
    double k2 = rhs(z + 0.5 * hh, pp[i] + 0.5 * hh * k1);
    double k3 = rhs(z + 0.5 * hh, pp[i] + 0.5 * hh * k2);
    double k4 = rhs(z + hh, pp[i] + hh * k3);
    pp[i + dir] = pp[i] + hh * (k1 + 2 * k2 + 2 * k3 + k4) / 6.;
  };
  for (int i = i0; i < n - 1; ++i) step(i, +1);
  for (int i = i0; i > 0; --i) step(i, -1);
  return {name, Z, gamma, [=](double z) {
            double x =
                std::min(std::max((z - z0) / h, 0.), (double)n - 1.0000001);
            int i = (int)x;
            double f = x - i;
            double p = pp[i] * (1. - f) + pp[i + 1] * f;
            double T = T_of(z);
            double q = q_of(z);
            return State{p / (((1. - q) + q / epsilon) * T), p, T};
          }};
}

double p_average(Profile const& prof, double a, double b) {
  if (prof.kink > a && prof.kink < b) {
    double left = simpson([&](double z) { return prof.at(z).p; }, a, prof.kink);
    double right =
        simpson([&](double z) { return prof.at(z).p; }, prof.kink, b);
    return (left + right) / (b - a);
  }
  return simpson([&](double z) { return prof.at(z).p; }, a, b) / (b - a);
}

std::shared_ptr<MeshBlockImpl> make_block(int nx, double Z, double gamma) {
  double cv_R = 1. / (gamma - 1.);
  std::string path = "/tmp/snapy-250-t1.yaml";
  std::ofstream os(path);
  os << std::setprecision(17);
  os << "reference-state:\n  Tref: 1.\n  Pref: 1.\n"
     << "species:\n  - name: dry\n    composition: {H: 2.}\n    cv_R: " << cv_R
     << "\n"
     << "geometry:\n  type: cartesian\n"
     << "  bounds: {x1min: 0., x1max: " << Z
     << ", x2min: 0., x2max: 1., x3min: 0., x3max: 1.}\n"
     << "  cells: {nx1: " << nx << ", nx2: 1, nx3: 1, nghost: 3}\n"
     << "dynamics:\n  equation-of-state:\n    type: ideal-gas\n    gammad: "
     << gamma << "\n"
     << "    density-floor: 1.e-30\n    pressure-floor: 1.e-30\n"
     << "    limiter: false\n"
     << "  reconstruct:\n"
     << "    vertical: {type: weno5, scale: false, shock: false}\n"
     << "    horizontal: {type: weno5, scale: false, shock: false}\n"
     << "  riemann-solver:\n    type: lmars\n"
     << "forcing:\n  const-gravity:\n    grav1: -1.\n    non-hydrostatic: 1.\n"
     << "integration:\n  type: rk3\n  implicit-scheme: 0\n"
     << "boundary-condition:\n  external:\n    x1-inner: reflecting\n"
     << "    x1-outer: reflecting\n";
  os.close();
  auto options = MeshBlockOptionsImpl::from_yaml(path);
  options->hydro()->icorr() = nullptr;
  auto block = std::make_shared<MeshBlockImpl>(options);
  block->to(torch::kCPU, torch::kFloat64);
  return block;
}

std::vector<int> grids_for(double Z) {
  std::vector<int> nx;
  for (double dz : {2., 1., 0.5, 0.25, 0.125, 0.05, 0.02}) {
    int n = (int)std::lround(Z / dz);
    if (n < 8 || n > 800) continue;
    if (nx.empty() || nx.back() != n) nx.push_back(n);
  }
  return nx;
}

struct Row {
  std::string regime;
  std::string form;
  double dz;
  double err;
  double gamma;
};

double face_error(std::shared_ptr<MeshBlockImpl> const& block,
                  torch::Tensor const& left, torch::Tensor const& right,
                  Profile const& prof) {
  auto x1f = block->pcoord->x1f.to(torch::kCPU, torch::kFloat64).contiguous();
  auto L = left.to(torch::kCPU, torch::kFloat64).contiguous().view({-1});
  auto R = right.to(torch::kCPU, torch::kFloat64).contiguous().view({-1});
  int il = block->pcoord->il();
  int iu = block->pcoord->iu();
  double worst = 0.;
  for (int i = il; i <= iu + 1; ++i) {
    double truth = prof.at(x1f[i].item<double>()).rho;
    double eL = std::abs(L[i].item<double>() - truth) / truth;
    double eR = std::abs(R[i].item<double>() - truth) / truth;
    worst = std::max(worst, std::max(eL, eR));
  }
  return worst;
}

}  // namespace

TEST(hydro, rho_ref_face_error_table) {
  // The constant-N^2 closed form reduces to the isothermal atmosphere at
  // N^2 = kappa.
  {
    double gamma = 1.4;
    double kappa = kappa_of(gamma);
    double z = 1.7;
    double pk = std::exp(-kappa * z);
    double p = std::pow(pk, 1. / kappa);
    double T = std::exp(kappa * z) * pk;
    EXPECT_NEAR(p, std::exp(-z), 1e-12);
    EXPECT_NEAR(T, 1., 1e-12);
  }

  std::vector<Profile> profiles;
  profiles.push_back(dry_adiabat("dry_adiabat/H2He", 1.4, 2.0));
  profiles.push_back(isothermal("isothermal/H2He", 1.4, 16.0));
  profiles.push_back(const_n2("const_N2/H2He", 1.4, 3.0));
  profiles.push_back(inversion("inversion/H2He", 1.4, 4.0));
  profiles.push_back(sharp_tropopause("tropopause_sharp/H2He", 1.4, 4.0));
  profiles.push_back(smooth_tropopause("tropopause_smooth/H2He", 1.4, 4.0));
  profiles.push_back(superadiabatic("superadiabatic/H2He", 1.4, 1.2));
  profiles.push_back(
      moist_sat("moist_H2O_jupiter/H2He", 1.4, 3.0, 18. / 2.3, 0.02, 5.));
  profiles.push_back(
      moist_sat("moist_CH4_neptune/H2He", 1.4, 3.0, 16. / 2.3, 0.05, 3.));
  profiles.push_back(heavy_vapour("heavy_vapour_mu_up/H2He", 1.4, 4.0, 2.3, 40.));
  profiles.push_back(dry_adiabat("dry_adiabat/CO2", 1.3, 2.0));
  profiles.push_back(isothermal("isothermal/CO2", 1.3, 16.0));
  profiles.push_back(
      moist_sat("moist_H2O/CO2", 1.3, 3.0, 18. / 44., 0.02, 5.));
  // N2 has the same gamma as H2-He, so equal dz/H is the same column. One
  // grid checks that the nondimensional error collapses.
  profiles.push_back(isothermal("isothermal/N2", 1.4, 4.0));

  char const* forms[] = {"smooth5", "isentrope", "none", "local_polytrope"};
  std::map<std::string, std::shared_ptr<MeshBlockImpl>> blocks;
  std::vector<Row> rows;

  for (auto const& prof : profiles) {
    auto sample = prof.at(0.3 * prof.Z);
    ASSERT_GT(sample.rho, 0.) << prof.name;
    ASSERT_GT(sample.p, 0.) << prof.name;
    ASSERT_GT(prof.at(0.).T, 0.) << prof.name;
    ASSERT_GT(prof.at(prof.Z).T, 0.05)
        << prof.name << " T(Z)=" << prof.at(prof.Z).T;

    for (int nx : grids_for(prof.Z)) {
      std::ostringstream key;
      key << std::setprecision(17) << nx << "|" << prof.Z << "|" << prof.gamma;
      if (!blocks.count(key.str())) {
        blocks[key.str()] = make_block(nx, prof.Z, prof.gamma);
      }
      auto block = blocks[key.str()];
      auto coord = block->pcoord;
      int nc1 = coord->options->nc1();
      auto x1f =
          coord->x1f.to(torch::kCPU, torch::kFloat64).contiguous().view({-1});
      ASSERT_EQ(x1f.numel(), nc1 + 1);
      double dz = (x1f[coord->il() + 1] - x1f[coord->il()]).item<double>();

      auto w = torch::zeros(
          {block->phydro->peos->nvar(), 1, 1, nc1}, torch::kFloat64);
      for (int i = 0; i < nc1; ++i) {
        double a = x1f[i].item<double>();
        double b = x1f[i + 1].item<double>();
        double p_lo = prof.at(a).p;
        double p_hi = prof.at(b).p;
        double rho_c = (p_lo - p_hi) / (b - a);
        double p_c = p_average(prof, a, b);
        ASSERT_GT(rho_c, 0.) << prof.name << " cell " << i;
        ASSERT_GT(p_c, 0.) << prof.name << " cell " << i;
        w[IDN][0][0][i] = rho_c;
        w[IPR][0][0][i] = p_c;
      }

      for (char const* form : forms) {
        auto faces = block->phydro->face_density_x1(w, form);
        double err = face_error(block, faces.first, faces.second, prof);
        rows.push_back({prof.name, form, dz, err, prof.gamma});
        EXPECT_TRUE(std::isfinite(err)) << prof.name << " " << form;
        EXPECT_LT(err, 10.) << prof.name << " " << form << " dz/H=" << dz;
      }

      auto a = block->phydro->face_density_x1(w, "smooth5");
      auto b = block->phydro->face_density_x1(w, "smooth5");
      EXPECT_TRUE(torch::equal(a.first, b.first));
      EXPECT_TRUE(torch::equal(a.second, b.second));
    }
  }

  std::ostringstream md;
  md << std::scientific << std::setprecision(3);
  md << "regime | form | dz/H | gamma | max relative face error\n";
  md << "---|---|---|---|---\n";
  for (auto const& row : rows) {
    md << row.regime << " | " << row.form << " | " << row.dz << " | "
       << row.gamma << " | " << row.err << "\n";
  }

  // Worst ratio to the best form at the same regime and dz/H.
  md << "\nform | worst error | worst ratio to the best form | where\n";
  md << "---|---|---|---\n";
  for (char const* form : forms) {
    double worst_err = 0.;
    double worst_ratio = 0.;
    std::string where_err, where_ratio;
    for (auto const& row : rows) {
      if (row.form != form) continue;
      if (row.err > worst_err) {
        worst_err = row.err;
        where_err = row.regime + " dz/H=" + std::to_string(row.dz);
      }
      double best = row.err;
      for (auto const& other : rows) {
        if (other.regime == row.regime && other.dz == row.dz)
          best = std::min(best, other.err);
      }
      double ratio = best > 0. ? row.err / best : 1.;
      if (ratio > worst_ratio) {
        worst_ratio = ratio;
        where_ratio = row.regime + " dz/H=" + std::to_string(row.dz);
      }
    }
    md << form << " | " << worst_err << " | " << worst_ratio << " | "
       << where_ratio << "\n";
  }
  md << "\nSame ratio, restricted to dz/H <= 0.5 (at least 2 cells per H).\n";
  md << "form | worst error | worst ratio to the best form | where\n";
  md << "---|---|---|---\n";
  for (char const* form : forms) {
    double worst_err = 0.;
    double worst_ratio = 0.;
    std::string where_ratio;
    for (auto const& row : rows) {
      if (row.form != form || row.dz > 0.5 + 1e-9) continue;
      if (row.err > worst_err) worst_err = row.err;
      double best = row.err;
      for (auto const& other : rows) {
        if (other.regime == row.regime && other.dz == row.dz)
          best = std::min(best, other.err);
      }
      double ratio = best > 0. ? row.err / best : 1.;
      if (ratio > worst_ratio) {
        worst_ratio = ratio;
        std::ostringstream loc;
        loc << std::scientific << std::setprecision(3) << row.regime
            << " dz/H=" << row.dz;
        where_ratio = loc.str();
      }
    }
    md << form << " | " << worst_err << " | " << worst_ratio << " | "
       << where_ratio << "\n";
  }

  std::cout << md.str() << std::flush;
  std::ofstream out("/tmp/snapy-250-t1-table.md");
  out << md.str();

  // Classic separation: on a tall isothermal column, smooth5 beats a
  // bottom-anchored isentrope. This is the issue's own misfit, checked here
  // as a face error so a swapped form fails the run.
  auto find = [&](std::string regime, std::string form, double dz) {
    for (auto const& row : rows) {
      if (row.regime == regime && row.form == form &&
          std::abs(row.dz - dz) < 1e-6)
        return row.err;
    }
    return -1.;
  };
  double iso_s = find("isothermal/H2He", "smooth5", 0.25);
  double iso_i = find("isothermal/H2He", "isentrope", 0.25);
  EXPECT_GT(iso_s, 0.);
  EXPECT_LT(iso_s, iso_i);
}
