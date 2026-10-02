// C/C++
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

// torch
#include <c10/core/InferenceMode.h>

// kintera
#include <kintera/constants.h>

#include <kintera/thermo/eval_uhs.hpp>
#include <kintera/thermo/log_svp.hpp>
#include <kintera/thermo/thermo.hpp>

// snap
#include <snap/snap.h>

#include <snap/eos/ideal_moist.hpp>

#include "hydro_rho_moist.hpp"

namespace snap {
namespace {

int64_t g_fb_left = 0;
int64_t g_fb_right = 0;

void print_wb_positivity_fallback() {
  std::fprintf(stderr, "WB_POSITIVITY_FALLBACK il=%lld ir=%lld\n",
               static_cast<long long>(g_fb_left),
               static_cast<long long>(g_fb_right));
  std::fflush(stderr);
}

void register_fallback_atexit() {
  static bool once = false;
  if (!once) {
    once = true;
    std::atexit(print_wb_positivity_fallback);
  }
}

struct CellMax {
  double value;
  int64_t i;
  int64_t j;
};

// i is the x2 index and j the x1 index, both relative to (js, is).
CellMax worst_cell(torch::Tensor const& field, int64_t is, int64_t js) {
  auto flat = field.reshape({-1});
  auto pos = flat.argmax().item<int64_t>();
  auto n1 = field.size(-1);
  auto n2 = field.size(-2);
  auto j_abs = pos % n1;
  auto i_abs = (pos / n1) % n2;
  return {flat[pos].item<double>(), i_abs - js, j_abs - is};
}

kintera::ThermoX thermo_x_for(kintera::ThermoY const& thermo_y,
                              torch::Device device) {
  static kintera::ThermoX tx;
  static bool init = false;
  if (!init) {
    tx = kintera::ThermoX(thermo_y->options);
    init = true;
  }
  if (tx->mu.device() != device) tx->to(device);
  return tx;
}

// Same latent-heat solve as ThermoXImpl::effective_cp. This deck has one
// reaction, and linalg_lstsq on a batch of 1x1 systems is most of the step;
// the 1x1 solution is the quotient. Any other reaction count uses the library.
torch::Tensor cp_mole(kintera::ThermoX tx, torch::Tensor const& temp,
                      torch::Tensor const& pres, torch::Tensor const& xfrac,
                      torch::Tensor const& gain, torch::Tensor const& conc) {
  if (!gain.defined() || gain.size(-1) != 1 || gain.size(-2) != 1) {
    return tx->effective_cp(temp, pres, xfrac, gain, conc);
  }
  kintera::LogSVPFunc::init(tx->options->nucleation());
  auto logsvp_ddT = kintera::LogSVPFunc::grad(temp);
  auto row_scale = gain.abs().amax(-1, /*keepdim=*/true);
  row_scale =
      torch::where(row_scale > 0., row_scale, torch::ones_like(row_scale));
  auto scaled = (gain / row_scale).squeeze(-1).squeeze(-1);
  auto rhs = (logsvp_ddT / row_scale.squeeze(-1)).squeeze(-1);
  auto rate =
      torch::where(scaled.abs() > 0., rhs / scaled, torch::zeros_like(scaled));
  auto enthalpy = kintera::eval_enthalpy_R(temp, conc, tx->options) *
                  kintera::constants::Rgas;
  auto cp =
      kintera::eval_cp_R(temp, conc, tx->options) * kintera::constants::Rgas;
  auto cp_normal = (cp * xfrac).sum(-1);
  auto cp_latent = (enthalpy.matmul(tx->stoich) * rate.unsqueeze(-1)).sum(-1);
  return cp_normal + cp_latent;
}

// Library extrapolate_dlnp applies one scalar dlnp. Here each cell has its
// own target pressure; the iteration is otherwise that routine with rainout
// left off (reset to the source mole fractions, keep the condensate).
torch::Tensor density_on_adiabat(kintera::ThermoX tx, torch::Tensor temp,
                                 torch::Tensor pres, torch::Tensor xfrac0,
                                 torch::Tensor s_target, char const* what) {
  c10::InferenceMode guard;
  int max_iter = tx->options->max_iter();
  double tol = 10. * tx->options->ftol();
  auto work_x = xfrac0.clone();
  torch::Tensor ds;
  double resid = std::numeric_limits<double>::infinity();
  int iter = 0;
  while (iter++ < max_iter) {
    work_x.copy_(xfrac0);
    auto gain = tx->forward(temp, pres, work_x);
    auto conc = tx->compute("TPX->V", {temp, pres, work_x});
    auto cp = cp_mole(tx, temp, pres, work_x, gain, conc);
    static bool cp_checked = false;
    if (!cp_checked && gain.defined() && gain.size(-1) == 1) {
      cp_checked = true;
      auto ref = tx->effective_cp(temp, pres, work_x, gain, conc);
      double scale = std::max(ref.abs().max().item<double>(), 1.0);
      double err = (cp - ref).abs().max().item<double>() / scale;
      std::fprintf(stderr, "WB_CP_CHECK max_rel=%.3e\n", err);
      std::fflush(stderr);
      TORCH_CHECK(err <= 1e-8,
                  "[Hydro] moist adiabat 1x1 cp does not match effective_cp, "
                  "max_rel=",
                  err);
    }
    auto entropy = tx->compute("TPV->S", {temp, pres, conc}) / conc.sum(-1);
    ds = s_target - entropy;
    resid = ds.abs().max().item<double>();
    if (!(resid < tol)) {
      temp.mul_(1. + ds / cp);
    } else {
      break;
    }
  }
  bool finite = temp.isfinite().all().item<bool>() &&
                work_x.isfinite().all().item<bool>() && std::isfinite(resid);
  if (!(resid < tol) || !finite) {
    CellMax at{resid, -1, -1};
    if (ds.defined() && ds.isfinite().any().item<bool>()) {
      at = worst_cell(ds.abs(), 0, 0);
    }
    print_wb_positivity_fallback();
    TORCH_CHECK(false, "[Hydro] wb-density-ref moist adiabat (", what,
                ") did not converge: iter=", iter, " max_iter=", max_iter,
                " resid=", resid, " tol=", tol, " finite=", finite,
                " worst_abs_i=", at.i, " worst_abs_j=", at.j,
                ". Positivity-fallback counts are the line above.");
  }
  auto conc = tx->compute("TPX->V", {temp, pres, work_x});
  auto rho = tx->compute("V->D", {conc});
  if (!rho.isfinite().all().item<bool>()) {
    print_wb_positivity_fallback();
    TORCH_CHECK(false, "[Hydro] wb-density-ref moist adiabat (", what,
                ") produced a non-finite density.");
  }
  return rho;
}

torch::Tensor molar_entropy(kintera::ThermoX tx, torch::Tensor const& temp,
                            torch::Tensor const& pres,
                            torch::Tensor const& xfrac) {
  auto conc = tx->compute("TPX->V", {temp, pres, xfrac});
  return tx->compute("TPV->S", {temp, pres, conc}) / conc.sum(-1);
}

}  // namespace

void note_wb_positivity_fallback(int64_t nleft, int64_t nright) {
  register_fallback_atexit();
  g_fb_left += nleft;
  g_fb_right += nright;
}

void log_wb_dref_t0(torch::Tensor const& rho, torch::Tensor const& dref, int is,
                    int iu, int js, int ju) {
  static bool logged = false;
  if (logged) return;
  logged = true;
  auto rel = (rho - dref).abs() / rho.abs().clamp_min(1e-300);
  auto interior = rel.slice(-1, is, iu + 1).slice(-2, js, ju + 1);
  auto at = worst_cell(interior, 0, 0);
  std::fprintf(stderr, "WB_DREF_T0 max_rel=%.17g i=%lld j=%lld\n", at.value,
               static_cast<long long>(at.i), static_cast<long long>(at.j));
  std::fflush(stderr);
}

void apply_moist_density_ref(IdealMoistImpl* moist, torch::Tensor const& w,
                             torch::Tensor const& pref,
                             torch::Tensor const& psf_lo, torch::Tensor& dref,
                             torch::Tensor& dsf, bool per_cell, int is, int iu,
                             int js, int ju) {
  register_fallback_atexit();
  TORCH_CHECK(moist && moist->pthermo,
              "[Hydro] moist density reference "
              "requires ideal-moist thermo");
  c10::InferenceMode guard;
  auto clock = std::chrono::steady_clock::now();
  int ny = moist->pthermo->options->vapor_ids().size() +
           moist->pthermo->options->cloud_ids().size() - 1;
  auto temp = moist->compute("W->T", {w}).contiguous();
  auto pres = w[IPR].contiguous();
  auto yfrac = w.narrow(0, ICY, ny).contiguous();
  auto xfrac = moist->pthermo->compute("Y->X", {yfrac}).contiguous();
  auto tx = thermo_x_for(moist->pthermo, temp.device());

  torch::Tensor s_src;
  torch::Tensor x_src;
  torch::Tensor temp_src;
  if (per_cell) {
    s_src = molar_entropy(tx, temp, pres, xfrac);
    x_src = xfrac;
    temp_src = temp;
  } else {
    auto temp_b = temp.narrow(-1, is, 1);
    auto pres_b = pres.narrow(-1, is, 1);
    auto x_b = xfrac.narrow(-2, is, 1);
    s_src = molar_entropy(tx, temp_b, pres_b, x_b).expand_as(temp);
    x_src = x_b.expand_as(xfrac).contiguous();
    temp_src = temp;
  }

  auto nbad_pres = (pres <= 0).sum().item<int64_t>() +
                   (pref <= 0).sum().item<int64_t>() +
                   (psf_lo <= 0).sum().item<int64_t>();
  auto nbad_temp = (temp <= 0).sum().item<int64_t>();
  if (nbad_pres != 0 || nbad_temp != 0) {
    print_wb_positivity_fallback();
    TORCH_CHECK(false,
                "[Hydro] wb-density-ref moist adiabat: non-positive "
                "pressure or temperature cells, n_pres=",
                nbad_pres, " n_temp=", nbad_temp);
  }

  static int calls = 0;
  static bool identity_done = false;
  if (!identity_done) {
    identity_done = true;
    torch::Tensor rho_hat;
    torch::Tensor rho_ref;
    if (per_cell) {
      rho_hat = density_on_adiabat(tx, temp.clone(), pres.clone(), xfrac, s_src,
                                   "identity");
      rho_ref = w[IDN];
    } else {
      auto temp_b = temp.narrow(-1, is, 1);
      auto pres_b = pres.narrow(-1, is, 1);
      auto x_b = xfrac.narrow(-2, is, 1).contiguous();
      auto s_b = s_src.narrow(-1, is, 1).contiguous();
      rho_hat = density_on_adiabat(tx, temp_b.clone(), pres_b.clone(), x_b, s_b,
                                   "identity-column");
      rho_ref = w[IDN].narrow(-1, is, 1);
    }
    auto rel = (rho_hat - rho_ref).abs() / rho_ref.abs().clamp_min(1e-300);
    // Column identity is one x1 row (the anchor). Cell identity is the field.
    torch::Tensor interior;
    if (per_cell) {
      interior = rel.slice(-1, is, iu + 1).slice(-2, js, ju + 1);
    } else {
      interior = rel.slice(-2, js, ju + 1);
    }
    auto at = worst_cell(interior, 0, 0);
    std::fprintf(stderr, "WB_IDENTITY form=%s max_rel=%.17g i=%lld j=%lld\n",
                 per_cell ? "moist_cell" : "moist_column", at.value,
                 static_cast<long long>(at.i), static_cast<long long>(at.j));
    std::fflush(stderr);
    if (!(at.value <= 1e-10)) {
      print_wb_positivity_fallback();
      TORCH_CHECK(false,
                  "[Hydro] wb-density-ref moist adiabat identity "
                  "failed: rho_ad(p_i; s_i, qt_i) vs rho_i max_rel=",
                  at.value, " (bar 1e-10) i=", at.i, " j=", at.j);
    }
  }

  auto pres_tgt = torch::stack({pref, psf_lo}, 0);
  auto lead = pres_tgt.sizes().vec();
  lead.push_back(x_src.size(-1));
  auto temp_tgt = temp_src.unsqueeze(0).expand_as(pres_tgt).contiguous();
  auto x_tgt = x_src.unsqueeze(0).expand(lead).contiguous();
  auto s_tgt = s_src.unsqueeze(0).expand_as(pres_tgt).contiguous();
  auto rho_tgt =
      density_on_adiabat(tx, temp_tgt, pres_tgt.clone(), x_tgt, s_tgt,
                         per_cell ? "moist_cell" : "moist_column");
  dref.copy_(rho_tgt[0]);
  dsf.copy_(rho_tgt[1]);

  if (calls < 5) {
    double sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - clock)
            .count();
    std::fprintf(stderr, "WB_MOIST_CALL form=%s n=%d seconds=%.6f\n",
                 per_cell ? "moist_cell" : "moist_column", calls, sec);
    std::fflush(stderr);
  }
  ++calls;
}

}  // namespace snap
