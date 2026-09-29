// C/C++
#include <string>

// snap
#include <snap/snap.h>

#include <snap/bc/bc_func.hpp>
#include <snap/mesh/meshblock.hpp>

#include "hydro.hpp"

namespace snap {

std::pair<torch::Tensor, torch::Tensor> HydroImpl::face_density_x1(
    torch::Tensor const& w_in, std::string const& form) {
  constexpr int DIM1 = 3;
  auto w = w_in.clone();
  bool phys_in = pmb->options->is_physical_boundary(0, 0, -1);
  bool phys_out = pmb->options->is_physical_boundary(0, 0, 1);

  auto refs = _hydro_ref_x1(w);
  auto psf_lo = std::get<0>(refs);
  auto pref = std::get<1>(refs);
  auto dsf = std::get<2>(refs);
  auto dref = std::get<3>(refs);
  auto density = w[IDN].clone();
  auto pressure = w[IPR].clone();

  if (form == "smooth5") {
    // production reference
  } else if (form == "none") {
    dref.zero_();
    dsf.zero_();
  } else if (form == "isentrope") {
    // Bottom interior cell's adiabat: rho_ref = rho_b (p / p_b)^(1/gamma).
    // The face uses the production face-pressure reference psf_lo.
    int is = pmb->pcoord->il();
    auto rho_b = density.narrow(-1, is, 1);
    auto p_b = pressure.narrow(-1, is, 1);
    auto gamma = peos->compute("W->A", {w.narrow(-1, is, 1)});
    auto inv_g = 1. / gamma;
    dref = rho_b * (pressure / p_b).pow(inv_g);
    dsf = rho_b * (psf_lo / p_b).pow(inv_g);
  } else if (form == "local_polytrope") {
    // n_i = dln p / dln rho across the two neighbours. At a cell, p = p_i so
    // dref = rho and the perturbation is zero. The face reference is the mean
    // of the two adjacent cells' polytropes evaluated at psf_lo.
    int nc1 = density.size(-1);
    int is = pmb->pcoord->il();
    auto gamma = peos->compute("W->A", {w.narrow(-1, is, 1)});
    auto dln_rho = torch::log(density.narrow(-1, 2, nc1 - 2) /
                              density.narrow(-1, 0, nc1 - 2));
    auto dln_p = torch::log(pressure.narrow(-1, 2, nc1 - 2) /
                            pressure.narrow(-1, 0, nc1 - 2));
    auto n_mid = torch::where(dln_rho.abs() < 1e-8, gamma, dln_p / dln_rho);
    n_mid = torch::where(n_mid.abs() < 0.05, gamma, n_mid);
    auto n = density.clone();
    n.narrow(-1, 1, nc1 - 2).copy_(n_mid);
    n.select(-1, 0).copy_(n_mid.select(-1, 0));
    n.select(-1, nc1 - 1).copy_(n_mid.select(-1, nc1 - 3));

    dref = density.clone();
    auto pface = psf_lo.narrow(-1, 1, nc1 - 1);
    auto from_b = density.narrow(-1, 0, nc1 - 1) *
                  (pface / pressure.narrow(-1, 0, nc1 - 1))
                      .pow(1. / n.narrow(-1, 0, nc1 - 1));
    auto from_a = density.narrow(-1, 1, nc1 - 1) *
                  (pface / pressure.narrow(-1, 1, nc1 - 1))
                      .pow(1. / n.narrow(-1, 1, nc1 - 1));
    dsf.narrow(-1, 1, nc1 - 1).copy_(0.5 * (from_b + from_a));
    dsf.select(-1, 0).copy_(dsf.select(-1, 1));
  } else {
    TORCH_CHECK(false, "[Hydro] unknown rho_ref form: ", form);
  }

  w[IPR] -= pref;
  w[IDN] -= dref;

  int ng = pmb->pcoord->options->nghost();
  int is = pmb->pcoord->il();
  int iu = pmb->pcoord->iu();
  for (int c : {(int)IPR, (int)IDN}) {
    if (phys_in && !is_outflow(pmb->options->bfuncs()[0])) {
      w[c].narrow(-1, is - ng, ng).copy_(w[c].narrow(-1, is, ng).flip(-1));
    }
    if (phys_out && !is_outflow(pmb->options->bfuncs()[1])) {
      w[c].narrow(-1, iu + 1, ng)
          .copy_(w[c].narrow(-1, iu + 1 - ng, ng).flip(-1));
    }
  }

  auto wtmp = precon1->forward(w, DIM1, /*floor=*/false);
  auto dl = wtmp[ILT][IDN] + dsf;
  auto dr = wtmp[IRT][IDN] + dsf;
  auto n1 = density.size(-1);
  auto rho_below = torch::cat(
      {density.narrow(-1, 0, 1), density.narrow(-1, 0, n1 - 1)}, -1);
  auto left = torch::where(dl > 0., dl, rho_below);
  auto right = torch::where(dr > 0., dr, density);
  return {left, right};
}

}  // namespace snap
