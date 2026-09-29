// C/C++
#include <string>

// snap
#include <snap/snap.h>

#include <snap/mesh/meshblock.hpp>

#include "hydro.hpp"

namespace snap {

std::pair<torch::Tensor, torch::Tensor> HydroImpl::_density_ref_x1(
    torch::Tensor const& w, torch::Tensor const& psf_lo,
    std::string const& form) const {
  auto density = w[IDN];
  auto pressure = w[IPR];
  auto dref = torch::empty_like(density);
  auto dsf = torch::empty_like(psf_lo);

  if (form == "none") {
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
    dsf = psf_lo.clone();
    dsf.narrow(-1, 1, nc1 - 1).copy_(0.5 * (from_b + from_a));
    dsf.select(-1, 0).copy_(dsf.select(-1, 1));
  } else {
    TORCH_CHECK(false, "[Hydro] unknown rho_ref form: ", form);
  }

  return {dref, dsf};
}

}  // namespace snap
