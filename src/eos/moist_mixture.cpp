// kintera
#include <kintera/constants.h>

#include <kintera/thermo/eval_uhs.hpp>
#include <kintera/thermo/thermo.hpp>

// snap
#include <snap/snap.h>

#include <snap/coord/coord_utils.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/mesh/meshblock.hpp>

#include "moist_mixture.hpp"

namespace snap {

void MoistMixtureImpl::reset() {
  TORCH_CHECK(options->thermo(), "[MoistMixture] thermo pointer is null");
  pthermo = kintera::ThermoYImpl::create(options->thermo(), this);

  // make gammad and weight consistent
  options->gammad(1. + 1. / options->thermo()->cref_R()[0]);
  options->weight(1. / pthermo->inv_mu[0].item<double>());

  // populate buffers
  ivol = register_buffer("ivol", torch::empty({0}, torch::kFloat64));
  temp = register_buffer("temp", torch::empty({0}, torch::kFloat64));
  cached_prim_.reset();
  cached_prim_version_ = -1;
}

double MoistMixtureImpl::species_weight(int n) const {
  return 1. / pthermo->inv_mu[n].item<double>();
}

double MoistMixtureImpl::species_cv_ref(int n) const {
  auto Ri = kintera::constants::Rgas * pthermo->inv_mu[n];
  return (options->thermo()->cref_R()[n] * Ri).item<double>();
}

torch::Tensor MoistMixtureImpl::specific_heat_cv(torch::Tensor prim,
                                                 torch::Tensor temp) {
  int ny = pthermo->options->vapor_ids().size() +
           pthermo->options->cloud_ids().size() - 1;
  auto ivol = pthermo->compute("DY->V", {prim[IDN], prim.narrow(0, ICY, ny)});
  return pthermo->compute("VT->cv", {ivol, temp}) / prim[IDN];
}

torch::Tensor MoistMixtureImpl::compute(
    std::string ab, std::vector<torch::Tensor> const& args) {
  if (ab == "W->U") {
    auto w = args[0];
    auto u = args.size() > 1 ? args[1] : torch::empty_like(w);
    _prim2cons(w, u);
    return u;
  } else if (ab == "U->W") {
    auto u = args[0];
    auto w = args.size() > 1 ? args[1] : torch::empty_like(u);
    _cons2prim(u, w);
    return w;
  } else if (ab == "W->I") {
    auto w = args[0];
    return _prim2intEng(w);
  } else if (ab == "W->T") {
    auto w = args[0];
    return _prim2temp(w);
  } else if (ab == "W->E") {
    auto w = args[0];
    return _prim2speciesEng(w);
  } else if (ab == "U->K") {
    auto u = args[0];
    return _cons2ke(u);
  } else if (ab == "UT->I") {
    auto u = args[0];
    auto temp = args[1];
    return _temp2intEng(u, temp);
  } else if (ab == "W->A") {
    auto w = args[0];
    _ensure_cache(w);
    return _adiabatic_index(ivol, temp);
  } else if (ab == "WA->L") {
    auto w = args[0];
    auto gamma = args[1];
    _ensure_cache(w);
    auto ct = _isothermal_sound_speed(ivol, temp, w[IDN]);
    return gamma.sqrt() * ct;
  } else {
    TORCH_CHECK(false, "Unknown abbreviation: ", ab);
  }
}

void MoistMixtureImpl::_prim2cons(torch::Tensor prim, torch::Tensor& cons) {
  auto pcoord = phydro->pmb->pcoord;

  apply_primitive_limiter_(prim);
  int ny = pthermo->options->vapor_ids().size() +
           pthermo->options->cloud_ids().size() - 1;

  // den -> den
  auto out = cons[IDN];
  torch::mul_out(out, (1. - prim.narrow(0, ICY, ny).sum(0)), prim[IDN]);

  // mixr -> den
  out = cons.narrow(0, ICY, ny);
  torch::mul_out(out, prim.narrow(0, ICY, ny), prim[IDN]);

  // vel -> mom
  out = cons.narrow(0, IVX, 3);
  torch::mul_out(out, prim.narrow(0, IVX, 3), prim[IDN]);

  coord_vec_lower_(out, pcoord->cosine_cell_kj);

  // KE
  auto ke = 0.5 * (prim.narrow(0, IVX, 3) * cons.narrow(0, IVX, 3)).sum(0);

  // IE
  cons[IPR] = _prim2intEng(prim);
  cons[IPR] += ke;

  apply_conserved_limiter_(cons);
}

void MoistMixtureImpl::_cons2prim(torch::Tensor cons, torch::Tensor& prim) {
  auto pcoord = phydro->pmb->pcoord;
  apply_conserved_limiter_(cons);

  int ny = pthermo->options->vapor_ids().size() +
           pthermo->options->cloud_ids().size() - 1;

  // den -> den
  auto out = prim[IDN];
  torch::sum_out(out, cons.narrow(0, ICY, ny), /*dim=*/0);
  out += cons[IDN];

  // den -> mixr
  out = prim.narrow(0, ICY, ny);
  torch::div_out(out, cons.narrow(0, ICY, ny), prim[IDN]);

  // mom -> vel
  out = prim.narrow(0, IVX, 3);
  torch::div_out(out, cons.narrow(0, IVX, 3), prim[IDN]);

  coord_vec_raise_(out, pcoord->cosine_cell_kj);

  auto ke = 0.5 * (prim.narrow(0, IVX, 3) * cons.narrow(0, IVX, 3)).sum(0);
  auto ie = cons[IPR] - ke;

  ivol.set_(pthermo->compute("DY->V", {prim[IDN], prim.narrow(0, ICY, ny)}));
  temp.set_(pthermo->compute("VU->T", {ivol, ie}));
  prim[IPR] = pthermo->compute("VT->P", {ivol, temp});

  apply_primitive_limiter_(prim);
  if (options->limiter()) {
    // Floors may have changed density, composition, or pressure. Defer the
    // refresh until a cached quantity is actually requested.
    cached_prim_.reset();
    cached_prim_version_ = -1;
  } else {
    _mark_cache(prim);
  }
}

torch::Tensor MoistMixtureImpl::_prim2intEng(torch::Tensor prim) {
  _ensure_cache(prim);
  return pthermo->compute("VT->U", {ivol, temp});
}

torch::Tensor MoistMixtureImpl::_prim2temp(torch::Tensor prim) {
  _ensure_cache(prim);
  return temp;
}

torch::Tensor MoistMixtureImpl::_prim2speciesEng(torch::Tensor prim) {
  auto pcoord = phydro->pmb->pcoord;

  int ny = pthermo->options->vapor_ids().size() +
           pthermo->options->cloud_ids().size() - 1;

  auto yfrac = prim.narrow(0, ICY, ny);

  _ensure_cache(prim);

  auto Rgas = kintera::constants::Rgas * pthermo->inv_mu;
  auto ie = eval_intEng_R(temp, ivol, pthermo->options) * Rgas * ivol;

  auto vel = prim.narrow(0, IVX, 3).clone();

  coord_vec_lower_(vel, pcoord->cosine_cell_kj);
  auto ke = 0.5 * (prim.narrow(0, IVX, 3) * vel).sum(0, /*keepdim=*/true);

  auto rhoc = prim[IDN] * yfrac;
  return ie.narrow(-1, 1, ny).permute({3, 0, 1, 2}) + ke * rhoc;
}

torch::Tensor MoistMixtureImpl::species_enthalpy(torch::Tensor prim) {
  auto pcoord = phydro->pmb->pcoord;
  int ngas = pthermo->options->vapor_ids().size();
  int ny = ngas + pthermo->options->cloud_ids().size() - 1;

  _ensure_cache(prim);
  auto conc = ivol * pthermo->inv_mu;

  // the per-species split of the flux enthalpy U + p + rho*KE, with the U of
  // "VT->U" and the p of "VT->P" (p = R T sum_gas c_n z_n): u_n + KE, plus
  // z_n R_n T for a vapour (a cloud has no pressure share)
  auto h = kintera::eval_intEng_R(temp, conc, pthermo->options);
  h.narrow(-1, 0, ngas) +=
      kintera::eval_czh(temp, conc.narrow(-1, 0, ngas), pthermo->options) *
      temp.unsqueeze(-1);
  h *= kintera::constants::Rgas * pthermo->inv_mu;

  auto vel = prim.narrow(0, IVX, 3).clone();
  coord_vec_lower_(vel, pcoord->cosine_cell_kj);
  auto ke = 0.5 * (prim.narrow(0, IVX, 3) * vel).sum(0);

  return h.narrow(-1, 1, ny).permute({3, 0, 1, 2}) + ke;
}

torch::Tensor MoistMixtureImpl::_cons2ke(torch::Tensor cons) {
  auto pcoord = phydro->pmb->pcoord;

  int ny = pthermo->options->vapor_ids().size() +
           pthermo->options->cloud_ids().size() - 1;
  auto rho = cons[IDN] + cons.narrow(0, ICY, ny).sum(0);
  auto mom = cons.narrow(0, IVX, 3).clone();
  coord_vec_raise_(mom, pcoord->cosine_cell_kj);

  return 0.5 * (cons.narrow(0, IVX, 3) * mom).sum(0) / rho;
}

torch::Tensor MoistMixtureImpl::_temp2intEng(torch::Tensor cons,
                                             torch::Tensor T) {
  int ny = pthermo->options->vapor_ids().size() +
           pthermo->options->cloud_ids().size() - 1;
  auto vec = T.sizes().vec();
  vec.push_back(ny + 1);

  auto V = torch::empty(vec, T.options());
  V.select(-1, IDN) = cons[IDN];
  V.narrow(-1, 1, ny) = cons.narrow(0, ICY, ny).permute({1, 2, 3, 0});
  return pthermo->compute("VT->U", {V, T});
}

torch::Tensor MoistMixtureImpl::_adiabatic_index(torch::Tensor V,
                                                 torch::Tensor T) {
  auto conc = V * pthermo->inv_mu;
  auto cp = kintera::eval_cp_R(T, conc, pthermo->options);
  auto cv = kintera::eval_cv_R(T, conc, pthermo->options);

  auto cp_vol = (conc * cp).sum(-1);
  auto cv_vol = (conc * cv).sum(-1);
  return cp_vol / cv_vol;
}

torch::Tensor MoistMixtureImpl::_isothermal_sound_speed(torch::Tensor V,
                                                        torch::Tensor T,
                                                        torch::Tensor dens) {
  int nvapor = pthermo->options->vapor_ids().size();
  auto conc_gas = (V * pthermo->inv_mu).narrow(-1, 0, nvapor);
  auto cz = kintera::eval_czh(T, conc_gas, pthermo->options);
  auto cz_ddC = kintera::eval_czh_ddC(T, conc_gas, pthermo->options);

  auto result = torch::addcmul(cz, cz_ddC, conc_gas);
  result *= conc_gas;

  auto ct = result.sum(-1);
  ct *= kintera::constants::Rgas * T / dens;
  ct.sqrt_();

  return ct;
}

bool MoistMixtureImpl::_cache_matches(torch::Tensor const& prim) const {
  return cached_prim_.defined() && cached_prim_.is_same(prim) &&
         cached_prim_version_ == prim._version();
}

void MoistMixtureImpl::_mark_cache(torch::Tensor const& prim) {
  cached_prim_ = prim;
  cached_prim_version_ = prim._version();
}

void MoistMixtureImpl::_ensure_cache(torch::Tensor const& prim) {
  if (_cache_matches(prim)) return;

  int ny = pthermo->options->vapor_ids().size() +
           pthermo->options->cloud_ids().size() - 1;
  ivol.set_(pthermo->compute("DY->V", {prim[IDN], prim.narrow(0, ICY, ny)}));
  temp.set_(pthermo->compute("PV->T", {prim[IPR], ivol}));
  _mark_cache(prim);
}

}  // namespace snap
