#include <kintera/utils/format.hpp>

// C/C++
#include <algorithm>

// snap
#include <snap/snap.h>

#include <snap/eos/ideal_moist.hpp>
#include <snap/mesh/meshblock.hpp>
#include <snap/utils/log.hpp>

#include "hydro.hpp"
#include "hydro_dispatch.hpp"
#include "hydro_rho_moist.hpp"

namespace snap {
HydroImpl::HydroImpl(const HydroOptions& options_, torch::nn::Module* p)
    : options(options_) {
  pmb = dynamic_cast<MeshBlockImpl const*>(p);
  reset();
}

void HydroImpl::reset() {
  TORCH_CHECK(pmb, "[Hydro] Parent MeshBlock is null");

  //// ---- (1) set up equation-of-state model ---- ////
  peos = EquationOfStateImpl::create(options->eos(), this);
  if (options->verbose()) {
    SINFO(Hydro) << "EOS type: " << peos->options->type() << "\n";
  }

  //// ---- (3) set up reconstruction-x1 model ---- ////
  precon1 = ReconstructImpl::create(options->recon1(), this, "recon1");
  if (options->verbose()) {
    SINFO(Hydro) << "Reconstruction-x1 type: "
                 << precon1->pinterp1->options->type() << "\n";
  }

  //// ---- (4) set up reconstruction-x23 model ---- ////
  precon23 = ReconstructImpl::create(options->recon23(), this, "recon23");
  if (options->verbose()) {
    SINFO(Hydro) << "Reconstruction-x2/x3 type: "
                 << precon23->pinterp1->options->type() << "\n";
  }

  //// ---- (5) set up riemann-solver model ---- ////
  priemann = RiemannSolverImpl::create(options->riemann(), this);
  if (options->verbose()) {
    SINFO(Hydro) << "Riemann solver type: " << priemann->options->type()
                 << "\n";
  }

  //// ---- (6) set up implicit solver ---- ////
  if (options->icorr()) {
    picorr = ImplicitHydroImpl::create(options->icorr(), this);
    if (options->verbose()) {
      SINFO(Hydro) << "Implicit correction type: " << picorr->options->type()
                   << "\n";
    }
  }

  //// ---- (7) set up sedimentation ---- ////
  if (options->sed() != nullptr) {
    psed = SedHydroImpl::create(options->sed(), this);
    if (options->verbose()) {
      SINFO(Hydro) << "Sedimentation particle ids: "
                   << fmt::format("{}", psed->options->sedvel()->particle_ids())
                   << "\n";
    }
  }

  //// ---- (8) set up forcings ---- ////
  auto forcing_names = _register_forcings_module();
  if (options->verbose()) {
    SINFO(Hydro) << "Forcings: " << fmt::format("{}", forcing_names) << "\n";
  }

  //// ---- (9) register all forcings ---- ////
  for (auto i = 0; i < forcings.size(); i++) {
    register_module(forcing_names[i], forcings[i].ptr());
  }

  //// ---- (10) populate buffers ---- ////
  int nc1 = pmb->options->coord()->nc1();
  int nc2 = pmb->options->coord()->nc2();
  int nc3 = pmb->options->coord()->nc3();
  int nvar = peos->nvar();

  if (nc1 > 1) {
    _flux1 = register_buffer(
        "F1", torch::zeros({nvar, nc3, nc2, nc1}, torch::kFloat64));
    _face_pressure1 =
        register_buffer("P1", torch::zeros({nc3, nc2, nc1}, torch::kFloat64));
  } else {
    _flux1 = register_buffer("F1", torch::Tensor());
    _face_pressure1 = register_buffer("P1", torch::Tensor());
  }

  if (nc2 > 1) {
    _flux2 = register_buffer(
        "F2", torch::zeros({nvar, nc3, nc2, nc1}, torch::kFloat64));
  } else {
    _flux2 = register_buffer("F2", torch::Tensor());
  }

  if (nc3 > 1) {
    _flux3 = register_buffer(
        "F3", torch::zeros({nvar, nc3, nc2, nc1}, torch::kFloat64));
  } else {
    _flux3 = register_buffer("F3", torch::Tensor());
  }

  _div = register_buffer("D",
                         torch::zeros({nvar, nc3, nc2, nc1}, torch::kFloat64));

  _positivity_hits =
      register_buffer("positivity_hits", torch::zeros({1}, torch::kInt64));
  _positivity_severe =
      register_buffer("positivity_severe", torch::zeros({1}, torch::kInt64));
  _positivity_min =
      register_buffer("positivity_min", torch::ones({1}, torch::kFloat64));
  _lim_cut = register_buffer("lim_cut", torch::zeros({1}, torch::kFloat64));
  _lim_flux = register_buffer("lim_flux", torch::zeros({1}, torch::kFloat64));
}

double HydroImpl::max_time_step(torch::Tensor w, torch::Tensor solid) const {
  auto sub3 = pmb->part({0, 0, 0}, PartOptions().exterior(false).ndim(3));

  torch::Tensor cs;
  if (options->eos()->type() == "aneos") {
    cs = peos->compute("W->L", {w});
  } else {
    auto gamma = peos->compute("W->A", {w});
    cs = peos->compute("WA->L", {w, gamma});
  }

  if (solid.defined()) {
    cs = torch::where(solid, 1.e-8, cs);
  }

  auto dt_min = torch::tensor({1.e9, 1.e9, 1.e9},
                              torch::dtype(torch::kFloat64).device(w.device()));

  auto icorr = options->icorr();

  if (icorr) {
    auto adv = pmb->pintg->options->cfl() / icorr->advection_cfl();
    if (cs.size(2) > 1) {
      auto denom1 =
          (!(icorr->scheme() & 1) || (cs.size(0) == 1 && cs.size(1) == 1))
              ? (w[IVX].abs() + cs)
              : w[IVX].abs() * adv;
      dt_min[0] = (pmb->pcoord->center_width1() / denom1).index(sub3).min();
    }

    if (cs.size(1) > 1) {
      auto denom2 = (!((icorr->scheme() >> 1) & 1)) ? (w[IVY].abs() + cs)
                                                    : w[IVY].abs() * adv;
      dt_min[1] = (pmb->pcoord->center_width2() / denom2).index(sub3).min();
    }

    if (cs.size(0) > 1) {
      auto denom3 = (!((icorr->scheme() >> 2) & 1)) ? (w[IVZ].abs() + cs)
                                                    : w[IVZ].abs() * adv;
      dt_min[2] = (pmb->pcoord->center_width3() / denom3).index(sub3).min();
    }

    // A horizontal wind that jumps by more than cs across an x1 face
    if (icorr->shear_cfl() > 0. && cs.size(2) > 1) {
      auto n1 = cs.size(2);
      auto csf = 0.5 * (cs.slice(2, 0, n1 - 1) + cs.slice(2, 1, n1));
      auto scale = icorr->shear_cfl() / pmb->pintg->options->cfl();
      auto face_bound = [&](int iv, torch::Tensor width) {
        auto lo = w[iv].slice(2, 0, n1 - 1);
        auto hi = w[iv].slice(2, 1, n1);
        auto prod = (lo.abs() * hi.abs()).clamp_min(1.e-30);
        auto dts = scale * csf * width.slice(2, 0, n1 - 1) / prod;
        return torch::where((hi - lo).abs() >= csf, dts, 1.e9)
            .index(sub3)
            .min();
      };
      if (cs.size(1) > 1) {
        dt_min[0] = torch::minimum(
            dt_min[0], face_bound(IVY, pmb->pcoord->center_width2()));
      }
      if (cs.size(0) > 1) {
        dt_min[0] = torch::minimum(
            dt_min[0], face_bound(IVZ, pmb->pcoord->center_width3()));
      }
    }
  } else {
    if (cs.size(2) > 1) {
      dt_min[0] = (pmb->pcoord->center_width1() / (w[IVX].abs() + cs))
                      .index(sub3)
                      .min();
    }

    if (cs.size(1) > 1) {
      dt_min[1] = (pmb->pcoord->center_width2() / (w[IVY].abs() + cs))
                      .index(sub3)
                      .min();
    }

    if (cs.size(0) > 1) {
      dt_min[2] = (pmb->pcoord->center_width3() / (w[IVZ].abs() + cs))
                      .index(sub3)
                      .min();
    }
  }

  double dt = dt_min.min().item<double>();
  if (pdiffusion) dt = std::min(dt, pdiffusion->max_time_step(w));
  return dt;
}

torch::Tensor HydroImpl::implicit_mass_correction() const {
  return picorr ? picorr->mass_correction() : torch::Tensor();
}

torch::Tensor HydroImpl::_apply_implicit_correction(torch::Tensor& du,
                                                    torch::Tensor const& w,
                                                    double dt,
                                                    Variables const& other) {
  if (!picorr) return torch::Tensor();

  // Implicit x1 solve has no cross-rank coupling, so nb1 > 1 would silently
  // solve each rank's own sub-column. Full column <=> both x1 faces physical.
  TORCH_CHECK(pmb->options->is_physical_boundary(0, 0, -1) &&
                  pmb->options->is_physical_boundary(0, 0, 1),
              "[Hydro] implicit scheme requires nb1 = 1 (no x1 decomposition): "
              "the vertical solve has no cross-rank coupling.");

  torch::Tensor wi;
  if (other.count("solid")) {
    wi = torch::where(other.at("solid").unsqueeze(0).expand_as(w),
                      other.at("fill_solid_hydro_w"), w);
    du.masked_fill_(other.at("solid").unsqueeze(0).expand_as(du), 0.0);
  } else {
    wi = w;
  }

  du[IPR].sub_(peos->internal_energy_offset(du));

  torch::Tensor gamma;
  if (options->eos()->type() == "aneos") {
    auto cs = peos->compute("W->L", {w});
    gamma = peos->compute("WL->A", {w, cs});
  } else {
    gamma = peos->compute("W->A", {wi});
  }
  auto correction = picorr->forward(du, wi, gamma, dt);
  du[IPR].add_(peos->internal_energy_offset(du));
  // picorr measured its delta after removing the EOS reference energy.
  // Diagnostics expose a conserved-state delta, so restore that reference
  // contribution using the corrected density and species tendencies.
  correction[IPR].add_(peos->internal_energy_offset(correction));

  return correction;
}

void HydroImpl::_revise_x1inner_lr(torch::Tensor const& wl,
                                   torch::Tensor const& wr) {
  int is = pmb->pcoord->il();
  wl[IPR].narrow(-1, is, 1) = wr[IPR].narrow(-1, is, 1);
  wl[IDN].narrow(-1, is, 1) = wr[IDN].narrow(-1, is, 1);
}

void HydroImpl::_revise_x1outer_lr(torch::Tensor const& wl,
                                   torch::Tensor const& wr) {
  int ie = pmb->pcoord->iu();
  wr[IPR].narrow(-1, ie + 1, 1) = wl[IPR].narrow(-1, ie + 1, 1);
  wr[IDN].narrow(-1, ie + 1, 1) = wl[IDN].narrow(-1, ie + 1, 1);
}

void HydroImpl::_revise_x1inner_ghost(torch::Tensor const& w) {
  auto pcoord = pmb->pcoord;
  int is = pcoord->il();
  auto grav = -options->grav()->grav1();

  auto gamma = peos->compute("W->A", {w.narrow(-1, is, 1)});
  auto gm = gamma - 1.;
  auto a = gm / gamma;
  auto K = w[IPR].narrow(-1, is, 1) / w[IDN].narrow(-1, is, 1).pow(gamma);

  for (int n = 0; n < pcoord->options->nghost(); ++n) {
    auto dz = pmb->pcoord->dx1v[is - n - 1];
    auto h =
        w[IPR].narrow(-1, is - n, 1).pow(a) + a * grav * dz / K.pow(1. / gamma);
    w[IPR].narrow(-1, is - n - 1, 1) = h.pow(1. / a);
    w[IDN].narrow(-1, is - n - 1, 1) =
        (w[IPR].narrow(-1, is - n - 1, 1) / K).pow(1. / gamma);
  }
}

void HydroImpl::_revise_x1outer_ghost(torch::Tensor const& w) {
  auto pcoord = pmb->pcoord;
  int ie = pcoord->iu();
  auto grav = -options->grav()->grav1();

  auto gamma = peos->compute("W->A", {w.narrow(-1, ie, 1)});
  auto gm = gamma - 1.;
  auto a = gm / gamma;
  auto K = w[IPR].narrow(-1, ie, 1) / w[IDN].narrow(-1, ie, 1).pow(gamma);

  for (int n = 0; n < pcoord->options->nghost(); ++n) {
    auto dz = pmb->pcoord->dx1v[ie + n];
    auto h =
        w[IPR].narrow(-1, ie + n, 1).pow(a) - a * grav * dz / K.pow(1. / gamma);
    w[IPR].narrow(-1, ie + n + 1, 1) = h.pow(1. / a);
    w[IDN].narrow(-1, ie + n + 1, 1) =
        (w[IPR].narrow(-1, ie + n + 1, 1) / K).pow(1. / gamma);
  }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
HydroImpl::_hydro_ref_x1(torch::Tensor const& w) const {
  auto pcoord = pmb->pcoord;
  int is = pcoord->il();
  int iu = pcoord->iu();
  double g = -options->grav()->grav1();  // downward magnitude (>0)

  // Global x1 reference across a vertical (nb1>1) decomposition: make the
  // hydrostatic reference continuous over the x1 process seams. The block
  // owning x1-outer anchors at the true domain top; every block below receives
  // the running seam-face pressure from the block above and passes on that
  // value plus its own interior hydrostatic drop (= its bottom-face pressure).
  // A serial top->bottom scan along the x1 process column. nb1 == 1 (no x1
  // neighbor) => an empty anchor tells the backend to compute the local top
  // anchor.
  torch::Tensor anchor;
  int below = -1;
  int above = -1;
  bool x1_split = false;
  auto layout = pmb->get_layout();
  if (layout && layout->has_process_group() && !layout->options->periodic_z() &&
      layout->options->pz() >
          1) {  // pz==nb1: relay only across a SPLIT x1 column (else unmatched
                // send/recv at nb1=1)
    x1_split = true;
    TORCH_CHECK(layout->options->blocks_per_process() == 1,
                "[Hydro] the x1 reference relay addresses block ranks as "
                "process ranks: one block per process only");
    auto iloc = layout->loc_of(layout->options->rank());
    above = layout->neighbor_rank(iloc, {0, 0, 1});   // toward x1-outer
    below = layout->neighbor_rank(iloc, {0, 0, -1});  // toward x1-inner
    constexpr int kWbRefTag = 0x7715;

    if (above >= 0) {
      std::vector<torch::Tensor> rbuf = {
          torch::empty({w.size(1), w.size(2), 1}, w.options())};
      layout->comm->recv(rbuf, above, kWbRefTag)->wait();
      anchor = rbuf[0];
    }
  }

  if (x1_uniform_ < 0) {
    auto d = pcoord->dx1f.to(torch::kCPU);
    x1_uniform_ =
        ((d.max() - d.min()).item<double>() < 1e-10 * d.mean().item<double>())
            ? 1
            : 0;
  }

  auto ref_options = w.options();
  auto ref_sizes = w.sizes().slice(1).vec();
  auto psf_lo = torch::empty(ref_sizes, ref_options);
  auto psf_hi = torch::empty(ref_sizes, ref_options);
  auto pref = torch::empty(ref_sizes, ref_options);
  auto dsf = torch::empty(ref_sizes, ref_options);
  auto dref = torch::empty(ref_sizes, ref_options);
  bool phys_in = pmb->options->is_physical_boundary(0, 0, -1);
  bool phys_out = pmb->options->is_physical_boundary(0, 0, 1);

  auto dx1f = pcoord->dx1f.contiguous();
  at::native::call_hydro_ref_x1(
      w.device().type(), w, dx1f, anchor, psf_lo, psf_hi, pref, dsf, dref, iu,
      g, x1_uniform_ == 1, phys_in, phys_out, options->wb_wall_clamp(),
      options->wb_rop_guard());

  // The backend computes the smooth5 density reference; the other forms
  // replace it. Both only change what the reconstruction sees as rho', never
  // p_ref, so the rest state is balanced identically for every form.
  auto const& form = options->wb_density_ref();
  if (form == "none") {
    dref.zero_();
    dsf.zero_();
  } else if (form == "isentrope") {
    // One adiabat per column through the bottom interior cell. It must be the
    // same adiabat on every block of the column, which the x1 relay does not
    // carry yet, so refuse a split x1 column rather than seam the reference.
    TORCH_CHECK(!x1_split && phys_in,
                "[Hydro] wb-density-ref: isentrope needs the whole x1 column "
                "on one block (nb1 == 1)");
    auto wb = w.narrow(-1, is, 1);
    auto gamma = peos->compute("W->A", {wb});
    auto rho_b = wb[IDN];
    auto p_b = wb[IPR];
    dref.copy_(rho_b * (pref / p_b).pow(1. / gamma));
    dsf.copy_(rho_b * (psf_lo / p_b).pow(1. / gamma));
  } else if (form == "local_polytrope") {
    // Ported from cshsgy/snapy study/250-t2-modes (612976c),
    // src/hydro/hydro_rho_ref_study.cpp:36-62. n_i = dln p / dln rho across
    // the two neighbours (the bottom cell's gamma where |dln rho| < 1e-8 or
    // |n| < 0.05; the edge cells copy their neighbour). At a cell p = p_i,
    // so dref = rho and the cell perturbation is zero. The face reference is
    // the mean of the two adjacent cells' polytropes evaluated at psf_lo.
    auto density = w[IDN];
    auto pressure = w[IPR];
    int nc1 = density.size(-1);
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

    dref.copy_(density);
    auto pface = psf_lo.narrow(-1, 1, nc1 - 1);
    auto from_b = density.narrow(-1, 0, nc1 - 1) *
                  (pface / pressure.narrow(-1, 0, nc1 - 1))
                      .pow(1. / n.narrow(-1, 0, nc1 - 1));
    auto from_a = density.narrow(-1, 1, nc1 - 1) *
                  (pface / pressure.narrow(-1, 1, nc1 - 1))
                      .pow(1. / n.narrow(-1, 1, nc1 - 1));
    dsf.copy_(psf_lo);
    dsf.narrow(-1, 1, nc1 - 1).copy_(0.5 * (from_b + from_a));
    dsf.select(-1, 0).copy_(dsf.select(-1, 1));
  } else if (form == "moist_cell" || form == "moist_column") {
    auto* moist = dynamic_cast<IdealMoistImpl*>(peos.get());
    TORCH_CHECK(moist, "[Hydro] wb-density-ref: ", form,
                " requires an ideal-moist equation of state");
    if (form == "moist_column") {
      // Same adiabat on every block of the column. The x1 relay does not
      // carry the anchor entropy, so refuse a split column.
      TORCH_CHECK(!x1_split && phys_in,
                  "[Hydro] wb-density-ref: moist_column needs the whole x1 "
                  "column on one block (nb1 == 1)");
    }
    apply_moist_density_ref(moist, w, pref, psf_lo, dref, dsf,
                            form == "moist_cell", is, iu, pcoord->jl(),
                            pcoord->ju());
  }

  log_wb_dref_t0(w[IDN], dref, is, iu, pcoord->jl(), pcoord->ju());

  if (below >= 0) {
    constexpr int kWbRefTag = 0x7715;
    std::vector<torch::Tensor> sbuf = {psf_lo.narrow(-1, is, 1).contiguous()};
    layout->comm->send(sbuf, below, kWbRefTag)->wait();
  }

  // Ghost-row reference exchange across x1 seams: overwrite this block's
  // ghost rows of (pref, dref) with the NEIGHBOR'S interior rows for the same
  // physical cells. The block-local ghost quadrature (w6e edge rows, and the
  // [lo,hi] guard whose accept/fallback decision is stencil-dependent) does
  // NOT reproduce the owner's interior values -- near a physical wall the
  // mismatch reaches ~1e2 Pa (measured), which enters
  // w' = w - ref as a spurious seam perturbation every step and makes the
  // dynamics nb1-dependent. After this exchange the perturbation field seen
  // by the reconstruction is identical on both sides of every seam, so the
  // seam-face states agree and the single-valued seam flux average becomes a
  // no-op. nb1=1: no seams, bit-unchanged.
  if (x1_split && (below >= 0 || above >= 0)) {
    constexpr int kWbGhostUpTag = 0x7717;
    constexpr int kWbGhostDnTag = 0x7718;
    int ng = is;  // il() == nghost
    std::vector<CommWorkPtr> sends;
    if (above >=
        0) {  // my top interior rows are the above block's lower ghosts
      std::vector<torch::Tensor> up = {
          torch::cat({pref.narrow(-1, iu - ng + 1, ng),
                      dref.narrow(-1, iu - ng + 1, ng)},
                     -1)
              .contiguous()};
      sends.push_back(layout->comm->send(up, above, kWbGhostUpTag));
    }
    if (below >=
        0) {  // my bottom interior rows are the below block's upper ghosts
      std::vector<torch::Tensor> dn = {
          torch::cat({pref.narrow(-1, is, ng), dref.narrow(-1, is, ng)}, -1)
              .contiguous()};
      sends.push_back(layout->comm->send(dn, below, kWbGhostDnTag));
    }
    if (below >= 0) {  // receive my lower ghost rows from below's top interior
      std::vector<torch::Tensor> rb = {
          torch::empty({w.size(1), w.size(2), 2 * ng}, w.options())};
      layout->comm->recv(rb, below, kWbGhostUpTag)->wait();
      pref.narrow(-1, 0, ng).copy_(rb[0].narrow(-1, 0, ng));
      dref.narrow(-1, 0, ng).copy_(rb[0].narrow(-1, ng, ng));
    }
    if (above >=
        0) {  // receive my upper ghost rows from above's bottom interior
      std::vector<torch::Tensor> ra = {
          torch::empty({w.size(1), w.size(2), 2 * ng}, w.options())};
      layout->comm->recv(ra, above, kWbGhostDnTag)->wait();
      pref.narrow(-1, iu + 1, ng).copy_(ra[0].narrow(-1, 0, ng));
      dref.narrow(-1, iu + 1, ng).copy_(ra[0].narrow(-1, ng, ng));
    }
    for (auto& sw : sends) sw->wait();
  }

  return {psf_lo, pref, dsf, dref};
}

std::shared_ptr<HydroImpl> HydroImpl::create(HydroOptions const& opts,
                                             torch::nn::Module* p,
                                             std::string const& name) {
  TORCH_CHECK(p, "[Hydro] Parent module is null");
  TORCH_CHECK(opts, "[Hydro] Options pointer is null");

  return p->register_module(name, Hydro(opts, p));
}

/*void check_recon(torch::Tensor wlr, int nghost, int extend_x1, int extend_x2,
                 int extend_x3) {
  auto interior =
      get_interior(wlr.sizes(), nghost, extend_x1, extend_x2, extend_x3);

  int dim = extend_x1 == 1 ? 1 : (extend_x2 == 1 ? 2 : 3);
  TORCH_CHECK(wlr.index(interior).select(1, IDN).min().item<double>() > 0.,
              "Negative density detected after reconstruction in dimension ",
              dim);
  TORCH_CHECK(wlr.index(interior).select(1, IPR).min().item<double>() > 0.,
              "Negative pressure detected after reconstruction in dimension ",
              dim);
}

void check_eos(torch::Tensor w, int nghost) {
  auto interior = get_interior(w.sizes(), nghost);
  TORCH_CHECK(w.index(interior)[IDN].min().item<double>() > 0.,
              "Negative density detected after EOS. ",
              "Suggestions: 1) Reducting the CFL number;",
              " 2) Activate EOS limiter and set the density floor");
  TORCH_CHECK(w.index(interior)[IPR].min().item<double>() > 0.,
              "Negative pressure detected after EOS. ",
              "Suggestions: 1) Reducting the CFL number; ",
              " 2) Activate EOS limiter and set the pressure floor");
}*/

}  // namespace snap
