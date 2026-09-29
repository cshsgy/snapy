#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>

#include <string>
#include <utility>
#include <vector>

// snap
#include <snap/eos/equation_of_state.hpp>
#include <snap/forcing/forcing.hpp>
#include <snap/implicit/implicit_hydro.hpp>
#include <snap/layout/layout.hpp>
#include <snap/recon/reconstruct.hpp>
#include <snap/riemann/riemann_solver.hpp>
#include <snap/sedimentation/sedimentation.hpp>

// arg
#include <snap/add_arg.h>

namespace snap {

class MeshBlockImpl;

struct HydroOptionsImpl {
  static std::shared_ptr<HydroOptionsImpl> create() {
    return std::make_shared<HydroOptionsImpl>();
  }
  static std::shared_ptr<HydroOptionsImpl> from_yaml(
      std::string const& filename, bool verbose = false);

  HydroOptionsImpl() = default;
  std::shared_ptr<HydroOptionsImpl> clone() const;
  void report(std::ostream& os) const {
    os << "-- hydro options --\n";
    os << "* verbose = " << verbose() << "\n"
       << "* disable_flux_x1 = " << disable_flux_x1() << "\n"
       << "* disable_flux_x2 = " << disable_flux_x2() << "\n"
       << "* disable_flux_x3 = " << disable_flux_x3() << "\n"
       << "* rho_ref = " << rho_ref() << "\n";
  }

  //! verbose
  ADD_ARG(bool, verbose) = false;

  ADD_ARG(bool, disable_flux_x1) = false;
  ADD_ARG(bool, disable_flux_x2) = false;
  ADD_ARG(bool, disable_flux_x3) = false;

  //! Keep the well-balanced x1 reference's stencils off the wall ghosts
  ADD_ARG(bool, wb_wall_clamp) = true;

  //! Density reference used by the well-balanced x1 reconstruction.
  //! smooth5 is the production reference and must stay bit-identical.
  //! isentrope, none, and local_polytrope replace only dref and dsf.
  ADD_ARG(std::string, rho_ref) = "smooth5";

  //! forcing options
  ADD_ARG(ConstGravityOptions, grav) = nullptr;
  ADD_ARG(CoriolisOptions, coriolis) = nullptr;
  ADD_ARG(DiffusionOptions, diffusion) = nullptr;
  ADD_ARG(BodyHeatOptions, bodyHeat) = nullptr;
  ADD_ARG(BotHeatOptions, botHeat) = nullptr;
  ADD_ARG(TopCoolOptions, topCool) = nullptr;
  ADD_ARG(RelaxBotCompOptions, relaxBotComp) = nullptr;
  ADD_ARG(RelaxBotTempOptions, relaxBotTemp) = nullptr;
  ADD_ARG(RelaxBotVeloOptions, relaxBotVelo) = nullptr;
  ADD_ARG(TopSpongeLyrOptions, topSpongeLyr) = nullptr;
  ADD_ARG(BotSpongeLyrOptions, botSpongeLyr) = nullptr;
  ADD_ARG(PlumeForcingOptions, plumeForcing) = nullptr;

  //! submodule options
  ADD_ARG(EquationOfStateOptions, eos) = nullptr;

  ADD_ARG(ReconstructOptions, recon1) = nullptr;
  ADD_ARG(ReconstructOptions, recon23) = nullptr;
  ADD_ARG(RiemannSolverOptions, riemann) = nullptr;

  ADD_ARG(ImplicitOptions, icorr) = nullptr;
  ADD_ARG(SedHydroOptions, sed) = nullptr;
};

using HydroOptions = std::shared_ptr<HydroOptionsImpl>;
using Variables = std::map<std::string, torch::Tensor>;

class HydroImpl : public torch::nn::Cloneable<HydroImpl> {
 public:
  //! \brief Create and register a `Hydro` module
  /*!
   * This function registers the created module as a submodule
   * of the given parent module `p`.
   *
   * \param[in] opts  options for creating the `Hydro` module
   * \param[in] p     parent module for registering the created module
   * \param[in] name  name for registering the created module
   * \return          created `Hydro` module
   */
  static std::shared_ptr<HydroImpl> create(HydroOptions const& opts,
                                           torch::nn::Module* p,
                                           std::string const& name = "hydro");

  //! options with which this `Hydro` was constructed
  HydroOptions options;

  //! non-owning reference to parent
  MeshBlockImpl const* pmb = nullptr;

  //! owning submodules
  EquationOfState peos = nullptr;
  RiemannSolver priemann = nullptr;

  Reconstruct precon1 = nullptr;
  Reconstruct precon23 = nullptr;

  ImplicitHydro picorr = nullptr;

  SedHydro psed = nullptr;

  //! forcings
  std::vector<torch::nn::AnyModule> forcings;
  Diffusion pdiffusion = nullptr;

  //! Constructor to initialize the layers
  HydroImpl() = default;
  explicit HydroImpl(const HydroOptions& options_,
                     torch::nn::Module* p = nullptr);
  void reset() override;

  double max_time_step(torch::Tensor hydro_w,
                       torch::Tensor solid = torch::Tensor()) const;

  //! Advance the conserved variables by one time step.
  torch::Tensor forward(double dt, torch::Tensor hydro_u,
                        Variables const& other);

  torch::Tensor flux1() const { return _flux1; }
  torch::Tensor flux2() const { return _flux2; }
  torch::Tensor flux3() const { return _flux3; }
  torch::Tensor face_pressure1() const { return _face_pressure1; }
  torch::Tensor implicit_mass_correction() const;

  //! dry-density increment the forcings added this stage (du[IDN])
  torch::Tensor forcing_dry_increment() const { return _forcing_dry; }

  //! cumulative count of (cell, species) entries with positivity theta < 1
  //! (diagnostic; accumulated when the EOS limiter is enabled)
  torch::Tensor positivity_hits() const { return _positivity_hits; }
  //! how HARD the limiter bites, not just how often
  torch::Tensor positivity_severe() const { return _positivity_severe; }
  torch::Tensor positivity_min() const { return _positivity_min; }
  //! x1 face flux the limiter removes, and the total offered (lifetime sums)
  torch::Tensor lim_cut() const { return _lim_cut; }
  torch::Tensor lim_flux() const { return _lim_flux; }

  //! RK stage currently being advanced, published by
  //! MeshBlockImpl::advance_local. The vertical implicit correction needs
  //! it because that solve is nonlinear in dt (see hydro_forward.cpp).
  //! -1 means "not set". Reaching the correction in that state means running
  //! the FULL-dt operator -- the one this fix replaces -- so it warns; it does
  //! not abort, because tests/test_forcing.cpp drives HydroImpl::forward
  //! directly, outside the stage loop, and must keep working.
  int rk_stage = -1;

 protected:
  void _revise_x1inner_ghost(torch::Tensor const& w);
  void _revise_x1outer_ghost(torch::Tensor const& w);

  void _revise_x1inner_lr(torch::Tensor const& wl, torch::Tensor const& wr);
  void _revise_x1outer_lr(torch::Tensor const& wl, torch::Tensor const& wt);

  // Per-column hydrostatic references for the well-balanced x1
  // reconstruction: {psf_lo (face pressure), pref (cell pressure), dsf (face
  // density), dref (cell density)}, rebuilt from the current field on every
  // call.
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
  _hydro_ref_x1(torch::Tensor const& w) const;

  //! Cell and face density references for a non-production rho_ref.
  //! Pressure reference is not touched. Index i is the lower face of cell i.
  std::pair<torch::Tensor, torch::Tensor> _density_ref_x1(
      torch::Tensor const& w, torch::Tensor const& psf_lo,
      std::string const& form) const;
  torch::Tensor _apply_implicit_correction(torch::Tensor& du,
                                           torch::Tensor const& w, double dt,
                                           Variables const& other);

 private:
  //! Register all forcing modules
  std::vector<std::string> _register_forcings_module();

  //! x1 grid uniformity (-1 unknown, 0 non-uniform, 1 uniform), probed once;
  //! selects the six-face vs log-mean cell-pressure reference
  mutable int x1_uniform_ = -1;

  torch::Tensor _flux1, _flux2, _flux3, _face_pressure1, _div, _forcing_dry;
  torch::Tensor _positivity_hits, _positivity_severe, _positivity_min;
  torch::Tensor _lim_cut, _lim_flux;
};

TORCH_MODULE(Hydro);

/*void check_recon(torch::Tensor wlr, int nghost, int extend_x1, int extend_x2,
                 int extend_x3);
void check_eos(torch::Tensor w, int nghost);*/
}  // namespace snap

#undef ADD_ARG
