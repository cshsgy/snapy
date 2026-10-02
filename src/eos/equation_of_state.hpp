#pragma once

// C/C++
#include <utility>
#include <vector>

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>

// kintera
#include <kintera/thermo/thermo.hpp>

// arg
#include <snap/add_arg.h>

namespace snap {

struct EquationOfStateOptionsImpl {
  static std::shared_ptr<EquationOfStateOptionsImpl> create() {
    return std::make_shared<EquationOfStateOptionsImpl>();
  }
  static std::shared_ptr<EquationOfStateOptionsImpl> from_yaml(
      std::string const& filename, bool verbose = false);

  EquationOfStateOptionsImpl() = default;
  void report(std::ostream& os) const {
    os << "-- equation of state options --\n";
    os << "* type = " << type() << "\n"
       << "* gammad = " << gammad() << "\n"
       << "* weight = " << weight() << "\n"
       << "* density_floor = " << density_floor() << "\n"
       << "* pressure_floor = " << pressure_floor() << "\n"
       << "* temperature_floor = " << temperature_floor() << "\n"
       << "* verbose = " << (verbose() ? "true" : "false") << "\n"
       << "* limiter = " << (limiter() ? "true" : "false") << "\n"
       << "* eos_file = " << eos_file() << "\n";
    if (thermo()) {
      os << "-- thermo options --\n";
      thermo()->report(os);
    }
  }

  ADD_ARG(std::string, type) = "moist-mixture";
  ADD_ARG(double, gammad) = 1.4;     // ratio of specific heats (cp/cv)
  ADD_ARG(double, weight) = 29.e-3;  // mean molecular weight in kg/mol

  ADD_ARG(double, density_floor) = 1.e-10;
  ADD_ARG(double, pressure_floor) = 1.e-10;
  ADD_ARG(double, temperature_floor) = 20.;
  ADD_ARG(bool, limiter) = false;
  ADD_ARG(std::string, eos_file) = "";
  ADD_ARG(bool, verbose) = false;

  //! submodules options
  ADD_ARG(kintera::ThermoOptions, thermo) = nullptr;
};
using EquationOfStateOptions = std::shared_ptr<EquationOfStateOptionsImpl>;

class HydroImpl;

class EquationOfStateImpl {
 public:
  //! Create and register an `EquationOfState` module
  /*!
   * This function registers the created module as a submodule
   * of the given parent module `p`.
   *
   * \param[in] opts  options for creating the `EquationOfState` module
   * \param[in] p     parent module for registering the created module
   * \param[in] name  name for registering the created module
   * \return          created `EquationOfState` module
   */
  static std::shared_ptr<EquationOfStateImpl> create(
      EquationOfStateOptions const& opts, torch::nn::Module* p,
      std::string const& name = "eos");

  //! options with which this `EquationOfState` was constructed
  EquationOfStateOptions options;

  //! non-owning reference to parent
  HydroImpl const* phydro = nullptr;

  EquationOfStateImpl() : options(EquationOfStateOptionsImpl::create()) {}
  explicit EquationOfStateImpl(EquationOfStateOptions const& options_,
                               torch::nn::Module* p = nullptr);
  virtual ~EquationOfStateImpl() = default;

  virtual int nvar() const { return 5; }

  //! \brief Return the molecular weight of species \p n.
  //!
  //! \param[in] n Index of the species for which to return the molecular
  //!              weight (defaults to 0).
  //! \return Molecular weight of the requested species, in kg/mol (or the
  //!         units consistent with the underlying thermodynamic model).
  virtual double species_weight(int n = 0) const { return 0.; }

  //! \brief Return the reference specific heat at constant volume of species \p
  //! n.
  //!
  //! \param[in] n Index of the species for which to return the reference
  //!              specific heat (defaults to 0).
  //! \return Reference specific heat at constant volume for the requested
  //!         species, in J/(kg·K) (or the units consistent with the underlying
  //!         thermodynamic model).
  virtual double species_cv_ref(int n = 0) const { return 0.; }

  //! \brief Return the local mixture specific heat at constant volume.
  //!
  //! \param[in] prim Primitive hydrodynamic variables.
  //! \param[in] temp Temperature.
  //! \return Specific heat at constant volume in J/(kg·K), or units
  //!         consistent with the underlying thermodynamic model.
  virtual torch::Tensor specific_heat_cv(torch::Tensor prim,
                                         torch::Tensor temp);

  //! \brief Return the additive internal-energy offset carried by conserved
  //! constituent densities.
  virtual torch::Tensor internal_energy_offset(torch::Tensor hydro_like) const;

  //! \brief Energy per unit mass each species carries through a face: its
  //! internal and kinetic energy plus its partial pressure over its density.
  //! \return (ny, nc3, nc2, nc1); undefined when the EOS has no such split.
  virtual torch::Tensor species_enthalpy(torch::Tensor prim) { return {}; }

  //! \brief Computes hydrodynamic variables from the given abbreviation
  /*!
   * These five abbreviations should be supported:
   *  - "W->U": convert primitive variables to conserved variables
   *  - "U->W": convert conserved variables to primitive variables
   *  - "WA->L": compute sound speed from primitive variables and adiabatic
   *  - "W->A": compute adiabatic index from conserved variables
   *  - "W->T": compute temperature
   *
   * \param[in] ab    abbreviation for the computation
   * \param[in] args  arguments for the computation
   * \return computed hydrodynamic variables
   */
  virtual torch::Tensor compute(std::string ab,
                                std::vector<torch::Tensor> const& args = {});

  // virtual torch::Tensor get_buffer(std::string) const;

  torch::Tensor forward(torch::Tensor cons,
                        torch::optional<torch::Tensor> out = torch::nullopt);

  //! \brief Apply the conserved variable limiter in place.
  //! \param whole_column repair a parentless cloud and the vapor over the
  //!        whole x1 column when it spans meshblocks; all blocks of the column
  //!        must then call together, as MeshBlockImpl::advance_local does
  //!        (#232).
  virtual void apply_conserved_limiter_(torch::Tensor const& cons,
                                        bool whole_column = false);

  //! \brief Apply the primitive variable limiter in place.
  virtual void apply_primitive_limiter_(torch::Tensor const& prim);

  //! \brief Zero the step's limiter marks on the device of \p like.
  void reset_limiter_marks(torch::Tensor const& like);

  //! [0]: a limiter call floored an interior density or energy; [1]: it found
  //! an interior NaN. Undefined (nothing marked) before the first reset.
  //! A side channel: the limiter calls set the marks as a side effect; the
  //! MeshBlock resets them at stage 0 and after a redo decision, and reads them
  //! in check_redo (MeshBlock and Mesh).
  torch::Tensor const& limiter_marks() const { return limiter_marks_; }

 private:
  // not a buffer: stage forcings get named_buffers()
  torch::Tensor limiter_marks_;

  //! Parent vapor slots and normalized stoichiometric mass fractions by cloud.
  std::vector<std::vector<std::pair<int, double>>> cloud_parent_cache_;

  void cache_cloud_parents_();
};

using EquationOfState = std::shared_ptr<EquationOfStateImpl>;

}  // namespace snap

#undef ADD_ARG
