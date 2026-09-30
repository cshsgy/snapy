#pragma once

// snap
#include "equation_of_state.hpp"

namespace snap {

class MoistMixtureImpl final : public torch::nn::Cloneable<MoistMixtureImpl>,
                               public EquationOfStateImpl {
 public:
  //! \cache
  torch::Tensor ivol, temp;

  //! submodules
  kintera::ThermoY pthermo = nullptr;

  // Constructor to initialize the layers
  MoistMixtureImpl() = default;
  explicit MoistMixtureImpl(EquationOfStateOptions const& options_,
                            torch::nn::Module* p = nullptr)
      : EquationOfStateImpl(options_, p) {
    reset();
  }
  void reset() override;
  // void pretty_print(std::ostream& os) const override;
  using EquationOfStateImpl::forward;

  int nvar() const override {
    return 4 + pthermo->options->vapor_ids().size() +
           pthermo->options->cloud_ids().size();
  }
  double species_weight(int n = 0) const override;
  double species_cv_ref(int n = 0) const override;
  torch::Tensor specific_heat_cv(torch::Tensor prim,
                                 torch::Tensor temp) override;
  torch::Tensor species_enthalpy(torch::Tensor prim) override;

  /*torch::Tensor get_buffer(std::string var) const override {
    return named_buffers()[var];
  }*/

  //! \brief Implementation of moist mixture equation of state.
  /*!
   * Conversions "W->A" and "WA->L" use cached thermodynamic variables for
   * efficiency.
   *
   * To ensure that the cache is up-to-date, the following order of calls should
   * be followed:
   *
   * If "W->A" is needed, it should be preceded immediately by "W->U" or "W->I".
   * if "WA->L" is needed, it should be preceded mmediately by "W->A".
   *
   * Any steps in between these calls may invalidate the cache.
   */
  torch::Tensor compute(std::string ab,
                        std::vector<torch::Tensor> const& args) override;

 private:
  //! \brief Convert primitive variables to conserved variables.
  /*
   * \param[in] prim  primitive variables
   * \param[out] out  conserved variables
   */
  void _prim2cons(torch::Tensor prim, torch::Tensor& out);

  //! \brief Convert conserved variables to primitive variables.
  /*
   * \param[in] cons  conserved variables
   * \param[ou] out   primitive variables
   */
  void _cons2prim(torch::Tensor cons, torch::Tensor& out);

  //! \brief calculate internal energy
  /*
   * \param[in] prim  primitive variables
   * \param[out] out  internal energy
   */
  torch::Tensor _prim2intEng(torch::Tensor prim);

  //! \brief calculate temperature.
  /*
   * \param[in] prim  primitive variables
   * \return          temperature
   */
  torch::Tensor _prim2temp(torch::Tensor prim);

  //! \brief calculate species energy (internal + kinetic).
  /*
   * \param[in] prim  primitive variables
   * \return          individual species energy
   */
  torch::Tensor _prim2speciesEng(torch::Tensor prim);

  //! \brief Convert conserved variables to kinetic energy.
  /*
   * \param[in] cons    conserved variables
   * \return            kinetic energy
   */
  torch::Tensor _cons2ke(torch::Tensor cons);

  //! \brief Convert temperature to kinetic energy.
  /*
   * \param[in] cons    conserved variables
   * \param[in] temp    temperature
   * \return            internal energy
   */
  torch::Tensor _temp2intEng(torch::Tensor cons, torch::Tensor temp);

  //! \brief Compute the adiabatic index
  /*
   * \param[in] ivol  inverse specific volume
   * \param[in] temp  temperature
   * \return          adiabatic index
   */
  torch::Tensor _adiabatic_index(torch::Tensor ivol, torch::Tensor temp);

  //! \brief Compute the isothermal sound speed
  /*
   * \param[in] temp  temperature
   * \param[in] ivol  inverse specific volume
   * \param[in] dens  total density
   * \return          isothermal sound speed
   */
  torch::Tensor _isothermal_sound_speed(torch::Tensor ivol, torch::Tensor temp,
                                        torch::Tensor dens);

  //! \brief Check/update cached thermodynamics for a primitive tensor.
  bool _cache_matches(torch::Tensor const& prim) const;
  void _mark_cache(torch::Tensor const& prim);
  void _ensure_cache(torch::Tensor const& prim);

  // A shallow tensor reference plus its ATen mutation version replaces the
  // former full-field cache copy and device reduction.
  torch::Tensor cached_prim_;
  int64_t cached_prim_version_ = -1;
};
TORCH_MODULE(MoistMixture);

}  // namespace snap
