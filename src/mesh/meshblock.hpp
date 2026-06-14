#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>

// harp
#include <harp/integrator/integrator.hpp>

// snap
#include <snap/bc/bc_func.hpp>
#include <snap/bc/internal_boundary.hpp>
#include <snap/coord/coordinate.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/layout/layout.hpp>
#include <snap/scalar/scalar.hpp>

// arg
#include <snap/add_arg.h>

namespace snap {

struct OutputOptionsImpl;
using OutputOptions = std::shared_ptr<OutputOptionsImpl>;

//! \brief  container for parameters to initialize a MeshBlock
/*!
 * This struct holds all the options required to initialize a MeshBlock.
 * It can be initialized from a YAML input file using the `from_yaml` method,
 * or by setting the individual options manually.
 */
struct MeshBlockOptionsImpl {
  static std::shared_ptr<MeshBlockOptionsImpl> create() {
    return std::make_shared<MeshBlockOptionsImpl>();
  }
  static std::shared_ptr<MeshBlockOptionsImpl> from_yaml(std::string input_file,
                                                         bool verbose = false);

  MeshBlockOptionsImpl() = default;
  void report(std::ostream& os) const {
    os << "-- meshblock options --\n";
    os << "* verbose = " << (verbose() ? "true" : "false") << "\n"
       << "* basename = " << basename() << "\n";
  }

  bool is_physical_boundary(int dy, int dx, int dz) const;

  std::string device_str() const;

  //! verbose
  ADD_ARG(bool, verbose) = false;

  //! output
  ADD_ARG(std::string, basename) = "";
  ADD_ARG(std::string, output_dir) = ".";
  ADD_ARG(std::vector<OutputOptions>, outputs);

  //! submodule options
  ADD_ARG(harp::IntegratorOptions, intg) = nullptr;
  ADD_ARG(CoordinateOptions, coord) = nullptr;
  ADD_ARG(HydroOptions, hydro) = nullptr;
  ADD_ARG(ScalarOptions, scalar) = nullptr;
  ADD_ARG(InternalBoundaryOptions, ib) = nullptr;

  //! boundary functions
  ADD_ARG(std::vector<bcfunc_t>, bfuncs);

  //! distribution layout
  ADD_ARG(LayoutOptions, layout) = nullptr;
};
using MeshBlockOptions = std::shared_ptr<MeshBlockOptionsImpl>;

using Variables = std::map<std::string, torch::Tensor>;
class OutputType;

struct PartOptions {
  //! if true, return the exterior part (with ghost zones);
  //! if false, return the interior part (without ghost zones)
  ADD_ARG(bool, exterior) = true;
  ADD_ARG(int, extend_x1) = 0;
  ADD_ARG(int, extend_x2) = 0;
  ADD_ARG(int, extend_x3) = 0;
  ADD_ARG(int, depth) = 99;
  ADD_ARG(int, ndim) = 4;
};

class MeshBlockImpl : public torch::nn::Cloneable<MeshBlockImpl> {
 public:
  //! options with which this `MeshBlock` was constructed
  MeshBlockOptions options;

  //! user output
  std::function<Variables(Variables const&)> user_output_callback;

  //! user forcing increments applied during advance_local
  std::function<Variables(Variables const&, double, int)> user_forcing_callback;

  //! outputs
  std::vector<std::shared_ptr<OutputType>> output_types;

  //! current cycle number
  int cycle = 0;

  //! Exchange buffers owned by the MeshBlock so Layout stays stateless.
  mutable std::vector<std::vector<torch::Tensor>> send_bufs, recv_bufs;

  //! submodules
  harp::Integrator pintg = nullptr;
  Coordinate pcoord = nullptr;
  InternalBoundary pib = nullptr;
  Hydro phydro = nullptr;
  Scalar pscalar = nullptr;

  Layout get_layout() const { return _playout; }

  //! Constructor to initialize the layers
  MeshBlockImpl() : options(MeshBlockOptionsImpl::create()) {}
  explicit MeshBlockImpl(MeshBlockOptions const& options_);
  ~MeshBlockImpl() override;
  void reset() override;

  //! \brief return an index tensor for part of the meshblock
  /*!
   * \param offset: tuple of (x1_offset, x2_offset, x3_offset)
   * \param opts: additional options
   * \return: vector of TensorIndex for each dimension
   */
  std::vector<torch::indexing::TensorIndex> part(
      std::tuple<int, int, int> offset, PartOptions const& opts) const;

  //! initialize the variables
  /*!
   * \param vars: variables to initialize
   * \return: initial simulation time
   */
  double initialize(Variables& vars, char const* restart_file = nullptr);
  void initialize_local(Variables& vars);

  //! Mesh-owned initialization path that preserves multi-block exchange order.
  void initialize_under_mesh(Variables& vars);
  void finalize_initialization(Variables& vars);

  //! compute the maximum allowable time step
  /*!
   * \param vars: current variables
   * \return: maximum time step
   */
  double max_time_step(Variables const& vars);
  double local_max_time_step(Variables const& vars) const;

  //! advance the variables by one time step
  /*!
   * \param vars: current variables
   * \param dt: time step
   * \param stage: current stage of the integrator
   */
  void forward(Variables& vars, double dt, int stage);
  void advance_local(Variables& vars, double dt, int stage);
  void exchange(Variables& vars, SyncOptions const& opts) const;

  //! Serialize state into MeshBlock-owned buffers without launching comms yet.
  void begin_exchange(Variables& vars, SyncOptions const& opts) const;

  //! Launch remote exchange work after begin_exchange prepared the buffers.
  void launch_exchange(SyncOptions const& opts,
                       std::vector<CommWorkPtr>& works) const;

  //! Wait on launched work and deserialize results back into the variables.
  void finalize_exchange(Variables& vars, SyncOptions const& opts,
                         std::vector<CommWorkPtr>& works) const;
  void exchange_ghost_zones(Variables& vars);

  //! make write outputs at the current time
  /*!
   * \param vars: current variables
   * \param current_time: current simulation time
   * \param final_write: if true, writing outputs as 'final' outputs
   */
  void make_outputs(Variables const& vars, double current_time,
                    bool final_write = false);

  //! print cycle info
  /*!
   * \param vars: current variables
   * \param time: current simulation time
   * \param dt: current time step
   */
  void print_cycle_info(Variables const& vars, double time, double dt) const;

  //! make final output and print diagnostics
  void finalize(Variables const& vars, double time);

  //! check if redo is needed
  /*!
   * \param vars: current variables
   * \return: > 0, redo is needed; 0, no redo; < 0, terminate simulation
   */
  int check_redo(Variables& vars);

 protected:
  //! initialize from restart file
  /*!
   * \param vars: variables to initialize
   * \param fname: restart filename
   * \return: last simulation time from the restart file
   */
  double _init_from_restart(Variables& vars, std::string fname);

 private:
  //! clock and cycle at time start
  clock_t _time_start;
  int _cycle_start = 0;

  //! distribution layout
  Layout _playout;

  //! stage registers
  torch::Tensor _hydro_u0, _hydro_u1;
  torch::Tensor _scalar_s0, _scalar_s1;
};

TORCH_MODULE(MeshBlock);
}  // namespace snap

#undef ADD_ARG
