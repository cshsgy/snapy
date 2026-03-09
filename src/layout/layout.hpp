#pragma once

// C/C++
#include <iostream>
#include <memory>
#include <tuple>

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>

#include <torch/csrc/distributed/c10d/Store.hpp>
// #include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Backend.hpp>

// snap
#include <snap/snap.h>

#include "connectivity.hpp"

// arg
#include <snap/add_arg.h>

namespace snap {

/*!
 * \brief Calculate buffer ID from directional offsets
 *
 * Converts 3D directional offsets into a linear buffer index.
 * For 2D layouts (dz=0), returns index in range [0,8].
 * For 3D layouts, returns index in range [0,26].
 *
 * \return linear buffer index
 */
inline int get_buffer_id(std::tuple<int, int, int> offset) {
  auto [dy, dx, dz] = offset;
  return (dx % 3 + 3) % 3 + ((dy % 3 + 3) % 3) * 3 + ((dz % 3 + 3) % 3) * 9;
}

//! get environment variable with default
inline std::string get_env(const char *name, const std::string &def) {
  const char *v = std::getenv(name);
  return v ? std::string(v) : def;
}

//! get global rank from environment variable
inline int get_rank() { return std::stoi(get_env("RANK", "0")); }

//! get local rank from environment variable
inline int get_local_rank() { return std::stoi(get_env("LOCAL_RANK", "0")); }

struct LayoutOptionsImpl {
  static std::shared_ptr<LayoutOptionsImpl> create() {
    return std::make_shared<LayoutOptionsImpl>();
  }
  static std::shared_ptr<LayoutOptionsImpl> from_yaml(
      std::string const &filename, bool verbose = false);

  LayoutOptionsImpl();

  void report(std::ostream &os) const {
    os << "-- layout options --\n";
    os << "* type = " << type() << "\n"
       << "* px = " << px() << "\n"
       << "* py = " << py() << "\n"
       << "* pz = " << pz() << "\n"
       << "* periodic_x = " << (periodic_x() ? "true" : "false") << "\n"
       << "* periodic_y = " << (periodic_y() ? "true" : "false") << "\n"
       << "* periodic_z = " << (periodic_z() ? "true" : "false") << "\n"
       << "* backend = " << backend() << "\n"
       << "* master_addr = " << master_addr() << "\n"
       << "* rank = " << rank() << "\n"
       << "* local_rank = " << local_rank() << "\n"
       << "* world_size = " << world_size() << "\n"
       << "* master_port = " << master_port() << "\n"
       << "* no_backend = " << (no_backend() ? "true" : "false") << "\n"
       << "* verbose = " << (verbose() ? "true" : "false") << "\n";
  }

  //! type of layout
  ADD_ARG(std::string, type) = "slab";

  //! number of processors in X
  ADD_ARG(int, px) = 1;

  //! number of processors in Y
  ADD_ARG(int, py) = 1;

  //! number of processors in Z
  ADD_ARG(int, pz) = 1;

  //! periodicity in X
  ADD_ARG(bool, periodic_x) = false;

  //! periodicity in Y
  ADD_ARG(bool, periodic_y) = false;

  //! periodicity in Z
  ADD_ARG(bool, periodic_z) = false;

  ADD_ARG(std::string, backend) = "gloo";
  ADD_ARG(std::string, master_addr) = "127.0.0.1";
  ADD_ARG(int, rank) = 0;
  ADD_ARG(int, root_rank) = 0;
  ADD_ARG(int, local_rank) = 0;
  ADD_ARG(int, world_size) = 1;
  ADD_ARG(int, master_port) = 29501;
  ADD_ARG(int, device_id) = -1;
  ADD_ARG(bool, verbose) = false;
  ADD_ARG(bool, no_backend) = false;
};
using LayoutOptions = std::shared_ptr<LayoutOptionsImpl>;

//! extra options for synchronization
struct SyncOptions {
  enum { DIM1 = 3, DIM2 = 2, DIM3 = 1 };

  int dz_min() const { return dim() == DIM2 || dim() == DIM3 ? 0 : -1; }
  int dz_max() const { return dim() == DIM2 || dim() == DIM3 ? 0 : 1; }

  int dx_min() const { return dim() == DIM3 || dim() == DIM1 ? 0 : -1; }
  int dx_max() const { return dim() == DIM3 || dim() == DIM1 ? 0 : 1; }

  int dy_min() const { return dim() == DIM2 || dim() == DIM1 ? 0 : -1; }
  int dy_max() const { return dim() == DIM2 || dim() == DIM1 ? 0 : 1; }

  ADD_ARG(bool, cross_panel_only) = false;
  ADD_ARG(bool, skip_corner) = true;
  ADD_ARG(bool, interpolate) = false;
  ADD_ARG(int, type) = kConserved;
  ADD_ARG(int, dim) = 0;
  ADD_ARG(int, phyid) = 0;
};

using Variables = std::map<std::string, torch::Tensor>;

class MeshBlockImpl;

class LayoutImpl {
 public:
  static std::shared_ptr<LayoutImpl> create(LayoutOptions const &opts,
                                            torch::nn::Module *p = nullptr,
                                            std::string const &name = "layout");

  //! exchange buffers
  /*!
   * The first index indicates the rank
   * The second index indicates the variable group
   */
  std::vector<std::vector<torch::Tensor>> send_bufs, recv_bufs;

  //! submodules
  at::intrusive_ptr<c10d::Store> store;
  std::shared_ptr<c10d::Backend> pg;

  //! options with which this `Layout` was constructed
  LayoutOptions options;

  LayoutImpl() : options(LayoutOptionsImpl::create()) {}
  LayoutImpl(const LayoutOptions &opts) : options(opts) {
    int P = options->px() * options->py() * options->pz();
    _rankof.resize(P);
  }

  std::tuple<int, int, int> get_procs() const {
    return {options->px(), options->py(), options->pz()};
  }

  bool is_root() const { return options->rank() == options->root_rank(); }

  virtual ~LayoutImpl() = default;

  virtual int rank_of(std::tuple<int, int, int> iloc) const {
    auto [rx, ry, rz] = iloc;

    int px = options->px();
    int py = options->py();
    int pz = options->pz();
    if (rx < 0 || rx >= px || ry < 0 || ry >= py || rz < 0 || rz >= pz)
      return -1;
    return _rankof[rz * (px * py) + ry * px + rx];
  }

  virtual std::tuple<int, int, int> loc_of(int rank) const { return {0, 0, 0}; }

  //! \brief Neighbor -> Z-order rank (3D)
  /*!
   * offset = (dx,dy,dz) <- {-1,0,1}. periodic flags control wrap;
   * otherwise off-domain -> -1.
   * iloc = (rx,ry,rz) are THIS rank's logical coords in the process grid (not
   * Morton code).
   */
  virtual int neighbor_rank(std::tuple<int, int, int> iloc,
                            std::tuple<int, int, int> offset) const {
    return -1;
  }

  //! Serialize variables
  virtual void serialize(MeshBlockImpl const *pmb, Variables &vars,
                         SyncOptions const &opts);

  //! Deserialize variables
  virtual void deserialize(MeshBlockImpl const *pmb, Variables &vars,
                           SyncOptions const &opts) const;

  //! fill corners after exchange
  virtual void fill_corners(MeshBlockImpl const *pmb, Variables &vars) const;

  //! \brief Perform ghost zone exchange
  /*!
   * Exchanges ghost zone data with neighboring processes using point-to-point
   * communication. This function serializes data, performs send/recv
   * operations, and deserializes received data into ghost zones.
   */
  virtual void forward(MeshBlockImpl const *pmb, Variables &vars,
                       SyncOptions const &opts,
                       std::vector<c10::intrusive_ptr<c10d::Work>> &works);

  void finalize(MeshBlockImpl const *pmb, Variables &vars,
                SyncOptions const &opts,
                std::vector<c10::intrusive_ptr<c10d::Work>> &works);

 protected:
  void _init_backend();

  // --- Backend initializers ---
  void _init_gloo();
  void _init_nccl();

  // --- NCCL specific ---
  void _group_start() const;
  c10::intrusive_ptr<c10d::Work> _group_end() const;

  // --- GPU specific ---
  void _sync_device() const;

  std::vector<Coord2> _coords2;
  std::vector<Coord3> _coords3;
  std::vector<int> _rankof;
};
using Layout = std::shared_ptr<LayoutImpl>;

class SlabLayoutImpl : public torch::nn::Cloneable<SlabLayoutImpl>,
                       public LayoutImpl {
 public:
  //! Constructor to initialize the layers
  SlabLayoutImpl() = default;
  SlabLayoutImpl(const LayoutOptions &opts) : LayoutImpl(opts) {
    options->type("slab");
    reset();
  }
  void reset() override;
  using LayoutImpl::forward;

  ~SlabLayoutImpl() = default;
  void pretty_print(std::ostream &os) const override;

  std::tuple<int, int, int> loc_of(int rank) const override;
  int neighbor_rank(std::tuple<int, int, int> iloc,
                    std::tuple<int, int, int> offset) const override;
};
TORCH_MODULE(SlabLayout);

class CubedLayoutImpl : public torch::nn::Cloneable<CubedLayoutImpl>,
                        public LayoutImpl {
 public:
  //! Constructor to initialize the layers
  CubedLayoutImpl() = default;
  CubedLayoutImpl(const LayoutOptions &opts) : LayoutImpl(opts) {
    options->type("cubed");
    reset();
  }
  void reset() override;

  ~CubedLayoutImpl() = default;
  void pretty_print(std::ostream &os) const override;

  std::tuple<int, int, int> loc_of(int rank) const override;
  int neighbor_rank(std::tuple<int, int, int> iloc,
                    std::tuple<int, int, int> offset) const override;
};
TORCH_MODULE(CubedLayout);

}  // namespace snap

#undef ADD_ARG
