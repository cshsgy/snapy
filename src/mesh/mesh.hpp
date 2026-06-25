#pragma once

// C/C++
#include <functional>
#include <memory>
#include <vector>

// torch
#include <torch/nn/cloneable.h>

// snap
#include <snap/mesh/meshblock.hpp>

// arg
#include <snap/add_arg.h>

namespace snap {

using MeshVariables = std::vector<Variables>;

struct MeshOptionsImpl {
  static std::shared_ptr<MeshOptionsImpl> create() {
    return std::make_shared<MeshOptionsImpl>();
  }
  static std::shared_ptr<MeshOptionsImpl> from_yaml(std::string input_file,
                                                    bool verbose = false);

  void report(std::ostream& os) const {
    os << "-- mesh options --\n";
    os << "* blocks_per_process = " << blocks_per_process() << "\n";
  }

  std::string device_str() const {
    return block() ? block()->device_str() : "";
  }

  ADD_ARG(MeshBlockOptions, block) = nullptr;
  ADD_ARG(int, blocks_per_process) = 1;
};
using MeshOptions = std::shared_ptr<MeshOptionsImpl>;

class MeshImpl : public torch::nn::Cloneable<MeshImpl> {
 public:
  MeshOptions options;
  std::vector<MeshBlock> blocks;

  MeshImpl() : options(MeshOptionsImpl::create()) {}
  explicit MeshImpl(MeshOptions const& options_);
  ~MeshImpl() override;
  void reset() override;

  double initialize(MeshVariables& vars, char const* restart_file = nullptr);
  double max_time_step(MeshVariables const& vars);
  void forward(MeshVariables& vars, double dt, int stage);
  void exchange(MeshVariables& vars, SyncOptions const& opts);
  void exchange_ghost_zones(MeshVariables& vars, int type = kConserved);
  void make_outputs(MeshVariables const& vars, double current_time,
                    bool final_write = false);
  void print_cycle_info(MeshVariables const& vars, double time,
                        double dt) const;
  int check_redo(MeshVariables& vars);
  void set_cycle(int cycle);
  void finalize(MeshVariables const& vars, double time);

 private:
  class BlockWorkerPool;

  void run_block_jobs(std::function<void(size_t)> func);

  std::shared_ptr<BlockWorkerPool> workers_;
};

TORCH_MODULE(Mesh);

}  // namespace snap

#undef ADD_ARG
