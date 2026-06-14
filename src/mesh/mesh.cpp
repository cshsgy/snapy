// C/C++
#include <c10/core/DeviceGuard.h>

#include <algorithm>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>

// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/mesh/mesh.hpp>
#include <snap/utils/log.hpp>
#include <snap/utils/signal_handler.hpp>

namespace snap {

namespace {
MeshBlockOptions clone_block_options(MeshBlockOptions const& src) {
  auto dst = std::make_shared<MeshBlockOptionsImpl>(*src);
  auto layout = std::make_shared<LayoutOptionsImpl>(*src->layout());
  dst->layout(layout);
  if (src->coord() != nullptr) {
    dst->coord(src->coord()->clone());
  }
  return dst;
}

template <typename Func>
void run_block_jobs(size_t count, torch::Device const& device, Func&& func) {
  std::vector<std::future<void>> jobs;
  jobs.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    jobs.push_back(std::async(std::launch::async, [&, i, device]() {
      c10::OptionalDeviceGuard device_guard(device);
      func(i);
    }));
  }
  for (auto& job : jobs) {
    job.get();
  }
}
}  // namespace

MeshOptions MeshOptionsImpl::from_yaml(std::string input_file, bool verbose) {
  auto block = MeshBlockOptionsImpl::from_yaml(input_file, verbose);
  auto options = MeshOptionsImpl::create();
  options->block(block);

  auto config = YAML::LoadFile(input_file);
  options->blocks_per_process(
      config["distribute"]["blocks_per_process"].as<int>(1));
  return options;
}

MeshImpl::MeshImpl(MeshOptions const& options_) : options(options_) { reset(); }

void MeshImpl::reset() {
  blocks.clear();
  TORCH_CHECK(options->block() != nullptr, "Mesh requires MeshBlockOptions");

  auto base = options->block();
  auto base_layout = base->layout();
  TORCH_CHECK(base_layout != nullptr, "Mesh requires LayoutOptions");
  TORCH_CHECK(options->blocks_per_process() > 0,
              "blocks_per_process must be positive");

  base_layout->blocks_per_process(options->blocks_per_process());
  base_layout->world_size(base_layout->process_world_size() *
                          options->blocks_per_process());

  for (int i = 0; i < options->blocks_per_process(); ++i) {
    auto block_opts = clone_block_options(base);
    auto layout = block_opts->layout();
    layout->blocks_per_process(options->blocks_per_process());
    layout->world_size(layout->process_world_size() *
                       options->blocks_per_process());
    layout->rank(layout->global_block_rank(layout->process_rank(), i));
    if (block_opts->coord() != nullptr) {
      block_opts->coord()->repartition(layout);
    }
    auto block =
        register_module("block" + std::to_string(i), MeshBlock(block_opts));
    blocks.push_back(block);
  }
}

double MeshImpl::initialize(MeshVariables& vars, char const* restart_file) {
  TORCH_CHECK(vars.size() == blocks.size(),
              "Mesh::initialize expects one Variables map per local MeshBlock");

  if (restart_file) {
    double current_time = 0.;
    for (int i = 0; i < blocks.size(); ++i) {
      auto block_time = blocks[i]->initialize(vars[i], restart_file);
      if (i == 0) {
        current_time = block_time;
      }
    }
    return current_time;
  }

  auto layout = blocks.front()->get_layout();
  if (layout->has_process_group()) {
    layout->comm->barrier();
  }
  SignalHandler::GetInstance();

  auto device = torch::Device(blocks.front()->options->device_str());
  std::vector<std::future<void>> jobs;
  jobs.reserve(blocks.size());
  for (int i = 0; i < blocks.size(); ++i) {
    jobs.push_back(std::async(std::launch::async, [&, i, device]() {
      c10::OptionalDeviceGuard device_guard(device);
      blocks[i]->initialize_under_mesh(vars[i]);
    }));
  }
  for (auto& job : jobs) {
    job.get();
  }

  return 0.;
}

double MeshImpl::max_time_step(MeshVariables const& vars) {
  TORCH_CHECK(
      vars.size() == blocks.size(),
      "Mesh::max_time_step expects one Variables map per local MeshBlock");

  double dt_local = 1.e99;
  for (int i = 0; i < blocks.size(); ++i) {
    dt_local = std::min(dt_local, blocks[i]->local_max_time_step(vars[i]));
  }

  auto dt_tensor = torch::tensor({dt_local}, torch::dtype(torch::kFloat64));
  std::vector<at::Tensor> dt_reduce = {dt_tensor};
  c10d::AllreduceOptions op;
  op.reduceOp = c10d::ReduceOp::MIN;
  auto layout = blocks.front()->get_layout();
  if (layout->has_process_group()) {
    layout->comm->allreduce(dt_reduce, op.reduceOp);
  }

  auto dt = dt_reduce[0].item<double>();
  auto redo = blocks.front()->pintg->current_redo;
  auto cfl = blocks.front()->pintg->options->cfl();
  return pow(2., -redo) * cfl * dt;
}

void MeshImpl::forward(MeshVariables& vars, double dt, int stage) {
  TORCH_CHECK(vars.size() == blocks.size(),
              "Mesh::forward expects one Variables map per local MeshBlock");

  if (blocks.size() == 1) {
    blocks[0]->advance_local(vars[0], dt, stage);
    blocks[0]->exchange_ghost_zones(vars[0]);
    return;
  }

  auto device = torch::Device(blocks.front()->options->device_str());
  run_block_jobs(blocks.size(), device, [&](size_t i) {
    blocks[i]->advance_local(vars[i], dt, stage);
  });
  run_block_jobs(blocks.size(), device,
                 [&](size_t i) { blocks[i]->exchange_ghost_zones(vars[i]); });
}

void MeshImpl::exchange_ghost_zones(MeshVariables& vars, int type) {
  TORCH_CHECK(vars.size() == blocks.size(),
              "Mesh::exchange_ghost_zones expects one Variables map per local "
              "MeshBlock");
  TORCH_CHECK(type == kPrimitive || type == kConserved || type == kScalar,
              "Mesh::exchange_ghost_zones received invalid variable type");

  SyncOptions opts;
  opts.interpolate(true).type(type);
  if (blocks.size() == 1) {
    blocks[0]->exchange(vars[0], opts);
    return;
  }

  auto device = torch::Device(blocks.front()->options->device_str());
  run_block_jobs(blocks.size(), device,
                 [&](size_t i) { blocks[i]->exchange(vars[i], opts); });
}

void MeshImpl::make_outputs(MeshVariables const& vars, double current_time,
                            bool final_write) {
  TORCH_CHECK(
      vars.size() == blocks.size(),
      "Mesh::make_outputs expects one Variables map per local MeshBlock");

  for (int i = 0; i < blocks.size(); ++i) {
    blocks[i]->make_outputs(vars[i], current_time, final_write);
  }
}

void MeshImpl::print_cycle_info(MeshVariables const& vars, double time,
                                double dt) const {
  TORCH_CHECK(vars.size() == blocks.size(),
              "Mesh::print_cycle_info expects one Variables map per local "
              "MeshBlock");

  auto root = blocks.front();
  auto pintg = root->pintg;
  if (pintg->options->ncycle_out() == 0 ||
      root->cycle % pintg->options->ncycle_out() != 0) {
    return;
  }

  const int dt_precision = std::numeric_limits<double>::max_digits10 - 3;
  bool compute_mass = false;
  bool compute_energy = false;

  if (vars.front().count("hydro_u")) {
    compute_mass = true;
    compute_energy = root->phydro->peos->nvar() > IPR;
  }

  SINFO() << "cycle=" << root->cycle << " redo=" << pintg->current_redo
          << std::scientific << std::setprecision(dt_precision)
          << " time=" << time << " dt=" << dt;

  c10d::ReduceOptions opsum;
  opsum.reduceOp = c10d::ReduceOp::SUM;
  opsum.rootRank = root->options->layout()->process_root_rank();

  torch::Tensor local_sum;
  if (compute_mass || compute_energy) {
    for (int i = 0; i < blocks.size(); ++i) {
      auto interior = blocks[i]->part({0, 0, 0}, PartOptions().exterior(false));
      auto vol = blocks[i]->pcoord->cell_volume();
      auto hydro_u_tot = vars[i].at("hydro_u") * vol;
      auto block_sum = hydro_u_tot.index(interior).sum({1, 2, 3});
      if (!local_sum.defined()) {
        local_sum = block_sum.clone();
      } else {
        local_sum += block_sum;
      }
    }
  }

  if (local_sum.defined()) {
    std::vector<at::Tensor> sum = {local_sum};
    if (root->get_layout()->has_process_group()) {
      root->get_layout()->comm->reduce(sum, opsum.reduceOp, opsum.rootRank);
    }

    if (compute_mass) {
      auto mass = sum[0][IDN];
      SINFO() << std::scientific << std::setprecision(dt_precision)
              << " mass0=" << mass.item<double>();

      int ny = local_sum.size(0) - 5;
      if (ny > 0) {
        for (int n = 0; n < ny; ++n) {
          mass += sum[0][ICY + n];
        }
        SINFO() << std::scientific << std::setprecision(dt_precision)
                << " masst=" << mass.item<double>();
      }
    }

    if (compute_energy) {
      SINFO() << std::scientific << std::setprecision(dt_precision)
              << " energy=" << sum[0][IPR].item<double>();
    }
  }

  SINFO() << std::endl;
}

int MeshImpl::check_redo(MeshVariables& vars) {
  TORCH_CHECK(vars.size() == blocks.size(),
              "Mesh::check_redo expects one Variables map per local MeshBlock");

  auto sig = snap::SignalHandler::GetInstance();
  if (!blocks.empty() && sig->CheckSignalFlags(blocks.front().get())) {
    return -1;
  }

  int redo = 0;
  for (int i = 0; i < blocks.size(); ++i) {
    int err = blocks[i]->check_redo(vars[i]);
    if (err < 0) return -1;
    redo = std::max(redo, err);
  }
  return redo;
}

void MeshImpl::set_cycle(int cycle) {
  for (auto& block : blocks) {
    block->cycle = cycle;
  }
}

void MeshImpl::finalize(MeshVariables const& vars, double time) {
  TORCH_CHECK(vars.size() == blocks.size(),
              "Mesh::finalize expects one Variables map per local MeshBlock");

  if (blocks.size() == 1) {
    blocks.front()->finalize(vars.front(), time);
    return;
  }

  make_outputs(vars, time, /*final_write=*/true);

  auto root = blocks.front();
  auto sig = SignalHandler::GetInstance();
  if (sig->GetSignalFlag(SIGTERM) != 0) {
    SINFO() << std::endl << "Terminating on Terminate signal" << std::endl;
  } else if (sig->GetSignalFlag(SIGINT) != 0) {
    SINFO() << std::endl << "Terminating on Interrupt signal" << std::endl;
  } else if (sig->GetSignalFlag(SIGALRM) != 0) {
    SINFO() << std::endl << "Terminating on wall-time limit" << std::endl;
  } else if (root->pintg->options->nlim() >= 0 &&
             root->cycle >= root->pintg->options->nlim()) {
    SINFO() << std::endl << "Terminating on cycle limit" << std::endl;
  } else if (time >= root->pintg->options->tlim()) {
    SINFO() << std::endl << "Terminating on time limit" << std::endl;
  } else {
    SINFO() << std::endl << "Terminating abnormally" << std::endl;
  }

  SINFO() << "time=" << time << " cycle=" << root->cycle << std::endl;
  SINFO() << "tlim=" << root->pintg->options->tlim()
          << " nlim=" << root->pintg->options->nlim() << std::endl;

  for (auto& block : blocks) {
    block->send_bufs.clear();
    block->send_bufs.shrink_to_fit();
    block->recv_bufs.clear();
    block->recv_bufs.shrink_to_fit();
  }

  auto layout = root->get_layout();
  if (layout->has_process_group()) {
    layout->comm->barrier();
    if (layout->comm->owns_process_group()) {
      layout->comm->shutdown();
    }
  }
}

}  // namespace snap
