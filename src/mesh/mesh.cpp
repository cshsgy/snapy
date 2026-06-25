// C/C++
#include <c10/core/DeviceGuard.h>

#include <algorithm>
#include <condition_variable>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <thread>

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

}  // namespace

class MeshImpl::BlockWorkerPool {
 public:
  BlockWorkerPool(size_t count, torch::Device device)
      : device_(std::move(device)), remaining_(count) {
    workers_.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      workers_.emplace_back([this, i]() { run(i); });
    }
  }

  ~BlockWorkerPool() { stop(); }

  void submit(std::function<void(size_t)> func) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      TORCH_CHECK(!stopping_, "Mesh block worker pool is stopping");
      func_ = std::move(func);
      error_ = nullptr;
      remaining_ = workers_.size();
      generation_ += 1;
    }
    cv_.notify_all();

    std::unique_lock<std::mutex> lock(mutex_);
    done_cv_.wait(lock, [&]() { return remaining_ == 0; });
    if (error_ != nullptr) {
      auto error = error_;
      error_ = nullptr;
      std::rethrow_exception(error);
    }
  }

  void stop() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (stopping_) return;
      stopping_ = true;
    }
    cv_.notify_all();
    for (auto& worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  }

 private:
  void run(size_t index) {
    c10::OptionalDeviceGuard device_guard(device_);
    size_t seen_generation = 0;
    while (true) {
      std::function<void(size_t)> func;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock,
                 [&]() { return stopping_ || generation_ != seen_generation; });
        if (stopping_) return;
        seen_generation = generation_;
        func = func_;
      }

      try {
        func(index);
      } catch (...) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (error_ == nullptr) {
          error_ = std::current_exception();
        }
      }

      {
        std::lock_guard<std::mutex> lock(mutex_);
        remaining_ -= 1;
        if (remaining_ == 0) {
          done_cv_.notify_one();
        }
      }
    }
  }

  torch::Device device_;
  std::vector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::condition_variable done_cv_;
  std::function<void(size_t)> func_;
  std::exception_ptr error_;
  size_t generation_ = 0;
  size_t remaining_ = 0;
  bool stopping_ = false;
};

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

MeshImpl::~MeshImpl() = default;

void MeshImpl::reset() {
  workers_.reset();
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

  if (blocks.size() > 1) {
    workers_ = std::make_shared<BlockWorkerPool>(
        blocks.size(), torch::Device(blocks.front()->options->device_str()));
  }
}

void MeshImpl::run_block_jobs(std::function<void(size_t)> func) {
  if (blocks.size() <= 1) {
    if (!blocks.empty()) {
      c10::OptionalDeviceGuard device_guard(
          torch::Device(blocks.front()->options->device_str()));
      func(0);
    }
    return;
  }

  TORCH_CHECK(workers_ != nullptr,
              "Mesh block worker pool is not initialized for multi-block mesh");
  workers_->submit(std::move(func));
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

  run_block_jobs([&](size_t i) { blocks[i]->initialize_under_mesh(vars[i]); });

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

  run_block_jobs(
      [&](size_t i) { blocks[i]->advance_local(vars[i], dt, stage); });
  run_block_jobs([&](size_t i) { blocks[i]->exchange_ghost_zones(vars[i]); });
}

void MeshImpl::exchange(MeshVariables& vars, SyncOptions const& opts) {
  TORCH_CHECK(vars.size() == blocks.size(),
              "Mesh::exchange expects one Variables map per local MeshBlock");

  if (blocks.size() == 1) {
    blocks[0]->exchange(vars[0], opts);
    return;
  }

  run_block_jobs([&](size_t i) { blocks[i]->exchange(vars[i], opts); });
}

void MeshImpl::exchange_ghost_zones(MeshVariables& vars, int type) {
  TORCH_CHECK(type == kPrimitive || type == kConserved || type == kScalar,
              "Mesh::exchange_ghost_zones received invalid variable type");

  SyncOptions opts;
  opts.interpolate(true).type(type);
  exchange(vars, opts);
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
