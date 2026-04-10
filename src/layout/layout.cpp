// yaml
#include <yaml-cpp/yaml.h>

#include <condition_variable>
#include <exception>
#include <map>

// base
#include <configure.h>  // gloo and nccl

// snap
#include <snap/mesh/meshblock.hpp>
#include <snap/utils/log.hpp>

#include "cubed_sphere_layout.hpp"
#include "layout.hpp"

namespace snap {

namespace {

struct LocalExchangeKey {
  int process_rank;
  int blocks_per_process;
  int dim;
  int phyid;
  int type;
  bool cross_panel_only;
  bool skip_corner;
  bool interpolate;
  std::string layout_type;

  bool operator<(LocalExchangeKey const& other) const {
    return std::tie(process_rank, blocks_per_process, dim, phyid, type,
                    cross_panel_only, skip_corner, interpolate, layout_type) <
           std::tie(other.process_rank, other.blocks_per_process, other.dim,
                    other.phyid, other.type, other.cross_panel_only,
                    other.skip_corner, other.interpolate, other.layout_type);
  }
};

struct LocalExchangeState {
  int generation = 0;
  int arrived = 0;
  int released = 0;
  bool ready = false;
  std::exception_ptr error;
};

struct OrderedPhaseKey {
  LocalExchangeKey exchange;
  int phase;

  bool operator<(OrderedPhaseKey const& other) const {
    return std::tie(exchange, phase) < std::tie(other.exchange, other.phase);
  }
};

struct OrderedPhaseState {
  int next_index = 0;
};

struct NCCLLaunchState {
  int generation = 0;
  int arrived = 0;
  int released = 0;
  bool ready = false;
  std::exception_ptr error;
  std::map<int, std::vector<c10::intrusive_ptr<c10d::Work>>> works_by_block;
};

struct RemoteExchangeOp {
  LayoutImpl* layout;
  int remote_process;
  int local_block;
  int remote_local_block;
  int buffer_id;
  int send_tag;
  int recv_tag;
  std::tuple<int, int, int> offset;
  std::tuple<int, int, int> peer_offset;
};

std::mutex g_local_exchange_mutex;
std::condition_variable g_local_exchange_cv;
std::map<std::pair<int, int>, LayoutImpl*> g_local_layouts;
std::map<LocalExchangeKey, LocalExchangeState> g_local_exchange_states;
std::map<LocalExchangeKey, NCCLLaunchState> g_nccl_launch_states;
std::map<OrderedPhaseKey, OrderedPhaseState> g_ordered_phase_states;
std::mutex g_process_comm_mutex;

std::vector<LayoutImpl*> local_layouts_for(LayoutImpl const& layout) {
  std::vector<LayoutImpl*> layouts(layout.options->blocks_per_process(),
                                   nullptr);

  for (auto const& [key, local_layout] : g_local_layouts) {
    if (key.first == layout.options->process_rank()) {
      TORCH_CHECK(
          key.second >= 0 && key.second < layout.options->blocks_per_process(),
          "invalid local block index ", key.second,
          " for process-local layout registry");
      layouts[key.second] = local_layout;
    }
  }

  for (int i = 0; i < layouts.size(); ++i) {
    TORCH_CHECK(layouts[i] != nullptr,
                "missing process-local layout registration for process ",
                layout.options->process_rank(), " local block ", i);
  }

  return layouts;
}

LocalExchangeKey make_local_exchange_key(LayoutImpl const& layout,
                                         SyncOptions const& opts) {
  return LocalExchangeKey{
      layout.options->process_rank(),
      layout.options->blocks_per_process(),
      opts.dim(),
      opts.phyid(),
      opts.type(),
      opts.cross_panel_only(),
      opts.skip_corner(),
      opts.interpolate(),
      layout.options->type(),
  };
}

OrderedPhaseKey make_ordered_phase_key(LayoutImpl const& layout,
                                       SyncOptions const& opts, int phase) {
  return OrderedPhaseKey{make_local_exchange_key(layout, opts), phase};
}

int exchange_order_index(SyncOptions const& opts,
                         std::tuple<int, int, int> offset) {
  int order = 0;
  for (int dy = opts.dy_min(); dy <= opts.dy_max(); ++dy) {
    for (int dx = opts.dx_min(); dx <= opts.dx_max(); ++dx) {
      if (dy == 0 && dx == 0) continue;
      if (opts.skip_corner() && std::abs(dy) + std::abs(dx) == 2) continue;
      if (offset == std::tuple<int, int, int>(dy, dx, 0)) {
        return order;
      }
      order += 1;
    }
  }
  return order;
}

auto make_remote_order_key(int process_rank, int remote_process,
                           int local_block, int remote_local_block,
                           std::tuple<int, int, int> offset,
                           std::tuple<int, int, int> peer_offset,
                           SyncOptions const& opts) {
  int lower_process = std::min(process_rank, remote_process);
  int upper_process = std::max(process_rank, remote_process);
  int lower_block =
      process_rank < remote_process ? local_block : remote_local_block;
  int upper_block =
      process_rank < remote_process ? remote_local_block : local_block;
  auto lower_to_upper_offset =
      process_rank < remote_process ? offset : peer_offset;
  return std::make_tuple(lower_process, upper_process, lower_block, upper_block,
                         exchange_order_index(opts, lower_to_upper_offset),
                         opts.phyid());
}

}  // namespace

std::vector<int> LayoutImpl::_active_remote_local_indices(
    SyncOptions const& opts) const {
  std::vector<
      std::pair<std::vector<std::tuple<int, int, int, int, int, int>>, int>>
      keyed_indices;
  auto layouts = local_layouts_for(*this);

  for (auto* local_layout : layouts) {
    auto keys = local_layout->_remote_order_keys(opts);
    if (keys.empty()) {
      continue;
    }

    keyed_indices.push_back(
        {std::move(keys), local_layout->options->local_block_index(
                              local_layout->options->rank())});
  }

  std::sort(
      keyed_indices.begin(), keyed_indices.end(),
      [](auto const& lhs, auto const& rhs) { return lhs.first < rhs.first; });

  std::vector<int> indices;
  indices.reserve(keyed_indices.size());
  for (auto const& [key, index] : keyed_indices) {
    (void)key;
    indices.push_back(index);
  }
  return indices;
}

std::vector<std::tuple<int, int, int, int, int, int>>
LayoutImpl::_remote_order_keys(SyncOptions const& opts) const {
  auto rank = options->rank();
  auto iloc = loc_of(rank);
  int local_block = options->local_block_index(rank);
  std::vector<std::tuple<int, int, int, int, int, int>> keys;

  for (int dy_ = opts.dy_min(); dy_ <= opts.dy_max(); ++dy_) {
    for (int dx_ = opts.dx_min(); dx_ <= opts.dx_max(); ++dx_) {
      if (dy_ == 0 && dx_ == 0) continue;
      if (opts.skip_corner() && std::abs(dy_) + std::abs(dx_) == 2) continue;

      auto offset = _remap_exchange_offset(iloc, dy_, dx_);
      int nb = neighbor_rank(iloc, offset);
      if (nb < 0 || nb == rank) continue;

      int remote_process = options->owner_process_rank(nb);
      if (remote_process == options->process_rank()) continue;

      int remote_local_block = options->local_block_index(nb);
      auto peer_offset = _peer_exchange_offset(nb, rank, opts, offset);
      keys.push_back(make_remote_order_key(
          options->process_rank(), remote_process, local_block,
          remote_local_block, offset, peer_offset, opts));
    }
  }

  std::sort(keys.begin(), keys.end());
  return keys;
}

void LayoutImpl::_launch_cubed_sphere_nccl_remote_ops(
    SyncOptions const& opts,
    std::map<int, std::vector<c10::intrusive_ptr<c10d::Work>>>& works_by_block)
    const {
  auto layouts = local_layouts_for(*this);
  std::vector<RemoteExchangeOp> remote_ops;

  for (auto* layout : layouts) {
    auto rank = layout->options->rank();
    auto iloc = layout->loc_of(rank);

    for (int dy = opts.dy_min(); dy <= opts.dy_max(); ++dy) {
      for (int dx = opts.dx_min(); dx <= opts.dx_max(); ++dx) {
        if (dy == 0 && dx == 0) continue;
        if (opts.skip_corner() && std::abs(dy) + std::abs(dx) == 2) continue;

        std::tuple<int, int, int> offset(dy, dx, 0);
        int nb = layout->neighbor_rank(iloc, offset);
        if (nb < 0) continue;

        int remote_process = layout->options->owner_process_rank(nb);
        if (remote_process == layout->options->process_rank()) continue;

        int local_block = layout->options->local_block_index(rank);
        int remote_local_block = layout->options->local_block_index(nb);
        auto peer_offset =
            layout->_peer_exchange_offset(nb, rank, opts, offset);
        remote_ops.push_back(
            {layout, remote_process, local_block, remote_local_block,
             get_buffer_id(offset),
             make_comm_tag(remote_local_block, peer_offset, opts.phyid()),
             make_comm_tag(local_block, offset, opts.phyid()), offset,
             peer_offset});
      }
    }
  }

  auto recv_ops = remote_ops;
  std::sort(recv_ops.begin(), recv_ops.end(),
            [&](RemoteExchangeOp const& lhs, RemoteExchangeOp const& rhs) {
              return std::make_tuple(lhs.remote_process, lhs.local_block,
                                     exchange_order_index(opts, lhs.offset)) <
                     std::make_tuple(rhs.remote_process, rhs.local_block,
                                     exchange_order_index(opts, rhs.offset));
            });

  auto send_ops = remote_ops;
  std::sort(
      send_ops.begin(), send_ops.end(),
      [&](RemoteExchangeOp const& lhs, RemoteExchangeOp const& rhs) {
        return std::make_tuple(lhs.remote_process, lhs.remote_local_block,
                               exchange_order_index(opts, lhs.peer_offset)) <
               std::make_tuple(rhs.remote_process, rhs.remote_local_block,
                               exchange_order_index(opts, rhs.peer_offset));
      });

  if (layouts.empty() || remote_ops.empty()) return;

  auto* root_layout = layouts.front();
  TORCH_CHECK(root_layout->has_process_group(),
              "[Layout:_launch_cubed_sphere_nccl_remote_ops] remote "
              "communication requires an initialized process group");

  std::lock_guard<std::mutex> lock(g_process_comm_mutex);
  root_layout->comm->group_start();
  for (auto const& op : recv_ops) {
    auto work =
        op.layout->comm->pg->recv(op.layout->owner()->recv_bufs[op.buffer_id],
                                  op.remote_process, op.recv_tag);
    if (work) {
      works_by_block[op.local_block].push_back(work);
    }
  }
  for (auto const& op : send_ops) {
    auto work =
        op.layout->comm->pg->send(op.layout->owner()->send_bufs[op.buffer_id],
                                  op.remote_process, op.send_tag);
    if (work) {
      works_by_block[op.local_block].push_back(work);
    }
  }
  root_layout->comm->group_end();
}

LayoutOptionsImpl::LayoutOptionsImpl() {
  // These enrionment variables will be set by torch.distributed.launch
  // Override by them if they are present
  auto process_rank_env = get_env("PROCESS_RANK", get_env("RANK", "0"));
  auto process_world_size_env =
      get_env("PROCESS_WORLD_SIZE", get_env("WORLD_SIZE", "1"));
  master_addr(get_env("MASTER_ADDR", "127.0.0.1"));
  master_port(std::stoi(get_env("MASTER_PORT", "29501")));
  process_rank(std::stoi(process_rank_env));
  rank(std::stoi(get_env("RANK", process_rank_env)));
  local_rank(std::stoi(get_env("LOCAL_RANK", "0")));
  process_world_size(std::stoi(process_world_size_env));
  world_size(std::stoi(get_env("WORLD_SIZE", process_world_size_env)));
  device_id(std::stoi(get_env("DEVICE_ID", "-1")));
}

LayoutOptions LayoutOptionsImpl::from_yaml(std::string const& filename,
                                           bool verbose) {
  auto op = LayoutOptionsImpl::create();
  auto config = YAML::LoadFile(filename);

  if (!config["distribute"]) return op;

  auto node = config["distribute"];

  op->type() = node["layout"].as<std::string>("slab");
  op->py(node["nb3"].as<int>(1));
  op->px(node["nb2"].as<int>(1));
  op->pz(node["nb1"].as<int>(1));
  op->backend() = get_env("BACKEND", node["backend"].as<std::string>("gloo"));
  op->verbose() = node["verbose"].as<bool>(verbose);

  if (op->verbose()) op->report(SINFO(LayoutOptions));

  return op;
}

std::shared_ptr<LayoutImpl> LayoutImpl::create(LayoutOptions const& options,
                                               MeshBlockImpl* p,
                                               std::string const& name) {
  (void)name;
  if (p == nullptr) options->no_backend(true);

  std::shared_ptr<LayoutImpl> pl;
  if (options->type() == "slab") {
    pl = std::make_shared<SlabLayoutImpl>(options, p);
  } else if (options->type() == "cubed") {
    pl = std::make_shared<CubedLayoutImpl>(options, p);
  } else if (options->type() == "cubed-sphere") {
    pl = std::make_shared<CubedSphereLayoutImpl>(options, p);
  } else {
    TORCH_CHECK(false, "Unsupported layout type: ", options->type());
  }

  if (!options->no_backend()) {
    std::lock_guard<std::mutex> lock(g_local_exchange_mutex);
    g_local_layouts[{options->process_rank(),
                     options->local_block_index(options->rank())}] = pl.get();
  }

  return pl;
}

LayoutImpl::~LayoutImpl() {
  if (options == nullptr || options->no_backend()) return;

  std::lock_guard<std::mutex> lock(g_local_exchange_mutex);
  g_local_layouts.erase(
      {options->process_rank(), options->local_block_index(options->rank())});
}

void LayoutImpl::_prepare_local_exchange(MeshBlockImpl const* pmb,
                                         SyncOptions const& opts) {
  if (options->blocks_per_process() <= 1) return;

  auto key = make_local_exchange_key(*this, opts);
  int expected = options->blocks_per_process();
  std::unique_lock<std::mutex> lock(g_local_exchange_mutex);
  auto& state = g_local_exchange_states[key];
  int generation = state.generation;

  state.arrived += 1;
  if (state.arrived == expected) {
    auto layouts = local_layouts_for(*this);
    lock.unlock();
    std::exception_ptr error;
    try {
      _copy_local_exchange_buffers(layouts, opts);
    } catch (...) {
      error = std::current_exception();
    }
    lock.lock();
    state.error = error;
    state.ready = true;
    state.released = expected;
    g_local_exchange_cv.notify_all();
  } else {
    g_local_exchange_cv.wait(
        lock, [&]() { return state.ready && state.generation == generation; });
  }

  if (state.error != nullptr) {
    auto error = state.error;
    state.released -= 1;
    if (state.released == 0) {
      state.arrived = 0;
      state.ready = false;
      state.error = nullptr;
      state.generation += 1;
      g_local_exchange_cv.notify_all();
    } else {
      g_local_exchange_cv.wait(
          lock, [&]() { return state.generation != generation; });
    }
    std::rethrow_exception(error);
  }

  state.released -= 1;
  if (state.released == 0) {
    state.arrived = 0;
    state.ready = false;
    state.error = nullptr;
    state.generation += 1;
    g_local_exchange_cv.notify_all();
  } else {
    g_local_exchange_cv.wait(lock,
                             [&]() { return state.generation != generation; });
  }
}

std::tuple<int, int, int> LayoutImpl::_remap_exchange_offset(
    std::tuple<int, int, int> iloc, int dy, int dx) const {
  int dx_sgn = 1;
  int dy_sgn = 1;

  if (options->periodic_x() && options->px() == 2 && std::get<0>(iloc) == 0) {
    dx_sgn = -1;
  }

  if (options->periodic_y() && options->py() == 2 && std::get<1>(iloc) == 0) {
    dy_sgn = -1;
  }

  return {dy_sgn * dy, dx_sgn * dx, 0};
}

std::tuple<int, int, int> LayoutImpl::_peer_exchange_offset(
    int peer_rank, int target_rank, SyncOptions const& opts,
    std::tuple<int, int, int> offset) const {
  (void)peer_rank;
  (void)target_rank;
  (void)opts;
  auto [dy, dx, dz] = offset;
  return {-dy, -dx, -dz};
}

void LayoutImpl::_copy_local_exchange_buffers(
    std::vector<LayoutImpl*> const& layouts, SyncOptions const& opts) const {
  for (auto* layout : layouts) {
    auto rank = layout->options->rank();
    auto iloc = layout->loc_of(rank);

    for (int dy_ = opts.dy_min(); dy_ <= opts.dy_max(); ++dy_) {
      for (int dx_ = opts.dx_min(); dx_ <= opts.dx_max(); ++dx_) {
        if (dy_ == 0 && dx_ == 0) continue;
        if (opts.skip_corner() && std::abs(dy_) + std::abs(dx_) == 2) continue;

        auto offset = layout->_remap_exchange_offset(iloc, dy_, dx_);
        int nb = layout->neighbor_rank(iloc, offset);
        if (nb < 0 || nb == rank) continue;
        if (layout->options->owner_process_rank(nb) !=
            layout->options->process_rank()) {
          continue;
        }

        int bid = get_buffer_id(offset);
        auto peer = layouts.at(layout->options->local_block_index(nb));
        auto peer_offset =
            layout->_peer_exchange_offset(nb, rank, opts, offset);
        int peer_bid = get_buffer_id(peer_offset);

        auto& send_bufs = layout->owner()->send_bufs;
        auto& peer_recv_bufs = peer->owner()->recv_bufs;
        for (int n = 0; n < send_bufs[bid].size(); ++n) {
          TORCH_CHECK(
              peer_recv_bufs[peer_bid][n].numel() == send_bufs[bid][n].numel(),
              "local exchange size mismatch from rank ", rank, " to rank ", nb,
              " send_offset=(", std::get<0>(offset), ",", std::get<1>(offset),
              ") recv_offset=(", std::get<0>(peer_offset), ",",
              std::get<1>(peer_offset),
              ") send_shape=", send_bufs[bid][n].sizes(),
              " recv_shape=", peer_recv_bufs[peer_bid][n].sizes());
          peer_recv_bufs[peer_bid][n].view({-1}).copy_(
              send_bufs[bid][n].reshape({-1}));
        }
      }
    }
  }
}

void LayoutImpl::serialize(MeshBlockImpl const* pmb, Variables& vars,
                           SyncOptions const& opts) {
  if (options->verbose()) {
    SINFO(Layout) << "serializing data into send buffers\n";
  }

  // Get my logical location
  auto iloc = loc_of(options->rank());

  // Iterate over all 2D neighbor directions
  int dy_min = opts.dy_min();
  int dy_max = opts.dy_max();
  int dx_min = opts.dx_min();
  int dx_max = opts.dx_max();

  for (int dy = dy_min; dy <= dy_max; ++dy)
    for (int dx = dx_min; dx <= dx_max; ++dx) {
      // Skip the center (self)
      if (dy == 0 && dx == 0) continue;
      if (opts.skip_corner() && std::abs(dy) + std::abs(dx) == 2) continue;
      if (pmb->options->is_physical_boundary(dy, dx, 0)) continue;

      std::tuple<int, int, int> offset(dy, dx, 0);
      int nb = neighbor_rank(iloc, offset);
      if (nb < 0) continue;  // no neighbor

      // Get the interior part for this direction
      auto sub = pmb->part(offset, PartOptions().exterior(false));

      // Copy data from mesh to send buffer
      int bid = get_buffer_id(offset);
      int count = 0;
      pmb->send_bufs[bid].resize(vars.size());
      pmb->recv_bufs[bid].resize(vars.size());
      for (auto& [name, var] : vars) {
        pmb->send_bufs[bid][count] = var.index(sub).clone();
        pmb->recv_bufs[bid][count] =
            torch::empty_like(pmb->send_bufs[bid][count]);
        count++;
      }
    }

  // comm->sync_stream();
}

void LayoutImpl::launch_exchange(
    MeshBlockImpl const* pmb, SyncOptions const& opts,
    std::vector<c10::intrusive_ptr<c10d::Work>>& works) {
  _prepare_local_exchange(pmb, opts);

  if (options->backend() == "nccl" && options->blocks_per_process() > 1 &&
      options->type() == "cubed-sphere") {
    auto key = make_local_exchange_key(*this, opts);
    int expected = options->blocks_per_process();
    int my_index = options->local_block_index(options->rank());
    std::unique_lock<std::mutex> lock(g_local_exchange_mutex);
    auto& state = g_nccl_launch_states[key];
    int generation = state.generation;

    state.arrived += 1;
    if (state.arrived == expected) {
      auto layouts = local_layouts_for(*this);
      lock.unlock();
      std::exception_ptr error;
      std::map<int, std::vector<c10::intrusive_ptr<c10d::Work>>> works_by_block;
      try {
        _launch_cubed_sphere_nccl_remote_ops(opts, works_by_block);
      } catch (...) {
        error = std::current_exception();
      }
      lock.lock();
      state.error = error;
      state.works_by_block = std::move(works_by_block);
      state.ready = true;
      state.released = expected;
      g_local_exchange_cv.notify_all();
    } else {
      g_local_exchange_cv.wait(lock, [&]() {
        return state.ready && state.generation == generation;
      });
    }

    if (state.error != nullptr) {
      auto error = state.error;
      state.released -= 1;
      if (state.released == 0) {
        state.arrived = 0;
        state.ready = false;
        state.error = nullptr;
        state.works_by_block.clear();
        state.generation += 1;
        g_nccl_launch_states.erase(key);
        g_local_exchange_cv.notify_all();
      } else {
        g_local_exchange_cv.wait(
            lock, [&]() { return state.generation != generation; });
      }
      std::rethrow_exception(error);
    }

    works = state.works_by_block[my_index];
    state.released -= 1;
    if (state.released == 0) {
      state.arrived = 0;
      state.ready = false;
      state.error = nullptr;
      state.works_by_block.clear();
      state.generation += 1;
      g_nccl_launch_states.erase(key);
      g_local_exchange_cv.notify_all();
    } else {
      g_local_exchange_cv.wait(
          lock, [&]() { return state.generation != generation; });
    }
    return;
  }

  if (options->backend() == "nccl" && options->blocks_per_process() > 1) {
    auto my_index = options->local_block_index(options->rank());
    auto active_indices = _active_remote_local_indices(opts);
    auto pos =
        std::find(active_indices.begin(), active_indices.end(), my_index);
    if (pos == active_indices.end()) {
      return;
    }

    auto key = make_ordered_phase_key(*this, opts, 0);
    auto my_order =
        static_cast<int>(std::distance(active_indices.begin(), pos));

    std::unique_lock<std::mutex> lock(g_local_exchange_mutex);
    auto& state = g_ordered_phase_states[key];
    g_local_exchange_cv.wait(lock,
                             [&]() { return state.next_index == my_order; });
    lock.unlock();

    exchange_remote(pmb, opts, works);

    lock.lock();
    state.next_index += 1;
    if (state.next_index == active_indices.size()) {
      g_ordered_phase_states.erase(key);
    }
    g_local_exchange_cv.notify_all();
    return;
  }

  exchange_remote(pmb, opts, works);
}

void LayoutImpl::exchange_remote(
    MeshBlockImpl const* pmb, SyncOptions const& opts,
    std::vector<c10::intrusive_ptr<c10d::Work>>& works) {
  TORCH_CHECK(!options->no_backend(),
              "[Layout:exchange_remote] backend is disabled");
  TORCH_CHECK(pmb != nullptr,
              "[Layout:exchange_remote] MeshBlock pointer is null");

  if (options->verbose()) {
    SINFO(Layout) << "performing communication\n";
  }

  // Get my rank
  auto rank = options->rank();

  // Get my logical location
  auto iloc = loc_of(rank);

  int dy_min = opts.dy_min();
  int dy_max = opts.dy_max();
  int dx_min = opts.dx_min();
  int dx_max = opts.dx_max();
  int dx_sgn = 1;
  int dy_sgn = 1;

  // swap the order of first block for periodic condition
  if (options->periodic_x() && options->px() == 2 && std::get<0>(iloc) == 0) {
    dx_sgn = -1;
  }

  if (options->periodic_y() && options->py() == 2 && std::get<1>(iloc) == 0) {
    dy_sgn = -1;
  }

  std::vector<RemoteExchangeOp> remote_ops;

  for (int dy_ = dy_min; dy_ <= dy_max; ++dy_)
    for (int dx_ = dx_min; dx_ <= dx_max; ++dx_) {
      int dy = dy_sgn * dy_;
      int dx = dx_sgn * dx_;

      // skip the center (self)
      if (dy == 0 && dx == 0) continue;
      if (opts.skip_corner() && std::abs(dy) + std::abs(dx) == 2) continue;
      if (pmb->options->is_physical_boundary(dy, dx, 0)) continue;

      std::tuple<int, int, int> offset(dy, dx, 0);
      int nb = neighbor_rank(iloc, offset);
      if (nb < 0) continue;  // no neighbor

      int r = get_buffer_id(offset);
      int remote_process = options->owner_process_rank(nb);
      bool is_remote = remote_process != options->process_rank();

      if (is_remote) {
        int remote_local_block = options->local_block_index(nb);
        int local_block = options->local_block_index(rank);
        auto peer_offset = _peer_exchange_offset(nb, rank, opts, offset);
        int send_id =
            make_comm_tag(remote_local_block, peer_offset, opts.phyid());
        int recv_id = make_comm_tag(local_block, offset, opts.phyid());
        remote_ops.push_back({this, remote_process, local_block,
                              remote_local_block, r, send_id, recv_id, offset,
                              peer_offset});
      } else if (nb == rank) {  // self-send
        int r1 = get_buffer_id(std::tuple<int, int, int>(-dy, -dx, 0));
        for (int n = 0; n < pmb->recv_bufs[r].size(); ++n)
          pmb->recv_bufs[r1][n].copy_(pmb->send_bufs[r][n]);
      }
    }

  if (remote_ops.empty()) return;
  TORCH_CHECK(has_process_group(),
              "[Layout:exchange_remote] remote communication requires an "
              "initialized process group");

  std::lock_guard<std::mutex> lock(g_process_comm_mutex);
  comm->group_start();

  if (options->backend() == "nccl") {
    std::sort(remote_ops.begin(), remote_ops.end(),
              [&](RemoteExchangeOp const& lhs, RemoteExchangeOp const& rhs) {
                return make_remote_order_key(
                           options->process_rank(), lhs.remote_process,
                           lhs.local_block, lhs.remote_local_block, lhs.offset,
                           lhs.peer_offset, opts) <
                       make_remote_order_key(
                           options->process_rank(), rhs.remote_process,
                           rhs.local_block, rhs.remote_local_block, rhs.offset,
                           rhs.peer_offset, opts);
              });

    for (auto const& op : remote_ops) {
      auto work = comm->pg->recv(pmb->recv_bufs[op.buffer_id],
                                 op.remote_process, op.recv_tag);
      if (work) {
        works.push_back(work);
      }
    }
    for (auto const& op : remote_ops) {
      auto work = comm->pg->send(pmb->send_bufs[op.buffer_id],
                                 op.remote_process, op.send_tag);
      if (work) {
        works.push_back(work);
      }
    }
  } else {
    for (auto const& op : remote_ops) {
      auto send_work = comm->pg->send(pmb->send_bufs[op.buffer_id],
                                      op.remote_process, op.send_tag);
      if (send_work) {
        works.push_back(send_work);
      }
      auto recv_work = comm->pg->recv(pmb->recv_bufs[op.buffer_id],
                                      op.remote_process, op.recv_tag);
      if (recv_work) {
        works.push_back(recv_work);
      }
    }
  }

  comm->group_end();
}

void LayoutImpl::deserialize(MeshBlockImpl const* pmb, Variables& vars,
                             SyncOptions const& opts) const {
  if (options->verbose()) {
    SINFO(Layout) << "deserializing data from receive buffers\n";
  }

  // comm->sync_device();

  // Get my logical location
  auto iloc = loc_of(options->rank());

  int dy_min = opts.dy_min();
  int dy_max = opts.dy_max();
  int dx_min = opts.dx_min();
  int dx_max = opts.dx_max();

  // Iterate over all 2D neighbor directions
  for (int dy = dy_min; dy <= dy_max; ++dy)
    for (int dx = dx_min; dx <= dx_max; ++dx) {
      // Skip the center (self)
      if (dy == 0 && dx == 0) continue;
      if (opts.skip_corner() && std::abs(dy) + std::abs(dx) == 2) continue;
      if (pmb->options->is_physical_boundary(dy, dx, 0)) continue;

      std::tuple<int, int, int> offset(dy, dx, 0);
      int nb = neighbor_rank(iloc, offset);
      if (nb < 0) continue;  // no neighbor

      // Get the exterior (ghost zone) part for this direction
      auto sub = pmb->part(offset, PartOptions().exterior(true));

      // Copy data from receive buffer to mesh ghost zones
      int bid = get_buffer_id(offset);
      int count = 0;
      for (auto& [name, var] : vars) {
        var.index_put_(sub, pmb->recv_bufs[bid][count++]);
      }
    }
}

void LayoutImpl::fill_corners(MeshBlockImpl const* pmb, Variables& vars) const {
  auto sub_left = pmb->part({0, -1, 0}, PartOptions().exterior(true));
  auto sub_right = pmb->part({0, +1, 0}, PartOptions().exterior(true));
  auto sub_bot = pmb->part({-1, 0, 0}, PartOptions().exterior(true));
  auto sub_top = pmb->part({+1, 0, 0}, PartOptions().exterior(true));

  // Fill-in left-bot inter-panel corners
  std::tuple<int, int, int> corner(/*dy=*/-1, /*dx=*/-1, 0);
  auto sub = pmb->part(corner, PartOptions().exterior(true));
  for (auto& [name, var] : vars) {
    auto var_left = var.index(sub_left).select(-3, 0).unsqueeze(-3);
    auto var_bot = var.index(sub_bot).select(-2, 0).unsqueeze(-2);
    var.index_put_(sub, 0.5 * (var_left + var_bot));
  }

  // Fill-in right-bot inter-panel corners
  corner = std::tuple<int, int, int>(/*dy=*/-1, /*dx=*/1, 0);
  sub = pmb->part(corner, PartOptions().exterior(true));
  for (auto& [name, var] : vars) {
    auto var_right = var.index(sub_right).select(-3, 0).unsqueeze(-3);
    auto var_bot = var.index(sub_bot).select(-2, -1).unsqueeze(-2);
    var.index_put_(sub, 0.5 * (var_right + var_bot));
  }

  // Fill-in left-top inter-panel corners
  corner = std::tuple<int, int, int>(/*dy=*/1, /*dx=*/-1, 0);
  sub = pmb->part(corner, PartOptions().exterior(true));
  for (auto& [name, var] : vars) {
    auto var_left = var.index(sub_left).select(-3, -1).unsqueeze(-3);
    auto var_top = var.index(sub_top).select(-2, 0).unsqueeze(-2);
    var.index_put_(sub, 0.5 * (var_left + var_top));
  }

  // Fill-in right-top inter-panel corners
  corner = std::tuple<int, int, int>(/*dy=*/1, /*dx=*/1, 0);
  sub = pmb->part(corner, PartOptions().exterior(true));
  for (auto& [name, var] : vars) {
    auto var_right = var.index(sub_right).select(-3, -1).unsqueeze(-3);
    auto var_top = var.index(sub_top).select(-2, -1).unsqueeze(-2);
    var.index_put_(sub, 0.5 * (var_right + var_top));
  }
}

void LayoutImpl::finalize(MeshBlockImpl const* pmb, Variables& vars,
                          SyncOptions const& opts,
                          std::vector<c10::intrusive_ptr<c10d::Work>>& works) {
  if (options->backend() == "nccl" && options->blocks_per_process() > 1) {
    auto my_index = options->local_block_index(options->rank());
    auto active_indices = _active_remote_local_indices(opts);
    auto pos =
        std::find(active_indices.begin(), active_indices.end(), my_index);

    if (pos == active_indices.end()) {
      deserialize(pmb, vars, opts);
      if (opts.skip_corner() && !opts.cross_panel_only()) {
        fill_corners(pmb, vars);
      }
      works.clear();
      return;
    }

    auto key = make_ordered_phase_key(*this, opts, 1);
    auto my_order =
        static_cast<int>(std::distance(active_indices.begin(), pos));

    std::unique_lock<std::mutex> lock(g_local_exchange_mutex);
    auto& state = g_ordered_phase_states[key];
    g_local_exchange_cv.wait(lock,
                             [&]() { return state.next_index == my_order; });
    lock.unlock();

    for (auto& work : works) {
      if (work) {
        work->wait();
      }
    }
    deserialize(pmb, vars, opts);
    if (opts.skip_corner() && !opts.cross_panel_only()) {
      fill_corners(pmb, vars);
    }
    works.clear();

    lock.lock();
    state.next_index += 1;
    if (state.next_index == active_indices.size()) {
      g_ordered_phase_states.erase(key);
    }
    g_local_exchange_cv.notify_all();
    return;
  }

  // Wait for all operations to complete
  for (auto& work : works) {
    if (work) {
      work->wait();
    }
  }

  // Deserialize received data into ghost zones
  deserialize(pmb, vars, opts);

  // Fill corners
  if (opts.skip_corner() && !opts.cross_panel_only()) {
    fill_corners(pmb, vars);
  }

  /*c10d::BarrierOptions op;
  op.device_ids = {options->local_rank()};
  pg->barrier(op)->wait();*/
  {
    if (has_process_group()) {
      std::lock_guard<std::mutex> lock(g_process_comm_mutex);
      comm->pg->barrier()->wait();
    }
  }

  works.clear();
}

void LayoutImpl::_init_process_group() {
  if (!use_process_group()) return;
  comm = ProcessGroupContext::create(options);
}

}  // namespace snap
