// C/C++
#include <map>
#include <sstream>

// base
#include <configure.h>

// torch
#include <c10/util/intrusive_ptr.h>

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>

// snap
#include <snap/utils/log.hpp>

#include "distributed.hpp"
#include "layout.hpp"
#include "process_group.hpp"

namespace snap {

std::mutex ProcessGroupContext::mutex_;

namespace {
std::string process_group_key(LayoutOptions const& opts) {
  std::ostringstream os;
  os << opts->backend() << "|" << opts->master_addr() << "|"
     << opts->master_port() << "|" << opts->process_rank() << "|"
     << opts->process_world_size() << "|" << opts->local_rank() << "|"
     << opts->device_id();
  return os.str();
}

std::string external_process_group_key(
    LayoutOptions const& opts,
    c10::intrusive_ptr<c10d::ProcessGroup> const& external_pg) {
  std::ostringstream os;
  os << "external|" << process_group_key(opts) << "|"
     << static_cast<void*>(external_pg.get());
  return os.str();
}
}  // namespace

std::shared_ptr<ProcessGroupContext> ProcessGroupContext::create(
    LayoutOptions const& opts) {
  static std::map<std::string, std::weak_ptr<ProcessGroupContext>> cache;

  std::lock_guard<std::mutex> lock(mutex_);
  auto external_pg = get_process_group();
  auto key = external_pg.defined()
                 ? external_process_group_key(opts, external_pg)
                 : process_group_key(opts);
  auto it = cache.find(key);
  if (it != cache.end()) {
    if (auto existing = it->second.lock()) {
      return existing;
    }
  }

  auto ctx =
      std::shared_ptr<ProcessGroupContext>(new ProcessGroupContext(opts));
  cache[key] = ctx;
  return ctx;
}

ProcessGroupContext::ProcessGroupContext(LayoutOptions const& opts)
    : options_(opts), backend(opts->backend()) {
  _init();
}

void ProcessGroupContext::_init() {
  if (options_->no_backend() || options_->process_world_size() <= 1) return;

  auto external_pg = get_process_group();
  if (external_pg.defined()) {
    _init_external(external_pg);
    return;
  }

  if (options_->verbose()) {
    std::cout << "[Process " << options_->process_rank() << ":"
              << options_->local_rank()
              << "] Initializing distributed environment\n";
  }

  c10d::TCPStoreOptions store_opts;
  store_opts.port = options_->master_port();
  store_opts.numWorkers = options_->process_world_size();
  store_opts.isServer = options_->process_rank() == 0;
  store =
      at::make_intrusive<c10d::TCPStore>(options_->master_addr(), store_opts);
  pg = c10::make_intrusive<c10d::ProcessGroup>(store, options_->process_rank(),
                                               options_->process_world_size());

  if (backend == "gloo") {
    _init_gloo();
  } else if (backend == "nccl") {
    _init_gloo();
    _init_nccl();
  } else {
    throw std::runtime_error("Unsupported BACKEND=" + backend);
  }

  pg->barrier()->wait();
  owns_process_group_ = true;

  if (options_->verbose()) {
    std::cout << "[Process " << options_->process_rank() << ":"
              << options_->local_rank()
              << "] Distributed environment initialized with backend="
              << backend << ", world_size=" << options_->process_world_size()
              << "\n";
  }
}

void ProcessGroupContext::_init_external(
    c10::intrusive_ptr<c10d::ProcessGroup> external_pg) {
  TORCH_CHECK(external_pg->getBackendName() == backend,
              "[ProcessGroup] external process group backend mismatch: "
              "LayoutOptions requests ",
              backend, " but registered ProcessGroup uses ",
              external_pg->getBackendName());
  pg = std::move(external_pg);
  owns_process_group_ = false;

  auto bound_device = pg->getBoundDeviceId();
  if (bound_device.has_value()) {
    options_->device_id(bound_device->index());
  }

  if (options_->verbose()) {
    std::cout << "[Process " << options_->process_rank() << ":"
              << options_->local_rank()
              << "] Referencing externally initialized distributed backend="
              << backend << ", world_size=" << options_->process_world_size()
              << "\n";
  }
}

void ProcessGroupContext::_init_gloo() {
  if (options_->verbose()) {
    std::cout << "[Process " << options_->process_rank() << ":"
              << options_->local_rank() << "] Using Gloo backend on CPU\n";
  }

  auto opts = c10d::ProcessGroupGloo::Options::create();
  opts->devices.push_back(c10d::ProcessGroupGloo::createDefaultDevice());

  auto backend_gloo = c10::static_intrusive_pointer_cast<c10d::Backend>(
      c10::make_intrusive<c10d::ProcessGroupGloo>(
          store, options_->process_rank(), options_->process_world_size(),
          opts));

  pg->setDefaultBackend(c10d::ProcessGroup::BackendType::GLOO);
  pg->setBackend(c10::DeviceType::CPU, c10d::ProcessGroup::BackendType::GLOO,
                 backend_gloo);
}

#ifdef NOT_USE_C10D_NCCL
void ProcessGroupContext::_init_nccl() {}
void ProcessGroupContext::group_start() const {}
void ProcessGroupContext::group_end() const {}
void ProcessGroupContext::sync_stream() const {}
void ProcessGroupContext::sync_device() const {}
#endif

}  // namespace snap
