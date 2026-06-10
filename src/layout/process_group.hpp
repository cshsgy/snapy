#pragma once

// C/C++
#include <memory>
#include <mutex>
#include <string>

// torch
#include <c10/util/intrusive_ptr.h>

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>

namespace snap {

struct LayoutOptionsImpl;
using LayoutOptions = std::shared_ptr<LayoutOptionsImpl>;

class ProcessGroupContext {
 public:
  static std::shared_ptr<ProcessGroupContext> create(LayoutOptions const& opts);

  at::intrusive_ptr<c10d::Store> store;
  c10::intrusive_ptr<c10d::ProcessGroup> pg;

  bool is_nccl() const { return backend == "nccl"; }
  bool owns_process_group() const { return owns_process_group_; }
  void group_start() const;
  void group_end() const;
  void sync_stream() const;
  void sync_device() const;

 private:
  explicit ProcessGroupContext(LayoutOptions const& opts);
  void _init();
  void _init_gloo();
  void _init_nccl();
  void _init_ucx();
  void _init_external(c10::intrusive_ptr<c10d::ProcessGroup> pg);

  LayoutOptions options_;
  std::string backend;
  bool owns_process_group_ = false;

  static std::mutex mutex_;
};

}  // namespace snap
