#pragma once

// C/C++
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// torch
#include <c10/util/intrusive_ptr.h>
#include <torch/torch.h>

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>

namespace snap {

struct LayoutOptionsImpl;
using LayoutOptions = std::shared_ptr<LayoutOptionsImpl>;

class CommWork {
 public:
  virtual ~CommWork() = default;
  virtual void wait() = 0;
};
using CommWorkPtr = std::shared_ptr<CommWork>;

class ProcessGroupContext {
 public:
  class UcxTransport {
   public:
    virtual ~UcxTransport() = default;
    virtual CommWorkPtr send(std::vector<torch::Tensor>& tensors, int peer,
                             int tag) = 0;
    virtual CommWorkPtr recv(std::vector<torch::Tensor>& tensors, int peer,
                             int tag) = 0;
    virtual void allreduce(std::vector<torch::Tensor>& tensors,
                           c10d::ReduceOp op) = 0;
    virtual void reduce(std::vector<torch::Tensor>& tensors, c10d::ReduceOp op,
                        int root) = 0;
    virtual void barrier() = 0;
    virtual void shutdown() = 0;
  };

  static std::shared_ptr<ProcessGroupContext> create(LayoutOptions const& opts);

  at::intrusive_ptr<c10d::Store> store;
  c10::intrusive_ptr<c10d::ProcessGroup> pg;

  bool is_ucx() const { return backend == "ucx"; }
  bool owns_process_group() const { return owns_process_group_; }
  bool initialized() const;
  CommWorkPtr send(std::vector<torch::Tensor>& tensors, int peer,
                   int tag) const;
  CommWorkPtr recv(std::vector<torch::Tensor>& tensors, int peer,
                   int tag) const;
  void allreduce(std::vector<torch::Tensor>& tensors, c10d::ReduceOp op) const;
  void reduce(std::vector<torch::Tensor>& tensors, c10d::ReduceOp op,
              int root) const;
  void barrier() const;
  void sync_stream() const;
  void sync_device() const;
  void shutdown() const;

 private:
  explicit ProcessGroupContext(LayoutOptions const& opts);
  void _init();
  void _init_gloo();
  void _init_ucx();
  void _init_external(c10::intrusive_ptr<c10d::ProcessGroup> pg);

  LayoutOptions options_;
  std::string backend;
  bool owns_process_group_ = false;
  std::shared_ptr<UcxTransport> ucx_;

  static std::mutex mutex_;
};

}  // namespace snap
