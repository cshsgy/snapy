// base
#include <configure.h>  // nccl

// torch
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

// snap
#include <snap/utils/log.hpp>
#include "layout.hpp"

namespace snap {

void LayoutImpl::_init_nccl() {
  auto opts = c10d::ProcessGroupNCCL::Options::create();

  // Rank -> GPU mapping
  int device_index;
  if (options->device_id() < 0) {
    device_index = options->local_rank();
    options->device_id(device_index);
  } else {
    device_index = options->device_id();
  }

  TORCH_CHECK(device_index < c10::cuda::device_count(), "[Layout] device_id error");

  torch::Device device(torch::kCUDA, device_index);
  c10::cuda::set_device(device_index);

  pg = std::make_shared<c10d::ProcessGroupNCCL>(store, options->rank(),
                                                options->world_size(), opts);
  pg->setBoundDeviceId(device);

  if (options->verbose()) {
    std::cout << "[Rank " << options->rank()
              << ":" << options->local_rank()
              << "] Using NCCL backend on GPU "
              << device_index << "\n";
  }
}

void LayoutImpl::_group_start() const {
  if (options->backend() == "nccl") {
    auto nccl = std::dynamic_pointer_cast<c10d::ProcessGroupNCCL>(pg);
    if (nccl) {
      std::cout << "start coalescing" << std::endl;
      nccl->startCoalescing();
    }
  }
}

c10::intrusive_ptr<c10d::Work> LayoutImpl::_group_end() const {
  if (options->backend() == "nccl") {
    auto nccl = std::dynamic_pointer_cast<c10d::ProcessGroupNCCL>(pg);
    if (nccl) {
      std::cout << "end coalescing" << std::endl;
      return nccl->endCoalescing();
    }
  }

  return nullptr;
}

void LayoutImpl::_sync_device() const {
  if (options->backend() == "nccl") {
    //at::cuda::getCurrentCUDAStream().synchronize();
    cudaDeviceSynchronize();
  }
}

}  // namespace snap
