// base
#include <configure.h>

// torch
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include "layout.hpp"
#include "process_group.hpp"

namespace snap {

void ProcessGroupContext::sync_stream() const {
  if (options_->device() == "cuda") {
    c10::cuda::getCurrentCUDAStream(options_->device_id()).synchronize();
  }
}

void ProcessGroupContext::sync_device() const {
  if (options_->device() == "cuda") {
    c10::cuda::device_synchronize();
  }
}

}  // namespace snap
