// base
#include <configure.h>

// C/C++
#include <set>

// torch
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include "process_group.hpp"

namespace snap {

void sync_tensor_streams(std::vector<torch::Tensor> const& tensors) {
  std::set<c10::DeviceIndex> devices;
  for (auto const& tensor : tensors) {
    if (tensor.is_cuda()) {
      devices.insert(tensor.device().index());
    }
  }

  for (auto device : devices) {
    c10::cuda::getCurrentCUDAStream(device).synchronize();
  }
}

}  // namespace snap
