#pragma once

// A CPU build of snapy must not enter a CUDA test just because a GPU is
// visible to this process. Torch's own availability is not the build.
#include <torch/torch.h>

#include "configure.h"

inline bool snapy_cuda_test_enabled() {
#ifdef USE_CUDA
  return torch::cuda::is_available();
#else
  return false;
#endif
}
