// gtest
#include <gtest/gtest.h>

// base
#include <configure.h>

// torch
#ifdef USE_CUDA
#include <c10/cuda/CUDAFunctions.h>
#endif

// snap
#include <snap/mesh/mesh.hpp>

using namespace snap;

namespace {
bool side_is_uniform(torch::Tensor const& side, double value) {
  return torch::allclose(side, torch::full_like(side, value));
}
}  // namespace

TEST(Mesh, multi_block_exchange) {
  auto block_opts =
      MeshBlockOptionsImpl::from_yaml("test_mesh_multi_block.yaml");
  auto mesh_opts = MeshOptionsImpl::create();
  mesh_opts->block(block_opts);
  mesh_opts->blocks_per_process(2);

  auto device = torch::Device(torch::kCPU);
  if (block_opts->layout()->device() == "cuda") {
    ASSERT_TRUE(torch::cuda::is_available());
    int device_index = block_opts->layout()->device_id();
    if (device_index < 0) device_index = block_opts->layout()->local_rank();
#ifdef USE_CUDA
    c10::cuda::set_device(device_index);
#endif
    device = torch::Device(torch::kCUDA, device_index);
  }

  auto mesh = Mesh(mesh_opts);
  if (device.is_cuda()) {
    mesh->to(device);
  }
  MeshVariables vars(mesh->blocks.size());

  bool saw_local_neighbor = false;
  bool saw_remote_neighbor = false;

  for (int i = 0; i < mesh->blocks.size(); ++i) {
    auto block = mesh->blocks[i];
    auto pcoord = block->pcoord;
    int nc1 = pcoord->options->nc1();
    int nc2 = pcoord->options->nc2();
    int nc3 = pcoord->options->nc3();

    auto hydro_w = torch::zeros(
        {5, nc3, nc2, nc1},
        torch::TensorOptions().dtype(torch::kFloat64).device(device));
    hydro_w[IDN].fill_(block->options->layout()->rank() + 1.0);
    hydro_w[IPR].fill_(1.0);
    vars[i]["hydro_w"] = hydro_w;
  }

  mesh->initialize(vars);

  for (int i = 0; i < mesh->blocks.size(); ++i) {
    auto block = mesh->blocks[i];
    auto layout = block->get_layout();
    auto rank = layout->options->rank();
    auto iloc = layout->loc_of(rank);

    for (auto offset : {std::tuple<int, int, int>{-1, 0, 0},
                        std::tuple<int, int, int>{1, 0, 0},
                        std::tuple<int, int, int>{0, -1, 0},
                        std::tuple<int, int, int>{0, 1, 0}}) {
      int nb = layout->neighbor_rank(iloc, offset);
      ASSERT_GE(nb, 0);

      auto side = vars[i]["hydro_w"].index(block->part(offset, PartOptions()));
      EXPECT_TRUE(side_is_uniform(side[IDN], nb + 1.0));

      if (layout->options->owner_process_rank(nb) ==
          layout->options->process_rank()) {
        saw_local_neighbor = true;
      } else {
        saw_remote_neighbor = true;
      }
    }
  }

  EXPECT_TRUE(saw_local_neighbor);
  EXPECT_TRUE(saw_remote_neighbor);
}
