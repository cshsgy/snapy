// gtest
#include <gtest/gtest.h>

#include <fstream>
#include <string>

#include "cuda_test_gate.hpp"

// torch
#include <torch/torch.h>

// snap
#include <snap/snap.h>

#include <snap/layout/layout.hpp>
#include <snap/mesh/mesh.hpp>

using namespace snap;

namespace {

// One process owns every block (6 * nb * nb). nx2 = nx3 = 8 is divisible by
// nb2 = 1, 2 and 4. The cross-panel interpolator reads a strip widened by
// cs_interp_margin. With that margin forced to 0, initialize indexes off the
// block at nb2 = 2 and at nb2 = 4. nb2 = 1 has no subdivided panel seam.
std::string write_card(int nb) {
  int blocks = 6 * nb * nb;
  std::string path =
      "test_cubed_sphere_exchange_" + std::to_string(nb) + ".yaml";
  std::ofstream o(path);
  o << "reference-state: {Tref: 0., Pref: 1.e5}\n"
       "species:\n"
       "  - name: dry\n"
       "    composition: {O: 0.42, N: 1.56, Ar: 0.01}\n"
       "    cv_R: 2.5\n"
       "dynamics:\n"
       "  equation-of-state: {type: ideal-gas}\n"
       "distribute:\n"
       "  layout: cubed-sphere\n"
       "  nb2: "
    << nb << "\n  nb3: " << nb << "\n  blocks_per_process: " << blocks
    << "\n  verbose: false\n"
       "geometry:\n"
       "  type: gnomonic-equiangle\n"
       "  cells: {nx1: 1, nx2: 8, nx3: 8, nghost: 2}\n"
       "  bounds: {x2min_pi: -0.25, x2max_pi: 0.25, x3min_pi: -0.25, "
       "x3max_pi: 0.25}\n"
       "boundary-condition:\n"
       "  external:\n"
       "    x1-inner: reflecting\n"
       "    x1-outer: reflecting\n"
       "    x2-inner: custom\n"
       "    x2-outer: custom\n"
       "    x3-inner: custom\n"
       "    x3-outer: custom\n"
       "integration: {type: rk3, cfl: 0.4, implicit-scheme: 0}\n";
  return path;
}

struct PanelFields {
  torch::Tensor x2f;      // (6, 9) interior faces, nghost stripped
  torch::Tensor dx2f;     // (6, 8) interior x2 cell widths
  torch::Tensor density;  // (6, 8, 8) interior hydro_u[IDN] after one stage
};

PanelFields run(int nb, torch::Device device = torch::kCPU) {
  auto block_opts = MeshBlockOptionsImpl::from_yaml(write_card(nb));
  // A card cannot name the device (the layout reads $DEVICE), so set it here:
  // the Mesh constructor picks the block worker pool's device from it.
  if (device.is_cuda()) {
    block_opts->layout()->device("cuda");
    block_opts->layout()->device_id(device.index());
  }
  auto mesh_opts = MeshOptionsImpl::create();
  mesh_opts->block(block_opts);
  mesh_opts->blocks_per_process(6 * nb * nb);
  auto mesh = Mesh(mesh_opts);
  MeshVariables vars(mesh->blocks.size());
  if (device.is_cuda()) {
    // The pool exists only for more than one block and reads the first
    // block's device_str(); each layout reads its own for the exchange.
    EXPECT_GT(mesh->blocks.size(), 1u);
    for (auto const &block : mesh->blocks) {
      EXPECT_EQ(block->options->device_str(), device.str());
    }
    // the pool itself is CUDA: one stream per block
    EXPECT_EQ(mesh->num_worker_streams(), mesh->blocks.size());
  } else {
    EXPECT_EQ(mesh->num_worker_streams(), 0u);
  }

  for (size_t i = 0; i < mesh->blocks.size(); ++i) {
    auto block = mesh->blocks[i];
    // The constructor still builds the block's tensors on the CPU.
    if (device.is_cuda()) block->to(device);
    auto coord = block->pcoord;
    int nvar = block->phydro->peos->nvar();
    auto w = torch::zeros(
        {nvar, coord->options->nc3(), coord->options->nc2(),
         coord->options->nc1()},
        torch::TensorOptions().dtype(torch::kFloat64).device(device));
    w[IDN] = 1. + 0.2 * coord->x2v.unsqueeze(0).unsqueeze(-1) +
             0.05 * coord->x3v.unsqueeze(1).unsqueeze(-1);
    w[IPR].fill_(1.e5);
    vars[i]["hydro_w"] = w;
  }
  mesh->initialize(vars);
  mesh->forward(vars, 1.e-4, 0);

  constexpr int kNx = 8;
  PanelFields out;
  out.x2f = torch::empty({6, kNx + 1}, torch::kFloat64);
  out.dx2f = torch::empty({6, kNx}, torch::kFloat64);
  out.density = torch::empty({6, kNx, kNx}, torch::kFloat64);

  for (size_t i = 0; i < mesh->blocks.size(); ++i) {
    auto block = mesh->blocks[i];
    auto coord = block->pcoord;
    int ng = coord->options->nghost();
    int nx2 = coord->options->nx2();
    int nx3 = coord->options->nx3();
    int ix2 = coord->options->ix2();
    int ix3 = coord->options->ix3();
    auto [rx, ry, face] =
        block->get_layout()->loc_of(block->options->layout()->rank());
    (void)rx;
    (void)ry;
    auto faces = coord->x2f.narrow(0, ng, nx2 + 1);
    out.x2f[face].narrow(0, ix2, nx2 + 1).copy_(faces.cpu());
    out.dx2f[face]
        .narrow(0, ix2, nx2)
        .copy_(coord->dx2f.narrow(0, ng, nx2).cpu());

    // hydro_u keeps a single x1 cell (nc1 == nx1); there is no x1 ghost pad.
    auto rho = vars[i]
                   .at("hydro_u")[IDN]
                   .slice(0, ng, ng + nx3)
                   .slice(1, ng, ng + nx2)
                   .select(2, 0);
    out.density[face]
        .slice(0, ix3, ix3 + nx3)
        .slice(1, ix2, ix2 + nx2)
        .copy_(rho.cpu());
  }
  return out;
}

}  // namespace

// Variable n of a remote exchange adds n * 65536 to the tag, so phyid 64 would
// reuse phyid 0's tag for the second variable.
TEST(CommTag, rejects_tags_that_collide_with_the_variable_offset) {
  std::tuple<int, int, int> offset{0, 1, 0};
  EXPECT_EQ(make_comm_tag(3, offset, 63), 63 * 1024 + 3 * 32 + 1);
  EXPECT_THROW(make_comm_tag(0, offset, 64), c10::Error);
  EXPECT_THROW(make_comm_tag(0, offset, -1), c10::Error);
}

// Decomposition invariance of the gnomonic faces, one metric (dx2f), and the
// conserved density after one RK stage.
TEST(CubedSphere, subdivided_panel_exchange_matches_one_block) {
  PanelFields ref;
  try {
    ref = run(1);
  } catch (std::exception const &e) {
    FAIL() << "nb2=1 " << e.what();
  }
  for (int nb : {2, 4}) {
    try {
      PanelFields got = run(nb);
      EXPECT_TRUE(torch::equal(got.x2f, ref.x2f)) << "x2f nb2=" << nb;
      EXPECT_TRUE(torch::equal(got.dx2f, ref.dx2f)) << "dx2f nb2=" << nb;
      EXPECT_TRUE(torch::equal(got.density, ref.density))
          << "hydro_u nb2=" << nb;
    } catch (std::exception const &e) {
      ADD_FAILURE() << "nb2=" << nb << " " << e.what();
    }
  }
}

TEST(CubedSphere, subdivided_panel_exchange_matches_one_block_cuda) {
  if (!snapy_cuda_test_enabled()) GTEST_SKIP() << "CUDA is not available";
  PanelFields ref;
  try {
    ref = run(1, torch::Device(torch::kCUDA, 0));
  } catch (std::exception const &e) {
    FAIL() << "nb2=1 " << e.what();
  }
  for (int nb : {2, 4}) {
    try {
      PanelFields got = run(nb, torch::Device(torch::kCUDA, 0));
      EXPECT_TRUE(torch::equal(got.x2f, ref.x2f)) << "x2f nb2=" << nb;
      EXPECT_TRUE(torch::equal(got.dx2f, ref.dx2f)) << "dx2f nb2=" << nb;
      EXPECT_TRUE(torch::equal(got.density, ref.density))
          << "hydro_u nb2=" << nb;
    } catch (std::exception const &e) {
      ADD_FAILURE() << "nb2=" << nb << " " << e.what();
    }
  }
}
