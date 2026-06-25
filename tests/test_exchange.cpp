// C/C++
#include <cstdlib>
#include <iostream>

// base
#include <configure.h>

// torch
#ifdef USE_CUDA
#include <c10/cuda/CUDAFunctions.h>
#endif

// snap
#include <snap/layout/cubed_sphere_layout.hpp>
#include <snap/mesh/mesh.hpp>

using namespace snap;

namespace {

constexpr std::tuple<int, int, int> kCardinalOffsets[] = {
    {-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}};

int parse_env_int(char const* name, int fallback) {
  char const* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') return fallback;
  return std::stoi(value);
}

bool parse_env_bool(char const* name, bool fallback) {
  char const* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') return fallback;
  return std::stoi(value) != 0;
}

double side_payload(int side) {
  constexpr double kValues[] = {0.125, 0.250, 0.375, 0.500};
  return kValues[side];
}

int offset_to_side(std::tuple<int, int, int> const& offset) {
  auto [dy, dx, _] = offset;
  if (dy == 0 && dx == -1) return SIDE_L;
  if (dy == 0 && dx == 1) return SIDE_R;
  if (dy == -1 && dx == 0) return SIDE_B;
  if (dy == 1 && dx == 0) return SIDE_T;
  return -1;
}

int source_side(Layout const& layout, std::tuple<int, int, int> const& iloc,
                std::tuple<int, int, int> const& offset, int nb) {
  auto face = std::get<2>(iloc);
  auto nb_face = std::get<2>(layout->loc_of(nb));
  int my_side = offset_to_side(offset);
  TORCH_CHECK(my_side >= 0, "invalid offset");

  if (face == nb_face) {
    return offset_to_side(std::tuple<int, int, int>(-std::get<0>(offset),
                                                    -std::get<1>(offset), 0));
  }

  return CS_FACE_EDGES[face][my_side].nside;
}

torch::Device select_device(LayoutOptions const& layout) {
  auto device = torch::Device(torch::kCPU);
  if (layout->device() == "cuda") {
    TORCH_CHECK(torch::cuda::is_available(),
                "CUDA is required for device=cuda");
    int device_index = layout->device_id();
    if (device_index < 0) device_index = layout->local_rank();
#ifdef USE_CUDA
    c10::cuda::set_device(device_index);
#endif
    device = torch::Device(torch::kCUDA, device_index);
  }
  return device;
}

void seed_hydro_state(MeshBlock block, Variables& vars, torch::Device device) {
  auto pcoord = block->pcoord;
  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();
  int rank = block->options->layout()->rank();

  auto hydro_w = torch::zeros(
      {5, nc3, nc2, nc1},
      torch::TensorOptions().dtype(torch::kFloat64).device(device));

  auto interior = block->part({0, 0, 0}, PartOptions().exterior(false));
  hydro_w.index(interior)[IDN].fill_(rank + 1.0);
  hydro_w.index(interior)[IPR].fill_(10.0 + rank);

  for (auto offset : kCardinalOffsets) {
    int side = offset_to_side(offset);
    auto part = block->part(offset, PartOptions().exterior(false));
    hydro_w.index(part)[IDN].fill_(rank + 1.0 + side_payload(side));
    hydro_w.index(part)[IPR].fill_(100.0 + 10.0 * rank + side);
  }

  vars["hydro_w"] = hydro_w;
}

void seed_scalar_state(MeshBlock block, Variables& vars, torch::Device device) {
  auto pcoord = block->pcoord;
  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();
  int rank = block->options->layout()->rank();
  int nscalar = block->pscalar->nvar();

  auto scalar_r = torch::zeros(
      {nscalar, nc3, nc2, nc1},
      torch::TensorOptions().dtype(torch::kFloat64).device(device));

  auto interior = block->part({0, 0, 0}, PartOptions().exterior(false));
  for (int n = 0; n < nscalar; ++n) {
    scalar_r.index(interior)[n].fill_(0.01 * (n + 1) + rank + 1.0);
  }

  for (auto offset : kCardinalOffsets) {
    int side = offset_to_side(offset);
    auto part = block->part(offset, PartOptions().exterior(false));
    for (int n = 0; n < nscalar; ++n) {
      scalar_r.index(part)[n].fill_(0.01 * (n + 1) + rank + 1.0 +
                                    side_payload(side));
    }
  }

  vars["scalar_r"] = scalar_r;
}

bool ghosts_match_expected(MeshBlock block, Variables const& vars,
                           bool& saw_local_neighbor,
                           bool& saw_remote_neighbor) {
  auto layout = block->get_layout();
  auto rank = layout->options->rank();
  auto iloc = layout->loc_of(rank);
  auto const& hydro_w = vars.at("hydro_w");

  if (!torch::isfinite(hydro_w).all().item<bool>()) return false;

  for (auto offset : kCardinalOffsets) {
    int nb = layout->neighbor_rank(iloc, offset);
    if (nb < 0) continue;

    auto ghost = block->part(offset, PartOptions().exterior(true));
    auto ghost_idn = hydro_w.index(ghost)[IDN];
    int side = source_side(layout, iloc, offset, nb);
    double expected_max = nb + 1.0 + side_payload(side);
    double min_allowed = nb + 1.0 + side_payload(SIDE_L);
    double actual_min = ghost_idn.min().item<double>();
    double actual_max = ghost_idn.max().item<double>();
    bool contains_expected =
        torch::isclose(ghost_idn, torch::full_like(ghost_idn, expected_max))
            .any()
            .item<bool>();

    if (actual_min < min_allowed || actual_max > expected_max ||
        !contains_expected) {
      std::cerr << "ghost mismatch on rank " << rank << " offset=("
                << std::get<0>(offset) << "," << std::get<1>(offset)
                << ") expected_max " << expected_max
                << " actual_min=" << actual_min << " actual_max=" << actual_max
                << " contains_expected=" << contains_expected << std::endl;
      return false;
    }

    if (layout->options->owner_process_rank(nb) ==
        layout->options->process_rank()) {
      saw_local_neighbor = true;
    } else {
      saw_remote_neighbor = true;
    }
  }

  return true;
}

bool scalar_ghosts_match_expected(MeshBlock block, Variables const& vars) {
  auto layout = block->get_layout();
  auto rank = layout->options->rank();
  auto iloc = layout->loc_of(rank);
  auto const& scalar_r = vars.at("scalar_r");

  if (!torch::isfinite(scalar_r).all().item<bool>()) return false;

  for (auto offset : kCardinalOffsets) {
    int nb = layout->neighbor_rank(iloc, offset);
    if (nb < 0) continue;

    auto ghost = block->part(offset, PartOptions().exterior(true));
    auto ghost_scalar = scalar_r.index(ghost);
    int side = source_side(layout, iloc, offset, nb);
    for (int n = 0; n < ghost_scalar.size(0); ++n) {
      double expected = 0.01 * (n + 1) + nb + 1.0 + side_payload(side);
      bool contains_expected =
          torch::isclose(ghost_scalar[n],
                         torch::full_like(ghost_scalar[n], expected))
              .any()
              .item<bool>();
      if (!contains_expected) {
        std::cerr << "scalar ghost mismatch on rank " << rank << " offset=("
                  << std::get<0>(offset) << "," << std::get<1>(offset)
                  << ") scalar=" << n << " expected=" << expected << std::endl;
        return false;
      }
    }
  }

  return true;
}

}  // namespace

int main(int argc, char** argv) {
  auto block_opts = MeshBlockOptionsImpl::from_yaml("test_exchange.yaml");
  auto mesh_opts = MeshOptionsImpl::create();
  mesh_opts->block(block_opts);
  mesh_opts->blocks_per_process(parse_env_int("BLOCKS_PER_PROCESS", 3));

  auto device = select_device(block_opts->layout());
  auto mesh = Mesh(mesh_opts);
  if (device.is_cuda()) {
    mesh->to(device);
  }

  MeshVariables vars(mesh->blocks.size());
  for (int i = 0; i < mesh->blocks.size(); ++i) {
    seed_hydro_state(mesh->blocks[i], vars[i], device);
    seed_scalar_state(mesh->blocks[i], vars[i], device);
  }

  SyncOptions opts;
  opts.interpolate(false).type(kPrimitive);
  MeshVariables prim_vars(mesh->blocks.size());
  for (int i = 0; i < mesh->blocks.size(); ++i) {
    prim_vars[i]["hydro_w"] = vars[i].at("hydro_w");
  }
  mesh->exchange(prim_vars, opts);

  SyncOptions scalar_opts;
  scalar_opts.interpolate(true).type(kScalar);
  MeshVariables scalar_vars(mesh->blocks.size());
  for (int i = 0; i < mesh->blocks.size(); ++i) {
    scalar_vars[i]["scalar_r"] = vars[i].at("scalar_r");
  }
  mesh->exchange(scalar_vars, scalar_opts);
  if (mesh->blocks.front()->get_layout()->has_process_group()) {
    mesh->blocks.front()->get_layout()->comm->barrier();
  }

  bool ok = true;
  bool saw_local_neighbor = false;
  bool saw_remote_neighbor = false;
  for (int i = 0; i < mesh->blocks.size(); ++i) {
    ok = ok && ghosts_match_expected(mesh->blocks[i], vars[i],
                                     saw_local_neighbor, saw_remote_neighbor);
    ok = ok && scalar_ghosts_match_expected(mesh->blocks[i], vars[i]);
  }

  bool expect_local = parse_env_bool("EXPECT_LOCAL_NEIGHBOR", true);
  bool expect_remote = parse_env_bool("EXPECT_REMOTE_NEIGHBOR", true);
  ok = ok && (saw_local_neighbor == expect_local);
  ok = ok && (saw_remote_neighbor == expect_remote);

  if (mesh->blocks.front()->get_layout()->has_process_group()) {
    mesh->blocks.front()->get_layout()->comm->barrier();
  }

  if (!ok) {
    std::cerr << "cubed-sphere exchange regression failed on process "
              << block_opts->layout()->process_rank() << std::endl;
    return 1;
  }

  return 0;
}
