// external
#include <gtest/gtest.h>

// C/C++
#include <string>
#include <thread>
#include <vector>

// torch
#include <torch/torch.h>

// snap
#include <snap/snap.h>

#include <snap/mesh/mesh.hpp>

// tests
#include "device_testing.hpp"

using namespace snap;

namespace {

// The vapor twin of test_parentless_cloud_nb1: the same 16-cell global column
// as one meshblock (nb1 = 1) or split along x1 (cubed layout). Vapor is 0.01 in
// the upper half and zero in the lower half except one over-drained cell at
// -deficit; the clouds are zero, so only the vapor repair acts. The global
// column total of vapor is positive, so the repair must succeed and keep the
// mass whatever nb1 is.
struct Repair {
  double before, after;
  std::string error;  // a block's apply_conserved_limiter_ exception, if any
  std::vector<double> vapor;  // interior vapor, one entry per global x1 cell
};

Repair repair_split_column(int nb1, torch::Device device, torch::Dtype dtype,
                           double deficit) {
  auto block_opts =
      MeshBlockOptionsImpl::from_yaml("test_parentless_cloud_nb1.yaml");
  if (nb1 > 1) {
    block_opts->layout()->type() = "cubed";
    block_opts->layout()->pz(nb1);
  }
  auto mesh_opts = MeshOptionsImpl::create();
  mesh_opts->block(block_opts);
  mesh_opts->blocks_per_process(nb1);

  auto mesh = Mesh(mesh_opts);
  mesh->to(device, dtype);
  EXPECT_EQ(static_cast<int>(mesh->blocks.size()), nb1);

  std::vector<torch::Tensor> cons(mesh->blocks.size());
  for (size_t b = 0; b < mesh->blocks.size(); ++b) {
    auto block = mesh->blocks[b];
    auto coord = block->pcoord;
    int il = coord->il(), iu = coord->iu();
    double x1min = coord->options->x1min();
    double dx = (coord->options->x1max() - x1min) / (iu - il + 1);
    cons[b] = torch::zeros({block->phydro->peos->nvar(), coord->options->nc3(),
                            coord->options->nc2(), coord->options->nc1()},
                           torch::device(device).dtype(dtype));
    cons[b][IDN].fill_(1.);
    cons[b][IPR].fill_(1.e8);
    for (int i = il; i <= iu; ++i) {
      int g = static_cast<int>(x1min + (i - il + 0.5) * dx);  // global cell
      double vapor = g >= 8 ? 0.01 : (g == 3 ? -deficit : 0.);
      cons[b][ICY].select(-1, i).fill_(vapor);
    }
  }

  auto total_mass = [&] {
    double mass = 0.;
    for (size_t b = 0; b < mesh->blocks.size(); ++b) {
      int il = mesh->blocks[b]->pcoord->il(),
          iu = mesh->blocks[b]->pcoord->iu();
      auto c = cons[b].slice(-1, il, iu + 1).to(torch::kFloat64);
      mass += (c[IDN] + c.narrow(0, ICY, 3).sum(0)).sum().item<double>();
    }
    return mass;
  };
  Repair r{total_mass(), 0., ""};

  // what advance_local() step (4) does on every block, one thread each, as
  // Mesh::forward runs them; an exception is recorded, not left to terminate
  std::vector<std::string> errors(mesh->blocks.size());
  std::vector<std::thread> threads;
  for (size_t b = 0; b < mesh->blocks.size(); ++b)
    threads.emplace_back([&, b] {
      try {
        mesh->blocks[b]->phydro->peos->apply_conserved_limiter_(
            cons[b], /*whole_column=*/true);
      } catch (std::exception const& e) {
        errors[b] = e.what();
      }
    });
  for (auto& t : threads) t.join();
  for (auto const& e : errors)
    if (!e.empty()) r.error = e.substr(0, e.find('\n'));

  r.after = total_mass();
  int ncell = 0;
  for (size_t b = 0; b < mesh->blocks.size(); ++b)
    ncell += mesh->blocks[b]->pcoord->iu() - mesh->blocks[b]->pcoord->il() + 1;
  r.vapor.assign(ncell, 0.);
  for (size_t b = 0; b < mesh->blocks.size(); ++b) {
    auto coord = mesh->blocks[b]->pcoord;
    int il = coord->il(), iu = coord->iu();
    double x1min = coord->options->x1min();
    double dx = (coord->options->x1max() - x1min) / (iu - il + 1);
    for (int i = il; i <= iu; ++i) {
      int g = static_cast<int>(x1min + (i - il + 0.5) * dx);
      r.vapor[g] = cons[b][ICY].select(-1, i).flatten()[0].item<double>();
    }
  }
  return r;
}

}  // namespace

// One meshblock: the scan reaches the over-drained cell with nothing below it
// and takes the deficit from the vapor above; the mass is kept.
TEST_P(DeviceTest, vapor_repair_keeps_the_mass_with_nb1_1) {
  double deficit = 1.e-4;
  auto r = repair_split_column(1, device, dtype, deficit);
  EXPECT_TRUE(r.error.empty()) << "nb1 = 1 threw: " << r.error;
  double tol = (dtype == torch::kFloat64 ? 1.e-12 : 1.e-6) * r.before;
  EXPECT_NEAR(r.after, r.before, tol);
}

// Two meshblocks along x1: the over-drained cell sits in the lower block and
// the positive vapor in the upper one. whole_column repairs that split column
// as one column, so each interior vapor cell matches the nb1 = 1 repair.
TEST_P(DeviceTest, vapor_repair_keeps_the_mass_with_nb1_2) {
  double deficit = 1.e-4;
  auto one = repair_split_column(1, device, dtype, deficit);
  auto two = repair_split_column(2, device, dtype, deficit);
  EXPECT_TRUE(one.error.empty()) << "nb1 = 1 threw: " << one.error;
  EXPECT_TRUE(two.error.empty()) << "nb1 = 2 threw: " << two.error;
  double tol = (dtype == torch::kFloat64 ? 1.e-12 : 1.e-6) * two.before;
  EXPECT_NEAR(two.after, two.before, tol);
  ASSERT_EQ(one.vapor.size(), two.vapor.size());
  for (size_t g = 0; g < one.vapor.size(); ++g)
    EXPECT_EQ(two.vapor[g], one.vapor[g]) << "vapor cell " << g;
}
