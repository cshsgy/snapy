// C/C++
#include <algorithm>
#include <filesystem>

// gtest
#include <gtest/gtest.h>

// base
#include <configure.h>

// snap
#include <snap/mesh/mesh.hpp>
#include <snap/output/output_type.hpp>

#ifdef NETCDFOUTPUT
#include <netcdf.h>

using namespace snap;
namespace fs = std::filesystem;

// No rank may return from a collective output call before the root's combined
// file is complete: each rank reads it the moment make_outputs returns.
TEST(Mesh, combined_output_is_complete_when_every_rank_returns) {
  auto block_opts =
      MeshBlockOptionsImpl::from_yaml("test_mesh_multi_block.yaml");
  auto nc = OutputOptionsImpl::create();
  nc->file_type("netcdf").variables({"d"});
  auto rst = OutputOptionsImpl::create();
  rst->file_type("restart");
  block_opts->outputs({nc, rst});
  block_opts->output_dir("output_barrier").basename("barrier");
  auto mesh_opts = MeshOptionsImpl::create();
  mesh_opts->block(block_opts);
  mesh_opts->blocks_per_process(2);

  auto mesh = Mesh(mesh_opts);
  auto layout = mesh->blocks.front()->get_layout();
  ASSERT_TRUE(layout->has_process_group()) << "one process: nothing to gate";
  int rank = layout->options->process_rank();
  // remove() each entry rather than remove_all(): a libtorch.so that exports
  // its own std::filesystem::remove_all can take precedence at link time, and
  // that copy crashes on a non-empty directory (i.e. on every rerun).
  if (rank == layout->options->process_root_rank() &&
      fs::exists("output_barrier")) {
    for (auto const& e : fs::directory_iterator("output_barrier"))
      fs::remove(e.path());
    fs::remove("output_barrier");
  }

  MeshVariables vars(mesh->blocks.size());
  for (int i = 0; i < mesh->blocks.size(); ++i) {
    auto c = mesh->blocks[i]->pcoord->options;
    auto w = torch::zeros({5, c->nc3(), c->nc2(), c->nc1()}, torch::kFloat64);
    w[IDN].fill_(1.0);
    w[IPR].fill_(1.0e5);
    vars[i]["hydro_w"] = w;
  }
  mesh->initialize(vars);  // its barrier orders the cleanup above

  // netcdf alone: a restart combine after it would hold the ranks at its entry
  for (auto& b : mesh->blocks) b->output_types[1]->next_time = 1.e30;
  mesh->make_outputs(vars, 0.);
  auto c = mesh->blocks.front()->options->coord();
  ASSERT_GT(c->global_nx2() * c->global_nx3(), c->nx2() * c->nx3());
  int ncid, id2, id3, varid;
  size_t n2 = 0, n3 = 0;
  bool opened = nc_open("output_barrier/barrier.out0.00000.nc", NC_NOWRITE,
                        &ncid) == NC_NOERR;
  EXPECT_TRUE(opened) << "rank " << rank << " returned before the combine";
  if (opened) {
    nc_inq_dimid(ncid, "x2", &id2);
    nc_inq_dimid(ncid, "x3", &id3);
    nc_inq_dimlen(ncid, id2, &n2);
    nc_inq_dimlen(ncid, id3, &n3);
    EXPECT_EQ(n2 * n3, (size_t)(c->global_nx2() * c->global_nx3()))
        << "rank " << rank;
    std::vector<double> rho(n2 * n3, 0.);
    nc_inq_varid(ncid, "rho", &varid);
    EXPECT_EQ(nc_get_var_double(ncid, varid, rho.data()), NC_NOERR);
    EXPECT_EQ((size_t)std::count(rho.begin(), rho.end(), 1.0), n2 * n3)
        << "rank " << rank << ": a block is missing from the combined file";
    nc_close(ncid);
  }

  for (auto& b : mesh->blocks) {
    b->output_types[0]->next_time = 1.e30;
    b->output_types[1]->next_time = 0.;
  }
  mesh->make_outputs(vars, 0.);
  EXPECT_TRUE(fs::exists("output_barrier/barrier.00000.restart"))
      << "rank " << rank << " returned before the restart bundle";
  int parts = 0;
  for (auto const& e : fs::directory_iterator("output_barrier"))
    parts += e.path().extension() == ".part";
  EXPECT_EQ(parts, 0) << "rank " << rank << ": the root was still bundling";
}
#endif
