// external
#include <gtest/gtest.h>

// base
#include <configure.h>

// C/C++
#ifdef NETCDFOUTPUT
#include <netcdf.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <future>
#include <memory>
#include <vector>

// torch
#include <torch/torch.h>

// kintera
#include <kintera/constants.h>

// snap
#include <snap/mesh/meshblock.hpp>
#include <snap/output/output_formats.hpp>
#include <snap/output/output_type.hpp>

using namespace snap;

namespace {

class TestOutputType : public OutputType {
 public:
  explicit TestOutputType(OutputOptions const& options) : OutputType(options) {}

  void load_hydro(MeshBlockImpl* pmb, Variables const& vars) {
    loadHydroOutputData(pmb, vars);
  }

  void load_scalar(MeshBlockImpl* pmb, Variables const& vars) {
    loadScalarOutputData(pmb, vars);
  }

  void load_diag(MeshBlockImpl* pmb, Variables const& vars) {
    loadDiagOutputData(pmb, vars);
  }

  void load_user_output(MeshBlockImpl* pmb, Variables const& vars) {
    loadUserOutputData(pmb, vars);
  }

  void append(std::string name, torch::Tensor const& tensor) {
    appendTensorSliceOutput("SCALARS", std::move(name), tensor, 4, 0,
                            tensor.size(0));
  }

  void append_reduced(std::string name, torch::Tensor const& tensor) {
    appendTensorOutput("SCALARS", std::move(name), tensor);
  }

  std::vector<int> output_shape(std::string const& name) const {
    for (auto* pdata = pfirst_data_; pdata != nullptr; pdata = pdata->pnext) {
      if (pdata->name == name) {
        return {pdata->data.GetDim4(), pdata->data.GetDim3(),
                pdata->data.GetDim2(), pdata->data.GetDim1()};
      }
    }
    throw std::runtime_error("missing output variable: " + name);
  }

  double output_value(std::string const& name, int n, int k, int j,
                      int i) const {
    for (auto* pdata = pfirst_data_; pdata != nullptr; pdata = pdata->pnext) {
      if (pdata->name == name) return pdata->data(n, k, j, i);
    }
    throw std::runtime_error("missing output variable: " + name);
  }

  std::vector<std::string> output_names() const {
    std::vector<std::string> names;
    for (auto* pdata = pfirst_data_; pdata != nullptr; pdata = pdata->pnext) {
      names.push_back(pdata->name);
    }
    return names;
  }

  double output_value(std::string const& name) const {
    for (auto* pdata = pfirst_data_; pdata != nullptr; pdata = pdata->pnext) {
      if (pdata->name == name) {
        return pdata->data(0, 0, 0, 0);
      }
    }
    throw std::runtime_error("missing output variable: " + name);
  }
};

std::shared_ptr<MeshBlockImpl> make_block(
    std::string eos_type, std::vector<std::string> scalars = {}) {
  std::vector<std::filesystem::path> candidates = {
      "test_coordinate.yaml",
      std::filesystem::path("tests") / "test_coordinate.yaml",
      std::filesystem::path("build") / "tests" / "test_coordinate.yaml",
  };
  auto it = std::find_if(candidates.begin(), candidates.end(),
                         [](std::filesystem::path const& path) {
                           return std::filesystem::exists(path);
                         });
  if (it == candidates.end()) {
    throw std::runtime_error("could not locate test_coordinate.yaml");
  }

  auto options = MeshBlockOptionsImpl::from_yaml(it->string());
  options->hydro()->eos()->type() = std::move(eos_type);
  options->hydro()->riemann()->type() = "lmars";
  options->scalar()->nvar() = scalars.size();
  options->scalar()->names() = std::move(scalars);
  return std::make_shared<MeshBlockImpl>(options);
}

std::shared_ptr<MeshBlockImpl> make_3d_block() {
  auto options = MeshBlockOptionsImpl::from_yaml("test_forcing_3d.yaml");
  return std::make_shared<MeshBlockImpl>(options);
}

bool contains(std::vector<std::string> const& names, std::string const& name) {
  return std::find(names.begin(), names.end(), name) != names.end();
}

std::filesystem::path save_stage_forcing(std::string const& name,
                                         double density_increment,
                                         bool unsupported = false,
                                         bool scalar = false) {
  auto path = std::filesystem::temp_directory_path() / (name + ".pt");
  torch::jit::Module module(name);
  std::string scalar_body =
      scalar ? "  scalar_ds = torch.full_like(variables[\"scalar_s\"], "
               "0.25)\n"
             : "";
  std::string output;
  if (unsupported) {
    output = "  return {\"unsupported\": hydro_du}\n";
  } else if (scalar) {
    output = "  return {\"hydro_du\": hydro_du, \"scalar_ds\": scalar_ds}\n";
  } else {
    output = "  return {\"hydro_du\": hydro_du}\n";
  }
  module.define(
      "def forward(self, variables: Dict[str, Tensor], dt: float, stage: int) "
      "-> Dict[str, Tensor]:\n"
      "  hydro_u = variables[\"hydro_u\"]\n"
      "  latitude = variables[\"coord.latitude\"]\n"
      "  hydro_du = torch.zeros_like(hydro_u)\n"
      "  hydro_du[0] = " +
      std::to_string(density_increment) + " + torch.zeros_like(latitude)\n" +
      scalar_body + output);
  module.save(path.string());
  return path;
}

}  // namespace

TEST(UserOutput, missing_callback_reports_meaningful_error) {
  auto options = OutputOptionsImpl::create();
  options->variables({"uov"});
  TestOutputType output(options);

  MeshBlockImpl block;
  Variables vars;
  vars["hydro_w"] = torch::ones({1, 1, 1, 1});

  try {
    output.load_user_output(&block, vars);
    FAIL() << "Expected missing user output callback to throw";
  } catch (std::exception const& exc) {
    auto msg = std::string(exc.what());
    EXPECT_NE(msg.find("set_user_output_func"), std::string::npos);
    EXPECT_NE(msg.find("uov"), std::string::npos);
  }
}

TEST(UserOutput, registered_callback_allows_uov_output) {
  auto options = OutputOptionsImpl::create();
  options->variables({"uov"});
  TestOutputType output(options);

  MeshBlockImpl block;
  block.user_output_callback = [](Variables const&) {
    Variables out;
    out["my_uov"] = torch::ones({1, 1, 1});
    return out;
  };

  Variables vars;
  vars["hydro_w"] = torch::ones({1, 1, 1, 1});

  EXPECT_NO_THROW(output.load_user_output(&block, vars));
}

TEST(OutputSlice, selects_boundary_cell_and_collapses_axis) {
  auto block = make_3d_block();
  auto opts = OutputOptionsImpl::create();
  opts->x1_slice(2.0);
  TestOutputType output(opts);

  int nc1 = block->pcoord->options->nc1();
  int nc2 = block->pcoord->options->nc2();
  int nc3 = block->pcoord->options->nc3();
  auto values = torch::arange(nc1 * nc2 * nc3, torch::kFloat64)
                    .reshape({1, nc3, nc2, nc1});
  output.append("field", values);
  output.out_is = block->pcoord->il();
  output.out_ie = block->pcoord->iu();
  output.out_js = block->pcoord->jl();
  output.out_je = block->pcoord->ju();
  output.out_ks = block->pcoord->kl();
  output.out_ke = block->pcoord->ku();

  ASSERT_TRUE(output.TransformOutputData(block.get()));
  EXPECT_EQ(output.islice, block->pcoord->il() + 2);
  EXPECT_EQ(output.output_shape("field"), (std::vector<int>{1, nc3, nc2, 1}));
  EXPECT_DOUBLE_EQ(output.output_value("field", 0, block->pcoord->kl(),
                                       block->pcoord->jl(), 0),
                   values
                       .index({0, block->pcoord->kl(), block->pcoord->jl(),
                               block->pcoord->il() + 2})
                       .item<double>());
  output.ClearOutputData();
}

TEST(OutputSlice, supports_multiple_axes_and_rejects_nonintersecting_block) {
  auto block = make_3d_block();
  auto opts = OutputOptionsImpl::create();
  opts->x2_slice(1.5);
  opts->x3_slice(2.5);
  TestOutputType output(opts);

  int nc1 = block->pcoord->options->nc1();
  int nc2 = block->pcoord->options->nc2();
  int nc3 = block->pcoord->options->nc3();
  output.append("field", torch::ones({1, nc3, nc2, nc1}, torch::kFloat64));
  output.out_is = block->pcoord->il();
  output.out_ie = block->pcoord->iu();
  output.out_js = block->pcoord->jl();
  output.out_je = block->pcoord->ju();
  output.out_ks = block->pcoord->kl();
  output.out_ke = block->pcoord->ku();

  ASSERT_TRUE(output.TransformOutputData(block.get()));
  EXPECT_EQ(output.output_shape("field"), (std::vector<int>{1, 1, 1, nc1}));
  output.ClearOutputData();

  auto outside_opts = OutputOptionsImpl::create();
  outside_opts->x1_slice(block->pcoord->options->x1max());
  TestOutputType outside(outside_opts);
  outside.append("field", torch::ones({1, nc3, nc2, nc1}, torch::kFloat64));
  EXPECT_FALSE(outside.TransformOutputData(block.get()));
  outside.ClearOutputData();
}

TEST(OutputSlice, preserves_axes_absent_from_reduced_outputs) {
  auto block = make_3d_block();
  auto opts = OutputOptionsImpl::create();
  opts->x2_slice(1.5);
  opts->x3_slice(2.5);
  TestOutputType output(opts);

  int nc1 = block->pcoord->options->nc1();
  output.append_reduced("avg_vel", torch::ones({3, nc1}, torch::kFloat64));
  output.out_is = block->pcoord->il();
  output.out_ie = block->pcoord->iu();
  output.out_js = block->pcoord->jl();
  output.out_je = block->pcoord->ju();
  output.out_ks = block->pcoord->kl();
  output.out_ke = block->pcoord->ku();

  ASSERT_TRUE(output.TransformOutputData(block.get()));
  EXPECT_EQ(output.output_shape("avg_vel"), (std::vector<int>{1, 1, 3, nc1}));
  output.ClearOutputData();
}

TEST(OutputSlice, maps_reduced_output_spatial_axes) {
  auto block = make_3d_block();
  auto opts = OutputOptionsImpl::create();
  opts->x2_slice(1.5);
  TestOutputType output(opts);

  int nc2 = block->pcoord->options->nc2();
  int nc3 = block->pcoord->options->nc3();
  auto values =
      torch::arange(2 * nc3 * nc2, torch::kFloat64).reshape({2, nc3, nc2});
  output.append_reduced("path_species", values);
  output.out_is = block->pcoord->il();
  output.out_ie = block->pcoord->iu();
  output.out_js = block->pcoord->jl();
  output.out_je = block->pcoord->ju();
  output.out_ks = block->pcoord->kl();
  output.out_ke = block->pcoord->ku();

  ASSERT_TRUE(output.TransformOutputData(block.get()));
  EXPECT_EQ(output.output_shape("path_species"),
            (std::vector<int>{1, 2, nc3, 1}));
  EXPECT_DOUBLE_EQ(
      output.output_value("path_species", 0, 0, block->pcoord->kl(), 0),
      values.index({0, block->pcoord->kl(), block->pcoord->jl() + 1})
          .item<double>());
  output.ClearOutputData();
}

TEST(OutputSlice, maps_reduced_output_sum_axes) {
  auto block = make_3d_block();
  auto opts = OutputOptionsImpl::create();
  opts->output_sumx2(true);
  TestOutputType output(opts);

  int nc2 = block->pcoord->options->nc2();
  int nc3 = block->pcoord->options->nc3();
  auto values =
      torch::arange(2 * nc3 * nc2, torch::kFloat64).reshape({2, nc3, nc2});
  output.append_reduced("path_species", values);
  output.out_is = block->pcoord->il();
  output.out_ie = block->pcoord->iu();
  output.out_js = block->pcoord->jl();
  output.out_je = block->pcoord->ju();
  output.out_ks = block->pcoord->kl();
  output.out_ke = block->pcoord->ku();

  ASSERT_TRUE(output.TransformOutputData(block.get()));
  EXPECT_EQ(output.output_shape("path_species"),
            (std::vector<int>{1, 2, nc3, 1}));
  EXPECT_DOUBLE_EQ(
      output.output_value("path_species", 0, 0, block->pcoord->kl(), 0),
      values
          .index({0, block->pcoord->kl(),
                  torch::indexing::Slice(block->pcoord->jl(),
                                         block->pcoord->ju() + 1)})
          .sum()
          .item<double>());
  output.ClearOutputData();
}

TEST(OutputSlice, yaml_coordinate_presence_activates_slice_and_rejects_sum) {
  auto opts = OutputOptionsImpl::from_yaml(
      YAML::Load("{type: netcdf, x1_slice: 1.25}"));
  ASSERT_TRUE(opts->x1_slice().has_value());
  EXPECT_DOUBLE_EQ(*opts->x1_slice(), 1.25);
  EXPECT_FALSE(opts->x2_slice().has_value());

  EXPECT_THROW(
      OutputOptionsImpl::from_yaml(YAML::Load("{type: netcdf, x1_slice: 1.25, "
                                              "output_sumx1: true}")),
      std::invalid_argument);
}

#ifdef NETCDFOUTPUT
TEST(OutputSlice, netcdf_writes_selected_coordinate_and_collapsed_dimension) {
  auto block = make_3d_block();
  auto dir = std::filesystem::temp_directory_path() /
             ("snapy_slice_" +
              std::to_string(reinterpret_cast<std::uintptr_t>(block.get())));
  block->options->output_dir(dir.string());
  block->options->basename("slice");

  auto opts = OutputOptionsImpl::create();
  opts->file_type("netcdf");
  opts->variables({"d"});
  opts->x1_slice(2.0);
  opts->combine(false);
  NetcdfOutput output(opts);

  int nc1 = block->pcoord->options->nc1();
  int nc2 = block->pcoord->options->nc2();
  int nc3 = block->pcoord->options->nc3();
  int nvar = block->phydro->peos->nvar();
  Variables vars;
  vars["hydro_w"] = torch::zeros({nvar, nc3, nc2, nc1}, torch::kFloat64);
  vars["hydro_u"] = torch::zeros_like(vars["hydro_w"]);
  vars["hydro_w"][IDN].copy_(
      torch::arange(nc1 * nc2 * nc3, torch::kFloat64).reshape({nc3, nc2, nc1}));

  output.write_output_file(block.get(), vars, 0.0, false);
  auto file = dir / "slice.block0.out0.00000.nc";

  int ncid;
  ASSERT_EQ(nc_open(file.c_str(), NC_NOWRITE, &ncid), NC_NOERR);
  int dimid;
  ASSERT_EQ(nc_inq_dimid(ncid, "x1", &dimid), NC_NOERR);
  size_t length;
  ASSERT_EQ(nc_inq_dimlen(ncid, dimid, &length), NC_NOERR);
  EXPECT_EQ(length, 1);

  int varid;
  ASSERT_EQ(nc_inq_varid(ncid, "x1", &varid), NC_NOERR);
  nc_type type;
  ASSERT_EQ(nc_inq_vartype(ncid, varid, &type), NC_NOERR);
  EXPECT_EQ(type, NC_FLOAT) << "double_precision is off by default";
  float coordinate;
  ASSERT_EQ(nc_get_var_float(ncid, varid, &coordinate), NC_NOERR);
  EXPECT_FLOAT_EQ(coordinate, 2.5F);
  EXPECT_EQ(nc_close(ncid), NC_NOERR);
  std::remove(file.c_str());
  std::remove(dir.c_str());
}

// double_precision: true writes NC_DOUBLE, and what reads back is the
// in-memory field bit for bit: steps of 1/3, which a float cannot hold
TEST(OutputPrecision, netcdf_double_precision_reads_back_exactly) {
  auto block = make_3d_block();
  auto dir = std::filesystem::temp_directory_path() /
             ("snapy_double_" +
              std::to_string(reinterpret_cast<std::uintptr_t>(block.get())));
  block->options->output_dir(dir.string());
  block->options->basename("double");

  NetcdfOutput output(OutputOptionsImpl::from_yaml(
      YAML::Load("{type: netcdf, variables: [d], double_precision: true, "
                 "combine: false}")));

  int nc1 = block->pcoord->options->nc1();
  int nc2 = block->pcoord->options->nc2();
  int nc3 = block->pcoord->options->nc3();
  int ng = block->pcoord->options->nghost();
  Variables vars;
  vars["hydro_w"] = torch::zeros({block->phydro->peos->nvar(), nc3, nc2, nc1},
                                 torch::kFloat64);
  vars["hydro_u"] = torch::zeros_like(vars["hydro_w"]);
  vars["hydro_w"][IDN].copy_(
      1. +
      torch::arange(nc1 * nc2 * nc3, torch::kFloat64).reshape({nc3, nc2, nc1}) /
          3.);
  output.write_output_file(block.get(), vars, 1. / 3., false);

  int ncid, varid;
  nc_type type;
  auto file = dir / "double.block0.out0.00000.nc";
  ASSERT_EQ(nc_open(file.c_str(), NC_NOWRITE, &ncid), NC_NOERR);
  ASSERT_EQ(nc_inq_varid(ncid, "rho", &varid), NC_NOERR);
  ASSERT_EQ(nc_inq_vartype(ncid, varid, &type), NC_NOERR);
  EXPECT_EQ(type, NC_DOUBLE);
  // the writer stores (x1, x3, x2), x1 slowest
  auto in = vars["hydro_w"][IDN]
                .slice(0, ng, nc3 - ng)
                .slice(1, ng, nc2 - ng)
                .slice(2, ng, nc1 - ng)
                .permute({2, 0, 1})
                .contiguous();
  auto out = torch::empty_like(in);
  ASSERT_EQ(nc_get_var_double(ncid, varid, out.data_ptr<double>()), NC_NOERR);
  EXPECT_TRUE(torch::equal(out, in))
      << "max |out - in| = " << (out - in).abs().max();
  double time;
  ASSERT_EQ(nc_inq_varid(ncid, "time", &varid), NC_NOERR);
  ASSERT_EQ(nc_get_var_double(ncid, varid, &time), NC_NOERR);
  EXPECT_EQ(time, 1. / 3.);
  EXPECT_EQ(nc_close(ncid), NC_NOERR);
  // remove() the known file and directory rather than remove_all(): a
  // libtorch.so that exports its own std::filesystem::remove_all can take
  // precedence at link time, and that copy crashes on a non-empty directory.
  std::filesystem::remove(file);
  std::filesystem::remove(dir);
}
#endif

TEST(UserForcing, scripted_stage_forcings_add_tendencies_in_list_order) {
  auto block = make_block("ideal-gas", {"tracer_a"});
  int nc1 = block->pcoord->options->nc1();
  int nc2 = block->pcoord->options->nc2();
  int nc3 = block->pcoord->options->nc3();

  Variables vars;
  vars["hydro_w"] = torch::zeros({block->phydro->peos->nvar(), nc3, nc2, nc1},
                                 torch::dtype(torch::kFloat64));
  vars["hydro_w"][IDN].fill_(1.0);
  vars["hydro_w"][IPR].fill_(3.0);
  vars["hydro_u"] = block->phydro->peos->compute("W->U", {vars["hydro_w"]});
  vars["scalar_s"] =
      torch::ones({1, nc3, nc2, nc1}, torch::dtype(torch::kFloat64));
  vars["scalar_r"] = vars["scalar_s"] / vars["hydro_u"][IDN].unsqueeze(0);

  auto first =
      save_stage_forcing("snapy_stage_forcing_first", 0.25, false, true);
  auto second = save_stage_forcing("snapy_stage_forcing_second", 0.5);
  block->set_user_stage_forcings({first.string(), second.string()});

  block->advance_local(vars, 0.0, 0);

  auto hydro_u = vars.at("hydro_u");
  EXPECT_TRUE(torch::allclose(
      hydro_u[IDN], torch::full_like(hydro_u[IDN], 1.75), 1.e-12, 1.e-12));
  EXPECT_TRUE(torch::allclose(vars["scalar_s"],
                              torch::full_like(vars["scalar_s"], 1.25), 1.e-12,
                              1.e-12));
  std::filesystem::remove(first);
  std::filesystem::remove(second);
}

TEST(UserForcing, scripted_module_is_shared_across_parallel_blocks) {
  auto first_block = make_block("ideal-gas");
  auto second_block = make_block("ideal-gas");
  auto forcing = save_stage_forcing("snapy_stage_forcing_parallel", 0.25);
  first_block->set_user_stage_forcings({forcing.string()});
  second_block->user_stage_forcings = first_block->user_stage_forcings;

  ASSERT_EQ(first_block->user_stage_forcings[0].get(),
            second_block->user_stage_forcings[0].get());

  auto advance = [](std::shared_ptr<MeshBlockImpl> const& block) {
    int nc1 = block->pcoord->options->nc1();
    int nc2 = block->pcoord->options->nc2();
    int nc3 = block->pcoord->options->nc3();
    Variables vars;
    vars["hydro_w"] = torch::zeros({block->phydro->peos->nvar(), nc3, nc2, nc1},
                                   torch::kFloat64);
    vars["hydro_w"][IDN].fill_(1.0);
    vars["hydro_w"][IPR].fill_(3.0);
    vars["hydro_u"] = block->phydro->peos->compute("W->U", {vars["hydro_w"]});
    block->advance_local(vars, 0.0, 0);
    return vars["hydro_u"][IDN].clone();
  };

  auto first = std::async(std::launch::async, advance, first_block);
  auto second = std::async(std::launch::async, advance, second_block);
  ASSERT_EQ(first.wait_for(std::chrono::seconds(10)),
            std::future_status::ready);
  ASSERT_EQ(second.wait_for(std::chrono::seconds(10)),
            std::future_status::ready);
  auto first_density = first.get();
  auto second_density = second.get();
  EXPECT_TRUE(
      torch::allclose(first_density, torch::full_like(first_density, 1.25)));
  EXPECT_TRUE(
      torch::allclose(second_density, torch::full_like(second_density, 1.25)));
  std::filesystem::remove(forcing);
}

TEST(UserForcing, scripted_stage_forcing_rejects_unsupported_keys) {
  auto block = make_block("ideal-gas");
  int nc1 = block->pcoord->options->nc1();
  int nc2 = block->pcoord->options->nc2();
  int nc3 = block->pcoord->options->nc3();

  Variables vars;
  vars["hydro_w"] = torch::zeros({block->phydro->peos->nvar(), nc3, nc2, nc1},
                                 torch::dtype(torch::kFloat64));
  vars["hydro_w"][IDN].fill_(1.0);
  vars["hydro_w"][IPR].fill_(3.0);
  vars["hydro_u"] = block->phydro->peos->compute("W->U", {vars["hydro_w"]});

  auto forcing =
      save_stage_forcing("snapy_stage_forcing_unsupported", 0.0, true);
  block->set_user_stage_forcings({forcing.string()});

  EXPECT_THROW(block->advance_local(vars, 0.0, 0), c10::Error);
  std::filesystem::remove(forcing);
}

TEST(OutputSelection, hydro_fields_depend_on_eos_and_requested_mode) {
  auto shallow_block = make_block("shallow-water");
  auto ideal_block = make_block("ideal-gas");

  Variables shallow_vars;
  shallow_vars["hydro_w"] = torch::ones(
      {shallow_block->phydro->peos->nvar(), 1, 4, 4}, torch::kFloat64);
  shallow_vars["hydro_u"] = torch::ones_like(shallow_vars["hydro_w"]);

  Variables ideal_vars;
  ideal_vars["hydro_w"] = torch::ones(
      {ideal_block->phydro->peos->nvar(), 1, 4, 4}, torch::kFloat64);
  ideal_vars["hydro_u"] = torch::ones_like(ideal_vars["hydro_w"]);

  auto prim_opts = OutputOptionsImpl::create();
  prim_opts->variables({"prim"});
  TestOutputType prim_output(prim_opts);
  prim_output.load_hydro(shallow_block.get(), shallow_vars);
  auto prim_names = prim_output.output_names();
  EXPECT_TRUE(contains(prim_names, "rho"));
  EXPECT_TRUE(contains(prim_names, "vel"));
  EXPECT_FALSE(contains(prim_names, "press"));
  prim_output.ClearOutputData();

  auto cons_opts = OutputOptionsImpl::create();
  cons_opts->variables({"cons"});
  TestOutputType cons_output(cons_opts);
  cons_output.load_hydro(shallow_block.get(), shallow_vars);
  auto shallow_cons_names = cons_output.output_names();
  EXPECT_TRUE(contains(shallow_cons_names, "dens"));
  EXPECT_TRUE(contains(shallow_cons_names, "mom"));
  EXPECT_FALSE(contains(shallow_cons_names, "Etot"));
  cons_output.ClearOutputData();

  cons_output.load_hydro(ideal_block.get(), ideal_vars);
  auto ideal_cons_names = cons_output.output_names();
  EXPECT_TRUE(contains(ideal_cons_names, "dens"));
  EXPECT_TRUE(contains(ideal_cons_names, "mom"));
  EXPECT_TRUE(contains(ideal_cons_names, "Etot"));
}

TEST(OutputSelection, scalar_fields_follow_primitive_and_conserved_requests) {
  auto block = make_block("ideal-gas", {"tracer_a", "tracer_b"});

  Variables vars;
  vars["scalar_r"] = torch::ones({2, 1, 4, 4}, torch::kFloat64);
  vars["scalar_s"] = torch::ones({2, 1, 4, 4}, torch::kFloat64);

  auto scalar_opts = OutputOptionsImpl::create();
  scalar_opts->variables({"scalar"});
  TestOutputType scalar_output(scalar_opts);
  scalar_output.load_scalar(block.get(), vars);

  auto names = scalar_output.output_names();
  EXPECT_TRUE(contains(names, "r_tracer_a"));
  EXPECT_TRUE(contains(names, "r_tracer_b"));
  EXPECT_TRUE(contains(names, "s_tracer_a"));
  EXPECT_TRUE(contains(names, "s_tracer_b"));
}

TEST(OutputSelection, scalar_statistics_are_explicitly_selected) {
  auto block = make_block("ideal-gas", {"tracer_a", "tracer_b"});

  Variables vars;
  vars["scalar_r"] = torch::ones({2, 1, 4, 4}, torch::kFloat64);
  vars["scalar_s"] = torch::ones({2, 1, 4, 4}, torch::kFloat64);

  auto scalar_opts = OutputOptionsImpl::create();
  scalar_opts->variables({"scalar_prim"});
  TestOutputType scalar_output(scalar_opts);
  scalar_output.load_scalar(block.get(), vars);
  auto scalar_names = scalar_output.output_names();
  EXPECT_TRUE(contains(scalar_names, "r_tracer_a"));
  EXPECT_FALSE(contains(scalar_names, "r_tracer_a_mean"));
  scalar_output.ClearOutputData();

  auto stat_opts = OutputOptionsImpl::create();
  stat_opts->variables({"scalar_stat"});
  TestOutputType stat_output(stat_opts);
  stat_output.load_scalar(block.get(), vars);
  auto stat_names = stat_output.output_names();
  EXPECT_TRUE(contains(stat_names, "r_tracer_a_mean"));
  EXPECT_TRUE(contains(stat_names, "r_tracer_a_std"));
  EXPECT_TRUE(contains(stat_names, "r_tracer_b_mean"));
  EXPECT_TRUE(contains(stat_names, "r_tracer_b_std"));
}

TEST(OutputSelection, primitive_statistics_are_explicitly_selected) {
  auto block = make_block("ideal-gas");

  Variables vars;
  vars["hydro_w"] =
      torch::ones({block->phydro->peos->nvar(), 1, 4, 4}, torch::kFloat64);
  vars["hydro_u"] = torch::ones_like(vars["hydro_w"]);

  auto prim_opts = OutputOptionsImpl::create();
  prim_opts->variables({"prim"});
  TestOutputType prim_output(prim_opts);
  prim_output.load_hydro(block.get(), vars);
  auto prim_names = prim_output.output_names();
  EXPECT_FALSE(contains(prim_names, "rho_mean"));
  EXPECT_FALSE(contains(prim_names, "vel1_mean"));
  prim_output.ClearOutputData();

  auto stat_opts = OutputOptionsImpl::create();
  stat_opts->variables({"prim_stat"});
  TestOutputType stat_output(stat_opts);
  stat_output.load_hydro(block.get(), vars);
  auto stat_names = stat_output.output_names();
  EXPECT_TRUE(contains(stat_names, "rho_mean"));
  EXPECT_TRUE(contains(stat_names, "rho_std"));
  EXPECT_TRUE(contains(stat_names, "press_mean"));
  EXPECT_TRUE(contains(stat_names, "press_std"));
  EXPECT_TRUE(contains(stat_names, "vel1_mean"));
  EXPECT_TRUE(contains(stat_names, "vel2_mean"));
  EXPECT_TRUE(contains(stat_names, "vel3_mean"));
  EXPECT_TRUE(contains(stat_names, "vel1_std"));
  EXPECT_TRUE(contains(stat_names, "vel2_std"));
  EXPECT_TRUE(contains(stat_names, "vel3_std"));
}

TEST(OutputDiagnostics, virtual_potential_temperature_uses_dry_gas_constant) {
  auto block = make_3d_block();
  int nc1 = block->pcoord->options->nc1();
  int nc2 = block->pcoord->options->nc2();
  int nc3 = block->pcoord->options->nc3();

  Variables vars;
  vars["hydro_w"] = torch::zeros({block->phydro->peos->nvar(), nc3, nc2, nc1},
                                 torch::kFloat64);
  vars["hydro_w"][IDN].fill_(1.0);
  vars["hydro_w"][IPR].fill_(1.e5);
  vars["hydro_w"][ICY].fill_(0.2);

  auto opts = OutputOptionsImpl::create();
  opts->variables({"thermo"});
  TestOutputType output(opts);
  output.load_diag(block.get(), vars);

  auto temp = block->phydro->peos->compute("W->T", {vars["hydro_w"]});
  auto Rd = kintera::constants::Rgas / block->phydro->peos->species_weight(0);
  auto factor = vars["hydro_w"][IPR] / (vars["hydro_w"][IDN] * Rd * temp);
  auto expected = output.output_value("theta") *
                  factor
                      .index({block->pcoord->kl(), block->pcoord->jl(),
                              block->pcoord->il()})
                      .item<double>();

  EXPECT_NEAR(output.output_value("theta_v"), expected, 1.e-10);
  output.ClearOutputData();
}

TEST(OutputDiagnostics, divergence_of_uniform_cartesian_velocity_is_zero) {
  auto block = make_3d_block();
  int nc1 = block->pcoord->options->nc1();
  int nc2 = block->pcoord->options->nc2();
  int nc3 = block->pcoord->options->nc3();

  Variables vars;
  vars["hydro_w"] = torch::zeros({block->phydro->peos->nvar(), nc3, nc2, nc1},
                                 torch::kFloat64);
  vars["hydro_w"][IDN].fill_(1.0);
  vars["hydro_w"][IPR].fill_(1.e5);
  vars["hydro_w"].narrow(0, IVX, 3).fill_(2.0);

  auto opts = OutputOptionsImpl::create();
  opts->variables({"diagnostics"});
  TestOutputType output(opts);
  output.load_diag(block.get(), vars);

  EXPECT_NEAR(output.output_value("div", 0, block->pcoord->kl(),
                                  block->pcoord->jl(), block->pcoord->il()),
              0.0, 1.e-12);
  EXPECT_NEAR(output.output_value("div_h", 0, block->pcoord->kl(),
                                  block->pcoord->jl(), block->pcoord->il()),
              0.0, 1.e-12);
  output.ClearOutputData();
}

TEST(OutputDiagnostics, cartesian_curl_of_solid_body_rotation) {
  auto block = make_3d_block();
  auto pcoord = block->pcoord;
  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();

  Variables vars;
  vars["hydro_w"] = torch::zeros({block->phydro->peos->nvar(), nc3, nc2, nc1},
                                 torch::kFloat64);
  vars["hydro_w"][IDN].fill_(1.0);
  vars["hydro_w"][IPR].fill_(1.e5);

  auto mesh = torch::meshgrid({pcoord->x3v, pcoord->x2v, pcoord->x1v}, "ij");
  vars["hydro_w"][IVX].zero_();
  vars["hydro_w"][IVY].copy_(-mesh[0]);
  vars["hydro_w"][IVZ].copy_(mesh[1]);

  auto opts = OutputOptionsImpl::create();
  opts->variables({"curl"});
  TestOutputType output(opts);
  output.load_diag(block.get(), vars);

  EXPECT_NEAR(output.output_value("curl", VEL1, pcoord->kl(), pcoord->jl(),
                                  pcoord->il()),
              2.0, 1.e-12);
  EXPECT_NEAR(output.output_value("curl", VEL2, pcoord->kl(), pcoord->jl(),
                                  pcoord->il()),
              0.0, 1.e-12);
  EXPECT_NEAR(output.output_value("curl", VEL3, pcoord->kl(), pcoord->jl(),
                                  pcoord->il()),
              0.0, 1.e-12);
  output.ClearOutputData();
}

TEST(OutputDiagnostics, gnomonic_diagnostics_are_finite) {
  auto block = make_block("ideal-gas");
  auto pcoord = block->pcoord;
  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();

  Variables vars;
  vars["hydro_w"] = torch::zeros({block->phydro->peos->nvar(), nc3, nc2, nc1},
                                 torch::kFloat64);
  vars["hydro_w"][IDN].fill_(1.0);
  vars["hydro_w"][IPR].fill_(1.e5);
  vars["hydro_w"].narrow(0, IVX, 3).fill_(1.0);

  auto opts = OutputOptionsImpl::create();
  opts->variables({"diagnostics"});
  TestOutputType output(opts);
  output.load_diag(block.get(), vars);

  EXPECT_TRUE(std::isfinite(output.output_value("curl", VEL1, pcoord->kl(),
                                                pcoord->jl(), pcoord->il())));
  EXPECT_TRUE(std::isfinite(
      output.output_value("div", 0, pcoord->kl(), pcoord->jl(), pcoord->il())));
  output.ClearOutputData();
}

TEST(OutputStatistics, primitive_statistics_are_time_weighted_and_reset) {
  auto block = make_block("ideal-gas");

  Variables vars;
  vars["hydro_w"] =
      torch::ones({block->phydro->peos->nvar(), 1, 4, 4}, torch::kFloat64);
  vars["hydro_u"] = torch::ones_like(vars["hydro_w"]);

  auto opts = OutputOptionsImpl::create();
  opts->variables({"prim_stat"});
  TestOutputType output(opts);

  vars["hydro_w"].fill_(1.0);
  output.AccumulateStats(vars, 0.0);
  output.load_hydro(block.get(), vars);
  EXPECT_DOUBLE_EQ(output.output_value("rho_mean"), 1.0);
  EXPECT_DOUBLE_EQ(output.output_value("rho_std"), 0.0);
  output.ClearOutputData();

  vars["hydro_w"].fill_(3.0);
  output.AccumulateStats(vars, 1.0);
  vars["hydro_w"].fill_(5.0);
  output.AccumulateStats(vars, 3.0);

  output.load_hydro(block.get(), vars);
  EXPECT_NEAR(output.output_value("rho_mean"), 13.0 / 3.0, 1.e-12);
  EXPECT_NEAR(output.output_value("rho_std"), std::sqrt(8.0 / 9.0), 1.e-12);
  EXPECT_NEAR(output.output_value("vel1_mean"), 13.0 / 3.0, 1.e-12);
  EXPECT_NEAR(output.output_value("vel1_std"), std::sqrt(8.0 / 9.0), 1.e-12);
  output.ClearOutputData();

  output.ResetStats(3.0);
  vars["hydro_w"].fill_(7.0);
  output.AccumulateStats(vars, 4.0);
  output.load_hydro(block.get(), vars);
  EXPECT_DOUBLE_EQ(output.output_value("rho_mean"), 7.0);
  EXPECT_DOUBLE_EQ(output.output_value("rho_std"), 0.0);
  output.ClearOutputData();

  TestOutputType stable_output(opts);
  vars["hydro_w"].fill_(1.e9 + 1.0);
  stable_output.AccumulateStats(vars, 0.0);
  vars["hydro_w"].fill_(1.e9 + 1.0);
  stable_output.AccumulateStats(vars, 1.0);
  vars["hydro_w"].fill_(1.e9 + 3.0);
  stable_output.AccumulateStats(vars, 2.0);
  stable_output.load_hydro(block.get(), vars);
  EXPECT_NEAR(stable_output.output_value("rho_mean"), 1.e9 + 2.0, 1.e-6);
  EXPECT_NEAR(stable_output.output_value("rho_std"), 1.0, 1.e-12);
}

TEST(OutputStatistics, scalar_statistics_are_time_weighted_and_reset) {
  auto block = make_block("ideal-gas", {"tracer_a"});

  Variables vars;
  vars["scalar_r"] = torch::ones({1, 1, 4, 4}, torch::kFloat64);

  auto opts = OutputOptionsImpl::create();
  opts->variables({"scalar_stat"});
  TestOutputType output(opts);

  vars["scalar_r"].fill_(1.0);
  output.AccumulateStats(vars, 0.0);
  output.load_scalar(block.get(), vars);
  EXPECT_DOUBLE_EQ(output.output_value("r_tracer_a_mean"), 1.0);
  EXPECT_DOUBLE_EQ(output.output_value("r_tracer_a_std"), 0.0);
  output.ClearOutputData();

  vars["scalar_r"].fill_(3.0);
  output.AccumulateStats(vars, 1.0);
  vars["scalar_r"].fill_(5.0);
  output.AccumulateStats(vars, 3.0);

  output.load_scalar(block.get(), vars);
  EXPECT_NEAR(output.output_value("r_tracer_a_mean"), 13.0 / 3.0, 1.e-12);
  EXPECT_NEAR(output.output_value("r_tracer_a_std"), std::sqrt(8.0 / 9.0),
              1.e-12);
  output.ClearOutputData();

  output.ResetStats(3.0);
  vars["scalar_r"].fill_(7.0);
  output.AccumulateStats(vars, 4.0);
  output.load_scalar(block.get(), vars);
  EXPECT_DOUBLE_EQ(output.output_value("r_tracer_a_mean"), 7.0);
  EXPECT_DOUBLE_EQ(output.output_value("r_tracer_a_std"), 0.0);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
