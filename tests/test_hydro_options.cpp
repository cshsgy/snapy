// C/C++
#include <cstdio>
#include <fstream>
#include <string>

// external
#include <gtest/gtest.h>

#include "cuda_test_gate.hpp"

// torch
#include <torch/torch.h>

// snap
#include <snap/hydro/hydro.hpp>
#include <snap/mesh/meshblock.hpp>

// Unknown keys under `dynamics:` must be rejected, not silently ignored: a
// typo'd or removed option would otherwise run as if applied, with nothing in
// the log. `positivity` stands in for such a key: the parser does not accept
// it, and the error must name it.
TEST(hydro_options, reject_unknown_dynamics_keys) {
  auto write = [](std::string const &fname, std::string const &extra) {
    std::ofstream f(fname);
    f << "dynamics:\n"
         "  equation-of-state:\n"
         "    type: ideal-gas\n"
         "  reconstruct:\n"
         "    vertical: {type: plm, scale: false, shock: false}\n"
         "    horizontal: {type: plm, scale: false, shock: false}\n"
         "  riemann-solver:\n"
         "    type: hllc\n"
         "  verbose: false\n"
      << extra;
  };
  std::string good = "test_hydro_options_good.yaml";
  std::string bad = "test_hydro_options_bad.yaml";

  write(good, "");
  EXPECT_NO_THROW(snap::HydroOptionsImpl::from_yaml(good));

  write(bad, "  positivity: true\n");
  try {
    snap::HydroOptionsImpl::from_yaml(bad);
    FAIL() << "Expected unknown key 'dynamics/positivity' to throw";
  } catch (std::exception const &exc) {
    auto msg = std::string(exc.what());
    EXPECT_NE(msg.find("dynamics/positivity"), std::string::npos) << msg;
  }

  std::remove(good.c_str());
  std::remove(bad.c_str());
}

// The equation-of-state block was never checked: `tracer-floor` (read by
// nothing), a typo or a wrong-case key all loaded silently and ran as if
// applied. Each must now be refused, and the error must name it.
TEST(hydro_options, reject_unknown_equation_of_state_keys) {
  std::string f = "test_eos_options_unknown.yaml";
  auto refused_naming = [&f](std::string const &key) {
    {
      std::ofstream o(f);
      o << "dynamics:\n"
           "  equation-of-state:\n"
           "    type: ideal-gas\n"
           "    "
        << key << ": 1.e-10\n";
    }
    try {
      snap::HydroOptionsImpl::from_yaml(f);
      return false;
    } catch (std::exception const &e) {
      return std::string(e.what()).find("dynamics/equation-of-state/" + key) !=
             std::string::npos;
    }
  };
  EXPECT_TRUE(refused_naming("tracer-floor")) << "tracer-floor accepted";
  EXPECT_TRUE(refused_naming("limter")) << "typo accepted";
  EXPECT_TRUE(refused_naming("Limiter")) << "wrong-case key accepted";
  std::remove(f.c_str());
}

// Every key snapy and kintera read from the block loads, and reaches its
// option: a whitelist missing a key that is read would refuse valid cards.
// uv-solver is set to a non-default value so the check sees it arrive.
TEST(hydro_options, every_equation_of_state_key_is_accepted_and_read) {
  std::string f = "test_eos_options_all.yaml";
  {
    std::ofstream o(f);
    o << "reference-state: {Tref: 300., Pref: 1.e5}\n"
         "species:\n"
         "  - name: dry\n"
         "    composition: {O: 0.42, N: 1.56, Ar: 0.01}\n"
         "    cv_R: 2.5\n"
         "dynamics:\n"
         "  equation-of-state:\n"
         "    type: ideal-gas\n"
         "    gammad: 1.3\n"
         "    weight: 2.e-3\n"
         "    density-floor: 1.e-9\n"
         "    pressure-floor: 1.e-8\n"
         "    temperature-floor: 15.\n"
         "    limiter: true\n"
         "    eos-file: unused.txt\n"
         "    verbose: true\n"
         "    max-iter: 30\n"
         "    ftol: 1.e-8\n"
         "    uv-solver: kkt\n";
  }
  snap::HydroOptions op;
  ASSERT_NO_THROW(op = snap::HydroOptionsImpl::from_yaml(f));
  auto eos = op->eos();
  EXPECT_EQ(eos->type(), "ideal-gas");
  EXPECT_EQ(eos->gammad(), 1.3);
  EXPECT_EQ(eos->weight(), 2.e-3);
  EXPECT_EQ(eos->density_floor(), 1.e-9);
  EXPECT_EQ(eos->pressure_floor(), 1.e-8);
  EXPECT_EQ(eos->temperature_floor(), 15.);
  EXPECT_TRUE(eos->limiter());
  EXPECT_EQ(eos->eos_file(), "unused.txt");
  EXPECT_TRUE(eos->verbose());
  ASSERT_TRUE(eos->thermo());
  EXPECT_EQ(eos->thermo()->max_iter(), 30);
  EXPECT_EQ(eos->thermo()->ftol(), 1.e-8);
  EXPECT_EQ(eos->thermo()->uv_solver(), "kkt");
  std::remove(f.c_str());
}

// snapy lets uv-solver through to kintera, which alone checks its value: a
// kintera older than 2.5.0 ignores the key, so a bad value would load.
TEST(hydro_options, invalid_uv_solver_is_refused_by_kintera) {
  std::string f = "test_eos_options_uv_solver.yaml";
  {
    std::ofstream o(f);
    o << "reference-state: {Tref: 300., Pref: 1.e5}\n"
         "species:\n"
         "  - name: dry\n"
         "    composition: {O: 0.42, N: 1.56, Ar: 0.01}\n"
         "    cv_R: 2.5\n"
         "dynamics:\n"
         "  equation-of-state:\n"
         "    type: ideal-gas\n"
         "    uv-solver: bogus\n";
  }
  try {
    snap::HydroOptionsImpl::from_yaml(f);
    ADD_FAILURE() << "uv-solver: bogus accepted";
  } catch (std::exception const &e) {
    EXPECT_NE(std::string(e.what()).find("Invalid UV solver"),
              std::string::npos)
        << e.what();
  }
  std::remove(f.c_str());
}

// a silently ignored fric-heat key would change a card's physics unlogged
TEST(hydro_options, reject_removed_and_unknown_forcing_keys) {
  auto write = [](std::string const &fname, std::string const &forcing) {
    std::ofstream f(fname);
    f << "dynamics:\n"
         "  equation-of-state:\n"
         "    type: ideal-gas\n"
         "forcing:\n"
      << forcing;
  };
  std::string good = "test_forcing_options_good.yaml";
  std::string removed = "test_forcing_options_removed.yaml";
  std::string unknown = "test_forcing_options_unknown.yaml";

  write(good, "  const-gravity: {grav1: -10.}\n");
  EXPECT_NO_THROW(snap::HydroOptionsImpl::from_yaml(good));

  auto refused_saying = [](std::string const &f, std::string const &needle) {
    try {
      snap::HydroOptionsImpl::from_yaml(f);
      return false;
    } catch (std::exception const &e) {
      return std::string(e.what()).find(needle) != std::string::npos;
    }
  };

  write(removed, "  fric-heat: {}\n");
  EXPECT_TRUE(refused_saying(removed, "has been removed"));

  write(unknown, "  no-such-forcing: {}\n");
  EXPECT_TRUE(refused_saying(unknown, "unknown key"));

  std::remove(good.c_str());
  std::remove(removed.c_str());
  std::remove(unknown.c_str());
}

// a scheme outside {0, 1, 9} was rejected only by the verbose report, while the
// x2/x3 acoustic CFL bound had already been dropped for it
TEST(hydro_options, reject_unsupported_implicit_scheme) {
  auto write = [](std::string const &fname, int scheme) {
    std::ofstream f(fname);
    f << "reference-state: {Tref: 300., Pref: 1.e5}\n"
         "species:\n"
         "  - name: dry\n"
         "    composition: {O: 0.42, N: 1.56, Ar: 0.01}\n"
         "    cv_R: 2.5\n"
         "geometry:\n"
         "  type: cartesian\n"
         "  bounds: {x1min: 0., x1max: 6., x2min: 0., x2max: 1., x3min: 0., "
         "x3max: 1.}\n"
         "  cells: {nx1: 6, nx2: 1, nx3: 1, nghost: 2}\n"
         "dynamics:\n"
         "  equation-of-state:\n"
         "    type: ideal-gas\n"
         "boundary-condition:\n"
         "  external: {x1-inner: reflecting, x1-outer: reflecting}\n"
         "integration:\n"
         "  type: rk3\n"
         "  implicit-scheme: "
      << scheme << "\n";
  };
  std::string f = "test_implicit_scheme.yaml";
  auto build = [&]() {
    std::make_shared<snap::MeshBlockImpl>(
        snap::MeshBlockOptionsImpl::from_yaml(f));
  };
  write(f, 1);
  EXPECT_NO_THROW(build());
  write(f, 3);
  try {
    build();
    ADD_FAILURE() << "unsupported implicit scheme accepted";
  } catch (std::exception const &e) {
    EXPECT_NE(std::string(e.what()).find("Unsupported implicit scheme"),
              std::string::npos)
        << e.what();
  }
  std::remove(f.c_str());
}

// No card in the tree sets wb-wall-clamp, so the shipped default IS the
// production path. test_hydro_ref_x1.cpp calls the dispatch directly and says
// nothing about which value a run gets; this pins the default and the key.
TEST(hydro_options, wb_wall_clamp_ships_enabled) {
  auto write = [](std::string const &fname, std::string const &extra) {
    std::ofstream f(fname);
    f << "dynamics:\n"
         "  equation-of-state:\n"
         "    type: ideal-gas\n"
      << extra;
  };
  std::string f = "test_wb_wall_clamp.yaml";

  write(f, "");
  EXPECT_TRUE(snap::HydroOptionsImpl::from_yaml(f)->wb_wall_clamp())
      << "wb-wall-clamp no longer ships enabled";

  // ...and the key is still read, so a card can still turn it off
  write(f, "  wb-wall-clamp: false\n");
  EXPECT_FALSE(snap::HydroOptionsImpl::from_yaml(f)->wb_wall_clamp());

  std::remove(f.c_str());
}

// The wb-wall-clamp option has exactly one wire into the solver, in
// HydroImpl::_hydro_ref_x1. Every other test drives call_hydro_ref_x1 with a
// literal bool, so replacing that wire with `false` leaves all of them green.
// This one runs both settings through a block and requires them to differ.
static void wb_wall_clamp_reaches_the_x1_reference(torch::Device device) {
  std::string f = "test_wb_wall_clamp_wire.yaml";
  {
    std::ofstream o(f);
    o << "reference-state:\n"
         "  Tref: 300.\n"
         "  Pref: 1.e5\n"
         "species:\n"
         "  - name: dry\n"
         "    composition: {O: 0.42, N: 1.56, Ar: 0.01}\n"
         "    cv_R: 2.5\n"
         "geometry:\n"
         "  type: cartesian\n"
         "  bounds: {x1min: 0., x1max: 16., x2min: 0., x2max: 1., x3min: 0., "
         "x3max: 1.}\n"
         "  cells: {nx1: 16, nx2: 1, nx3: 1, nghost: 3}\n"
         "dynamics:\n"
         "  equation-of-state:\n"
         "    type: ideal-gas\n"
         "  reconstruct:\n"
         "    vertical: {type: weno5, scale: false, shock: false}\n"
         "    horizontal: {type: weno5, scale: false, shock: false}\n"
         "forcing:\n"
         "  const-gravity:\n"
         "    grav1: -1.\n"
         "boundary-condition:\n"
         "  external:\n"
         "    x1-inner: reflecting\n"
         "    x1-outer: reflecting\n";
  }

  auto run = [&f, device](bool clamp) {
    auto options = snap::MeshBlockOptionsImpl::from_yaml(f);
    options->hydro()->wb_wall_clamp() = clamp;
    auto block = std::make_shared<snap::MeshBlockImpl>(options);
    block->to(device, torch::kFloat64);
    auto coord = block->pcoord;
    auto w = torch::zeros({block->phydro->peos->nvar(), coord->options->nc3(),
                           coord->options->nc2(), coord->options->nc1()},
                          torch::dtype(torch::kFloat64).device(device));
    // A stratification the wall rows can disagree about: on a uniform column
    // the two references coincide whatever the clamp does.
    w[snap::IDN] = 1. + 0.05 * coord->x1v;
    w[snap::IPR].fill_(1.e5);
    snap::Variables vars;
    vars["hydro_w"] = w;
    block->initialize(vars);
    return block->phydro->forward(1.e-3, vars.at("hydro_u"), vars);
  };

  auto clamped = run(true);
  auto unclamped = run(false);
  ASSERT_EQ(clamped.device(), device);
  ASSERT_EQ(unclamped.device(), device);
  EXPECT_FALSE(torch::allclose(clamped, unclamped, 1.e-13, 1.e-13))
      << "wb-wall-clamp changed nothing: the option no longer reaches "
         "HydroImpl::_hydro_ref_x1";

  std::remove(f.c_str());
}

TEST(hydro_options, wb_wall_clamp_reaches_the_x1_reference) {
  wb_wall_clamp_reaches_the_x1_reference(torch::kCPU);
}

TEST(hydro_options, wb_wall_clamp_reaches_the_x1_reference_cuda) {
  if (!snapy_cuda_test_enabled()) GTEST_SKIP() << "CUDA is not available";
  wb_wall_clamp_reaches_the_x1_reference(torch::Device(torch::kCUDA, 0));
}
