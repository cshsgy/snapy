// C/C++
#include <cmath>
#include <limits>
#include <vector>

// gtest
#include <gtest/gtest.h>

#include "cuda_test_gate.hpp"

// torch
#include <torch/torch.h>

// snap
#include <snap/snap.h>

#include <snap/hydro/hydro_dispatch.hpp>

namespace {

struct RefOutput {
  torch::Tensor psf_lo;
  torch::Tensor psf_hi;
  torch::Tensor pref;
  torch::Tensor dsf;
  torch::Tensor dref;
};

RefOutput allocate_output(torch::Tensor const& w) {
  auto sizes = w.sizes().slice(1).vec();
  return {torch::empty(sizes, w.options()), torch::empty(sizes, w.options()),
          torch::empty(sizes, w.options()), torch::empty(sizes, w.options()),
          torch::empty(sizes, w.options())};
}

RefOutput tensor_reference(torch::Tensor const& w, torch::Tensor const& dx1f,
                           torch::Tensor const& anchor_in, int iu, double grav,
                           bool uniform, bool phys_in, bool phys_out) {
  int nc1 = w.size(-1);
  auto rho = w[snap::IDN];
  auto dp = grav * rho * dx1f;
  auto cum = torch::cumsum(dp, -1);
  auto anchor =
      anchor_in.defined()
          ? anchor_in
          : (w[snap::IPR].select(-1, iu) *
             torch::exp(-grav * 0.5 * dx1f[iu] /
                        (w[snap::IPR].select(-1, iu) / rho.select(-1, iu))))
                .unsqueeze(-1);

  auto out = allocate_output(w);
  out.psf_lo.copy_(anchor + cum.select(-1, iu).unsqueeze(-1) - cum + dp);
  out.psf_hi.copy_(out.psf_lo - dp);
  out.psf_lo.clamp_min_(std::numeric_limits<double>::min());
  out.psf_hi.clamp_min_(std::numeric_limits<double>::min());

  if (uniform) {
    out.pref.copy_(0.5 * (out.psf_lo + out.psf_hi));
    auto faces =
        torch::cat({out.psf_lo, out.psf_hi.narrow(-1, nc1 - 1, 1)}, -1);
    constexpr double w6[6] = {11. / 1440., -31. / 480., 401. / 720.,
                              401. / 720., -31. / 480., 11. / 1440.};
    auto six = w6[0] * faces.narrow(-1, 0, nc1 - 4);
    for (int k = 1; k < 6; ++k) {
      six += w6[k] * faces.narrow(-1, k, nc1 - 4);
    }
    auto lo = torch::minimum(out.psf_lo, out.psf_hi).narrow(-1, 2, nc1 - 4);
    auto hi = torch::maximum(out.psf_lo, out.psf_hi).narrow(-1, 2, nc1 - 4);
    auto mid = out.pref.narrow(-1, 2, nc1 - 4);
    mid.copy_(torch::where((six >= lo) & (six <= hi), six, mid));

    constexpr double w6e[2][6] = {
        {95. / 288., 1427. / 1440., -133. / 240., 241. / 720., -173. / 1440.,
         3. / 160.},
        {-3. / 160., 637. / 1440., 511. / 720., -43. / 240., 77. / 1440.,
         -11. / 1440.},
    };
    if (!phys_in) {
      for (int j : {0, 1}) {
        auto val = w6e[j][0] * faces.select(-1, 0);
        for (int m = 1; m < 6; ++m) {
          val += w6e[j][m] * faces.select(-1, m);
        }
        auto flo =
            torch::minimum(out.psf_lo.select(-1, j), out.psf_hi.select(-1, j));
        auto fhi =
            torch::maximum(out.psf_lo.select(-1, j), out.psf_hi.select(-1, j));
        auto cur = out.pref.select(-1, j);
        cur.copy_(torch::where((val >= flo) & (val <= fhi), val, cur));
      }
    }
    if (!phys_out) {
      for (int j : {nc1 - 2, nc1 - 1}) {
        int sigma = j - (nc1 - 5);
        auto val = w6e[4 - sigma][5] * faces.select(-1, nc1 - 5);
        for (int m = 1; m < 6; ++m) {
          val += w6e[4 - sigma][5 - m] * faces.select(-1, nc1 - 5 + m);
        }
        auto flo =
            torch::minimum(out.psf_lo.select(-1, j), out.psf_hi.select(-1, j));
        auto fhi =
            torch::maximum(out.psf_lo.select(-1, j), out.psf_hi.select(-1, j));
        auto cur = out.pref.select(-1, j);
        cur.copy_(torch::where((val >= flo) & (val <= fhi), val, cur));
      }
    }
  } else {
    auto ratio = out.psf_lo / out.psf_hi;
    out.pref.copy_(torch::where((ratio - 1.).abs() < 1.e-6,
                                0.5 * (out.psf_lo + out.psf_hi),
                                dp / torch::log(ratio)));
  }

  auto rop = rho / w[snap::IPR];
  auto lo_edge = rop.narrow(-1, 0, 1);
  auto hi_edge = rop.narrow(-1, nc1 - 1, 1);
  auto pad = torch::cat({lo_edge, lo_edge, rop, hi_edge, hi_edge}, -1);
  auto rs = (pad.narrow(-1, 0, nc1) + 4. * pad.narrow(-1, 1, nc1) +
             6. * pad.narrow(-1, 2, nc1) + 4. * pad.narrow(-1, 3, nc1) +
             pad.narrow(-1, 4, nc1)) /
            16.;
  auto rf = rs.clone();
  rf.narrow(-1, 1, nc1 - 1)
      .copy_(0.5 * (rs.narrow(-1, 0, nc1 - 1) + rs.narrow(-1, 1, nc1 - 1)));
  out.dref.copy_(out.pref * rs);
  out.dsf.copy_(out.psf_lo * rf);
  return out;
}

struct Inputs {
  torch::Tensor w;
  torch::Tensor dx1f;
  torch::Tensor anchor;
};

Inputs make_inputs(torch::ScalarType dtype, bool uniform, bool with_anchor,
                   int nc1 = 10) {
  auto options = torch::TensorOptions().dtype(dtype).device(torch::kCPU);
  constexpr int nc3 = 2;
  constexpr int nc2 = 3;
  auto w = torch::zeros({snap::IPR + 1, nc3, nc2, nc1}, options);
  auto x = torch::arange(nc1, options).view({1, 1, nc1});
  auto column = torch::arange(nc3 * nc2, options).view({nc3, nc2, 1});
  w[snap::IDN].copy_(1.0 + 0.01 * x + 0.002 * column);
  w[snap::IPR].copy_(100.0 - 0.15 * x + 0.03 * column);
  auto dx1f = uniform ? torch::full({nc1}, 0.1, options)
                      : torch::linspace(0.08, 0.13, nc1, options);
  torch::Tensor anchor;
  if (with_anchor) {
    anchor = (95.0 + 0.02 * column).contiguous();
  }
  return {w, dx1f, anchor};
}

constexpr int kIu =
    7;  // nc1 = 10 -> il = 2, six interior cells (the `wide` path)

void dispatch(Inputs const& in, RefOutput const& out, bool uniform,
              bool phys_in, bool phys_out, bool wall_clamp = false,
              int iu = kIu) {
  at::native::call_hydro_ref_x1(in.w.device().type(), in.w, in.dx1f, in.anchor,
                                out.psf_lo, out.psf_hi, out.pref, out.dsf,
                                out.dref, iu, 1.0, uniform, phys_in, phys_out,
                                wall_clamp);
}

//! the cells a block owns: [il, iu]. tensor_reference has no clamped branch on
//! purpose -- the clamp is tested by the PROPERTY below, not against a second
//! copy of the same arithmetic.
torch::Tensor interior(torch::Tensor const& t, int iu = kIu) {
  int il = t.size(-1) - 1 - iu;
  return t.narrow(-1, il, iu - il + 1);
}

//! scale rho and p in the GHOST cells only, leaving [il, iu] untouched
Inputs with_perturbed_ghosts(Inputs const& in, double factor, int iu = kIu) {
  auto w = in.w.clone();
  int nc1 = w.size(-1);
  int il = nc1 - 1 - iu;
  w[snap::IDN].narrow(-1, 0, il).mul_(factor);
  w[snap::IDN].narrow(-1, iu + 1, nc1 - 1 - iu).mul_(factor);
  return {w, in.dx1f, in.anchor};
}

void expect_close(RefOutput const& actual, RefOutput const& expected,
                  double rtol, double atol) {
  EXPECT_TRUE(torch::allclose(actual.psf_lo, expected.psf_lo, rtol, atol));
  EXPECT_TRUE(torch::allclose(actual.psf_hi, expected.psf_hi, rtol, atol));
  EXPECT_TRUE(torch::allclose(actual.pref, expected.pref, rtol, atol));
  EXPECT_TRUE(torch::allclose(actual.dsf, expected.dsf, rtol, atol));
  EXPECT_TRUE(torch::allclose(actual.dref, expected.dref, rtol, atol));
}

TEST(HydroRefX1Dispatch, cpu_matches_tensor_reference) {
  for (auto dtype : {torch::kFloat32, torch::kFloat64}) {
    for (bool uniform : {false, true}) {
      for (bool with_anchor : {false, true}) {
        auto in = make_inputs(dtype, uniform, with_anchor);
        bool physical = !with_anchor;
        auto expected = tensor_reference(in.w, in.dx1f, in.anchor, 7, 1.0,
                                         uniform, physical, physical);
        auto actual = allocate_output(in.w);
        dispatch(in, actual, uniform, physical, physical);
        double tolerance = dtype == torch::kFloat32 ? 2.e-5 : 2.e-12;
        expect_close(actual, expected, tolerance, tolerance);
      }
    }
  }
}

// At a PHYSICAL x1 wall the reference must be built from the
// cells the block owns. Whatever a boundary function writes into the ghosts --
// a reflecting mirror, a fixed-temperature fill -- is not a hydrostatic
// continuation of the column, and a reference that reads it puts a standing
// force on a resting atmosphere. The property, independent of how either side
// computes it: perturb the ghosts, and every INTERIOR output must not move.
TEST(HydroRefX1Dispatch, wall_clamp_keeps_the_interior_free_of_the_ghosts) {
  for (bool uniform : {false, true}) {
    auto base = make_inputs(torch::kFloat64, uniform, /*with_anchor=*/false);
    auto moved = with_perturbed_ghosts(base, 1.3);

    auto a = allocate_output(base.w);
    auto b = allocate_output(moved.w);
    dispatch(base, a, uniform, true, true, /*wall_clamp=*/true);
    dispatch(moved, b, uniform, true, true, /*wall_clamp=*/true);

    // pref/dsf/dref all read the ghosts without the clamp (the control below)
    EXPECT_TRUE(torch::equal(interior(a.pref), interior(b.pref)));
    EXPECT_TRUE(torch::equal(interior(a.dsf), interior(b.dsf)));
    EXPECT_TRUE(torch::equal(interior(a.dref), interior(b.dref)));
  }
}

// ...and the control, so the test above cannot pass on a build where the clamp
// does nothing: with it OFF the same perturbation MUST reach the interior.
TEST(HydroRefX1Dispatch, without_the_clamp_the_ghosts_do_reach_the_interior) {
  auto base = make_inputs(torch::kFloat64, /*uniform=*/true, false);
  auto moved = with_perturbed_ghosts(base, 1.3);

  auto a = allocate_output(base.w);
  auto b = allocate_output(moved.w);
  dispatch(base, a, true, true, true, /*wall_clamp=*/false);
  dispatch(moved, b, true, true, true, /*wall_clamp=*/false);

  EXPECT_FALSE(torch::equal(interior(a.pref), interior(b.pref)));
  EXPECT_FALSE(torch::equal(interior(a.dsf), interior(b.dsf)));
  EXPECT_FALSE(torch::equal(interior(a.dref), interior(b.dref)));
}

TEST(HydroRefX1Dispatch, cuda_matches_cpu) {
  if (!snapy_cuda_test_enabled()) {
    GTEST_SKIP() << "CUDA is not available";
  }

  for (bool uniform : {false, true}) {
    // clamp=false at a seam, then clamp=true at physical walls: the clamped
    // rows are CUDA code too (same impl header, different launch)
    for (bool physical : {false, true}) {
      auto cpu_in = make_inputs(torch::kFloat64, uniform, !physical);
      auto expected = allocate_output(cpu_in.w);
      dispatch(cpu_in, expected, uniform, physical, physical, physical);

      Inputs gpu_in = {cpu_in.w.to(torch::kCUDA), cpu_in.dx1f.to(torch::kCUDA),
                       cpu_in.anchor.defined() ? cpu_in.anchor.to(torch::kCUDA)
                                               : cpu_in.anchor};
      auto actual = allocate_output(gpu_in.w);
      dispatch(gpu_in, actual, uniform, physical, physical, physical);
      expect_close({actual.psf_lo.cpu(), actual.psf_hi.cpu(), actual.pref.cpu(),
                    actual.dsf.cpu(), actual.dref.cpu()},
                   expected, 2.e-12, 2.e-12);
    }
  }
}

// A block too thin to carry a one-sided row (4 interior cells) must still keep
// the ghosts out of its interior -- it falls back to the 2-point mean. This is
// the case the `wide` guard exists for, and a version that lets the interior
// stencil run at the wall fails it.
TEST(HydroRefX1Dispatch, a_thin_block_keeps_the_interior_free_of_the_ghosts) {
  constexpr int nc1 = 8, iu = 5;  // il = 2, four interior cells
  auto base = make_inputs(torch::kFloat64, /*uniform=*/true, false, nc1);
  auto moved = with_perturbed_ghosts(base, 1.3, iu);

  auto a = allocate_output(base.w);
  auto b = allocate_output(moved.w);
  dispatch(base, a, true, true, true, /*wall_clamp=*/true, iu);
  dispatch(moved, b, true, true, true, /*wall_clamp=*/true, iu);

  EXPECT_TRUE(torch::equal(interior(a.pref, iu), interior(b.pref, iu)));
  EXPECT_TRUE(torch::equal(interior(a.dsf, iu), interior(b.dsf, iu)));
  EXPECT_TRUE(torch::equal(interior(a.dref, iu), interior(b.dref, iu)));
}

// ---------------------------------------------------------------------------
// The clamp must not cost ACCURACY where it is not needed.
//
// A one-sided row spans six faces, so on a block with fewer than five x1 cells
// it reaches one face past the owned range -- but at the OPPOSITE end from the
// wall. At a SEAM that face is the neighbour's, and reading it is correct; only
// a second physical wall makes it unreadable. A guard that retreats to the
// two-point mean on every thin block therefore drops the reference from sixth
// to second order on exactly the blocks an nb1 decomposition creates, and puts
// a mixed-order step back into the pref rows that hydro.cpp then exchanges
// across the seam -- the step that exchange exists to remove.
//
// The property below is the one that matters and the one such a guard breaks:
// the reference at a physical wall must be the SAME whether the wall's block
// owns four cells or twelve.

struct Column {
  torch::Tensor w, dx1f;
};

//! An isothermal hydrostatic column of `n` interior cells: mirrored ghosts at
//! the inner wall (what a reflecting BC writes, and not a continuation of the
//! column), the true continuation at the outer seam (what an exchange writes).
//! `g0` is the global index of the first interior cell.
Column hydrostatic_slab(int n, int ng, int g0, double dz, double H, double p0,
                        double rt) {
  auto opt = torch::TensorOptions().dtype(torch::kFloat64);
  int nc1 = n + 2 * ng;
  auto w = torch::zeros({snap::IPR + 1, 1, 1, nc1}, opt);
  auto prs_t = w[snap::IPR],
       rho_t = w[snap::IDN];  // accessor() is deleted on an rvalue
  auto prs = prs_t.accessor<double, 3>();
  auto rho = rho_t.accessor<double, 3>();
  for (int i = 0; i < nc1; ++i) {
    int g = g0 + (i - ng);
    double p = p0 * std::exp(-(g + 0.5) * dz / H);
    if (g <
        0) {  // below the physical wall: the even mirror, deliberately wrong
      p = p0 * std::exp(-(-g - 0.5) * dz / H);
    }
    prs[0][0][i] = p;
    rho[0][0][i] = p / rt;
  }
  return {w, torch::full({nc1}, dz, opt)};
}

TEST(HydroRefX1Dispatch,
     the_wall_reference_does_not_depend_on_the_blocks_x1_size) {
  constexpr int ng = 3;
  constexpr double rt = 287.0 * 300.0, g = 9.8;
  const double H = rt / g, dz = 0.2 * H, p0 = 1.0e5;

  // one block owning twelve cells, wall below, seam above
  auto big = hydrostatic_slab(12, ng, 0, dz, H, p0, rt);
  auto a = allocate_output(big.w);
  at::native::call_hydro_ref_x1(torch::kCPU, big.w, big.dx1f, torch::Tensor(),
                                a.psf_lo, a.psf_hi, a.pref, a.dsf, a.dref,
                                ng + 11, g, true, true, false, true);

  // the bottom FOUR of those cells as their own block -- same wall, same
  // column, and the seam anchor the relay would have handed it (hydro.cpp's
  // kWbRefTag)
  auto small = hydrostatic_slab(4, ng, 0, dz, H, p0, rt);
  auto anchor = a.psf_lo.narrow(-1, ng + 4, 1).clone();
  auto b = allocate_output(small.w);
  at::native::call_hydro_ref_x1(torch::kCPU, small.w, small.dx1f, anchor,
                                b.psf_lo, b.psf_hi, b.pref, b.dsf, b.dref,
                                ng + 3, g, true, true, false, true);

  auto want = a.pref.narrow(-1, ng, 4);
  auto got = b.pref.narrow(-1, ng, 4);
  // 3.3e-3 relative is what a blanket thin-block fallback costs here (~333 Pa
  // at 1e5 Pa), so 1e-14 separates "the same reference" from "a different one"
  // by eleven orders and is not a round-off threshold in disguise.
  EXPECT_LT(((got - want).abs() / want).max().item<double>(), 1.e-14)
      << "want " << want << "\ngot  " << got;
  EXPECT_TRUE(
      torch::equal(a.psf_lo.narrow(-1, ng, 4), b.psf_lo.narrow(-1, ng, 4)));
}

// ...and the control for the thin test above it: a four-cell block with walls
// at BOTH ends is the one case where no one-sided row fits, so it really does
// keep the two-point mean -- and with the clamp OFF the ghosts must reach the
// interior, or that test would pass on a build where the clamp does nothing.
TEST(HydroRefX1Dispatch, without_the_clamp_a_thin_blocks_ghosts_do_reach_it) {
  constexpr int nc1 = 8, iu = 5;  // il = 2, four interior cells
  auto base = make_inputs(torch::kFloat64, /*uniform=*/true, false, nc1);
  auto moved = with_perturbed_ghosts(base, 1.3, iu);

  auto a = allocate_output(base.w);
  auto b = allocate_output(moved.w);
  dispatch(base, a, true, true, true, /*wall_clamp=*/false, iu);
  dispatch(moved, b, true, true, true, /*wall_clamp=*/false, iu);

  EXPECT_FALSE(torch::equal(interior(a.pref, iu), interior(b.pref, iu)));
}

}  // namespace
