// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/cpu/Loops.h>
#include <torch/torch.h>

// snap
#include <snap/snap.h>

#include "riemann_dispatch.hpp"

// impl
#include "lmars_impl.h"

namespace snap {

void call_lmars_cpu(at::TensorIterator& iter, int dim) {
  int grain_size = iter.numel() / at::get_num_threads();

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_lmars_cpu", [&] {
    auto nhydro = at::native::ensure_nonempty_size(iter.output(0), 0);
    auto stride = at::native::ensure_nonempty_stride(iter.output(0), 0);
    auto ny = nhydro - ICY;

    iter.for_each(
        [&](char** data, const int64_t* strides, int64_t n) {
          for (int i = 0; i < n; i++) {
            auto out = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
            auto face_pressure =
                reinterpret_cast<scalar_t*>(data[1] + i * strides[1]);
            auto wl = reinterpret_cast<scalar_t*>(data[2] + i * strides[2]);
            auto wr = reinterpret_cast<scalar_t*>(data[3] + i * strides[3]);
            auto elr = reinterpret_cast<scalar_t*>(data[4] + i * strides[4]);
            auto glr = reinterpret_cast<scalar_t*>(data[5] + i * strides[5]);
            lmars_impl(out, wl, wr, *elr, *(elr + stride), *glr,
                       *(glr + stride), dim, ny, stride, stride, face_pressure);
          }
        },
        grain_size);
  });
}

void call_lmars_mps(at::TensorIterator& iter, int dim) {
  auto flx = iter.output(0);
  auto face_pressure = iter.output(1).squeeze(0);
  auto wl = iter.input(0);
  auto wr = iter.input(1);
  auto hlr = iter.input(2);  // el
  auto glr = iter.input(3);

  auto hl = hlr[ILT].clone();
  auto hr = hlr[IRT].clone();

  auto gammal = glr[ILT];
  auto gammar = glr[IRT];

  int ny = wl.size(0) - ICY;

  // dim, ivx
  // 3, IVX
  // 2, IVX + 1
  // 1, IVX + 2
  auto ivx = IPR - dim;
  auto ivy = IVX + ((ivx - IVX) + 1) % 3;
  auto ivz = IVX + ((ivx - IVX) + 2) % 3;

  // energy -> enthalpy
  hl += 0.5 * wl.narrow(0, IVX, 3).square().sum(0) + wl[IPR] / wl[IDN];
  hr += 0.5 * wr.narrow(0, IVX, 3).square().sum(0) + wr[IPR] / wr[IDN];

  auto rhobar = 0.5 * (wl[IDN] + wr[IDN]);
  auto gamma_bar = 0.5 * (gammal + gammar);
  auto cbar = torch::sqrt(gamma_bar * 0.5 * (wl[IPR] + wr[IPR]) / rhobar);
  auto pbar =
      0.5 * (wl[IPR] + wr[IPR]) + 0.5 * (rhobar * cbar) * (wl[ivx] - wr[ivx]);
  face_pressure.copy_(pbar);
  auto ubar =
      0.5 * (wl[ivx] + wr[ivx]) + 0.5 / (rhobar * cbar) * (wl[IPR] - wr[IPR]);

  // left flux
  auto fluxl = torch::zeros_like(wl);
  auto fluxr = torch::zeros_like(wr);

  fluxl[IDN] = ubar * wl[IDN] *
               (torch::ones_like(wl[IDN]) - wl.narrow(0, ICY, ny).sum(0));
  fluxl.narrow(0, ICY, ny) = ubar * wl[IDN] * wl.narrow(0, ICY, ny);
  fluxl.narrow(0, IVX, 3) = ubar * wl[IDN] * wl.narrow(0, IVX, 3);
  fluxl[ivx] += pbar;
  fluxl[IPR] = ubar * wl[IDN] * hl;

  // right flux
  fluxr[IDN] = ubar * wr[IDN] *
               (torch::ones_like(wr[IDN]) - wr.narrow(0, ICY, ny).sum(0));
  fluxr.narrow(0, ICY, ny) = ubar * wr[IDN] * wr.narrow(0, ICY, ny);
  fluxr.narrow(0, IVX, 3) = ubar * wr[IDN] * wr.narrow(0, IVX, 3);
  fluxr[ivx] += pbar;
  fluxr[IPR] = ubar * wr[IDN] * hr;

  auto ui = (ubar > 0).to(torch::kInt);
  torch::add_out(flx, ui * fluxl, (1 - ui) * fluxr);
}

}  // namespace snap

namespace at::native {

DEFINE_DISPATCH(call_lmars);
REGISTER_ALL_CPU_DISPATCH(call_lmars, &snap::call_lmars_cpu);
REGISTER_MPS_DISPATCH(call_lmars, &snap::call_lmars_mps);

}  // namespace at::native
