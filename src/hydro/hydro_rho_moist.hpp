#pragma once

// torch
#include <torch/torch.h>

namespace snap {

class IdealMoistImpl;

// #250 O3 step 5. Replaces dref and dsf with densities on kintera's
// reversible saturated adiabat (rainout off). per_cell uses each cell's own
// entropy and total water; otherwise the bottom interior cell's, broadcast
// up the column. Aborts if the Newton misses any cell, ghosts included.
void apply_moist_density_ref(IdealMoistImpl* moist, torch::Tensor const& w,
                             torch::Tensor const& pref,
                             torch::Tensor const& psf_lo, torch::Tensor& dref,
                             torch::Tensor& dsf, bool per_cell, int is, int iu,
                             int js, int ju);

// Reconstruction positivity fallback (dl<=0 / dr<=0). The running total is
// reprinted before a moist-adiabat abort and again at process exit.
void note_wb_positivity_fallback(int64_t nleft, int64_t nright);

// First call only: max |rho-dref|/rho over the interior, with (i, j) =
// (x2, x1) counted from the first interior cell.
void log_wb_dref_t0(torch::Tensor const& rho, torch::Tensor const& dref, int is,
                    int iu, int js, int ju);

}  // namespace snap
