# Gate 2 arm A, CPU — FAIL

Stopped before uranus. Every B2 reference is VOID, and arm A misses the zero-clamp bar on 14 of 16 cards. The test was not changed to make those bars pass. No plan revision. No CUDA. No PR.

Base: snapy main `5eeb9b6761ae484a98b3993aae18314fc58cf862`. Port of `8c6b3fc` / `842a116` is `5281c51fd272662c8625142c01c29584af42afb7` plus `562d4979b92e4f48d5bd1a5043b9d9e81068ef75`. This result sits on top of those. CPU, fp64, one process per card, dt = 1. kintera 2.5.13 (cmake and the wheel import). `Rgas = 8.31446`. `use_nasa9_cp = 0`, `use_h2_cp = 0` on every card. Either flag on would be FAIL; neither was.

## What was checked

`MoistMixtureImpl::species_enthalpy` (`src/eos/moist_mixture.cpp:208-211`) calls `kintera::eval_intEng_R` and `kintera::eval_czh`, multiplies by `Rgas * inv_mu` (`:222-226`), then adds KE (`:252`). It is not a hard-coded linear formula. The five mutations are the branches at `:215-251`.

Theta is rebuilt from `rho_s`, cell volume, dt = 1, and the outgoing area-weighted B2 species flux, with margin `1 - 4096 eps`. It is not read from the code. A clamp cell is interior `IDN < 1e-10` or any species density `< 0`, counted on `u+du` before repair. Ghosts were 0 everywhere. NaNs were 0 everywhere.

B2 flux equals this build's `limiter: false` arm bitwise on all 16 cards (`max abs 0`). That is the "your build" reference. A separate binary of unmodified `5eeb9b6` was not run, so ideal-moist versus that tree is not claimed.

## Bars

O3 (vapor and cloud), T, I1, and I2: PASS on all 16. Worst O3 enthalpy ratio is `9.42e-4` (moist-mixture donors, vapor). Worst `|dT|` is `1.14e-13` K (ideal-moist donors). Worst I1 ratio is `2.75e-5`. I2 neighbor is bitwise on every card, and `|h-KE| / S_n = 0`.

`M_n` versus the YAML composition: relative `1.20e-16`. Not VOID.

Limited-face counts, full slab, match the list on every card (adv 5, settling 5, x2 30, donors 4, mixed 5, vapor-u0 5). IDN flux is bitwise A versus B2 on every card. Species carry `R_s` ratio is 0. Worst energy ratio `1.21e-4`, worst momentum ratio `1.30e-4`, worst column ratio `4.73e-5`, all under 1. Those four comparisons are still VOID, not PASS, because the B2 reference is void on every card.

B2 census, interior clamps: adv/hllc/vapor-u0 1, settling 1, donors 2, mixed 1, x2 6. Both EOS. The drained cell is the expected negative species cell (`min` species density `-0.020` on the uniform cards, `-0.0268` donors, `-0.0221` mixed). On every card except settling that same cell also has `IDN < 0` (`min` `-0.97` adv, `-1.30` donors, `-2.14` mixed, `-0.97` x2). Settling keeps `IDN` at `+0.97` and only the species goes negative. Zero clamps and zero NaNs were required. All 16 B2 references are VOID. VOID counts as FAIL.

Arm A census: PASS only on both settling cards (0 clamps; `min` species `1.8e-14`). FAIL on the other 14. Arm A does not limit dry `IDN`. On those 14 the species densities stay positive (`min` about `9e-15`) and the clamp is `IDN` under the floor only (`idn_only` 1, or 2 on donors, or 6 on x2). The bar says A counts 0. It does not.

x2, both EOS: B2's repair throws `Failed to fix vapor mass fractions` after the census is stored (`repair_threw_b2 = 1`). The drain is along x2, and `fix_vapor` only repairs an x1 column. Arm A does not throw. The other 14 cards repair without throwing. The throw is recorded; the flux dumps were still written. Repair was not disabled.

Moist-mixture mutations 1–5, all 8 cards: each turns O3 and the energy carry red by at least `1e3` times the tolerance. Smallest O3 ratio is `9.04e10` (mutation 4). Smallest energy ratio is `1.22e10`.

## Uranus

Not started. `examples/uranus.yaml` was not run. Step 4 waits on card bars that are not all PASS.
